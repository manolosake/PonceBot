#!/usr/bin/env python3
"""
Non-intrusive worker telemetry sidecar.

Goals:
- Emit periodic heartbeat (10-30s) for running jobs/roles without restarting codexbot.
- Emit activity events that help infer "coding" based on filesystem churn:
  - worktree file edits
  - artifacts dir updates
- Publish a compact JSON snapshot with TTL semantics for external consumers.

Transport:
- journald logs (service stdout)
- snapshot file (atomic write) at $CODEXBOT_WORKER_TELEMETRY_OUT

This sidecar reads the orchestrator SQLite DB in read-only mode.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class JobView:
    job_id: str
    role: str
    state: str
    created_at: float
    updated_at: float
    artifacts_dir: str | None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except Exception:
        return default


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    # Read-only URI avoids writes and reduces contention with codexbot.
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=3.0)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_running_jobs(conn: sqlite3.Connection) -> list[JobView]:
    rows = conn.execute(
        """
        SELECT job_id, role, state, created_at, updated_at, artifacts_dir
        FROM jobs
        WHERE state='running'
        ORDER BY updated_at DESC
        """
    ).fetchall()
    out: list[JobView] = []
    for r in rows:
        out.append(
            JobView(
                job_id=str(r["job_id"]),
                role=str(r["role"]),
                state=str(r["state"]),
                created_at=float(r["created_at"]),
                updated_at=float(r["updated_at"]),
                artifacts_dir=(None if r["artifacts_dir"] is None else str(r["artifacts_dir"])),
            )
        )
    return out


def _fetch_workspace_leases(conn: sqlite3.Connection) -> dict[str, tuple[str, int]]:
    # job_id -> (role, slot)
    rows = conn.execute("SELECT role, slot, job_id FROM workspace_leases").fetchall()
    out: dict[str, tuple[str, int]] = {}
    for r in rows:
        out[str(r["job_id"])] = (str(r["role"]), int(r["slot"]))
    return out


def _iter_recent_changes(root: Path, since_ts: float, *, max_items: int) -> Iterable[tuple[str, float]]:
    # Walk the tree with pruning. We only care about "some" evidence of churn.
    if not root.exists():
        return []

    skip_dirs = {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".codexbot_tmp",
        ".codexbot_preview",
    }

    found: list[tuple[str, float]] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune dirs in-place.
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                if len(found) >= max_items:
                    return found
                p = Path(dirpath) / fn
                try:
                    st = p.stat()
                except FileNotFoundError:
                    continue
                m = float(st.st_mtime)
                if m >= since_ts:
                    # Keep paths relative to root for compact output.
                    rel = str(p.relative_to(root))
                    found.append((rel, m))
    except Exception:
        # Best-effort only; never break the sidecar loop.
        return found
    return found


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    tmp.write_text(data + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    db_path = Path(os.environ.get("CODEXBOT_ORCH_DB", "/home/aponce/codexbot/data/jobs.sqlite"))
    out_path = Path(os.environ.get("CODEXBOT_WORKER_TELEMETRY_OUT", "/home/aponce/codexbot/data/worker_telemetry.json"))

    interval_s = _env_float("CODEXBOT_WORKER_TELEMETRY_INTERVAL_S", 15.0)  # 10-30s
    ttl_s = _env_int("CODEXBOT_WORKER_TELEMETRY_TTL_S", 60)  # consumer considers stale if older than TTL
    max_changes = _env_int("CODEXBOT_WORKER_TELEMETRY_MAX_CHANGES", 25)

    worktrees_root = Path(os.environ.get("CODEXBOT_WORKTREES_ROOT", "/home/aponce/codexbot/data/worktrees"))

    last_scan_ts = time.time()
    last_activity_by_job: dict[str, float] = {}

    while True:
        now = time.time()
        try:
            conn = _connect_ro(db_path)
        except Exception as e:
            print(f"worker_telemetry: db_open_failed err={e!r} db={db_path}", flush=True)
            time.sleep(min(interval_s, 5.0))
            continue

        try:
            running = _fetch_running_jobs(conn)
            leases = _fetch_workspace_leases(conn)
        except Exception as e:
            print(f"worker_telemetry: db_query_failed err={e!r}", flush=True)
            try:
                conn.close()
            except Exception:
                pass
            time.sleep(interval_s)
            continue
        finally:
            try:
                conn.close()
            except Exception:
                pass

        snapshot: dict[str, Any] = {
            "ts": now,
            "ttl_s": ttl_s,
            "interval_s": interval_s,
            "running_jobs": [],
        }

        for j in running:
            since = last_activity_by_job.get(j.job_id, last_scan_ts)

            wt_slot = leases.get(j.job_id)
            wt_path: Path | None = None
            if wt_slot is not None:
                role, slot = wt_slot
                wt_path = worktrees_root / role / f"slot{slot}"

            wt_changes = list(_iter_recent_changes(wt_path, since, max_items=max_changes)) if wt_path else []
            art_path = Path(j.artifacts_dir) if j.artifacts_dir else None
            art_changes = list(_iter_recent_changes(art_path, since, max_items=max_changes)) if art_path else []

            # Infer "coding" if there is filesystem churn.
            activity_kinds: list[str] = []
            if wt_changes:
                activity_kinds.append("worktree_edit")
            if art_changes:
                activity_kinds.append("artifacts_update")

            if activity_kinds:
                last_activity_by_job[j.job_id] = now

            item = {
                "job_id": j.job_id,
                "role": j.role,
                "heartbeat_ts": now,
                "last_activity_ts": last_activity_by_job.get(j.job_id),
                "activity_kinds": activity_kinds,
                "worktree": (str(wt_path) if wt_path else None),
                "artifacts_dir": j.artifacts_dir,
                "worktree_changes": [{"path": p, "ts": t} for (p, t) in wt_changes],
                "artifacts_changes": [{"path": p, "ts": t} for (p, t) in art_changes],
            }
            snapshot["running_jobs"].append(item)

            # Heartbeat log line; machine-parseable.
            # NOTE: keep short; details are in the snapshot file.
            print(
                "worker_heartbeat"
                f" role={j.role}"
                f" job={j.job_id}"
                f" activity={'/'.join(activity_kinds) if activity_kinds else 'none'}"
                f" last_activity_age_s={('' if last_activity_by_job.get(j.job_id) is None else int(now - float(last_activity_by_job[j.job_id])))}"
                f" ttl_s={ttl_s}",
                flush=True,
            )

        try:
            _atomic_write_json(out_path, snapshot)
        except Exception as e:
            print(f"worker_telemetry: write_failed err={e!r} out={out_path}", flush=True)

        last_scan_ts = now
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())

