from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OPEN_STATES = ("queued", "running", "waiting_deps", "blocked", "review", "ready_for_merge")
CLOSED_STATES = ("done", "failed", "cancelled", "canceled", "skipped", "paused")


def utc_iso(ts: float | None = None) -> str:
    value = time.time() if ts is None else float(ts)
    return datetime.fromtimestamp(value, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[2]


def default_db_path() -> Path:
    for key in ("BOT_ORCHESTRATOR_DB_PATH", "ORCH_JOBS_DB"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser()

    root = repo_root_from_module()
    candidate = root / "data" / "jobs.sqlite"
    if candidate.exists():
        return candidate

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        for guess in (parent / "data" / "jobs.sqlite", parent / "jobs.sqlite"):
            if guess.exists():
                return guess
    return candidate


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    resolved = Path(db_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"jobs database not found: {resolved}")
    uri = f"file:{resolved}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _preview(text: Any, limit: int = 180) -> str:
    raw = str(text or "").replace("\n", " ").strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3].rstrip() + "..."


def _is_idle_no_open_jobs_self_check(row: dict[str, Any]) -> bool:
    role = str(row.get("role") or "").strip().lower()
    text = str(row.get("input_text") or "").strip()
    return role == "sre" and text.startswith(
        "Preflight guard and closure protocol for idle scheduler state (no open jobs)."
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _select_existing(columns: set[str], wanted: tuple[str, ...]) -> str:
    selected = [name for name in wanted if name in columns]
    if not selected:
        return ""
    return ", ".join(selected)


def fetch_open_jobs(conn: sqlite3.Connection, *, now: float | None = None) -> list[dict[str, Any]]:
    columns = _table_columns(conn, "jobs")
    wanted = (
        "job_id",
        "parent_job_id",
        "role",
        "state",
        "owner",
        "input_text",
        "created_at",
        "updated_at",
        "due_at",
        "stalled_since",
        "retry_count",
        "max_retries",
    )
    select_cols = _select_existing(columns, wanted)
    if not select_cols:
        return []

    placeholders = ",".join("?" for _ in OPEN_STATES)
    rows = conn.execute(
        f"""
        SELECT {select_cols}
        FROM jobs
        WHERE state IN ({placeholders})
        ORDER BY updated_at ASC, created_at ASC
        """,
        OPEN_STATES,
    ).fetchall()
    checked_at = time.time() if now is None else float(now)
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if _is_idle_no_open_jobs_self_check(item):
            continue
        created_at = _safe_float(item.get("created_at"))
        updated_at = _safe_float(item.get("updated_at"))
        stalled_since = _safe_float(item.get("stalled_since"))
        item["created_at_iso"] = utc_iso(created_at) if created_at else ""
        item["updated_at_iso"] = utc_iso(updated_at) if updated_at else ""
        item["age_seconds"] = max(0, int(checked_at - created_at)) if created_at else None
        item["idle_seconds"] = max(0, int(checked_at - updated_at)) if updated_at else None
        item["stalled_seconds"] = max(0, int(checked_at - stalled_since)) if stalled_since else None
        if "input_text" in item:
            item["input_preview"] = _preview(item.pop("input_text"))
        out.append(item)
    return out


def fetch_counts(conn: sqlite3.Connection) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    state_counts: Counter[str] = Counter()
    role_state_counts: dict[str, Counter[str]] = {}
    rows = conn.execute(
        """
        SELECT role, state, COUNT(*) AS n
        FROM jobs
        GROUP BY role, state
        ORDER BY role, state
        """
    ).fetchall()
    for row in rows:
        role = str(row["role"] or "unknown")
        state = str(row["state"] or "unknown")
        n = int(row["n"] or 0)
        state_counts[state] += n
        role_state_counts.setdefault(role, Counter())[state] += n
    return dict(state_counts), {role: dict(counts) for role, counts in role_state_counts.items()}


def build_liveness_payload(db_path: Path | None = None, *, now: float | None = None) -> dict[str, Any]:
    checked_at = time.time() if now is None else float(now)
    resolved = Path(db_path) if db_path is not None else default_db_path()
    with connect_readonly(resolved) as conn:
        open_jobs = fetch_open_jobs(conn, now=checked_at)
    oldest = max((int(job.get("age_seconds") or 0) for job in open_jobs), default=0)
    return {
        "ok": True,
        "checked_at": checked_at,
        "checked_at_iso": utc_iso(checked_at),
        "db_path": str(Path(resolved).expanduser()),
        "active_states": list(OPEN_STATES),
        "closed_states": list(CLOSED_STATES),
        "open_jobs_count": len(open_jobs),
        "open_jobs": open_jobs,
        "oldest_open_age_seconds": oldest,
    }
