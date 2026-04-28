#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, UTC
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _updated_age_seconds(updated_at: object, now_epoch: float) -> float:
    if updated_at in (None, ""):
        return 0.0
    try:
        raw_ts = float(updated_at)
        if raw_ts > 1e11:
            raw_ts = raw_ts / 1000.0
        return max(0.0, float(now_epoch) - raw_ts)
    except Exception:
        pass

    try:
        raw = str(updated_at).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, float(now_epoch) - parsed.timestamp())
    except Exception:
        return 0.0


_ACTIVE_STATES = ("blocked", "blocked_approval", "waiting_deps", "running", "queued")
_REQUIRED_JOB_COLUMNS = {
    "job_id",
    "role",
    "state",
    "depends_on",
    "blocked_reason",
    "updated_at",
    "created_at",
    "parent_job_id",
    "labels",
}


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _coerce_depends_on_list(depends_on: object) -> tuple[list[str], bool]:
    try:
        raw_deps = json.loads(str(depends_on or "[]"))
    except Exception:
        return [], True

    if not isinstance(raw_deps, list):
        return [], True

    deps = [dep.strip() for dep in raw_deps if isinstance(dep, str) and dep.strip()]
    return deps, False


def _error_payload(ticket_id: str, db_path: Path, error: str, recommendation: str) -> dict:
    return {
        "generated_at": _utc_now(),
        "ticket_id": str(ticket_id),
        "db_path": str(db_path),
        "error": error,
        "recommendation": recommendation,
    }


def _schema_error(con: sqlite3.Connection) -> str | None:
    table_row = con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("jobs",),
    ).fetchone()
    if table_row is None:
        return "schema_missing"

    columns = {str(row["name"]) for row in con.execute("PRAGMA table_info(jobs)").fetchall()}
    if not _REQUIRED_JOB_COLUMNS.issubset(columns):
        return "schema_missing"
    return None


def _select_rows_for_ticket(con: sqlite3.Connection, ticket_id: str) -> tuple[list[dict], str]:
    rows = [
        dict(r)
        for r in con.execute(
            f"""
            SELECT job_id, role, state, COALESCE(depends_on,'[]') AS depends_on,
                   COALESCE(blocked_reason,'') AS blocked_reason, updated_at
            FROM jobs
            WHERE parent_job_id = ?
              AND state IN ({",".join("?" for _ in _ACTIVE_STATES)})
            ORDER BY created_at
            """,
            (str(ticket_id), *_ACTIVE_STATES),
        ).fetchall()
    ]
    if rows:
        return rows, "parent_job_id"

    # Fallback for lanes where ticket lineage is stored in labels JSON.
    escaped_ticket_id = _escape_like(str(ticket_id))
    rows = [
        dict(r)
        for r in con.execute(
            f"""
            SELECT job_id, role, state, COALESCE(depends_on,'[]') AS depends_on,
                   COALESCE(blocked_reason,'') AS blocked_reason, updated_at
            FROM jobs
            WHERE (labels LIKE ? ESCAPE '\\' OR labels LIKE ? ESCAPE '\\')
              AND state IN ({",".join("?" for _ in _ACTIVE_STATES)})
            ORDER BY created_at
            """,
            (
                f'%"ticket": "{escaped_ticket_id}"%',
                f'%"ticket":"{escaped_ticket_id}"%',
                *_ACTIVE_STATES,
            ),
        ).fetchall()
    ]
    return rows, "labels_ticket_fallback"


def main() -> int:
    ap = argparse.ArgumentParser(description="Replayable blocker validity check for proactive sweeps")
    ap.add_argument("--db", default="/home/aponce/codexbot/data/jobs.sqlite")
    ap.add_argument("--ticket-id", required=True)
    ap.add_argument("--stale-seconds", type=float, default=300.0)
    args = ap.parse_args()

    db_path = Path(args.db).expanduser()
    if not db_path.is_file():
        print(
            json.dumps(
                _error_payload(
                    str(args.ticket_id),
                    db_path,
                    "db_missing",
                    "provide_existing_jobs_sqlite_path",
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    schema_error = _schema_error(con)
    if schema_error is not None:
        print(
            json.dumps(
                _error_payload(
                    str(args.ticket_id),
                    db_path,
                    schema_error,
                    "run_against_jobs_database_with_required_jobs_schema",
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    rows, selector_strategy = _select_rows_for_ticket(con, str(args.ticket_id))

    now = time.time()
    for r in rows:
        r["updated_age_s"] = round(_updated_age_seconds(r.get("updated_at"), now), 1)

    blocked = [r for r in rows if str(r.get("state", "")).lower() in {"blocked", "blocked_approval", "waiting_deps"}]
    stale = [r for r in blocked if float(r.get("updated_age_s") or 0.0) > float(args.stale_seconds)]
    ids = {str(r.get("job_id") or "") for r in rows}

    invalid_wait = []
    for r in blocked:
        if str(r.get("state", "")).lower() != "waiting_deps":
            continue
        deps, malformed_deps = _coerce_depends_on_list(r.get("depends_on"))
        if malformed_deps or not deps or not any(d in ids for d in deps):
            invalid_wait.append(str(r.get("job_id") or ""))

    payload = {
        "generated_at": _utc_now(),
        "ticket_id": str(args.ticket_id),
        "selector_strategy": selector_strategy,
        "candidate_count": len(rows),
        "blocked_or_waiting_count": len(blocked),
        "stale_threshold_seconds": float(args.stale_seconds),
        "stale_blocked_count": len(stale),
        "invalid_wait_dependency_count": len(invalid_wait),
        "invalid_wait_job_ids": invalid_wait,
        "recommendation": (
            "cancel_or_reseed_invalid_or_stale_blockers"
            if (len(stale) > 0 or len(invalid_wait) > 0)
            else "keep_blockers_valid_no_cancel"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 2 if (len(stale) > 0 or len(invalid_wait) > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
