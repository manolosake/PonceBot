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


def main() -> int:
    ap = argparse.ArgumentParser(description="Replayable blocker validity check for proactive sweeps")
    ap.add_argument("--db", default="/home/aponce/codexbot/data/jobs.sqlite")
    ap.add_argument("--ticket-id", required=True)
    ap.add_argument("--stale-seconds", type=float, default=300.0)
    args = ap.parse_args()

    db_path = Path(args.db).expanduser()
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    rows = [
        dict(r)
        for r in con.execute(
            """
            SELECT job_id, role, state, COALESCE(depends_on,'[]') AS depends_on,
                   COALESCE(blocked_reason,'') AS blocked_reason, updated_at
            FROM jobs
            WHERE parent_job_id = ?
              AND state IN ('blocked','blocked_approval','waiting_deps','running','queued')
            ORDER BY created_at
            """,
            (str(args.ticket_id),),
        ).fetchall()
    ]

    now = time.time()
    for r in rows:
        r["updated_age_s"] = round(now - float(r.get("updated_at") or now), 1)

    blocked = [r for r in rows if str(r.get("state", "")).lower() in {"blocked", "blocked_approval", "waiting_deps"}]
    stale = [r for r in blocked if float(r.get("updated_age_s") or 0.0) > float(args.stale_seconds)]
    ids = {str(r.get("job_id") or "") for r in rows}

    invalid_wait = []
    for r in blocked:
        if str(r.get("state", "")).lower() != "waiting_deps":
            continue
        try:
            deps = json.loads(str(r.get("depends_on") or "[]"))
        except Exception:
            deps = []
        if not deps or not any(str(d) in ids for d in deps):
            invalid_wait.append(str(r.get("job_id") or ""))

    payload = {
        "generated_at": _utc_now(),
        "ticket_id": str(args.ticket_id),
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
