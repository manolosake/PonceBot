from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from .common import build_liveness_payload, connect_readonly, default_db_path, fetch_counts, json_dumps, utc_iso


def build_snapshot(db_path: Path | None = None, *, now: float | None = None) -> dict[str, Any]:
    checked_at = time.time() if now is None else float(now)
    resolved = Path(db_path) if db_path is not None else default_db_path()
    liveness = build_liveness_payload(resolved, now=checked_at)
    with connect_readonly(resolved) as conn:
        state_counts, role_state_counts = fetch_counts(conn)
    return {
        "ok": True,
        "checked_at": checked_at,
        "checked_at_iso": utc_iso(checked_at),
        "db_path": str(Path(resolved).expanduser()),
        "open_jobs": int(liveness["open_jobs_count"]),
        "open_job_ids": [str(job.get("job_id") or "") for job in liveness["open_jobs"]],
        "state_counts": state_counts,
        "role_state_counts": role_state_counts,
    }


def _format_text(payload: dict[str, Any]) -> str:
    lines = [
        f"checked_at={payload.get('checked_at_iso')}",
        f"db_path={payload.get('db_path')}",
        f"open_jobs={payload.get('open_jobs')}",
        "open_job_ids=" + ",".join(str(x) for x in payload.get("open_job_ids", [])),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit a read-only orchestrator status snapshot.")
    parser.add_argument("--db", type=Path, default=None, help="Path to jobs.sqlite. Defaults to BOT_ORCHESTRATOR_DB_PATH.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    try:
        payload = build_snapshot(args.db)
    except Exception as exc:
        error = {"ok": False, "error": str(exc), "checked_at_iso": utc_iso()}
        print(json_dumps(error) if args.json else f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json_dumps(payload) if args.json else _format_text(payload))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
