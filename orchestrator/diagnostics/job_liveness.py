from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import build_liveness_payload, json_dumps, utc_iso


def _format_text(payload: dict) -> str:
    lines = [
        f"checked_at={payload.get('checked_at_iso')}",
        f"db_path={payload.get('db_path')}",
        f"open_jobs_count={payload.get('open_jobs_count')}",
        f"oldest_open_age_seconds={payload.get('oldest_open_age_seconds')}",
    ]
    for job in payload.get("open_jobs", []):
        lines.append(
            "open_job "
            + " ".join(
                f"{key}={job.get(key)}"
                for key in ("job_id", "role", "state", "owner", "updated_at_iso", "idle_seconds")
                if key in job
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit read-only liveness details for open orchestrator jobs.")
    parser.add_argument("--db", type=Path, default=None, help="Path to jobs.sqlite. Defaults to BOT_ORCHESTRATOR_DB_PATH.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    try:
        payload = build_liveness_payload(args.db)
    except Exception as exc:
        error = {"ok": False, "error": str(exc), "checked_at_iso": utc_iso()}
        print(json_dumps(error) if args.json else f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json_dumps(payload) if args.json else _format_text(payload))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
