from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .common import json_dumps, utc_iso


def _read_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        return None, f"{path}: {exc}"
    if not raw.strip():
        return None, f"{path}: empty file"
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return None, f"{path}: invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path}: expected JSON object"
    return payload, None


def _open_count_from_status(payload: dict[str, Any]) -> int | None:
    for key in ("open_jobs", "open_jobs_count"):
        value = payload.get(key)
        if isinstance(value, int):
            return int(value)
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    ids = payload.get("open_job_ids")
    if isinstance(ids, list):
        return len(ids)
    return None


def _open_count_from_jobs(payload: dict[str, Any]) -> int | None:
    value = payload.get("open_jobs_count")
    if isinstance(value, int):
        return int(value)
    try:
        if value is not None:
            return int(value)
    except Exception:
        pass
    jobs = payload.get("open_jobs")
    if isinstance(jobs, list):
        return len(jobs)
    return None


def evaluate_gate(status: dict[str, Any], jobs: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    status_count = _open_count_from_status(status)
    jobs_count = _open_count_from_jobs(jobs)

    if status.get("ok") is False:
        reasons.append("status snapshot reported ok=false")
    if jobs.get("ok") is False:
        reasons.append("job liveness reported ok=false")
    if status_count is None:
        reasons.append("status snapshot missing open job count")
    if jobs_count is None:
        reasons.append("job liveness missing open job count")
    if status_count is not None and jobs_count is not None and status_count != jobs_count:
        reasons.append(f"open job count mismatch: status={status_count} jobs={jobs_count}")
    if status_count not in (None, 0):
        reasons.append(f"status snapshot has open_jobs={status_count}")
    if jobs_count not in (None, 0):
        reasons.append(f"job liveness has open_jobs_count={jobs_count}")

    return len(reasons) == 0, reasons


def render_gate(status: dict[str, Any], jobs: dict[str, Any], *, ok: bool, reasons: list[str]) -> str:
    status_count = _open_count_from_status(status)
    jobs_count = _open_count_from_jobs(jobs)
    payload = {
        "gate": "PASS" if ok else "NO-GO",
        "checked_at_iso": utc_iso(),
        "status_open_jobs": status_count,
        "jobs_open_jobs": jobs_count,
        "reasons": reasons,
    }
    return json_dumps(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the idle-no-open-jobs close gate.")
    parser.add_argument("--status", type=Path, required=True, help="Path to status_snapshot JSON.")
    parser.add_argument("--jobs", type=Path, required=True, help="Path to job_liveness JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Path to write the gate result.")
    args = parser.parse_args(argv)

    status, status_error = _read_json_file(args.status)
    jobs, jobs_error = _read_json_file(args.jobs)
    read_errors = [err for err in (status_error, jobs_error) if err]

    ok = False
    reasons: list[str]
    if read_errors:
        reasons = read_errors
        status = status or {}
        jobs = jobs or {}
    else:
        ok, reasons = evaluate_gate(status or {}, jobs or {})

    rendered = render_gate(status or {}, jobs or {}, ok=ok, reasons=reasons)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
