#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def meta(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"exists": False, "size_bytes": 0, "sha256": "", "mtime_epoch": 0.0}
    b = path.read_bytes()
    st = path.stat()
    return {"exists": True, "size_bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "mtime_epoch": float(st.st_mtime)}


def contract_tuple(m: dict[str, Any]) -> tuple[bool, int, str]:
    return (bool(m.get("exists")), int(m.get("size_bytes", 0)), str(m.get("sha256", "")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--smoke-stable-dir", required=True)
    ap.add_argument("--files", default="changes.patch,git_status.txt")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--sleep-seconds", type=float, default=1.0)
    args = ap.parse_args()

    root = Path(args.root_dir).resolve()
    smoke = Path(args.smoke_stable_dir).resolve()
    summary = Path(args.summary).resolve()
    files = [x.strip() for x in args.files.split(",") if x.strip()]

    started_epoch = time.time()
    started_utc = utc_now()
    root_t0 = {f: meta(root / f) for f in files}
    smoke_snap = {f: meta(smoke / f) for f in files}
    summary_t0 = meta(summary)
    time.sleep(max(args.sleep_seconds, 0.0))
    root_t1 = {f: meta(root / f) for f in files}
    summary_t1 = meta(summary)

    mismatches = []
    late = []
    mtime_after_summary = []
    mtime_after_guard_start = []
    empty_terminal = []
    for f in files:
        if contract_tuple(root_t1[f]) != contract_tuple(smoke_snap[f]):
            mismatches.append({"file": f, "root_terminal": root_t1[f], "smoke": smoke_snap[f]})
        for field in ("exists", "size_bytes", "sha256", "mtime_epoch"):
            if root_t0[f].get(field) != root_t1[f].get(field):
                late.append({"file": f, "field": field, "t0": root_t0[f].get(field), "t1": root_t1[f].get(field)})
        if (not root_t1[f]["exists"]) or int(root_t1[f]["size_bytes"]) <= 0:
            empty_terminal.append({"file": f, "meta": root_t1[f]})
        if summary_t1["exists"] and float(root_t1[f]["mtime_epoch"]) > float(summary_t1["mtime_epoch"]):
            mtime_after_summary.append({"file": f, "file_mtime_epoch": root_t1[f]["mtime_epoch"], "summary_mtime_epoch": summary_t1["mtime_epoch"]})
        if float(root_t1[f]["mtime_epoch"]) > started_epoch:
            mtime_after_guard_start.append({"file": f, "file_mtime_epoch": root_t1[f]["mtime_epoch"], "guard_started_epoch": started_epoch})

    errors = []
    if mismatches:
        errors.append({"type": "root_smoke_mismatch_terminal", "count": len(mismatches), "details": mismatches})
    if late:
        errors.append({"type": "late_root_mutation_after_summary", "count": len(late), "details": late})
    if empty_terminal:
        errors.append({"type": "root_contract_file_empty_or_missing_terminal", "count": len(empty_terminal), "details": empty_terminal})
    if mtime_after_summary:
        errors.append({"type": "contract_file_mtime_newer_than_final_summary", "count": len(mtime_after_summary), "details": mtime_after_summary})
    if mtime_after_guard_start:
        errors.append({"type": "contract_file_mtime_newer_than_guard_start", "count": len(mtime_after_guard_start), "details": mtime_after_guard_start})

    report = {
        "check": "postfinal_root_smoke_guard",
        "checked_at_utc": utc_now(),
        "check_started_at_utc": started_utc,
        "decision_timestamps": {
            "guard_started_epoch": started_epoch,
            "summary_mtime_epoch": summary_t1["mtime_epoch"],
            "guard_checked_at_utc": utc_now(),
        },
        "root_snapshot_t0": root_t0,
        "root_snapshot_t1": root_t1,
        "smoke_snapshot": smoke_snap,
        "summary_snapshot_t0": summary_t0,
        "summary_snapshot_t1": summary_t1,
        "mismatch_count": len(mismatches),
        "late_mutation_count": len(late),
        "mtime_after_summary_count": len(mtime_after_summary),
        "mtime_after_guard_start_count": len(mtime_after_guard_start),
        "terminal_non_empty_violations": len(empty_terminal),
        "errors": errors,
        "root_smoke_consistency": len(mismatches) == 0,
        "status": "PASS" if not errors else "FAIL",
        "exit_code": 0 if not errors else 2,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
