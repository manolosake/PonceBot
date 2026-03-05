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


def stat_meta(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"exists": False, "size_bytes": 0, "sha256": "", "mtime_epoch": 0.0}
    raw = path.read_bytes()
    st = path.stat()
    return {
        "exists": True,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mtime_epoch": float(st.st_mtime),
    }


def compare(root_dir: Path, smoke_dir: Path, files: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    root_snap: dict[str, Any] = {}
    smoke_snap: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    for rel in files:
        r = stat_meta(root_dir / rel)
        s = stat_meta(smoke_dir / rel)
        root_snap[rel] = r
        smoke_snap[rel] = s
        if r != s:
            mismatches.append({"file": rel, "root": r, "smoke_stable": s})
    return mismatches, root_snap, smoke_snap


def detect_late_mutation(snapshot_t0: dict[str, Any], snapshot_t1: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in sorted(set(snapshot_t0) | set(snapshot_t1)):
        a = snapshot_t0.get(rel, {})
        b = snapshot_t1.get(rel, {})
        for field in ("exists", "size_bytes", "sha256", "mtime_epoch"):
            if a.get(field) != b.get(field):
                out.append({"file": rel, "field": field, "t0": a.get(field), "t1": b.get(field)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-final root vs smoke_stable contract guard.")
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--smoke-stable-dir", required=True)
    ap.add_argument("--files", default="changes.patch,git_status.txt")
    ap.add_argument("--report", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--sleep-seconds", type=float, default=1.5)
    args = ap.parse_args()

    root_dir = Path(args.root_dir).resolve()
    smoke_dir = Path(args.smoke_stable_dir).resolve()
    files = [x.strip() for x in args.files.split(",") if x.strip()]
    summary_path = Path(args.summary).resolve()
    report_path = Path(args.report).resolve()

    mismatch, root_t0, smoke_t0 = compare(root_dir, smoke_dir, files)
    check_started = utc_now()
    summary_stat = stat_meta(summary_path)

    time.sleep(max(args.sleep_seconds, 0.0))
    root_t1 = {rel: stat_meta(root_dir / rel) for rel in files}
    late_mut = detect_late_mutation(root_t0, root_t1)
    summary_after = stat_meta(summary_path)
    summary_mut = detect_late_mutation({"summary": summary_stat}, {"summary": summary_after})

    errors: list[dict[str, Any]] = []
    if mismatch:
        errors.append({"type": "root_smoke_mismatch", "count": len(mismatch), "details": mismatch})
    if late_mut:
        errors.append({"type": "late_root_mutation_after_summary", "count": len(late_mut), "details": late_mut})
    if summary_mut:
        errors.append({"type": "summary_mutation_during_guard_window", "count": len(summary_mut), "details": summary_mut})
    summary_mtime = float(summary_after.get("mtime_epoch", 0.0))
    mtime_after_summary: list[dict[str, Any]] = []
    for rel, meta in root_t1.items():
        if float(meta.get("mtime_epoch", 0.0)) > summary_mtime:
            mtime_after_summary.append(
                {
                    "file": rel,
                    "file_mtime_epoch": float(meta.get("mtime_epoch", 0.0)),
                    "summary_mtime_epoch": summary_mtime,
                }
            )
    if mtime_after_summary:
        errors.append(
            {
                "type": "contract_file_mtime_newer_than_final_summary",
                "count": len(mtime_after_summary),
                "details": mtime_after_summary,
            }
        )

    report = {
        "check": "postfinal_root_smoke_guard",
        "checked_at_utc": utc_now(),
        "check_started_at_utc": check_started,
        "root_dir": str(root_dir),
        "smoke_stable_dir": str(smoke_dir),
        "files": files,
        "root_snapshot_t0": root_t0,
        "root_snapshot_t1": root_t1,
        "smoke_snapshot": smoke_t0,
        "summary_path": str(summary_path),
        "summary_snapshot_t0": summary_stat,
        "summary_snapshot_t1": summary_after,
        "summary_mtime_epoch": summary_mtime,
        "mismatch_count": len(mismatch),
        "late_mutation_count": len(late_mut),
        "summary_mutation_count": len(summary_mut),
        "mtime_after_summary_count": len(mtime_after_summary),
        "errors": errors,
        "root_smoke_consistency": len(mismatch) == 0,
        "root_mtime_not_newer_than_summary": len(mtime_after_summary) == 0,
        "status": "PASS" if not errors else "FAIL",
        "exit_code": 0 if not errors else 2,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
