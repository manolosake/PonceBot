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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--files", default="changes.patch,git_status.txt")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--interval-seconds", type=float, default=0.5)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    root = Path(args.root_dir).resolve()
    files = [x.strip() for x in args.files.split(",") if x.strip()]
    samples = max(args.samples, 2)
    interval = max(args.interval_seconds, 0.1)

    snaps = []
    for i in range(samples):
        snap = {"sample_index": i, "captured_at_utc": utc_now(), "files": {f: meta(root / f) for f in files}}
        snaps.append(snap)
        if i < samples - 1:
            time.sleep(interval)

    errors = []
    for s in snaps:
        for f, m in s["files"].items():
            if (not m["exists"]) or int(m["size_bytes"]) <= 0:
                errors.append({"type": "blank_or_missing_in_probe", "sample_index": s["sample_index"], "file": f, "meta": m})
    first, last = snaps[0]["files"], snaps[-1]["files"]
    for f in files:
        for field in ("exists", "size_bytes", "sha256", "mtime_epoch"):
            if first[f].get(field) != last[f].get(field):
                errors.append({"type": "drift_between_probe_samples", "file": f, "field": field, "first": first[f].get(field), "last": last[f].get(field)})

    report = {
        "check": "terminal_live_probe",
        "root_dir": str(root),
        "files": files,
        "samples_requested": samples,
        "captured_at_utc": utc_now(),
        "snapshots": snaps,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "exit_code": 0 if not errors else 2,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
