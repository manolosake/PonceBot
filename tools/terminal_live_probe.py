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


def file_meta(path: Path) -> dict[str, Any]:
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-close live probe sampler.")
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--files", default="changes.patch,git_status.txt")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--interval-seconds", type=float, default=0.8)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    root = Path(args.root_dir).resolve()
    files = [x.strip() for x in args.files.split(",") if x.strip()]
    samples = max(args.samples, 2)
    interval = max(args.interval_seconds, 0.1)

    snapshots: list[dict[str, Any]] = []
    for i in range(samples):
        snap = {"sample_index": i, "captured_at_utc": utc_now(), "files": {}}
        for rel in files:
            snap["files"][rel] = file_meta(root / rel)
        snapshots.append(snap)
        if i < samples - 1:
            time.sleep(interval)

    errors: list[dict[str, Any]] = []
    # Invariant A: no blank/missing contract files in any sample.
    for snap in snapshots:
        for rel, meta in snap["files"].items():
            if (not meta["exists"]) or int(meta["size_bytes"]) <= 0:
                errors.append(
                    {
                        "type": "blank_or_missing_in_probe",
                        "sample_index": snap["sample_index"],
                        "file": rel,
                        "meta": meta,
                    }
                )

    # Invariant B: no drift between first and last sample.
    first = snapshots[0]["files"]
    last = snapshots[-1]["files"]
    for rel in files:
        a = first.get(rel, {})
        b = last.get(rel, {})
        for field in ("exists", "size_bytes", "sha256", "mtime_epoch"):
            if a.get(field) != b.get(field):
                errors.append(
                    {
                        "type": "drift_between_probe_samples",
                        "file": rel,
                        "field": field,
                        "first": a.get(field),
                        "last": b.get(field),
                    }
                )

    report = {
        "check": "terminal_live_probe",
        "root_dir": str(root),
        "files": files,
        "samples_requested": samples,
        "captured_at_utc": utc_now(),
        "snapshots": snapshots,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "exit_code": 0 if not errors else 2,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
