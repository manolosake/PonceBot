#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"exists": False, "size_bytes": 0, "sha256": ""}
    raw = path.read_bytes()
    return {"exists": True, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Hard-fail if root bundle contract diverges from smoke bundle.")
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--smoke-dir", required=True)
    ap.add_argument("--files", required=True, help="Comma-separated relative file paths.")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    root_dir = Path(args.root_dir).resolve()
    smoke_dir = Path(args.smoke_dir).resolve()
    files = [x.strip() for x in args.files.split(",") if x.strip()]

    checks: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for rel in files:
        root_meta = file_meta(root_dir / rel)
        smoke_meta = file_meta(smoke_dir / rel)
        equal = root_meta == smoke_meta
        row = {"file": rel, "root": root_meta, "smoke": smoke_meta, "match": equal}
        checks.append(row)
        if not equal:
            mismatches.append(row)

    report = {
        "check": "root_smoke_contract_checker",
        "root_dir": str(root_dir),
        "smoke_dir": str(smoke_dir),
        "files": files,
        "checked_at_utc": utc_now(),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
        "exit_code": 0 if not mismatches else 2,
    }

    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
