#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot(path: Path) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    size = int(path.stat().st_size) if exists else 0
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size,
        "sha256": _sha256(path) if size > 0 else "",
        "captured_at_utc": _utc_now(),
    }


def run_independent_check(artifacts_dir: Path, validation_report: Path) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    if not validation_report.exists():
        return {
            "status": "FAIL",
            "mismatches": [{"type": "missing_validation_report", "path": str(validation_report)}],
            "checks": {"validation_report_exists": False},
            "artifacts_dir": str(artifacts_dir),
            "captured_at_utc": _utc_now(),
        }

    payload = json.loads(validation_report.read_text(encoding="utf-8"))
    checks["validation_report_exists"] = True
    checks["validation_report_status"] = payload.get("status", "UNKNOWN")
    checks["validation_artifact_dir_matches"] = payload.get("artifacts_dir", "") == str(artifacts_dir)
    if not checks["validation_artifact_dir_matches"]:
        mismatches.append(
            {
                "type": "artifact_dir_mismatch",
                "expected": str(artifacts_dir),
                "reported": payload.get("artifacts_dir", ""),
            }
        )

    files = payload.get("files", {})
    for name in ("git_status.txt", "changes.patch"):
        reported = files.get(name, {})
        real_path = artifacts_dir / name
        real = _snapshot(real_path)
        if not real["exists"]:
            mismatches.append({"type": "missing_file", "file": name, "path": str(real_path)})
            continue
        if real["size_bytes"] <= 0:
            mismatches.append({"type": "size_zero", "file": name, "path": str(real_path)})
        if str(reported.get("path", "")) != str(real_path):
            mismatches.append(
                {
                    "type": "path_mismatch",
                    "file": name,
                    "expected_path": str(real_path),
                    "reported_path": str(reported.get("path", "")),
                }
            )
        if int(reported.get("size_bytes", 0)) != int(real["size_bytes"]):
            mismatches.append(
                {
                    "type": "size_mismatch",
                    "file": name,
                    "expected_size": int(real["size_bytes"]),
                    "reported_size": int(reported.get("size_bytes", 0)),
                }
            )
        if str(reported.get("sha256", "")) != str(real["sha256"]):
            mismatches.append(
                {
                    "type": "sha256_mismatch",
                    "file": name,
                    "expected_sha256": str(real["sha256"]),
                    "reported_sha256": str(reported.get("sha256", "")),
                }
            )

    return {
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "checks": checks,
        "validation_report_path": str(validation_report),
        "artifacts_dir": str(artifacts_dir),
        "captured_at_utc": _utc_now(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Independent S-02 checker (real stat+sha256 vs report).")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--validation-report", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    report_path = (
        Path(args.validation_report).expanduser().resolve()
        if args.validation_report
        else artifacts_dir / "s02_trace_validation.json"
    )
    out_path = Path(args.out).expanduser().resolve() if args.out else artifacts_dir / "s02_independent_check.json"

    result = run_independent_check(artifacts_dir, report_path)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
