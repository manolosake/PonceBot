#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _check_non_empty(path: Path) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    size = int(path.stat().st_size) if exists else 0
    return {
        "path": str(path),
        "exists": bool(exists),
        "size_bytes": int(size),
        "ok": bool(exists and size > 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Local integrity guard for SRE evidence bundle.")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--required", default="merge_tree.log,changes.patch,git_status.txt,git_status")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    summary_path = Path(args.summary).resolve() if args.summary else (artifacts_dir / "summary.json")
    out_path = Path(args.out).resolve() if args.out else (artifacts_dir / "bundle_integrity_guard_report.json")

    required = [x.strip() for x in args.required.split(",") if x.strip()]
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    # Contract 1: summary JSON must parse.
    summary_ok = False
    summary_obj: dict[str, Any] = {}
    if not summary_path.exists():
        errors.append(f"missing summary: {summary_path}")
    else:
        try:
            raw = summary_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                errors.append("summary is not a JSON object")
            else:
                summary_ok = True
                summary_obj = parsed
        except Exception as ex:
            errors.append(f"summary parse error: {ex}")
    checks.append(
        {
            "key": "summary_json_parseable",
            "path": str(summary_path),
            "ok": summary_ok,
        }
    )

    # Contract 2: required artifacts must be non-empty.
    required_results: list[dict[str, Any]] = []
    for name in required:
        result = _check_non_empty(artifacts_dir / name)
        required_results.append({"name": name, **result})
        checks.append({"key": f"required_{name}_non_empty", "ok": result["ok"], "size_bytes": result["size_bytes"]})
        if not result["ok"]:
            errors.append(f"required artifact empty or missing: {name}")

    status = "PASS" if not errors else "FAIL"
    report = {
        "status": status,
        "generated_at_utc": _utc_now(),
        "artifacts_dir": str(artifacts_dir),
        "summary_path": str(summary_path),
        "required_files": required_results,
        "checks": checks,
        "errors": errors,
        "summary_status": summary_obj.get("status", "") if summary_ok else "",
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

