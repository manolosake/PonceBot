#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate minimum backend evidence package.")
    ap.add_argument("--execution-dir", required=True, help="Path to artifacts execution directory")
    ap.add_argument("--strict-nonempty", action="store_true", help="Require required files to be non-empty")
    args = ap.parse_args()

    execution_dir = Path(args.execution_dir).expanduser().resolve()
    errors: list[str] = []
    checks: dict[str, Any] = {}

    required_files = [
        "repo_patch.diff",
        "git_status.txt",
        "verify.log",
        "contract_validation_report.json",
        "visual_assets/run1_summary.json",
        "visual_assets/run2_summary.json",
        "visual_assets/run_compare_summary.json",
        "risk_summary.md",
    ]

    for rel in required_files:
        p = execution_dir / rel
        exists = p.exists() and p.is_file()
        checks[f"exists:{rel}"] = exists
        if not exists:
            errors.append(f"missing file: {rel}")
            continue
        size = p.stat().st_size
        checks[f"size:{rel}"] = size
        if args.strict_nonempty and size <= 0:
            errors.append(f"empty file: {rel}")

    verify_path = execution_dir / "verify.log"
    if verify_path.exists():
        verify_text = verify_path.read_text(encoding="utf-8", errors="replace")
        has_cmd = "make verify" in verify_text
        has_exit0 = "EXIT_CODE=0" in verify_text
        checks["verify_contains_make_verify"] = has_cmd
        checks["verify_contains_exit_code_0"] = has_exit0
        if not has_cmd:
            errors.append("verify.log missing make verify command trace")
        if not has_exit0:
            errors.append("verify.log missing EXIT_CODE=0")

    contract_path = execution_dir / "contract_validation_report.json"
    if contract_path.exists():
        try:
            contract = _read_json(contract_path)
            c_pass = str(contract.get("status", "")).upper() == "PASS"
            c_errors_empty = contract.get("errors", []) == []
            checks["contract_status_pass"] = c_pass
            checks["contract_errors_empty"] = c_errors_empty
            if not c_pass:
                errors.append("contract_validation_report.json status is not PASS")
            if not c_errors_empty:
                errors.append("contract_validation_report.json errors is not []")
        except Exception as ex:
            errors.append(f"contract_validation_report.json parse error: {ex}")

    compare_path = execution_dir / "visual_assets" / "run_compare_summary.json"
    if compare_path.exists():
        try:
            compare = _read_json(compare_path)
            compare_pass = str(compare.get("status", "")).upper() == "PASS"
            checks["run_compare_status_pass"] = compare_pass
            if not compare_pass:
                errors.append("run_compare_summary.json status is not PASS")
        except Exception as ex:
            errors.append(f"run_compare_summary.json parse error: {ex}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "execution_dir": str(execution_dir),
        "checks": checks,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
