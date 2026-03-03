#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PATCH_CANDIDATES = ("changes.patch", "repo_patch.diff")
TEST_LOG_CANDIDATES = ("tests.log", "test.log", "verify.log", "test_orchestrator.log")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _first_existing(base: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = base / name
        if p.exists() and p.is_file():
            return p
    return None


def validate_bundle(artifacts_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}

    patch_path = _first_existing(artifacts_dir, PATCH_CANDIDATES)
    status_path = artifacts_dir / "git_status.txt"
    test_log_path = _first_existing(artifacts_dir, TEST_LOG_CANDIDATES)

    checks["patch_file"] = str(patch_path) if patch_path else None
    checks["git_status_file"] = str(status_path) if status_path.exists() else None
    checks["test_log_file"] = str(test_log_path) if test_log_path else None

    if patch_path is None:
        errors.append("missing diff file (changes.patch or repo_patch.diff)")
    if not status_path.exists():
        errors.append("missing git_status.txt")
    if test_log_path is None:
        errors.append("missing test log (tests.log/test.log/verify.log/test_orchestrator.log)")

    if patch_path is not None:
        patch_text = _read_text(patch_path).strip()
        patch_non_empty = bool(patch_text)
        checks["patch_non_empty"] = patch_non_empty
        if not patch_non_empty:
            errors.append(f"{patch_path.name} is empty")

    if status_path.exists():
        status_text = _read_text(status_path).strip()
        status_non_empty = bool(status_text)
        checks["git_status_non_empty"] = status_non_empty
        if not status_non_empty:
            errors.append("git_status.txt is empty")

    if test_log_path is not None:
        log_text = _read_text(test_log_path)
        log_non_empty = bool(log_text.strip())
        checks["test_log_non_empty"] = log_non_empty
        if not log_non_empty:
            errors.append(f"{test_log_path.name} is empty")
        has_pass_fail_marker = ("EXIT_CODE=" in log_text) or ("PASS" in log_text) or ("FAIL" in log_text)
        checks["test_log_has_result_marker"] = has_pass_fail_marker
        if not has_pass_fail_marker:
            errors.append(f"{test_log_path.name} missing explicit PASS/FAIL or EXIT_CODE marker")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": checks,
        "artifacts_dir": str(artifacts_dir),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate minimum evidence bundle and emit standardized summary.")
    ap.add_argument("--artifacts-dir", required=True, help="Directory containing evidence bundle files")
    ap.add_argument("--summary-out", default="", help="Optional path for evidence_summary.json output")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    result = validate_bundle(artifacts_dir)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)

    summary_out = Path(args.summary_out).expanduser().resolve() if args.summary_out else artifacts_dir / "evidence_summary.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(payload + "\n", encoding="utf-8")

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
