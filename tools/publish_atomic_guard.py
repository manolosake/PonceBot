#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_issue(issues: list[dict[str, Any]], rule: str, expected: Any, actual: Any) -> None:
    issues.append({"rule": rule, "expected": expected, "actual": actual})


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


def _parse_git_status_paths(git_status_path: Path) -> list[str]:
    tracked: set[str] = set()
    for raw in git_status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                tracked.add(parts[-1].strip())
                continue
        if line.startswith("?? "):
            tracked.add(line[3:].strip())
            continue
        if len(line) > 3 and line[2] == " ":
            rest = line[3:].strip()
        elif len(line) > 2 and line[1] == " ":
            rest = line[2:].strip()
        else:
            rest = line.strip()
        if not rest:
            continue
        if "->" in rest:
            rest = rest.split("->", 1)[1].strip()
        tracked.add(rest)
    return sorted(p for p in tracked if p)


def _parse_patch_paths(changes_patch_path: Path) -> list[str]:
    diff_re = re.compile(r"^diff --git a/(.+?) b/(.+)$")
    out: set[str] = set()
    for line in changes_patch_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = diff_re.match(line)
        if m:
            rhs = m.group(2).strip()
            if rhs:
                out.add(rhs)
    return sorted(out)


def _extract_declared_paths(payload: dict[str, Any]) -> list[str]:
    candidates: list[Any] = [payload.get("declared_files"), payload.get("files")]
    nested = payload.get("patch_apply_check")
    if isinstance(nested, dict):
        candidates.extend([nested.get("declared_files"), nested.get("files")])

    for c in candidates:
        if not isinstance(c, list):
            continue
        out: set[str] = set()
        for item in c:
            if isinstance(item, str) and item.strip():
                out.add(item.strip())
            elif isinstance(item, dict):
                p = str(item.get("path", "")).strip()
                if p:
                    out.add(p)
        if out:
            return sorted(out)
    return []


def run_guard(artifacts_dir: Path) -> dict[str, Any]:
    artifacts_dir = artifacts_dir.expanduser().resolve()
    issues: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    git_status_path = artifacts_dir / "git_status.txt"
    changes_patch_path = artifacts_dir / "changes.patch"
    patch_apply_check_path = artifacts_dir / "patch_apply_check.json"

    checks["git_status_exists"] = git_status_path.exists() and git_status_path.is_file()
    checks["changes_patch_exists"] = changes_patch_path.exists() and changes_patch_path.is_file()
    checks["patch_apply_check_exists"] = patch_apply_check_path.exists() and patch_apply_check_path.is_file()
    if not checks["git_status_exists"]:
        _append_issue(issues, "git_status_exists", True, False)
    if not checks["changes_patch_exists"]:
        _append_issue(issues, "changes_patch_exists", True, False)
    if not checks["patch_apply_check_exists"]:
        _append_issue(issues, "patch_apply_check_exists", True, False)
    if issues:
        return {
            "status": "FAIL",
            "exit_code": 2,
            "artifacts_dir": str(artifacts_dir),
            "checks": checks,
            "issues": issues,
            "captured_at_utc": _utc_now(),
        }

    status_paths = _parse_git_status_paths(git_status_path)
    patch_paths = _parse_patch_paths(changes_patch_path)
    declared_paths = _extract_declared_paths(_read_json(patch_apply_check_path))

    missing_from_patch_vs_status = sorted(set(status_paths) - set(patch_paths))
    extra_in_patch_vs_status = sorted(set(patch_paths) - set(status_paths))
    missing_from_patch_vs_declared = sorted(set(declared_paths) - set(patch_paths))
    extra_in_patch_vs_declared = sorted(set(patch_paths) - set(declared_paths))

    checks["changes_patch_non_empty"] = changes_patch_path.stat().st_size > 0
    checks["declared_paths_exist"] = len(declared_paths) > 0
    checks["status_paths_covered_by_patch"] = len(missing_from_patch_vs_status) == 0
    checks["patch_paths_within_status"] = len(extra_in_patch_vs_status) == 0
    checks["declared_paths_covered_by_patch"] = len(missing_from_patch_vs_declared) == 0
    checks["patch_paths_within_declared"] = len(extra_in_patch_vs_declared) == 0

    # Atomic-write regression guard: if apply-check has files, patch cannot be empty.
    if checks["declared_paths_exist"] and not checks["changes_patch_non_empty"]:
        _append_issue(issues, "changes_patch_non_empty_when_declared_paths_exist", True, False)
    if missing_from_patch_vs_status:
        _append_issue(issues, "status_paths_covered_by_patch", [], missing_from_patch_vs_status)
    if extra_in_patch_vs_status:
        _append_issue(issues, "patch_paths_within_status", [], extra_in_patch_vs_status)
    if missing_from_patch_vs_declared:
        _append_issue(issues, "declared_paths_covered_by_patch", [], missing_from_patch_vs_declared)
    if extra_in_patch_vs_declared:
        _append_issue(issues, "patch_paths_within_declared", [], extra_in_patch_vs_declared)

    return {
        "status": "PASS" if not issues else "FAIL",
        "exit_code": 0 if not issues else 2,
        "artifacts_dir": str(artifacts_dir),
        "checks": checks,
        "path_sets": {
            "git_status_paths": status_paths,
            "changes_patch_paths": patch_paths,
            "patch_apply_check_declared_paths": declared_paths,
            "missing_from_patch_vs_status": missing_from_patch_vs_status,
            "extra_in_patch_vs_status": extra_in_patch_vs_status,
            "missing_from_patch_vs_declared": missing_from_patch_vs_declared,
            "extra_in_patch_vs_declared": extra_in_patch_vs_declared,
        },
        "issues": issues,
        "captured_at_utc": _utc_now(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Atomic publish guard: blocks mismatches between git_status, changes.patch and patch_apply_check."
    )
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else artifacts_dir / "publish_atomic_guard_report.json"
    log_path = Path(args.log).expanduser().resolve() if args.log else artifacts_dir / "publish_atomic_guard.log"

    result = run_guard(artifacts_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (
            f"TIMESTAMP={_utc_now()}\n"
            f"ARTIFACTS_DIR={result['artifacts_dir']}\n"
            f"STATUS={result['status']}\n"
            f"EXIT_CODE={result['exit_code']}\n"
            f"ISSUE_COUNT={len(result['issues'])}\n"
        ),
        encoding="utf-8",
    )
    print(payload, end="")
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
