#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _parse_status_paths(status_text: str) -> list[str]:
    paths: list[str] = []
    for raw in status_text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        # name-status format: M<TAB>path or R100<TAB>old<TAB>new
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                path = parts[-1]
                if path:
                    paths.append(path)
                continue
        # porcelain format: XY<space>path (or XY<space>old -> new)
        if len(line) < 4:
            continue
        payload = line[3:]
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        paths.append(payload)
    return paths


def _parse_patch_paths(patch_text: str) -> list[str]:
    out: list[str] = []
    prefix = "diff --git a/"
    for line in patch_text.splitlines():
        if not line.startswith(prefix):
            continue
        # diff --git a/foo b/foo
        parts = line.split()
        if len(parts) < 4:
            continue
        b_path = parts[3]
        if b_path.startswith("b/"):
            b_path = b_path[2:]
        out.append(b_path)
    return out


def evaluate_bundle(
    artifacts_dir: Path,
    repo_root: Path,
    *,
    require_non_empty_patch: bool = True,
    run_apply_check: bool = True,
) -> dict[str, Any]:
    changes_patch = artifacts_dir / "changes.patch"
    git_status = artifacts_dir / "git_status.txt"

    errors: list[str] = []
    checks: dict[str, bool] = {}

    checks["changes_patch_exists"] = changes_patch.exists()
    checks["git_status_exists"] = git_status.exists()

    if not checks["changes_patch_exists"]:
        errors.append("missing changes.patch")
    if not checks["git_status_exists"]:
        errors.append("missing git_status.txt")

    if errors:
        return {
            "status": "FAIL",
            "exit_code": 2,
            "artifacts_dir": str(artifacts_dir),
            "errors": errors,
            "checks": checks,
            "mismatches": [],
        }

    patch_text = changes_patch.read_text(encoding="utf-8", errors="replace")
    status_text = git_status.read_text(encoding="utf-8", errors="replace")

    patch_size = changes_patch.stat().st_size
    status_size = git_status.stat().st_size

    checks["changes_patch_non_empty"] = patch_size > 0
    checks["git_status_non_empty"] = status_size > 0

    if require_non_empty_patch and not checks["changes_patch_non_empty"]:
        errors.append("changes.patch is empty")
    if not checks["git_status_non_empty"]:
        errors.append("git_status.txt is empty")

    status_paths = _parse_status_paths(status_text)
    patch_paths = _parse_patch_paths(patch_text)

    missing_in_patch = sorted(p for p in status_paths if p not in patch_paths)
    checks["status_paths_covered_by_patch"] = len(missing_in_patch) == 0
    if missing_in_patch:
        errors.append("git_status paths missing in patch")

    apply_check_ok = False
    apply_check_stdout = ""
    apply_check_stderr = ""
    apply_check_returncode = None
    if run_apply_check and checks["changes_patch_non_empty"]:
        proc = subprocess.run(
            ["git", "apply", "--check", "--reverse", str(changes_patch)],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        apply_check_returncode = int(proc.returncode)
        apply_check_stdout = (proc.stdout or "").strip()
        apply_check_stderr = (proc.stderr or "").strip()
        apply_check_ok = proc.returncode == 0
        checks["patch_apply_check_reverse_ok"] = apply_check_ok
        if not apply_check_ok:
            errors.append("git apply --check --reverse failed")
    else:
        checks["patch_apply_check_reverse_ok"] = False if run_apply_check else True

    mismatches = [
        {
            "type": "missing_in_patch",
            "path": p,
        }
        for p in missing_in_patch
    ]

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "exit_code": 0 if status == "PASS" else 2,
        "artifacts_dir": str(artifacts_dir),
        "changes_patch": {
            "path": str(changes_patch),
            "size_bytes": patch_size,
        },
        "git_status": {
            "path": str(git_status),
            "size_bytes": status_size,
        },
        "checks": checks,
        "errors": errors,
        "mismatches": mismatches,
        "inventory": {
            "status_paths": status_paths,
            "patch_paths": patch_paths,
        },
        "patch_apply_check": {
            "ran": bool(run_apply_check and checks["changes_patch_non_empty"]),
            "ok": apply_check_ok,
            "returncode": apply_check_returncode,
            "stdout": apply_check_stdout,
            "stderr": apply_check_stderr,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Guardrail for wormhole artifacts bundle consistency")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--report-out", default="")
    ap.add_argument("--allow-empty-patch", action="store_true")
    ap.add_argument("--skip-apply-check", action="store_true")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    repo_root = Path(args.repo_root).resolve()

    result = evaluate_bundle(
        artifacts_dir,
        repo_root,
        require_non_empty_patch=not args.allow_empty_patch,
        run_apply_check=not args.skip_apply_check,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report_out:
        out = Path(args.report_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
