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
    mtime = float(path.stat().st_mtime) if exists else 0.0
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size,
        "sha256": _sha256(path) if size > 0 else "",
        "mtime_epoch": mtime,
        "captured_at_utc": _utc_now(),
    }


def _to_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _to_float(v: Any, default: float = -1.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _append_check(
    mismatches: list[dict[str, Any]],
    *,
    check: str,
    ok: bool,
    expected: Any,
    actual: Any,
) -> None:
    if ok:
        return
    mismatches.append(
        {
            "check": check,
            "expected": expected,
            "actual": actual,
        }
    )


def run_independent_check(artifacts_dir: Path, expected_branch: str) -> dict[str, Any]:
    validation_path = artifacts_dir / "s02_trace_validation.json"
    post_path = artifacts_dir / "post_publish_check.json"
    gs_path = artifacts_dir / "git_status.txt"
    patch_path = artifacts_dir / "changes.patch"

    mismatches: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    required = [validation_path, post_path, gs_path, patch_path]
    for p in required:
        ok = p.exists() and p.is_file()
        checks[f"exists::{p.name}"] = ok
        _append_check(
            mismatches,
            check=f"exists::{p.name}",
            ok=ok,
            expected=True,
            actual=ok,
        )
    if mismatches:
        return {
            "status": "FAIL",
            "exit_code": 1,
            "artifacts_dir": str(artifacts_dir),
            "expected_branch": expected_branch,
            "checks": checks,
            "mismatches": mismatches,
            "captured_at_utc": _utc_now(),
        }

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    post = json.loads(post_path.read_text(encoding="utf-8"))

    snap_gs = _snapshot(gs_path)
    snap_patch = _snapshot(patch_path)
    actual = {"git_status.txt": snap_gs, "changes.patch": snap_patch}

    for name in ("git_status.txt", "changes.patch"):
        expected_from_validation = validation.get("files", {}).get(name, {})
        expected_from_post = post.get("actual_after_publish", {}).get(name, {})
        snap = actual[name]

        _append_check(
            mismatches,
            check=f"{name}::path_vs_validation",
            ok=snap["path"] == str(expected_from_validation.get("path", "")),
            expected=expected_from_validation.get("path", ""),
            actual=snap["path"],
        )
        _append_check(
            mismatches,
            check=f"{name}::size_vs_validation",
            ok=snap["size_bytes"] == _to_int(expected_from_validation.get("size_bytes")),
            expected=expected_from_validation.get("size_bytes"),
            actual=snap["size_bytes"],
        )
        _append_check(
            mismatches,
            check=f"{name}::sha_vs_validation",
            ok=snap["sha256"] == str(expected_from_validation.get("sha256", "")),
            expected=expected_from_validation.get("sha256", ""),
            actual=snap["sha256"],
        )
        _append_check(
            mismatches,
            check=f"{name}::mtime_vs_validation",
            ok=snap["mtime_epoch"] == _to_float(expected_from_validation.get("mtime_epoch")),
            expected=expected_from_validation.get("mtime_epoch"),
            actual=snap["mtime_epoch"],
        )
        _append_check(
            mismatches,
            check=f"{name}::size_vs_post_publish",
            ok=snap["size_bytes"] == _to_int(expected_from_post.get("size_bytes")),
            expected=expected_from_post.get("size_bytes"),
            actual=snap["size_bytes"],
        )
        _append_check(
            mismatches,
            check=f"{name}::sha_vs_post_publish",
            ok=snap["sha256"] == str(expected_from_post.get("sha256", "")),
            expected=expected_from_post.get("sha256", ""),
            actual=snap["sha256"],
        )
        _append_check(
            mismatches,
            check=f"{name}::mtime_vs_post_publish",
            ok=snap["mtime_epoch"] == _to_float(expected_from_post.get("mtime_epoch")),
            expected=expected_from_post.get("mtime_epoch"),
            actual=snap["mtime_epoch"],
        )
        _append_check(
            mismatches,
            check=f"{name}::size_gt_zero",
            ok=snap["size_bytes"] > 0,
            expected=">0",
            actual=snap["size_bytes"],
        )

    expected_from_post = post.get("branch_provenance", {}).get("expected_branch", "")
    reported_branch = post.get("branch_provenance", {}).get("trace_branch", "")
    _append_check(
        mismatches,
        check="branch::expected_matches_runtime",
        ok=expected_branch == expected_from_post,
        expected=expected_branch,
        actual=expected_from_post,
    )
    _append_check(
        mismatches,
        check="branch::trace_matches_runtime",
        ok=reported_branch == expected_branch,
        expected=expected_branch,
        actual=reported_branch,
    )

    status = "PASS" if not mismatches else "FAIL"
    return {
        "status": status,
        "exit_code": 0 if status == "PASS" else 1,
        "artifacts_dir": str(artifacts_dir),
        "expected_branch": expected_branch,
        "checks": checks,
        "filesystem_snapshots": actual,
        "mismatches": mismatches,
        "captured_at_utc": _utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent S-02 checker against real filesystem state.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    result = run_independent_check(artifacts_dir, args.expected_branch)

    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    out_path = Path(args.out).expanduser().resolve() if args.out else artifacts_dir / "s02_independent_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
