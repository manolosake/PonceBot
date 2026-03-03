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


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_reported_branch(artifacts_dir: Path, override: str) -> str:
    if override.strip():
        return override.strip()
    summary_path = artifacts_dir / "bundle_s02_summary.json"
    if summary_path.exists():
        try:
            payload = _read_json(summary_path)
            return str(payload.get("trace", {}).get("branch", "")).strip()
        except Exception:  # noqa: BLE001
            pass
    manifest_path = artifacts_dir / "s02_final_manifest_signed.json"
    if manifest_path.exists():
        try:
            payload = _read_json(manifest_path)
            return str(payload.get("order_branch_reported", "")).strip()
        except Exception:  # noqa: BLE001
            pass
    return ""


def _append_mismatch(mismatches: list[dict[str, Any]], check: str, expected: Any, actual: Any) -> None:
    mismatches.append({"check": check, "expected": expected, "actual": actual})


def run_close_gate(artifacts_dir: Path, expected_order_branch: str, reported_branch_override: str) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    critical_names = [
        "git_status.txt",
        "changes.patch",
        "s02_trace_validation.json",
        "post_publish_check.json",
        "bundle_s02_summary.json",
    ]
    critical_paths = [artifacts_dir / n for n in critical_names]
    snapshots = {p.name: _snapshot(p) for p in critical_paths}

    for name, snap in snapshots.items():
        exists = bool(snap["exists"])
        non_empty = int(snap["size_bytes"]) > 0
        checks[f"{name}::exists"] = exists
        checks[f"{name}::size_gt_zero"] = non_empty
        if not exists:
            _append_mismatch(mismatches, f"{name}::exists", True, exists)
        elif not non_empty:
            _append_mismatch(mismatches, f"{name}::size_gt_zero", ">0", snap["size_bytes"])

    reported_order_branch = _detect_reported_branch(artifacts_dir, reported_branch_override)
    checks["branch::reported_present"] = bool(reported_order_branch)
    checks["branch::match_expected"] = reported_order_branch == expected_order_branch
    if not checks["branch::reported_present"]:
        _append_mismatch(mismatches, "branch::reported_present", True, False)
    if not checks["branch::match_expected"]:
        _append_mismatch(mismatches, "branch::match_expected", expected_order_branch, reported_order_branch)

    validation_path = artifacts_dir / "s02_trace_validation.json"
    post_path = artifacts_dir / "post_publish_check.json"
    manifest_path = artifacts_dir / "s02_final_manifest_signed.json"
    checks["drift_inputs_present"] = validation_path.exists() and post_path.exists() and manifest_path.exists()
    if not checks["drift_inputs_present"]:
        _append_mismatch(
            mismatches,
            "drift::required_inputs_present",
            True,
            checks["drift_inputs_present"],
        )

    drift_detected = False
    if checks["drift_inputs_present"]:
        validation = _read_json(validation_path)
        post = _read_json(post_path)
        manifest = _read_json(manifest_path)
        manifest_files = manifest.get("files", {})
        for name in ("git_status.txt", "changes.patch"):
            snap = snapshots[name]
            expected_validation = validation.get("files", {}).get(name, {})
            expected_post = post.get("actual_after_publish", {}).get(name, {})
            expected_manifest = manifest_files.get(name, {})

            if snap["path"] != str(expected_validation.get("path", "")):
                drift_detected = True
                _append_mismatch(mismatches, f"{name}::path_vs_validation", expected_validation.get("path", ""), snap["path"])
            if snap["size_bytes"] != _to_int(expected_validation.get("size_bytes")):
                drift_detected = True
                _append_mismatch(
                    mismatches,
                    f"{name}::size_vs_validation",
                    expected_validation.get("size_bytes", -1),
                    snap["size_bytes"],
                )
            if snap["sha256"] != str(expected_validation.get("sha256", "")):
                drift_detected = True
                _append_mismatch(
                    mismatches,
                    f"{name}::sha_vs_validation",
                    expected_validation.get("sha256", ""),
                    snap["sha256"],
                )
            if snap["mtime_epoch"] != _to_float(expected_validation.get("mtime_epoch")):
                drift_detected = True
                _append_mismatch(
                    mismatches,
                    f"{name}::mtime_vs_validation",
                    expected_validation.get("mtime_epoch", -1.0),
                    snap["mtime_epoch"],
                )

            if snap["size_bytes"] != _to_int(expected_post.get("size_bytes")):
                drift_detected = True
                _append_mismatch(mismatches, f"{name}::size_vs_post", expected_post.get("size_bytes", -1), snap["size_bytes"])
            if snap["sha256"] != str(expected_post.get("sha256", "")):
                drift_detected = True
                _append_mismatch(mismatches, f"{name}::sha_vs_post", expected_post.get("sha256", ""), snap["sha256"])

            if snap["size_bytes"] != _to_int(expected_manifest.get("size_bytes")):
                drift_detected = True
                _append_mismatch(
                    mismatches,
                    f"{name}::size_vs_manifest",
                    expected_manifest.get("size_bytes", -1),
                    snap["size_bytes"],
                )
            if snap["sha256"] != str(expected_manifest.get("sha256", "")):
                drift_detected = True
                _append_mismatch(
                    mismatches,
                    f"{name}::sha_vs_manifest",
                    expected_manifest.get("sha256", ""),
                    snap["sha256"],
                )

    status = "PASS" if not mismatches else "FAIL"
    return {
        "artifacts_dir": str(artifacts_dir),
        "expected_order_branch": expected_order_branch,
        "reported_order_branch": reported_order_branch,
        "drift_detected": bool(drift_detected),
        "checks": checks,
        "snapshots": snapshots,
        "mismatches": mismatches,
        "status": status,
        "exit_code": 0 if status == "PASS" else 1,
        "captured_at_utc": _utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="S-02 immutable close gate (drift + branch provenance).")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--expected-order-branch", required=True)
    parser.add_argument("--reported-order-branch", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else artifacts_dir / "s02_close_gate_report.json"
    log_path = Path(args.log).expanduser().resolve() if args.log else artifacts_dir / "s02_close_gate.log"

    result = run_close_gate(artifacts_dir, args.expected_order_branch.strip(), args.reported_order_branch.strip())
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (
            f"TIMESTAMP={_utc_now()}\n"
            f"ARTIFACTS_DIR={result['artifacts_dir']}\n"
            f"EXPECTED_ORDER_BRANCH={result['expected_order_branch']}\n"
            f"REPORTED_ORDER_BRANCH={result['reported_order_branch']}\n"
            f"DRIFT_DETECTED={result['drift_detected']}\n"
            f"STATUS={result['status']}\n"
            f"EXIT_CODE={result['exit_code']}\n"
            f"MISMATCH_COUNT={len(result['mismatches'])}\n"
        ),
        encoding="utf-8",
    )
    print(payload, end="")
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
