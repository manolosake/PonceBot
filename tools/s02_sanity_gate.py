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


def _kv_map(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        if "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def run_sanity_gate(artifacts_dir: Path) -> dict[str, Any]:
    required_files = [
        "git_status.txt",
        "changes.patch",
        "s02_trace_validation.json",
        "post_publish_check.json",
        "s02_final_manifest_signed.json",
        "s02_independent_check.json",
        "bundle_s02_summary.json",
        "verify.log",
        "preflight_report.json",
    ]
    critical_non_empty = {
        "git_status.txt",
        "changes.patch",
        "s02_trace_validation.json",
        "post_publish_check.json",
        "s02_final_manifest_signed.json",
        "s02_independent_check.json",
        "bundle_s02_summary.json",
    }

    errors: list[str] = []
    checks: dict[str, Any] = {}
    files: dict[str, Any] = {}

    for name in required_files:
        snap = _snapshot(artifacts_dir / name)
        files[name] = snap
        exists = bool(snap["exists"])
        checks[f"{name}::exists"] = exists
        if not exists:
            errors.append(f"missing required artifact: {name}")
            continue
        if name in critical_non_empty:
            non_empty = int(snap["size_bytes"]) > 0
            checks[f"{name}::size_gt_zero"] = non_empty
            if not non_empty:
                errors.append(f"critical artifact is empty: {name}")
        valid_hash = bool(snap["sha256"]) and len(str(snap["sha256"])) == 64
        checks[f"{name}::sha256_present"] = valid_hash
        if int(snap["size_bytes"]) > 0 and not valid_hash:
            errors.append(f"invalid sha256 for artifact: {name}")

    patch_path = artifacts_dir / "changes.patch"
    if patch_path.exists():
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        kv = _kv_map(patch_text)
        no_diff = kv.get("NO_DIFF", "").lower() == "true"
        checks["changes.patch::no_diff_mode"] = no_diff
        if no_diff:
            for key in ("captured_at_utc", "head_sha", "branch", "justification"):
                present = bool(kv.get(key, ""))
                checks[f"changes.patch::no_diff_{key}_present"] = present
                if not present:
                    errors.append(f"NO_DIFF missing required key: {key}")

    manifest_path = artifacts_dir / "s02_final_manifest_signed.json"
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("s02_final_manifest_signed.json is not valid JSON")
            manifest = {}
        manifest_files = manifest.get("files", {}) if isinstance(manifest, dict) else {}
        checks["manifest::files_present"] = isinstance(manifest_files, dict)
        if not isinstance(manifest_files, dict):
            errors.append("manifest.files missing or invalid")
        else:
            for name in ("git_status.txt", "changes.patch"):
                entry = manifest_files.get(name, {})
                snap = files.get(name, {})
                if not entry:
                    errors.append(f"manifest missing file entry: {name}")
                    continue
                if str(entry.get("path", "")) != str(snap.get("path", "")):
                    errors.append(f"manifest path mismatch for {name}")
                if int(entry.get("size_bytes", -1)) != int(snap.get("size_bytes", -2)):
                    errors.append(f"manifest size mismatch for {name}")
                if str(entry.get("sha256", "")) != str(snap.get("sha256", "")):
                    errors.append(f"manifest sha256 mismatch for {name}")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "exit_code": 0 if status == "PASS" else 1,
        "artifacts_dir": str(artifacts_dir),
        "required_files": required_files,
        "critical_non_empty": sorted(critical_non_empty),
        "checks": checks,
        "files": files,
        "errors": errors,
        "captured_at_utc": _utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity gate for S-02 close before PASS publication.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out).expanduser().resolve() if args.out else artifacts_dir / "s02_sanity_gate.json"
    log_path = Path(args.log).expanduser().resolve() if args.log else artifacts_dir / "s02_sanity_gate.log"

    result = run_sanity_gate(artifacts_dir)
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    log_path.write_text(
        (
            f"TIMESTAMP={_utc_now()}\n"
            f"ARTIFACTS_DIR={artifacts_dir}\n"
            f"STATUS={result['status']}\n"
            f"ERROR_COUNT={len(result['errors'])}\n"
            f"EXIT_CODE={result['exit_code']}\n"
            f"ERRORS={json.dumps(result['errors'], ensure_ascii=False)}\n"
        ),
        encoding="utf-8",
    )
    print(payload, end="")
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
