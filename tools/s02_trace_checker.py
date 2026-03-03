#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _kv_map(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot(path: Path) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    data: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "size_bytes": 0,
        "sha256": "",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not exists:
        return data
    st = path.stat()
    data["size_bytes"] = int(st.st_size)
    if st.st_size > 0:
        data["sha256"] = _sha256(path)
    return data


def _valid_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value or ""))


def validate_s02_trace(artifacts_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}

    git_status = artifacts_dir / "git_status.txt"
    patch = artifacts_dir / "changes.patch"

    snap_before = {
        "git_status.txt": _snapshot(git_status),
        "changes.patch": _snapshot(patch),
    }

    checks["git_status_exists"] = snap_before["git_status.txt"]["exists"]
    checks["changes_patch_exists"] = snap_before["changes.patch"]["exists"]
    if not checks["git_status_exists"]:
        errors.append("missing git_status.txt")
    if not checks["changes_patch_exists"]:
        errors.append("missing changes.patch")
    if errors:
        return {
            "status": "FAIL",
            "errors": errors,
            "checks": checks,
            "files": snap_before,
            "artifacts_dir": str(artifacts_dir),
        }

    gs_text = git_status.read_text(encoding="utf-8", errors="replace")
    gs_non_empty = bool(gs_text.strip())
    checks["git_status_non_empty"] = gs_non_empty
    if not gs_non_empty:
        errors.append("git_status.txt is empty")

    patch_text = patch.read_text(encoding="utf-8", errors="replace")
    patch_non_empty = bool(patch_text.strip())
    checks["changes_patch_non_empty"] = patch_non_empty
    if not patch_non_empty:
        errors.append("changes.patch is empty")
        return {
            "status": "FAIL",
            "errors": errors,
            "checks": checks,
            "artifacts_dir": str(artifacts_dir),
        }

    kv = _kv_map(patch_text)
    no_diff = kv.get("NO_DIFF", "").lower() == "true"
    checks["no_diff_mode"] = no_diff

    if no_diff:
        required = ("captured_at_utc", "head_sha", "branch", "justification")
        for key in required:
            ok = bool(kv.get(key, ""))
            checks[f"no_diff_{key}_present"] = ok
            if not ok:
                errors.append(f"NO_DIFF missing {key}")
        ts = kv.get("captured_at_utc", "")
        sha = kv.get("head_sha", "")
        checks["no_diff_timestamp_valid"] = bool(TS_RE.match(ts))
        checks["no_diff_sha_valid"] = bool(SHA_RE.match(sha))
        if ts and not checks["no_diff_timestamp_valid"]:
            errors.append("NO_DIFF captured_at_utc must be UTC ISO8601 (YYYY-MM-DDTHH:MM:SSZ)")
        if sha and not checks["no_diff_sha_valid"]:
            errors.append("NO_DIFF head_sha must be 40-char lowercase hex")

    # Hard guard: size/hash must be present and valid for both files.
    for fname, snap in snap_before.items():
        size = int(snap["size_bytes"])
        digest = str(snap["sha256"])
        checks[f"{fname}_size_gt_zero"] = size > 0
        if size <= 0:
            errors.append(f"{fname} size_bytes must be > 0")
        checks[f"{fname}_sha256_valid"] = _valid_sha256(digest)
        if not _valid_sha256(digest):
            errors.append(f"{fname} sha256 missing/invalid")

    # Re-snapshot at end to detect desynchronization during validation.
    snap_after = {
        "git_status.txt": _snapshot(git_status),
        "changes.patch": _snapshot(patch),
    }
    for fname in ("git_status.txt", "changes.patch"):
        before = snap_before[fname]
        after = snap_after[fname]
        unchanged = (before["size_bytes"] == after["size_bytes"]) and (before["sha256"] == after["sha256"])
        checks[f"{fname}_stable_during_validation"] = unchanged
        if not unchanged:
            errors.append(f"{fname} changed during validation (possible stale report)")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": checks,
        "files": snap_before,
        "files_after_validation": snap_after,
        "artifacts_dir": str(artifacts_dir),
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate S-02 traceability bundle files.")
    ap.add_argument("--artifacts-dir", required=True, help="Path to bundle artifacts directory")
    ap.add_argument("--out", default="", help="Optional JSON output path")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    result = validate_s02_trace(artifacts_dir)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)

    out_path = Path(args.out).expanduser().resolve() if args.out else artifacts_dir / "s02_trace_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload + "\n", encoding="utf-8")

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
