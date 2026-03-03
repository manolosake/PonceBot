#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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


def validate_s02_trace(artifacts_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}

    git_status = artifacts_dir / "git_status.txt"
    patch = artifacts_dir / "changes.patch"

    checks["git_status_exists"] = git_status.exists() and git_status.is_file()
    checks["changes_patch_exists"] = patch.exists() and patch.is_file()
    if not checks["git_status_exists"]:
        errors.append("missing git_status.txt")
    if not checks["changes_patch_exists"]:
        errors.append("missing changes.patch")
    if errors:
        return {
            "status": "FAIL",
            "errors": errors,
            "checks": checks,
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

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": checks,
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
