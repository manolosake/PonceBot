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
        "sha256": _sha256(path) if exists else "",
    }


def run_verify(artifacts_dir: Path, manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files", [])
    if not isinstance(files, list):
        raise ValueError("manifest.files must be a list")

    issues: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    current: dict[str, Any] = {}

    for entry in files:
        if not isinstance(entry, dict):
            issues.append({"rule": "manifest_entry_object", "expected": "object", "actual": str(type(entry))})
            continue
        p = Path(str(entry.get("path", ""))).expanduser().resolve()
        key = p.name
        snap = _snapshot(p)
        current[key] = snap
        try:
            p.relative_to(artifacts_dir)
            path_within_artifacts = True
        except ValueError:
            path_within_artifacts = False

        exp_exists = bool(entry.get("exists", True))
        exp_size = int(entry.get("size_bytes", -1))
        exp_sha = str(entry.get("sha256", ""))

        checks[f"{key}::path_within_artifacts_dir"] = path_within_artifacts
        c1 = snap["exists"] == exp_exists
        c2 = snap["size_bytes"] == exp_size
        c3 = snap["sha256"] == exp_sha
        checks[f"{key}::exists_match"] = c1
        checks[f"{key}::size_match"] = c2
        checks[f"{key}::sha256_match"] = c3
        if not path_within_artifacts:
            issues.append(
                {
                    "rule": f"{key}::path_within_artifacts_dir",
                    "expected": str(artifacts_dir),
                    "actual": str(p),
                }
            )
        if not c1:
            issues.append({"rule": f"{key}::exists_match", "expected": exp_exists, "actual": snap["exists"]})
        if not c2:
            issues.append({"rule": f"{key}::size_match", "expected": exp_size, "actual": snap["size_bytes"]})
        if not c3:
            issues.append({"rule": f"{key}::sha256_match", "expected": exp_sha, "actual": snap["sha256"]})

    return {
        "status": "PASS" if not issues else "FAIL",
        "captured_at_utc": _utc_now(),
        "artifacts_dir": str(artifacts_dir),
        "manifest_path": str(manifest_path),
        "checks": checks,
        "current_snapshots": current,
        "errors": issues,
        "exit_code": 0 if not issues else 2,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-write verify against sealed bundle manifest.")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else artifacts_dir / "bundle_immutability_manifest.json"
    out_path = Path(args.out).expanduser().resolve() if args.out else artifacts_dir / "bundle_postwrite_verify_report.json"

    result = run_verify(artifacts_dir, manifest_path)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
