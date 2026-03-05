#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"exists": False, "size_bytes": 0, "sha256": ""}
    raw = path.read_bytes()
    return {"exists": True, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-seal integrity gate for artifact bundles.")
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--files", required=True, help="Comma-separated relative paths to enforce.")
    ap.add_argument("--manifest", required=True, help="Manifest path used for seal/verify.")
    ap.add_argument("--report", required=True)
    ap.add_argument("--mode", choices=("seal", "verify"), required=True)
    ap.add_argument("--pass-summary", default="", help="Optional summary JSON to enforce non-empty after PASS.")
    args = ap.parse_args()

    root_dir = Path(args.root_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    report_path = Path(args.report).resolve()
    files = [x.strip() for x in args.files.split(",") if x.strip()]

    current = {rel: file_meta(root_dir / rel) for rel in files}
    errors: list[dict[str, Any]] = []
    manifest_obj: dict[str, Any] = {}

    if args.mode == "seal":
        manifest_obj = {
            "schema_version": 1,
            "sealed_at_utc": utc_now(),
            "root_dir": str(root_dir),
            "files": current,
        }
        manifest_path.write_text(json.dumps(manifest_obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    else:
        if not manifest_path.exists():
            errors.append({"type": "missing_manifest", "path": str(manifest_path)})
        else:
            manifest_obj = load_json(manifest_path)
            expected = manifest_obj.get("files", {})
            for rel in files:
                exp = expected.get(rel, {"exists": False, "size_bytes": 0, "sha256": ""})
                cur = current.get(rel, {"exists": False, "size_bytes": 0, "sha256": ""})
                if exp != cur:
                    errors.append({"type": "root_manifest_mismatch", "file": rel, "expected": exp, "current": cur})

    pass_summary_enforced = False
    if args.pass_summary:
        ps = Path(args.pass_summary).resolve()
        if ps.exists():
            obj = load_json(ps)
            if str(obj.get("status", "")).upper() == "PASS":
                pass_summary_enforced = True
                for rel, meta in current.items():
                    if (not meta.get("exists")) or int(meta.get("size_bytes", 0)) <= 0:
                        errors.append(
                            {
                                "type": "post_pass_blank_or_missing",
                                "file": rel,
                                "current": meta,
                                "reason": "summary_reported_pass_but_file_is_blank_or_missing",
                            }
                        )

    report = {
        "check": "postseal_integrity_gate",
        "mode": args.mode,
        "root_dir": str(root_dir),
        "manifest": str(manifest_path),
        "files": files,
        "current": current,
        "manifest_loaded": bool(manifest_obj),
        "pass_summary_enforced": pass_summary_enforced,
        "checked_at_utc": utc_now(),
        "errors": errors,
        "root_manifest_consistency": len(errors) == 0,
        "status": "PASS" if not errors else "FAIL",
        "exit_code": 0 if not errors else 2,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
