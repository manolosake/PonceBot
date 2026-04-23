#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _resolve_artifact_path(*, artifacts_dir: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path or "").strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (artifacts_dir / candidate).resolve()


def validate_evidence(
    *,
    artifacts_dir: Path,
    summary_name: str = "final_evidence.json",
    allow_empty_artifacts: bool = False,
    allow_missing_files: bool = False,
) -> tuple[bool, dict[str, Any]]:
    artifacts_root = artifacts_dir.expanduser().resolve()
    summary_path = (artifacts_root / summary_name).resolve()
    try:
        summary_path.relative_to(artifacts_root)
    except ValueError:
        return False, {
            "ok": False,
            "reason": "summary_outside_dir",
            "summary_path": str(summary_path),
        }
    if not summary_path.is_file():
        return False, {
            "ok": False,
            "reason": "summary_missing",
            "summary_path": str(summary_path),
        }

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, {
            "ok": False,
            "reason": "summary_invalid_json",
            "summary_path": str(summary_path),
            "error": str(exc),
        }

    if not isinstance(payload, dict):
        return False, {
            "ok": False,
            "reason": "summary_not_object",
            "summary_path": str(summary_path),
        }

    summary = str(payload.get("summary") or "").strip()
    if not summary:
        return False, {
            "ok": False,
            "reason": "summary_missing_text",
            "summary_path": str(summary_path),
        }

    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        return False, {
            "ok": False,
            "reason": "artifacts_not_list",
            "summary_path": str(summary_path),
        }

    artifacts = [str(item or "").strip() for item in raw_artifacts if str(item or "").strip()]
    if not artifacts and not allow_empty_artifacts:
        return False, {
            "ok": False,
            "reason": "artifacts_empty",
            "summary_path": str(summary_path),
        }

    missing_files: list[str] = []
    outside_dir: list[str] = []
    resolved_artifacts: list[str] = []
    for raw_path in artifacts:
        resolved = _resolve_artifact_path(artifacts_dir=artifacts_root, raw_path=raw_path)
        resolved_artifacts.append(str(resolved))
        try:
            resolved.relative_to(artifacts_root)
        except ValueError:
            outside_dir.append(str(resolved))
            continue
        if not allow_missing_files and not resolved.exists():
            missing_files.append(str(resolved))

    if outside_dir:
        return False, {
            "ok": False,
            "reason": "artifact_outside_dir",
            "summary_path": str(summary_path),
            "outside_artifacts": outside_dir,
        }

    if missing_files:
        return False, {
            "ok": False,
            "reason": "artifact_missing",
            "summary_path": str(summary_path),
            "missing_artifacts": missing_files,
        }

    return True, {
        "ok": True,
        "summary_path": str(summary_path),
        "artifact_count": len(artifacts),
        "artifacts": resolved_artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate backend final evidence bundle contract.")
    parser.add_argument("--artifacts-dir", required=True, help="directory that contains final_evidence.json")
    parser.add_argument("--summary-name", default="final_evidence.json", help="summary filename inside artifacts dir")
    parser.add_argument("--allow-empty-artifacts", action="store_true", help="accept empty artifacts list")
    parser.add_argument("--allow-missing-files", action="store_true", help="skip existence checks for listed artifacts")
    args = parser.parse_args(argv)

    ok, payload = validate_evidence(
        artifacts_dir=Path(args.artifacts_dir).expanduser().resolve(),
        summary_name=str(args.summary_name or "final_evidence.json").strip() or "final_evidence.json",
        allow_empty_artifacts=bool(args.allow_empty_artifacts),
        allow_missing_files=bool(args.allow_missing_files),
    )
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
