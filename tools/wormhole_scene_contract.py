#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {
    "contract_name",
    "contract_version",
    "scene_version",
    "seed",
    "quality_presets",
    "viewport_defaults",
    "parameter_ranges",
    "traceability",
}
REQUIRED_PRESETS = {"cinematic", "balanced", "performance"}
REQUIRED_VIEWPORTS = {"desktop", "tablet", "mobile"}
CRITICAL_SCENE_KEYS = {
    "particle_density",
    "ring_layers",
    "meridian_lines",
    "volumetric_intensity",
    "distortion_strength",
    "parallax_depth",
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _git(cmd: list[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return ""
    return (p.stdout or "").strip()


def _load_contract(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("contract must be a JSON object")
    return data


def validate_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    missing = sorted(REQUIRED_TOP_LEVEL - set(contract.keys()))
    if missing:
        errors.append(f"missing top-level keys: {', '.join(missing)}")

    presets = contract.get("quality_presets")
    if not isinstance(presets, dict):
        errors.append("quality_presets must be an object")
    else:
        missing_presets = sorted(REQUIRED_PRESETS - set(presets.keys()))
        if missing_presets:
            errors.append(f"missing quality presets: {', '.join(missing_presets)}")
        for name in REQUIRED_PRESETS & set(presets.keys()):
            p = presets.get(name)
            if not isinstance(p, dict):
                errors.append(f"quality_presets.{name} must be an object")
                continue
            missing_keys = sorted(CRITICAL_SCENE_KEYS - set(p.keys()))
            if missing_keys:
                errors.append(f"quality_presets.{name} missing keys: {', '.join(missing_keys)}")

    viewport_defaults = contract.get("viewport_defaults")
    if not isinstance(viewport_defaults, dict):
        errors.append("viewport_defaults must be an object")
    else:
        missing_viewports = sorted(REQUIRED_VIEWPORTS - set(viewport_defaults.keys()))
        if missing_viewports:
            errors.append(f"viewport_defaults missing keys: {', '.join(missing_viewports)}")
        if isinstance(presets, dict):
            for vp in REQUIRED_VIEWPORTS & set(viewport_defaults.keys()):
                val = viewport_defaults.get(vp)
                if val not in presets:
                    errors.append(f"viewport_defaults.{vp} references unknown preset: {val}")

    seed = contract.get("seed")
    if not isinstance(seed, dict):
        errors.append("seed must be an object")
    else:
        if "value" not in seed:
            errors.append("seed.value is required")
        elif not isinstance(seed.get("value"), int):
            errors.append("seed.value must be integer")

    return errors


def _build_trace(
    contract: dict[str, Any],
    *,
    root: Path,
    ticket_id: str,
    expected_branch: str,
    artifacts_dir: Path,
    canonical_json_sha256: str,
) -> dict[str, Any]:
    canonical = _canonical_json(contract)
    scene_signature = _sha256_text(canonical)
    if scene_signature != canonical_json_sha256:
        raise ValueError("canonical hash mismatch while building trace")

    head_sha = _git(["git", "rev-parse", "HEAD"], root)
    reported_branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)

    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "ticket_id": ticket_id,
        "expected_order_branch": expected_branch,
        "reported_branch": reported_branch,
        "branch_matches_expected": bool(expected_branch and reported_branch == expected_branch),
        "head_sha": head_sha,
        "artifacts_dir": str(artifacts_dir),
        "contract": {
            "name": contract.get("contract_name"),
            "version": contract.get("contract_version"),
            "scene_version": contract.get("scene_version"),
            "seed": contract.get("seed"),
        },
        "scene_signature_sha256": scene_signature,
        "scene_signature_semantics": "canonical_json_sha256",
    }


def cmd_validate(args: argparse.Namespace) -> int:
    contract_path = Path(args.contract).resolve()
    contract = _load_contract(contract_path)
    source_file_sha256 = _sha256_file(contract_path)
    errors = validate_contract(contract)
    payload = {
        "status": "PASS" if not errors else "FAIL",
        "contract_path": str(contract_path),
        "errors": errors,
    }
    out = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        rp = Path(args.report).resolve()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(out + "\n", encoding="utf-8")
    print(out)
    return 0 if not errors else 2


def cmd_export(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    contract_path = Path(args.contract).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    contract = _load_contract(contract_path)
    errors = validate_contract(contract)
    if errors:
        payload = {
            "status": "FAIL",
            "reason": "invalid_contract",
            "errors": errors,
            "contract_path": str(contract_path),
        }
        out = json.dumps(payload, ensure_ascii=False, indent=2)
        report_path = artifacts_dir / "wormhole_scene_contract_export_report.json"
        report_path.write_text(out + "\n", encoding="utf-8")
        print(out)
        return 2

    contract_out_path = artifacts_dir / "wormhole_scene_contract.json"
    canonical = _canonical_json(contract)
    canonical_json_sha256 = _sha256_text(canonical)
    contract_out_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    exported_file_sha256 = _sha256_file(contract_out_path)

    trace = _build_trace(
        contract,
        root=root,
        ticket_id=args.ticket_id,
        expected_branch=args.expected_branch,
        artifacts_dir=artifacts_dir,
        canonical_json_sha256=canonical_json_sha256,
    )

    trace_path = artifacts_dir / "wormhole_scene_trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    signature_path = artifacts_dir / "wormhole_scene_contract.sha256"
    signature_path.write_text(canonical_json_sha256 + "\n", encoding="utf-8")

    report = {
        "status": "PASS",
        "contract_source": str(contract_path),
        "contract_exported": str(contract_out_path),
        "trace_report": str(trace_path),
        "signature_file": str(signature_path),
        "hash_semantics": {
            "canonical_json_sha256": "SHA-256 of canonicalized JSON object (sorted keys, compact separators); stable across whitespace and key ordering in files.",
            "file_byte_sha256": "SHA-256 of raw file bytes; sensitive to formatting, key order, and newline differences.",
        },
        "hashes": {
            "canonical_json_sha256": canonical_json_sha256,
            "contract_source_file_sha256": source_file_sha256,
            "contract_export_file_sha256": exported_file_sha256,
        },
        "scene_signature_sha256": trace["scene_signature_sha256"],
        "branch_matches_expected": trace["branch_matches_expected"],
    }
    report_path = artifacts_dir / "wormhole_scene_contract_export_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Validate and export wormhole scene contract with reproducible trace")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate contract schema")
    v.add_argument("--contract", default="docs/contracts/wormhole_scene_contract.v1.json")
    v.add_argument("--report", default="")
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("export", help="export contract and traceability artifacts")
    e.add_argument("--repo-root", default=".")
    e.add_argument("--contract", default="docs/contracts/wormhole_scene_contract.v1.json")
    e.add_argument("--artifacts-dir", required=True)
    e.add_argument("--ticket-id", required=True)
    e.add_argument("--expected-branch", required=True)
    e.set_defaults(func=cmd_export)

    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
