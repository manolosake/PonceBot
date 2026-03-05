#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_integrity(
    artifacts_dir: Path,
    contract_source: Path | None = None,
    *,
    expected_branch: str = "",
    expected_ticket_id: str = "",
) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}

    report_path = artifacts_dir / "wormhole_scene_contract_export_report.json"
    if not report_path.exists():
        return {
            "status": "FAIL",
            "reason": "missing_export_report",
            "errors": [f"missing file: {report_path}"],
            "artifacts_dir": str(artifacts_dir),
            "checks": checks,
        }

    report = _load_json(report_path)
    checks["report_status_pass"] = report.get("status") == "PASS"
    if not checks["report_status_pass"]:
        errors.append("export report status must be PASS")

    expected_semantics = {
        "canonical_json_sha256": "SHA-256 of canonicalized JSON object (sorted keys, compact separators); stable across whitespace and key ordering in files.",
        "file_byte_sha256": "SHA-256 of raw file bytes; sensitive to formatting, key order, and newline differences.",
    }
    checks["hash_semantics_present"] = report.get("hash_semantics") == expected_semantics
    if not checks["hash_semantics_present"]:
        errors.append("export report hash_semantics block is missing or unexpected")

    hashes = report.get("hashes")
    checks["hashes_block_present"] = isinstance(hashes, dict)
    if not checks["hashes_block_present"]:
        errors.append("export report must include hashes object")
        hashes = {}

    contract_exported = Path(str(report.get("contract_exported", "")))
    signature_file = Path(str(report.get("signature_file", "")))
    trace_report = Path(str(report.get("trace_report", "")))
    if contract_source is None:
        contract_source_value = report.get("contract_source", "")
        contract_source = Path(str(contract_source_value))
    assert contract_source is not None

    for label, path in [
        ("contract_source", contract_source),
        ("contract_exported", contract_exported),
        ("signature_file", signature_file),
        ("trace_report", trace_report),
    ]:
        checks[f"{label}_exists"] = path.exists()
        if not checks[f"{label}_exists"]:
            errors.append(f"missing file: {path}")

    if any(not checks[k] for k in checks if k.endswith("_exists")):
        return {
            "status": "FAIL",
            "reason": "missing_required_files",
            "errors": errors,
            "artifacts_dir": str(artifacts_dir),
            "checks": checks,
        }

    source_payload = _load_json(contract_source)
    exported_payload = _load_json(contract_exported)
    canonical_source = _sha256_text(_canonical_json(source_payload))
    canonical_exported = _sha256_text(_canonical_json(exported_payload))
    source_file_sha = _sha256_file(contract_source)
    export_file_sha = _sha256_file(contract_exported)
    signature_sha = signature_file.read_text(encoding="utf-8").strip()

    trace = _load_json(trace_report)
    trace_scene_signature = trace.get("scene_signature_sha256", "")
    reported_branch = str(trace.get("reported_branch", "") or "")
    reported_ticket_id = str(trace.get("ticket_id", "") or "")
    reported_artifacts_dir = str(trace.get("artifacts_dir", "") or "")

    checks["canonical_source_matches_report"] = canonical_source == hashes.get("canonical_json_sha256")
    if not checks["canonical_source_matches_report"]:
        errors.append("canonical_json_sha256 mismatch against contract_source")

    checks["canonical_export_matches_report"] = canonical_exported == hashes.get("canonical_json_sha256")
    if not checks["canonical_export_matches_report"]:
        errors.append("canonical_json_sha256 mismatch against contract_exported")

    checks["source_file_hash_matches_report"] = source_file_sha == hashes.get("contract_source_file_sha256")
    if not checks["source_file_hash_matches_report"]:
        errors.append("contract_source_file_sha256 mismatch")

    checks["export_file_hash_matches_report"] = export_file_sha == hashes.get("contract_export_file_sha256")
    if not checks["export_file_hash_matches_report"]:
        errors.append("contract_export_file_sha256 mismatch")

    checks["signature_matches_canonical"] = signature_sha == hashes.get("canonical_json_sha256")
    if not checks["signature_matches_canonical"]:
        errors.append("signature file does not match canonical_json_sha256")

    checks["trace_matches_canonical"] = trace_scene_signature == hashes.get("canonical_json_sha256")
    if not checks["trace_matches_canonical"]:
        errors.append("trace scene_signature_sha256 does not match canonical_json_sha256")

    checks["report_scene_signature_matches_canonical"] = (
        report.get("scene_signature_sha256") == hashes.get("canonical_json_sha256")
    )
    if not checks["report_scene_signature_matches_canonical"]:
        errors.append("report scene_signature_sha256 does not match canonical_json_sha256")

    checks["trace_artifacts_dir_matches_runtime"] = reported_artifacts_dir == str(artifacts_dir)
    if not checks["trace_artifacts_dir_matches_runtime"]:
        errors.append("trace artifacts_dir does not match evaluated artifacts_dir")

    if expected_branch:
        checks["expected_branch_matches_trace"] = reported_branch == expected_branch
        if not checks["expected_branch_matches_trace"]:
            errors.append(
                f"reported_branch mismatch: expected '{expected_branch}' observed '{reported_branch}'"
            )

    if expected_ticket_id:
        checks["expected_ticket_id_matches_trace"] = reported_ticket_id == expected_ticket_id
        if not checks["expected_ticket_id_matches_trace"]:
            errors.append(
                f"ticket_id mismatch: expected '{expected_ticket_id}' observed '{reported_ticket_id}'"
            )

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "artifacts_dir": str(artifacts_dir),
        "expected_branch": expected_branch,
        "reported_branch": reported_branch,
        "expected_ticket_id": expected_ticket_id,
        "reported_ticket_id": reported_ticket_id,
        "checks": checks,
        "canonical_json_sha256": hashes.get("canonical_json_sha256", ""),
        "contract_source_file_sha256": source_file_sha,
        "contract_export_file_sha256": export_file_sha,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Integrity gate for wormhole scene contract export artifacts")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--contract-source", default="")
    ap.add_argument("--expected-branch", default="")
    ap.add_argument("--expected-ticket-id", default="")
    ap.add_argument("--report-out", default="")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    contract_source = Path(args.contract_source).resolve() if args.contract_source else None

    result = evaluate_integrity(
        artifacts_dir,
        contract_source=contract_source,
        expected_branch=args.expected_branch,
        expected_ticket_id=args.expected_ticket_id,
    )
    out = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report_out:
        report_out = Path(args.report_out).resolve()
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(out + "\n", encoding="utf-8")
    print(out)
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
