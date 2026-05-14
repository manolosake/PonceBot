#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import ntpath
import re
import sys
from pathlib import Path
from typing import Any

DELIVERED_OUTCOMES = {"shipped_to_main", "published_project"}
ROOT_CAUSED_TERMINAL_OUTCOMES = {
    "blocked_need_operator",
    "rejected_low_value",
    "failed_root_caused",
}
VALID_OUTCOMES = DELIVERED_OUTCOMES | ROOT_CAUSED_TERMINAL_OUTCOMES
FACTORY_CAPABILITY_FAMILIES = {
    "choose",
    "selection",
    "delegate",
    "validation",
    "merge",
    "ship",
    "deploy",
    "learn",
    "governor",
    "studio",
    "autonomy",
    "outcome",
    "delivery",
    "quality",
    "revenue",
}
FACTORY_DELTA_PLACEHOLDERS = {
    "done",
    "n/a",
    "na",
    "none",
    "todo",
    "tbd",
    "proof",
    "proof.log",
    "test",
    "tests",
    "validation",
    "quality",
    "delivery",
    "ship",
    "learn",
    "merge",
    "deploy",
    "selection",
    "delegate",
    "outcome",
    "studio",
    "governor",
    "autonomy",
    "revenue",
}
COMMERCIAL_EVIDENCE_PLACEHOLDERS = FACTORY_DELTA_PLACEHOLDERS | {
    "backend",
    "backend docs",
    "docs",
    "documentation",
    "file",
    "log",
    "proof artifact",
    "updated",
    "updated docs",
}
COMMERCIAL_RELEVANCE_TERMS = {
    "money",
    "revenue",
    "sell",
    "rent",
    "pricing",
    "buyer",
    "customer",
    "save",
    "savings",
    "less waste",
    "fewer ghost",
    "profitable",
    "retention",
    "lead",
    "factory leverage",
    "selection quality",
    "delivery reliability",
    "deployability",
    "learning",
    "speed",
}
COMMERCIAL_RELEVANCE_TOKEN_VARIANTS = {
    "money": {"money"},
    "revenue": {"revenue", "revenues"},
    "sell": {"sell", "sells", "selling", "sold"},
    "rent": {"rent", "rents", "rental", "rented", "renting"},
    "pricing": {"price", "prices", "priced", "pricing"},
    "buyer": {"buyer", "buyers"},
    "customer": {"customer", "customers"},
    "save": {"save", "saves", "saved", "saving"},
    "savings": {"savings"},
    "profitable": {"profit", "profits", "profitable", "profitability"},
    "retention": {"retention"},
    "lead": {"lead", "leads"},
    "deployability": {"deployability", "deployable"},
    "learning": {"learn", "learns", "learned", "learning"},
    "speed": {"speed", "speeds"},
}


def _is_absolute_artifact_path(raw_path: str) -> bool:
    # Use both local-path semantics and Windows semantics so checks are platform-agnostic.
    return Path(raw_path).is_absolute() or ntpath.isabs(raw_path)


def _artifact_candidate_path(raw_path: str) -> Path:
    # Interpret Windows-style separators consistently for platform-agnostic safety checks.
    return Path(raw_path.replace("\\", "/"))


def _resolve_artifact_path(*, artifacts_dir: Path, raw_path: str) -> Path:
    candidate = _artifact_candidate_path(str(raw_path or "").strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (artifacts_dir / candidate).resolve()


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_non_empty_text_list(value: Any) -> bool:
    return isinstance(value, list) and any(_has_text(item) for item in value)


def _is_substantive_factory_delta_text(value: Any) -> bool:
    if not _has_text(value):
        return False
    text = str(value).strip()
    if text.lower() in FACTORY_DELTA_PLACEHOLDERS:
        return False
    words = text.split()
    non_space_chars = sum(1 for char in text if not char.isspace())
    return len(words) >= 2 or non_space_chars >= 16


def _is_substantive_commercial_evidence_text(value: Any) -> bool:
    if not _has_text(value):
        return False
    text = str(value).strip()
    if text.lower() in COMMERCIAL_EVIDENCE_PLACEHOLDERS:
        return False
    words = text.split()
    non_space_chars = sum(1 for char in text if not char.isspace())
    return len(words) >= 2 or non_space_chars >= 16


def _is_commercially_relevant_text(*values: Any) -> bool:
    normalized = " ".join(str(value or "").strip().lower() for value in values)
    for phrase in (term for term in COMMERCIAL_RELEVANCE_TERMS if " " in term):
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
        if re.search(pattern, normalized):
            return True

    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    for term in (term for term in COMMERCIAL_RELEVANCE_TERMS if " " not in term):
        if tokens & COMMERCIAL_RELEVANCE_TOKEN_VARIANTS.get(term, {term}):
            return True
    return False


def _parse_bool_flag(value: str | bool | None) -> bool:
    if value is True or value is None:
        return True
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected one of on/off, true/false, yes/no, or 1/0; got {value!r}")


def _has_diff_evidence(delivery_contract: dict[str, Any]) -> bool:
    return (
        _has_text(delivery_contract.get("diff_artifact"))
        or _has_text(delivery_contract.get("diffstat"))
        or _has_non_empty_text_list(delivery_contract.get("changed_files"))
    )


def _has_validation_evidence(delivery_contract: dict[str, Any]) -> bool:
    return _has_non_empty_text_list(delivery_contract.get("validation_artifacts")) or _has_text(
        delivery_contract.get("validation")
    )


def _has_ship_evidence(delivery_contract: dict[str, Any]) -> bool:
    return _has_text(delivery_contract.get("ship_evidence")) or _has_non_empty_text_list(
        delivery_contract.get("ship_artifacts")
    )


def _factory_delta_error(
    *,
    payload: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any] | None:
    factory_delta = payload.get("factory_delta")
    if not isinstance(factory_delta, dict):
        return {
            "ok": False,
            "reason": "factory_delta_missing",
            "summary_path": str(summary_path),
        }

    capability_changed = factory_delta.get("capability_changed")
    if not _has_text(capability_changed):
        return {
            "ok": False,
            "reason": "factory_delta_capability_changed_missing",
            "summary_path": str(summary_path),
        }
    if not _has_text(factory_delta.get("measurable_delta")):
        return {
            "ok": False,
            "reason": "factory_delta_measurable_delta_missing",
            "summary_path": str(summary_path),
        }
    if not _has_text(factory_delta.get("evidence")):
        return {
            "ok": False,
            "reason": "factory_delta_evidence_missing",
            "summary_path": str(summary_path),
        }
    if not _is_substantive_factory_delta_text(capability_changed):
        return {
            "ok": False,
            "reason": "factory_delta_capability_changed_too_generic",
            "summary_path": str(summary_path),
            "capability_changed": str(capability_changed).strip(),
        }
    measurable_delta = factory_delta.get("measurable_delta")
    if not _is_substantive_factory_delta_text(measurable_delta):
        return {
            "ok": False,
            "reason": "factory_delta_measurable_delta_too_generic",
            "summary_path": str(summary_path),
            "measurable_delta": str(measurable_delta).strip(),
        }
    evidence = factory_delta.get("evidence")
    if not _is_substantive_factory_delta_text(evidence):
        return {
            "ok": False,
            "reason": "factory_delta_evidence_too_generic",
            "summary_path": str(summary_path),
            "evidence": str(evidence).strip(),
        }

    normalized_capability = str(capability_changed).strip().lower()
    if not any(family in normalized_capability for family in FACTORY_CAPABILITY_FAMILIES):
        return {
            "ok": False,
            "reason": "factory_delta_capability_not_factory_relevant",
            "summary_path": str(summary_path),
            "capability_changed": str(capability_changed).strip(),
        }

    return None


def _commercial_evidence_error(
    *,
    payload: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any] | None:
    commercial_evidence = payload.get("commercial_evidence")
    if not isinstance(commercial_evidence, dict):
        return {
            "ok": False,
            "reason": "commercial_evidence_missing",
            "summary_path": str(summary_path),
        }

    monetization_path = commercial_evidence.get("monetization_or_savings_path")
    if not _has_text(monetization_path):
        return {
            "ok": False,
            "reason": "commercial_evidence_monetization_path_missing",
            "summary_path": str(summary_path),
        }
    evidence = commercial_evidence.get("evidence")
    if not _has_text(evidence):
        return {
            "ok": False,
            "reason": "commercial_evidence_evidence_missing",
            "summary_path": str(summary_path),
        }
    if not _is_substantive_commercial_evidence_text(monetization_path):
        return {
            "ok": False,
            "reason": "commercial_evidence_monetization_path_too_generic",
            "summary_path": str(summary_path),
            "monetization_or_savings_path": str(monetization_path).strip(),
        }
    if not _is_substantive_commercial_evidence_text(evidence):
        return {
            "ok": False,
            "reason": "commercial_evidence_evidence_too_generic",
            "summary_path": str(summary_path),
            "evidence": str(evidence).strip(),
        }
    if not _is_commercially_relevant_text(monetization_path, evidence):
        return {
            "ok": False,
            "reason": "commercial_evidence_not_commercially_relevant",
            "summary_path": str(summary_path),
        }

    return None


def _check_artifact_references(
    *,
    artifacts_root: Path,
    artifacts: list[str],
    allow_missing_files: bool,
) -> dict[str, list[str]]:
    missing_files: list[str] = []
    outside_dir: list[str] = []
    non_files: list[str] = []
    empty_files: list[str] = []
    resolved_artifacts: list[str] = []
    for raw_path in artifacts:
        if _is_absolute_artifact_path(raw_path):
            outside_dir.append(raw_path)
            continue
        resolved = _resolve_artifact_path(artifacts_dir=artifacts_root, raw_path=raw_path)
        resolved_artifacts.append(str(resolved))
        try:
            resolved.relative_to(artifacts_root)
        except ValueError:
            outside_dir.append(str(resolved))
            continue
        if not allow_missing_files and not resolved.exists():
            missing_files.append(str(resolved))
            continue
        if resolved.exists() and not resolved.is_file():
            non_files.append(str(resolved))
            continue
        if resolved.is_file() and resolved.stat().st_size == 0:
            empty_files.append(str(resolved))

    return {
        "resolved_artifacts": resolved_artifacts,
        "outside_dir": outside_dir,
        "missing_files": missing_files,
        "non_files": non_files,
        "empty_files": empty_files,
    }


def _nested_contract_artifacts(delivery_contract: dict[str, Any]) -> list[str]:
    artifacts: list[str] = []
    diff_artifact = delivery_contract.get("diff_artifact")
    if _has_text(diff_artifact):
        artifacts.append(str(diff_artifact).strip())

    for key in ("validation_artifacts", "ship_artifacts"):
        raw_artifacts = delivery_contract.get(key)
        if not isinstance(raw_artifacts, list):
            continue
        artifacts.extend(str(item or "").strip() for item in raw_artifacts if str(item or "").strip())

    return artifacts


def _delivery_contract_artifact_error(
    *,
    summary_path: Path,
    outcome: str,
    violations: dict[str, list[str]],
) -> dict[str, Any] | None:
    if violations["outside_dir"]:
        return {
            "ok": False,
            "reason": "delivery_contract_artifact_outside_dir",
            "summary_path": str(summary_path),
            "outcome": outcome,
            "outside_artifacts": violations["outside_dir"],
        }
    if violations["missing_files"]:
        return {
            "ok": False,
            "reason": "delivery_contract_artifact_missing",
            "summary_path": str(summary_path),
            "outcome": outcome,
            "missing_artifacts": violations["missing_files"],
        }
    if violations["non_files"]:
        return {
            "ok": False,
            "reason": "delivery_contract_artifact_not_file",
            "summary_path": str(summary_path),
            "outcome": outcome,
            "not_file_artifacts": violations["non_files"],
        }
    if violations["empty_files"]:
        return {
            "ok": False,
            "reason": "delivery_contract_artifact_empty",
            "summary_path": str(summary_path),
            "outcome": outcome,
            "empty_artifacts": violations["empty_files"],
        }
    return None


def validate_evidence(
    *,
    artifacts_dir: Path,
    summary_name: str = "final_evidence.json",
    allow_empty_artifacts: bool = False,
    allow_missing_files: bool = False,
    require_factory_delta: bool = False,
    require_commercial_evidence: bool = False,
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

    next_action = payload.get("next_action")
    if next_action is not None:
        return False, {
            "ok": False,
            "reason": "next_action_open",
            "summary_path": str(summary_path),
            "next_action": next_action,
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

    artifact_violations = _check_artifact_references(
        artifacts_root=artifacts_root,
        artifacts=artifacts,
        allow_missing_files=allow_missing_files,
    )

    if artifact_violations["outside_dir"]:
        return False, {
            "ok": False,
            "reason": "artifact_outside_dir",
            "summary_path": str(summary_path),
            "outside_artifacts": artifact_violations["outside_dir"],
        }

    if artifact_violations["missing_files"]:
        return False, {
            "ok": False,
            "reason": "artifact_missing",
            "summary_path": str(summary_path),
            "missing_artifacts": artifact_violations["missing_files"],
        }

    if artifact_violations["non_files"]:
        return False, {
            "ok": False,
            "reason": "artifact_not_file",
            "summary_path": str(summary_path),
            "not_file_artifacts": artifact_violations["non_files"],
        }

    if artifact_violations["empty_files"]:
        return False, {
            "ok": False,
            "reason": "artifact_empty",
            "summary_path": str(summary_path),
            "empty_artifacts": artifact_violations["empty_files"],
        }

    outcome = payload.get("outcome")
    if not _has_text(outcome):
        return False, {
            "ok": False,
            "reason": "outcome_missing",
            "summary_path": str(summary_path),
        }
    outcome = str(outcome).strip()
    if outcome not in VALID_OUTCOMES:
        return False, {
            "ok": False,
            "reason": "outcome_invalid",
            "summary_path": str(summary_path),
            "outcome": outcome,
        }

    if outcome in DELIVERED_OUTCOMES:
        delivery_contract = payload.get("delivery_contract")
        if not isinstance(delivery_contract, dict):
            return False, {
                "ok": False,
                "reason": "delivery_contract_missing",
                "summary_path": str(summary_path),
                "outcome": outcome,
            }
        if not _has_text(delivery_contract.get("branch")):
            return False, {
                "ok": False,
                "reason": "branch_missing",
                "summary_path": str(summary_path),
                "outcome": outcome,
            }
        if not _has_diff_evidence(delivery_contract):
            return False, {
                "ok": False,
                "reason": "diff_evidence_missing",
                "summary_path": str(summary_path),
                "outcome": outcome,
            }
        if not _has_validation_evidence(delivery_contract):
            return False, {
                "ok": False,
                "reason": "validation_evidence_missing",
                "summary_path": str(summary_path),
                "outcome": outcome,
            }
        if not _has_ship_evidence(delivery_contract):
            return False, {
                "ok": False,
                "reason": "ship_evidence_missing",
                "summary_path": str(summary_path),
                "outcome": outcome,
            }
        nested_artifacts = _nested_contract_artifacts(delivery_contract)
        nested_violations = _check_artifact_references(
            artifacts_root=artifacts_root,
            artifacts=nested_artifacts,
            allow_missing_files=False,
        )
        nested_error = _delivery_contract_artifact_error(
            summary_path=summary_path,
            outcome=outcome,
            violations=nested_violations,
        )
        if nested_error is not None:
            return False, nested_error

    if outcome in ROOT_CAUSED_TERMINAL_OUTCOMES and not _has_text(payload.get("root_cause")):
        return False, {
            "ok": False,
            "reason": "root_cause_missing",
            "summary_path": str(summary_path),
            "outcome": outcome,
        }

    if require_factory_delta:
        factory_delta_error = _factory_delta_error(payload=payload, summary_path=summary_path)
        if factory_delta_error is not None:
            return False, factory_delta_error

    if require_commercial_evidence:
        commercial_evidence_error = _commercial_evidence_error(
            payload=payload,
            summary_path=summary_path,
        )
        if commercial_evidence_error is not None:
            return False, commercial_evidence_error

    return True, {
        "ok": True,
        "summary_path": str(summary_path),
        "artifact_count": len(artifacts),
        "artifacts": artifact_violations["resolved_artifacts"],
        "outcome": outcome,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate backend final evidence bundle contract.")
    parser.add_argument("--artifacts-dir", required=True, help="directory that contains final_evidence.json")
    parser.add_argument("--summary-name", default="final_evidence.json", help="summary filename inside artifacts dir")
    parser.add_argument("--allow-empty-artifacts", action="store_true", help="accept empty artifacts list")
    parser.add_argument("--allow-missing-files", action="store_true", help="skip existence checks for listed artifacts")
    parser.add_argument(
        "--require-factory-delta",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool_flag,
        help="require final_evidence.json to state an explicit factory capability delta",
    )
    parser.add_argument(
        "--require-commercial-evidence",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool_flag,
        help="require final_evidence.json to state explicit commercial or factory-value evidence",
    )
    args = parser.parse_args(argv)

    ok, payload = validate_evidence(
        artifacts_dir=Path(args.artifacts_dir).expanduser().resolve(),
        summary_name=str(args.summary_name or "final_evidence.json").strip() or "final_evidence.json",
        allow_empty_artifacts=bool(args.allow_empty_artifacts),
        allow_missing_files=bool(args.allow_missing_files),
        require_factory_delta=bool(args.require_factory_delta),
        require_commercial_evidence=bool(args.require_commercial_evidence),
    )
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
