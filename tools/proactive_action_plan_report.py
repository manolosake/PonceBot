#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.queue import OrchestratorQueue
from orchestrator.status_service import StatusService
from orchestrator.storage import SQLiteTaskStorage


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _ascii(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.encode("ascii", errors="replace").decode("ascii")


def _one_line(value: Any, *, default: str = "-") -> str:
    text = _ascii(value).replace("\r", " ").replace("\n", " ").strip()
    return text or default


def _markdown_table_cell(value: Any, *, default: str = "-") -> str:
    return _one_line(value, default=default).replace("|", "\\|")


def _compact_scalar_text(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return _one_line(value, default="")


def _publication_recovery_evidence(item: dict[str, Any]) -> str:
    evidence: list[str] = []
    github_url = _compact_scalar_text(item.get("github_url"))
    latest_head = _compact_scalar_text(item.get("latest_head"))
    source_order_id = _compact_scalar_text(item.get("source_order_id"))
    if github_url:
        evidence.append(f"url={github_url}")
    if latest_head:
        evidence.append(f"head={latest_head}")
    if source_order_id:
        evidence.append(f"order={source_order_id}")
    return "; ".join(evidence) or "-"


def _publication_recovery_status(item: dict[str, Any]) -> str:
    status = _one_line(item.get("status"), default="-")
    reason = _one_line(item.get("reason"), default="")
    if reason:
        return f"{status}: {reason}"
    return status


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def _source_line(source: Any) -> str:
    if not isinstance(source, dict):
        return _one_line(source)
    parts = []
    kind = _one_line(source.get("kind"), default="")
    key = _one_line(source.get("key"), default="")
    summary = _one_line(source.get("summary"), default="")
    if kind:
        parts.append(kind)
    if key:
        parts.append(key)
    prefix = "/".join(parts)
    if prefix and summary:
        return f"{prefix}: {summary}"
    return prefix or summary or "-"


def _append_list(lines: list[str], title: str, values: Any) -> None:
    lines.append(f"- {title}:")
    items = [_one_line(item) for item in _as_list(values)]
    if items:
        for item in items:
            lines.append(f"  - {item}")
    else:
        lines.append("  - -")


def _append_contract(lines: list[str], title: str, contract: Any) -> None:
    if not isinstance(contract, dict):
        return
    lines.append(f"- {title}:")
    allowed_outcomes = [_one_line(item) for item in _as_list(contract.get("allowed_outcomes"))]
    if allowed_outcomes:
        lines.append("  - allowed_outcomes:")
        for item in allowed_outcomes:
            lines.append(f"    - {item}")
    by_outcome = contract.get("required_fields_by_outcome")
    if isinstance(by_outcome, dict):
        lines.append("  - required_fields_by_outcome:")
        for outcome, raw_fields in by_outcome.items():
            fields = [_one_line(item) for item in _as_list(raw_fields)]
            if fields:
                lines.append(f"    - {_one_line(outcome)}: {', '.join(fields)}")
            else:
                lines.append(f"    - {_one_line(outcome)}: -")
    else:
        fields = [_one_line(item) for item in _as_list(contract.get("required_fields"))]
        if fields:
            lines.append("  - required_fields:")
            for field in fields:
                lines.append(f"    - {field}")
    definition = _one_line(contract.get("definition"), default="")
    if definition and definition != "-":
        lines.append(f"  - definition: {definition}")
    validation = [_one_line(item) for item in _as_list(contract.get("suggested_validation"))]
    if validation:
        lines.append("  - suggested_validation:")
        for item in validation:
            lines.append(f"    - {item}")


def _append_mapping(lines: list[str], title: str, mapping: Any) -> None:
    if not isinstance(mapping, dict):
        return
    items = [
        f"{_one_line(key, default='-')}={_one_line(value, default='-')}"
        for key, value in mapping.items()
        if _one_line(value, default="") not in {"", "-"}
    ]
    if not items:
        return
    lines.append(f"- {title}:")
    for item in items:
        lines.append(f"  - {item}")


def _append_current_target_identifiers(lines: list[str], packet: Any) -> None:
    if not isinstance(packet, dict):
        return
    current_target_facts = packet.get("current_target_facts") if isinstance(packet.get("current_target_facts"), dict) else {}
    for key in ("recovery_id", "project_key", "project_name", "project_path", "target_label", "required_action"):
        value = packet.get(key)
        if value is None:
            value = current_target_facts.get(key)
        text = _one_line(value, default="")
        if text and text != "-":
            lines.append(f"- {key}: {text}")


def _find_top_order(report: dict[str, Any], order_id: Any) -> dict[str, Any]:
    oid = str(order_id or "").strip()
    if not oid:
        return {}
    for lane in list(report.get("lanes") or []):
        if not isinstance(lane, dict):
            continue
        for order in list(lane.get("orders") or []):
            if isinstance(order, dict) and str(order.get("order_id") or "").strip() == oid:
                return order
    return {}


def build_report(*, db_path: Path, chat_id: int | None, limit: int) -> dict[str, Any]:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    return svc.proactive_action_plan(chat_id=chat_id, limit=limit)


def build_receipt(
    *,
    db_path: Path,
    chat_id: int | None,
    state: str,
    summary: str | None = None,
    next_action: str | None = None,
    actor: str | None = None,
    details: dict[str, Any] | None = None,
    order_id: str | None = None,
    rank: int | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    return svc.proactive_action_plan_receipt(
        chat_id=chat_id,
        state=state,
        summary=summary,
        next_action=next_action,
        actor=actor,
        details=details,
        order_id=order_id,
        rank=rank,
        limit=limit,
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lane_counts = summary.get("lanes") if isinstance(summary.get("lanes"), dict) else {}
    selection_quality = summary.get("selection_quality") if isinstance(summary.get("selection_quality"), dict) else {}
    publication_recovery = (
        report.get("publication_recovery") if isinstance(report.get("publication_recovery"), dict) else None
    )
    lanes = [lane for lane in list(report.get("lanes") or []) if isinstance(lane, dict)]

    lines = [
        "# Proactive Action Plan report",
        "",
        f"- Generated: {_one_line(report.get('generated_at'))}",
        f"- Chat: {_one_line(report.get('chat_id'), default='all')}",
        f"- Limit: {_one_line(report.get('limit'))}",
        "",
        "## Summary",
        "",
        f"- Active proactive orders: {_one_line(summary.get('active_proactive_orders'), default='0')}",
        f"- Returned: {_one_line(summary.get('returned'), default='0')}",
        f"- Top lane: {_one_line(summary.get('top_lane'))}",
        f"- Top action: {_one_line(summary.get('top_action'))}",
        f"- Selection needs review: {_one_line(selection_quality.get('needs_review'), default='0')}",
        "",
        "## Lane Counts",
        "",
    ]

    if lanes:
        for lane in lanes:
            lane_key = str(lane.get("lane") or "").strip()
            label = _one_line(lane.get("label") or lane_key)
            count = lane_counts.get(lane_key, lane.get("count"))
            lines.append(f"- {label}: {_one_line(count, default='0')}")
    else:
        lines.append("- No lanes returned.")

    if publication_recovery is not None:
        lines.extend(
            [
                "",
                "## Publication Recovery",
                "",
                f"- Open items: {_one_line(publication_recovery.get('count'), default='0')}",
                f"- Truncated: {_one_line(publication_recovery.get('truncated'), default='False')}",
            ]
        )
        items = [item for item in list(publication_recovery.get("items") or []) if isinstance(item, dict)]
        if items:
            lines.extend(
                [
                    "",
                    "| Project | Target | Evidence | Required action | Missing | Status |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
            )
            for item in items:
                target = item.get("github_repo") or item.get("project_path")
                missing = ", ".join(_one_line(entry, default="") for entry in _as_list(item.get("missing_fields")))
                if not missing.strip():
                    missing = _one_line(item.get("missing_json"))
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _markdown_table_cell(item.get("project_name")),
                            _markdown_table_cell(target),
                            _markdown_table_cell(_publication_recovery_evidence(item)),
                            _markdown_table_cell(item.get("required_action")),
                            _markdown_table_cell(missing),
                            _markdown_table_cell(_publication_recovery_status(item)),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("- No open publication recovery items.")

    top_execution_packet = report.get("top_execution_packet") if isinstance(report.get("top_execution_packet"), dict) else None
    if top_execution_packet is not None:
        lines.extend(["", "## Top Execution Packet", ""])
        for key in ("owner_role", "lane", "order_id", "action", "inspect_endpoint", "handoff_endpoint"):
            lines.append(f"- {key}: {_one_line(top_execution_packet.get(key))}")
        _append_current_target_identifiers(lines, top_execution_packet)
        _append_mapping(lines, "current_target_facts", top_execution_packet.get("current_target_facts"))
        _append_list(lines, "missing_fields", top_execution_packet.get("missing_fields"))
        _append_list(lines, "acceptance_criteria", top_execution_packet.get("acceptance_criteria"))
        _append_list(lines, "evidence_required", top_execution_packet.get("evidence_required"))
        _append_list(lines, "suggested_validation", top_execution_packet.get("suggested_validation"))
        _append_list(lines, "definition_of_done", top_execution_packet.get("definition_of_done"))
        _append_contract(
            lines,
            "outcome_contract",
            top_execution_packet.get("outcome_contract"),
        )
        _append_contract(
            lines,
            "studio_decision_evidence_contract",
            top_execution_packet.get("studio_decision_evidence_contract"),
        )

        top_order = _find_top_order(report, top_execution_packet.get("order_id"))
        selection = top_order.get("selection_quality") if isinstance(top_order.get("selection_quality"), dict) else {}
        if selection:
            lines.extend(["", "### Top Order Selection Quality", ""])
            lines.append(f"- status: {_one_line(selection.get('status'))}")
            _append_list(lines, "flags", selection.get("flags"))
            lines.append(f"- summary: {_one_line(selection.get('summary'))}")
            if "evidence_sources" in selection:
                lines.append("- evidence_sources:")
                sources = [_source_line(source) for source in _as_list(selection.get("evidence_sources"))]
                if sources:
                    for source in sources:
                        lines.append(f"  - {source}")
                else:
                    lines.append("  - -")
            churn_risk = selection.get("churn_risk") if isinstance(selection.get("churn_risk"), dict) else {}
            if churn_risk:
                lines.append(f"- churn_risk: {_one_line(churn_risk.get('status'))}")
                _append_list(lines, "churn_flags", churn_risk.get("flags"))
                counters = churn_risk.get("counters") if isinstance(churn_risk.get("counters"), dict) else {}
                if counters:
                    counter_text = ", ".join(f"{_one_line(key)}={_one_line(value)}" for key, value in sorted(counters.items()))
                    lines.append(f"- churn_counters: {counter_text}")

    next_delegate = summary.get("next_delegate") if isinstance(summary.get("next_delegate"), dict) else None
    if next_delegate is not None:
        lines.extend(["", "## Next Delegate", ""])
        for key in ("owner_role", "lane", "order_id", "action", "inspect_endpoint", "handoff_endpoint"):
            lines.append(f"- {key}: {_one_line(next_delegate.get(key))}")
        _append_current_target_identifiers(lines, next_delegate)
        _append_mapping(lines, "current_target_facts", next_delegate.get("current_target_facts"))
        _append_list(lines, "missing_fields", next_delegate.get("missing_fields"))
        _append_list(lines, "acceptance_criteria", next_delegate.get("acceptance_criteria"))
        _append_list(lines, "evidence_required", next_delegate.get("evidence_required"))
        _append_list(lines, "suggested_validation", next_delegate.get("suggested_validation"))
        _append_list(lines, "definition_of_done", next_delegate.get("definition_of_done"))
        lines.append(f"- assignment_prompt: {_one_line(next_delegate.get('assignment_prompt'))}")
        _append_contract(
            lines,
            "outcome_contract",
            next_delegate.get("outcome_contract"),
        )
        _append_contract(
            lines,
            "factory_delta_contract",
            next_delegate.get("factory_delta_contract"),
        )
        _append_contract(
            lines,
            "studio_decision_evidence_contract",
            next_delegate.get("studio_decision_evidence_contract"),
        )

    for lane in lanes:
        label = _one_line(lane.get("label") or lane.get("lane"))
        orders = [order for order in list(lane.get("orders") or []) if isinstance(order, dict)]
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"- Recommended next action: {_one_line(lane.get('recommended_next_action'))}",
                "",
                "| Rank | Order | Stage | Verdict | Selection | Next action |",
                "| ---: | --- | --- | --- | --- | --- |",
            ]
        )
        if orders:
            for order in orders:
                order_label = order.get("order_id_short") or order.get("order_id")
                order_selection = order.get("selection_quality") if isinstance(order.get("selection_quality"), dict) else {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _one_line(order.get("rank")),
                            _one_line(order_label),
                            _one_line(order.get("current_stage")),
                            _one_line(order.get("readiness_verdict")),
                            _one_line(order_selection.get("status")),
                            _one_line(order.get("next_action")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("| - | - | - | - | - | No orders in this lane. |")

    return "\n".join(lines) + "\n"


def render_receipt_markdown(receipt_report: dict[str, Any]) -> str:
    selection = receipt_report.get("selection") if isinstance(receipt_report.get("selection"), dict) else {}
    summary = receipt_report.get("summary") if isinstance(receipt_report.get("summary"), dict) else {}
    order_identity = receipt_report.get("order_identity") if isinstance(receipt_report.get("order_identity"), dict) else None
    receipt = receipt_report.get("receipt") if isinstance(receipt_report.get("receipt"), dict) else {}
    persisted_receipt = (
        receipt_report.get("persisted_receipt")
        if isinstance(receipt_report.get("persisted_receipt"), dict)
        else None
    )
    details = receipt.get("details") if isinstance(receipt.get("details"), dict) else {}
    receipt_counts = (
        receipt_report.get("receipt_counts_by_state")
        if isinstance(receipt_report.get("receipt_counts_by_state"), dict)
        else {}
    )
    receipt_history = [
        item for item in list(receipt_report.get("receipt_history") or []) if isinstance(item, dict)
    ]

    lines = [
        "# Proactive Action Plan Receipt",
        "",
        f"- Generated: {_one_line(receipt_report.get('generated_at'))}",
        f"- Chat: {_one_line(receipt_report.get('chat_id'), default='all')}",
        "",
        "## Selection",
        "",
        f"- Order id: {_one_line(selection.get('order_id'))}",
        f"- Rank: {_one_line(selection.get('rank'))}",
        f"- Matched by: {_one_line(selection.get('matched_by'))}",
        "",
        "## Summary",
        "",
        f"- Active proactive orders: {_one_line(summary.get('active_proactive_orders'), default='0')}",
        f"- Returned: {_one_line(summary.get('returned'), default='0')}",
        f"- Top lane: {_one_line(summary.get('top_lane'))}",
        f"- Top action: {_one_line(summary.get('top_action'))}",
        "",
        "## Selected Order",
        "",
    ]

    if order_identity:
        for key in (
            "rank",
            "order_id",
            "order_id_short",
            "title",
            "priority",
            "phase",
            "current_stage",
            "readiness_state",
            "readiness_verdict",
            "decision",
            "next_action",
        ):
            if key in order_identity:
                lines.append(f"- {key}: {_one_line(order_identity.get(key))}")
        selection_quality = (
            order_identity.get("selection_quality")
            if isinstance(order_identity.get("selection_quality"), dict)
            else {}
        )
        if selection_quality:
            lines.extend(["", "### Selection Context", ""])
            for key in ("status", "recommended_owner_role", "delegation_reason", "delegation_focus", "summary"):
                if key in selection_quality:
                    lines.append(f"- {key}: {_one_line(selection_quality.get(key))}")
            if "flags" in selection_quality:
                _append_list(lines, "flags", selection_quality.get("flags"))
            if "evidence_sources" in selection_quality:
                lines.append("- evidence_sources:")
                sources = [_source_line(source) for source in _as_list(selection_quality.get("evidence_sources"))]
                if sources:
                    for source in sources:
                        lines.append(f"  - {source}")
                else:
                    lines.append("  - -")
        handoff = order_identity.get("handoff") if isinstance(order_identity.get("handoff"), dict) else {}
        if handoff:
            lines.extend(
                [
                    "",
                    "### Delegation",
                    "",
                    f"- suggested_role: {_one_line(handoff.get('suggested_role'))}",
                    f"- suggested_endpoint: {_one_line(handoff.get('suggested_endpoint'))}",
                    f"- inspect_path: {_one_line(handoff.get('inspect_path'))}",
                    f"- assignment_prompt: {_one_line(handoff.get('assignment_prompt'))}",
                ]
            )
            for key, title in (
                ("evidence_expectations", "evidence_expectations"),
                ("suggested_validation", "suggested_validation"),
                ("definition_of_done", "definition_of_done"),
            ):
                if key in handoff:
                    _append_list(lines, title, handoff.get(key))
        lines.append("")
    else:
        lines.extend(["No proactive order matched the requested selector.", ""])

    lines.extend(
        [
            "## Receipt",
            "",
            f"- event_type: {_one_line(receipt.get('event_type'))}",
            f"- state: {_one_line(receipt.get('state'))}",
            f"- summary: {_one_line(receipt.get('summary'))}",
            f"- next_action: {_one_line(receipt.get('next_action'))}",
            f"- actor: {_one_line(receipt.get('actor'))}",
            f"- persisted: {_one_line(receipt.get('persisted'))}",
            f"- persistence_reason: {_one_line(receipt.get('persistence_reason'))}",
            f"- order_id: {_one_line(receipt.get('order_id'))}",
            f"- job_id: {_one_line(receipt.get('job_id'))}",
            "",
            "### Details",
            "",
            "```json",
            json.dumps(details, sort_keys=True, ensure_ascii=True, indent=2),
            "```",
            "",
            "## Persisted Receipt",
            "",
        ]
    )

    if persisted_receipt:
        for key in ("state", "summary", "next_action", "actor", "recorded_at", "order_id", "job_id"):
            lines.append(f"- {key}: {_one_line(persisted_receipt.get(key))}")
        lines.append("")
    else:
        lines.extend(["No persisted receipt is available for this selection.", ""])

    lines.extend(
        [
            "## Receipt Counts",
            "",
            f"- Total receipts: {_one_line(receipt_report.get('receipt_count'), default='0')}",
        ]
    )
    if receipt_counts:
        for state in sorted(receipt_counts):
            lines.append(f"- {state}: {_one_line(receipt_counts.get(state), default='0')}")
    else:
        lines.append("- By state: none")
    lines.append("")

    lines.extend(
        [
            "## Receipt History",
            "",
            "| # | State | Actor | Recorded at | Summary | Next action |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    if receipt_history:
        for index, history_item in enumerate(receipt_history, start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        _one_line(index),
                        _one_line(history_item.get("state")),
                        _one_line(history_item.get("actor")),
                        _one_line(history_item.get("recorded_at")),
                        _one_line(history_item.get("summary")),
                        _one_line(history_item.get("next_action")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | - | - | - | No receipt history. | - |")
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline proactive action-plan report from the orchestrator SQLite database.")
    parser.add_argument("--db", required=True, type=Path, help="Path to the orchestrator SQLite database.")
    parser.add_argument("--format", choices=("json", "md", "markdown"), default="json", help="Output format.")
    parser.add_argument("--receipt", action="store_true", help="Record and render a proactive action-plan receipt payload.")
    parser.add_argument("--limit", type=_positive_int, default=20, help="Maximum proactive orders to return.")
    parser.add_argument("--rank", type=_positive_int, help="Select a proactive order by rank for receipt mode.")
    parser.add_argument("--order-id", help="Select a proactive order by order id for receipt mode. Takes precedence over --rank.")
    parser.add_argument("--chat-id", type=int, help="Optional chat id scope.")
    parser.add_argument("--state", choices=("acknowledged", "in_progress", "completed"), help="Receipt state.")
    parser.add_argument("--summary", help="Receipt summary.")
    parser.add_argument("--next-action", help="Receipt next action.")
    parser.add_argument("--actor", help="Receipt actor.")
    parser.add_argument("--details-json", help="Receipt details as a JSON object.")
    parser.add_argument("--output", type=Path, help="Optional output file path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = _parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"database not found: {db_path}", file=err)
        return 2

    details: dict[str, Any] | None = None
    if args.receipt:
        if not args.state:
            print("receipt mode requires --state (acknowledged, in_progress, or completed)", file=err)
            return 2
        if not args.order_id and args.rank is None:
            print("receipt mode requires either --order-id or --rank", file=err)
            return 2
        if args.details_json:
            try:
                parsed_details = json.loads(args.details_json)
            except json.JSONDecodeError as exc:
                print(f"--details-json must be a valid JSON object: {exc.msg}", file=err)
                return 2
            if not isinstance(parsed_details, dict):
                print("--details-json must be a JSON object", file=err)
                return 2
            details = parsed_details

    try:
        if args.receipt:
            report = build_receipt(
                db_path=db_path,
                chat_id=args.chat_id,
                state=args.state,
                summary=args.summary,
                next_action=args.next_action,
                actor=args.actor,
                details=details,
                order_id=args.order_id,
                rank=args.rank,
                limit=args.limit,
            )
        else:
            report = build_report(db_path=db_path, chat_id=args.chat_id, limit=args.limit)
    except Exception as exc:
        print(f"failed to render proactive action-plan report: {exc}", file=err)
        return 2

    if args.format == "json":
        rendered = json.dumps(report, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    else:
        rendered = render_receipt_markdown(report) if args.receipt else render_markdown(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        out.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
