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


def _one_line_list(value: Any, *, default: str = "-") -> str:
    if isinstance(value, list):
        parts = [_one_line(item, default="") for item in value]
        text = ", ".join(part for part in parts if part)
        return text or default
    return _one_line(value, default=default)


def _parse_filter_values(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        for part in str(raw or "").split(","):
            value = part.strip().lower()
            if not value or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
    return normalized


def _format_filter_values(values: Any, *, default: str = "all") -> str:
    if isinstance(values, list):
        return _one_line_list(values, default=default)
    return _one_line(values, default=default)


def build_report(
    *,
    db_path: Path,
    chat_id: int | None,
    limit: int,
    categories: list[str] | None = None,
    urgencies: list[str] | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    return svc.operator_focus(chat_id=chat_id, limit=limit, categories=categories, urgencies=urgencies, sources=sources)


def build_handoff(
    *,
    db_path: Path,
    chat_id: int | None,
    action_id: str | None = None,
    rank: int | None = None,
    categories: list[str] | None = None,
    urgencies: list[str] | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    return svc.operator_focus_handoff(
        chat_id=chat_id,
        action_id=action_id,
        rank=rank,
        categories=categories,
        urgencies=urgencies,
        sources=sources,
    )


def build_briefing(
    *,
    db_path: Path,
    chat_id: int | None,
    action_id: str | None = None,
    rank: int | None = None,
    categories: list[str] | None = None,
    urgencies: list[str] | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    return svc.operator_focus_briefing(
        chat_id=chat_id,
        action_id=action_id,
        rank=rank,
        categories=categories,
        urgencies=urgencies,
        sources=sources,
    )


def build_briefing_bundle(
    *,
    db_path: Path,
    chat_id: int | None,
    limit: int,
    categories: list[str] | None = None,
    urgencies: list[str] | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    return svc.operator_focus_briefing_bundle(
        chat_id=chat_id,
        limit=limit,
        categories=categories,
        urgencies=urgencies,
        sources=sources,
    )


def build_receipt(
    *,
    db_path: Path,
    chat_id: int | None,
    state: str,
    summary: str | None = None,
    next_action: str | None = None,
    actor: str | None = None,
    details: dict[str, Any] | None = None,
    action_id: str | None = None,
    rank: int | None = None,
    categories: list[str] | None = None,
    urgencies: list[str] | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    return svc.operator_focus_receipt(
        chat_id=chat_id,
        action_id=action_id,
        rank=rank,
        categories=categories,
        urgencies=urgencies,
        sources=sources,
        state=state,
        summary=summary,
        next_action=next_action,
        actor=actor,
        details=details,
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    source_counts = summary.get("source_counts") if isinstance(summary.get("source_counts"), dict) else {}
    available_source_counts = summary.get("available_source_counts") if isinstance(summary.get("available_source_counts"), dict) else {}
    filters = summary.get("filters") if isinstance(summary.get("filters"), dict) else {}
    available = summary.get("available") if isinstance(summary.get("available"), dict) else {}
    items = [item for item in list(report.get("items") or []) if isinstance(item, dict)]

    lines = [
        "# Operator Focus",
        "",
        f"- Generated: {_one_line(report.get('generated_at'))}",
        f"- Chat: {_one_line(report.get('chat_id'), default='all')}",
        f"- Limit: {_one_line(report.get('limit'))}",
        "",
        "## Summary",
        "",
        f"- Returned: {_one_line(summary.get('returned'), default='0')}",
        f"- Health: {_one_line(summary.get('health_level'))}",
        f"- Top action: {_one_line(summary.get('top_action_id'))}",
        f"- Top category: {_one_line(summary.get('top_category'))}",
        "",
        "## Filters",
        "",
        f"- Active categories: {_format_filter_values(filters.get('categories'))}",
        f"- Active urgencies: {_format_filter_values(filters.get('urgencies'))}",
        f"- Active sources: {_format_filter_values(filters.get('sources'))}",
        f"- Available categories: {_format_filter_values(available.get('categories'))}",
        f"- Available urgencies: {_format_filter_values(available.get('urgencies'))}",
        f"- Available sources: {_format_filter_values(available.get('sources'))}",
        f"- Available total: {_one_line(available.get('total'), default='0')}",
        f"- Filtered out: {_one_line(summary.get('filtered_out'), default='0')}",
        "",
        "## Counts",
        "",
        f"- Control room: {_one_line(source_counts.get('control_room'), default='0')}",
        f"- Proactive priorities: {_one_line(source_counts.get('proactive_priorities'), default='0')}",
        f"- Proactive health: {_one_line(source_counts.get('proactive_health'), default='0')}",
        f"- Available control room: {_one_line(available_source_counts.get('control_room'), default='0')}",
        f"- Available proactive priorities: {_one_line(available_source_counts.get('proactive_priorities'), default='0')}",
        f"- Available proactive health: {_one_line(available_source_counts.get('proactive_health'), default='0')}",
        "",
        "## Focus Items",
        "",
        "| Rank | Urgency | Category | Label | Target | Next action | Receipt status |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]

    if items:
        for item in items:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _one_line(item.get("rank")),
                        _one_line(item.get("urgency")),
                        _one_line(item.get("category")),
                        _one_line(item.get("label")),
                        _one_line(item.get("target")),
                        _one_line(item.get("next_action")),
                        _one_line(item.get("receipt_state"), default="new"),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | - | - | No operator focus items. | - | - | - |")

    lines.extend(["", "## Triage Details", ""])
    if items:
        for item in items:
            latest_receipt = item.get("latest_receipt") if isinstance(item.get("latest_receipt"), dict) else {}
            lines.extend(
                [
                    f"### {_one_line(item.get('rank'))}. {_one_line(item.get('label'))}",
                    "",
                    f"- Reason: {_one_line(item.get('reason'))}",
                    f"- Inspect path: {_one_line(item.get('inspect_path'))}",
                    f"- Inspect target: {_one_line(item.get('inspect_target'))}",
                    f"- Action target: {_one_line(item.get('action_target'))}",
                    f"- Source: {_one_line(item.get('source'))}",
                    f"- Source signals: {_one_line_list(item.get('source_signals'))}",
                    f"- Receipt state: {_one_line(item.get('receipt_state'), default='new')}",
                    f"- Latest receipt summary: {_one_line(latest_receipt.get('summary'))}",
                    f"- Latest receipt actor: {_one_line(latest_receipt.get('actor'))}",
                    f"- Latest receipt next_action: {_one_line(latest_receipt.get('next_action'))}",
                    f"- Latest receipt recorded_at: {_one_line(latest_receipt.get('recorded_at'))}",
                    "",
                ]
            )
    else:
        lines.append("- No triage details.")

    return "\n".join(lines) + "\n"


def _append_list(lines: list[str], title: str, values: Any) -> None:
    lines.extend([f"## {title}", ""])
    if isinstance(values, list) and values:
        for value in values:
            lines.append(f"- {_one_line(value)}")
    else:
        text = _one_line(values)
        if text != "-":
            lines.append(f"- {text}")
        else:
            lines.append("- None.")
    lines.append("")


def render_handoff_markdown(handoff: dict[str, Any]) -> str:
    selection = handoff.get("selection") if isinstance(handoff.get("selection"), dict) else {}
    summary = handoff.get("summary") if isinstance(handoff.get("summary"), dict) else {}
    filters = summary.get("filters") if isinstance(summary.get("filters"), dict) else {}
    available = summary.get("available") if isinstance(summary.get("available"), dict) else {}
    item = handoff.get("item") if isinstance(handoff.get("item"), dict) else None

    lines = [
        "# Operator Focus Handoff",
        "",
        f"- Generated: {_one_line(handoff.get('generated_at'))}",
        f"- Chat: {_one_line(handoff.get('chat_id'), default='all')}",
        "",
        "## Selection",
        "",
        f"- Action id: {_one_line(selection.get('action_id'))}",
        f"- Rank: {_one_line(selection.get('rank'))}",
        f"- Matched by: {_one_line(selection.get('matched_by'))}",
        "",
        "## Summary",
        "",
        f"- Returned: {_one_line(summary.get('returned'), default='0')}",
        f"- Health: {_one_line(summary.get('health_level'))}",
        f"- Top action: {_one_line(summary.get('top_action_id'))}",
        f"- Top category: {_one_line(summary.get('top_category'))}",
        f"- Filtered out: {_one_line(summary.get('filtered_out'), default='0')}",
        f"- Active categories: {_format_filter_values(filters.get('categories'))}",
        f"- Active urgencies: {_format_filter_values(filters.get('urgencies'))}",
        f"- Active sources: {_format_filter_values(filters.get('sources'))}",
        f"- Available categories: {_format_filter_values(available.get('categories'))}",
        f"- Available urgencies: {_format_filter_values(available.get('urgencies'))}",
        f"- Available sources: {_format_filter_values(available.get('sources'))}",
        f"- Available total: {_one_line(available.get('total'), default='0')}",
        "",
    ]

    if not item:
        lines.extend(
            [
                "## Selected Item",
                "",
                "No handoff item matched the requested selector.",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    delegate = item.get("delegate_contract") if isinstance(item.get("delegate_contract"), dict) else {}
    lines.extend(
        [
            "## Selected Item",
            "",
            f"- Rank: {_one_line(item.get('rank'))}",
            f"- Action id: {_one_line(item.get('action_id'))}",
            f"- Urgency: {_one_line(item.get('urgency'))}",
            f"- Category: {_one_line(item.get('category'))}",
            f"- Label: {_one_line(item.get('label'))}",
            f"- Next action: {_one_line(item.get('next_action'))}",
            f"- Target: {_one_line(item.get('target'))}",
            f"- Inspect path: {_one_line(item.get('inspect_path'))}",
            f"- Inspect target: {_one_line(item.get('inspect_target'))}",
            f"- Action target: {_one_line(item.get('action_target'))}",
            f"- Order id: {_one_line(item.get('order_id'))}",
            f"- Job id: {_one_line(item.get('job_id'))}",
            f"- Repo id: {_one_line(item.get('repo_id'))}",
            f"- Source: {_one_line(item.get('source'))}",
            f"- Source signals: {_one_line_list(item.get('source_signals'))}",
            f"- Reason: {_one_line(item.get('reason'))}",
            "",
            "## Delegate Contract",
            "",
            f"- Delegate role: {_one_line(delegate.get('delegate_role'))}",
            f"- Task title: {_one_line(delegate.get('task_title'))}",
            f"- Source action id: {_one_line(delegate.get('source_action_id'))}",
            f"- Handoff endpoint: {_one_line(delegate.get('handoff_endpoint'))}",
            f"- Inspect endpoint: {_one_line(delegate.get('inspect_endpoint'))}",
            "",
        ]
    )

    _append_list(lines, "Acceptance Criteria", delegate.get("acceptance_criteria"))
    _append_list(lines, "Definition Of Done", delegate.get("definition_of_done"))
    _append_list(lines, "Evidence Required", delegate.get("evidence_required"))
    _append_list(lines, "Suggested Tests", delegate.get("suggested_tests"))
    _append_list(lines, "Risk Notes", delegate.get("risk_notes"))

    lines.extend(
        [
            "## Task Prompt",
            "",
            "```text",
            _ascii(delegate.get("task_prompt")).rstrip(),
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_briefing_markdown(briefing: dict[str, Any]) -> str:
    selection = briefing.get("selection") if isinstance(briefing.get("selection"), dict) else {}
    summary = briefing.get("summary") if isinstance(briefing.get("summary"), dict) else {}
    filters = summary.get("filters") if isinstance(summary.get("filters"), dict) else {}
    available = summary.get("available") if isinstance(summary.get("available"), dict) else {}
    item_identity = briefing.get("item_identity") if isinstance(briefing.get("item_identity"), dict) else None
    packet = briefing.get("briefing_packet") if isinstance(briefing.get("briefing_packet"), dict) else None

    lines = [
        "# Operator Focus Briefing",
        "",
        f"- Generated: {_one_line(briefing.get('generated_at'))}",
        f"- Chat: {_one_line(briefing.get('chat_id'), default='all')}",
        "",
        "## Selection",
        "",
        f"- Action id: {_one_line(selection.get('action_id'))}",
        f"- Rank: {_one_line(selection.get('rank'))}",
        f"- Matched by: {_one_line(selection.get('matched_by'))}",
        "",
        "## Summary",
        "",
        f"- Returned: {_one_line(summary.get('returned'), default='0')}",
        f"- Health: {_one_line(summary.get('health_level'))}",
        f"- Top action: {_one_line(summary.get('top_action_id'))}",
        f"- Top category: {_one_line(summary.get('top_category'))}",
        f"- Filtered out: {_one_line(summary.get('filtered_out'), default='0')}",
        f"- Active categories: {_format_filter_values(filters.get('categories'))}",
        f"- Active urgencies: {_format_filter_values(filters.get('urgencies'))}",
        f"- Active sources: {_format_filter_values(filters.get('sources'))}",
        f"- Available categories: {_format_filter_values(available.get('categories'))}",
        f"- Available urgencies: {_format_filter_values(available.get('urgencies'))}",
        f"- Available sources: {_format_filter_values(available.get('sources'))}",
        f"- Available total: {_one_line(available.get('total'), default='0')}",
        "",
        "## Selected Item Identity",
        "",
    ]

    if item_identity:
        for key in ("rank", "action_id", "urgency", "category", "label", "source", "order_id", "job_id", "repo_id"):
            if key in item_identity:
                lines.append(f"- {key}: {_one_line(item_identity.get(key))}")
        lines.append("")
    else:
        lines.extend(["No selected item identity matched the requested selector.", ""])

    if not packet:
        lines.extend(
            [
                "## Briefing Packet",
                "",
                "No operator focus briefing packet matched the requested selector.",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Briefing Packet",
            "",
            f"- owner_role: {_one_line(packet.get('owner_role'))}",
            f"- action: {_one_line(packet.get('action'))}",
            f"- inspect_endpoint: {_one_line(packet.get('inspect_endpoint'))}",
            f"- handoff_endpoint: {_one_line(packet.get('handoff_endpoint'))}",
            "",
        ]
    )

    _append_list(lines, "Evidence Required", packet.get("evidence_required"))
    _append_list(lines, "Suggested Validation", packet.get("suggested_validation"))
    _append_list(lines, "Definition Of Done", packet.get("definition_of_done"))

    lines.extend(
        [
            "## Assignment Prompt",
            "",
            "```text",
            _ascii(packet.get("assignment_prompt")).rstrip(),
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_receipt_markdown(receipt_report: dict[str, Any]) -> str:
    selection = receipt_report.get("selection") if isinstance(receipt_report.get("selection"), dict) else {}
    summary = receipt_report.get("summary") if isinstance(receipt_report.get("summary"), dict) else {}
    filters = summary.get("filters") if isinstance(summary.get("filters"), dict) else {}
    available = summary.get("available") if isinstance(summary.get("available"), dict) else {}
    item_identity = receipt_report.get("item_identity") if isinstance(receipt_report.get("item_identity"), dict) else None
    receipt = receipt_report.get("receipt") if isinstance(receipt_report.get("receipt"), dict) else {}
    details = receipt.get("details") if isinstance(receipt.get("details"), dict) else {}

    lines = [
        "# Operator Focus Receipt",
        "",
        f"- Generated: {_one_line(receipt_report.get('generated_at'))}",
        f"- Chat: {_one_line(receipt_report.get('chat_id'), default='all')}",
        "",
        "## Selection",
        "",
        f"- Action id: {_one_line(selection.get('action_id'))}",
        f"- Rank: {_one_line(selection.get('rank'))}",
        f"- Matched by: {_one_line(selection.get('matched_by'))}",
        "",
        "## Summary",
        "",
        f"- Returned: {_one_line(summary.get('returned'), default='0')}",
        f"- Health: {_one_line(summary.get('health_level'))}",
        f"- Top action: {_one_line(summary.get('top_action_id'))}",
        f"- Top category: {_one_line(summary.get('top_category'))}",
        f"- Filtered out: {_one_line(summary.get('filtered_out'), default='0')}",
        f"- Active categories: {_format_filter_values(filters.get('categories'))}",
        f"- Active urgencies: {_format_filter_values(filters.get('urgencies'))}",
        f"- Active sources: {_format_filter_values(filters.get('sources'))}",
        f"- Available categories: {_format_filter_values(available.get('categories'))}",
        f"- Available urgencies: {_format_filter_values(available.get('urgencies'))}",
        f"- Available sources: {_format_filter_values(available.get('sources'))}",
        f"- Available total: {_one_line(available.get('total'), default='0')}",
        "",
        "## Selected Item Identity",
        "",
    ]

    if item_identity:
        for key in ("rank", "action_id", "urgency", "category", "label", "source", "order_id", "job_id", "repo_id"):
            if key in item_identity:
                lines.append(f"- {key}: {_one_line(item_identity.get(key))}")
        lines.append("")
    else:
        lines.extend(["No operator focus receipt item matched the requested selector.", ""])

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
            f"- recorded_at: {_one_line(receipt.get('recorded_at'))}",
            f"- order_id: {_one_line(receipt.get('order_id'))}",
            f"- job_id: {_one_line(receipt.get('job_id'))}",
            "",
            "### Details",
            "",
            "```json",
            json.dumps(details, sort_keys=True, ensure_ascii=True, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_briefing_bundle_markdown(bundle: dict[str, Any]) -> str:
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    filters = summary.get("filters") if isinstance(summary.get("filters"), dict) else {}
    available = summary.get("available") if isinstance(summary.get("available"), dict) else {}
    briefings = [briefing for briefing in list(bundle.get("briefings") or []) if isinstance(briefing, dict)]

    lines = [
        "# Operator Focus Briefing Bundle",
        "",
        f"- Generated: {_one_line(bundle.get('generated_at'))}",
        f"- Chat: {_one_line(bundle.get('chat_id'), default='all')}",
        f"- Limit: {_one_line(bundle.get('limit'))}",
        "",
        "## Summary",
        "",
        f"- Returned: {_one_line(summary.get('returned'), default=str(len(briefings)))}",
        f"- Health: {_one_line(summary.get('health_level'))}",
        f"- Top action: {_one_line(summary.get('top_action_id'))}",
        f"- Top category: {_one_line(summary.get('top_category'))}",
        f"- Filtered out: {_one_line(summary.get('filtered_out'), default='0')}",
        f"- Active categories: {_format_filter_values(filters.get('categories'))}",
        f"- Active urgencies: {_format_filter_values(filters.get('urgencies'))}",
        f"- Active sources: {_format_filter_values(filters.get('sources'))}",
        f"- Available categories: {_format_filter_values(available.get('categories'))}",
        f"- Available urgencies: {_format_filter_values(available.get('urgencies'))}",
        f"- Available sources: {_format_filter_values(available.get('sources'))}",
        f"- Available total: {_one_line(available.get('total'), default='0')}",
        "",
        "## Briefings",
        "",
    ]

    if not briefings:
        lines.extend(["No operator focus briefing packets matched the requested filters.", ""])
        return "\n".join(lines) + "\n"

    for index, briefing in enumerate(briefings, start=1):
        selection = briefing.get("selection") if isinstance(briefing.get("selection"), dict) else {}
        item_identity = briefing.get("item_identity") if isinstance(briefing.get("item_identity"), dict) else {}
        packet = briefing.get("briefing_packet") if isinstance(briefing.get("briefing_packet"), dict) else {}
        title = _one_line(item_identity.get("label"), default=f"Briefing {index}")
        lines.extend(
            [
                f"### {index}. {title}",
                "",
                f"- Action id: {_one_line(selection.get('action_id') or item_identity.get('action_id'))}",
                f"- Rank: {_one_line(selection.get('rank') or item_identity.get('rank'))}",
                f"- Matched by: {_one_line(selection.get('matched_by'))}",
                f"- Urgency: {_one_line(item_identity.get('urgency'))}",
                f"- Category: {_one_line(item_identity.get('category'))}",
                f"- Source: {_one_line(item_identity.get('source'))}",
                f"- Order id: {_one_line(item_identity.get('order_id'))}",
                f"- Job id: {_one_line(item_identity.get('job_id'))}",
                f"- Repo id: {_one_line(item_identity.get('repo_id'))}",
                "",
                "#### Briefing Packet",
                "",
                f"- owner_role: {_one_line(packet.get('owner_role'))}",
                f"- action: {_one_line(packet.get('action'))}",
                f"- inspect_endpoint: {_one_line(packet.get('inspect_endpoint'))}",
                f"- handoff_endpoint: {_one_line(packet.get('handoff_endpoint'))}",
                "",
            ]
        )
        _append_list(lines, "Evidence Required", packet.get("evidence_required"))
        _append_list(lines, "Suggested Validation", packet.get("suggested_validation"))
        _append_list(lines, "Definition Of Done", packet.get("definition_of_done"))
        lines.extend(
            [
                "## Assignment Prompt",
                "",
                "```text",
                _ascii(packet.get("assignment_prompt")).rstrip(),
                "```",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline operator-focus report from the orchestrator SQLite database.")
    parser.add_argument("--db", required=True, type=Path, help="Path to the orchestrator SQLite database.")
    parser.add_argument("--format", choices=("json", "md", "markdown"), default="json", help="Output format.")
    output_kind = parser.add_mutually_exclusive_group()
    output_kind.add_argument("--handoff", action="store_true", help="Render the selected operator-focus handoff payload.")
    output_kind.add_argument("--briefing", action="store_true", help="Render the selected operator-focus briefing payload.")
    output_kind.add_argument("--briefings", action="store_true", help="Render an operator-focus briefing bundle.")
    output_kind.add_argument("--receipt", action="store_true", help="Record and render an operator-focus receipt payload.")
    parser.add_argument("--limit", type=_positive_int, default=5, help="Maximum ranked focus items to return.")
    parser.add_argument("--rank", type=_positive_int, help="Select a handoff, briefing, or receipt focus item by rank.")
    parser.add_argument("--action-id", help="Select a handoff, briefing, or receipt focus item by action id. Takes precedence over --rank.")
    parser.add_argument("--chat-id", type=int, help="Optional chat id scope.")
    parser.add_argument("--category", action="append", help="Filter by focus category. Repeat or use comma-separated values.")
    parser.add_argument("--urgency", action="append", help="Filter by urgency. Repeat or use comma-separated values.")
    parser.add_argument("--source", action="append", help="Filter by source. Repeat or use comma-separated values.")
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

    categories = _parse_filter_values(args.category)
    urgencies = _parse_filter_values(args.urgency)
    sources = _parse_filter_values(args.source)
    details: dict[str, Any] | None = None

    if args.receipt:
        if not args.state:
            print("receipt mode requires --state (acknowledged, in_progress, or completed)", file=err)
            return 2
        if not args.action_id and args.rank is None:
            print("receipt mode requires either --action-id or --rank", file=err)
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
        if args.handoff:
            report = build_handoff(
                db_path=db_path,
                chat_id=args.chat_id,
                action_id=args.action_id,
                rank=args.rank,
                categories=categories,
                urgencies=urgencies,
                sources=sources,
            )
        elif args.briefing:
            report = build_briefing(
                db_path=db_path,
                chat_id=args.chat_id,
                action_id=args.action_id,
                rank=args.rank,
                categories=categories,
                urgencies=urgencies,
                sources=sources,
            )
        elif args.briefings:
            report = build_briefing_bundle(
                db_path=db_path,
                chat_id=args.chat_id,
                limit=args.limit,
                categories=categories,
                urgencies=urgencies,
                sources=sources,
            )
        elif args.receipt:
            report = build_receipt(
                db_path=db_path,
                chat_id=args.chat_id,
                action_id=args.action_id,
                rank=args.rank,
                categories=categories,
                urgencies=urgencies,
                sources=sources,
                state=args.state,
                summary=args.summary,
                next_action=args.next_action,
                actor=args.actor,
                details=details,
            )
        else:
            report = build_report(
                db_path=db_path,
                chat_id=args.chat_id,
                limit=args.limit,
                categories=categories,
                urgencies=urgencies,
                sources=sources,
            )
    except Exception as exc:
        print(f"failed to render operator focus report: {exc}", file=err)
        return 2

    if args.format == "json":
        rendered = json.dumps(report, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    elif args.handoff:
        rendered = render_handoff_markdown(report)
    elif args.briefing:
        rendered = render_briefing_markdown(report)
    elif args.briefings:
        rendered = render_briefing_bundle_markdown(report)
    elif args.receipt:
        rendered = render_receipt_markdown(report)
    else:
        rendered = render_markdown(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        out.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
