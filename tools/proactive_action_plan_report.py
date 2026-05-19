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


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lane_counts = summary.get("lanes") if isinstance(summary.get("lanes"), dict) else {}
    selection_quality = summary.get("selection_quality") if isinstance(summary.get("selection_quality"), dict) else {}
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

    top_execution_packet = report.get("top_execution_packet") if isinstance(report.get("top_execution_packet"), dict) else None
    if top_execution_packet is not None:
        lines.extend(["", "## Top Execution Packet", ""])
        for key in ("owner_role", "lane", "order_id", "action", "inspect_endpoint", "handoff_endpoint"):
            lines.append(f"- {key}: {_one_line(top_execution_packet.get(key))}")
        _append_list(lines, "acceptance_criteria", top_execution_packet.get("acceptance_criteria"))
        _append_list(lines, "evidence_required", top_execution_packet.get("evidence_required"))
        _append_list(lines, "suggested_validation", top_execution_packet.get("suggested_validation"))
        _append_list(lines, "definition_of_done", top_execution_packet.get("definition_of_done"))

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

    next_delegate = summary.get("next_delegate") if isinstance(summary.get("next_delegate"), dict) else None
    if next_delegate is not None:
        lines.extend(["", "## Next Delegate", ""])
        for key in ("owner_role", "lane", "order_id", "action", "inspect_endpoint", "handoff_endpoint"):
            lines.append(f"- {key}: {_one_line(next_delegate.get(key))}")

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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline proactive action-plan report from the orchestrator SQLite database.")
    parser.add_argument("--db", required=True, type=Path, help="Path to the orchestrator SQLite database.")
    parser.add_argument("--format", choices=("json", "md", "markdown"), default="json", help="Output format.")
    parser.add_argument("--limit", type=_positive_int, default=20, help="Maximum proactive orders to return.")
    parser.add_argument("--chat-id", type=int, help="Optional chat id scope.")
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

    try:
        report = build_report(db_path=db_path, chat_id=args.chat_id, limit=args.limit)
    except Exception as exc:
        print(f"failed to render proactive action-plan report: {exc}", file=err)
        return 2

    if args.format == "json":
        rendered = json.dumps(report, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
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
