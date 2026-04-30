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
        "| Rank | Urgency | Category | Label | Target | Next action |",
        "| ---: | --- | --- | --- | --- | --- |",
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
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | - | - | No operator focus items. | - | - |")

    lines.extend(["", "## Triage Details", ""])
    if items:
        for item in items:
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
                    "",
                ]
            )
    else:
        lines.append("- No triage details.")

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline operator-focus report from the orchestrator SQLite database.")
    parser.add_argument("--db", required=True, type=Path, help="Path to the orchestrator SQLite database.")
    parser.add_argument("--format", choices=("json", "md", "markdown"), default="json", help="Output format.")
    parser.add_argument("--limit", type=_positive_int, default=5, help="Maximum ranked focus items to return.")
    parser.add_argument("--chat-id", type=int, help="Optional chat id scope.")
    parser.add_argument("--category", action="append", help="Filter by focus category. Repeat or use comma-separated values.")
    parser.add_argument("--urgency", action="append", help="Filter by urgency. Repeat or use comma-separated values.")
    parser.add_argument("--source", action="append", help="Filter by source. Repeat or use comma-separated values.")
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

    try:
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
