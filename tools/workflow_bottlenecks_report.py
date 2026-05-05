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
    text = text.replace("|", "\\|")
    return text or default


def build_report(*, db_path: Path, chat_id: int | None, limit: int) -> dict[str, Any]:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    return svc.workflow_bottlenecks(chat_id=chat_id, limit=limit)


def _order_label(order: dict[str, Any]) -> str:
    return _one_line(order.get("order_id_short") or order.get("order_id"))


def _stage_label(stage: dict[str, Any]) -> str:
    return _one_line(stage.get("label") or stage.get("stage"))


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    stages = [stage for stage in list(report.get("stages") or []) if isinstance(stage, dict)]

    lines = [
        "# Workflow Bottleneck Board",
        "",
        f"- Generated: {_one_line(report.get('generated_at'))}",
        f"- Chat: {_one_line(report.get('chat_id'), default='all')}",
        f"- Limit: {_one_line(report.get('limit'))}",
        "",
        "## Summary",
        "",
        f"- Orders total: {_one_line(summary.get('orders_total'), default='0')}",
        f"- Bottleneck stage: {_one_line(summary.get('bottleneck_stage'))}",
        f"- Bottleneck score: {_one_line(summary.get('bottleneck_score'), default='0')}",
        f"- Bottleneck overdue count: {_one_line(summary.get('bottleneck_overdue_count'), default='0')}",
        f"- Recommended next action: {_one_line(summary.get('recommended_next_action'))}",
        "",
        "## Stages",
        "",
        "| Stage | Orders | Blocked | Failed | Overdue | Running | Pending | Oldest updated | Recommended next action |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    if stages:
        for stage in stages:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _stage_label(stage),
                        _one_line(stage.get("orders_count"), default="0"),
                        _one_line(stage.get("blocked_count"), default="0"),
                        _one_line(stage.get("failed_count"), default="0"),
                        _one_line(stage.get("overdue_count"), default="0"),
                        _one_line(stage.get("running_count"), default="0"),
                        _one_line(stage.get("pending_count"), default="0"),
                        _one_line(stage.get("oldest_updated_at")),
                        _one_line(stage.get("recommended_next_action")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | 0 | 0 | 0 | 0 | 0 | 0 | - | No workflow stages returned. |")

    lines.extend(["", "## Order Samples"])

    if not stages:
        lines.extend(
            [
                "",
                "### No Stages",
                "",
                "| Order | Priority | Phase | Readiness | Overdue seconds | Blocker summary | Next action |",
                "| --- | ---: | --- | --- | ---: | --- | --- |",
                "| - | 0 | - | - | 0 | - | No workflow stages returned. |",
            ]
        )
        return "\n".join(lines) + "\n"

    for stage in stages:
        orders = [order for order in list(stage.get("orders") or []) if isinstance(order, dict)]
        lines.extend(
            [
                "",
                f"### {_stage_label(stage)}",
                "",
                "| Order | Priority | Phase | Readiness | Overdue seconds | Blocker summary | Next action |",
                "| --- | ---: | --- | --- | ---: | --- | --- |",
            ]
        )
        if orders:
            for order in orders:
                readiness = _one_line(
                    (
                        f"{_one_line(order.get('readiness_verdict'))}"
                        f" / {_one_line(order.get('readiness_state'))}"
                    )
                )
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _order_label(order),
                            _one_line(order.get("priority"), default="0"),
                            _one_line(order.get("phase")),
                            readiness,
                            _one_line(order.get("current_stage_overdue_by_seconds"), default="0"),
                            _one_line(order.get("blocker_summary")),
                            _one_line(order.get("next_action")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("| - | 0 | - | - | 0 | - | No orders in this stage. |")

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline workflow bottleneck board report from the orchestrator SQLite database.")
    parser.add_argument("--db", required=True, type=Path, help="Path to the orchestrator SQLite database.")
    parser.add_argument("--format", choices=("json", "md", "markdown"), default="json", help="Output format.")
    parser.add_argument("--limit", type=_positive_int, default=50, help="Maximum order samples per stage to return.")
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
        print(f"failed to render workflow bottleneck board report: {exc}", file=err)
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
