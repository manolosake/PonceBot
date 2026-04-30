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


def build_report(*, db_path: Path, chat_id: int | None, limit: int) -> dict[str, Any]:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    return svc.operator_focus(chat_id=chat_id, limit=limit)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    source_counts = summary.get("source_counts") if isinstance(summary.get("source_counts"), dict) else {}
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
        "## Counts",
        "",
        f"- Control room: {_one_line(source_counts.get('control_room'), default='0')}",
        f"- Proactive priorities: {_one_line(source_counts.get('proactive_priorities'), default='0')}",
        f"- Proactive health: {_one_line(source_counts.get('proactive_health'), default='0')}",
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

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline operator-focus report from the orchestrator SQLite database.")
    parser.add_argument("--db", required=True, type=Path, help="Path to the orchestrator SQLite database.")
    parser.add_argument("--format", choices=("json", "md", "markdown"), default="json", help="Output format.")
    parser.add_argument("--limit", type=_positive_int, default=5, help="Maximum ranked focus items to return.")
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
