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


LANE_LABELS = {
    "ready": "Ready",
    "blocked": "Blocked",
    "not_ready": "Not ready",
    "released": "Released",
}


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


def build_report(
    *,
    db_path: Path,
    chat_id: int | None,
    limit: int,
    include_released: bool,
) -> dict[str, Any]:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    return svc.release_readiness_board(
        chat_id=chat_id,
        limit=limit,
        include_released=include_released,
    )


def _lane_payload(report: dict[str, Any], lane: str) -> dict[str, Any]:
    lanes = report.get("lanes") if isinstance(report.get("lanes"), dict) else {}
    payload = lanes.get(lane) if isinstance(lanes.get(lane), dict) else {}
    return payload


def _checks_summary(order: dict[str, Any]) -> str:
    checks = order.get("checks_by_status") if isinstance(order.get("checks_by_status"), dict) else {}
    if not checks:
        return "-"
    parts = [f"{_one_line(key)}={_one_line(checks.get(key), default='0')}" for key in sorted(checks)]
    return ", ".join(parts) if parts else "-"


def _blocker_summary(order: dict[str, Any]) -> str:
    blocker = order.get("primary_blocker") if isinstance(order.get("primary_blocker"), dict) else {}
    if not blocker:
        return "-"
    stage = _one_line(blocker.get("stage"))
    summary = _one_line(blocker.get("summary"))
    if stage == "-":
        return summary
    if summary == "-":
        return stage
    return f"{stage}: {summary}"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    by_lane = summary.get("by_lane") if isinstance(summary.get("by_lane"), dict) else {}

    lines = [
        "# Release Readiness Board",
        "",
        f"- Generated: {_one_line(report.get('generated_at'))}",
        f"- Chat: {_one_line(report.get('chat_id'), default='all')}",
        f"- Limit: {_one_line(report.get('limit'))}",
        f"- Include released: {_one_line(report.get('include_released'), default='False')}",
        "",
        "## Summary",
        "",
        f"- Orders returned: {_one_line(summary.get('returned', summary.get('orders_total')), default='0')}",
        f"- Ready: {_one_line(by_lane.get('ready', summary.get('ready')), default='0')}",
        f"- Blocked: {_one_line(by_lane.get('blocked', summary.get('blocked')), default='0')}",
        f"- Not ready: {_one_line(by_lane.get('not_ready', summary.get('not_ready')), default='0')}",
        f"- Released: {_one_line(by_lane.get('released', summary.get('released')), default='0')}",
    ]

    for lane in ("ready", "blocked", "not_ready", "released"):
        payload = _lane_payload(report, lane)
        orders = [order for order in list(payload.get("orders") or []) if isinstance(order, dict)]
        label = LANE_LABELS.get(lane, lane)
        count = payload.get("count", by_lane.get(lane, 0))
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"- Count: {_one_line(count, default='0')}",
                "",
                "| Rank | Order | Chat | Stage | Verdict | Checks | Blocker | Next action |",
                "| ---: | --- | ---: | --- | --- | --- | --- | --- |",
            ]
        )
        if orders:
            for order in orders:
                order_label = order.get("order_id_short") or order.get("order_id")
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _one_line(order.get("rank")),
                            _one_line(order_label),
                            _one_line(order.get("chat_id")),
                            _one_line(order.get("current_stage")),
                            _one_line(order.get("readiness_verdict")),
                            _checks_summary(order),
                            _blocker_summary(order),
                            _one_line(order.get("next_action")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("| - | - | - | - | - | - | - | No orders in this lane. |")

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline release-readiness board report from the orchestrator SQLite database.")
    parser.add_argument("--db", required=True, type=Path, help="Path to the orchestrator SQLite database.")
    parser.add_argument("--format", choices=("json", "md", "markdown"), default="json", help="Output format.")
    parser.add_argument("--limit", type=_positive_int, default=50, help="Maximum proactive orders to return.")
    parser.add_argument("--chat-id", type=int, help="Optional chat id scope.")
    parser.add_argument("--include-released", action="store_true", help="Include released orders in the board.")
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
        report = build_report(
            db_path=db_path,
            chat_id=args.chat_id,
            limit=args.limit,
            include_released=args.include_released,
        )
    except Exception as exc:
        print(f"failed to render release-readiness board report: {exc}", file=err)
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
