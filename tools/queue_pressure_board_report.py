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
    return svc.queue_pressure_board(chat_id=chat_id, limit=limit)


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "-"


def _job_label(job: dict[str, Any]) -> str:
    return _one_line(job.get("job_id_short") or job.get("job_id"))


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    pressure_rows = [row for row in list(report.get("pressure_by_role") or []) if isinstance(row, dict)]
    actions = [row for row in list(report.get("recommended_actions") or []) if isinstance(row, dict)]
    top_jobs = [row for row in list(report.get("top_jobs") or report.get("backlog_samples") or []) if isinstance(row, dict)]

    lines = [
        "# Queue Pressure Board",
        "",
        f"- Generated: {_one_line(report.get('generated_at'))}",
        f"- Chat: {_one_line(report.get('chat_id'), default='all')}",
        f"- Limit: {_one_line(report.get('limit'))}",
        "",
        "## Summary",
        "",
        f"- Pressure level: {_one_line(summary.get('pressure_level'))}",
        f"- Next action: {_one_line(summary.get('next_action'))}",
        (
            f"- Capacity: max_parallel={_one_line(summary.get('max_parallel_total'), default='0')}, "
            f"running={_one_line(summary.get('running'), default='0')}, "
            f"idle={_one_line(summary.get('idle'), default='0')}, "
            f"saturation={_percent(summary.get('saturation'))}"
        ),
        (
            f"- Backlog: queued={_one_line(summary.get('queued'), default='0')}, "
            f"waiting_deps={_one_line(summary.get('waiting_deps'), default='0')}, "
            f"blocked_approval={_one_line(summary.get('blocked_approval'), default='0')}, "
            f"blocked={_one_line(summary.get('blocked'), default='0')}, "
            f"waiting_total={_one_line(summary.get('waiting_total'), default='0')}, "
            f"backlog_total={_one_line(summary.get('backlog_total'), default='0')}, "
            f"stalled={_one_line(summary.get('stalled'), default='0')}"
        ),
        f"- Roles: total={_one_line(summary.get('roles_total'), default='0')}, critical={_one_line(summary.get('roles_critical'), default='0')}, attention={_one_line(summary.get('roles_attention'), default='0')}",
        f"- Samples returned: {_one_line(summary.get('samples_returned'), default='0')}",
        "",
        "## Pressure By Role",
        "",
        "| Role | Level | Capacity | Running | Idle | Saturation | Queued | Waiting | Approval | Blocked | Stalled | Backlog | Next action |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    if pressure_rows:
        for row in pressure_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _one_line(row.get("role")),
                        _one_line(row.get("pressure_level")),
                        _one_line(row.get("max_parallel"), default="0"),
                        _one_line(row.get("running"), default="0"),
                        _one_line(row.get("idle"), default="0"),
                        _percent(row.get("saturation")),
                        _one_line(row.get("queued"), default="0"),
                        _one_line(row.get("waiting_deps"), default="0"),
                        _one_line(row.get("blocked_approval"), default="0"),
                        _one_line(row.get("blocked"), default="0"),
                        _one_line(row.get("stalled"), default="0"),
                        _one_line(row.get("backlog_total"), default="0"),
                        _one_line(row.get("next_action")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | - | 0 | 0 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | No role pressure rows returned. |")

    lines.extend(
        [
            "",
            "## Recommended Actions",
            "",
            "| Rank | Action | Role | Count | Target | Next action |",
            "| ---: | --- | --- | ---: | --- | --- |",
        ]
    )
    if actions:
        for idx, action in enumerate(actions, start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(idx),
                        _one_line(action.get("label") or action.get("action_id")),
                        _one_line(action.get("role")),
                        _one_line(action.get("count"), default="0"),
                        _one_line(action.get("target")),
                        _one_line(action.get("next_action")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | - | - | 0 | - | No recommended actions returned. |")

    lines.extend(
        [
            "",
            "## Backlog / Top Job Samples",
            "",
            "| Kind | Job | Role | State | Priority | Chat | Updated | Next action |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    if top_jobs:
        for job in top_jobs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _one_line(job.get("sample_kind")),
                        _job_label(job),
                        _one_line(job.get("role")),
                        _one_line(job.get("state")),
                        _one_line(job.get("priority"), default="0"),
                        _one_line(job.get("chat_id")),
                        _one_line(job.get("updated_at")),
                        _one_line(job.get("next_action")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | - | - | - | 0 | - | - | No backlog samples returned. |")

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline queue-pressure board report from the orchestrator SQLite database.")
    parser.add_argument("--db", required=True, type=Path, help="Path to the orchestrator SQLite database.")
    parser.add_argument("--format", choices=("json", "md", "markdown"), default="json", help="Output format.")
    parser.add_argument("--limit", type=_positive_int, default=50, help="Maximum backlog job samples to return.")
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
        print(f"failed to render queue-pressure board report: {exc}", file=err)
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
