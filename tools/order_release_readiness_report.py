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


def _evidence_count(check: dict[str, Any]) -> int:
    return len([item for item in list(check.get("evidence") or []) if isinstance(item, dict)])


def _format_ref(ref: dict[str, Any]) -> str:
    bits: list[str] = []
    for key in (
        "kind",
        "key",
        "event_type",
        "decision_kind",
        "role",
        "job_id_short",
        "state",
        "path",
        "artifact_id",
        "summary",
    ):
        value = ref.get(key)
        if value is None or value == "":
            continue
        bits.append(f"{key}={_one_line(value)}")
    return "; ".join(bits) if bits else _one_line(ref)


def build_report(
    *,
    db_path: Path,
    order_id: str,
    trace_limit: int,
    child_limit: int,
    log_limit: int,
) -> dict[str, Any] | None:
    storage = SQLiteTaskStorage(db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=None)
    svc = StatusService(orch_q=orch_q, role_profiles=None, cache_ttl_seconds=0)
    packet = svc.order_release_readiness(
        order_id,
        trace_limit=trace_limit,
        child_limit=child_limit,
        log_limit=log_limit,
    )
    if not packet:
        return None
    digest = svc.order_handoff_digest(
        order_id,
        trace_limit=trace_limit,
        child_limit=child_limit,
        log_limit=log_limit,
    )
    return {"release_readiness": packet, "handoff_digest": digest}


def render_markdown(report: dict[str, Any]) -> str:
    packet = report.get("release_readiness") if isinstance(report.get("release_readiness"), dict) else {}
    digest = report.get("handoff_digest") if isinstance(report.get("handoff_digest"), dict) else {}
    order = packet.get("order") if isinstance(packet.get("order"), dict) else {}
    workflow = packet.get("workflow") if isinstance(packet.get("workflow"), dict) else {}
    readiness = packet.get("release_readiness") if isinstance(packet.get("release_readiness"), dict) else {}
    counts = packet.get("counts") if isinstance(packet.get("counts"), dict) else {}

    order_id = _one_line(packet.get("order_id") or digest.get("order_id"))
    title = _one_line(order.get("title") or (digest.get("order") or {}).get("title") if isinstance(digest.get("order"), dict) else "")
    status = _one_line(order.get("status") or (digest.get("order") or {}).get("status") if isinstance(digest.get("order"), dict) else "")
    phase = _one_line(readiness.get("phase") or digest.get("phase") or order.get("phase"))
    current_stage = _one_line(readiness.get("current_stage") or digest.get("current_stage") or workflow.get("current_stage"))
    state = _one_line(readiness.get("state") or digest.get("state"))
    verdict = _one_line(readiness.get("verdict") or digest.get("verdict"))
    summary = _one_line(readiness.get("summary") or digest.get("summary"))
    next_action = _one_line(readiness.get("next_action") or digest.get("next_action"))

    lines = [
        "# Order Release Readiness",
        "",
        f"- Order: {order_id}",
        f"- Title: {title}",
        f"- Status: {status}",
        f"- Phase: {phase}",
        f"- Current stage: {current_stage}",
        f"- Readiness state: {state}",
        f"- Verdict: {verdict}",
        f"- Summary: {summary}",
        f"- Next action: {next_action}",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence | Summary |",
        "| --- | --- | ---: | --- |",
    ]

    checks = [check for check in list(readiness.get("checks") or []) if isinstance(check, dict)]
    if checks:
        for check in checks:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _one_line(check.get("key")),
                        _one_line(check.get("status")),
                        str(_evidence_count(check)),
                        _one_line(check.get("summary")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | - | 0 | No readiness checks found. |")

    lines.extend(
        [
            "",
            "## Evidence Counts",
            "",
            f"- Children considered: {_one_line(counts.get('children_considered'))}",
            f"- Traces considered: {_one_line(counts.get('traces_considered'))}",
            f"- Decision log considered: {_one_line(counts.get('decision_log_considered'))}",
            f"- Artifacts considered: {_one_line(counts.get('artifacts_considered'))}",
        ]
    )

    refs = [ref for ref in list(digest.get("evidence_refs") or []) if isinstance(ref, dict)]
    lines.extend(["", "## Recent Evidence Refs", ""])
    if refs:
        for ref in refs:
            lines.append(f"- {_format_ref(ref)}")
    else:
        lines.append("- No recent evidence refs found.")

    blockers = [blocker for blocker in list(readiness.get("blockers") or digest.get("blockers") or []) if isinstance(blocker, dict)]
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:
            lines.append(f"- {_one_line(blocker.get('stage'))}: {_one_line(blocker.get('summary'))}")

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render offline release-readiness evidence for a proactive order.")
    parser.add_argument("--db", required=True, type=Path, help="Path to the orchestrator SQLite database.")
    parser.add_argument("--order-id", required=True, help="Order/root job id to report.")
    parser.add_argument("--format", choices=("json", "md", "markdown"), default="json", help="Output format.")
    parser.add_argument("--trace-limit", type=_positive_int, default=100, help="Maximum trace events to consider.")
    parser.add_argument("--child-limit", type=_positive_int, default=200, help="Maximum child jobs to consider.")
    parser.add_argument("--log-limit", type=_positive_int, default=200, help="Maximum decision log entries to consider.")
    parser.add_argument("--output", type=Path, help="Optional output file path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = _parse_args(argv)

    report = build_report(
        db_path=args.db,
        order_id=args.order_id,
        trace_limit=args.trace_limit,
        child_limit=args.child_limit,
        log_limit=args.log_limit,
    )
    if report is None:
        print(f"order not found: {args.order_id}", file=err)
        return 1

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
