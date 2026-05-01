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


def _cell(value: Any, *, default: str = "-") -> str:
    return _one_line(value, default=default).replace("|", "\\|")


def _format_ref(ref: dict[str, Any]) -> str:
    bits: list[str] = []
    for key in (
        "kind",
        "key",
        "value",
        "role",
        "job_id_short",
        "state",
        "path",
        "artifact_id",
        "trace_event_id",
        "event_type",
        "decision_kind",
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
    return svc.order_handoff_digest(
        order_id,
        trace_limit=trace_limit,
        child_limit=child_limit,
        log_limit=log_limit,
    )


def render_markdown(report: dict[str, Any]) -> str:
    order = report.get("order") if isinstance(report.get("order"), dict) else {}
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    evidence_counts = report.get("evidence_counts") if isinstance(report.get("evidence_counts"), dict) else {}
    checks = [check for check in list(report.get("checks") or []) if isinstance(check, dict)]
    blockers = [blocker for blocker in list(report.get("blockers") or []) if isinstance(blocker, dict)]
    recent_jobs = [job for job in list(report.get("recent_jobs") or []) if isinstance(job, dict)]
    recent_artifacts = [artifact for artifact in list(report.get("recent_artifacts") or []) if isinstance(artifact, dict)]
    evidence_refs = [ref for ref in list(report.get("evidence_refs") or []) if isinstance(ref, dict)]
    operator_actions = [_one_line(action, default="") for action in list(report.get("operator_actions") or [])]
    release_manager_actions = [_one_line(action, default="") for action in list(report.get("release_manager_actions") or [])]

    lines = [
        "# Order Handoff Digest",
        "",
        f"- Order: {_one_line(report.get('order_id'))}",
        f"- Title: {_one_line(order.get('title'))}",
        f"- Status: {_one_line(order.get('status'))}",
        f"- Phase: {_one_line(report.get('phase'))}",
        f"- Current stage: {_one_line(report.get('current_stage'))}",
        f"- State: {_one_line(report.get('state'))}",
        f"- Verdict: {_one_line(report.get('verdict'))}",
        f"- Summary: {_one_line(report.get('summary'))}",
        f"- Next action: {_one_line(report.get('next_action'))}",
        "",
        "## Readiness Details",
        "",
        f"- Applies: {_one_line(readiness.get('applies'))}",
        f"- Scope: {_one_line(readiness.get('scope'))}",
        f"- Merge ready: {_one_line(readiness.get('merge_ready'))}",
        f"- Merged to main: {_one_line(readiness.get('merged_to_main'))}",
        f"- Deploy status: {_one_line(readiness.get('deploy_status'))}",
        f"- Deploy summary: {_one_line(readiness.get('deploy_summary'))}",
        f"- Deployed commit: {_one_line(readiness.get('deployed_commit'))}",
        f"- Checks total: {_one_line(readiness.get('checks_total'), default='0')}",
        f"- Checks by status: {_one_line(readiness.get('checks_by_status'))}",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence | Summary |",
        "| --- | --- | ---: | --- |",
    ]

    if checks:
        for check in checks:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(check.get("key")),
                        _cell(check.get("status")),
                        _cell(check.get("evidence_count"), default="0"),
                        _cell(check.get("summary")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | - | 0 | No readiness checks found. |")

    lines.extend(["", "## Blockers", ""])
    if blockers:
        for blocker in blockers:
            job = blocker.get("job") if isinstance(blocker.get("job"), dict) else {}
            job_ref = _one_line(job.get("job_id_short") or job.get("job_id"), default="-")
            lines.append(f"- {_one_line(blocker.get('stage'))}: {_one_line(blocker.get('summary'))} (job: {job_ref})")
    else:
        lines.append("- No blockers.")

    lines.extend(
        [
            "",
            "## Evidence Counts",
            "",
            f"- Children: {_one_line(evidence_counts.get('children'), default='0')}",
            f"- Traces: {_one_line(evidence_counts.get('traces'), default='0')}",
            f"- Decision log: {_one_line(evidence_counts.get('decision_log'), default='0')}",
            f"- Delegation log: {_one_line(evidence_counts.get('delegation_log'), default='0')}",
            f"- Artifacts: {_one_line(evidence_counts.get('artifacts'), default='0')}",
            f"- Handoff refs: {_one_line(evidence_counts.get('handoff_refs'), default='0')}",
            "",
            "## Recent Jobs",
            "",
            "| Job | Role | State | Title | Result summary | Next action |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    if recent_jobs:
        for job in recent_jobs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(job.get("job_id_short") or job.get("job_id")),
                        _cell(job.get("role")),
                        _cell(job.get("state")),
                        _cell(job.get("title")),
                        _cell(job.get("result_summary")),
                        _cell(job.get("result_next_action")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| - | - | - | No recent jobs found. | - | - |")

    lines.extend(["", "## Recent Artifacts", ""])
    if recent_artifacts:
        for artifact in recent_artifacts:
            lines.append(
                "- "
                + "; ".join(
                    [
                        f"kind={_one_line(artifact.get('kind'))}",
                        f"role={_one_line(artifact.get('role'))}",
                        f"job={_one_line(artifact.get('job_id_short') or artifact.get('job_id'))}",
                        f"path={_one_line(artifact.get('path'))}",
                        f"artifact_id={_one_line(artifact.get('artifact_id'))}",
                    ]
                )
            )
    else:
        lines.append("- No recent artifacts found.")

    lines.extend(["", "## Evidence Refs", ""])
    if evidence_refs:
        for ref in evidence_refs:
            lines.append(f"- {_format_ref(ref)}")
    else:
        lines.append("- No evidence refs found.")

    lines.extend(["", "## Operator Actions", ""])
    if operator_actions:
        for action in operator_actions:
            if action:
                lines.append(f"- {action}")
    else:
        lines.append("- No operator actions.")

    lines.extend(["", "## Release Manager Actions", ""])
    if release_manager_actions:
        for action in release_manager_actions:
            if action:
                lines.append(f"- {action}")
    else:
        lines.append("- No release manager actions.")

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline order handoff digest from the orchestrator SQLite database.")
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

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"database not found: {db_path}", file=err)
        return 2

    report = build_report(
        db_path=db_path,
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
