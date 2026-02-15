from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .schemas.result import TaskResult
from .schemas.task import Task


class RunnerError(RuntimeError):
    pass


def _extract_structured_json(text: str) -> tuple[str, dict[str, Any]]:
    """
    Best-effort extraction of a JSON object embedded in the agent output.

    Convention encouraged by prompting: include a ```json fenced block``` with a single JSON object.
    """
    raw = (text or "").strip()
    if not raw:
        return "", {}

    # 1) Look for a fenced JSON block.
    fence = "```json"
    if fence in raw:
        idx = raw.rfind(fence)
        tail = raw[idx + len(fence) :]
        end = tail.find("```")
        if end != -1:
            candidate = tail[:end].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    # Remove the fenced block from the human summary.
                    human = (raw[:idx] + raw[idx + len(fence) + end + len("```") :]).strip()
                    return human, parsed
            except Exception:
                pass

    # 2) Try parsing the last JSON object by scanning from the end.
    # This is intentionally conservative: only attempt when a "}" appears near the end.
    last_brace = raw.rfind("}")
    if last_brace != -1 and last_brace >= max(0, len(raw) - 12000):
        head = raw[: last_brace + 1]
        for start in range(head.rfind("{"), -1, -1):
            if head[start] != "{":
                continue
            candidate = head[start:]
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                human = raw[:start].strip()
                return human, parsed

    return raw, {}


def run_task(task: Task, *, executor: Any, cfg: Any | None = None) -> TaskResult:
    """Execute an orchestration task with a caller-provided executor.

    The executor should expose a `run(task, **kwargs)` method or a `start(job)`-style
    interface. This keeps the orchestrator runner independent from bot internals and
    allows phased rollout.
    """

    if task.requires_approval and not bool(task.trace.get("approved", False)):
        return TaskResult(
            status="blocked",
            summary="Execution blocked: task requires explicit approval.",
            artifacts=[],
            logs="",
            next_action="approve",
            structured_digest={"job_id": task.job_id, "reason": "requires_approval"},
        )

    start = time.time()

    try:
        if hasattr(executor, "run_task"):
            out = executor.run_task(task)
            duration = time.time() - start
            return _coerce_result(out, task, duration, status="ok")

        if hasattr(executor, "start"):
            exec_result = executor.start(task)
            return _coerce_result(exec_result, task, time.time() - start, status="ok")

        raise RunnerError("No compatible executor interface found")
    except Exception as e:
        return TaskResult(
            status="error",
            summary="Execution failed",
            artifacts=[],
            logs=str(e),
            next_action=None,
            structured_digest={"job_id": task.job_id},
        )


def _coerce_result(value: Any, task: Task, duration_s: float, *, status: str) -> TaskResult:
    if isinstance(value, TaskResult):
        return value

    if isinstance(value, dict):
        raw_summary = str(value.get("summary", "completed"))
        human_summary, parsed = _extract_structured_json(raw_summary)
        summary = human_summary or str(parsed.get("summary") or "").strip() or raw_summary
        artifacts = list(value.get("artifacts", []) or [])
        logs = str(value.get("logs", ""))
        next_action = value.get("next_action")
        structured = value.get("structured_digest", {})
        if not isinstance(structured, dict):
            structured = {"notes": structured}
        if parsed:
            structured = {**structured, **parsed}
        return TaskResult(
            status=str(value.get("status", status)),
            summary=summary,
            artifacts=artifacts,
            logs=logs,
            next_action=None if next_action is None else str(next_action),
            structured_digest={"job_id": task.job_id, "duration_s": duration_s, **structured},
        )

    return TaskResult(
        status=status,
        summary="completed" if value is None else str(value),
        artifacts=[],
        logs="",
        next_action=None,
        structured_digest={"job_id": task.job_id, "duration_s": duration_s},
    )
