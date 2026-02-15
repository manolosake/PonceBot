from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .schemas.result import TaskResult
from .schemas.task import Task


class RunnerError(RuntimeError):
    pass


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
        summary = str(value.get("summary", "completed"))
        artifacts = list(value.get("artifacts", []) or [])
        logs = str(value.get("logs", ""))
        next_action = value.get("next_action")
        structured = value.get("structured_digest", {})
        if not isinstance(structured, dict):
            structured = {"notes": structured}
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
