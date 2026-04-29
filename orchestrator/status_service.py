from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import hashlib
import json
import time

from .queue import OrchestratorQueue
from .schemas.task import Task


def _max_parallel_by_role(role_profiles: dict[str, dict[str, Any]] | None) -> dict[str, int]:
    """
    Keep aligned with OrchestratorQueue._max_parallel_by_role().
    """
    out: dict[str, int] = {
        "jarvis": 1,
        "frontend": 2,
        "backend": 2,
        "qa": 2,
        "sre": 2,
        "product_ops": 1,
        "security": 1,
        "research": 1,
        "release_mgr": 1,
        "architect_local": 1,
        "implementer_local": 1,
        "reviewer_local": 1,
    }
    for key, cfg in (role_profiles or {}).items():
        if not isinstance(cfg, dict):
            continue
        role = str(cfg.get("role") or key).strip().lower()
        if not role:
            continue
        try:
            out[role] = int(cfg.get("max_parallel_jobs", out.get(role, 1)))
        except Exception:
            pass
    # Normalize non-positive to 1.
    for r in list(out.keys()):
        try:
            out[r] = max(1, int(out[r]))
        except Exception:
            out[r] = 1
    return out


def _task_title(t: Task, *, max_chars: int = 120) -> str:
    s = (t.input_text or "").strip().replace("\n", " ")
    if len(s) > max_chars:
        s = s[:max_chars] + "..."
    return s


def _task_to_status(t: Task) -> dict[str, Any]:
    tr = t.trace or {}

    live_phase = str(tr.get("live_phase") or "").strip() or None

    slot = tr.get("live_workspace_slot")
    try:
        slot_i = int(slot) if slot is not None else None
    except Exception:
        slot_i = None

    live_at = tr.get("live_at")
    try:
        live_at_f = float(live_at) if live_at is not None else None
    except Exception:
        live_at_f = None

    pid = tr.get("live_pid")
    try:
        pid_i = int(pid) if pid is not None else None
    except Exception:
        pid_i = None

    live_workdir = str(tr.get("live_workdir") or "").strip() or None

    stdout_tail = str(tr.get("live_stdout_tail") or "").strip()
    if len(stdout_tail) > 1600:
        stdout_tail = stdout_tail[-1600:]
    stdout_tail = stdout_tail or None

    stderr_tail = str(tr.get("live_stderr_tail") or "").strip()
    if len(stderr_tail) > 1600:
        stderr_tail = stderr_tail[-1600:]
    stderr_tail = stderr_tail or None

    result_summary = str(tr.get("result_summary") or "").strip()
    if len(result_summary) > 600:
        result_summary = result_summary[:600] + "..."
    result_summary = result_summary or None

    result_next_action = str(tr.get("result_next_action") or "").strip() or None

    approved = bool(tr.get("approved", False))

    return {
        "job_id": t.job_id,
        "job_id_short": t.job_id[:8],
        "role": t.role,
        "state": t.state,
        "priority": int(t.priority or 2),
        "request_type": t.request_type,
        "mode_hint": t.mode_hint,
        "requires_approval": bool(t.requires_approval),
        "approved": approved,
        "owner": t.owner,
        "chat_id": int(t.chat_id),
        "user_id": (int(t.user_id) if t.user_id is not None else None),
        "parent_job_id": t.parent_job_id,
        "created_at": float(t.created_at),
        "updated_at": float(t.updated_at),
        "title": _task_title(t),
        "live_phase": live_phase,
        "live_at": live_at_f,
        "live_pid": pid_i,
        "live_workdir": live_workdir,
        "live_workspace_slot": slot_i,
        "live_stdout_tail": stdout_tail,
        "live_stderr_tail": stderr_tail,
        "result_summary": result_summary,
        "result_next_action": result_next_action,
    }

def _assign_running_to_workers(
    role: str,
    running: list[Task],
    max_parallel: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    """
    Assign running tasks to worker slots using live_workspace_slot when available,
    otherwise assign sequentially.

    Returns (workers, occupied_slots).
    """
    workers: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for i in range(1, max_parallel + 1):
        workers.append(
            {
                "worker_id": f"{role}:{i}",
                "role": role,
                "slot": i,
                "current": None,
                "next": None,
            }
        )

    # Prefer explicit slot if present.
    remaining: list[Task] = []
    for t in running:
        slot = (t.trace or {}).get("live_workspace_slot")
        try:
            slot_i = int(slot) if slot is not None else None
        except Exception:
            slot_i = None
        if slot_i is not None and 1 <= slot_i <= max_parallel and slot_i not in occupied:
            workers[slot_i - 1]["current"] = _task_to_status(t)
            occupied.add(slot_i)
        else:
            remaining.append(t)

    # Fill unassigned slots.
    for t in remaining:
        slot_i = None
        for i in range(1, max_parallel + 1):
            if i not in occupied and workers[i - 1]["current"] is None:
                slot_i = i
                break
        if slot_i is None:
            break
        workers[slot_i - 1]["current"] = _task_to_status(t)
        occupied.add(slot_i)

    return workers, occupied


_DELIVERY_ROLES = {"backend", "frontend", "sre", "security", "implementer_local"}
_VALIDATION_ROLES = {"qa", "release_mgr", "reviewer_local"}
_CONTROLLER_ROLES = {"skynet"}


def _count_task_states(tasks: list[Task]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tasks:
        state = str(t.state or "").strip().lower() or "unknown"
        counts[state] = counts.get(state, 0) + 1
    return counts


def _latest_task(tasks: list[Task]) -> Task | None:
    if not tasks:
        return None
    return max(tasks, key=lambda t: float(t.updated_at or 0.0))


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _task_kind(task: Task | None) -> str | None:
    if task is None:
        return None
    labels = task.labels or {}
    kind = str(labels.get("kind") or "").strip().lower()
    return kind or None


def _workflow_task_ref(task: Task | None) -> dict[str, Any] | None:
    if task is None:
        return None
    return {
        "job_id": task.job_id,
        "job_id_short": task.job_id[:8],
        "role": str(task.role or "").strip().lower(),
        "state": str(task.state or "").strip().lower(),
        "kind": _task_kind(task),
        "updated_at": float(task.updated_at or 0.0),
    }


def _controller_failure_summary(task: Task | None) -> str | None:
    if task is None:
        return None
    trace = task.trace or {}
    stdout_tail = str(trace.get("live_stdout_tail") or "").strip()
    stderr_tail = str(trace.get("live_stderr_tail") or "").strip()
    result_summary = str(trace.get("result_summary") or "").strip()
    if "not supported when using Codex with a ChatGPT account" in stdout_tail:
        return "Configured Codex model is unsupported for the current ChatGPT-backed account."
    if result_summary:
        return result_summary[:280]
    if stderr_tail:
        return stderr_tail[:280]
    if stdout_tail:
        return stdout_tail[:280]
    return None


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            s = str(item or "").strip()
            if s:
                out.append(s)
        return out
    return []


def _artifact_refs_from_task(task: Task) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    base = {
        "job_id": task.job_id,
        "job_id_short": task.job_id[:8],
        "role": str(task.role or "").strip().lower(),
    }
    artifacts_dir = str(task.artifacts_dir or "").strip()
    if artifacts_dir:
        refs.append({**base, "kind": "artifacts_dir", "path": artifacts_dir})

    trace = task.trace or {}
    for key in ("result_artifacts", "artifacts"):
        for path in _coerce_str_list(trace.get(key)):
            refs.append({**base, "kind": key, "path": path})
    return refs


def _artifact_refs_from_trace_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        base = {
            "trace_event_id": event.get("id"),
            "job_id": event.get("job_id"),
            "job_id_short": (str(event.get("job_id"))[:8] if event.get("job_id") else None),
            "role": event.get("agent_role"),
        }
        artifact_id = str(event.get("artifact_id") or "").strip()
        if artifact_id:
            refs.append({**base, "kind": "trace_artifact_id", "artifact_id": artifact_id})
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        for key in ("result_artifacts", "artifacts"):
            for path in _coerce_str_list(payload.get(key)):
                refs.append({**base, "kind": f"trace_payload_{key}", "path": path})
    return refs


def _derive_stage_status(
    *,
    running: bool,
    failed: bool,
    blocked: bool,
    done: bool,
    pending: bool,
) -> str:
    if running:
        return "running"
    if failed:
        return "failed"
    if blocked:
        return "blocked"
    if done:
        return "done"
    if pending:
        return "pending"
    return "pending"


def _build_order_workflow(
    *,
    order_row: dict[str, Any],
    root_task: Task | None,
    children: list[Task],
) -> dict[str, Any]:
    root_trace = dict((root_task.trace or {}) if root_task is not None else {})
    delivery_children = [t for t in children if str(t.role or "").strip().lower() in _DELIVERY_ROLES]
    validation_children = [t for t in children if str(t.role or "").strip().lower() in _VALIDATION_ROLES]
    controller_children = [t for t in children if str(t.role or "").strip().lower() in _CONTROLLER_ROLES]

    delivery_counts = _count_task_states(delivery_children)
    validation_counts = _count_task_states(validation_children)
    controller_counts = _count_task_states(controller_children)

    latest_delivery = _latest_task(delivery_children)
    latest_validation = _latest_task(validation_children)
    latest_controller = _latest_task(controller_children)

    applied_count = _coerce_int(root_trace.get("proactive_slices_applied"))
    validated_count = _coerce_int(root_trace.get("proactive_slices_validated"))
    closed_count = _coerce_int(root_trace.get("proactive_slices_closed"))
    quality_gate_status = str(root_trace.get("proactive_quality_gate_status") or "").strip().lower()
    improvement_verified = bool(root_trace.get("proactive_improvement_verified", False))
    merge_ready = bool(root_trace.get("merge_ready", False))
    merged_to_main = bool(root_trace.get("merged_to_main", False))
    deploy_status = str(root_trace.get("deploy_status") or "").strip().lower()
    deploy_summary = str(root_trace.get("deploy_summary") or "").strip()
    deployed_commit = str(root_trace.get("deployed_commit") or "").strip() or None

    plan_status = _derive_stage_status(
        running=bool(root_task is not None and str(root_task.state or "").strip().lower() == "running"),
        failed=bool(root_task is not None and str(root_task.state or "").strip().lower() == "failed"),
        blocked=bool(root_task is not None and str(root_task.state or "").strip().lower() in {"blocked", "blocked_approval", "waiting_deps"}),
        done=bool(root_task is not None and str(root_task.state or "").strip().lower() in {"done", "cancelled"}),
        pending=(root_task is None),
    )

    delivery_status = _derive_stage_status(
        running=delivery_counts.get("running", 0) > 0,
        failed=delivery_counts.get("failed", 0) > 0,
        blocked=(delivery_counts.get("blocked", 0) + delivery_counts.get("blocked_approval", 0) + delivery_counts.get("waiting_deps", 0)) > 0,
        done=(applied_count > 0) or (delivery_counts.get("done", 0) > 0),
        pending=(len(delivery_children) == 0),
    )

    validation_status = _derive_stage_status(
        running=validation_counts.get("running", 0) > 0,
        failed=validation_counts.get("failed", 0) > 0,
        blocked=(validation_counts.get("blocked", 0) + validation_counts.get("blocked_approval", 0) + validation_counts.get("waiting_deps", 0)) > 0,
        done=(validated_count > 0) or (validation_counts.get("done", 0) > 0),
        pending=(applied_count > 0) or (quality_gate_status == "applied") or (len(validation_children) == 0),
    )

    controller_done = bool(improvement_verified or closed_count > 0)
    controller_summary = _controller_failure_summary(latest_controller)
    controller_status = _derive_stage_status(
        running=controller_counts.get("running", 0) > 0,
        failed=controller_counts.get("failed", 0) > 0,
        blocked=(controller_counts.get("blocked", 0) + controller_counts.get("blocked_approval", 0) + controller_counts.get("waiting_deps", 0)) > 0,
        done=controller_done,
        pending=(validated_count > 0) or (len(controller_children) == 0),
    )

    deploy_done = deploy_status in {"ok", "scheduled"}
    deploy_failed = deploy_status == "failed"
    deploy_status_label = _derive_stage_status(
        running=bool(merged_to_main and deploy_status in {"queued", "running", "scheduled"}),
        failed=deploy_failed,
        blocked=False,
        done=deploy_done,
        pending=bool(controller_done or merge_ready or merged_to_main),
    )

    blockers: list[dict[str, Any]] = []
    if delivery_status in {"failed", "blocked"}:
        blockers.append(
            {
                "stage": "delivery",
                "summary": str((latest_delivery.trace or {}).get("result_summary") or latest_delivery.blocked_reason or "Delivery work is blocked.")[:280]
                if latest_delivery is not None
                else "Delivery work is blocked.",
                "job": _workflow_task_ref(latest_delivery),
            }
        )
    if validation_status in {"failed", "blocked"}:
        blockers.append(
            {
                "stage": "validation",
                "summary": str((latest_validation.trace or {}).get("result_summary") or latest_validation.blocked_reason or "Validation did not pass.")[:280]
                if latest_validation is not None
                else "Validation did not pass.",
                "job": _workflow_task_ref(latest_validation),
            }
        )
    if controller_status in {"failed", "blocked"}:
        blockers.append(
            {
                "stage": "skynet_review",
                "summary": controller_summary or "Skynet review did not complete cleanly.",
                "job": _workflow_task_ref(latest_controller),
            }
        )
    if deploy_failed:
        blockers.append(
            {
                "stage": "deploy",
                "summary": deploy_summary[:280] if deploy_summary else "Deploy failed after merge.",
                "job": None,
            }
        )

    stages = [
        {
            "stage": "skynet_plan",
            "label": "Skynet plan",
            "status": plan_status,
            "job": _workflow_task_ref(root_task),
            "counts": {"children_total": len(children)},
            "summary": str(root_trace.get("result_summary") or "").strip()[:280] or None,
        },
        {
            "stage": "delivery",
            "label": "Delivery",
            "status": delivery_status,
            "job": _workflow_task_ref(latest_delivery),
            "counts": delivery_counts,
            "summary": (
                f"applied={applied_count} started={_coerce_int(root_trace.get('proactive_slices_started'))}"
                if (applied_count > 0 or delivery_children)
                else None
            ),
        },
        {
            "stage": "validation",
            "label": "Validation",
            "status": validation_status,
            "job": _workflow_task_ref(latest_validation),
            "counts": validation_counts,
            "summary": (
                f"validated={validated_count} quality_gate={quality_gate_status or 'n/a'}"
                if (validated_count > 0 or quality_gate_status or validation_children)
                else None
            ),
        },
        {
            "stage": "skynet_review",
            "label": "Skynet review",
            "status": controller_status,
            "job": _workflow_task_ref(latest_controller),
            "counts": controller_counts,
            "summary": controller_summary,
        },
        {
            "stage": "deploy",
            "label": "Deploy",
            "status": deploy_status_label,
            "job": None,
            "counts": {},
            "summary": deploy_summary[:280] if deploy_summary else None,
        },
    ]

    current_stage = "deploy"
    for status_priority in ("running", "failed", "blocked", "pending"):
        for stage in stages:
            if str(stage.get("status") or "") == status_priority:
                current_stage = str(stage.get("stage") or "deploy")
                break
        if current_stage != "deploy":
            break

    return {
        "order_id": str(order_row.get("order_id") or ""),
        "order_id_short": str(order_row.get("order_id") or "")[:8],
        "phase": str(order_row.get("phase") or "planning"),
        "current_stage": current_stage,
        "merge_ready": merge_ready,
        "merged_to_main": merged_to_main,
        "deploy_status": deploy_status or None,
        "deploy_summary": deploy_summary or None,
        "deployed_commit": deployed_commit,
        "stages": stages,
        "blockers": blockers,
    }


_BLOCKING_TASK_STATES = {"failed", "blocked", "blocked_approval", "waiting_deps"}


def _readiness_trace_evidence(key: str, value: Any) -> dict[str, Any]:
    return {"kind": "trace", "key": key, "value": value}


def _readiness_job_evidence(task: Task | None) -> list[dict[str, Any]]:
    ref = _workflow_task_ref(task)
    return [{"kind": "job", **ref}] if ref is not None else []


def _readiness_artifact_evidence(artifacts: list[dict[str, Any]], *, roles: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        role = str(artifact.get("role") or "").strip().lower()
        if role not in roles:
            continue
        item: dict[str, Any] = {"kind": "artifact", "role": role}
        for key in ("job_id", "job_id_short", "path", "artifact_id"):
            if artifact.get(key):
                item[key] = artifact.get(key)
        out.append(item)
        if len(out) >= 3:
            break
    return out


def _readiness_decision_evidence(decision_log: list[dict[str, Any]], *, kinds: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in decision_log:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in kinds:
            continue
        out.append(
            {
                "kind": "decision_log",
                "decision_kind": kind,
                "state": str(item.get("state") or "").strip().lower(),
                "job_id": item.get("job_id"),
                "job_id_short": item.get("job_id_short"),
                "summary": str(item.get("summary") or "").strip()[:280] or None,
            }
        )
        if len(out) >= 3:
            break
    return out


def _readiness_trace_event_evidence(traces: list[dict[str, Any]], *, event_types: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in traces:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type") or "").strip().lower()
        if event_type not in event_types:
            continue
        out.append(
            {
                "kind": "trace_event",
                "trace_event_id": event.get("id"),
                "event_type": event_type,
                "job_id": event.get("job_id"),
                "job_id_short": (str(event.get("job_id"))[:8] if event.get("job_id") else None),
                "role": event.get("agent_role"),
                "artifact_id": event.get("artifact_id"),
            }
        )
        if len(out) >= 3:
            break
    return out


def _task_summary(task: Task | None, fallback: str) -> str:
    if task is None:
        return fallback
    trace = task.trace or {}
    summary = str(trace.get("result_summary") or task.blocked_reason or "").strip()
    return summary[:280] if summary else fallback


def _has_patch_artifact_evidence(task: Task) -> bool:
    trace = task.trace or {}
    patch_info = trace.get("local_patch_info")
    if not isinstance(patch_info, dict):
        patch_info = trace.get("patch_info")
    changed = patch_info.get("changed_files") if isinstance(patch_info, dict) else None
    return bool(
        trace.get("slice_patch_applied")
        or (isinstance(changed, list) and any(str(item or "").strip() for item in changed))
        or _coerce_str_list(trace.get("result_artifacts"))
        or _coerce_str_list(trace.get("artifacts"))
        or str(task.artifacts_dir or "").strip()
    )


def _has_validation_evidence(task: Task) -> bool:
    trace = task.trace or {}
    patch_info = trace.get("local_patch_info")
    if not isinstance(patch_info, dict):
        patch_info = trace.get("patch_info")
    return bool(
        trace.get("slice_validation_ok")
        or trace.get("review_ready")
        or trace.get("improvement_verified")
        or trace.get("quality_gate_status") in {"validated", "reviewed_ready", "closed"}
        or trace.get("slice_status") in {"validated", "reviewed_ready", "closed"}
        or (isinstance(patch_info, dict) and bool(patch_info.get("validation_ok")))
    )


def _build_release_readiness(
    *,
    order_row: dict[str, Any],
    root_task: Task | None,
    children: list[Task],
    workflow: dict[str, Any],
    decision_log: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    root_trace = dict((root_task.trace or {}) if root_task is not None else {})
    proactive_keys_present = any(str(key).startswith("proactive_") for key in root_trace.keys())
    applies = bool(root_trace.get("proactive_lane", False) or proactive_keys_present)
    phase = str(order_row.get("phase") or "planning")
    current_stage = str(workflow.get("current_stage") or "skynet_plan")
    check_keys = ["delivery_applied", "validation_passed", "controller_signoff", "release_evidence", "merge_ready"]

    if not applies:
        return {
            "schema_version": 1,
            "scope": "proactive_order",
            "applies": False,
            "state": "not_applicable",
            "verdict": "n/a",
            "summary": "Release readiness applies only to proactive orders.",
            "current_stage": current_stage,
            "phase": phase,
            "checks": [
                {"key": key, "status": "na", "summary": "Not a proactive order.", "evidence": []}
                for key in check_keys
            ],
            "blockers": [],
            "next_action": "Use the standard order workflow fields.",
        }

    delivery_children = [t for t in children if str(t.role or "").strip().lower() in _DELIVERY_ROLES]
    validation_children = [t for t in children if str(t.role or "").strip().lower() in _VALIDATION_ROLES and str(t.role or "").strip().lower() != "release_mgr"]
    release_children = [t for t in children if str(t.role or "").strip().lower() == "release_mgr"]
    controller_children = [t for t in children if str(t.role or "").strip().lower() in _CONTROLLER_ROLES]

    latest_delivery = _latest_task(delivery_children)
    latest_validation = _latest_task(validation_children)
    latest_release = _latest_task(release_children)
    latest_controller = _latest_task(controller_children)

    applied_count = _coerce_int(root_trace.get("proactive_slices_applied"))
    validated_count = _coerce_int(root_trace.get("proactive_slices_validated"))
    closed_count = _coerce_int(root_trace.get("proactive_slices_closed"))
    quality_gate_status = str(root_trace.get("proactive_quality_gate_status") or "").strip().lower()
    merge_ready = bool(root_trace.get("merge_ready", False))
    merged_to_main = bool(root_trace.get("merged_to_main", False))
    deploy_status = str(root_trace.get("deploy_status") or "").strip().lower()
    already_released = bool(merged_to_main or deploy_status in {"ok", "scheduled"})

    delivery_blocked = [t for t in delivery_children if str(t.state or "").strip().lower() in _BLOCKING_TASK_STATES]
    validation_blocked = [t for t in validation_children if str(t.state or "").strip().lower() in _BLOCKING_TASK_STATES]
    release_blocked = [t for t in release_children if str(t.state or "").strip().lower() in _BLOCKING_TASK_STATES]
    controller_blocked = [t for t in controller_children if str(t.state or "").strip().lower() in _BLOCKING_TASK_STATES]

    blockers: list[dict[str, Any]] = []
    for stage, tasks, fallback in (
        ("delivery", delivery_blocked, "Delivery work is blocked."),
        ("validation", validation_blocked, "Validation did not pass."),
        ("release_evidence", release_blocked, "Release evidence is blocked."),
        ("controller_signoff", controller_blocked, "Controller signoff did not complete cleanly."),
    ):
        if not tasks:
            continue
        task = _latest_task(tasks)
        blockers.append({"stage": stage, "summary": _task_summary(task, fallback), "job": _workflow_task_ref(task)})
    if deploy_status == "failed":
        blockers.append({"stage": "release", "summary": str(root_trace.get("deploy_summary") or "Deploy failed after merge.")[:280], "job": None})

    delivery_applied = bool(applied_count > 0 or any(str(t.state or "").strip().lower() == "done" and _has_patch_artifact_evidence(t) for t in delivery_children))
    validation_passed = bool(
        validated_count > 0
        or quality_gate_status in {"validated", "reviewed_ready", "closed"}
        or any(str(t.state or "").strip().lower() == "done" and _has_validation_evidence(t) for t in [*validation_children, *release_children])
    )
    controller_signoff = bool(
        closed_count > 0
        or root_trace.get("proactive_improvement_verified")
        or root_trace.get("proactive_improvement_closed")
        or root_trace.get("proactive_no_change_validated")
        or any(str(t.state or "").strip().lower() == "done" and _has_validation_evidence(t) for t in controller_children)
    )
    release_evidence = bool(
        root_trace.get("proactive_improvement_closed")
        or root_trace.get("proactive_no_change_validated")
        or any(str(t.state or "").strip().lower() == "done" and (_has_validation_evidence(t) or _has_patch_artifact_evidence(t)) for t in release_children)
        or _readiness_decision_evidence(decision_log, kinds={"release", "release_evidence"})
        or _readiness_trace_event_evidence(traces, event_types={"order.proactive_verified_improvement_closed", "order.proactive_no_change_closed"})
    )
    merge_ready_pass = bool(merge_ready or already_released)

    checks = [
        {
            "key": "delivery_applied",
            "status": "fail" if delivery_blocked else ("pass" if delivery_applied else "pending"),
            "summary": (
                "Delivery changes are applied."
                if delivery_applied
                else ("Delivery is blocked." if delivery_blocked else "Waiting for applied delivery evidence.")
            ),
            "evidence": (
                [_readiness_trace_evidence("proactive_slices_applied", applied_count)] if applied_count > 0 else _readiness_job_evidence(latest_delivery)
            ),
        },
        {
            "key": "validation_passed",
            "status": "fail" if validation_blocked else ("pass" if validation_passed else "pending"),
            "summary": (
                "Validation evidence is present."
                if validation_passed
                else ("Validation is blocked." if validation_blocked else "Waiting for validation evidence.")
            ),
            "evidence": (
                [_readiness_trace_evidence("proactive_slices_validated", validated_count), _readiness_trace_evidence("proactive_quality_gate_status", quality_gate_status)]
                if validation_passed
                else _readiness_job_evidence(latest_validation)
            ),
        },
        {
            "key": "controller_signoff",
            "status": "fail" if controller_blocked else ("pass" if controller_signoff else "pending"),
            "summary": (
                "Controller signoff is present."
                if controller_signoff
                else ("Controller signoff is blocked." if controller_blocked else "Waiting for controller signoff.")
            ),
            "evidence": (
                [_readiness_trace_evidence("proactive_slices_closed", closed_count)]
                if closed_count > 0
                else _readiness_job_evidence(latest_controller)
            ),
        },
        {
            "key": "release_evidence",
            "status": "fail" if release_blocked else ("pass" if release_evidence else "pending"),
            "summary": (
                "Release evidence is present."
                if release_evidence
                else ("Release evidence is blocked." if release_blocked else "Waiting for release evidence.")
            ),
            "evidence": (
                [
                    *_readiness_artifact_evidence(artifacts, roles={"release_mgr", "qa", "reviewer_local", "skynet"}),
                    *_readiness_decision_evidence(decision_log, kinds={"release", "release_evidence"}),
                    *_readiness_trace_event_evidence(traces, event_types={"order.proactive_verified_improvement_closed", "order.proactive_no_change_closed"}),
                ][:5]
                or ([_readiness_trace_evidence("proactive_improvement_closed", True)] if root_trace.get("proactive_improvement_closed") else _readiness_job_evidence(latest_release))
            ),
        },
        {
            "key": "merge_ready",
            "status": "pass" if merge_ready_pass else "pending",
            "summary": (
                "Order is already merged or released."
                if already_released
                else ("Order is marked ready for merge." if merge_ready else "Waiting for merge-ready signal.")
            ),
            "evidence": [
                _readiness_trace_evidence("merge_ready", merge_ready),
                _readiness_trace_evidence("merged_to_main", merged_to_main),
            ],
        },
    ]

    if blockers:
        state = "blocked"
        verdict = "no_go"
        summary = f"Release is blocked at {blockers[0]['stage']}."
        next_action = blockers[0]["summary"]
    elif already_released:
        state = "released"
        verdict = "go"
        summary = "Proactive order is already merged or released."
        next_action = "Monitor deploy status and post-release evidence."
    elif all(str(check.get("status") or "") == "pass" for check in checks):
        state = "ready"
        verdict = "go"
        summary = "Proactive order has delivery, validation, controller, release, and merge-ready evidence."
        next_action = "Release or merge the order branch."
    else:
        state = "not_ready"
        verdict = "wait"
        first_pending = next((check for check in checks if str(check.get("status") or "") == "pending"), None)
        pending_key = str(first_pending.get("key") or "release_readiness") if first_pending else "release_readiness"
        summary = f"Proactive order is waiting on {pending_key}."
        next_action = str(first_pending.get("summary") or "Continue the proactive workflow.") if first_pending else "Continue the proactive workflow."

    return {
        "schema_version": 1,
        "scope": "proactive_order",
        "applies": True,
        "state": state,
        "verdict": verdict,
        "summary": summary,
        "current_stage": current_stage,
        "phase": phase,
        "checks": checks,
        "blockers": blockers,
        "next_action": next_action,
    }


@dataclass
class StatusService:
    orch_q: OrchestratorQueue
    role_profiles: dict[str, dict[str, Any]] | None = None
    cache_ttl_seconds: int = 2
    factory_snapshot_fn: Callable[[int | None], dict[str, Any]] | None = None

    def autonomy_board(self, *, chat_id: int | None = None, limit: int = 50) -> dict[str, Any]:
        lim = max(1, min(200, int(limit)))
        generated_at = float(time.time())
        if chat_id is None:
            orders = self.orch_q.list_orders_global(status="active", limit=lim)
        else:
            orders = self.orch_q.list_orders(chat_id=int(chat_id), status="active", limit=lim)

        rows: list[dict[str, Any]] = []
        readiness_counts: dict[str, int] = {}
        stage_counts: dict[str, int] = {}
        for order_row in orders:
            oid = str((order_row or {}).get("order_id") or "").strip()
            if not oid:
                continue
            try:
                root_task = self.orch_q.get_job(oid)
            except Exception:
                root_task = None
            try:
                children = [t for t in self.orch_q.jobs_by_parent(parent_job_id=oid, limit=200) if t.job_id != oid]
            except Exception:
                children = []
            try:
                traces = self.orch_q.list_trace_events(order_id=oid, limit=100)
            except Exception:
                traces = []
            try:
                decision_log = self.orch_q.list_decision_log(order_id=oid, limit=50)
            except Exception:
                decision_log = []

            tasks = ([root_task] if root_task is not None else []) + children
            artifacts: list[dict[str, Any]] = []
            for task in tasks:
                artifacts.extend(_artifact_refs_from_task(task))
            artifacts.extend(_artifact_refs_from_trace_events(traces))

            workflow = _build_order_workflow(order_row=order_row, root_task=root_task, children=children)
            readiness = _build_release_readiness(
                order_row=order_row,
                root_task=root_task,
                children=children,
                workflow=workflow,
                decision_log=decision_log,
                traces=traces,
                artifacts=artifacts,
            )
            children_by_state = _count_task_states(children)
            children_by_role: dict[str, int] = {}
            for child in children:
                role = str(child.role or "").strip().lower() or "unknown"
                children_by_role[role] = children_by_role.get(role, 0) + 1

            readiness_state = str(readiness.get("state") or "unknown")
            current_stage = str(workflow.get("current_stage") or readiness.get("current_stage") or "skynet_plan")
            readiness_counts[readiness_state] = readiness_counts.get(readiness_state, 0) + 1
            stage_counts[current_stage] = stage_counts.get(current_stage, 0) + 1
            blockers = list(readiness.get("blockers") or workflow.get("blockers") or [])
            updated_at = float((order_row or {}).get("updated_at") or (root_task.updated_at if root_task is not None else 0.0) or 0.0)

            rows.append(
                {
                    "order_id": oid,
                    "order_id_short": oid[:8],
                    "chat_id": int((order_row or {}).get("chat_id") or (root_task.chat_id if root_task is not None else 0) or 0),
                    "title": str((order_row or {}).get("title") or (_task_title(root_task) if root_task is not None else "")),
                    "priority": int((order_row or {}).get("priority") or (root_task.priority if root_task is not None else 2) or 2),
                    "phase": str((order_row or {}).get("phase") or "planning"),
                    "current_stage": current_stage,
                    "readiness_state": readiness_state,
                    "readiness_verdict": str(readiness.get("verdict") or "unknown"),
                    "blockers": blockers,
                    "next_action": str(readiness.get("next_action") or ""),
                    "updated_at": updated_at,
                    "children_total": len(children),
                    "children_by_state": children_by_state,
                    "children_by_role": children_by_role,
                    "merge_ready": bool(workflow.get("merge_ready")),
                    "merged_to_main": bool(workflow.get("merged_to_main")),
                }
            )

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": generated_at,
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "limit": lim,
            "summary": {
                "orders_total": len(rows),
                "by_readiness_state": readiness_counts,
                "by_current_stage": stage_counts,
            },
            "orders": rows,
        }

    def order_evidence_packet(
        self,
        order_id: str,
        *,
        trace_limit: int = 100,
        child_limit: int = 200,
        log_limit: int = 200,
    ) -> dict[str, Any]:
        oid = str(order_id or "").strip()
        if not oid:
            return {}

        root_task = self.orch_q.get_job(oid)
        if root_task is None:
            return {}

        child_lim = max(1, min(2000, int(child_limit)))
        trace_lim = max(1, min(5000, int(trace_limit)))
        log_lim = max(1, min(2000, int(log_limit)))

        children = [t for t in self.orch_q.jobs_by_parent(parent_job_id=oid, limit=child_lim) if t.job_id != oid]
        order_row: dict[str, Any] | None = None
        try:
            order_row = self.orch_q.get_order(oid, chat_id=int(root_task.chat_id))
        except Exception:
            order_row = None
        if not isinstance(order_row, dict):
            order_row = {
                "order_id": oid,
                "chat_id": int(root_task.chat_id),
                "status": str(root_task.state or ""),
                "phase": "planning",
                "intent_type": "",
                "project_id": None,
                "source_message_id": root_task.reply_to_message_id,
                "reply_to_message_id": root_task.reply_to_message_id,
                "priority": int(root_task.priority or 2),
                "title": _task_title(root_task),
                "updated_at": float(root_task.updated_at or 0.0),
            }

        traces = self.orch_q.list_trace_events(order_id=oid, limit=trace_lim)
        decision_log = self.orch_q.list_decision_log(order_id=oid, limit=log_lim)
        delegation_log = self.orch_q.list_delegation_log(root_ticket_id=oid, limit=log_lim)
        tasks = [root_task, *children]
        artifacts: list[dict[str, Any]] = []
        for task in tasks:
            artifacts.extend(_artifact_refs_from_task(task))
        artifacts.extend(_artifact_refs_from_trace_events(traces))

        child_statuses = [_task_to_status(t) for t in children]
        counts_by_state = _count_task_states(children)
        counts_by_role: dict[str, int] = {}
        for child in children:
            role = str(child.role or "").strip().lower() or "unknown"
            counts_by_role[role] = counts_by_role.get(role, 0) + 1

        workflow = _build_order_workflow(order_row=order_row, root_task=root_task, children=children)
        release_readiness = _build_release_readiness(
            order_row=order_row,
            root_task=root_task,
            children=children,
            workflow=workflow,
            decision_log=decision_log,
            traces=traces,
            artifacts=artifacts,
        )

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": float(time.time()),
            "order_id": oid,
            "order_id_short": oid[:8],
            "order": {
                "order_id": oid,
                "chat_id": int(order_row.get("chat_id") or root_task.chat_id),
                "status": str(order_row.get("status") or ""),
                "phase": str(order_row.get("phase") or "planning"),
                "intent_type": str(order_row.get("intent_type") or ""),
                "project_id": (str(order_row.get("project_id") or "").strip() or None),
                "source_message_id": order_row.get("source_message_id"),
                "reply_to_message_id": order_row.get("reply_to_message_id"),
                "priority": int(order_row.get("priority") or root_task.priority or 2),
                "title": str(order_row.get("title") or _task_title(root_task)),
                "updated_at": float(order_row.get("updated_at") or root_task.updated_at or 0.0),
            },
            "root": _task_to_status(root_task),
            "workflow": workflow,
            "release_readiness": release_readiness,
            "children": child_statuses,
            "traces": traces,
            "decision_log": decision_log,
            "delegation_log": delegation_log,
            "artifacts": artifacts,
            "counts": {
                "children": len(child_statuses),
                "children_by_state": counts_by_state,
                "children_by_role": counts_by_role,
                "traces": len(traces),
                "decision_log": len(decision_log),
                "delegation_log": len(delegation_log),
                "artifacts": len(artifacts),
            },
        }

    def _build_alerts(
        self,
        *,
        queued_total: int,
        blocked_approval_total: int,
        failed_total: int,
        stalled_tasks: list[dict[str, Any]],
        staleness_seconds: float | None,
    ) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        if blocked_approval_total > 0:
            alerts.append(
                {
                    "kind": "approval_blocked",
                    "severity": "warning",
                    "count": int(blocked_approval_total),
                    "summary": f"{blocked_approval_total} job(s) esperando aprobación",
                }
            )
        if failed_total > 0:
            alerts.append(
                {
                    "kind": "failed_jobs",
                    "severity": "critical",
                    "count": int(failed_total),
                    "summary": f"{failed_total} job(s) en estado failed",
                }
            )
        if len(stalled_tasks) > 0:
            alerts.append(
                {
                    "kind": "stalled_tasks",
                    "severity": "warning",
                    "count": int(len(stalled_tasks)),
                    "summary": f"{len(stalled_tasks)} job(s) stale en waiting_deps/blocked_approval",
                }
            )
        if queued_total >= 10:
            alerts.append(
                {
                    "kind": "queue_pressure",
                    "severity": "warning",
                    "count": int(queued_total),
                    "summary": f"Presión de cola: {queued_total} job(s) en queued",
                }
            )
        if staleness_seconds is not None and float(staleness_seconds) > 60.0:
            alerts.append(
                {
                    "kind": "snapshot_stale",
                    "severity": "warning",
                    "count": 1,
                    "summary": f"Datos con staleness de {round(float(staleness_seconds), 1)}s",
                }
            )
        return alerts

    def _build_risks(
        self,
        *,
        blocked_approval_total: int,
        stalled_tasks: list[dict[str, Any]],
        queued_total: int,
    ) -> list[dict[str, Any]]:
        risks: list[dict[str, Any]] = []
        if blocked_approval_total > 0:
            risks.append(
                {
                    "risk_id": "approval_dependency",
                    "level": "medium",
                    "source": "orchestrator",
                    "summary": "Dependencia de aprobación humana puede frenar entregas.",
                    "impact": "throughput",
                    "count": int(blocked_approval_total),
                }
            )
        if len(stalled_tasks) > 0:
            risks.append(
                {
                    "risk_id": "stale_dependencies",
                    "level": "high",
                    "source": "orchestrator",
                    "summary": "Jobs stale en dependencias incrementan lead time.",
                    "impact": "latency",
                    "count": int(len(stalled_tasks)),
                }
            )
        if queued_total >= 10:
            risks.append(
                {
                    "risk_id": "queue_backlog",
                    "level": "medium",
                    "source": "scheduler",
                    "summary": "Backlog alto puede degradar SLA.",
                    "impact": "sla",
                    "count": int(queued_total),
                }
            )
        return risks

    def _build_pending_decisions(
        self,
        *,
        orders: list[dict[str, Any]],
        blocked_requires_approval: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in blocked_requires_approval[:20]:
            out.append(
                {
                    "kind": "job_approval",
                    "priority": int(t.get("priority") or 2),
                    "state": str(t.get("state") or "blocked_approval"),
                    "order_id": str(t.get("parent_job_id") or "").strip() or None,
                    "job_id": str(t.get("job_id") or ""),
                    "job_id_short": str(t.get("job_id_short") or ""),
                    "title": str(t.get("title") or ""),
                    "next_action": "Aprobar o rechazar ejecución",
                    "updated_at": t.get("updated_at"),
                }
            )

        seen: set[tuple[str, str]] = set()
        for o in orders:
            oid = str(o.get("order_id") or "").strip()
            if not oid:
                continue
            try:
                rows = self.orch_q.list_decision_log(order_id=oid, limit=20)
            except Exception:
                rows = []
            for row in rows:
                action = str((row or {}).get("next_action") or "").strip()
                if not action:
                    continue
                key = (oid, action.lower())
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "kind": "order_decision",
                        "order_id": oid,
                        "job_id": (row or {}).get("job_id"),
                        "state": str((row or {}).get("state") or ""),
                        "summary": str((row or {}).get("summary") or ""),
                        "next_action": action,
                        "updated_at": (row or {}).get("ts"),
                    }
                )
                if len(out) >= 50:
                    return out
        return out

    def snapshot(self, *, chat_id: int | None = None) -> dict[str, Any]:
        """
        Compute a snapshot of worker/task status.

        Cache policy:
        - Reads a persisted cache entry if present and unexpired.
        - Writes the computed snapshot back to persisted cache.
        """
        key = "status_snapshot:" + (str(int(chat_id)) if chat_id is not None else "all")
        cached = None
        try:
            cached = self.orch_q.get_status_cache(key)
        except Exception:
            cached = None
        if isinstance(cached, dict):
            return cached

        now = time.time()
        max_parallel = _max_parallel_by_role(self.role_profiles)
        roles = sorted(max_parallel.keys())

        # Orders/autopilot: initial source of truth for "what the system is trying to do".
        if chat_id is None:
            orders = self.orch_q.list_orders_global(status="active", limit=50)
        else:
            orders = self.orch_q.list_orders(chat_id=int(chat_id), status="active", limit=50)

        orders_out: list[dict[str, Any]] = []
        workflows_out: list[dict[str, Any]] = []
        for o in orders:
            oid = str(o.get("order_id") or "").strip()
            if not oid:
                continue
            root_task = self.orch_q.get_job(oid)
            children = self.orch_q.jobs_by_parent(parent_job_id=oid, limit=200)
            counts: dict[str, int] = {}
            for c in children:
                counts[c.state] = counts.get(c.state, 0) + 1
            workflow = _build_order_workflow(order_row=o, root_task=root_task, children=children)
            orders_out.append(
                {
                    "order_id": oid,
                    "order_id_short": oid[:8],
                    "chat_id": int(o.get("chat_id") or 0),
                    "status": str(o.get("status") or ""),
                    "phase": str(o.get("phase") or "planning"),
                    "intent_type": str(o.get("intent_type") or "order_project_new"),
                    "project_id": (str(o.get("project_id") or "").strip() or None),
                    "source_message_id": o.get("source_message_id"),
                    "reply_to_message_id": o.get("reply_to_message_id"),
                    "priority": int(o.get("priority") or 2),
                    "title": str(o.get("title") or ""),
                    "updated_at": float(o.get("updated_at") or 0.0),
                    "children_counts": counts,
                    "workflow": workflow,
                }
            )
            workflows_out.append(workflow)

        workers_out: list[dict[str, Any]] = []
        # Scope counts: when chat_id is provided, only count jobs for that chat.
        counts_ok = True
        try:
            role_health = self.orch_q.get_role_health(chat_id=chat_id)
        except Exception:
            role_health = {}
            counts_ok = False
        try:
            queued_total = int(self.orch_q.get_queued_count(chat_id=chat_id))
            waiting_deps_total = int(self.orch_q.get_waiting_deps_count(chat_id=chat_id))
            blocked_approval_total = int(self.orch_q.get_blocked_approval_count(chat_id=chat_id))
            running_total = int(self.orch_q.get_running_count(chat_id=chat_id))
        except Exception:
            queued_total = 0
            waiting_deps_total = 0
            blocked_approval_total = 0
            running_total = 0
            counts_ok = False
        blocked_total = 0
        try:
            for rec in (role_health or {}).values():
                blocked_total += int((rec or {}).get("blocked", 0) or 0)
        except Exception:
            blocked_total = 0
            counts_ok = False

        blocked_requires_approval: list[dict[str, Any]] = []
        try:
            blocked = self.orch_q.peek(state="blocked_approval", limit=200, chat_id=chat_id)
            blocked_sorted = sorted(blocked, key=lambda t: float(t.updated_at), reverse=True)
            for t in blocked_sorted[:30]:
                blocked_requires_approval.append(_task_to_status(t))
                if len(blocked_requires_approval) >= 12:
                    break
        except Exception:
            blocked_requires_approval = []

        try:
            stalled_tasks = self.orch_q.list_stalled_tasks(stale_after_seconds=1800.0, limit=40)
        except Exception:
            stalled_tasks = []
        stalled_out = [_task_to_status(t) for t in stalled_tasks[:20]]

        try:
            projects = self.orch_q.list_projects(status=None, limit=400)
        except Exception:
            projects = []

        role_queue_rows: list[dict[str, Any]] = []
        failed_total = 0
        for role in sorted((role_health or {}).keys()):
            rec = role_health.get(role) or {}
            failed_total += int(rec.get("failed", 0) or 0)
            role_queue_rows.append(
                {
                    "role": role,
                    "queued": int(rec.get("queued", 0) or 0),
                    "waiting_deps": int(rec.get("waiting_deps", 0) or 0),
                    "blocked_approval": int(rec.get("blocked_approval", 0) or 0),
                    "running": int(rec.get("running", 0) or 0),
                    "blocked": int(rec.get("blocked", 0) or 0),
                    "done": int(rec.get("done", 0) or 0),
                    "failed": int(rec.get("failed", 0) or 0),
                }
            )
        newest_updated_at = 0.0
        no_open_jobs = (
            len(orders_out) == 0
            and queued_total == 0
            and running_total == 0
            and waiting_deps_total == 0
            and blocked_approval_total == 0
            and blocked_total == 0
        )
        if counts_ok and not no_open_jobs:
            for role in roles:
                n = int(max_parallel.get(role) or 1)
                running = self.orch_q.list_role_tasks_for_status(role=role, state="running", limit=200, chat_id=chat_id)
                queued = self.orch_q.list_role_tasks_for_status(role=role, state="queued", limit=200, chat_id=chat_id)

                # Sort queued by the scheduler's intended order (priority asc, created_at asc).
                queued_sorted = sorted(queued, key=lambda t: (int(t.priority or 2), float(t.created_at)))

                workers, occupied = _assign_running_to_workers(role, running, n)

                # Fill next tasks into idle workers.
                q_idx = 0
                for i in range(1, n + 1):
                    if workers[i - 1]["current"] is not None:
                        continue
                    if q_idx < len(queued_sorted):
                        workers[i - 1]["next"] = _task_to_status(queued_sorted[q_idx])
                        q_idx += 1

                for t in running + queued:
                    try:
                        newest_updated_at = max(newest_updated_at, float(t.updated_at))
                    except Exception:
                        pass

                workers_out.extend(workers)

        staleness_seconds = max(0.0, float(now - newest_updated_at)) if newest_updated_at > 0 else None
        alerts = self._build_alerts(
            queued_total=int(queued_total),
            blocked_approval_total=int(blocked_approval_total),
            failed_total=int(failed_total),
            stalled_tasks=stalled_out,
            staleness_seconds=staleness_seconds,
        )
        risks = self._build_risks(
            blocked_approval_total=int(blocked_approval_total),
            stalled_tasks=stalled_out,
            queued_total=int(queued_total),
        )
        decisions_pending = self._build_pending_decisions(
            orders=orders_out,
            blocked_requires_approval=blocked_requires_approval,
        )
        live_view = {
            "transport": "sse",
            "event": "snapshot",
            "target_latency_seconds": max(1, min(5, int(self.cache_ttl_seconds) + 2)),
            "staleness_seconds": staleness_seconds,
        }

        factory_extra: dict[str, Any] = {}
        if self.factory_snapshot_fn is not None:
            try:
                maybe_extra = self.factory_snapshot_fn(chat_id)
                if isinstance(maybe_extra, dict):
                    factory_extra = maybe_extra
            except Exception:
                factory_extra = {}

        payload: dict[str, Any] = {
            "api_version": "v1",
            "schema_version": 2,
            "generated_at": float(now),
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "orders_active": orders_out,
            "order_workflows": workflows_out,
            "autonomy_board": self.autonomy_board(chat_id=chat_id, limit=50),
            "projects": projects,
            "workers": workers_out,
            "queued_total": int(queued_total),
            "waiting_deps_total": int(waiting_deps_total),
            "blocked_approval_total": int(blocked_approval_total),
            "running_total": int(running_total),
            "blocked_total": int(blocked_total),
            "queue_by_role": role_queue_rows,
            "blocked_requires_approval": blocked_requires_approval,
            "stalled_tasks": stalled_out,
            "alerts": alerts,
            "risks": risks,
            "decisions_pending": decisions_pending,
            "live_view": live_view,
            "source_newest_updated_at": (float(newest_updated_at) if newest_updated_at > 0 else None),
            "staleness_seconds": staleness_seconds,
        }
        for key in ("factory", "repos", "heartbeats", "lanes", "models", "slo", "mailbox"):
            if key in factory_extra:
                payload[key] = factory_extra.get(key)
        extra_alerts = factory_extra.get("alerts") if isinstance(factory_extra.get("alerts"), list) else []
        if extra_alerts:
            payload["alerts"] = [*(payload.get("alerts") or []), *extra_alerts]
        extra_risks = factory_extra.get("risks") if isinstance(factory_extra.get("risks"), list) else []
        if extra_risks:
            payload["risks"] = [*(payload.get("risks") or []), *extra_risks]

        # Add a stable hash for SSE clients.
        try:
            canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            payload["snapshot_hash"] = hashlib.sha256(canon.encode("utf-8")).hexdigest()
        except Exception:
            payload["snapshot_hash"] = ""

        try:
            self.orch_q.set_status_cache(key, payload=payload, ttl_seconds=int(self.cache_ttl_seconds))
        except Exception:
            pass
        return payload
