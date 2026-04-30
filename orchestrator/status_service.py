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
_WORKFLOW_STAGE_ORDER = ("skynet_plan", "delivery", "validation", "skynet_review", "deploy")
_WORKFLOW_STAGE_LABELS = {
    "skynet_plan": "Skynet plan",
    "delivery": "Delivery",
    "validation": "Validation",
    "skynet_review": "Skynet review",
    "deploy": "Deploy",
}
_DEFAULT_WORKFLOW_SLA_TIER = "P1"
_WORKFLOW_STAGE_SLA_SECONDS_BY_TIER = {
    "P1": {
        "skynet_plan": 15 * 60,
        "delivery": 60 * 60,
        "validation": 30 * 60,
        "skynet_review": 30 * 60,
        "deploy": 30 * 60,
    },
}


def _normalize_sla_tier(sla_tier: str | None) -> str:
    tier = str(sla_tier or "").strip().upper()
    aliases = {
        "": _DEFAULT_WORKFLOW_SLA_TIER,
        "1": "P1",
        "HIGH": "P1",
        "CRITICAL": "P1",
    }
    tier = aliases.get(tier, tier)
    return tier if tier in _WORKFLOW_STAGE_SLA_SECONDS_BY_TIER else _DEFAULT_WORKFLOW_SLA_TIER


def _derive_workflow_sla_tier(root_task: Task | None) -> tuple[str, str]:
    trace = (root_task.trace or {}) if root_task is not None else {}
    labels = (root_task.labels or {}) if root_task is not None else {}
    for source, value in (
        ("root_task.trace.workflow_sla_tier", trace.get("workflow_sla_tier")),
        ("root_task.trace.sla_tier", trace.get("sla_tier")),
        ("root_task.labels.sla_tier", labels.get("sla_tier")),
    ):
        if str(value or "").strip():
            return _normalize_sla_tier(str(value)), source
    return _DEFAULT_WORKFLOW_SLA_TIER, "default"


def _workflow_stage_sla_seconds(sla_tier: str, stage: str) -> int:
    tier = _normalize_sla_tier(sla_tier)
    return int((_WORKFLOW_STAGE_SLA_SECONDS_BY_TIER.get(tier) or {}).get(stage, 0) or 0)


_TERMINAL_TASK_STATES = {"done", "failed", "cancelled"}


def _workflow_stage_tasks(stage: str, root_task: Task | None, children: list[Task]) -> list[Task]:
    stage_name = str(stage or "").strip()
    if stage_name == "skynet_plan":
        return [root_task] if root_task is not None else []
    if stage_name == "delivery":
        return [t for t in children if str(t.role or "").strip().lower() in _DELIVERY_ROLES]
    if stage_name == "validation":
        return [t for t in children if str(t.role or "").strip().lower() in (_VALIDATION_ROLES - {"release_mgr"})]
    if stage_name == "skynet_review":
        return [t for t in children if str(t.role or "").strip().lower() in _CONTROLLER_ROLES]
    if stage_name == "deploy":
        return [t for t in children if str(t.role or "").strip().lower() == "release_mgr"]
    return []


def _workflow_stage_sla_view(
    *,
    sla_tier: str | None,
    tier_source: str,
    stage: str,
    root_task: Task | None,
    children: list[Task],
    order_row: dict[str, Any],
    now: float,
) -> dict[str, Any]:
    tier = _normalize_sla_tier(sla_tier)
    stage_name = str(stage or "").strip()
    sla_seconds = _workflow_stage_sla_seconds(tier, stage_name)
    stage_tasks = _workflow_stage_tasks(stage_name, root_task, children)

    started_at: float | None = None
    source = "none"

    non_terminal = [
        t
        for t in stage_tasks
        if str(t.state or "").strip().lower() not in _TERMINAL_TASK_STATES and _coerce_float(t.created_at) is not None
    ]
    if non_terminal:
        task = min(non_terminal, key=lambda t: float(t.created_at or 0.0))
        started_at = float(task.created_at or 0.0)
        source = "oldest_non_terminal_stage_task_created_at"
    else:
        created_stage_tasks = [t for t in stage_tasks if _coerce_float(t.created_at) is not None]
        if created_stage_tasks:
            task = max(created_stage_tasks, key=lambda t: float(t.created_at or 0.0))
            started_at = float(task.created_at or 0.0)
            source = "latest_stage_task_created_at"

    if started_at is None:
        order_created_at = _coerce_float((order_row or {}).get("created_at"))
        if order_created_at is not None:
            started_at = order_created_at
            source = "order_created_at"
        elif root_task is not None:
            root_created_at = _coerce_float(root_task.created_at)
            if root_created_at is not None:
                started_at = root_created_at
                source = "root_task_created_at"

    age_seconds: int | None = None
    deadline_at: float | None = None
    overdue_by_seconds = 0
    if started_at is not None:
        age_seconds = max(0, int(float(now) - float(started_at)))
        deadline_at = float(started_at) + float(sla_seconds)
        if sla_seconds > 0:
            overdue_by_seconds = max(0, age_seconds - int(sla_seconds))

    return {
        "sla_tier": tier,
        "sla_seconds": int(sla_seconds),
        "started_at": started_at,
        "age_seconds": age_seconds,
        "deadline_at": deadline_at,
        "overdue": overdue_by_seconds > 0,
        "overdue_by_seconds": overdue_by_seconds,
        "source": source,
        "tier_source": tier_source,
    }


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


def _compact_jsonable_dict(value: Any, *, max_items: int = 20) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, item in list(value.items())[:max_items]:
        if not isinstance(key, str):
            key = str(key)
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[key] = item
        elif isinstance(item, dict):
            out[key] = _compact_jsonable_dict(item, max_items=max_items)
        elif isinstance(item, (list, tuple)):
            out[key] = [
                _compact_jsonable_dict(x, max_items=max_items) if isinstance(x, dict) else x
                for x in list(item)[:max_items]
                if isinstance(x, (str, int, float, bool, dict)) or x is None
            ]
    return out


def _compact_jsonable_list(value: Any, *, max_items: int = 20) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[Any] = []
    for item in list(value)[:max_items]:
        if isinstance(item, dict):
            compact = _compact_jsonable_dict(item)
            if compact:
                out.append(compact)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            out.append(item)
    return out


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


_PROACTIVE_MARKERS = ("[proactive:", "proactive sprint")
_DECISION_RANK = {
    "release": 0,
    "unblock": 1,
    "advance": 2,
    "monitor": 3,
}
_STAGE_RANK = {stage: idx for idx, stage in enumerate(_WORKFLOW_STAGE_ORDER)}


def _is_proactive_order(order_row: dict[str, Any], root_task: Task | None) -> bool:
    trace = dict((root_task.trace or {}) if root_task is not None else {})
    if bool(trace.get("proactive_lane")):
        return True
    if any(str(key).startswith("proactive_") for key in trace.keys()):
        return True

    haystack = " ".join(
        [
            str((order_row or {}).get("title") or ""),
            str((order_row or {}).get("body") or ""),
            str(root_task.input_text if root_task is not None else ""),
        ]
    ).lower()
    return any(marker in haystack for marker in _PROACTIVE_MARKERS)


def _primary_blocker(order: dict[str, Any]) -> dict[str, Any] | None:
    blockers = list(order.get("blockers") or [])
    for blocker in blockers:
        if isinstance(blocker, dict):
            return blocker
    return None


def _priority_decision(order: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None]:
    readiness_state = str(order.get("readiness_state") or "").strip().lower()
    readiness_verdict = str(order.get("readiness_verdict") or "").strip().lower()
    stage = str(order.get("current_stage") or "skynet_plan").strip()
    merge_ready = bool(order.get("merge_ready"))
    merged_to_main = bool(order.get("merged_to_main"))
    blocker = _primary_blocker(order)

    if merged_to_main or readiness_state == "released":
        return "monitor", "Already merged or released; keep it behind unreleased proactive orders.", blocker
    if merge_ready or readiness_state == "ready" or readiness_verdict == "go":
        return "release", "Ready/go order should be released before spending attention on blocked or not-ready work.", blocker
    if blocker is not None or readiness_state == "blocked" or readiness_verdict == "no_go":
        blocker_summary = str((blocker or {}).get("summary") or "").strip()
        blocker_stage = str((blocker or {}).get("stage") or stage).strip()
        why = f"Blocked at {blocker_stage}; clearing this restores flow for an active proactive order."
        if blocker_summary:
            why = f"{why} Primary blocker: {blocker_summary}"
        return "unblock", why, blocker
    return "advance", f"Not ready yet; advance {stage} evidence before it can be released.", blocker


def _priority_next_action(order: dict[str, Any], decision: str, blocker: dict[str, Any] | None) -> str:
    existing = str(order.get("next_action") or "").strip()
    if decision == "release":
        return existing or "Release or merge the order branch."
    if decision == "monitor":
        return existing or "Monitor deploy status and post-release evidence."
    if blocker is not None:
        summary = str(blocker.get("summary") or "").strip()
        if summary:
            return summary
    return existing or "Advance the proactive workflow evidence."


def _handoff_action_profile(
    *,
    release_readiness: dict[str, Any],
    workflow: dict[str, Any],
) -> dict[str, Any]:
    state = str(release_readiness.get("state") or "unknown").strip().lower()
    verdict = str(release_readiness.get("verdict") or "unknown").strip().lower()
    summary = str(release_readiness.get("summary") or "").strip()
    current_stage = str(workflow.get("current_stage") or release_readiness.get("current_stage") or "skynet_plan")
    blockers = [b for b in list(release_readiness.get("blockers") or workflow.get("blockers") or []) if isinstance(b, dict)]
    next_action = str(release_readiness.get("next_action") or "").strip()

    if state == "ready" or verdict == "go":
        decision = "ready_go"
        primary_action = next_action or "Release or merge the order branch."
        operator_actions = [
            "Confirm every handoff checklist item is pass.",
            "Keep the order evidence packet open while release_mgr performs the release.",
        ]
        release_manager_actions = [
            primary_action,
            "Record release evidence when the release completes.",
        ]
    elif state == "released":
        decision = "already_released"
        primary_action = next_action or "Monitor deploy status and post-release evidence."
        operator_actions = [
            primary_action,
            "Watch for deploy regressions or missing post-release evidence.",
        ]
        release_manager_actions = [
            "Confirm release evidence is attached to the order.",
        ]
    elif state == "blocked" or verdict == "no_go":
        decision = "blocked_no_go"
        primary_blocker = blockers[0] if blockers else {}
        blocker_summary = str(primary_blocker.get("summary") or "").strip()
        primary_action = blocker_summary or next_action or "Clear the release blocker before handoff."
        operator_actions = [
            primary_action,
            "Route the blocker to the owning role before release_mgr spends release capacity.",
        ]
        release_manager_actions = [
            "Do not release until blockers are cleared and readiness is recomputed.",
        ]
    elif state == "not_applicable" or verdict == "n/a":
        decision = "not_applicable"
        primary_action = next_action or "Use the standard order workflow fields."
        operator_actions = [
            primary_action,
            "Treat this as a non-proactive order handoff.",
        ]
        release_manager_actions = []
    else:
        decision = "not_ready_wait"
        primary_action = next_action or f"Advance {current_stage} evidence before release handoff."
        operator_actions = [
            primary_action,
            "Re-run this handoff after the pending readiness check changes state.",
        ]
        release_manager_actions = [
            "Wait for readiness to become go before releasing.",
        ]

    return {
        "state": state,
        "verdict": verdict,
        "decision": decision,
        "summary": summary,
        "primary_action": primary_action,
        "operator_actions": operator_actions,
        "release_manager_actions": release_manager_actions,
        "blockers": blockers,
    }


def _handoff_evidence_refs(packet: dict[str, Any], *, max_items: int = 12) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []

    def add(ref: dict[str, Any]) -> None:
        if not isinstance(ref, dict) or len(refs) >= max_items:
            return
        compact: dict[str, Any] = {}
        for key in (
            "kind",
            "key",
            "value",
            "role",
            "job_id",
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
            if value is not None and value != "":
                compact[key] = value
        if compact and compact not in refs:
            refs.append(compact)

    readiness = packet.get("release_readiness") if isinstance(packet.get("release_readiness"), dict) else {}
    for check in list((readiness or {}).get("checks") or []):
        if not isinstance(check, dict):
            continue
        for evidence in list(check.get("evidence") or []):
            if isinstance(evidence, dict):
                add({"check": check.get("key"), **evidence})

    for artifact in list(packet.get("artifacts") or []):
        if isinstance(artifact, dict):
            add({"kind": "artifact", **artifact})

    for decision in list(packet.get("decision_log") or []):
        if not isinstance(decision, dict):
            continue
        add(
            {
                "kind": "decision_log",
                "decision_kind": decision.get("kind"),
                "state": decision.get("state"),
                "job_id": decision.get("job_id"),
                "job_id_short": decision.get("job_id_short"),
                "summary": str(decision.get("summary") or "").strip()[:280],
            }
        )

    for trace in list(packet.get("traces") or []):
        if isinstance(trace, dict):
            add(
                {
                    "kind": "trace_event",
                    "trace_event_id": trace.get("id"),
                    "event_type": trace.get("event_type"),
                    "job_id": trace.get("job_id"),
                    "job_id_short": (str(trace.get("job_id"))[:8] if trace.get("job_id") else None),
                    "role": trace.get("agent_role"),
                    "artifact_id": trace.get("artifact_id"),
                }
            )

    return refs


@dataclass
class StatusService:
    orch_q: OrchestratorQueue
    role_profiles: dict[str, dict[str, Any]] | None = None
    cache_ttl_seconds: int = 2
    factory_snapshot_fn: Callable[[int | None], dict[str, Any]] | None = None
    proactive_health_fn: Callable[[], dict[str, Any]] | None = None

    def proactive_health(self) -> dict[str, Any]:
        generated_at = float(time.time())
        if self.proactive_health_fn is None:
            return {
                "api_version": "v1",
                "schema_version": 1,
                "generated_at": generated_at,
                "status": "not_configured",
                "operational_status": "not_configured",
                "trend_status": "not_configured",
                "alert_level": "OK",
                "alert_active": False,
                "summary_reason": "proactive_health_fn_not_configured",
                "report_found": False,
                "stale_report": False,
                "report_age_s": None,
                "report_path": None,
                "recommended_actions": [],
                "anomalies": [],
                "trend_flags": [],
                "autonomy_funnel": {},
                "orders": [],
                "factory": {},
            }

        try:
            raw = self.proactive_health_fn()
        except Exception:
            raw = None
        if not isinstance(raw, dict):
            return {
                "api_version": "v1",
                "schema_version": 1,
                "generated_at": generated_at,
                "status": "unavailable",
                "operational_status": "unavailable",
                "trend_status": "unavailable",
                "alert_level": "OK",
                "alert_active": False,
                "summary_reason": "proactive_health_fn_unavailable",
                "report_found": False,
                "stale_report": False,
                "report_age_s": None,
                "report_path": None,
                "recommended_actions": [],
                "anomalies": [],
                "trend_flags": [],
                "autonomy_funnel": {},
                "orders": [],
                "factory": {},
            }

        def _text(name: str, default: str) -> str:
            value = str(raw.get(name) or "").strip()
            return value if value else default

        def _bool(name: str, default: bool = False) -> bool:
            value = raw.get(name)
            if value is None:
                return bool(default)
            return bool(value)

        report_age_raw = raw.get("report_age_s")
        try:
            report_age_s = int(report_age_raw) if report_age_raw is not None else None
        except Exception:
            report_age_s = None

        alert_level = _text("alert_level", "OK").upper()
        if alert_level not in ("OK", "WARN", "CRITICAL"):
            alert_level = "OK"

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": generated_at,
            "status": _text("status", "unavailable"),
            "operational_status": _text("operational_status", "unavailable"),
            "trend_status": _text("trend_status", "unavailable"),
            "alert_level": alert_level,
            "alert_active": _bool("alert_active", alert_level in ("WARN", "CRITICAL")),
            "summary_reason": _text("summary_reason", "unavailable"),
            "report_found": _bool("report_found", False),
            "stale_report": _bool("stale_report", False),
            "report_age_s": report_age_s,
            "report_path": (_text("report_path", "") or None),
            "recommended_actions": _compact_jsonable_list(raw.get("recommended_actions"), max_items=20),
            "anomalies": _compact_jsonable_list(raw.get("anomalies"), max_items=20),
            "trend_flags": _compact_jsonable_list(raw.get("trend_flags"), max_items=20),
            "autonomy_funnel": _compact_jsonable_dict(raw.get("autonomy_funnel")),
            "orders": _compact_jsonable_list(raw.get("orders") or raw.get("order_reports"), max_items=20),
            "factory": _compact_jsonable_dict(raw.get("factory")),
        }

    def _proactive_health_alert(self, proactive_health: dict[str, Any]) -> dict[str, Any] | None:
        if not bool(proactive_health.get("alert_active")):
            return None
        level = str(proactive_health.get("alert_level") or "").strip().upper()
        severity = {"CRITICAL": "critical", "WARN": "warning"}.get(level, "info")
        summary = str(proactive_health.get("summary_reason") or "").strip() or "proactive health alert active"
        return {
            "kind": "proactive_health",
            "severity": severity,
            "summary": summary,
            "alert_level": level or "OK",
            "operational_status": proactive_health.get("operational_status"),
            "trend_status": proactive_health.get("trend_status"),
            "report_age_s": proactive_health.get("report_age_s"),
            "stale_report": bool(proactive_health.get("stale_report")),
            "report_path": proactive_health.get("report_path"),
        }

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
                    "workflow_stages": [
                        {
                            "stage": str(stage.get("stage") or ""),
                            "status": str(stage.get("status") or "pending"),
                            "summary": stage.get("summary"),
                        }
                        for stage in list(workflow.get("stages") or [])
                        if str(stage.get("stage") or "") in _WORKFLOW_STAGE_ORDER
                    ],
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

    def workflow_bottlenecks(self, chat_id: int | None = None, limit: int = 50) -> dict[str, Any]:
        lim = max(1, min(200, int(limit)))
        board = self.autonomy_board(chat_id=chat_id, limit=lim)
        generated_at = float(board.get("generated_at") or time.time())
        stage_index = {stage: idx for idx, stage in enumerate(_WORKFLOW_STAGE_ORDER)}
        stages: dict[str, dict[str, Any]] = {
            stage: {
                "stage": stage,
                "label": _WORKFLOW_STAGE_LABELS.get(stage, stage),
                "orders_count": 0,
                "blocked_count": 0,
                "failed_count": 0,
                "running_count": 0,
                "pending_count": 0,
                "oldest_updated_at": None,
                "recommended_next_action": "No active orders at this stage.",
                "orders": [],
            }
            for stage in _WORKFLOW_STAGE_ORDER
        }

        def _compact_blockers(order: dict[str, Any], stage: str) -> str:
            summaries: list[str] = []
            for blocker in list(order.get("blockers") or []):
                if not isinstance(blocker, dict):
                    continue
                blocker_stage = str(blocker.get("stage") or "").strip()
                normalized_stage = {
                    "controller_signoff": "skynet_review",
                    "release_evidence": "deploy",
                    "release": "deploy",
                }.get(blocker_stage, blocker_stage)
                if normalized_stage != stage:
                    continue
                summary = str(blocker.get("summary") or "").strip()
                if summary:
                    summaries.append(summary[:280])
            return "; ".join(summaries[:2])

        def _stage_status(order: dict[str, Any], stage: str) -> str:
            for item in list(order.get("workflow_stages") or []):
                if isinstance(item, dict) and str(item.get("stage") or "") == stage:
                    return str(item.get("status") or "pending").strip().lower() or "pending"
            if str(order.get("current_stage") or "") == stage:
                state = str(order.get("readiness_state") or "").strip().lower()
                if state == "blocked":
                    return "blocked"
                if state in {"not_ready", "not_applicable"}:
                    return "pending"
            return "pending"

        for order in list(board.get("orders") or []):
            if not isinstance(order, dict):
                continue
            current_stage = str(order.get("current_stage") or "skynet_plan").strip()
            if current_stage not in stages:
                current_stage = "skynet_plan"
            status = _stage_status(order, current_stage)
            stage = stages[current_stage]
            stage["orders_count"] = int(stage["orders_count"]) + 1
            if status == "failed":
                stage["failed_count"] = int(stage["failed_count"]) + 1
            elif status in {"blocked", "blocked_approval", "waiting_deps"}:
                stage["blocked_count"] = int(stage["blocked_count"]) + 1
            elif status == "running":
                stage["running_count"] = int(stage["running_count"]) + 1
            elif status == "pending":
                stage["pending_count"] = int(stage["pending_count"]) + 1

            updated_at = _coerce_float(order.get("updated_at"))
            oldest = _coerce_float(stage.get("oldest_updated_at"))
            if updated_at is not None and (oldest is None or updated_at < oldest):
                stage["oldest_updated_at"] = updated_at

            compact_order = {
                "order_id": str(order.get("order_id") or ""),
                "order_id_short": str(order.get("order_id_short") or str(order.get("order_id") or "")[:8]),
                "title": str(order.get("title") or ""),
                "priority": int(order.get("priority") or 2),
                "phase": str(order.get("phase") or "planning"),
                "current_stage": current_stage,
                "readiness_state": str(order.get("readiness_state") or "unknown"),
                "readiness_verdict": str(order.get("readiness_verdict") or "unknown"),
                "blocker_summary": _compact_blockers(order, current_stage),
                "next_action": str(order.get("next_action") or ""),
                "updated_at": updated_at,
            }
            orders = list(stage.get("orders") or [])
            orders.append(compact_order)
            orders.sort(
                key=lambda o: (
                    int(o.get("priority") or 2),
                    float(o.get("updated_at") or float("inf")),
                    str(o.get("order_id") or ""),
                )
            )
            stage["orders"] = orders[:lim]

        for stage_name, stage in stages.items():
            orders = list(stage.get("orders") or [])
            if orders:
                first_action = str(orders[0].get("next_action") or orders[0].get("blocker_summary") or "").strip()
                stage["recommended_next_action"] = first_action or f"Advance the oldest {stage_name} order."

        ordered_stages = [stages[stage] for stage in _WORKFLOW_STAGE_ORDER]

        def _rank_key(stage: dict[str, Any]) -> tuple[int, int, float, int]:
            name = str(stage.get("stage") or "")
            blocked_failed = int(stage.get("blocked_count") or 0) + int(stage.get("failed_count") or 0)
            oldest = _coerce_float(stage.get("oldest_updated_at"))
            return (
                -blocked_failed,
                -int(stage.get("orders_count") or 0),
                oldest if oldest is not None else float("inf"),
                stage_index.get(name, 999),
            )

        bottleneck = min(ordered_stages, key=_rank_key) if ordered_stages else None
        bottleneck_stage = str((bottleneck or {}).get("stage") or _WORKFLOW_STAGE_ORDER[0])
        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": generated_at,
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "limit": lim,
            "summary": {
                "orders_total": int((board.get("summary") or {}).get("orders_total") or 0),
                "bottleneck_stage": bottleneck_stage,
                "bottleneck_score": int((bottleneck or {}).get("blocked_count") or 0) + int((bottleneck or {}).get("failed_count") or 0),
                "recommended_next_action": str((bottleneck or {}).get("recommended_next_action") or ""),
            },
            "stages": ordered_stages,
        }

    def proactive_priorities(self, chat_id: int | None = None, limit: int = 20) -> dict[str, Any]:
        lim = max(1, min(200, int(limit)))
        generated_at = float(time.time())
        board = self.autonomy_board(chat_id=chat_id, limit=200)

        if chat_id is None:
            active_rows = self.orch_q.list_orders_global(status="active", limit=200)
        else:
            active_rows = self.orch_q.list_orders(chat_id=int(chat_id), status="active", limit=200)
        active_by_id = {str(row.get("order_id") or ""): row for row in active_rows if isinstance(row, dict)}

        ranked: list[dict[str, Any]] = []
        for order in list(board.get("orders") or []):
            if not isinstance(order, dict):
                continue
            oid = str(order.get("order_id") or "").strip()
            order_row = active_by_id.get(oid)
            if not oid or not isinstance(order_row, dict):
                continue
            try:
                root_task = self.orch_q.get_job(oid)
            except Exception:
                root_task = None
            if not _is_proactive_order(order_row, root_task):
                continue

            decision, why, blocker = _priority_decision(order)
            primary_blocker = None
            if blocker is not None:
                primary_blocker = {
                    "stage": str(blocker.get("stage") or ""),
                    "summary": str(blocker.get("summary") or "").strip(),
                    "job": blocker.get("job"),
                }

            priority = int(order.get("priority") or 2)
            current_stage = str(order.get("current_stage") or "skynet_plan")
            stage_rank = int(_STAGE_RANK.get(current_stage, -1))
            decision_rank = int(_DECISION_RANK.get(decision, 9))
            updated_at = _coerce_float(order.get("updated_at")) or 0.0

            ranked.append(
                {
                    "rank": 0,
                    "order_id": oid,
                    "order_id_short": str(order.get("order_id_short") or oid[:8]),
                    "title": str(order.get("title") or ""),
                    "priority": priority,
                    "phase": str(order.get("phase") or "planning"),
                    "current_stage": current_stage,
                    "readiness_state": str(order.get("readiness_state") or "unknown"),
                    "readiness_verdict": str(order.get("readiness_verdict") or "unknown"),
                    "decision": decision,
                    "why": why,
                    "primary_blocker": primary_blocker,
                    "next_action": _priority_next_action(order, decision, blocker),
                    "score_breakdown": {
                        "decision_rank": decision_rank,
                        "priority": priority,
                        "stage_rank": stage_rank,
                        "updated_at": updated_at,
                    },
                    "merge_ready": bool(order.get("merge_ready")),
                    "merged_to_main": bool(order.get("merged_to_main")),
                    "updated_at": updated_at,
                }
            )

        ranked.sort(
            key=lambda order: (
                int((order.get("score_breakdown") or {}).get("decision_rank") or 9),
                int(order.get("priority") or 2),
                -int((order.get("score_breakdown") or {}).get("stage_rank") or -1),
                float(order.get("updated_at") or 0.0),
                str(order.get("order_id") or ""),
            )
        )
        for idx, order in enumerate(ranked, start=1):
            order["rank"] = idx

        orders = ranked[:lim]
        top = None
        if orders:
            first = orders[0]
            top = {
                "order_id": first.get("order_id"),
                "order_id_short": first.get("order_id_short"),
                "title": first.get("title"),
                "decision": first.get("decision"),
                "why": first.get("why"),
                "next_action": first.get("next_action"),
            }

        by_decision: dict[str, int] = {}
        for order in ranked:
            decision = str(order.get("decision") or "unknown")
            by_decision[decision] = by_decision.get(decision, 0) + 1

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": generated_at,
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "limit": lim,
            "summary": {
                "active_proactive_orders": len(ranked),
                "returned": len(orders),
                "by_decision": by_decision,
                "top_decision": (str(top.get("decision")) if isinstance(top, dict) else None),
            },
            "top": top,
            "orders": orders,
        }

    def proactive_action_plan(self, chat_id: int | None = None, limit: int = 20) -> dict[str, Any]:
        priorities = self.proactive_priorities(chat_id=chat_id, limit=limit)
        generated_at = float(priorities.get("generated_at") or time.time())
        lim = int(priorities.get("limit") or max(1, min(200, int(limit))))
        lane_defs = [
            ("release", "Release"),
            ("unblock", "Unblock"),
            ("advance", "Advance"),
            ("monitor", "Monitor"),
        ]
        lanes_by_key: dict[str, list[dict[str, Any]]] = {key: [] for key, _label in lane_defs}

        compact_keys = [
            "rank",
            "order_id",
            "order_id_short",
            "title",
            "priority",
            "phase",
            "current_stage",
            "readiness_state",
            "readiness_verdict",
            "why",
            "primary_blocker",
            "next_action",
            "updated_at",
            "merge_ready",
            "merged_to_main",
        ]
        for order in list(priorities.get("orders") or []):
            if not isinstance(order, dict):
                continue
            lane = str(order.get("decision") or "").strip().lower()
            if lane not in lanes_by_key:
                lane = "advance"
            lanes_by_key[lane].append({key: order.get(key) for key in compact_keys})

        lanes: list[dict[str, Any]] = []
        for lane, label in lane_defs:
            orders = lanes_by_key[lane]
            recommended = str((orders[0].get("next_action") if orders else "") or "")
            lanes.append(
                {
                    "lane": lane,
                    "label": label,
                    "count": len(orders),
                    "recommended_next_action": recommended,
                    "orders": orders,
                }
            )

        top_order = next((order for lane in lanes for order in lane["orders"]), None)
        top_lane = None
        if isinstance(top_order, dict):
            top_rank = int(top_order.get("rank") or 0)
            for lane in lanes:
                if any(int((order or {}).get("rank") or 0) == top_rank for order in lane["orders"]):
                    top_lane = str(lane.get("lane") or "")
                    break

        lane_counts = {str(lane.get("lane") or ""): int(lane.get("count") or 0) for lane in lanes}
        returned = sum(lane_counts.values())
        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": generated_at,
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "limit": lim,
            "summary": {
                "active_proactive_orders": int((priorities.get("summary") or {}).get("active_proactive_orders") or 0),
                "returned": returned,
                "lanes": lane_counts,
                "top_lane": top_lane,
                "top_action": (str(top_order.get("next_action") or "") if isinstance(top_order, dict) else None),
            },
            "lanes": lanes,
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

    def order_release_readiness(
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
        tasks = [root_task, *children]
        artifacts: list[dict[str, Any]] = []
        for task in tasks:
            artifacts.extend(_artifact_refs_from_task(task))
        artifacts.extend(_artifact_refs_from_trace_events(traces))

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
            "workflow": workflow,
            "release_readiness": release_readiness,
            "counts": {
                "children_considered": len(children),
                "traces_considered": len(traces),
                "decision_log_considered": len(decision_log),
                "artifacts_considered": len(artifacts),
            },
        }

    def order_handoff_digest(
        self,
        order_id: str,
        trace_limit: int = 100,
        child_limit: int = 200,
        log_limit: int = 200,
    ) -> dict[str, Any] | None:
        packet = self.order_evidence_packet(
            order_id,
            trace_limit=trace_limit,
            child_limit=child_limit,
            log_limit=log_limit,
        )
        if not packet:
            return None

        order = packet.get("order") if isinstance(packet.get("order"), dict) else {}
        workflow = packet.get("workflow") if isinstance(packet.get("workflow"), dict) else {}
        readiness = packet.get("release_readiness") if isinstance(packet.get("release_readiness"), dict) else {}
        counts = packet.get("counts") if isinstance(packet.get("counts"), dict) else {}
        action_profile = _handoff_action_profile(release_readiness=readiness, workflow=workflow)

        checks: list[dict[str, Any]] = []
        check_counts: dict[str, int] = {}
        for check in list(readiness.get("checks") or []):
            if not isinstance(check, dict):
                continue
            status = str(check.get("status") or "unknown").strip().lower() or "unknown"
            check_counts[status] = check_counts.get(status, 0) + 1
            checks.append(
                {
                    "key": str(check.get("key") or ""),
                    "status": status,
                    "summary": str(check.get("summary") or "").strip()[:280] or None,
                    "evidence_count": len([item for item in list(check.get("evidence") or []) if isinstance(item, dict)]),
                }
            )

        recent_jobs: list[dict[str, Any]] = []
        jobs = [j for j in [packet.get("root"), *list(packet.get("children") or [])] if isinstance(j, dict)]
        for job in sorted(jobs, key=lambda j: float(j.get("updated_at") or 0.0), reverse=True)[:8]:
            recent_jobs.append(
                {
                    "job_id": job.get("job_id"),
                    "job_id_short": job.get("job_id_short"),
                    "role": job.get("role"),
                    "state": job.get("state"),
                    "title": job.get("title"),
                    "updated_at": job.get("updated_at"),
                    "result_summary": job.get("result_summary"),
                    "result_next_action": job.get("result_next_action"),
                }
            )

        recent_artifacts: list[dict[str, Any]] = []
        for artifact in list(packet.get("artifacts") or [])[:8]:
            if not isinstance(artifact, dict):
                continue
            recent_artifacts.append(
                {
                    "kind": artifact.get("kind") or "artifact",
                    "role": artifact.get("role"),
                    "job_id": artifact.get("job_id"),
                    "job_id_short": artifact.get("job_id_short"),
                    "path": artifact.get("path"),
                    "artifact_id": artifact.get("artifact_id"),
                }
            )

        evidence_refs = _handoff_evidence_refs(packet, max_items=12)
        state = str(action_profile.get("state") or readiness.get("state") or "unknown").strip().lower()
        verdict = str(action_profile.get("verdict") or readiness.get("verdict") or "unknown").strip().lower()
        current_stage = str(workflow.get("current_stage") or readiness.get("current_stage") or "skynet_plan")
        phase = str(workflow.get("phase") or readiness.get("phase") or order.get("phase") or "planning")

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": float(time.time()),
            "order_id": str(packet.get("order_id") or ""),
            "order_id_short": str(packet.get("order_id_short") or "")[:8],
            "phase": phase,
            "current_stage": current_stage,
            "state": state,
            "verdict": verdict,
            "summary": str(action_profile.get("summary") or readiness.get("summary") or "").strip(),
            "next_action": str(action_profile.get("primary_action") or readiness.get("next_action") or "").strip(),
            "blockers": list(action_profile.get("blockers") or readiness.get("blockers") or workflow.get("blockers") or []),
            "order": {
                "status": order.get("status"),
                "priority": order.get("priority"),
                "title": order.get("title"),
                "intent_type": order.get("intent_type"),
                "project_id": order.get("project_id"),
                "updated_at": order.get("updated_at"),
            },
            "readiness": {
                "applies": bool(readiness.get("applies")),
                "state": state,
                "verdict": verdict,
                "scope": readiness.get("scope"),
                "merge_ready": bool(workflow.get("merge_ready")),
                "merged_to_main": bool(workflow.get("merged_to_main")),
                "deploy_status": workflow.get("deploy_status"),
                "deploy_summary": workflow.get("deploy_summary"),
                "deployed_commit": workflow.get("deployed_commit"),
                "checks_total": len(checks),
                "checks_by_status": check_counts,
            },
            "checks": checks,
            "evidence_counts": {
                "children": int(counts.get("children") or 0),
                "traces": int(counts.get("traces") or 0),
                "decision_log": int(counts.get("decision_log") or 0),
                "delegation_log": int(counts.get("delegation_log") or 0),
                "artifacts": int(counts.get("artifacts") or 0),
                "handoff_refs": len(evidence_refs),
            },
            "recent_jobs": recent_jobs,
            "recent_artifacts": recent_artifacts,
            "evidence_refs": evidence_refs,
            "operator_actions": list(action_profile.get("operator_actions") or []),
            "release_manager_actions": list(action_profile.get("release_manager_actions") or []),
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

    def control_room(self, *, chat_id: int | None = None) -> dict[str, Any]:
        snap = self.snapshot(chat_id=chat_id)

        def _as_list(name: str) -> list[dict[str, Any]]:
            items = snap.get(name) or []
            return [x for x in items if isinstance(x, dict)]

        def _count(name: str) -> int:
            try:
                return int(snap.get(name) or 0)
            except Exception:
                return 0

        alerts = _as_list("alerts")
        risks = _as_list("risks")
        pending_decisions = _as_list("decisions_pending")
        blocked_approvals = _as_list("blocked_requires_approval")
        stalled_tasks = _as_list("stalled_tasks")
        order_workflows = _as_list("order_workflows")
        workers = _as_list("workers")

        critical_alerts = [a for a in alerts if str(a.get("severity") or "").strip().lower() == "critical"]
        warning_alerts = [a for a in alerts if str(a.get("severity") or "").strip().lower() == "warning"]

        if critical_alerts:
            health_level = "critical"
        elif blocked_approvals or stalled_tasks or risks or warning_alerts:
            health_level = "attention"
        else:
            health_level = "ok"

        reasons: list[str] = []
        if critical_alerts:
            reasons.append(f"{len(critical_alerts)} critical alert(s)")
        if blocked_approvals:
            reasons.append(f"{len(blocked_approvals)} approval(s) blocked")
        if stalled_tasks:
            reasons.append(f"{len(stalled_tasks)} stalled task(s)")
        if risks:
            reasons.append(f"{len(risks)} risk(s)")
        if warning_alerts and not critical_alerts:
            reasons.append(f"{len(warning_alerts)} warning alert(s)")
        health_summary = "OK: no operator attention required." if not reasons else "Attention: " + "; ".join(reasons[:4]) + "."

        running_workers = sum(1 for w in workers if isinstance(w.get("current"), dict))
        workers_with_next = sum(1 for w in workers if isinstance(w.get("next"), dict))
        total_workers = len(workers)
        idle_workers = max(0, total_workers - running_workers)
        saturation = round((running_workers / total_workers) if total_workers else 0.0, 3)

        role_rows: dict[str, dict[str, int | str]] = {}
        for w in workers:
            role = str(w.get("role") or "").strip().lower() or "unknown"
            row = role_rows.setdefault(role, {"role": role, "total": 0, "running": 0, "idle": 0, "with_next": 0})
            row["total"] = int(row["total"]) + 1
            if isinstance(w.get("current"), dict):
                row["running"] = int(row["running"]) + 1
            else:
                row["idle"] = int(row["idle"]) + 1
            if isinstance(w.get("next"), dict):
                row["with_next"] = int(row["with_next"]) + 1
        by_role = sorted(role_rows.values(), key=lambda r: str(r.get("role") or ""))[:10]

        queued_total = _count("queued_total")
        blocked_approval_total = _count("blocked_approval_total")
        recommended_actions: list[dict[str, Any]] = []

        def _add_action(action_id: str, label: str, target: str, count: int, reason: str) -> None:
            if count <= 0:
                return
            recommended_actions.append(
                {
                    "action_id": action_id,
                    "label": label,
                    "target": target,
                    "count": int(count),
                    "reason": reason,
                }
            )

        _add_action(
            "approve_blocked_jobs",
            "Approve or reject blocked jobs",
            "/api/v1/status/decisions",
            max(blocked_approval_total, len(blocked_approvals)),
            "Jobs are waiting for operator approval.",
        )
        _add_action(
            "resolve_pending_decisions",
            "Resolve pending decisions",
            "/api/v1/status/decisions",
            len(pending_decisions),
            "Pending decisions are blocking progress or confidence.",
        )
        _add_action(
            "inspect_stalled_tasks",
            "Inspect stalled tasks",
            "/api/v1/status/alerts",
            len(stalled_tasks),
            "Tasks appear stale and may need intervention.",
        )
        _add_action(
            "inspect_critical_alerts",
            "Inspect critical alerts",
            "/api/v1/status/alerts",
            len(critical_alerts),
            "Critical alerts require immediate review.",
        )
        queue_pressure = next((a for a in alerts if str(a.get("kind") or "") == "queue_pressure"), None)
        _add_action(
            "reduce_queue_pressure",
            "Reduce queue pressure",
            "/api/v1/orchestration/agents-live",
            int((queue_pressure or {}).get("count") or 0),
            "Queued work has reached the pressure threshold.",
        )
        low_saturation_with_work = queued_total if queued_total > 0 and total_workers > 0 and saturation < 0.5 else 0
        _add_action(
            "check_worker_availability",
            "Check worker availability",
            "/api/v1/orchestration/agents-live",
            low_saturation_with_work,
            "Queued work exists while worker saturation is low.",
        )

        bottleneck = self.workflow_bottlenecks(chat_id=chat_id, limit=50)
        bottleneck_summary = bottleneck.get("summary") if isinstance(bottleneck.get("summary"), dict) else {}
        workflow_bottleneck = {
            "stage": str(bottleneck_summary.get("bottleneck_stage") or ""),
            "score": int(bottleneck_summary.get("bottleneck_score") or 0),
            "recommended_next_action": str(bottleneck_summary.get("recommended_next_action") or ""),
            "orders_total": int(bottleneck_summary.get("orders_total") or 0),
        }

        action_plan = self.proactive_action_plan(chat_id=chat_id, limit=20)
        action_plan_summary = action_plan.get("summary") if isinstance(action_plan.get("summary"), dict) else {}
        lanes = action_plan_summary.get("lanes") if isinstance(action_plan_summary.get("lanes"), dict) else {}
        proactive_action_plan = {
            "active_proactive_orders": int(action_plan_summary.get("active_proactive_orders") or 0),
            "top_lane": action_plan_summary.get("top_lane"),
            "top_action": action_plan_summary.get("top_action"),
            "lanes": {str(k): int(v or 0) for k, v in lanes.items()},
        }

        top_action = str(proactive_action_plan.get("top_action") or "").strip()
        compact_recommended_actions = recommended_actions[:5]
        if top_action:
            compact_recommended_actions.append(
                {
                    "action_id": "follow_proactive_action_plan",
                    "label": "Follow proactive action plan",
                    "target": "/api/v1/orchestration/proactive-action-plan",
                    "count": 1,
                    "reason": f"Top proactive lane recommends: {top_action}",
                }
            )

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": snap.get("generated_at"),
            "chat_id": snap.get("chat_id"),
            "snapshot_hash": snap.get("snapshot_hash"),
            "health": {
                "level": health_level,
                "summary": health_summary,
            },
            "queue": {
                "queued_runnable": queued_total,
                "waiting_deps": _count("waiting_deps_total"),
                "blocked_approval": blocked_approval_total,
                "running": _count("running_total"),
                "blocked_legacy": _count("blocked_total"),
                "by_role": list(snap.get("queue_by_role") or [])[:10],
            },
            "orders": {
                "active_count": len(list(snap.get("orders_active") or [])),
                "workflows": order_workflows[:10],
            },
            "workers": {
                "total": total_workers,
                "running": running_workers,
                "idle": idle_workers,
                "with_next": workers_with_next,
                "saturation": saturation,
                "by_role": by_role,
            },
            "attention": {
                "pending_decisions": pending_decisions[:10],
                "blocked_approvals": blocked_approvals[:10],
                "alerts": alerts[:10],
                "risks": risks[:10],
                "stalled_tasks": stalled_tasks[:10],
            },
            "workflow_bottleneck": workflow_bottleneck,
            "proactive_action_plan": proactive_action_plan,
            "recommended_actions": compact_recommended_actions,
            "staleness_seconds": snap.get("staleness_seconds"),
        }

    def operator_focus(self, chat_id: int | None = None, limit: int = 5) -> dict[str, Any]:
        lim = max(1, min(20, int(limit)))
        generated_at = float(time.time())
        focus_chat_id = int(chat_id) if chat_id is not None else None

        category_order = {
            "critical_alert": 0,
            "approval": 1,
            "proactive_release": 2,
            "proactive_unblock": 3,
            "stalled": 4,
            "workflow_bottleneck": 5,
            "queue": 6,
            "proactive_advance": 7,
            "proactive_monitor": 8,
        }
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        source_counts = {"control_room": 0, "proactive_priorities": 0, "proactive_health": 0}

        def _text(value: Any, default: str = "") -> str:
            s = str(value or "").strip()
            return s if s else default

        def _num(value: Any, default: int = 0) -> int:
            try:
                return int(value or 0)
            except Exception:
                return int(default)

        def _float_or_none(value: Any) -> float | None:
            try:
                if value is None:
                    return None
                return float(value)
            except Exception:
                return None

        def _first_dict(items: Any) -> dict[str, Any]:
            for item in list(items or []):
                if isinstance(item, dict):
                    return item
            return {}

        def _score(category: str, urgency: str, updated_at: float | None) -> int:
            category_rank = int(category_order.get(category, 99))
            urgency_rank = int(urgency_order.get(urgency, 9))
            recency = 0
            if updated_at is not None and generated_at > updated_at:
                recency = max(0, min(99, int((generated_at - updated_at) // 60)))
            return max(0, 10000 - category_rank * 1000 - urgency_rank * 100 - recency)

        def _item(
            *,
            action_id: str,
            category: str,
            urgency: str,
            label: str,
            reason: str,
            next_action: str,
            target: str,
            count: int,
            order_id: str | None,
            job_id: str | None,
            source: str,
            source_signals: list[str],
            updated_at: float | None = None,
        ) -> dict[str, Any]:
            return {
                "rank": 0,
                "action_id": action_id,
                "category": category,
                "urgency": urgency,
                "label": label,
                "reason": reason,
                "next_action": next_action,
                "target": target,
                "count": int(max(1, count)),
                "order_id": order_id,
                "job_id": job_id,
                "source": source,
                "source_signals": source_signals,
                "score": _score(category, urgency, updated_at),
                "updated_at": updated_at,
            }

        items: list[dict[str, Any]] = []

        control = self.control_room(chat_id=focus_chat_id)
        control_health = control.get("health") if isinstance(control.get("health"), dict) else {}
        health_level = _text(control_health.get("level"), "ok")
        attention = control.get("attention") if isinstance(control.get("attention"), dict) else {}
        first_blocked = _first_dict(attention.get("blocked_approvals"))
        first_pending = _first_dict(attention.get("pending_decisions"))
        first_stalled = _first_dict(attention.get("stalled_tasks"))

        control_action_map = {
            "inspect_critical_alerts": ("critical_alert", "critical", first_blocked or first_pending),
            "approve_blocked_jobs": ("approval", "high", first_blocked),
            "resolve_pending_decisions": ("approval", "high", first_pending),
            "inspect_stalled_tasks": ("stalled", "high", first_stalled),
            "reduce_queue_pressure": ("queue", "medium", {}),
            "check_worker_availability": ("queue", "medium", {}),
        }
        for action in list(control.get("recommended_actions") or []):
            if not isinstance(action, dict):
                continue
            source_action_id = _text(action.get("action_id"))
            mapped = control_action_map.get(source_action_id)
            if mapped is None:
                continue
            category, urgency, ref = mapped
            updated_at = _float_or_none(ref.get("updated_at")) if isinstance(ref, dict) else None
            count = _num(action.get("count"), 1)
            items.append(
                _item(
                    action_id=source_action_id,
                    category=category,
                    urgency=urgency,
                    label=_text(action.get("label"), source_action_id.replace("_", " ").title()),
                    reason=_text(action.get("reason"), "Control room recommends operator attention."),
                    next_action=_text(action.get("label"), "Review operator action."),
                    target=_text(action.get("target"), "/api/v1/orchestration/control-room"),
                    count=count,
                    order_id=(_text(ref.get("order_id")) or None) if isinstance(ref, dict) else None,
                    job_id=(_text(ref.get("job_id")) or None) if isinstance(ref, dict) else None,
                    source="control_room",
                    source_signals=[source_action_id, category],
                    updated_at=updated_at,
                )
            )
            source_counts["control_room"] += 1

        workflow = control.get("workflow_bottleneck") if isinstance(control.get("workflow_bottleneck"), dict) else {}
        bottleneck_score = _num(workflow.get("score"), 0)
        if bottleneck_score > 0:
            stage = _text(workflow.get("stage"), "workflow")
            items.append(
                _item(
                    action_id=f"workflow_bottleneck:{stage}",
                    category="workflow_bottleneck",
                    urgency="medium",
                    label="Clear workflow bottleneck",
                    reason=f"{bottleneck_score} blocked or failed order(s) at {stage}.",
                    next_action=_text(workflow.get("recommended_next_action"), "Inspect the bottleneck stage."),
                    target="/api/v1/orchestration/workflow-bottlenecks",
                    count=bottleneck_score,
                    order_id=None,
                    job_id=None,
                    source="control_room",
                    source_signals=["workflow_bottleneck", stage],
                    updated_at=None,
                )
            )
            source_counts["control_room"] += 1

        priorities = self.proactive_priorities(chat_id=focus_chat_id, limit=20)
        proactive_category = {
            "release": ("proactive_release", "high", "Release proactive order"),
            "unblock": ("proactive_unblock", "high", "Unblock proactive order"),
            "advance": ("proactive_advance", "medium", "Advance proactive order"),
            "monitor": ("proactive_monitor", "low", "Monitor proactive order"),
        }
        for order in list(priorities.get("orders") or []):
            if not isinstance(order, dict):
                continue
            decision = _text(order.get("decision"), "advance")
            category, urgency, label = proactive_category.get(decision, proactive_category["advance"])
            oid = _text(order.get("order_id"))
            if not oid:
                continue
            updated_at = _float_or_none(order.get("updated_at"))
            source_signals = [
                f"decision:{decision}",
                f"stage:{_text(order.get('current_stage'), 'unknown')}",
                f"readiness:{_text(order.get('readiness_state'), 'unknown')}",
            ]
            items.append(
                _item(
                    action_id=f"{category}:{oid[:8]}",
                    category=category,
                    urgency=urgency,
                    label=label,
                    reason=_text(order.get("why"), "Proactive order needs operator focus."),
                    next_action=_text(order.get("next_action"), "Review the proactive order."),
                    target=f"/api/v1/orchestration/orders/release-readiness?order_id={oid}",
                    count=1,
                    order_id=oid,
                    job_id=None,
                    source="proactive_priorities",
                    source_signals=source_signals,
                    updated_at=updated_at,
                )
            )
            source_counts["proactive_priorities"] += 1

        health = self.proactive_health()
        health_status = _text(health.get("status"), "not_configured")
        if health_status != "not_configured" and bool(health.get("alert_active")):
            alert_level = _text(health.get("alert_level"), "OK").upper()
            urgency = "critical" if alert_level == "CRITICAL" else "high"
            if alert_level == "CRITICAL":
                health_level = "critical"
            elif health_level == "ok":
                health_level = "attention"
            recommended = _first_dict(health.get("recommended_actions"))
            recommended_text = _text(recommended.get("summary") or recommended.get("action")) if recommended else ""
            items.append(
                _item(
                    action_id="proactive_health_alert",
                    category="critical_alert",
                    urgency=urgency,
                    label="Inspect proactive health",
                    reason=_text(health.get("summary_reason"), "Proactive health alert is active."),
                    next_action=recommended_text or "Review the proactive health report.",
                    target="/api/v1/status/proactive-health",
                    count=1,
                    order_id=None,
                    job_id=None,
                    source="proactive_health",
                    source_signals=[
                        f"alert_level:{alert_level}",
                        f"operational:{_text(health.get('operational_status'), 'unknown')}",
                        f"trend:{_text(health.get('trend_status'), 'unknown')}",
                    ],
                    updated_at=None,
                )
            )
            source_counts["proactive_health"] += 1

        items.sort(
            key=lambda item: (
                int(category_order.get(str(item.get("category") or ""), 99)),
                int(urgency_order.get(str(item.get("urgency") or ""), 9)),
                -int(item.get("score") or 0),
                float(item.get("updated_at") or 0.0),
                str(item.get("action_id") or ""),
            )
        )
        returned = items[:lim]
        for idx, item in enumerate(returned, start=1):
            item["rank"] = idx

        top = returned[0] if returned else {}
        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": generated_at,
            "chat_id": focus_chat_id,
            "limit": lim,
            "summary": {
                "returned": len(returned),
                "health_level": health_level if health_level in {"ok", "attention", "critical"} else "ok",
                "top_action_id": top.get("action_id") if top else None,
                "top_category": top.get("category") if top else None,
                "source_counts": source_counts,
            },
            "items": returned,
        }

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
