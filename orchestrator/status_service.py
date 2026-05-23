from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import time
import urllib.parse

from .queue import OrchestratorQueue
from .runbooks import load_runbooks
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
_OPERATOR_FOCUS_RECEIPT_EVENT_TYPE = "operator_focus_receipt"
_OPERATOR_FOCUS_RECEIPT_HISTORY_LIMIT = 5
_OPERATOR_FOCUS_RECEIPT_STATUSES = {"acknowledged", "in_progress", "completed"}
_OPERATOR_FOCUS_RECEIPT_STALE_AFTER_SECONDS = {
    "acknowledged": 30 * 60,
    "in_progress": 60 * 60,
}


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


def _compact_factory_focus(proactive_health: dict[str, Any], *, limit: int) -> dict[str, Any]:
    factory = proactive_health.get("factory") if isinstance(proactive_health.get("factory"), dict) else {}

    def _int_value(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    if bool(factory.get("hard_stop")):
        state = "hard_stop"
    elif bool(factory.get("soft_pause_active")):
        state = "soft_pause"
    else:
        state = "active"

    raw_targets = [item for item in list(factory.get("next_targets") or []) if isinstance(item, dict)]
    target_count = _int_value(factory.get("next_target_count"), len(raw_targets))
    target_limit = max(1, int(limit))
    compact_targets: list[dict[str, Any]] = []
    for index, target in enumerate(raw_targets[:target_limit], start=1):
        heartbeat = target.get("runtime_heartbeat") if isinstance(target.get("runtime_heartbeat"), dict) else {}
        compact_targets.append(
            {
                "rank": target.get("rank") if target.get("rank") is not None else index,
                "repo_id": target.get("repo_id"),
                "attention_reason": target.get("attention_reason"),
                "recommended_action": target.get("recommended_action"),
                "priority": target.get("priority"),
                "coverage_state": target.get("coverage_state"),
                "order_id": target.get("order_id"),
                "order_phase": target.get("order_phase"),
                "order_activity_age_s": target.get("order_activity_age_s"),
                "heartbeat_state": target.get("heartbeat_state") or heartbeat.get("state"),
                "heartbeat_age_s": target.get("heartbeat_age_s") or heartbeat.get("heartbeat_age_s"),
                "agent_key": target.get("agent_key") or heartbeat.get("agent_key"),
                "role": target.get("role") or heartbeat.get("role"),
            }
        )

    return {
        "state": state,
        "pause_reason": factory.get("pause_reason"),
        "soft_pause_until": factory.get("soft_pause_until"),
        "registered_repos": _int_value(factory.get("registered_repos")),
        "enabled_repos": _int_value(factory.get("enabled_repos")),
        "uncovered_enabled_repos": _int_value(factory.get("uncovered_enabled_repos")),
        "stale_heartbeats": _int_value(factory.get("stale_heartbeats")),
        "next_target_count": target_count,
        "next_targets_truncated": bool(factory.get("next_targets_truncated")) or target_count > len(compact_targets),
        "next_targets": compact_targets,
    }


def _artifact_refs_from_task(task: Task) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    base = {
        "job_id": task.job_id,
        "job_id_short": task.job_id[:8],
        "role": str(task.role or "").strip().lower(),
        "ts": float(task.updated_at or task.created_at or 0.0),
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
            "ts": event.get("ts"),
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


def _recent_evidence_timeline(
    *,
    tasks: list[Task],
    traces: list[dict[str, Any]],
    decision_log: list[dict[str, Any]],
    delegation_log: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    max_items: int = 12,
) -> list[dict[str, Any]]:
    task_ts_by_job = {t.job_id: float(t.updated_at or t.created_at or 0.0) for t in tasks if t is not None}
    task_role_by_job = {t.job_id: str(t.role or "").strip().lower() for t in tasks if t is not None}
    trace_ts_by_event = {str(e.get("id")): _coerce_float(e.get("ts")) for e in traces if isinstance(e, dict) and e.get("id")}
    entries: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(
        *,
        ts: Any,
        source: str,
        kind: str,
        role: Any = None,
        job_id: Any = None,
        job_id_short: Any = None,
        summary: Any = None,
        path: Any = None,
        artifact_id: Any = None,
    ) -> None:
        ts_f = _coerce_float(ts)
        if ts_f is None:
            ts_f = 0.0
        jid = str(job_id or "").strip()
        jid_short = str(job_id_short or "").strip() or (jid[:8] if jid else None)
        entry: dict[str, Any] = {
            "ts": float(ts_f),
            "source": str(source or "").strip().lower() or "evidence",
            "kind": str(kind or "").strip().lower() or "evidence",
        }
        role_s = str(role or "").strip().lower()
        if role_s:
            entry["role"] = role_s
        if jid_short:
            entry["job_id_short"] = jid_short
        summary_s = str(summary or "").replace("\r", " ").replace("\n", " ").strip()
        if summary_s:
            entry["summary"] = summary_s[:220]
        path_s = str(path or "").strip()
        if path_s:
            entry["path"] = path_s
        artifact_s = str(artifact_id or "").strip()
        if artifact_s:
            entry["artifact_id"] = artifact_s

        key = (
            entry.get("source"),
            entry.get("kind"),
            entry.get("role"),
            entry.get("job_id_short"),
            entry.get("summary"),
            entry.get("path"),
            entry.get("artifact_id"),
        )
        if key in seen:
            return
        seen.add(key)
        entries.append(entry)

    for task in tasks:
        if task is None:
            continue
        tr = task.trace or {}
        summary = str(tr.get("result_summary") or tr.get("result_next_action") or _task_title(task)).strip()
        add(
            ts=task.updated_at or task.created_at,
            source="job",
            kind=str(task.state or "job"),
            role=task.role,
            job_id=task.job_id,
            summary=summary,
        )

    for event in traces:
        if not isinstance(event, dict):
            continue
        add(
            ts=event.get("ts"),
            source="trace",
            kind=event.get("event_type") or "trace_event",
            role=event.get("agent_role"),
            job_id=event.get("job_id"),
            summary=event.get("message"),
            artifact_id=event.get("artifact_id"),
        )

    for item in decision_log:
        if not isinstance(item, dict):
            continue
        add(
            ts=item.get("ts"),
            source="decision_log",
            kind=item.get("kind") or "decision",
            role=task_role_by_job.get(str(item.get("job_id") or "")),
            job_id=item.get("job_id"),
            job_id_short=item.get("job_id_short"),
            summary=item.get("summary") or item.get("next_action"),
        )

    for item in delegation_log:
        if not isinstance(item, dict):
            continue
        summary_bits = [
            str(item.get("edge_type") or "delegated"),
            str(item.get("to_key") or "").strip(),
            f"to {str(item.get('to_job_id_short') or item.get('to_job_id') or '').strip()}",
        ]
        add(
            ts=item.get("ts"),
            source="delegation_log",
            kind=item.get("edge_type") or "delegated",
            role=item.get("to_role"),
            job_id=item.get("to_job_id"),
            job_id_short=item.get("to_job_id_short"),
            summary=" ".join(bit for bit in summary_bits if bit.strip()),
        )

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        job_id = str(artifact.get("job_id") or "").strip()
        trace_event_id = str(artifact.get("trace_event_id") or "").strip()
        ts = artifact.get("ts")
        if _coerce_float(ts) is None and trace_event_id:
            ts = trace_ts_by_event.get(trace_event_id)
        if _coerce_float(ts) is None and job_id:
            ts = task_ts_by_job.get(job_id)
        path = artifact.get("path")
        artifact_id = artifact.get("artifact_id")
        add(
            ts=ts,
            source="artifact",
            kind=artifact.get("kind") or "artifact",
            role=artifact.get("role") or task_role_by_job.get(job_id),
            job_id=job_id,
            job_id_short=artifact.get("job_id_short"),
            summary=path or artifact_id,
            path=path,
            artifact_id=artifact_id,
        )

    capped = max(1, min(100, int(max_items)))
    return sorted(entries, key=lambda item: float(item.get("ts") or 0.0), reverse=True)[:capped]


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


def _release_target_evidence(root_trace: dict[str, Any]) -> list[dict[str, Any]]:
    trace = dict(root_trace or {})
    evidence: list[dict[str, Any]] = []

    order_branch = str(trace.get("order_branch") or "").strip()
    if order_branch:
        evidence.append(_readiness_trace_evidence("order_branch", order_branch))

    merged_to_main = bool(trace.get("merged_to_main", False))
    deploy_status = str(trace.get("deploy_status") or "").strip().lower()
    if merged_to_main:
        evidence.append(_readiness_trace_evidence("merged_to_main", merged_to_main))
    if deploy_status in {"ok", "scheduled"}:
        evidence.append(_readiness_trace_evidence("deploy_status", deploy_status))
        deployed_commit = str(trace.get("deployed_commit") or "").strip()
        if deployed_commit:
            evidence.append(_readiness_trace_evidence("deployed_commit", deployed_commit))

    publication = trace.get("github_publication") if isinstance(trace.get("github_publication"), dict) else {}
    publication_ok = bool(publication.get("ok", False))
    publication_target = (
        str(publication.get("github_repo") or "").strip()
        or str(publication.get("remote_url") or "").strip()
        or str(publication.get("project_path") or "").strip()
    )
    if publication_ok and publication_target:
        evidence.append({"kind": "trace", "key": "github_publication", "value": publication_target})

    delivery = trace.get("project_incubator_delivery") if isinstance(trace.get("project_incubator_delivery"), dict) else {}
    delivery_ok = bool(delivery.get("ok", False))
    delivery_target = (
        str(delivery.get("github_remote_url") or "").strip()
        or str(delivery.get("project_path") or "").strip()
        or str(delivery.get("project_head") or "").strip()
    )
    if delivery_ok and delivery_target and bool(trace.get("project_incubator_external_delivery", False)):
        evidence.append({"kind": "trace", "key": "project_incubator_delivery", "value": delivery_target})

    studio_outcome = str(trace.get("studio_terminal_outcome") or "").strip().lower()
    studio_summary = str(trace.get("studio_terminal_outcome_summary") or trace.get("result_summary") or "").strip()
    studio_summary_l = studio_summary.lower()
    if studio_outcome == "published_project" and studio_summary and any(
        marker in studio_summary_l for marker in ("github.com", "/home/", "project=")
    ):
        evidence.append({"kind": "trace", "key": "studio_terminal_outcome", "value": studio_summary[:280]})

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        key = str(item.get("key") or "")
        value = str(item.get("value") or "")
        marker = (key, value)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(item)
        if len(out) >= 5:
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
    check_keys = [
        "delivery_applied",
        "validation_passed",
        "controller_signoff",
        "release_evidence",
        "release_target_evidence",
        "merge_ready",
    ]

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
    release_target_evidence = _release_target_evidence(root_trace)
    has_release_target_evidence = bool(release_target_evidence)

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
            "key": "release_target_evidence",
            "status": "pass" if has_release_target_evidence else "pending",
            "summary": (
                "Release target evidence is present."
                if has_release_target_evidence
                else "Missing concrete release target evidence: add order_branch, merged/deployed status, or published project/repo evidence."
            ),
            "evidence": release_target_evidence,
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
        summary = "Proactive order has delivery, validation, controller, release, target, and merge-ready evidence."
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
    "selection_review": 1.5,
    "advance": 2,
    "monitor": 3,
}
_STAGE_RANK = {stage: idx for idx, stage in enumerate(_WORKFLOW_STAGE_ORDER)}
_HANDOFF_STAGE_ROLE = {
    "delivery": "implementer_local",
    "implementation": "implementer_local",
    "validation": "reviewer_local",
    "qa": "reviewer_local",
    "review": "reviewer_local",
    "controller_signoff": "architect_local",
    "skynet_review": "architect_local",
    "deploy": "release_mgr",
    "release": "release_mgr",
    "release_evidence": "release_mgr",
}
_HANDOFF_ROLES = {"implementer_local", "reviewer_local", "release_mgr", "architect_local"}
RELEASE_READINESS_LANES = ("ready", "blocked", "not_ready", "released")
_RELEASE_READINESS_LANE_SET = set(RELEASE_READINESS_LANES)


def normalize_release_readiness_lanes(lanes: Any) -> list[str]:
    if lanes is None:
        return []
    raw_values: list[Any]
    if isinstance(lanes, str):
        raw_values = [lanes]
    else:
        try:
            raw_values = list(lanes)
        except TypeError:
            raw_values = [lanes]

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in str(raw or "").split(","):
            lane = part.strip().lower()
            if not lane or lane in seen:
                continue
            if lane not in _RELEASE_READINESS_LANE_SET:
                raise ValueError(f"invalid_release_readiness_lane:{lane}")
            normalized.append(lane)
            seen.add(lane)
    return normalized


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
    if readiness_state == "ready" or readiness_verdict == "go":
        return "release", "Ready/go order should be released before spending attention on blocked or not-ready work.", blocker
    if blocker is not None or readiness_state == "blocked" or readiness_verdict == "no_go":
        blocker_summary = str((blocker or {}).get("summary") or "").strip()
        blocker_stage = str((blocker or {}).get("stage") or stage).strip()
        why = f"Blocked at {blocker_stage}; clearing this restores flow for an active proactive order."
        if blocker_summary:
            why = f"{why} Primary blocker: {blocker_summary}"
        return "unblock", why, blocker
    if merge_ready:
        return "advance", f"Merge-ready signal exists, but readiness is waiting on {stage} evidence before release.", blocker
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


_SELECTION_TEXT_MARKERS = (
    "buyer",
    "commercial",
    "customer",
    "deploy",
    "factory",
    "github",
    "merge",
    "monetiz",
    "price",
    "publish",
    "release",
    "retention",
    "revenue",
    "saving",
    "selection",
    "ship",
    "user",
    "validat",
    "value",
)
_SELECTION_PLACEHOLDER_TEXT = {
    "n/a",
    "na",
    "none",
    "not assessed",
    "not available",
    "tbd",
    "todo",
    "unknown",
}


def _selection_one_line(value: Any, *, max_chars: int = 220) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def _selection_flatten_text(value: Any, *, limit: int = 16) -> list[str]:
    out: list[str] = []

    def walk(item: Any) -> None:
        if len(out) >= limit:
            return
        if isinstance(item, dict):
            for nested in item.values():
                walk(nested)
                if len(out) >= limit:
                    break
        elif isinstance(item, (list, tuple)):
            for nested in item:
                walk(nested)
                if len(out) >= limit:
                    break
        else:
            text = _selection_one_line(item)
            if text:
                out.append(text)

    walk(value)
    return out


def _has_substantive_selection_text(value: Any, *, require_marker: bool = False) -> bool:
    for text in _selection_flatten_text(value):
        normalized = " ".join(text.lower().split())
        if not normalized or normalized in _SELECTION_PLACEHOLDER_TEXT:
            continue
        if len(normalized) < 24:
            continue
        if require_marker and not any(marker in normalized for marker in _SELECTION_TEXT_MARKERS):
            continue
        return True
    return False


def _selection_evidence_sources(*, order_row: dict[str, Any] | None, root_task: Task | None) -> list[dict[str, Any]]:
    trace = dict((root_task.trace or {}) if root_task is not None else {})
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(key: str, value: Any, *, require_marker: bool = False) -> None:
        if key in seen:
            return
        if not _has_substantive_selection_text(value, require_marker=require_marker):
            return
        seen.add(key)
        sources.append({"kind": "trace", "key": key, "summary": _selection_one_line(value)})

    for key in ("proactive_improvement_verified", "proactive_no_change_validated"):
        if bool(trace.get(key)) and key not in seen:
            seen.add(key)
            sources.append({"kind": "trace", "key": key, "summary": "true"})

    commercial_evidence = trace.get("commercial_evidence")
    if isinstance(commercial_evidence, dict):
        add("commercial_evidence", commercial_evidence, require_marker=True)
    add("commercial_evidence_target", trace.get("commercial_evidence_target"))

    studio_decision_evidence = trace.get("studio_decision_evidence")
    if isinstance(studio_decision_evidence, dict):
        add("studio_decision_evidence", studio_decision_evidence)

    factory_value = trace.get("factory_value")
    if isinstance(factory_value, dict):
        score = _coerce_int(factory_value.get("score"))
        dimensions = factory_value.get("dimensions") if isinstance(factory_value.get("dimensions"), dict) else {}
        dimension_ok = any(
            bool(value.get("ok")) for value in dimensions.values() if isinstance(value, dict)
        )
        if score >= 50 or dimension_ok or _has_substantive_selection_text(factory_value, require_marker=True):
            add("factory_value", factory_value)

    terminal_outcome = str(trace.get("studio_terminal_outcome") or "").strip()
    terminal_summary = trace.get("studio_terminal_outcome_summary") or trace.get("outcome_summary") or trace.get("result_summary")
    if terminal_outcome and _has_substantive_selection_text(terminal_summary, require_marker=True):
        add("studio_terminal_outcome", f"{terminal_outcome}: {terminal_summary}", require_marker=True)

    add("outcome_summary", trace.get("outcome_summary"), require_marker=True)
    add("result_summary", trace.get("result_summary"), require_marker=True)

    body = str((order_row or {}).get("body") or "").strip()
    if _has_substantive_selection_text(body, require_marker=True):
        key = "order_body"
        if key not in seen:
            sources.append({"kind": "order", "key": key, "summary": _selection_one_line(body)})

    return sources[:6]


def _selection_review_action(order: dict[str, Any]) -> str:
    oid = str(order.get("order_id_short") or order.get("order_id") or "this order").strip()
    return (
        f"Selection review for order {oid}: make a kill/continue/replan decision before more implementation churn; "
        "record concrete business, factory-value, or studio-decision evidence if continuing."
    )


_FACTORY_DELTA_GUARD_COMMAND = (
    "python3 tools/backend_done_evidence_guard.py --artifacts-dir <dir> --require-factory-delta"
)


def _factory_delta_contract_from_trace(root_task: Task | None) -> dict[str, Any] | None:
    trace = dict((root_task.trace or {}) if root_task is not None else {})
    if str(trace.get("studio_selected_type") or "").strip() != "DEEP_IMPROVEMENT":
        return None

    studio_cycle_id = str(trace.get("studio_cycle_id") or "").strip()
    expected_delta = str(trace.get("expected_measurable_delta") or "").strip()
    contract = {
        "required_fields": ["capability_changed", "measurable_delta", "evidence"],
        "capability_changed": (
            "Name the internal factory capability changed by this Studio DEEP_IMPROVEMENT work."
        ),
        "measurable_delta": expected_delta
        or "State the measurable before/after or acceptance delta this work is expected to move.",
        "evidence": (
            "Return a structured factory_delta object and cite concrete validation evidence for the claimed delta."
        ),
        "definition": (
            "Final structured output must include factory_delta with capability_changed, measurable_delta, "
            f"and evidence; validate with `{_FACTORY_DELTA_GUARD_COMMAND}`."
        ),
        "evidence_required": [
            "structured factory_delta.capability_changed",
            "structured factory_delta.measurable_delta",
            "structured factory_delta.evidence",
            f"factory delta guard result from `{_FACTORY_DELTA_GUARD_COMMAND}` or equivalent validation",
        ],
        "suggested_validation": [
            _FACTORY_DELTA_GUARD_COMMAND,
        ],
    }
    if studio_cycle_id:
        contract["studio_cycle_id"] = studio_cycle_id
    if expected_delta:
        contract["expected_measurable_delta"] = expected_delta
    return contract


def _selection_churn_risk_metadata(
    order: dict[str, Any],
    *,
    root_task: Task | None = None,
) -> dict[str, Any]:
    trace = dict((root_task.trace or {}) if root_task is not None else {})
    started_count = _coerce_int(trace.get("proactive_slices_started"))
    applied_count = _coerce_int(trace.get("proactive_slices_applied"))
    validated_count = _coerce_int(trace.get("proactive_slices_validated"))
    closed_count = _coerce_int(trace.get("proactive_slices_closed"))
    validated_or_closed = max(validated_count, closed_count)

    children_total = _coerce_int(order.get("children_total"))
    children_by_role = order.get("children_by_role") if isinstance(order.get("children_by_role"), dict) else {}
    delivery_children = sum(
        _coerce_int(count)
        for role, count in children_by_role.items()
        if str(role or "").strip().lower() in _DELIVERY_ROLES
    )

    churn_flags: list[str] = []
    if started_count >= 3 and started_count >= validated_or_closed + 2:
        churn_flags.append("started_slices_exceed_validated_or_closed")
    if delivery_children >= 2 and validated_or_closed == 0:
        churn_flags.append("repeated_delivery_children_without_validation")
    elif children_total >= 3 and validated_or_closed == 0:
        churn_flags.append("multiple_child_jobs_without_validation")

    needs_review = bool(churn_flags and (started_count > 0 or applied_count > 0 or children_total > 0))
    return {
        "status": "needs_review" if needs_review else "ok",
        "flags": churn_flags,
        "summary": (
            "Implementation activity is outpacing validation; review selection before more delivery delegation."
            if needs_review
            else "No implementation churn risk detected."
        ),
        "counters": {
            "proactive_slices_started": started_count,
            "proactive_slices_applied": applied_count,
            "proactive_slices_validated": validated_count,
            "proactive_slices_closed": closed_count,
            "children_total": children_total,
            "delivery_children": delivery_children,
        },
    }


def _selection_quality_metadata(
    order: dict[str, Any],
    *,
    order_row: dict[str, Any] | None = None,
    root_task: Task | None = None,
) -> dict[str, Any]:
    decision = str(order.get("decision") or "").strip().lower()
    readiness_state = str(order.get("readiness_state") or "").strip().lower()
    readiness_verdict = str(order.get("readiness_verdict") or "").strip().lower()
    primary_blocker = order.get("primary_blocker") if isinstance(order.get("primary_blocker"), dict) else None
    merge_ready = bool(order.get("merge_ready"))
    merged_to_main = bool(order.get("merged_to_main"))

    flags: list[str] = []
    if decision != "advance":
        return {
            "status": "ok",
            "flags": flags,
            "summary": "Selection-quality gate only reviews advance/not-ready proactive work.",
            "recommended_owner_role": None,
        }
    if merge_ready or merged_to_main or readiness_state in {"ready", "released"} or readiness_verdict == "go":
        return {
            "status": "ok",
            "flags": ["release_ready_exempt"],
            "summary": "Release-ready or already merged work is exempt from selection review.",
            "recommended_owner_role": None,
        }
    if primary_blocker is not None or readiness_state == "blocked" or readiness_verdict == "no_go":
        return {
            "status": "ok",
            "flags": ["blocked_exempt"],
            "summary": "Blocked work should route through unblock ownership before selection review.",
            "recommended_owner_role": None,
        }
    if readiness_state not in {"not_ready", "unknown", ""} and readiness_verdict not in {"wait", "unknown", ""}:
        return {
            "status": "ok",
            "flags": flags,
            "summary": "Order is not in the advance/not-ready selection-review lane.",
            "recommended_owner_role": None,
        }

    churn_risk = _selection_churn_risk_metadata(order, root_task=root_task)
    churn_needs_review = str(churn_risk.get("status") or "").strip().lower() == "needs_review"
    evidence_sources = _selection_evidence_sources(order_row=order_row, root_task=root_task)
    if evidence_sources and not churn_needs_review:
        return {
            "status": "ok",
            "flags": ["selection_evidence_present"],
            "summary": "Commercial, factory-value, or studio-decision evidence is present.",
            "recommended_owner_role": None,
            "evidence_sources": evidence_sources,
            "churn_risk": churn_risk,
        }

    if churn_needs_review and evidence_sources:
        flags = ["implementation_churn_without_validation"]
        flags.extend(str(flag) for flag in list(churn_risk.get("flags") or []) if str(flag or "").strip())
        flags.append("selection_evidence_present")
    else:
        flags = ["weak_selection_evidence", "advance_without_commercial_factory_or_studio_evidence"]
        if churn_needs_review:
            flags.append("implementation_churn_without_validation")
            flags.extend(str(flag) for flag in list(churn_risk.get("flags") or []) if str(flag or "").strip())
    return {
        "status": "needs_review",
        "flags": flags,
        "summary": (
            "Implementation churn is outpacing validation; make a kill/continue/replan decision before more delivery delegation."
            if churn_needs_review
            else "Advance work has no substantive commercial, factory-value, or studio-decision evidence; review selection before more delivery churn."
        ),
        "recommended_owner_role": "architect_local",
        "evidence_sources": evidence_sources,
        "churn_risk": churn_risk,
    }


def _handoff_role_for_stage(stage: str) -> str:
    return _HANDOFF_STAGE_ROLE.get(str(stage or "").strip().lower(), "implementer_local")


def _proactive_handoff_metadata(order: dict[str, Any], decision: str) -> dict[str, Any]:
    oid = str(order.get("order_id") or "").strip()
    stage = str(order.get("current_stage") or "").strip().lower()
    handoff_path = f"/api/v1/orchestration/orders/handoff-digest?order_id={oid}" if oid else "/api/v1/orchestration/orders/handoff-digest"

    decision_key = str(decision or "").strip().lower()
    selection_quality = order.get("selection_quality") if isinstance(order.get("selection_quality"), dict) else {}
    selection_needs_review = (
        decision_key in {"advance", "selection_review"}
        and str(selection_quality.get("status") or "").strip().lower() == "needs_review"
    )
    primary_blocker = order.get("primary_blocker") if isinstance(order.get("primary_blocker"), dict) else None
    blocker_stage = str((primary_blocker or {}).get("stage") or stage).strip().lower()
    blocker_job = (primary_blocker or {}).get("job")
    blocker_role = str((blocker_job or {}).get("role") or "").strip().lower() if isinstance(blocker_job, dict) else ""

    if decision_key in {"release", "monitor"}:
        suggested_role = "release_mgr"
    elif decision_key == "unblock":
        suggested_role = blocker_role if blocker_role in _HANDOFF_ROLES else _handoff_role_for_stage(blocker_stage)
    elif selection_needs_review:
        suggested_role = "architect_local"
    else:
        suggested_role = _handoff_role_for_stage(stage)

    next_action = str(order.get("next_action") or "").strip() or "Advance the proactive workflow evidence."
    title = str(order.get("title") or "").strip() or "proactive order"

    if selection_needs_review:
        action = _selection_review_action(order)
        definition_of_done = [
            "Make an explicit kill/continue/replan decision for this proactive order.",
            "If continuing, record the business, factory-value, or studio-decision evidence that justifies implementation.",
            "If killing or pausing, update the order next action so delivery churn stops.",
        ]
        checklist = [
            "Open the handoff digest.",
            "Inspect existing order and root trace evidence for commercial value, factory value, and studio decision rationale.",
            "Do not implement the delivery slice until selection evidence is recorded or the order is killed/paused.",
        ]
        evidence_expectations = [
            "kill/continue decision",
            "business or factory-value evidence",
            "studio-decision rationale or selected-bet evidence",
        ]
    elif decision_key == "release":
        action = ""
        definition_of_done = [
            "Release or merge the order branch.",
            "Record release evidence on the order.",
            "Confirm the proactive order remains ready/go after release.",
        ]
        checklist = [
            "Open the handoff digest.",
            "Verify readiness checks are pass or explicitly accepted.",
            "Perform the release action and attach evidence.",
        ]
        evidence_expectations = [
            "release command or merge result",
            "release evidence artifact or trace event",
            "post-release status summary",
        ]
    elif decision_key == "monitor":
        action = ""
        definition_of_done = [
            "Confirm deployed or merged status is still healthy.",
            "Attach post-release evidence or a no-action note.",
        ]
        checklist = [
            "Open the handoff digest.",
            "Inspect deploy and merge evidence.",
            "Record follow-up only if a regression or missing evidence is found.",
        ]
        evidence_expectations = [
            "deploy status",
            "merge or release reference",
            "post-release verification note",
        ]
    elif decision_key == "unblock":
        action = ""
        definition_of_done = [
            next_action,
            "Recompute readiness after the blocker is cleared.",
        ]
        checklist = [
            "Open the handoff digest.",
            f"Inspect the blocker at {blocker_stage or stage or 'the current stage'}.",
            "Complete the owning specialist task and attach evidence.",
        ]
        evidence_expectations = [
            "blocker resolution summary",
            "specialist task result",
            "updated readiness or workflow status",
        ]
    else:
        action = ""
        definition_of_done = [
            next_action,
            "Attach evidence that advances the proactive workflow.",
        ]
        checklist = [
            "Open the handoff digest.",
            f"Work the {stage or 'current'} stage for this order.",
            "Record concrete evidence before handing back to the operator.",
        ]
        evidence_expectations = [
            "implementation or review result",
            "test or validation output when applicable",
            "updated order next action",
        ]

    payload = {
        "suggested_role": suggested_role,
        "suggested_endpoint": handoff_path,
        "inspect_path": handoff_path,
        "definition_of_done": definition_of_done,
        "checklist": checklist,
        "evidence_expectations": evidence_expectations,
        "title": title,
    }
    factory_delta_contract = order.get("factory_delta_contract")
    if isinstance(factory_delta_contract, dict):
        payload["factory_delta_contract"] = factory_delta_contract
    if action:
        payload["action"] = action
    return payload


def _proactive_packet_text(value: Any, default: str = "") -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").strip()
    return text or default


def _proactive_packet_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        source = list(value)
    elif value is None:
        source = []
    else:
        source = [value]
    return [_proactive_packet_text(item) for item in source if _proactive_packet_text(item)]


def _proactive_lane_execution_packet(order: dict[str, Any], lane: str, label: str) -> dict[str, Any]:
    handoff = order.get("handoff") if isinstance(order.get("handoff"), dict) else {}
    lane_key = _proactive_packet_text(lane, "advance").lower()
    lane_label = _proactive_packet_text(label, lane_key.title())
    selection_quality = order.get("selection_quality") if isinstance(order.get("selection_quality"), dict) else {}
    selection_needs_review = (
        lane_key in {"advance", "selection_review"}
        and str(selection_quality.get("status") or "").strip().lower() == "needs_review"
    )
    oid = _proactive_packet_text(order.get("order_id"))
    display_oid = _proactive_packet_text(order.get("order_id_short"), oid[:8] if oid else "unknown")
    stage = _proactive_packet_text(order.get("current_stage"), "skynet_plan")
    action = (
        _selection_review_action(order) if selection_needs_review else _proactive_packet_text(handoff.get("action"))
        or _proactive_packet_text(order.get("next_action"))
        or f"Advance the {lane_label.lower()} lane order."
    )

    owner_role = "architect_local" if selection_needs_review else _proactive_packet_text(handoff.get("suggested_role"))
    if not owner_role:
        if lane_key in {"release", "monitor"}:
            owner_role = "release_mgr"
        else:
            blocker = order.get("primary_blocker") if isinstance(order.get("primary_blocker"), dict) else {}
            blocker_job = blocker.get("job") if isinstance(blocker.get("job"), dict) else {}
            blocker_role = _proactive_packet_text(blocker_job.get("role")).lower()
            owner_role = blocker_role if blocker_role in _HANDOFF_ROLES else _handoff_role_for_stage(stage)

    fallback_handoff_endpoint = (
        f"/api/v1/orchestration/orders/handoff-digest?order_id={oid}"
        if oid
        else "/api/v1/orchestration/orders/handoff-digest"
    )
    handoff_endpoint = _proactive_packet_text(handoff.get("suggested_endpoint"), fallback_handoff_endpoint)
    inspect_endpoint = _proactive_packet_text(handoff.get("inspect_path"), handoff_endpoint)

    acceptance_criteria = _proactive_packet_list(handoff.get("acceptance_criteria") or handoff.get("checklist"))
    if not acceptance_criteria:
        acceptance_criteria = [
            f"Inspect {inspect_endpoint}.",
            action,
            f"Stay within proactive lane {lane_key} and order {display_oid}.",
        ]

    definition_of_done = _proactive_packet_list(handoff.get("definition_of_done"))
    if not definition_of_done:
        definition_of_done = [
            action,
            "Record the operator-visible outcome before handing back.",
        ]

    evidence_required = _proactive_packet_list(handoff.get("evidence_required") or handoff.get("evidence_expectations"))
    if not evidence_required:
        evidence_required = [
            "completion summary",
            "updated readiness or workflow status",
        ]

    suggested_validation = _proactive_packet_list(handoff.get("suggested_validation") or handoff.get("suggested_tests"))
    if not suggested_validation:
        suggested_validation = [
            "Re-read the inspect endpoint after the action.",
            "Verify acceptance criteria, definition of done, and evidence expectations are satisfied.",
        ]

    if selection_needs_review:
        acceptance_criteria = [
            f"Inspect {inspect_endpoint}.",
            "Make an explicit kill/continue/replan decision before delivery work continues.",
            "Record business, factory-value, or studio-decision evidence for any continue decision.",
        ]
        definition_of_done = [
            "The order has a kill/continue/replan decision.",
            "Continuing work has concrete selection evidence, or killed/paused work has an updated next action.",
        ]
        evidence_required = [
            "kill/continue decision",
            "business or factory-value evidence",
            "studio-decision rationale or selected-bet evidence",
        ]
        suggested_validation = [
            "Re-read the inspect endpoint after the selection decision.",
            "Verify implementation work has not continued without selection evidence.",
        ]

    factory_delta_contract = order.get("factory_delta_contract")
    if not isinstance(factory_delta_contract, dict):
        factory_delta_contract = handoff.get("factory_delta_contract") if isinstance(handoff.get("factory_delta_contract"), dict) else None
    if isinstance(factory_delta_contract, dict):
        for item in _proactive_packet_list(factory_delta_contract.get("evidence_required")):
            if item not in evidence_required:
                evidence_required.append(item)
        for item in _proactive_packet_list(factory_delta_contract.get("suggested_validation")):
            if item not in suggested_validation:
                suggested_validation.append(item)
        definition = _proactive_packet_text(factory_delta_contract.get("definition"))
        if definition and definition not in definition_of_done:
            definition_of_done.append(definition)

    assignment_prompt = "" if selection_needs_review else _proactive_packet_text(handoff.get("assignment_prompt"))
    if not assignment_prompt:
        prompt_lines = [
            f"ROLE: {owner_role}.",
            "",
            f"Task: {_proactive_packet_text(handoff.get('title'), _proactive_packet_text(order.get('title'), 'Proactive order'))}",
            f"Scope: Work only on proactive action-plan lane {lane_key} for order {display_oid}.",
            f"Inspect endpoint: {inspect_endpoint}",
            f"Handoff endpoint: {handoff_endpoint}",
            f"Action: {action}",
            "",
            "Acceptance criteria:",
            *(f"- {criterion}" for criterion in acceptance_criteria),
            "",
            "Definition of done:",
            *(f"- {criterion}" for criterion in definition_of_done),
            "",
            "Evidence required:",
            *(f"- {evidence}" for evidence in evidence_required),
            "",
            "Suggested validation:",
            *(f"- {validation}" for validation in suggested_validation),
        ]
        if isinstance(factory_delta_contract, dict):
            prompt_lines.extend(
                [
                    "",
                    "Factory delta contract:",
                    *(f"- {field}" for field in _proactive_packet_list(factory_delta_contract.get("required_fields"))),
                    f"- validation: {_FACTORY_DELTA_GUARD_COMMAND}",
                ]
            )
        assignment_prompt = "\n".join(prompt_lines)
    elif isinstance(factory_delta_contract, dict) and _FACTORY_DELTA_GUARD_COMMAND not in assignment_prompt:
        assignment_prompt = (
            assignment_prompt.rstrip()
            + "\n\nFactory delta contract:\n"
            + "\n".join(f"- {field}" for field in _proactive_packet_list(factory_delta_contract.get("required_fields")))
            + f"\n- validation: {_FACTORY_DELTA_GUARD_COMMAND}"
        )

    packet = {
        "owner_role": owner_role,
        "action": action,
        "inspect_endpoint": inspect_endpoint,
        "handoff_endpoint": handoff_endpoint,
        "acceptance_criteria": acceptance_criteria,
        "definition_of_done": definition_of_done,
        "evidence_required": evidence_required,
        "suggested_validation": suggested_validation,
        "assignment_prompt": assignment_prompt,
        "order_id": oid or None,
        "lane": lane_key,
    }
    if isinstance(factory_delta_contract, dict):
        packet["factory_delta_contract"] = factory_delta_contract
    return packet


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


def _order_handoff_contract(
    *,
    order_id: str,
    title: str,
    current_stage: str,
    action_profile: dict[str, Any],
) -> dict[str, Any]:
    oid = str(order_id or "").strip()
    display_oid = oid[:8] or "unknown"
    title_text = str(title or "").strip() or "order"
    stage = str(current_stage or "skynet_plan").strip() or "skynet_plan"
    decision = str(action_profile.get("decision") or "not_ready_wait").strip().lower()
    blockers = [b for b in list(action_profile.get("blockers") or []) if isinstance(b, dict)]
    primary_blocker = blockers[0] if blockers else {}
    blocker_stage = str(primary_blocker.get("stage") or stage).strip() or stage
    blocker_job = primary_blocker.get("job") if isinstance(primary_blocker.get("job"), dict) else {}
    blocker_role = str(blocker_job.get("role") or "").strip().lower()
    inspect_endpoint = f"/api/v1/orchestration/orders/handoff-digest?order_id={oid}" if oid else "/api/v1/orchestration/orders/handoff-digest"
    readiness_endpoint = f"/api/v1/orchestration/orders/release-readiness?order_id={oid}" if oid else "/api/v1/orchestration/orders/release-readiness"
    action = str(action_profile.get("primary_action") or "").strip() or "Advance the order handoff."

    if decision in {"ready_go", "already_released"}:
        owner_role = "release_mgr"
    elif decision == "blocked_no_go":
        owner_role = blocker_role or _handoff_role_for_stage(blocker_stage or stage)
    else:
        owner_role = _handoff_role_for_stage(stage)

    if decision == "ready_go":
        evidence_required = [
            "readiness remains go",
            "release command or merge result",
            "release evidence artifact or trace event",
        ]
        suggested_validation = [
            "Inspect the handoff digest checks before release.",
            "Verify the release-readiness endpoint still reports go.",
        ]
        definition_of_done = [
            "Order is released or merged.",
            "Release evidence is attached to the order.",
            "Post-release status is recorded.",
        ]
        assignment_prompt = (
            f"{owner_role}: release order {display_oid} ({title_text}) only while readiness is go. "
            f"Inspect {inspect_endpoint}, validate {readiness_endpoint}, perform the release or merge, "
            "and return release evidence plus post-release status."
        )
    elif decision == "already_released":
        evidence_required = [
            "release or merge reference",
            "deploy status",
            "post-release verification note",
        ]
        suggested_validation = [
            "Confirm release evidence is present.",
            "Check deploy status for regressions or missing post-release notes.",
        ]
        definition_of_done = [
            "Released state is confirmed.",
            "Any missing post-release evidence or regression is recorded as follow-up.",
        ]
        assignment_prompt = (
            f"{owner_role}: verify released order {display_oid} ({title_text}). "
            f"Inspect {inspect_endpoint}, confirm release evidence through {readiness_endpoint}, "
            "and return a post-release verification note or bounded follow-up."
        )
    elif decision == "blocked_no_go":
        action = action or "Clear the release blocker before handoff."
        evidence_required = [
            "blocker resolution summary",
            "owning specialist task result",
            "updated readiness status",
        ]
        suggested_validation = [
            "Re-run or inspect release readiness after the blocker is cleared.",
            "Confirm no_go changes before asking release_mgr to act.",
        ]
        definition_of_done = [
            "Primary blocker is resolved or clearly reclassified.",
            "Readiness is recomputed with fresh evidence.",
            "Release remains held until readiness is go.",
        ]
        assignment_prompt = (
            f"{owner_role}: clear the {blocker_stage} blocker for order {display_oid} ({title_text}). "
            f"Inspect {inspect_endpoint}, validate with {readiness_endpoint}, attach blocker evidence, "
            "and do not release unless readiness later becomes go."
        )
    elif decision == "not_applicable":
        owner_role = _handoff_role_for_stage(stage)
        evidence_required = [
            "standard workflow status",
            "next action note",
        ]
        suggested_validation = [
            "Use the standard order workflow fields.",
            "Confirm whether proactive release readiness applies before routing release work.",
        ]
        definition_of_done = [
            "Standard handoff fields identify the next owner and action.",
        ]
        assignment_prompt = (
            f"{owner_role}: inspect order {display_oid} ({title_text}) using {inspect_endpoint}; "
            "return the standard workflow status and next action."
        )
    else:
        evidence_required = [
            f"{stage} progress evidence",
            "updated readiness status",
            "next action note",
        ]
        suggested_validation = [
            "Inspect the handoff digest and current readiness checks.",
            "Confirm the release-readiness endpoint does not permit release yet.",
        ]
        definition_of_done = [
            f"{stage} evidence is advanced or a concrete blocker is recorded.",
            "Readiness is recomputed after the stage work.",
            "Release remains held until readiness is go.",
        ]
        assignment_prompt = (
            f"{owner_role}: advance {stage} readiness for order {display_oid} ({title_text}). "
            f"Inspect {inspect_endpoint}, validate current state with {readiness_endpoint}, attach evidence, "
            "and do not release until readiness becomes go."
        )

    return {
        "owner_role": owner_role,
        "decision": decision,
        "action": action,
        "inspect_endpoint": inspect_endpoint,
        "release_readiness_endpoint": readiness_endpoint,
        "evidence_required": evidence_required,
        "suggested_validation": suggested_validation,
        "definition_of_done": definition_of_done,
        "assignment_prompt": assignment_prompt,
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
    runbooks_path: Path | None = None

    def runbook_status(self, *, now: float | None = None, runbooks_path: Path | None = None) -> dict[str, Any]:
        generated_at = float(time.time() if now is None else now)
        path = runbooks_path or self.runbooks_path or (Path(__file__).with_name("runbooks.yaml"))
        path_exists = path.exists()
        runbooks = load_runbooks(path) if path_exists else []

        summary = {
            "total": 0,
            "enabled": 0,
            "disabled": 0,
            "due": 0,
            "overdue": 0,
            "never_run_enabled": 0,
        }
        items: list[dict[str, Any]] = []

        for rb in runbooks:
            summary["total"] += 1
            enabled = bool(rb.enabled)
            if enabled:
                summary["enabled"] += 1
            else:
                summary["disabled"] += 1

            last_raw = self.orch_q.get_runbook_last_run(runbook_id=rb.runbook_id)
            try:
                last_run = float(last_raw)
            except Exception:
                last_run = 0.0
            has_last_run = last_run > 0.0

            if not enabled:
                status = "disabled"
                last_run_at = float(last_run) if has_last_run else None
                next_run_at = None
                due = False
                overdue = False
                due_in_seconds = None
                overdue_by_seconds = None
            elif not has_last_run:
                status = "due"
                last_run_at = None
                next_run_at = generated_at
                due = True
                overdue = True
                due_in_seconds = 0
                overdue_by_seconds = 0
                summary["never_run_enabled"] += 1
            else:
                last_run_at = float(last_run)
                next_run_at = float(last_run + float(rb.interval_seconds))
                due = generated_at >= next_run_at
                overdue = bool(due)
                if due:
                    status = "due"
                    due_in_seconds = 0
                    overdue_by_seconds = max(0, int(generated_at - next_run_at))
                else:
                    status = "scheduled"
                    due_in_seconds = max(0, int(next_run_at - generated_at))
                    overdue_by_seconds = None

            if due:
                summary["due"] += 1
            if overdue:
                summary["overdue"] += 1

            items.append(
                {
                    "runbook_id": rb.runbook_id,
                    "role": rb.role,
                    "enabled": enabled,
                    "status": status,
                    "interval_seconds": int(rb.interval_seconds),
                    "last_run_at": last_run_at,
                    "next_run_at": next_run_at,
                    "due": bool(due),
                    "overdue": bool(overdue),
                    "due_in_seconds": due_in_seconds,
                    "overdue_by_seconds": overdue_by_seconds,
                    "mode_hint": rb.mode_hint,
                    "priority": int(rb.priority),
                }
            )

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": generated_at,
            "runbooks_path": str(path),
            "runbooks_path_exists": bool(path_exists),
            "summary": summary,
            "items": items,
        }

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
            sla_tier, sla_tier_source = _derive_workflow_sla_tier(root_task)
            stage_sla_by_name: dict[str, dict[str, Any]] = {}
            for stage in list(workflow.get("stages") or []):
                stage_name = str(stage.get("stage") or "")
                if stage_name not in _WORKFLOW_STAGE_ORDER:
                    continue
                stage_sla_by_name[stage_name] = _workflow_stage_sla_view(
                    sla_tier=sla_tier,
                    tier_source=sla_tier_source,
                    stage=stage_name,
                    root_task=root_task,
                    children=children,
                    order_row=order_row,
                    now=generated_at,
                )
            current_stage_sla = stage_sla_by_name.get(current_stage, {})
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
                    "sla_tier": sla_tier,
                    "sla_tier_source": sla_tier_source,
                    "current_stage_overdue": bool(current_stage_sla.get("overdue")),
                    "current_stage_overdue_by_seconds": int(current_stage_sla.get("overdue_by_seconds") or 0),
                    "current_stage_deadline_at": current_stage_sla.get("deadline_at"),
                    "workflow_stages": [
                        {
                            "stage": str(stage.get("stage") or ""),
                            "status": str(stage.get("status") or "pending"),
                            "summary": stage.get("summary"),
                            "sla_tier": (stage_sla_by_name.get(str(stage.get("stage") or "")) or {}).get("sla_tier"),
                            "sla_seconds": (stage_sla_by_name.get(str(stage.get("stage") or "")) or {}).get("sla_seconds"),
                            "started_at": (stage_sla_by_name.get(str(stage.get("stage") or "")) or {}).get("started_at"),
                            "deadline_at": (stage_sla_by_name.get(str(stage.get("stage") or "")) or {}).get("deadline_at"),
                            "elapsed_seconds": (stage_sla_by_name.get(str(stage.get("stage") or "")) or {}).get("age_seconds"),
                            "overdue": bool((stage_sla_by_name.get(str(stage.get("stage") or "")) or {}).get("overdue")),
                            "overdue_by_seconds": int((stage_sla_by_name.get(str(stage.get("stage") or "")) or {}).get("overdue_by_seconds") or 0),
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

    def release_readiness_board(
        self,
        *,
        chat_id: int | None = None,
        limit: int = 50,
        include_released: bool = False,
        lanes: Any = None,
    ) -> dict[str, Any]:
        lim = max(1, min(200, int(limit)))
        active_lanes = normalize_release_readiness_lanes(lanes)
        active_lane_set = set(active_lanes)
        show_released = bool(include_released or "released" in active_lane_set)
        generated_at = float(time.time())
        source_limit = max(lim, 2000)
        if chat_id is None:
            orders = self.orch_q.list_orders_global(status="active", limit=source_limit)
        else:
            orders = self.orch_q.list_orders(chat_id=int(chat_id), status="active", limit=source_limit)

        lane_order = {"ready": 0, "blocked": 1, "not_ready": 2, "released": 3}
        candidates: list[dict[str, Any]] = []
        for order_row in orders:
            oid = str((order_row or {}).get("order_id") or "").strip()
            if not oid:
                continue
            try:
                root_task = self.orch_q.get_job(oid)
            except Exception:
                root_task = None
            if not _is_proactive_order(order_row, root_task):
                continue

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

            readiness_state = str(readiness.get("state") or "unknown").strip().lower() or "unknown"
            if readiness_state == "ready":
                lane = "ready"
            elif readiness_state == "blocked":
                lane = "blocked"
            elif readiness_state == "released":
                lane = "released"
            else:
                lane = "not_ready"
            if lane == "released" and not show_released:
                continue
            if active_lane_set and lane not in active_lane_set:
                continue

            checks_by_status: dict[str, int] = {}
            for check in list(readiness.get("checks") or []):
                if not isinstance(check, dict):
                    continue
                status = str(check.get("status") or "unknown").strip().lower() or "unknown"
                checks_by_status[status] = checks_by_status.get(status, 0) + 1

            blockers = [b for b in list(readiness.get("blockers") or workflow.get("blockers") or []) if isinstance(b, dict)]
            blocker = blockers[0] if blockers else None
            primary_blocker: dict[str, Any] | None = None
            if blocker is not None:
                job = blocker.get("job") if isinstance(blocker.get("job"), dict) else {}
                primary_blocker = {
                    "stage": str(blocker.get("stage") or "").strip() or None,
                    "summary": str(blocker.get("summary") or "").strip()[:280] or None,
                    "job_id": job.get("job_id") if isinstance(job, dict) else None,
                    "job_id_short": job.get("job_id_short") if isinstance(job, dict) else None,
                    "role": job.get("role") if isinstance(job, dict) else None,
                    "state": job.get("state") if isinstance(job, dict) else None,
                }

            updated_at = float((order_row or {}).get("updated_at") or (root_task.updated_at if root_task is not None else 0.0) or 0.0)
            title = str((order_row or {}).get("title") or (_task_title(root_task) if root_task is not None else "")).strip()
            priority = int((order_row or {}).get("priority") or (root_task.priority if root_task is not None else 2) or 2)
            current_stage = str(workflow.get("current_stage") or readiness.get("current_stage") or "skynet_plan")
            candidates.append(
                {
                    "rank": 0,
                    "order_id": oid,
                    "order_id_short": oid[:8],
                    "chat_id": int((order_row or {}).get("chat_id") or (root_task.chat_id if root_task is not None else 0) or 0),
                    "title": title,
                    "priority": priority,
                    "phase": str((order_row or {}).get("phase") or readiness.get("phase") or workflow.get("phase") or "planning"),
                    "current_stage": current_stage,
                    "readiness_state": readiness_state,
                    "readiness_verdict": str(readiness.get("verdict") or "unknown").strip().lower() or "unknown",
                    "release_lane": lane,
                    "summary": str(readiness.get("summary") or "").strip(),
                    "next_action": str(readiness.get("next_action") or "").strip(),
                    "primary_blocker": primary_blocker,
                    "checks_by_status": checks_by_status,
                    "handoff_endpoint": f"/api/v1/orchestration/orders/handoff-digest?order_id={oid}",
                    "release_readiness_endpoint": f"/api/v1/orchestration/orders/release-readiness?order_id={oid}",
                    "merge_ready": bool(workflow.get("merge_ready")),
                    "merged_to_main": bool(workflow.get("merged_to_main")),
                    "updated_at": updated_at,
                }
            )

        candidates.sort(
            key=lambda item: (
                lane_order.get(str(item.get("release_lane") or ""), 99),
                int(item.get("priority") or 2),
                float(item.get("updated_at") or 0.0),
                str(item.get("order_id") or ""),
            )
        )
        returned = candidates[:lim]
        for idx, item in enumerate(returned, start=1):
            item["rank"] = idx

        lanes = {
            lane: {
                "lane": lane,
                "count": 0,
                "orders": [],
            }
            for lane in ("ready", "blocked", "not_ready", "released")
        }
        for item in returned:
            lane = str(item.get("release_lane") or "not_ready")
            lanes.setdefault(lane, {"lane": lane, "count": 0, "orders": []})
            lanes[lane]["orders"].append(item)
            lanes[lane]["count"] = len(lanes[lane]["orders"])

        by_lane = {lane: int(payload.get("count") or 0) for lane, payload in lanes.items()}
        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": generated_at,
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "limit": lim,
            "include_released": bool(include_released),
            "lane_filters": active_lanes,
            "summary": {
                "orders_total": len(returned),
                "returned": len(returned),
                "active_lanes": active_lanes,
                "by_lane": by_lane,
                "ready": by_lane.get("ready", 0),
                "blocked": by_lane.get("blocked", 0),
                "not_ready": by_lane.get("not_ready", 0),
                "released": by_lane.get("released", 0),
            },
            "lanes": lanes,
        }

    def queue_pressure_board(
        self,
        *,
        chat_id: int | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        lim = max(1, min(200, int(limit)))
        snap = self.snapshot(chat_id=chat_id)
        generated_at = float(snap.get("generated_at") or time.time())
        max_parallel = _max_parallel_by_role(self.role_profiles)

        role_rows: dict[str, dict[str, Any]] = {}
        for row in list(snap.get("queue_by_role") or []):
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "").strip().lower()
            if role:
                role_rows[role] = row

        worker_capacity: dict[str, int] = {}
        worker_running: dict[str, int] = {}
        for worker in list(snap.get("workers") or []):
            if not isinstance(worker, dict):
                continue
            role = str(worker.get("role") or "").strip().lower()
            if not role:
                continue
            worker_capacity[role] = worker_capacity.get(role, 0) + 1
            if isinstance(worker.get("current"), dict):
                worker_running[role] = worker_running.get(role, 0) + 1

        stalled_by_role: dict[str, int] = {}
        for item in list(snap.get("stalled_tasks") or []):
            if not isinstance(item, dict):
                continue
            if chat_id is not None:
                try:
                    if int(item.get("chat_id") or 0) != int(chat_id):
                        continue
                except Exception:
                    continue
            role = str(item.get("role") or "").strip().lower()
            if role:
                stalled_by_role[role] = stalled_by_role.get(role, 0) + 1

        pressure_rows: list[dict[str, Any]] = []
        roles = sorted(set(max_parallel) | set(role_rows) | set(worker_capacity))
        for role in roles:
            row = role_rows.get(role) or {}
            capacity = max(1, int(worker_capacity.get(role) or max_parallel.get(role) or 1))
            running = int(row.get("running", worker_running.get(role, 0)) or 0)
            queued = int(row.get("queued", 0) or 0)
            waiting = int(row.get("waiting_deps", 0) or 0)
            blocked_approval = int(row.get("blocked_approval", 0) or 0)
            blocked = int(row.get("blocked", 0) or 0)
            stalled = int(stalled_by_role.get(role, 0) or 0)
            idle = max(0, capacity - running)
            waiting_total = waiting + blocked_approval + blocked
            backlog_total = queued + waiting_total
            saturation = round(min(1.0, float(running) / float(capacity)), 3)

            if blocked_approval > 0 or stalled > 0 or (queued > 0 and idle == 0 and running >= capacity):
                level = "critical"
            elif backlog_total > 0 or saturation >= 0.75:
                level = "attention"
            else:
                level = "ok"

            if blocked_approval > 0:
                next_action = f"Review {blocked_approval} blocked approval job(s) for {role}."
            elif stalled > 0:
                next_action = f"Inspect {stalled} stalled job(s) for {role}."
            elif queued > 0 and idle == 0:
                next_action = f"Role {role} is saturated; consider freeing a worker or reprioritizing queued work."
            elif waiting > 0 or blocked > 0:
                next_action = f"Resolve dependencies for {waiting_total} waiting/blocked job(s) in {role}."
            elif queued > 0 and idle > 0:
                next_action = f"Dispatch queued {role} work into {idle} idle slot(s)."
            elif running > 0:
                next_action = f"Monitor {running} running {role} job(s)."
            else:
                next_action = f"No queue action needed for {role}."

            pressure_rows.append(
                {
                    "role": role,
                    "pressure_level": level,
                    "max_parallel": capacity,
                    "running": running,
                    "idle": idle,
                    "saturation": saturation,
                    "queued": queued,
                    "waiting_deps": waiting,
                    "blocked_approval": blocked_approval,
                    "blocked": blocked,
                    "waiting_total": waiting_total,
                    "backlog_total": backlog_total,
                    "stalled": stalled,
                    "next_action": next_action,
                }
            )

        level_rank = {"critical": 0, "attention": 1, "ok": 2}
        pressure_rows.sort(
            key=lambda item: (
                level_rank.get(str(item.get("pressure_level") or "ok"), 9),
                -int(item.get("backlog_total") or 0),
                -float(item.get("saturation") or 0.0),
                str(item.get("role") or ""),
            )
        )

        idle_slots_by_role: dict[str, list[int]] = {}
        for worker in list(snap.get("workers") or []):
            if not isinstance(worker, dict) or isinstance(worker.get("current"), dict):
                continue
            role = str(worker.get("role") or "").strip().lower()
            if not role:
                continue
            try:
                slot = int(worker.get("slot") or 0)
            except Exception:
                slot = 0
            if slot > 0:
                idle_slots_by_role.setdefault(role, []).append(slot)
        for slots in idle_slots_by_role.values():
            slots.sort()

        dispatch_candidates: list[tuple[int, float, str, str, int, Task]] = []
        for row in pressure_rows:
            role = str(row.get("role") or "").strip().lower()
            idle = int(row.get("idle") or 0)
            queued = int(row.get("queued") or 0)
            if not role or idle <= 0 or queued <= 0:
                continue
            slots = idle_slots_by_role.get(role) or list(range(1, idle + 1))
            try:
                tasks = self.orch_q.list_role_tasks_for_status(role=role, state="queued", limit=max(lim, idle), chat_id=chat_id)
            except Exception:
                tasks = []
            tasks = sorted(tasks, key=lambda t: (int(t.priority or 2), float(t.created_at), str(t.job_id)))
            for slot, task in zip(slots[:idle], tasks):
                dispatch_candidates.append((int(task.priority or 2), float(task.created_at), str(task.job_id), role, int(slot), task))

        dispatch_plan: list[dict[str, Any]] = []
        for plan_rank, (_, _, _, role, slot, task) in enumerate(sorted(dispatch_candidates, key=lambda item: item[:3])[:lim], start=1):
            item = _task_to_status(task)
            item["role"] = role
            item["slot"] = slot
            item["plan_rank"] = plan_rank
            item["next_action"] = f"Dispatch this queued {role} job into idle slot {slot}."
            dispatch_plan.append(item)

        queued_total = int(snap.get("queued_total") or 0)
        waiting_deps_total = int(snap.get("waiting_deps_total") or 0)
        blocked_approval_total = int(snap.get("blocked_approval_total") or 0)
        blocked_total = int(snap.get("blocked_total") or 0)
        running_total = int(snap.get("running_total") or 0)
        stalled_total = sum(int(row.get("stalled") or 0) for row in pressure_rows)
        total_capacity = sum(max(1, int(row.get("max_parallel") or 1)) for row in pressure_rows)
        idle_total = max(0, total_capacity - running_total)
        overall_saturation = round(min(1.0, float(running_total) / float(total_capacity)), 3) if total_capacity > 0 else 0.0

        if blocked_approval_total > 0 or stalled_total > 0 or any(row.get("pressure_level") == "critical" for row in pressure_rows):
            overall_level = "critical"
        elif queued_total > 0 or waiting_deps_total > 0 or blocked_total > 0 or overall_saturation >= 0.75:
            overall_level = "attention"
        else:
            overall_level = "ok"

        recommended_actions: list[dict[str, Any]] = []

        def _add_action(action_id: str, label: str, count: int, target: str, next_action: str, role: str | None = None) -> None:
            if count <= 0:
                return
            recommended_actions.append(
                {
                    "action_id": action_id,
                    "label": label,
                    "role": role,
                    "count": int(count),
                    "target": target,
                    "next_action": next_action,
                }
            )

        _add_action(
            "review_blocked_approvals",
            "Review blocked approvals",
            blocked_approval_total,
            "/api/v1/orchestration/control-room",
            "Approve, reject, or reroute jobs blocked on operator approval.",
        )
        _add_action(
            "inspect_stalled_jobs",
            "Inspect stalled jobs",
            stalled_total,
            "/api/v1/orchestration/agents-live",
            "Inspect stalled waiting or approval-blocked jobs and clear the dependency.",
        )
        for row in pressure_rows:
            if row.get("pressure_level") != "critical":
                continue
            role = str(row.get("role") or "")
            if int(row.get("queued") or 0) > 0 and int(row.get("idle") or 0) == 0:
                _add_action(
                    f"relieve_saturated_{role}",
                    f"Relieve saturated {role}",
                    int(row.get("queued") or 0),
                    "/api/v1/orchestration/overview",
                    str(row.get("next_action") or ""),
                    role=role,
                )
        _add_action(
            "dispatch_idle_capacity",
            "Dispatch idle capacity",
            len(dispatch_plan),
            "/api/v1/orchestration/queue-pressure-board#dispatch-plan",
            f"Follow the dispatch plan for {len(dispatch_plan)} queued job(s) that match idle role capacity.",
        )
        if not recommended_actions:
            recommended_actions.append(
                {
                    "action_id": "monitor_queue",
                    "label": "Monitor queue",
                    "role": None,
                    "count": 0,
                    "target": "/api/v1/orchestration/overview",
                    "next_action": "No immediate queue intervention is needed.",
                }
            )

        samples: list[dict[str, Any]] = []
        for state in ("blocked_approval", "waiting_deps", "queued"):
            try:
                tasks = self.orch_q.peek(state=state, limit=max(lim, 20), chat_id=chat_id)
            except Exception:
                tasks = []
            if state == "queued":
                tasks = sorted(tasks, key=lambda t: (int(t.priority or 2), float(t.created_at), str(t.job_id)))
            else:
                tasks = sorted(tasks, key=lambda t: (float(t.updated_at), int(t.priority or 2), str(t.job_id)))
            for task in tasks:
                item = _task_to_status(task)
                item["sample_kind"] = "blocked" if state == "blocked_approval" else ("waiting" if state == "waiting_deps" else "queued")
                item["next_action"] = (
                    "Record an approval decision."
                    if state == "blocked_approval"
                    else ("Clear dependency or unblock parent work." if state == "waiting_deps" else "Dispatch when capacity is available.")
                )
                samples.append(item)
                if len(samples) >= lim:
                    break
            if len(samples) >= lim:
                break

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": generated_at,
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "limit": lim,
            "summary": {
                "pressure_level": overall_level,
                "next_action": str(recommended_actions[0].get("next_action") or ""),
                "roles_total": len(pressure_rows),
                "roles_critical": sum(1 for row in pressure_rows if row.get("pressure_level") == "critical"),
                "roles_attention": sum(1 for row in pressure_rows if row.get("pressure_level") == "attention"),
                "max_parallel_total": total_capacity,
                "running": running_total,
                "idle": idle_total,
                "saturation": overall_saturation,
                "queued": queued_total,
                "waiting_deps": waiting_deps_total,
                "blocked_approval": blocked_approval_total,
                "blocked": blocked_total,
                "waiting_total": waiting_deps_total + blocked_approval_total + blocked_total,
                "backlog_total": queued_total + waiting_deps_total + blocked_approval_total + blocked_total,
                "stalled": stalled_total,
                "dispatchable_jobs": len(dispatch_plan),
                "dispatch_roles": len({str(item.get("role") or "") for item in dispatch_plan if str(item.get("role") or "")}),
                "samples_returned": len(samples),
            },
            "pressure_by_role": pressure_rows,
            "recommended_actions": recommended_actions[:10],
            "dispatch_plan": dispatch_plan,
            "top_jobs": samples,
            "backlog_samples": samples,
            "signals": {
                "alerts": list(snap.get("alerts") or [])[:10],
                "risks": list(snap.get("risks") or [])[:10],
                "blocked_approvals": list(snap.get("blocked_requires_approval") or [])[:10],
                "stalled_tasks": list(snap.get("stalled_tasks") or [])[:10],
            },
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
                "overdue_count": 0,
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
            current_stage_overdue = bool(order.get("current_stage_overdue"))
            current_stage_overdue_by_seconds = int(order.get("current_stage_overdue_by_seconds") or 0)
            if current_stage_overdue:
                stage["overdue_count"] = int(stage["overdue_count"]) + 1

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
                "sla_tier": str(order.get("sla_tier") or ""),
                "sla_tier_source": str(order.get("sla_tier_source") or ""),
                "current_stage_overdue": current_stage_overdue,
                "current_stage_overdue_by_seconds": current_stage_overdue_by_seconds,
                "current_stage_deadline_at": order.get("current_stage_deadline_at"),
                "updated_at": updated_at,
            }
            orders = list(stage.get("orders") or [])
            orders.append(compact_order)
            orders.sort(
                key=lambda o: (
                    0 if bool(o.get("current_stage_overdue")) else 1,
                    -int(o.get("current_stage_overdue_by_seconds") or 0),
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
                if bool(orders[0].get("current_stage_overdue")):
                    overdue_by = int(orders[0].get("current_stage_overdue_by_seconds") or 0)
                    stage["recommended_next_action"] = first_action or f"Clear overdue {stage_name} order ({overdue_by}s past SLA)."
                else:
                    stage["recommended_next_action"] = first_action or f"Advance the oldest {stage_name} order."

        ordered_stages = [stages[stage] for stage in _WORKFLOW_STAGE_ORDER]

        def _rank_key(stage: dict[str, Any]) -> tuple[int, int, int, float, int]:
            name = str(stage.get("stage") or "")
            blocked_failed = int(stage.get("blocked_count") or 0) + int(stage.get("failed_count") or 0)
            overdue = int(stage.get("overdue_count") or 0)
            oldest = _coerce_float(stage.get("oldest_updated_at"))
            return (
                -blocked_failed,
                -overdue,
                -int(stage.get("orders_count") or 0),
                oldest if oldest is not None else float("inf"),
                stage_index.get(name, 999),
            )

        bottleneck = min(ordered_stages, key=_rank_key) if ordered_stages else None
        bottleneck_stage = str((bottleneck or {}).get("stage") or _WORKFLOW_STAGE_ORDER[0])
        bottleneck_blocked_failed = int((bottleneck or {}).get("blocked_count") or 0) + int((bottleneck or {}).get("failed_count") or 0)
        bottleneck_overdue = int((bottleneck or {}).get("overdue_count") or 0)
        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": generated_at,
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "limit": lim,
            "summary": {
                "orders_total": int((board.get("summary") or {}).get("orders_total") or 0),
                "bottleneck_stage": bottleneck_stage,
                "bottleneck_score": bottleneck_blocked_failed + bottleneck_overdue,
                "bottleneck_overdue_count": bottleneck_overdue,
                "recommended_next_action": str((bottleneck or {}).get("recommended_next_action") or ""),
            },
            "stages": ordered_stages,
        }

    def workflow_bottleneck_handoff(
        self,
        chat_id: int | None = None,
        stage: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        lim = max(1, min(200, int(limit)))
        board = self.workflow_bottlenecks(chat_id=chat_id, limit=lim)
        summary = board.get("summary") if isinstance(board.get("summary"), dict) else {}
        raw_stage = "" if stage is None else str(stage).strip()
        selected_stage = raw_stage.lower() if raw_stage else str(summary.get("bottleneck_stage") or _WORKFLOW_STAGE_ORDER[0])
        if raw_stage and selected_stage not in _WORKFLOW_STAGE_ORDER:
            valid = ",".join(_WORKFLOW_STAGE_ORDER)
            raise ValueError(f"invalid_stage:{raw_stage}; valid_stages={valid}")

        stages = [item for item in list(board.get("stages") or []) if isinstance(item, dict)]
        stage_by_name = {str(item.get("stage") or ""): item for item in stages}
        stage_payload = stage_by_name.get(selected_stage)
        if stage_payload is None:
            stage_payload = {
                "stage": selected_stage,
                "label": _WORKFLOW_STAGE_LABELS.get(selected_stage, selected_stage),
                "orders_count": 0,
                "blocked_count": 0,
                "failed_count": 0,
                "overdue_count": 0,
                "running_count": 0,
                "pending_count": 0,
                "oldest_updated_at": None,
                "recommended_next_action": "No active orders at this stage.",
                "orders": [],
            }
        orders = [item for item in list(stage_payload.get("orders") or []) if isinstance(item, dict)][:lim]
        query: dict[str, str] = {"stage": selected_stage, "limit": str(lim)}
        if chat_id is not None:
            query["chat_id"] = str(int(chat_id))
        query_string = urllib.parse.urlencode(query)
        inspect_endpoint = f"/api/v1/orchestration/workflow-bottlenecks?{query_string}"
        handoff_endpoint = f"/api/v1/orchestration/workflow-bottlenecks/handoff?{query_string}"

        owner_by_stage = {
            "skynet_plan": "skynet",
            "delivery": "implementer_local",
            "validation": "qa",
            "skynet_review": "skynet",
            "deploy": "release_mgr",
        }
        action_by_stage = {
            "skynet_plan": "Clarify the plan, scope, and delegation path for the selected orders.",
            "delivery": "Unblock implementation and move the selected orders toward validation.",
            "validation": "Run focused validation and produce actionable pass/fail evidence.",
            "skynet_review": "Resolve review blockers and decide whether the selected orders can advance.",
            "deploy": "Prepare release evidence and complete deployment readiness checks.",
        }
        validation_by_stage = {
            "skynet_plan": "Confirm each selected order has bounded scope, owner, and next delegated job.",
            "delivery": "Run the focused unit or integration command covering the implemented slice.",
            "validation": "Run the acceptance test command and capture pass/fail output.",
            "skynet_review": "Review evidence, blockers, and final-sweep output before advancing.",
            "deploy": "Run release-readiness checks and verify evidence links are present.",
        }
        stage_label = str(stage_payload.get("label") or _WORKFLOW_STAGE_LABELS.get(selected_stage, selected_stage))
        if orders:
            action = str(stage_payload.get("recommended_next_action") or action_by_stage.get(selected_stage) or "Advance selected orders.")
            top_order_ids = ", ".join(str(order.get("order_id_short") or order.get("order_id") or "") for order in orders[:3]).strip(", ")
            assignment_prompt = (
                f"ROLE: {owner_by_stage.get(selected_stage, 'skynet')}. Take the workflow bottleneck handoff for "
                f"{stage_label}. Inspect {inspect_endpoint}, prioritize the top orders ({top_order_ids or 'listed'}), "
                "perform the action, and report evidence against the acceptance criteria."
            )
        else:
            action = f"No active orders are currently selected for {stage_label}; monitor the stage and re-run the handoff if demand appears."
            assignment_prompt = (
                f"ROLE: {owner_by_stage.get(selected_stage, 'skynet')}. No orders are currently selected for "
                f"{stage_label}. Inspect {inspect_endpoint} and confirm there is no active handoff work before closing this packet."
            )

        handoff_packet = {
            "owner_role": owner_by_stage.get(selected_stage, "skynet"),
            "action": action,
            "inspect_endpoint": inspect_endpoint,
            "handoff_endpoint": handoff_endpoint,
            "acceptance_criteria": [
                "Selected orders are inspected and triaged in priority order.",
                "Each blocker or failed stage has a recorded next action or resolution.",
                "The selected stage can advance, or the remaining blocker is explicit and assigned.",
            ]
            if orders
            else ["No selected orders exist for this stage, and the empty state is confirmed."],
            "definition_of_done": [
                "Top selected orders have current status, owner, and next action updated.",
                "Validation or review evidence is attached where the stage requires it.",
                "The workflow bottleneck score is reduced or the residual risk is documented.",
            ]
            if orders
            else ["The handoff is closed as no-op with the inspected stage and timestamp recorded."],
            "evidence_required": [
                "Order ids handled and their resulting states.",
                "Commands, checks, review notes, or release evidence used to validate the action.",
                "Any remaining blocker summary with owner and follow-up endpoint.",
            ]
            if orders
            else ["Confirmation that the stage has no selected orders."],
            "suggested_validation": validation_by_stage.get(selected_stage, "Inspect the stage and capture the resulting evidence."),
            "assignment_prompt": assignment_prompt,
        }

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": board.get("generated_at"),
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "limit": lim,
            "selection": {
                "requested_stage": (raw_stage or None),
                "selected_stage": selected_stage,
                "defaulted": not bool(raw_stage),
                "valid_stages": list(_WORKFLOW_STAGE_ORDER),
                "orders_selected": len(orders),
            },
            "summary": summary,
            "stage": {k: v for k, v in stage_payload.items() if k != "orders"},
            "orders": orders,
            "handoff_packet": handoff_packet,
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
            decision_rank = float(_DECISION_RANK.get(decision, 9))
            updated_at = _coerce_float(order.get("updated_at")) or 0.0
            next_action = _priority_next_action(order, decision, blocker)

            priority_item = {
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
                "next_action": next_action,
                "score_breakdown": {
                    "decision_rank": decision_rank,
                    "priority": priority,
                    "stage_rank": stage_rank,
                    "updated_at": updated_at,
                },
                "merge_ready": bool(order.get("merge_ready")),
                "merged_to_main": bool(order.get("merged_to_main")),
                "children_total": int(order.get("children_total") or 0),
                "children_by_role": order.get("children_by_role") if isinstance(order.get("children_by_role"), dict) else {},
                "updated_at": updated_at,
            }
            factory_delta_contract = _factory_delta_contract_from_trace(root_task)
            if factory_delta_contract is not None:
                priority_item["factory_delta_contract"] = factory_delta_contract
            selection_quality = _selection_quality_metadata(priority_item, order_row=order_row, root_task=root_task)
            priority_item["selection_quality"] = selection_quality
            if str(selection_quality.get("status") or "").strip().lower() == "needs_review":
                decision = "selection_review"
                decision_rank = float(_DECISION_RANK[decision])
                priority_item["decision"] = decision
                priority_item["score_breakdown"]["decision_rank"] = decision_rank
                priority_item["next_action"] = _selection_review_action(priority_item)
            priority_item["handoff"] = _proactive_handoff_metadata(priority_item, decision)
            ranked.append(priority_item)

        ranked.sort(
            key=lambda order: (
                float((order.get("score_breakdown") or {}).get("decision_rank") or 9),
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
                "selection_quality": first.get("selection_quality"),
            }

        by_decision: dict[str, int] = {}
        by_selection_quality: dict[str, int] = {}
        for order in ranked:
            decision = str(order.get("decision") or "unknown")
            by_decision[decision] = by_decision.get(decision, 0) + 1
            selection_quality = order.get("selection_quality") if isinstance(order.get("selection_quality"), dict) else {}
            selection_status = str(selection_quality.get("status") or "unknown")
            by_selection_quality[selection_status] = by_selection_quality.get(selection_status, 0) + 1

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
                "by_selection_quality": by_selection_quality,
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
            ("selection_review", "Selection Review"),
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
            "decision",
            "why",
            "primary_blocker",
            "next_action",
            "handoff",
            "updated_at",
            "merge_ready",
            "merged_to_main",
            "children_total",
            "children_by_role",
            "selection_quality",
            "factory_delta_contract",
        ]
        for order in list(priorities.get("orders") or []):
            if not isinstance(order, dict):
                continue
            lane = str(order.get("decision") or "").strip().lower()
            selection_quality = order.get("selection_quality") if isinstance(order.get("selection_quality"), dict) else {}
            if lane == "advance" and str(selection_quality.get("status") or "").strip().lower() == "needs_review":
                lane = "selection_review"
            if lane not in lanes_by_key:
                lane = "advance"
            compact_order = {key: order.get(key) for key in compact_keys}
            compact_order["decision"] = lane
            lanes_by_key[lane].append(compact_order)

        lanes: list[dict[str, Any]] = []
        for lane, label in lane_defs:
            orders = lanes_by_key[lane]
            recommended = str((orders[0].get("next_action") if orders else "") or "")
            lane_top_order: dict[str, Any] | None = None
            lane_top_rank: int | None = None
            for order in orders:
                if not isinstance(order, dict):
                    continue
                try:
                    rank = int(order.get("rank") or 0)
                except Exception:
                    continue
                if rank <= 0:
                    continue
                if lane_top_rank is None or rank < lane_top_rank:
                    lane_top_order = order
                    lane_top_rank = rank
            execution_packet = (
                _proactive_lane_execution_packet(lane_top_order, lane, label)
                if isinstance(lane_top_order, dict)
                else None
            )
            lanes.append(
                {
                    "lane": lane,
                    "label": label,
                    "count": len(orders),
                    "recommended_next_action": recommended,
                    "execution_packet": execution_packet,
                    "orders": orders,
                }
            )

        top_order: dict[str, Any] | None = None
        top_lane = None
        top_rank: int | None = None
        for lane in lanes:
            for order in lane["orders"]:
                if not isinstance(order, dict):
                    continue
                try:
                    rank = int(order.get("rank") or 0)
                except Exception:
                    continue
                if rank <= 0:
                    continue
                if top_rank is None or rank < top_rank:
                    top_rank = rank
                    top_order = order
                    top_lane = str(lane.get("lane") or "")

        top_execution_packet = None
        if isinstance(top_order, dict) and top_lane:
            for lane in lanes:
                if str(lane.get("lane") or "") == top_lane:
                    packet = lane.get("execution_packet")
                    top_execution_packet = packet if isinstance(packet, dict) else None
                    break
        next_delegate = None
        if isinstance(top_execution_packet, dict):
            next_delegate = {
                "owner_role": top_execution_packet.get("owner_role"),
                "order_id": top_execution_packet.get("order_id"),
                "lane": top_execution_packet.get("lane"),
                "action": top_execution_packet.get("action"),
                "inspect_endpoint": top_execution_packet.get("inspect_endpoint"),
                "handoff_endpoint": top_execution_packet.get("handoff_endpoint"),
            }
            if isinstance(top_execution_packet.get("factory_delta_contract"), dict):
                next_delegate["factory_delta_contract"] = top_execution_packet.get("factory_delta_contract")

        lane_counts = {str(lane.get("lane") or ""): int(lane.get("count") or 0) for lane in lanes}
        returned = sum(lane_counts.values())
        selection_quality_counts: dict[str, int] = {}
        for lane in lanes:
            for order in list(lane.get("orders") or []):
                if not isinstance(order, dict):
                    continue
                selection_quality = order.get("selection_quality") if isinstance(order.get("selection_quality"), dict) else {}
                status = str(selection_quality.get("status") or "unknown")
                selection_quality_counts[status] = selection_quality_counts.get(status, 0) + 1
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
                "selection_quality": selection_quality_counts,
                "top_lane": top_lane,
                "top_action": (str(top_order.get("next_action") or "") if isinstance(top_order, dict) else None),
                "next_delegate": next_delegate,
            },
            "lanes": lanes,
            "top_execution_packet": top_execution_packet,
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

        runbook_status = self.runbook_status()
        runbook_summary = runbook_status.get("summary") if isinstance(runbook_status.get("summary"), dict) else {}
        due_runbooks = [
            item
            for item in list(runbook_status.get("items") or [])
            if isinstance(item, dict) and (bool(item.get("due")) or bool(item.get("overdue")))
        ]
        runbooks = {
            "summary": runbook_summary,
            "due_items": due_runbooks[:10],
        }
        due_or_overdue_runbooks = max(
            int(runbook_summary.get("due") or 0),
            int(runbook_summary.get("overdue") or 0),
        )
        _add_action(
            "inspect_due_runbooks",
            "Inspect due runbooks",
            "/api/v1/orchestration/runbooks",
            due_or_overdue_runbooks,
            "Scheduled runbooks are due or overdue.",
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
            "runbooks": runbooks,
            "workflow_bottleneck": workflow_bottleneck,
            "proactive_action_plan": proactive_action_plan,
            "recommended_actions": compact_recommended_actions,
            "staleness_seconds": snap.get("staleness_seconds"),
        }

    def operator_focus(
        self,
        chat_id: int | None = None,
        limit: int = 5,
        *,
        categories: Any = None,
        urgencies: Any = None,
        sources: Any = None,
        receipt_states: list[str] | None = None,
    ) -> dict[str, Any]:
        lim = max(1, min(20, int(limit)))
        generated_at = float(time.time())
        focus_chat_id = int(chat_id) if chat_id is not None else None

        category_order = {
            "critical_alert": 0,
            "approval": 1,
            "proactive_release": 2,
            "proactive_unblock": 3,
            "proactive_health": 4,
            "stalled": 5,
            "workflow_bottleneck": 6,
            "queue": 7,
            "proactive_advance": 8,
            "proactive_monitor": 9,
        }
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        source_keys = ("control_room", "proactive_priorities", "proactive_health")
        source_counts = {key: 0 for key in source_keys}

        def _normalize_filter(values: Any) -> list[str]:
            if values is None:
                return []
            raw_values = [values] if isinstance(values, str) else list(values or [])
            normalized: list[str] = []
            seen: set[str] = set()
            for raw in raw_values:
                for part in str(raw or "").split(","):
                    value = part.strip().lower()
                    if not value or value in seen:
                        continue
                    normalized.append(value)
                    seen.add(value)
            return normalized

        active_filters = {
            "categories": _normalize_filter(categories),
            "urgencies": _normalize_filter(urgencies),
            "sources": _normalize_filter(sources),
            "receipt_states": _normalize_filter(receipt_states),
        }

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

        def _order_id_from_ref(ref: dict[str, Any]) -> str | None:
            for key in ("order_id", "parent_job_id"):
                value = _text(ref.get(key))
                if value:
                    return value
            return None

        def _append_chat_scope(path: str) -> str:
            base = _text(path)
            if not base or focus_chat_id is None:
                return base
            separator = "&" if "?" in base else "?"
            return f"{base}{separator}chat_id={focus_chat_id}"

        def _triage_targets(
            *,
            category: str,
            target: str,
            order_id: str | None,
            job_id: str | None,
        ) -> dict[str, Any]:
            oid = _text(order_id)
            jid = _text(job_id)
            base_target = _text(target)

            inspect_path = _append_chat_scope(base_target)
            inspect_target = None
            action_target = None

            if oid:
                inspect_target = f"order:{oid[:8]}"
                if category.startswith("proactive_") and "order_id=" in base_target:
                    inspect_path = base_target
                else:
                    inspect_path = f"/api/v1/orchestration/orders/evidence?order_id={oid}"
            if jid:
                action_target = f"job:{jid[:8]}"
                if not inspect_target:
                    inspect_target = action_target
                    inspect_path = _append_chat_scope("/api/v1/status/snapshot")
            if action_target is None and inspect_target is not None:
                action_target = inspect_target
            if inspect_target is None and base_target:
                inspect_target = base_target
            if action_target is None and base_target:
                action_target = base_target

            return {
                "inspect_path": inspect_path or None,
                "inspect_target": inspect_target,
                "action_target": action_target,
            }

        def _focus_handoff_metadata(
            *,
            source: str,
            category: str,
            action_id: str,
            label: str,
            next_action: str,
            target: str,
            reason: str,
            evidence_type: str | None = None,
        ) -> dict[str, Any]:
            endpoint = _append_chat_scope(target) or "/api/v1/orchestration/operator-focus"
            category_key = _text(category)
            evidence_key = _text(evidence_type).lower()

            if source == "control_room" and category_key == "approval":
                suggested_role = "reviewer_local"
                definition_of_done = [
                    next_action,
                    "Approve, reject, or annotate the blocked operator decision.",
                ]
                checklist = [
                    "Open the decision endpoint.",
                    "Inspect the blocked job or pending decision context.",
                    "Record the approval decision and rationale.",
                ]
                evidence_expectations = [
                    "approval or rejection decision",
                    "decision rationale",
                    "updated job or decision state",
                ]
            elif source == "proactive_health":
                if any(token in evidence_key for token in ("release", "deploy", "merge")):
                    suggested_role = "release_mgr"
                elif any(token in evidence_key for token in ("qa", "review")):
                    suggested_role = "reviewer_local"
                else:
                    suggested_role = "implementer_local"
                definition_of_done = [
                    next_action,
                    "Attach evidence that resolves or explains the proactive health alert.",
                ]
                checklist = [
                    "Open the proactive health endpoint.",
                    "Inspect the recommended action and source signals.",
                    "Record concrete evidence or a bounded follow-up.",
                ]
                evidence_expectations = [
                    "proactive health report status",
                    "recommended action result",
                    "updated evidence or follow-up note",
                ]
            else:
                suggested_role = "implementer_local"
                definition_of_done = [
                    next_action,
                    "Record the operator-visible outcome before handing back.",
                ]
                checklist = [
                    "Open the suggested endpoint.",
                    "Inspect the referenced control-room signals.",
                    "Complete the bounded operator action.",
                ]
                evidence_expectations = [
                    "control-room signal review",
                    "operator action result",
                    "updated status snapshot or trace",
                ]

            return {
                "suggested_role": suggested_role,
                "suggested_endpoint": endpoint,
                "inspect_path": endpoint,
                "definition_of_done": definition_of_done,
                "checklist": checklist,
                "evidence_expectations": evidence_expectations,
                "title": _text(label, action_id.replace("_", " ").title()),
            }

        def _fallback_owner_role(*, source: str, category: str) -> str:
            category_key = _text(category).lower()
            source_key = _text(source).lower()
            if category_key == "approval":
                return "reviewer_local"
            if category_key in {"proactive_release", "proactive_monitor"}:
                return "release_mgr"
            if category_key == "proactive_health":
                return "implementer_local"
            if category_key == "stalled":
                return "sre"
            if source_key == "proactive_priorities":
                return "implementer_local"
            if source_key == "control_room":
                return "reviewer_local"
            return "implementer_local"

        def _compact_owner_action_evidence(item: dict[str, Any]) -> dict[str, Any]:
            handoff = item.get("handoff") if isinstance(item.get("handoff"), dict) else {}
            owner_role = _text((handoff or {}).get("suggested_role")) or _fallback_owner_role(
                source=_text(item.get("source")),
                category=_text(item.get("category")),
            )
            evidence_required = [
                text
                for text in (_text(value) for value in list((handoff or {}).get("evidence_expectations") or []))
                if text
            ][:3]
            if not evidence_required:
                category = _text(item.get("category"), "operator action").replace("_", " ")
                evidence_required = [f"{category} completion evidence"]
            inspect_path = (
                _text((handoff or {}).get("inspect_path"))
                or _text((handoff or {}).get("suggested_endpoint"))
                or _text(item.get("inspect_path"))
                or _text(item.get("target"))
                or None
            )
            return {
                "owner_role": owner_role,
                "action": _text(item.get("next_action"), "Review operator focus item."),
                "evidence_required": evidence_required,
                "inspect_path": inspect_path,
            }

        def _delegate_contract(item: dict[str, Any]) -> dict[str, Any]:
            handoff = item.get("handoff") if isinstance(item.get("handoff"), dict) else {}
            owner_evidence = item.get("owner_action_evidence") if isinstance(item.get("owner_action_evidence"), dict) else {}
            delegate_role = (
                _text(handoff.get("suggested_role"))
                or _text(owner_evidence.get("owner_role"))
                or _fallback_owner_role(source=_text(item.get("source")), category=_text(item.get("category")))
            )
            task_title = _text(handoff.get("title")) or _text(item.get("label")) or _text(item.get("action_id"), "Operator focus item")
            action = _text(item.get("next_action"), _text(owner_evidence.get("action"), "Review operator focus item."))
            handoff_endpoint = (
                _text(handoff.get("suggested_endpoint"))
                or _text(item.get("target"))
                or "/api/v1/orchestration/operator-focus"
            )
            inspect_endpoint = (
                _text(handoff.get("inspect_path"))
                or _text(item.get("inspect_path"))
                or handoff_endpoint
            )
            definition_of_done = [
                text
                for text in (_text(value) for value in list(handoff.get("definition_of_done") or []))
                if text
            ]
            if not definition_of_done:
                definition_of_done = [action, "Record the operator-visible outcome before handing back."]
            acceptance_criteria = [
                text
                for text in (_text(value) for value in list(handoff.get("checklist") or []))
                if text
            ]
            if not acceptance_criteria:
                acceptance_criteria = [
                    f"Inspect {inspect_endpoint}.",
                    action,
                    "Stay within the referenced order, job, endpoint, or focus item scope.",
                ]
            evidence_required = [
                text
                for text in (
                    _text(value)
                    for value in list(owner_evidence.get("evidence_required") or handoff.get("evidence_expectations") or [])
                )
                if text
            ]
            if not evidence_required:
                evidence_required = ["completion summary", "updated status or trace evidence"]
            suggested_tests = [
                "Re-read the inspect endpoint after the action.",
                "Verify the acceptance criteria and evidence expectations are satisfied.",
            ]
            risk_notes = [
                "Do not edit outside the referenced endpoint, order, job, repository, or focus item scope.",
                "Preserve existing endpoint, auth, filter, rank, and not-found behavior.",
            ]

            prompt_lines = [
                f"ROLE: {delegate_role}.",
                "",
                f"Task: {task_title}",
                f"Scope: Work only on action { _text(item.get('action_id'), 'unknown') } and the referenced endpoint/order/job context.",
                f"Inspect endpoint: {inspect_endpoint}",
                f"Handoff endpoint: {handoff_endpoint}",
                f"Action: {action}",
                "",
                "Acceptance criteria:",
                *(f"- {criterion}" for criterion in acceptance_criteria),
                "",
                "Definition of done:",
                *(f"- {criterion}" for criterion in definition_of_done),
                "",
                "Evidence required:",
                *(f"- {evidence}" for evidence in evidence_required),
                "",
                "Suggested validation:",
                *(f"- {test}" for test in suggested_tests),
                "",
                "Warning: do not edit outside the referenced endpoint, order, job, repository, or focus item scope.",
            ]

            return {
                "delegate_role": delegate_role,
                "task_title": task_title,
                "task_prompt": "\n".join(prompt_lines),
                "handoff_endpoint": handoff_endpoint,
                "inspect_endpoint": inspect_endpoint,
                "acceptance_criteria": acceptance_criteria,
                "definition_of_done": definition_of_done,
                "evidence_required": evidence_required,
                "suggested_tests": suggested_tests,
                "risk_notes": risk_notes,
                "source_action_id": _text(item.get("action_id")) or None,
            }

        def _briefing_packet(item: dict[str, Any]) -> dict[str, Any]:
            handoff = item.get("handoff") if isinstance(item.get("handoff"), dict) else {}
            owner_evidence = item.get("owner_action_evidence") if isinstance(item.get("owner_action_evidence"), dict) else {}
            delegate = item.get("delegate_contract") if isinstance(item.get("delegate_contract"), dict) else {}
            owner_role = (
                _text(owner_evidence.get("owner_role"))
                or _text(delegate.get("delegate_role"))
                or _fallback_owner_role(source=_text(item.get("source")), category=_text(item.get("category")))
            )
            action = _text(owner_evidence.get("action")) or _text(item.get("next_action"), "Review operator focus item.")
            inspect_endpoint = (
                _text(delegate.get("inspect_endpoint"))
                or _text(owner_evidence.get("inspect_path"))
                or _text(handoff.get("inspect_path"))
                or _text(item.get("inspect_path"))
                or _text(item.get("target"))
                or "/api/v1/orchestration/operator-focus"
            )
            handoff_endpoint = (
                _text(delegate.get("handoff_endpoint"))
                or _text(handoff.get("suggested_endpoint"))
                or _text(item.get("target"))
                or "/api/v1/orchestration/operator-focus"
            )
            evidence_required = [
                text
                for text in (
                    _text(value)
                    for value in list(delegate.get("evidence_required") or owner_evidence.get("evidence_required") or [])
                )
                if text
            ][:3]
            if not evidence_required:
                evidence_required = ["completion summary"]
            suggested_validation = [
                text
                for text in (_text(value) for value in list(delegate.get("suggested_tests") or []))
                if text
            ][:3]
            if not suggested_validation:
                suggested_validation = ["Re-read the inspect endpoint after the action."]
            definition_of_done = [
                text
                for text in (_text(value) for value in list(delegate.get("definition_of_done") or handoff.get("definition_of_done") or []))
                if text
            ][:3]
            if not definition_of_done:
                definition_of_done = [action, "Record the operator-visible outcome before handing back."]
            assignment_prompt = (
                f"ROLE: {owner_role}. Action: {action} Inspect {inspect_endpoint}; "
                f"handoff via {handoff_endpoint}; return evidence for action { _text(item.get('action_id'), 'unknown') }."
            )
            return {
                "owner_role": owner_role,
                "action": action,
                "inspect_endpoint": inspect_endpoint,
                "handoff_endpoint": handoff_endpoint,
                "evidence_required": evidence_required,
                "suggested_validation": suggested_validation,
                "definition_of_done": definition_of_done,
                "assignment_prompt": assignment_prompt,
            }

        def _focus_packet(item: dict[str, Any] | None) -> dict[str, Any] | None:
            if not item:
                return None
            keys = (
                "rank",
                "action_id",
                "urgency",
                "category",
                "label",
                "next_action",
                "target",
                "inspect_path",
                "inspect_target",
                "action_target",
                "order_id",
                "job_id",
                "repo_id",
                "source",
                "source_signals",
                "reason",
                "evidence_type",
                "priority",
                "count",
                "handoff",
                "owner_action_evidence",
                "delegate_contract",
                "briefing_packet",
                "receipt_state",
                "latest_receipt",
                "receipt_count",
                "receipt_counts_by_state",
                "receipt_history",
                "receipt_follow_up",
            )
            return {key: item.get(key) for key in keys if key in item}

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
            repo_id: str | None = None,
            source: str,
            source_signals: list[str],
            updated_at: float | None = None,
            handoff: dict[str, Any] | None = None,
            evidence_type: str | None = None,
            priority: str | None = None,
        ) -> dict[str, Any]:
            triage = _triage_targets(category=category, target=target, order_id=order_id, job_id=job_id)
            packet = {
                "rank": 0,
                "action_id": action_id,
                "category": category,
                "urgency": urgency,
                "label": label,
                "reason": reason,
                "next_action": next_action,
                "target": target,
                "inspect_path": triage.get("inspect_path"),
                "inspect_target": triage.get("inspect_target"),
                "action_target": triage.get("action_target"),
                "count": int(max(1, count)),
                "order_id": order_id,
                "job_id": job_id,
                "repo_id": repo_id,
                "source": source,
                "source_signals": source_signals,
                "score": _score(category, urgency, updated_at),
                "updated_at": updated_at,
            }
            if evidence_type:
                packet["evidence_type"] = evidence_type
            if priority:
                packet["priority"] = priority
            if isinstance(handoff, dict) and handoff:
                packet["handoff"] = handoff
            packet["owner_action_evidence"] = _compact_owner_action_evidence(packet)
            packet["delegate_contract"] = _delegate_contract(packet)
            packet["briefing_packet"] = _briefing_packet(packet)
            return packet

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
            label = _text(action.get("label"), source_action_id.replace("_", " ").title())
            reason = _text(action.get("reason"), "Control room recommends operator attention.")
            next_action = label or "Review operator action."
            target = _text(action.get("target"), "/api/v1/orchestration/control-room")
            items.append(
                _item(
                    action_id=source_action_id,
                    category=category,
                    urgency=urgency,
                    label=label,
                    reason=reason,
                    next_action=next_action,
                    target=target,
                    count=count,
                    order_id=(_order_id_from_ref(ref) if isinstance(ref, dict) else None),
                    job_id=(_text(ref.get("job_id")) or None) if isinstance(ref, dict) else None,
                    source="control_room",
                    source_signals=[source_action_id, category],
                    updated_at=updated_at,
                    handoff=_focus_handoff_metadata(
                        source="control_room",
                        category=category,
                        action_id=source_action_id,
                        label=label,
                        next_action=next_action,
                        target=target,
                        reason=reason,
                    ),
                )
            )
            source_counts["control_room"] += 1

        workflow = control.get("workflow_bottleneck") if isinstance(control.get("workflow_bottleneck"), dict) else {}
        bottleneck_score = _num(workflow.get("score"), 0)
        if bottleneck_score > 0:
            stage = _text(workflow.get("stage"), "workflow")
            label = "Clear workflow bottleneck"
            reason = f"{bottleneck_score} blocked or failed order(s) at {stage}."
            next_action = _text(workflow.get("recommended_next_action"), "Inspect the bottleneck stage.")
            target = "/api/v1/orchestration/workflow-bottlenecks"
            items.append(
                _item(
                    action_id=f"workflow_bottleneck:{stage}",
                    category="workflow_bottleneck",
                    urgency="medium",
                    label=label,
                    reason=reason,
                    next_action=next_action,
                    target=target,
                    count=bottleneck_score,
                    order_id=None,
                    job_id=None,
                    source="control_room",
                    source_signals=["workflow_bottleneck", stage],
                    updated_at=None,
                    handoff=_focus_handoff_metadata(
                        source="control_room",
                        category="workflow_bottleneck",
                        action_id=f"workflow_bottleneck:{stage}",
                        label=label,
                        next_action=next_action,
                        target=target,
                        reason=reason,
                    ),
                )
            )
            source_counts["control_room"] += 1

        priorities = self.proactive_priorities(chat_id=focus_chat_id, limit=20)
        proactive_category = {
            "release": ("proactive_release", "high", "Release proactive order"),
            "unblock": ("proactive_unblock", "high", "Unblock proactive order"),
            "selection_review": ("proactive_selection_review", "high", "Review proactive selection"),
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
                    handoff=(order.get("handoff") if isinstance(order.get("handoff"), dict) else None),
                )
            )
            source_counts["proactive_priorities"] += 1

        health = self.proactive_health()
        health_status = _text(health.get("status"), "not_configured")
        if health_status != "not_configured" and bool(health.get("alert_active")):
            alert_level = _text(health.get("alert_level"), "OK").upper()
            if alert_level == "CRITICAL":
                health_level = "critical"
            elif health_level == "ok":
                health_level = "attention"

            def _health_urgency(priority: str) -> str:
                mapped = {"P0": "critical", "P1": "high", "P2": "medium"}.get(priority.upper())
                if mapped:
                    return mapped
                return {"CRITICAL": "critical", "WARN": "high"}.get(alert_level, "medium")

            def _health_signals(action: dict[str, Any] | None = None) -> list[str]:
                signals = [
                    f"alert_level:{alert_level}",
                    f"operational:{_text(health.get('operational_status'), 'unknown')}",
                    f"trend:{_text(health.get('trend_status'), 'unknown')}",
                ]
                if action:
                    priority = _text(action.get("priority"))
                    evidence_type = _text(action.get("evidence_type"))
                    order_id = _text(action.get("order_id"))
                    repo_id = _text(action.get("repo_id"))
                    if priority:
                        signals.append(f"priority:{priority}")
                    if evidence_type:
                        signals.append(f"evidence_type:{evidence_type}")
                    if order_id:
                        signals.append(f"order_id:{order_id}")
                    if repo_id:
                        signals.append(f"repo_id:{repo_id}")
                return signals

            health_actions = [action for action in list(health.get("recommended_actions") or []) if isinstance(action, dict)]
            seen_health_action_ids: set[str] = set()
            for idx, action in enumerate(health_actions, start=1):
                priority = _text(action.get("priority"))
                evidence_type = _text(action.get("evidence_type"), "recommendation")
                order_id = _text(action.get("order_id")) or None
                repo_id = _text(action.get("repo_id")) or None
                selector = order_id or repo_id or str(idx)
                action_id = f"proactive_health:{evidence_type}:{selector}"
                if action_id in seen_health_action_ids:
                    action_id = f"{action_id}:{idx}"
                seen_health_action_ids.add(action_id)
                label_subject = evidence_type.replace("_", " ").strip() or "health recommendation"
                label = f"Triage {label_subject}"
                reason = _text(action.get("reason"), _text(health.get("summary_reason"), "Proactive health alert is active."))
                next_action = _text(action.get("action") or action.get("summary"), "Review the proactive health report.")
                target = "/api/v1/status/proactive-health"
                items.append(
                    _item(
                        action_id=action_id,
                        category="proactive_health",
                        urgency=_health_urgency(priority),
                        label=label,
                        reason=reason,
                        next_action=next_action,
                        target=target,
                        count=_num(action.get("count"), 1),
                        order_id=order_id,
                        job_id=None,
                        repo_id=repo_id,
                        source="proactive_health",
                        source_signals=_health_signals(action),
                        updated_at=None,
                        handoff=_focus_handoff_metadata(
                            source="proactive_health",
                            category="proactive_health",
                            action_id=action_id,
                            label=label,
                            next_action=next_action,
                            target=target,
                            reason=reason,
                            evidence_type=evidence_type,
                        ),
                        evidence_type=evidence_type,
                        priority=priority or None,
                    )
                )
                source_counts["proactive_health"] += 1

            if not health_actions:
                label = "Inspect proactive health"
                reason = _text(health.get("summary_reason"), "Proactive health alert is active.")
                next_action = "Review the proactive health report."
                target = "/api/v1/status/proactive-health"
                items.append(
                    _item(
                        action_id="proactive_health_alert",
                        category="proactive_health",
                        urgency=_health_urgency(""),
                        label=label,
                        reason=reason,
                        next_action=next_action,
                        target=target,
                        count=1,
                        order_id=None,
                        job_id=None,
                        source="proactive_health",
                        source_signals=_health_signals(),
                        updated_at=None,
                        handoff=_focus_handoff_metadata(
                            source="proactive_health",
                            category="proactive_health",
                            action_id="proactive_health_alert",
                            label=label,
                            next_action=next_action,
                            target=target,
                            reason=reason,
                        ),
                    )
                )
                source_counts["proactive_health"] += 1

        self._decorate_operator_focus_receipts(items, now=generated_at)

        def _receipt_state_selectors(item: dict[str, Any]) -> set[str]:
            selectors: set[str] = set()
            receipt_state = str(item.get("receipt_state") or "").strip().lower() or "new"
            if receipt_state:
                selectors.add(receipt_state)
            follow_up = item.get("receipt_follow_up") if isinstance(item.get("receipt_follow_up"), dict) else {}
            severity = str((follow_up or {}).get("severity") or "").strip().lower()
            if severity in {"follow_up", "escalation"}:
                selectors.add(severity)
            return selectors

        def _matches_receipt_states(item: dict[str, Any]) -> bool:
            filters = active_filters["receipt_states"]
            if not filters:
                return True
            selectors = _receipt_state_selectors(item)
            return any(value in selectors for value in filters)

        available_source_counts = {key: 0 for key in source_keys}
        for item in items:
            source = str(item.get("source") or "").strip().lower()
            if source:
                available_source_counts[source] = int(available_source_counts.get(source, 0)) + 1
        available_receipt_states = sorted(
            {
                selector
                for item in items
                for selector in _receipt_state_selectors(item)
            }
        )
        available = {
            "total": len(items),
            "categories": sorted({str(item.get("category") or "").strip().lower() for item in items if str(item.get("category") or "").strip()}),
            "urgencies": sorted({str(item.get("urgency") or "").strip().lower() for item in items if str(item.get("urgency") or "").strip()}),
            "sources": sorted({str(item.get("source") or "").strip().lower() for item in items if str(item.get("source") or "").strip()}),
            "receipt_states": available_receipt_states,
        }

        filtered_items = [
            item
            for item in items
            if (not active_filters["categories"] or str(item.get("category") or "").strip().lower() in active_filters["categories"])
            and (not active_filters["urgencies"] or str(item.get("urgency") or "").strip().lower() in active_filters["urgencies"])
            and (not active_filters["sources"] or str(item.get("source") or "").strip().lower() in active_filters["sources"])
            and _matches_receipt_states(item)
        ]
        source_counts = {key: 0 for key in source_keys}
        for item in filtered_items:
            source = str(item.get("source") or "").strip().lower()
            if source:
                source_counts[source] = int(source_counts.get(source, 0)) + 1

        filtered_items.sort(
            key=lambda item: (
                int(category_order.get(str(item.get("category") or ""), 99)),
                int(urgency_order.get(str(item.get("urgency") or ""), 9)),
                -int(item.get("score") or 0),
                float(item.get("updated_at") or 0.0),
                str(item.get("action_id") or ""),
            )
        )
        returned = filtered_items[:lim]
        for idx, item in enumerate(returned, start=1):
            item["rank"] = idx
        receipt_follow_up_count = sum(1 for item in returned if isinstance(item.get("receipt_follow_up"), dict))
        receipt_escalation_count = sum(
            1
            for item in returned
            if isinstance(item.get("receipt_follow_up"), dict)
            and str((item.get("receipt_follow_up") or {}).get("severity") or "") == "escalation"
        )

        release_board_endpoint = _append_chat_scope("/api/v1/orchestration/release-readiness-board?limit=20")
        release_lanes: dict[str, Any] = {
            "total": 0,
            "by_lane": {"ready": 0, "blocked": 0, "not_ready": 0, "released": 0},
            "endpoint": release_board_endpoint,
        }
        try:
            release_board = self.release_readiness_board(chat_id=focus_chat_id, limit=20)
        except Exception:
            release_board = {}
        if isinstance(release_board, dict):
            release_summary = release_board.get("summary") if isinstance(release_board.get("summary"), dict) else {}
            board_by_lane = release_summary.get("by_lane") if isinstance(release_summary.get("by_lane"), dict) else {}
            by_lane = dict(release_lanes["by_lane"])
            for lane in ("ready", "blocked", "not_ready", "released"):
                by_lane[lane] = _num(board_by_lane.get(lane), 0)
            release_lanes["by_lane"] = by_lane
            release_lanes["total"] = _num(
                release_summary.get("orders_total", release_summary.get("returned")),
                sum(by_lane.values()),
            )

            top_order: dict[str, Any] | None = None
            lanes = release_board.get("lanes") if isinstance(release_board.get("lanes"), dict) else {}
            for lane in ("ready", "blocked", "not_ready", "released"):
                lane_payload = lanes.get(lane) if isinstance(lanes.get(lane), dict) else {}
                for order in list(lane_payload.get("orders") or []):
                    if isinstance(order, dict):
                        top_order = order
                        break
                if top_order is not None:
                    break
            if top_order is not None:
                release_lanes["top_order"] = {
                    "rank": top_order.get("rank"),
                    "order_id": top_order.get("order_id"),
                    "order_id_short": top_order.get("order_id_short"),
                    "title": top_order.get("title"),
                    "release_lane": top_order.get("release_lane"),
                    "readiness_state": top_order.get("readiness_state"),
                    "summary": top_order.get("summary"),
                    "next_action": top_order.get("next_action"),
                    "endpoint": top_order.get("release_readiness_endpoint"),
                    "handoff_endpoint": top_order.get("handoff_endpoint"),
                }

        top = returned[0] if returned else {}
        top_focus = _focus_packet(top if top else None)
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
                "top_focus": top_focus,
                "filters": active_filters,
                "available": available,
                "filtered_out": len(items) - len(filtered_items),
                "source_counts": source_counts,
                "available_source_counts": available_source_counts,
                "receipt_follow_up_count": receipt_follow_up_count,
                "receipt_escalation_count": receipt_escalation_count,
                "release_lanes": release_lanes,
            },
            "top_focus": top_focus,
            "items": returned,
        }

    def _operator_focus_receipt_action_id(self, receipt: dict[str, Any]) -> str:
        details = receipt.get("details") if isinstance(receipt.get("details"), dict) else {}
        item_identity = details.get("item_identity") if isinstance(details.get("item_identity"), dict) else {}
        selection = details.get("selection") if isinstance(details.get("selection"), dict) else {}
        for value in (item_identity.get("action_id"), selection.get("action_id")):
            action_id = str(value or "").strip()
            if action_id:
                return action_id
        return ""

    def _compact_operator_focus_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        details = receipt.get("details") if isinstance(receipt.get("details"), dict) else {}
        persisted_details = details.get("operator_focus_details")
        compact = {
            "state": receipt.get("state"),
            "summary": receipt.get("summary"),
            "next_action": receipt.get("next_action"),
            "actor": details.get("actor"),
            "recorded_at": receipt.get("ts"),
            "ts": receipt.get("ts"),
            "order_id": receipt.get("order_id"),
            "job_id": receipt.get("job_id"),
        }
        if isinstance(persisted_details, dict) and persisted_details:
            compact["details"] = dict(persisted_details)
        selection = details.get("selection")
        if isinstance(selection, dict) and selection:
            compact["selection"] = dict(selection)
        item_identity = details.get("item_identity")
        if isinstance(item_identity, dict) and item_identity:
            compact["item_identity"] = dict(item_identity)
        return {key: value for key, value in compact.items() if value is not None}

    def _operator_focus_receipt_rollup_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        action_id: str,
        history_limit: int = _OPERATOR_FOCUS_RECEIPT_HISTORY_LIMIT,
    ) -> dict[str, Any]:
        aid = str(action_id or "").strip()
        if not aid:
            return {
                "receipt_count": 0,
                "receipt_counts_by_state": {},
                "receipt_history": [],
                "latest_receipt": None,
            }

        receipts: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("kind") or "").strip() != _OPERATOR_FOCUS_RECEIPT_EVENT_TYPE:
                continue
            if self._operator_focus_receipt_action_id(row) != aid:
                continue
            try:
                ts = float(row.get("ts") or 0.0)
            except Exception:
                ts = 0.0
            receipts.append((ts, row))

        receipts.sort(key=lambda item: item[0], reverse=True)
        counts_by_state: dict[str, int] = {}
        for _, receipt in receipts:
            state = str(receipt.get("state") or "").strip().lower() or "unknown"
            counts_by_state[state] = int(counts_by_state.get(state, 0)) + 1

        history = [
            self._compact_operator_focus_receipt(receipt)
            for _, receipt in receipts[: max(1, int(history_limit))]
        ]
        latest = history[0] if history else None
        return {
            "receipt_count": len(receipts),
            "receipt_counts_by_state": counts_by_state,
            "receipt_history": history,
            "latest_receipt": latest,
        }

    def _operator_focus_receipt_follow_up(
        self,
        latest_receipt: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any] | None:
        state = str(latest_receipt.get("state") or "").strip().lower()
        stale_after_seconds = _OPERATOR_FOCUS_RECEIPT_STALE_AFTER_SECONDS.get(state)
        if stale_after_seconds is None:
            return None
        try:
            recorded_at = float(latest_receipt.get("recorded_at"))
        except Exception:
            return None
        if recorded_at <= 0:
            return None
        age_seconds = max(0, int(float(now) - recorded_at))
        if age_seconds < stale_after_seconds:
            return None
        severity = "escalation" if age_seconds >= stale_after_seconds * 2 else "follow_up"
        if severity == "escalation":
            message = f"Latest {state} receipt is stale beyond escalation threshold."
            next_action = "Escalate for owner status or record a completed receipt."
        else:
            message = f"Latest {state} receipt is stale and needs operator follow-up."
            next_action = "Follow up with the receipt owner for current status."
        return {
            "active": True,
            "severity": severity,
            "state": state,
            "age_seconds": age_seconds,
            "stale_after_seconds": int(stale_after_seconds),
            "message": message,
            "next_action": next_action,
        }

    def _latest_operator_focus_receipt(self, *, order_id: str, action_id: str) -> dict[str, Any] | None:
        oid = str(order_id or "").strip()
        aid = str(action_id or "").strip()
        if not oid or not aid:
            return None
        try:
            rows = self.orch_q.list_decision_log(order_id=oid, limit=50)
        except Exception:
            return None

        receipts = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("kind") or "").strip() == _OPERATOR_FOCUS_RECEIPT_EVENT_TYPE
            and self._operator_focus_receipt_action_id(row) == aid
        ]
        if not receipts:
            return None
        return max(receipts, key=lambda row: float(row.get("ts") or 0.0))

    def _decorate_operator_focus_receipts(self, items: list[dict[str, Any]], *, now: float | None = None) -> None:
        generated_at = float(time.time() if now is None else now)
        rows_by_order_id: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            order_id = str(item.get("order_id") or "").strip()
            action_id = str(item.get("action_id") or "").strip()
            if not order_id or not action_id or order_id in rows_by_order_id:
                continue
            try:
                rows_by_order_id[order_id] = self.orch_q.list_decision_log(order_id=order_id, limit=500)
            except Exception:
                rows_by_order_id[order_id] = []

        for item in items:
            if not isinstance(item, dict):
                continue
            item["receipt_state"] = "new"
            order_id = str(item.get("order_id") or "").strip()
            action_id = str(item.get("action_id") or "").strip()
            if not order_id or not action_id:
                continue
            rollup = self._operator_focus_receipt_rollup_from_rows(
                rows_by_order_id.get(order_id, []),
                action_id=action_id,
            )
            item["receipt_count"] = rollup["receipt_count"]
            item["receipt_counts_by_state"] = rollup["receipt_counts_by_state"]
            item["receipt_history"] = rollup["receipt_history"]
            latest = rollup.get("latest_receipt")
            if not isinstance(latest, dict) or not latest:
                continue
            state = str(latest.get("state") or "").strip().lower()
            item["receipt_state"] = state or "new"
            item["latest_receipt"] = latest
            follow_up = self._operator_focus_receipt_follow_up(latest, now=generated_at)
            if follow_up:
                item["receipt_follow_up"] = follow_up

    def _select_operator_focus_item(
        self,
        report: dict[str, Any],
        *,
        action_id: str | None = None,
        rank: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        items = [item for item in list(report.get("items") or []) if isinstance(item, dict)]
        normalized_action_id = str(action_id or "").strip()
        selected: dict[str, Any] | None = None
        matched_by = "top"

        if normalized_action_id:
            selected = next(
                (item for item in items if str(item.get("action_id") or "").strip() == normalized_action_id),
                None,
            )
            matched_by = "action_id"
        elif rank is not None:
            selected = next(
                (
                    item
                    for item in items
                    if isinstance(item.get("rank"), int) and int(item.get("rank")) == int(rank)
                ),
                None,
            )
            matched_by = "rank"
        elif items:
            selected = items[0]

        selection = {
            "action_id": normalized_action_id or None,
            "rank": int(rank) if rank is not None else None,
            "matched_by": matched_by if selected else None,
        }
        return selected, selection

    def _operator_focus_item_identity(self, item: dict[str, Any] | None) -> dict[str, Any] | None:
        if not item:
            return None
        keys = (
            "rank",
            "action_id",
            "urgency",
            "category",
            "label",
            "source",
            "order_id",
            "job_id",
            "repo_id",
            "receipt_state",
            "receipt_count",
            "receipt_counts_by_state",
            "receipt_history",
            "latest_receipt",
            "receipt_follow_up",
        )
        return {key: item.get(key) for key in keys if item.get(key) is not None}

    def _fallback_operator_focus_briefing_packet(self, item: dict[str, Any]) -> dict[str, Any]:
        def _text(value: Any, default: str = "") -> str:
            text = str(value or "").strip()
            return text if text else default

        action_id = _text(item.get("action_id"), "unknown")
        source = _text(item.get("source"), "operator_focus")
        action = _text(item.get("next_action"), _text(item.get("label"), "Review operator focus item."))
        inspect_endpoint = (
            _text(item.get("inspect_path"))
            or _text(item.get("target"))
            or "/api/v1/orchestration/operator-focus"
        )
        handoff_endpoint = _text(item.get("target")) or inspect_endpoint

        return {
            "owner_role": "operator",
            "action": action,
            "inspect_endpoint": inspect_endpoint,
            "handoff_endpoint": handoff_endpoint,
            "evidence_required": ["completion summary"],
            "suggested_validation": ["Re-read the inspect endpoint after the action."],
            "definition_of_done": [action, "Record the operator-visible outcome before handing back."],
            "assignment_prompt": (
                f"Action: {action} Inspect {inspect_endpoint}; return evidence for "
                f"{source} action {action_id}."
            ),
        }

    def operator_focus_handoff(
        self,
        *,
        chat_id: int | None = None,
        action_id: str | None = None,
        rank: int | None = None,
        categories: list[str] | None = None,
        urgencies: list[str] | None = None,
        sources: list[str] | None = None,
        receipt_states: list[str] | None = None,
    ) -> dict[str, Any]:
        report = self.operator_focus(
            chat_id=chat_id,
            limit=20,
            categories=categories,
            urgencies=urgencies,
            sources=sources,
            receipt_states=receipt_states,
        )
        selected, selection = self._select_operator_focus_item(report, action_id=action_id, rank=rank)
        selected_packet = dict(selected) if selected else None
        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": report.get("generated_at"),
            "chat_id": report.get("chat_id"),
            "selection": selection,
            "summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
            "item": selected_packet,
        }

    def operator_focus_briefing(
        self,
        *,
        chat_id: int | None = None,
        action_id: str | None = None,
        rank: int | None = None,
        categories: list[str] | None = None,
        urgencies: list[str] | None = None,
        sources: list[str] | None = None,
        receipt_states: list[str] | None = None,
    ) -> dict[str, Any]:
        report = self.operator_focus(
            chat_id=chat_id,
            limit=20,
            categories=categories,
            urgencies=urgencies,
            sources=sources,
            receipt_states=receipt_states,
        )
        selected, selection = self._select_operator_focus_item(report, action_id=action_id, rank=rank)
        briefing_packet = None
        if selected:
            existing = selected.get("briefing_packet")
            briefing_packet = dict(existing) if isinstance(existing, dict) and existing else self._fallback_operator_focus_briefing_packet(selected)

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": report.get("generated_at"),
            "chat_id": report.get("chat_id"),
            "selection": selection,
            "summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
            "item_identity": self._operator_focus_item_identity(selected),
            "briefing_packet": briefing_packet,
        }

    def operator_focus_receipt(
        self,
        *,
        chat_id: int | None = None,
        state: str,
        summary: str | None = None,
        next_action: str | None = None,
        actor: str | None = None,
        details: dict[str, Any] | None = None,
        action_id: str | None = None,
        rank: int | None = None,
        categories: list[str] | None = None,
        urgencies: list[str] | None = None,
        sources: list[str] | None = None,
        receipt_states: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_state = str(state or "").strip().lower()
        if normalized_state not in _OPERATOR_FOCUS_RECEIPT_STATUSES:
            raise ValueError("invalid_operator_focus_receipt_state")

        def _text(value: Any, default: str = "") -> str:
            text = str(value or "").strip()
            return text if text else default

        report = self.operator_focus(
            chat_id=chat_id,
            limit=20,
            categories=categories,
            urgencies=urgencies,
            sources=sources,
            receipt_states=receipt_states,
        )
        selected, selection = self._select_operator_focus_item(report, action_id=action_id, rank=rank)
        generated_at = float(time.time())
        item_identity = self._operator_focus_item_identity(selected)

        receipt_summary = _text(summary)
        if not receipt_summary and selected:
            receipt_summary = f"Operator focus item {normalized_state}: {_text(selected.get('label'), _text(selected.get('action_id'), 'selected item'))}"
        receipt_next_action = _text(next_action) or None
        receipt_details = dict(details) if isinstance(details, dict) else {}
        actor_name = _text(actor) or None

        persisted = False
        persistence_reason = "focus_item_not_selected"
        order_id = _text(selected.get("order_id")) if selected else ""
        job_id = _text(selected.get("job_id")) if selected else ""
        if selected and order_id:
            persistence_details = {
                "event_type": _OPERATOR_FOCUS_RECEIPT_EVENT_TYPE,
                "actor": actor_name,
                "selection": dict(selection),
                "item_identity": item_identity or {},
                "operator_focus_details": receipt_details,
            }
            self.orch_q.append_decision_log(
                order_id=order_id,
                job_id=job_id or order_id,
                kind=_OPERATOR_FOCUS_RECEIPT_EVENT_TYPE,
                state=normalized_state,
                summary=receipt_summary,
                next_action=receipt_next_action,
                details=persistence_details,
            )
            persisted = True
            persistence_reason = "decision_log_appended"
        elif selected:
            persistence_reason = "focus_item_not_tied_to_order"

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": report.get("generated_at"),
            "chat_id": report.get("chat_id"),
            "selection": selection,
            "summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
            "item_identity": item_identity,
            "receipt": {
                "event_type": _OPERATOR_FOCUS_RECEIPT_EVENT_TYPE,
                "state": normalized_state,
                "summary": receipt_summary,
                "next_action": receipt_next_action,
                "actor": actor_name,
                "details": receipt_details,
                "persisted": persisted,
                "persistence_reason": persistence_reason,
                "recorded_at": generated_at,
                "order_id": order_id or None,
                "job_id": (job_id or order_id or None) if selected else None,
            },
        }

    def operator_focus_receipt_trail(
        self,
        *,
        chat_id: int | None = None,
        action_id: str | None = None,
        rank: int | None = None,
        categories: list[str] | None = None,
        urgencies: list[str] | None = None,
        sources: list[str] | None = None,
        receipt_states: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        try:
            receipt_limit = int(limit)
        except Exception:
            receipt_limit = 20
        receipt_limit = max(1, min(100, receipt_limit))

        report = self.operator_focus(
            chat_id=chat_id,
            limit=20,
            categories=categories,
            urgencies=urgencies,
            sources=sources,
            receipt_states=receipt_states,
        )
        selected, selection = self._select_operator_focus_item(report, action_id=action_id, rank=rank)
        item_identity = self._operator_focus_item_identity(selected)
        empty_rollup = {
            "receipt_count": 0,
            "receipt_counts_by_state": {},
            "receipt_history": [],
            "latest_receipt": None,
        }
        rollup = empty_rollup

        order_id = str(selected.get("order_id") or "").strip() if selected else ""
        action = str(selected.get("action_id") or "").strip() if selected else ""
        if order_id and action:
            try:
                rows = self.orch_q.list_decision_log(order_id=order_id, limit=500)
            except Exception:
                rows = []
            rollup = self._operator_focus_receipt_rollup_from_rows(
                rows,
                action_id=action,
                history_limit=receipt_limit,
            )

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": report.get("generated_at"),
            "chat_id": report.get("chat_id"),
            "selection": selection,
            "summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
            "item_identity": item_identity,
            "receipt_count": rollup["receipt_count"],
            "receipt_counts_by_state": rollup["receipt_counts_by_state"],
            "latest_receipt": rollup["latest_receipt"],
            "receipts": rollup["receipt_history"],
        }

    def operator_focus_briefing_bundle(
        self,
        *,
        chat_id: int | None = None,
        limit: int = 5,
        categories: list[str] | None = None,
        urgencies: list[str] | None = None,
        sources: list[str] | None = None,
        receipt_states: list[str] | None = None,
    ) -> dict[str, Any]:
        report = self.operator_focus(
            chat_id=chat_id,
            limit=limit,
            categories=categories,
            urgencies=urgencies,
            sources=sources,
            receipt_states=receipt_states,
        )
        briefings: list[dict[str, Any]] = []
        for item in list(report.get("items") or []):
            if not isinstance(item, dict):
                continue
            existing = item.get("briefing_packet")
            briefing_packet = (
                dict(existing)
                if isinstance(existing, dict) and existing
                else self._fallback_operator_focus_briefing_packet(item)
            )
            briefings.append(
                {
                    "selection": {
                        "action_id": item.get("action_id"),
                        "rank": item.get("rank"),
                        "matched_by": "rank",
                    },
                    "item_identity": self._operator_focus_item_identity(item),
                    "briefing_packet": briefing_packet,
                }
            )

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": report.get("generated_at"),
            "chat_id": report.get("chat_id"),
            "limit": report.get("limit"),
            "summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
                "briefings": briefings,
        }

    def operator_focus_digest(
        self,
        *,
        chat_id: int | None = None,
        limit: int = 5,
        categories: list[str] | None = None,
        urgencies: list[str] | None = None,
        sources: list[str] | None = None,
        receipt_states: list[str] | None = None,
    ) -> dict[str, Any]:
        report = self.operator_focus(
            chat_id=chat_id,
            limit=limit,
            categories=categories,
            urgencies=urgencies,
            sources=sources,
            receipt_states=receipt_states,
        )
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        items = [item for item in list(report.get("items") or []) if isinstance(item, dict)]

        def _text(value: Any, default: str = "") -> str:
            text = str(value or "").strip()
            return text if text else default

        def _count_by(key: str, *, default: str = "unknown") -> dict[str, int]:
            counts: dict[str, int] = {}
            for item in items:
                value = _text(item.get(key), default).lower()
                counts[value] = int(counts.get(value, 0)) + 1
            return dict(sorted(counts.items()))

        follow_up_counts = {"active": 0, "follow_up": 0, "escalation": 0}
        follow_ups: list[dict[str, Any]] = []
        for item in items:
            follow_up = item.get("receipt_follow_up") if isinstance(item.get("receipt_follow_up"), dict) else None
            if not follow_up:
                continue
            severity = _text(follow_up.get("severity"), "follow_up").lower()
            follow_up_counts["active"] += 1
            follow_up_counts[severity] = int(follow_up_counts.get(severity, 0)) + 1
            follow_ups.append(
                {
                    "rank": item.get("rank"),
                    "action_id": item.get("action_id"),
                    "urgency": item.get("urgency"),
                    "category": item.get("category"),
                    "label": item.get("label"),
                    "receipt_state": item.get("receipt_state", "new"),
                    "severity": severity,
                    "message": follow_up.get("message"),
                    "next_action": follow_up.get("next_action"),
                    "age_seconds": follow_up.get("age_seconds"),
                }
            )

        def _suggested_commands(rank: Any, *, inspect_path: Any = None) -> dict[str, str]:
            try:
                rank_i = max(1, int(rank or 1))
            except Exception:
                rank_i = 1
            commands = {}
            inspect = str(inspect_path or "").strip()
            if inspect:
                commands["inspect"] = inspect
            commands.update(
                {
                    "brief": f"/focus brief chat {rank_i}",
                    "handoff": f"/focus handoff chat {rank_i}",
                    "trail": f"/focus trail chat {rank_i}",
                    "ack": f"/focus ack chat {rank_i} <summary>",
                    "start": f"/focus start chat {rank_i} <summary>",
                    "done": f"/focus done chat {rank_i} <summary>",
                }
            )
            return commands

        def _compact_action(item: dict[str, Any]) -> dict[str, Any]:
            handoff = item.get("handoff") if isinstance(item.get("handoff"), dict) else {}
            delegate = item.get("delegate_contract") if isinstance(item.get("delegate_contract"), dict) else {}
            receipt_counts = item.get("receipt_counts_by_state") if isinstance(item.get("receipt_counts_by_state"), dict) else {}
            inspect_path = item.get("inspect_path") or handoff.get("inspect_path") or delegate.get("inspect_endpoint")
            out = {
                "rank": item.get("rank"),
                "action_id": item.get("action_id"),
                "urgency": item.get("urgency"),
                "category": item.get("category"),
                "label": item.get("label"),
                "next_action": item.get("next_action"),
                "target": item.get("target"),
                "inspect_path": inspect_path,
                "inspect_target": item.get("inspect_target"),
                "action_target": item.get("action_target"),
                "handoff_endpoint": handoff.get("suggested_endpoint") or delegate.get("handoff_endpoint") or item.get("target"),
                "source": item.get("source"),
                "source_signals": item.get("source_signals"),
                "reason": item.get("reason"),
                "receipt_state": item.get("receipt_state", "new"),
                "receipt_count": item.get("receipt_count", 0),
                "receipt_counts_by_state": receipt_counts,
                "latest_receipt": item.get("latest_receipt"),
                "receipt_follow_up": item.get("receipt_follow_up"),
                "suggested_commands": _suggested_commands(item.get("rank"), inspect_path=inspect_path),
            }
            return {key: value for key, value in out.items() if value is not None}

        digest_summary = {
            "returned": summary.get("returned", len(items)),
            "health_level": summary.get("health_level"),
            "top_action_id": summary.get("top_action_id"),
            "top_category": summary.get("top_category"),
            "filters": summary.get("filters") if isinstance(summary.get("filters"), dict) else {},
            "available": summary.get("available") if isinstance(summary.get("available"), dict) else {},
            "filtered_out": summary.get("filtered_out", 0),
            "counts": {
                "by_urgency": _count_by("urgency"),
                "by_category": _count_by("category"),
                "by_source": _count_by("source"),
                "by_receipt_state": _count_by("receipt_state", default="new"),
                "follow_ups": follow_up_counts,
            },
            "release_lanes": summary.get("release_lanes") if isinstance(summary.get("release_lanes"), dict) else {},
        }

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": report.get("generated_at"),
            "chat_id": report.get("chat_id"),
            "limit": report.get("limit"),
            "summary": digest_summary,
            "top_actions": [_compact_action(item) for item in items],
            "follow_ups": follow_ups,
            "suggested_commands": {
                "refresh": "/focus digest chat",
                "global": "/focus digest all",
                "brief_top": "/focus brief chat 1",
                "handoff_top": "/focus handoff chat 1",
                "trail_top": "/focus trail chat 1",
                "ack_top": "/focus ack chat 1 <summary>",
                "start_top": "/focus start chat 1 <summary>",
                "done_top": "/focus done chat 1 <summary>",
            },
        }

    def operator_shift_brief(
        self,
        chat_id: int | None = None,
        limit: int = 5,
        categories: list[str] | None = None,
        urgencies: list[str] | None = None,
        sources: list[str] | None = None,
        receipt_states: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            lim = int(limit)
        except Exception:
            lim = 5
        lim = max(1, min(20, lim))

        control = self.control_room(chat_id=chat_id)
        proactive_health = self.proactive_health()
        digest = self.operator_focus_digest(
            chat_id=chat_id,
            limit=lim,
            categories=categories,
            urgencies=urgencies,
            sources=sources,
            receipt_states=receipt_states,
        )
        briefing_bundle = self.operator_focus_briefing_bundle(
            chat_id=chat_id,
            limit=lim,
            categories=categories,
            urgencies=urgencies,
            sources=sources,
            receipt_states=receipt_states,
        )

        attention = control.get("attention") if isinstance(control.get("attention"), dict) else {}
        queue = control.get("queue") if isinstance(control.get("queue"), dict) else {}
        orders = control.get("orders") if isinstance(control.get("orders"), dict) else {}
        runbooks = control.get("runbooks") if isinstance(control.get("runbooks"), dict) else {}
        runbook_summary = runbooks.get("summary") if isinstance(runbooks.get("summary"), dict) else {}
        health = control.get("health") if isinstance(control.get("health"), dict) else {}
        digest_summary = digest.get("summary") if isinstance(digest.get("summary"), dict) else {}
        digest_counts = digest_summary.get("counts") if isinstance(digest_summary.get("counts"), dict) else {}
        follow_up_counts = digest_counts.get("follow_ups") if isinstance(digest_counts.get("follow_ups"), dict) else {}
        follow_up_items = [item for item in list(digest.get("follow_ups") or [])[:lim] if isinstance(item, dict)]

        top_actions: list[dict[str, Any]] = []
        for item in list(digest.get("top_actions") or [])[:lim]:
            if not isinstance(item, dict):
                continue
            suggested_commands = (
                item.get("suggested_commands") if isinstance(item.get("suggested_commands"), dict) else {}
            )
            compact = {
                "rank": item.get("rank"),
                "action_id": item.get("action_id"),
                "urgency": item.get("urgency"),
                "category": item.get("category"),
                "label": item.get("label"),
                "next_action": item.get("next_action"),
                "target": item.get("target"),
                "inspect_path": item.get("inspect_path"),
                "handoff_endpoint": item.get("handoff_endpoint"),
                "source": item.get("source"),
                "receipt_state": item.get("receipt_state", "new"),
                "receipt_follow_up": item.get("receipt_follow_up") if isinstance(item.get("receipt_follow_up"), dict) else None,
                "suggested_commands": suggested_commands,
            }
            top_actions.append({key: value for key, value in compact.items() if value is not None})

        def _command_plan_for(item: dict[str, Any]) -> dict[str, Any]:
            commands = item.get("suggested_commands") if isinstance(item.get("suggested_commands"), dict) else {}
            plan_commands: dict[str, str] = {}
            inspect = str(commands.get("inspect") or item.get("inspect_path") or "").strip()
            if inspect:
                plan_commands["inspect"] = inspect
            for key in ("brief", "handoff", "trail", "ack", "start", "done"):
                value = str(commands.get(key) or "").strip()
                if value:
                    plan_commands[key] = value
            endpoints: dict[str, str] = {}
            for key, source_key in (("inspect", "inspect_path"), ("handoff", "handoff_endpoint")):
                value = str(item.get(source_key) or "").strip()
                if value:
                    endpoints[key] = value
            out = {
                "rank": item.get("rank"),
                "action_id": item.get("action_id"),
                "label": item.get("label"),
                "commands": plan_commands,
                "endpoints": endpoints,
            }
            return {key: value for key, value in out.items() if value}

        command_plan = [_command_plan_for(item) for item in top_actions]

        return {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": digest.get("generated_at") or control.get("generated_at"),
            "chat_id": digest.get("chat_id"),
            "limit": lim,
            "snapshot_summary": {
                "health_level": health.get("level"),
                "queue": {
                    "queued_runnable": int(queue.get("queued_runnable") or 0),
                    "waiting_deps": int(queue.get("waiting_deps") or 0),
                    "blocked_approval": int(queue.get("blocked_approval") or 0),
                    "running": int(queue.get("running") or 0),
                    "blocked_legacy": int(queue.get("blocked_legacy") or 0),
                },
                "orders": {
                    "active_count": int(orders.get("active_count") or 0),
                },
                "alerts": {
                    "count": len(list(attention.get("alerts") or [])),
                    "stalled_tasks": len(list(attention.get("stalled_tasks") or [])),
                },
                "risks": {
                    "count": len(list(attention.get("risks") or [])),
                },
                "decisions": {
                    "pending": len(list(attention.get("pending_decisions") or [])),
                    "blocked_approvals": len(list(attention.get("blocked_approvals") or [])),
                },
                "runbooks": {
                    "due": int(runbook_summary.get("due") or 0),
                    "overdue": int(runbook_summary.get("overdue") or 0),
                },
            },
            "proactive_health": proactive_health,
            "operator_focus_digest": digest,
            "receipt_follow_ups": {
                "counts": follow_up_counts,
                "items": follow_up_items,
            },
            "briefings": list(briefing_bundle.get("briefings") or [])[:lim],
            "next_actions": top_actions,
            "command_plan": command_plan,
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
