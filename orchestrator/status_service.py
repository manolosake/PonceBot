from __future__ import annotations

from dataclasses import dataclass
from typing import Any
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


@dataclass
class StatusService:
    orch_q: OrchestratorQueue
    role_profiles: dict[str, dict[str, Any]] | None = None
    cache_ttl_seconds: int = 2

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
        for o in orders:
            oid = str(o.get("order_id") or "").strip()
            if not oid:
                continue
            children = self.orch_q.jobs_by_parent(parent_job_id=oid, limit=200)
            counts: dict[str, int] = {}
            for c in children:
                counts[c.state] = counts.get(c.state, 0) + 1
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
                }
            )

        workers_out: list[dict[str, Any]] = []
        # Scope counts: when chat_id is provided, only count jobs for that chat.
        try:
            role_health = self.orch_q.get_role_health(chat_id=chat_id)
        except Exception:
            role_health = {}
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
        blocked_total = 0
        try:
            for rec in (role_health or {}).values():
                blocked_total += int((rec or {}).get("blocked", 0) or 0)
        except Exception:
            blocked_total = 0

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
        if no_open_jobs:
            workers_out = []

        for role in roles:
            n = int(max_parallel.get(role) or 1)
            running = self.orch_q.peek(role=role, state="running", limit=200, chat_id=chat_id)
            queued = self.orch_q.peek(role=role, state="queued", limit=200, chat_id=chat_id)

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

        payload: dict[str, Any] = {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": float(now),
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "orders_active": orders_out,
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
