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
            running_total = int(self.orch_q.get_running_count(chat_id=chat_id))
        except Exception:
            queued_total = 0
            running_total = 0
        blocked_total = 0
        try:
            for rec in (role_health or {}).values():
                blocked_total += int((rec or {}).get("blocked", 0) or 0)
        except Exception:
            blocked_total = 0

        blocked_requires_approval: list[dict[str, Any]] = []
        try:
            blocked = self.orch_q.peek(state="blocked", limit=200, chat_id=chat_id)
            blocked_sorted = sorted(blocked, key=lambda t: float(t.updated_at), reverse=True)
            for t in blocked_sorted[:30]:
                blocked_requires_approval.append(_task_to_status(t))
                if len(blocked_requires_approval) >= 12:
                    break
        except Exception:
            blocked_requires_approval = []
        newest_updated_at = 0.0

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

        payload: dict[str, Any] = {
            "api_version": "v1",
            "schema_version": 1,
            "generated_at": float(now),
            "chat_id": (int(chat_id) if chat_id is not None else None),
            "orders_active": orders_out,
            "workers": workers_out,
            "queued_total": int(queued_total),
            "running_total": int(running_total),
            "blocked_total": int(blocked_total),
            "blocked_requires_approval": blocked_requires_approval,
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
