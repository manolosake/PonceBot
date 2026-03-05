from __future__ import annotations

from typing import Any

from .schemas.task import Task
from .storage import SQLiteTaskStorage


class OrchestratorQueue:
    """Thin in-process coordinator around persistent queue + role policy."""

    def __init__(
        self,
        storage: SQLiteTaskStorage,
        *,
        role_profiles: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._storage = storage
        self._profiles = role_profiles or {}

    def submit_task(self, task: Task) -> str:
        return self._storage.submit_task(task)

    def submit_batch(self, tasks: list[Task]) -> list[str]:
        return self._storage.submit_batch(tasks)

    def take_next(
        self,
        role: str | None = None,
        limit_roles: set[str] | None = None,
    ) -> Task | None:
        if self._storage.is_paused_globally():
            return None
        max_parallel = self._max_parallel_by_role()
        return self._storage.claim_next(role=role, limit_roles=limit_roles, max_parallel_by_role=max_parallel)

    def update_state(self, job_id: str, state: str, **metadata: Any) -> bool:
        return self._storage.update_state(job_id=job_id, state=state, **metadata)

    def update_trace(self, job_id: str, **metadata: Any) -> bool:
        return self._storage.update_trace(job_id=job_id, **metadata)

    def bump_retry(self, job_id: str, *, due_at: float, error: str | None = None) -> bool:
        return self._storage.bump_retry(job_id=job_id, due_at=due_at, error=error)

    def cancel(self, job_id: str) -> bool:
        return self._storage.cancel(job_id)

    def recover_stale_running(self) -> int:
        return self._storage.recover_stale_running()

    def pause_role(self, role: str) -> None:
        self._storage.pause_role(role)

    def resume_role(self, role: str) -> None:
        self._storage.resume_role(role)

    def pause_all_roles(self) -> None:
        self._storage.pause_all_roles()

    def resume_all_roles(self) -> None:
        self._storage.resume_all_roles()

    def is_paused_globally(self) -> bool:
        return self._storage.is_orchestrator_paused()

    def cancel_running_jobs(self) -> int:
        return self._storage.cancel_running_jobs()

    def cancel_by_states(
        self,
        *,
        states: tuple[str, ...],
        reason: str = "bulk_cancel",
        chat_id: int | None = None,
        exclude_job_ids: set[str] | None = None,
    ) -> int:
        return self._storage.cancel_by_states(states=states, reason=reason, chat_id=chat_id, exclude_job_ids=exclude_job_ids)

    def set_job_approved(self, job_id: str, approved: bool = True) -> bool:
        return self._storage.set_job_approved(job_id, approved=approved)

    def clear_job_approval(self, job_id: str) -> bool:
        return self._storage.clear_job_approval(job_id)

    def get_role_health(self, *, chat_id: int | None = None) -> dict[str, dict[str, int]]:
        return self._storage.get_role_health(chat_id=chat_id)

    def get_job(self, job_id: str) -> Task | None:
        return self._storage.get_job(job_id)

    def delete_job(self, job_id: str) -> bool:
        return self._storage.delete_job(job_id)

    def get_agent_thread(self, *, chat_id: int, role: str) -> str | None:
        return self._storage.get_agent_thread(chat_id=chat_id, role=role)

    def set_agent_thread(self, *, chat_id: int, role: str, thread_id: str) -> None:
        self._storage.set_agent_thread(chat_id=chat_id, role=role, thread_id=thread_id)

    def clear_agent_thread(self, *, chat_id: int, role: str) -> bool:
        return self._storage.clear_agent_thread(chat_id=chat_id, role=role)

    def clear_agent_threads(self, *, chat_id: int) -> int:
        return self._storage.clear_agent_threads(chat_id=chat_id)

    def lease_workspace(self, *, role: str, job_id: str, slots: int) -> int | None:
        return self._storage.lease_workspace(role=role, job_id=job_id, slots=slots)

    def release_workspace(self, *, job_id: str) -> bool:
        return self._storage.release_workspace(job_id=job_id)

    def get_workspace_lease(self, *, job_id: str) -> tuple[str, int] | None:
        return self._storage.get_workspace_lease(job_id=job_id)

    def get_runbook_last_run(self, *, runbook_id: str) -> float:
        return self._storage.get_runbook_last_run(runbook_id=runbook_id)

    def set_runbook_last_run(self, *, runbook_id: str, ts: float) -> None:
        self._storage.set_runbook_last_run(runbook_id=runbook_id, ts=ts)

    def jobs_by_parent(self, *, parent_job_id: str, limit: int = 200) -> list[Task]:
        return self._storage.jobs_by_parent(parent_job_id=parent_job_id, limit=limit)

    def inbox(self, *, role: str | None = None, limit: int = 25) -> list[Task]:
        return self._storage.inbox(role=role, limit=limit)

    def get_queued_count(self, *, chat_id: int | None = None) -> int:
        return self._storage.queued_count(chat_id=chat_id)

    def get_running_count(self, *, chat_id: int | None = None) -> int:
        return self._storage.running_count(chat_id=chat_id)

    def get_waiting_deps_count(self, *, chat_id: int | None = None) -> int:
        return self._storage.waiting_deps_count(chat_id=chat_id)

    def get_blocked_approval_count(self, *, chat_id: int | None = None) -> int:
        return self._storage.blocked_approval_count(chat_id=chat_id)

    def list_stalled_tasks(
        self,
        *,
        stale_after_seconds: float = 1800.0,
        limit: int = 100,
    ) -> list[Task]:
        return self._storage.list_stalled_tasks(stale_after_seconds=stale_after_seconds, limit=limit)

    def peek(
        self,
        *,
        role: str | None = None,
        state: str | None = None,
        chat_id: int | None = None,
        limit: int = 20,
    ) -> list[Task]:
        return self._storage.peek(role=role, state=state, chat_id=chat_id, limit=limit)

    def dequeue_with_budget(
        self,
        role: str | None = None,
        *,
        limit_roles: set[str] | None = None,
        budget: float | None = None,
    ) -> Task | None:
        max_parallel = self._max_parallel_by_role()
        return self._storage.dequeue_with_budget(
            role=role,
            limit_roles=limit_roles,
            budget=budget,
            max_parallel_by_role=max_parallel,
        )

    def get_role_backlog(self, *, state: str | None = None) -> dict[str, dict[str, int]]:
        return self._storage.get_role_backlog(state=state)

    def set_cost_cap(self, role: str, max_cost_usd: float) -> None:
        self._storage.set_cost_cap(role, max_cost_usd)

    def jobs_by_state(self, *, state: str, limit: int = 50, chat_id: int | None = None) -> list[Task]:
        return self._storage.jobs_by_state(state=state, limit=limit, chat_id=chat_id)

    def claim_autonomous_due_jobs(self, *, limit: int = 5) -> list[Task]:
        return self._storage.claim_autonomous_due_jobs(limit=limit)

    # CEO Orders (autopilot scope)
    def upsert_order(
        self,
        *,
        order_id: str,
        chat_id: int,
        title: str,
        body: str,
        status: str = "active",
        priority: int = 2,
        intent_type: str | None = None,
        source_message_id: int | None = None,
        reply_to_message_id: int | None = None,
        phase: str | None = None,
        project_id: str | None = None,
    ) -> None:
        self._storage.upsert_order(
            order_id=order_id,
            chat_id=chat_id,
            title=title,
            body=body,
            status=status,
            priority=priority,
            intent_type=intent_type,
            source_message_id=source_message_id,
            reply_to_message_id=reply_to_message_id,
            phase=phase,
            project_id=project_id,
        )

    def list_orders(self, *, chat_id: int, status: str | None, limit: int = 50) -> list[dict[str, Any]]:
        return self._storage.list_orders(chat_id=chat_id, status=status, limit=limit)

    def list_orders_global(self, *, status: str | None, limit: int = 50) -> list[dict[str, Any]]:
        return self._storage.list_orders_global(status=status, limit=limit)

    def get_order(self, order_id: str, *, chat_id: int) -> dict[str, Any] | None:
        return self._storage.get_order(order_id, chat_id=chat_id)

    def set_order_status(self, order_id: str, *, chat_id: int, status: str) -> bool:
        return self._storage.set_order_status(order_id, chat_id=chat_id, status=status)

    def latest_active_order(self, *, chat_id: int) -> dict[str, Any] | None:
        return self._storage.latest_active_order(chat_id=chat_id)

    def set_order_phase(self, order_id: str, *, chat_id: int, phase: str) -> bool:
        return self._storage.set_order_phase(order_id, chat_id=chat_id, phase=phase)

    def append_audit_event(self, *, event_type: str, actor: str = "system", details: dict[str, Any] | None = None) -> None:
        self._storage.append_audit_event(event_type=event_type, actor=actor, details=details)

    def list_audit_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self._storage.list_audit_events(limit=limit)

    def upsert_project(
        self,
        *,
        project_id: str,
        name: str,
        path: str,
        runtime_mode: str = "venv",
        ports: list[str] | None = None,
        status: str = "active",
        created_by: str = "jarvis",
    ) -> None:
        self._storage.upsert_project(
            project_id=project_id,
            name=name,
            path=path,
            runtime_mode=runtime_mode,
            ports=ports,
            status=status,
            created_by=created_by,
        )

    def list_projects(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return self._storage.list_projects(status=status, limit=limit)

    def hard_reset_bootstrap(self, *, reason: str = "hard_reset_bootstrap") -> dict[str, int]:
        return self._storage.hard_reset_bootstrap(reason=reason)

    # Cached status snapshots (dashboard support).
    def get_status_cache(self, cache_key: str) -> dict[str, Any] | None:
        return self._storage.get_status_cache(cache_key)

    def set_status_cache(self, cache_key: str, *, payload: dict[str, Any], ttl_seconds: int) -> None:
        self._storage.set_status_cache(cache_key, payload=payload, ttl_seconds=ttl_seconds)

    # Structured logs (dashboard/audit).
    def append_decision_log(
        self,
        *,
        order_id: str,
        job_id: str,
        kind: str,
        state: str,
        summary: str,
        next_action: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._storage.append_decision_log(
            order_id=order_id,
            job_id=job_id,
            kind=kind,
            state=state,
            summary=summary,
            next_action=next_action,
            details=details,
        )

    def list_decision_log(self, *, order_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self._storage.list_decision_log(order_id=order_id, limit=limit)

    def append_delegation_edge(
        self,
        *,
        root_ticket_id: str,
        from_job_id: str,
        to_job_id: str,
        edge_type: str,
        to_role: str | None = None,
        to_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._storage.append_delegation_edge(
            root_ticket_id=root_ticket_id,
            from_job_id=from_job_id,
            to_job_id=to_job_id,
            edge_type=edge_type,
            to_role=to_role,
            to_key=to_key,
            details=details,
        )

    def list_delegation_log(self, *, root_ticket_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return self._storage.list_delegation_log(root_ticket_id=root_ticket_id, limit=limit)

    def append_worker_activity(
        self,
        *,
        ts: float | None = None,
        role: str,
        worker_slot: int | None,
        worker_id: str | None,
        job_id: str | None,
        state: str | None,
        phase: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._storage.append_worker_activity(
            ts=ts,
            role=role,
            worker_slot=worker_slot,
            worker_id=worker_id,
            job_id=job_id,
            state=state,
            phase=phase,
            details=details,
        )

    def list_worker_activity(
        self,
        *,
        role: str | None = None,
        since_ts: float | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        return self._storage.list_worker_activity(role=role, since_ts=since_ts, limit=limit)

    def _max_parallel_by_role(self) -> dict[str, int]:
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
        for key, cfg in self._profiles.items():
            role = cfg.get("role") or key
            try:
                out[role] = int(cfg.get("max_parallel_jobs", out.get(role, 1)))
            except Exception:
                pass
        return out
