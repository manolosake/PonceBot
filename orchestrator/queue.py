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

    def set_job_approved(self, job_id: str, approved: bool = True) -> bool:
        return self._storage.set_job_approved(job_id, approved=approved)

    def clear_job_approval(self, job_id: str) -> bool:
        return self._storage.clear_job_approval(job_id)

    def get_role_health(self) -> dict[str, dict[str, int]]:
        return self._storage.get_role_health()

    def get_job(self, job_id: str) -> Task | None:
        return self._storage.get_job(job_id)

    def get_queued_count(self) -> int:
        return self._storage.queued_count()

    def get_running_count(self) -> int:
        return self._storage.running_count()

    def jobs_by_state(self, *, state: str, limit: int = 50) -> list[Task]:
        return self._storage.jobs_by_state(state=state, limit=limit)

    def _max_parallel_by_role(self) -> dict[str, int]:
        out: dict[str, int] = {
            "ceo": 1,
            "frontend": 2,
            "backend": 2,
            "qa": 2,
            "sre": 2,
        }
        for key, cfg in self._profiles.items():
            role = cfg.get("role") or key
            try:
                out[role] = int(cfg.get("max_parallel_jobs", out.get(role, 1)))
            except Exception:
                pass
        return out
