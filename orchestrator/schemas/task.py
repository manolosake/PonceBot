from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import time
import uuid


JobRole = str
RequestType = str
JobState = str


@dataclass(frozen=True)
class Task:
    job_id: str
    source: str
    role: JobRole
    input_text: str
    request_type: RequestType
    priority: int
    model: str
    effort: str
    mode_hint: str
    requires_approval: bool
    max_cost_window_usd: float
    created_at: float
    updated_at: float
    due_at: float | None
    state: JobState
    chat_id: int
    user_id: int | None
    reply_to_message_id: int | None
    trace: dict[str, str | int | float | bool | list[str]] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        source: str,
        role: str,
        input_text: str,
        request_type: str,
        priority: int,
        model: str,
        effort: str,
        mode_hint: str,
        requires_approval: bool,
        max_cost_window_usd: float,
        chat_id: int,
        user_id: int | None = None,
        reply_to_message_id: int | None = None,
        due_at: float | None = None,
        state: str = "queued",
        trace: dict[str, str | int | float | bool | list[str]] | None = None,
        job_id: str | None = None,
        created_at: float | None = None,
    ) -> "Task":
        now = float(created_at if created_at is not None else time.time())
        return cls(
            job_id=job_id or str(uuid.uuid4()),
            source=source,
            role=role,
            input_text=input_text,
            request_type=request_type,
            priority=priority,
            model=model,
            effort=effort,
            mode_hint=mode_hint,
            requires_approval=requires_approval,
            max_cost_window_usd=float(max_cost_window_usd),
            created_at=now,
            updated_at=now,
            due_at=due_at,
            state=state,
            chat_id=chat_id,
            user_id=user_id,
            reply_to_message_id=reply_to_message_id,
            trace=trace or {},
        )

    def with_updates(self, **kwargs: Any) -> "Task":
        return Task(
            job_id=kwargs.get("job_id", self.job_id),
            source=kwargs.get("source", self.source),
            role=kwargs.get("role", self.role),
            input_text=kwargs.get("input_text", self.input_text),
            request_type=kwargs.get("request_type", self.request_type),
            priority=kwargs.get("priority", self.priority),
            model=kwargs.get("model", self.model),
            effort=kwargs.get("effort", self.effort),
            mode_hint=kwargs.get("mode_hint", self.mode_hint),
            requires_approval=kwargs.get("requires_approval", self.requires_approval),
            max_cost_window_usd=kwargs.get("max_cost_window_usd", self.max_cost_window_usd),
            created_at=kwargs.get("created_at", self.created_at),
            updated_at=kwargs.get("updated_at", time.time()),
            due_at=kwargs.get("due_at", self.due_at),
            state=kwargs.get("state", self.state),
            chat_id=kwargs.get("chat_id", self.chat_id),
            user_id=kwargs.get("user_id", self.user_id),
            reply_to_message_id=kwargs.get("reply_to_message_id", self.reply_to_message_id),
            trace=kwargs.get("trace", dict(self.trace)),
        )

    def trace_json(self) -> str:
        return json.dumps(self.trace or {}, ensure_ascii=False)

    @classmethod
    def from_trace_json(cls, value: str | None) -> dict[str, Any]:
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
        return {}
