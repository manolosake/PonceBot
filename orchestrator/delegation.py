from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json


_ALLOWED_ROLES = ("ceo", "frontend", "backend", "qa", "sre")
_ALLOWED_MODES = ("ro", "rw", "full")


@dataclass(frozen=True)
class TaskSpec:
    key: str
    role: str
    text: str
    mode_hint: str = "ro"
    priority: int = 2
    depends_on: list[str] = field(default_factory=list)
    requires_approval: bool = False


def parse_ceo_subtasks(structured: dict[str, Any] | str | None) -> list[TaskSpec]:
    """
    Parse the CEO structured output into TaskSpec entries.

    Expects a dict with a 'subtasks' list. If a string is provided, attempts to parse JSON.
    """
    if structured is None:
        return []
    payload: Any = structured
    if isinstance(structured, str):
        s = structured.strip()
        if not s:
            return []
        try:
            payload = json.loads(s)
        except Exception:
            return []
    if not isinstance(payload, dict):
        return []
    items = payload.get("subtasks")
    if not isinstance(items, list):
        return []

    out: list[TaskSpec] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("key") or "").strip()
        role = str(raw.get("role") or "").strip().lower()
        text = str(raw.get("text") or "").strip()
        if not key or not text:
            continue
        if role not in _ALLOWED_ROLES:
            continue
        mode = str(raw.get("mode_hint") or "ro").strip().lower()
        if mode not in _ALLOWED_MODES:
            mode = "ro"
        try:
            priority = int(raw.get("priority") or 2)
        except Exception:
            priority = 2
        if priority < 1:
            priority = 1
        if priority > 3:
            priority = 3

        deps_raw = raw.get("depends_on") or []
        deps: list[str] = []
        if isinstance(deps_raw, list):
            deps = [str(x).strip() for x in deps_raw if str(x).strip()]

        requires_approval = bool(raw.get("requires_approval", False))
        if mode == "full":
            requires_approval = True

        out.append(
            TaskSpec(
                key=key,
                role=role,
                text=text,
                mode_hint=mode,
                priority=priority,
                depends_on=deps,
                requires_approval=requires_approval,
            )
        )
    return out

