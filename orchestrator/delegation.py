from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re
import json


_ALLOWED_ROLES = ("jarvis", "frontend", "backend", "qa", "sre", "product_ops", "security", "research", "release_mgr", "architect_local", "implementer_local", "reviewer_local")
_ALLOWED_MODES = ("ro", "rw", "full")
_ALLOWED_SLA = ("normal", "high", "urgent")


@dataclass(frozen=True)
class TaskSpec:
    key: str
    role: str
    text: str
    # Empty string means "use the role profile default" (safer than guessing here).
    mode_hint: str = ""
    priority: int = 2
    depends_on: list[str] = field(default_factory=list)
    requires_approval: bool = False
    acceptance_criteria: list[str] = field(default_factory=list)
    definition_of_done: list[str] = field(default_factory=list)
    eta_minutes: int | None = None
    sla_tier: str = ""


def _to_string_list(value: Any, *, max_items: int = 8, max_chars: int = 220) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in value:
        s = str(raw or "").strip()
        if not s:
            continue
        if len(s) > max_chars:
            s = s[:max_chars].rstrip() + "..."
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def parse_jarvis_subtasks(structured: dict[str, Any] | str | None) -> list[TaskSpec]:
    """Parse Jarvis structured output into TaskSpec entries.

    Expects a dict with a subtasks list. If a string is provided, attempts to parse JSON.

    Back-compat aliases:
    - role="ceo" -> role="jarvis"
    - role="orchestrator" -> role="jarvis"
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
        if role in ("ceo", "orchestrator"):
            role = "jarvis"
        if role not in _ALLOWED_ROLES:
            continue

        # mode_hint is optional. If omitted, we let the runner apply role profile defaults.
        if "mode_hint" in raw:
            mode = str(raw.get("mode_hint") or "").strip().lower()
            if mode not in _ALLOWED_MODES:
                mode = "ro"
        else:
            mode = ""

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

        acceptance = _to_string_list(raw.get("acceptance_criteria") or raw.get("acceptance") or [])
        dod = _to_string_list(raw.get("definition_of_done") or raw.get("dod") or [])

        eta_minutes: int | None = None
        try:
            if raw.get("eta_minutes") is not None:
                eta_minutes = int(raw.get("eta_minutes"))
        except Exception:
            eta_minutes = None
        if eta_minutes is not None:
            eta_minutes = max(5, min(7 * 24 * 60, int(eta_minutes)))

        sla_tier = str(raw.get("sla_tier") or "").strip().lower()
        if sla_tier not in _ALLOWED_SLA:
            sla_tier = ""

        out.append(
            TaskSpec(
                key=key,
                role=role,
                text=text,
                mode_hint=mode,
                priority=priority,
                depends_on=deps,
                requires_approval=requires_approval,
                acceptance_criteria=acceptance,
                definition_of_done=dod,
                eta_minutes=eta_minutes,
                sla_tier=sla_tier,
            )
        )
    return out


def _extract_reseed_text(structured: dict[str, Any] | str | None) -> str:
    if structured is None:
        return ""
    if isinstance(structured, str):
        return structured
    try:
        return json.dumps(structured)
    except Exception:
        return str(structured)


def parse_orchestrator_subtasks(structured: dict[str, Any] | str | None) -> list[TaskSpec]:
    parsed = parse_jarvis_subtasks(structured)
    if parsed:
        return parsed

    raw_text = _extract_reseed_text(structured).lower()
    if "proactive_idle_watchdog" not in raw_text and "proactive_local_reseed" not in raw_text:
        return parsed

    ticket_id = ""
    try:
        match = re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", raw_text)
        if match:
            ticket_id = match.group(0)
    except Exception:
        ticket_id = ""

    ticket_line = f" Ticket: {ticket_id}." if ticket_id else ""
    text = (
        "Provide a single, bounded implementer-ready slice for proactive idle watchdog reseed."
        " Include exact file(s), one concrete change, validation command, and risks."
        f"{ticket_line}"
    )

    return [
        TaskSpec(
            key="architect_local_reseed",
            role="architect_local",
            text=text,
            mode_hint="ro",
            priority=2,
            acceptance_criteria=["Return one implementer-ready slice with file-scoped change and validation."],
            definition_of_done=["Implementer can apply a single bounded change without extra planning."],
        )
    ]


# Backwards-compatible alias; do not document.
parse_ceo_subtasks = parse_jarvis_subtasks
