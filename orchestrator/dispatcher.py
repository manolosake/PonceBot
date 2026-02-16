from __future__ import annotations

from typing import Any

from .schemas.task import Task


_ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    # Orchestrator role (formerly "ceo"). This is the coordinator agent; the human is the CEO.
    "orchestrator": (
        "status",
        "plan",
        "coordin",
        "prior",
        "report",
        "resumen",
        "siguiente",
        "idea",
        "decid",
        "estrateg",
        "orquest",
    ),
    "frontend": (
        "ui",
        "ux",
        "frontend",
        "front",
        "screenshot",
        "visual",
        "css",
        "html",
        "componente",
        "pantalla",
        "diseño",
    ),
    "backend": (
        "api",
        "endpoint",
        "base de datos",
        "db",
        "schema",
        "servidor",
        "backend",
        "codex",
        "python",
        "error",
        "bug",
        "refactor",
        "service",
    ),
    "qa": (
        "test",
        "prueba",
        "qa",
        "coverage",
        "assert",
        "bugs",
        "regresi",
        "fail",
        "lint",
    ),
    "sre": (
        "deploy",
        "systemd",
        "servicio",
        "service",
        "monitor",
        "cpu",
        "mem",
        "disco",
        "storage",
        "health",
        "log",
        "alert",
    ),
}

_REQUEST_TYPES = {"status", "review", "maintenance", "task"}


def _score_role(text_l: str, role: str) -> int:
    score = 0
    for k in _ROLE_KEYWORDS.get(role, ()):
        if k in text_l:
            score += 1
    return score


def _explicit_role(text_l: str) -> str | None:
    tl = (text_l or "").lower()

    # Explicit orchestrator markers (support legacy @ceo alias).
    if "@orchestrator" in tl or "@ceo" in tl:
        return "orchestrator"

    for role in ("frontend", "backend", "qa", "sre"):
        marker = f"@{role}"
        if marker in tl:
            return role
    return None


def detect_role(text: str, *, default_role: str = "backend") -> str:
    text_l = (text or "").lower()
    direct = _explicit_role(text_l)
    if direct:
        return direct

    scored = {role: _score_role(text_l, role) for role in _ROLE_KEYWORDS}
    best = max(scored.items(), key=lambda kv: kv[1], default=("backend", 0))
    if best[1] > 0:
        return best[0]
    return (default_role or "backend").strip().lower() or "backend"


def detect_request_type(text_l: str) -> str:
    t = (text_l or "").lower()
    if t.startswith("/status") or "estado" in t or "status" in t:
        return "status"
    if "revis" in t or "review" in t:
        return "review"
    if "mantenimiento" in t or "cron" in t or "monitor" in t:
        return "maintenance"
    return "task"


def choose_model_by_role(role: str, model_override: str | None = None, default_model: str = "gpt-4.1") -> str:
    if model_override:
        return model_override.strip()
    return default_model


def choose_effort_by_role(role: str) -> str:
    return {"orchestrator": "high", "qa": "high", "sre": "high"}.get(role, "medium")


def choose_priority(text: str) -> int:
    t = (text or "").lower()
    if any(w in t for w in ("urgent", "urgente", "crítico", "bloque", "bloqueo", "prod")):
        return 1
    if any(w in t for w in ("importante", "necesito", "review")):
        return 2
    return 2


def to_task(
    text: str,
    *,
    context: dict[str, Any],
) -> Task:
    # text is expected to be already normalised.
    normalized_context = context or {}
    source = normalized_context.get("source", "telegram")
    chat_id = int(normalized_context.get("chat_id", 0))
    user_id = normalized_context.get("user_id")
    reply_to_message_id = normalized_context.get("reply_to_message_id")
    model = str(normalized_context.get("model", "gpt-5.2"))
    effort = str(normalized_context.get("effort", "medium"))
    default_role = str(normalized_context.get("default_role") or "backend").strip().lower() or "backend"

    # Allow explicit @role markers to override any context-provided default.
    explicit = _explicit_role((text or "").lower())
    role = explicit or normalized_context.get("role") or detect_role(text, default_role=default_role)

    # Legacy alias: if callers still pass role=ceo, normalize.
    if role == "ceo":
        role = "orchestrator"

    if role not in ("orchestrator", "frontend", "backend", "qa", "sre"):
        role = "backend"

    request_type = detect_request_type(text)
    if request_type not in _REQUEST_TYPES:
        request_type = "task"

    due_at = normalized_context.get("due_at")
    priority = int(normalized_context.get("priority", choose_priority(text)))
    mode_hint = str(normalized_context.get("mode_hint", "ro"))
    if mode_hint not in ("ro", "rw", "full"):
        mode_hint = "ro"

    max_cost_window_usd = float(normalized_context.get("max_cost_window_usd", 8.0))
    requires_approval = bool(normalized_context.get("requires_approval", False))
    if mode_hint == "full":
        requires_approval = True

    return Task.new(
        source=str(source),
        role=str(role),
        input_text=(text or "").strip(),
        request_type=str(request_type),
        priority=priority,
        model=str(model),
        effort=str(effort),
        mode_hint=mode_hint,
        requires_approval=requires_approval,
        max_cost_window_usd=max_cost_window_usd,
        chat_id=chat_id,
        user_id=user_id,
        reply_to_message_id=reply_to_message_id,
        due_at=due_at,
        trace=dict(normalized_context.get("trace") or {}),
    )
