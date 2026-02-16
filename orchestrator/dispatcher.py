from __future__ import annotations

from typing import Any
import re

from .schemas.task import Task


_ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    # Jarvis role (formerly "orchestrator"/"ceo"). This is the coordinator agent; the human is the CEO.
    "jarvis": (
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
        "jarvis",
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
    "product_ops": (
        "acceptance",
        "criter",
        "scope",
        "mvp",
        "metrics",
        "product",
        "requirements",
        "spec",
    ),
    "security": (
        "security",
        "secur",
        "secret",
        "token",
        "leak",
        "vuln",
        "rbac",
        "auth",
        "ssrf",
    ),
    "research": (
        "research",
        "state of the art",
        "sota",
        "benchmark",
        "paper",
        "openclaw",
        "gap",
    ),
    "release_mgr": (
        "release",
        "merge",
        "branch",
        "qa gate",
        "deploy",
        "changelog",
        "version",
    ),
}

_REQUEST_TYPES = {"status", "query", "review", "maintenance", "task"}

_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
_TASK_VERB_RE = re.compile(
    # Conservative "do/change/build" intent. If present, treat as work even if the text contains
    # words like "status"/"estado" (which otherwise cause false positives).
    r"\b("
    r"arregl\w*|corrig\w*|cambi\w*|modific\w*|ajust\w*|"
    r"agreg\w*|anad\w*|añad\w*|quit\w*|elimin\w*|"
    r"implement\w*|cre\w*|haz|hagan|"
    r"fix\w*|remove\w*|add\w*|update\w*|format\w*|refactor\w*|improv\w*"
    r")\b",
    flags=re.IGNORECASE,
)


def _score_role(text_l: str, role: str) -> int:
    """
    Keyword scoring is intentionally conservative.

    Grounded motivation: naive substring checks cause false positives in Spanish
    (e.g. "quien" contains "ui"). For short keywords (<=3 chars), require token match.
    """
    score = 0
    tokens: set[str] | None = None
    for k in _ROLE_KEYWORDS.get(role, ()):
        kk = (k or "").strip().lower()
        if not kk:
            continue
        if " " in kk:
            if kk in text_l:
                score += 2
            continue
        if len(kk) <= 3:
            if tokens is None:
                tokens = set(_TOKEN_RE.findall(text_l))
            if kk in tokens:
                score += 2
            continue
        if kk in text_l:
            score += 1
    return score


def _explicit_role(text_l: str) -> str | None:
    tl = (text_l or "").lower()

    # Explicit jarvis markers (support legacy @orchestrator/@ceo aliases).
    if "@jarvis" in tl or "@orchestrator" in tl or "@ceo" in tl:
        return "jarvis"

    for role in ("frontend", "backend", "qa", "sre", "product_ops", "security", "research", "release_mgr"):
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

    def _looks_like_task() -> bool:
        # "I want X changed" should be treated as work, not as a status/query fast-path.
        if "quiero que " in t or "i want " in t:
            return True
        return _TASK_VERB_RE.search(t) is not None

    def _looks_like_status() -> bool:
        # Explicit command.
        if t.startswith("/status"):
            return True

        # Natural-language status checks (Spanish + English).
        if any(
            k in t
            for k in (
                "estan trabajando",
                "están trabajando",
                "siguen trabajando",
                "ya acabaron",
                "ya terminaron",
                "ya termino",
                "ya terminó",
                "en que van",
                "en qué van",
                "progreso",
                "avance",
                "que estan haciendo",
                "qué están haciendo",
                "que están haciendo",
                "que esta haciendo",
                "qué está haciendo",
                "que está haciendo",
                "haciendo el equipo",
                "que hacen",
                "qué hacen",
                "still running",
                "are you done",
                "did you finish",
            )
        ):
            return True

        # Token-based: avoid substring false positives ("job role status", etc).
        tokens = set(_TOKEN_RE.findall(t))
        if "status" in tokens or "estado" in tokens:
            # Status-like when it references the system/service/host.
            if any(w in t for w in ("servidor", "server", "host", "service", "servicio", "bot")):
                return True
            # Also treat the bare "status"/"estado" as a status request.
            bare = t.strip(" ?!.")
            if bare in ("status", "estado"):
                return True
        return False

    # IMPORTANT: task intent wins over status keywords.
    if _looks_like_task():
        if "revis" in t or "review" in t:
            return "review"
        if "mantenimiento" in t or "cron" in t or "monitor" in t:
            return "maintenance"
        return "task"

    if _looks_like_status():
        return "status"
    # Queries (CEO questions) that should not auto-delegate.
    if any(
        k in t
        for k in (
            "quien soy",
            "quién soy",
            "who am i",
            "que tienes pendiente",
            "qué tienes pendiente",
            "pendientes",
            "backlog",
            "what's pending",
            "whats pending",
            "cuantos empleados",
            "cuántos empleados",
            "cuantos trabajadores",
            "equipo tenemos",
            "a quien tenemos en el equipo",
            "qué modelos",
            "que modelos",
            "modelo usan",
            "modelos usan",
            "que es sre",
            "qué es sre",
        )
    ):
        return "query"
    if "?" in t and not any(k in t for k in ("haz ", "implement", "crea ", "arregla", "build", "deploy", "refactor", "agrega ")):
        return "query"
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
    return {"jarvis": "high", "qa": "high", "sre": "high"}.get(role, "medium")


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

    # Legacy aliases: if callers still pass role=ceo/orchestrator, normalize.
    if role in ("ceo", "orchestrator"):
        role = "jarvis"

    if role not in ("jarvis", "frontend", "backend", "qa", "sre", "product_ops", "security", "research", "release_mgr"):
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
