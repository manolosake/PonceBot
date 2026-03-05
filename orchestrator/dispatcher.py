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
_INTENT_TYPES = {"query", "order_project_new", "order_project_change"}

_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
_TASK_VERB_RE = re.compile(
    # Conservative "do/change/build" intent. If present, treat as work even if the text contains
    # words like "status"/"estado" (which otherwise cause false positives).
    r"\b("
    r"arregl\w*|corrig\w*|cambi\w*|modific\w*|ajust\w*|"
    r"agreg\w*|anad\w*|añad\w*|quit\w*|elimin\w*|actualiz\w*|mejor\w*|"
    r"implement\w*|crea\w*|crear\w*|haz|hagan|"
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
    for k in _ROLE_KEYWORDS.get(role, ()):  # pragma: no branch
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
    # Conversational statements (non-imperative) should stay in Jarvis query lane.
    if any(t.startswith(k) for k in ("creo que ", "pienso que ", "me parece ", "considero que ", "i think ")):
        return "query"
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


def detect_ceo_intent(text: str, *, reply_context: dict[str, Any] | None = None) -> str:
    """
    Classify a top-level CEO message into:
    - query
    - order_project_new
    - order_project_change

    Grounded rule set:
    - Queries never create/advance CEO orders.
    - Replies/corrections default to `order_project_change`.
    - New orders/projects require explicit project scope signals.
    - Plain one-shot asks (without project scope) stay in conversational lane.
    """
    req_type = detect_request_type(text)
    if req_type in ("query", "status"):
        return "query"

    raw = (text or "").strip().lower()

    project_scope_terms = (
        "project",
        "proyecto",
        "repo",
        "repository",
        "dashboard",
        "feature",
        "branch",
        "pull request",
        "merge",
        "release",
        "app",
        "application",
        "backend",
        "frontend",
        "api",
        "database",
        "db ",
        "schema",
        "migration",
        "migracion",
        "migración",
        "deploy",
        "infra",
        "infrastructure",
    )
    has_project_scope = any(k in raw for k in project_scope_terms)

    has_reply = bool(reply_context and isinstance(reply_context, dict) and any(reply_context.values()))
    if has_reply:
        return "order_project_change"

    has_change_signal = any(
        k in raw
        for k in (
            "change ",
            "update ",
            "modify ",
            "adjust ",
            "fix this",
            "sobre eso",
            "de eso",
            "ajusta",
            "corrige",
            "modifica",
            "actualiza",
            "cambia",
            "iterate",
            "iteration",
            "follow up",
        )
    )
    if has_change_signal and has_project_scope:
        return "order_project_change"

    if any(
        k in raw
        for k in (
            "new project",
            "nuevo proyecto",
            "start project",
            "inicia proyecto",
            "create project",
            "crear proyecto",
            "new order",
            "nueva orden",
            "build from scratch",
            "desde cero",
        )
    ):
        return "order_project_new"

    ideation_signal = any(
        k in raw
        for k in (
            "ideas",
            "idea ",
            "propuestas",
            "suggest",
            "suggestion",
            "brainstorm",
            "que proyectos",
            "qué proyectos",
            "proyectos interesantes",
            "podemos hacer",
        )
    )
    execution_signal = any(
        k in raw
        for k in (
            "implement",
            "build",
            "develop",
            "create",
            "ship",
            "deploy",
            "setup",
            "set up",
            "execute",
            "run",
            "haz ",
            "quiero que",
            "agrega",
            "añade",
            "anade",
            "pon ",
            "ejecuta",
            "implementa",
            "desarrolla",
            "construye",
            "crea",
            "inicia",
            "asigna",
            "deleg",
        )
    )
    if ideation_signal and not execution_signal:
        return "query"

    # Safety: only create a persistent project/order when there is explicit execution intent.
    if has_project_scope and req_type in ("task", "maintenance", "review") and execution_signal:
        return "order_project_new"

    return "query"


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

    override_request_type = str(normalized_context.get("request_type") or "").strip().lower()
    request_type = override_request_type or detect_request_type(text)
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

    trace = dict(normalized_context.get("trace") or {})
    intent_type = str(trace.get("intent_type") or normalized_context.get("intent_type") or "").strip().lower()
    if intent_type in _INTENT_TYPES:
        trace["intent_type"] = intent_type

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
        trace=trace,
    )
