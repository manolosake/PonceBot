#!/usr/bin/env python3
"""
Telegram -> Codex bridge ("codexbot").

Design goals:
- No third-party Python deps (stdlib only).
- Long-polling Telegram Bot API (no public webhook endpoint required).
- Runs Codex CLI non-interactively, using a local OSS provider (Ollama by default).
- Safety by default: plain messages run in read-only sandbox; writes require /rw.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import queue
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import http.client
import ssl
import signal
import urllib.error
import urllib.parse
import urllib.request
import base64
import hashlib
import hmac
import secrets
from html import escape as _html_escape
import dataclasses
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.agents import load_agent_profiles
from orchestrator.dispatcher import detect_request_type, to_task
from orchestrator.delegation import parse_orchestrator_subtasks
from orchestrator.prompting import build_agent_prompt
from orchestrator.queue import OrchestratorQueue
from orchestrator.runbooks import Runbook, due as runbook_due, load_runbooks, to_task as runbook_to_task
from orchestrator.schemas.task import Task
from orchestrator.screenshot import (
    Viewport,
    capture as capture_screenshot,
    capture_html as capture_screenshot_html,
    capture_html_file as capture_screenshot_html_file,
    validate_screenshot_url,
)
from orchestrator.storage import SQLiteTaskStorage
from orchestrator.scheduler import OrchestratorScheduler
from orchestrator.runner import run_task as run_orchestrator_task
from orchestrator.workspaces import WorktreeLease, collect_git_artifacts, ensure_worktree_pool, prepare_clean_workspace
from orchestrator.status_service import StatusService
from orchestrator.status_http import start_status_http_server

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


LOG = logging.getLogger("codexbot")


TELEGRAM_MSG_LIMIT = 4096


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SKILL_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")

# Ticket card UX: keep one editable message per ticket (no spam).
_TICKET_CARD_LOCK = threading.Lock()
_TICKET_CARD_LAST_EDIT: dict[tuple[int, str], float] = {}
_TICKET_CARD_MIN_EDIT_INTERVAL_S = 2.0


def _codex_home_dir() -> Path:
    """
    Resolve CODEX_HOME (default: ~/.codex). Keep this in sync with Codex CLI conventions.
    """
    v = os.environ.get("CODEX_HOME", "").strip()
    base = Path(v).expanduser() if v else (Path.home() / ".codex")
    try:
        return base.resolve()
    except Exception:
        return base


def _skills_root_dir() -> Path:
    return _codex_home_dir() / "skills"


def _skill_segment_ok(name: str) -> bool:
    """
    Only allow simple skill names like: imagegen, gh-fix-ci, notion-research-documentation.
    (No slashes, no '..', no weird whitespace.)
    """
    s = (name or "").strip()
    if not s or s in (".", ".."):
        return False
    if s != Path(s).name:
        return False
    return bool(_SKILL_SEGMENT_RE.fullmatch(s))


def _safe_filename(name: str, *, fallback: str = "upload.bin", max_len: int = 120) -> str:
    """
    Sanitize a filename so it is safe to use on disk and inside prompts.
    Keeps ASCII alnum + '._-' and replaces everything else with '_'.
    """
    base = (name or "").strip()
    base = Path(base).name  # drop any path parts
    base = _SAFE_FILENAME_RE.sub("_", base).strip("._-")
    if not base:
        base = fallback
    if max_len > 0 and len(base) > max_len:
        base = base[:max_len]
    return base


def _normalize_slash_aliases(text: str) -> str:
    """
    Human-friendly aliases:
    - /reset -> /new
    - /m -> /model
    - /p -> /permissions
    - /s -> /status
    - /x -> /cancel
    - /v -> /voice
    """
    t = (text or "").strip()
    if not t.startswith("/"):
        return text

    # Telegram can include the bot username in group commands: /cmd@BotName args...
    # Strip it so commands work consistently everywhere.
    if "@" in t:
        head, sep, tail = t.partition(" ")
        if "@" in head:
            head = head.split("@", 1)[0]
            t = head + (sep + tail if sep else "")

    if t == "/reset":
        return "/new"
    if t == "/m":
        return "/model"
    if t.startswith("/m "):
        return "/model " + t[len("/m ") :]
    if t == "/p":
        return "/permissions"
    if t.startswith("/p "):
        return "/permissions " + t[len("/p ") :]
    if t == "/s":
        return "/status"
    if t == "/x":
        return "/cancel"
    if t == "/v":
        return "/voice"
    if t.startswith("/v "):
        return "/voice " + t[len("/v ") :]
    return text


_GREETING_PREFIXES = (
    "hola",
    "buenos dias",
    "buenos días",
    "buenas tardes",
    "buenas noches",
    "buen dia",
    "buenas",
    "hey",
    "hi",
    "hello",
    "hello,",
    "hi,",
    "hey,",
    "hola,",
)


def _is_greeting(text: str) -> bool:
    """
    Identify short pleasantries that should be answered immediately.

    This avoids enqueueing tiny greetings as full Jarvis tasks and avoids showing
    only a ticket card for a plain "hola" message.
    """
    raw = (text or "").strip()
    if not raw:
        return False

    t = raw.lower()
    # Keep it strict: short text or explicit greeting prefixes only.
    compact = re.sub(r"\s+", " ", t).strip(" .,!¿?¡")
    if not compact:
        return False

    if compact in _GREETING_PREFIXES:
        return True
    for pfx in _GREETING_PREFIXES:
        if compact.startswith(f"{pfx} "):
            return True
    return False


def _is_purge_queue_request(text: str) -> bool:
    """
    Detect "clear the queue" intents in natural language.

    Grounded motivation: CEO control-plane actions should be deterministic and immediate
    (no Codex, no delegation, no ticket spam).
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if raw.startswith("/"):
        return False

    t = re.sub(r"\s+", " ", raw.lower()).strip(" .,!¿?¡")
    if not t:
        return False

    # Must reference the queue/backlog explicitly to avoid accidental triggers.
    if not any(w in t for w in ("cola", "queue", "backlog")):
        return False

    # Require an action verb.
    return any(
        w in t
        for w in (
            "limpia",
            "limpiar",
            "vacia",
            "vacía",
            "vaciar",
            "purga",
            "purgar",
            "borra",
            "borrar",
            "clear",
            "purge",
        )
    )


def _humanize_orchestrator_role(role: str) -> str:
    """Human-readable role label for bot responses."""
    r = (role or "").strip().lower()
    if r == "jarvis":
        return "Jarvis"
    if r == "product_ops":
        return "Product Ops"
    if r == "release_mgr":
        return "Release Manager"
    return " ".join(part.capitalize() for part in re.split(r"[_-]", r) if part) or "Jarvis"


def _parse_employee_forward(text: str) -> tuple[str | None, str]:
    """
    Optional "Empleado:" forwarded-message mode.

    Supported formats (robust to extra ':' and newlines):
    - "Empleado: Juan Perez\\n<message...>"
    - "Empleado: Juan Perez: <message...>"
    - "Empleado:\\nJuan Perez\\n<message...>"

    Returns (employee_name_or_none, message_text).
    If no employee forwarding is detected, returns (None, original_text).
    """
    raw = text if isinstance(text, str) else ""
    if not raw:
        return None, ""

    # Only trigger on an explicit label to avoid breaking common "Riesgo: ..." style inputs.
    stripped = raw.lstrip()
    low = stripped.lower()
    if not (low.startswith("empleado:") or low.startswith("employee:")):
        return None, raw

    # Split the first line as "Empleado: <rest...>".
    first_line, _sep_nl, rest_lines = stripped.partition("\n")
    _label, _sep_colon, after = first_line.partition(":")
    after = after.strip()

    # If the first line has no payload, treat the next non-empty line as the employee name.
    if not after:
        lines = rest_lines.splitlines()
        name = ""
        idx = 0
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx < len(lines):
            name = lines[idx].strip()
            msg = "\n".join(lines[idx + 1 :])
        else:
            msg = ""
        return (name or None), msg

    # If the first line contains an additional ":" then interpret it as "name: message".
    name, c2, msg_inline = after.partition(":")
    name = name.strip()
    msg_inline = msg_inline.lstrip() if c2 else ""

    msg = msg_inline
    if rest_lines:
        msg = (msg + ("\n" if msg else "") + rest_lines) if msg is not None else rest_lines
    return (name or None), (msg or "")


def _markdownish_to_telegram_html(text: str) -> str:
    """
    Convert a minimal, safe subset of markdown-ish text to Telegram HTML:
    - Triple backtick fences -> <pre><code>...</code></pre>
    - Inline `code` -> <code>...</code>
    Everything else is HTML-escaped.
    """
    raw = text or ""
    parts = raw.split("```")
    out_parts: list[str] = []
    for i, part in enumerate(parts):
        is_code_block = (i % 2) == 1
        if is_code_block:
            # Drop an optional language tag on the first line: ```python\n...
            if "\n" in part:
                first, rest = part.split("\n", 1)
                if re.fullmatch(r"[A-Za-z0-9_+-]{1,20}", (first or "").strip() or ""):
                    part = rest
            out_parts.append("<pre><code>" + _html_escape(part) + "</code></pre>")
            continue

        # Normal text: handle inline code first so emphasis/link markup doesn't apply inside <code>.
        inline_parts = part.split("`")
        rebuilt: list[str] = []
        for j, p in enumerate(inline_parts):
            if (j % 2) == 1:
                rebuilt.append("<code>" + _html_escape(p) + "</code>")
            else:
                rebuilt.append(_apply_simple_markup(_html_escape(p)))
        out_parts.append("".join(rebuilt))
    return "".join(out_parts)


def _apply_simple_markup(escaped_text: str) -> str:
    """
    Apply a small subset of markdown-ish formatting on already-escaped text:
    - # Heading -> <b>Heading</b>
    - Bullet lists (- / *) -> •
    - [label](https://url) -> <a href="url">label</a>
    - **bold** -> <b>bold</b>
    - *italic* / _italic_ -> <i>italic</i>
    Conservative by design: doesn't span newlines.
    """
    s = escaped_text or ""

    # Headings and list markers: do this line-wise.
    lines: list[str] = []
    for ln in s.splitlines(keepends=True):
        nl = "\n" if ln.endswith("\n") else ""
        body = ln[:-1] if nl else ln

        m = re.match(r"^(#{1,6})\s+(.*)$", body)
        if m:
            body = "<b>" + (m.group(2) or "").strip() + "</b>"
            lines.append(body + nl)
            continue

        if body.startswith("- "):
            body = "• " + body[2:]
        elif body.startswith("* "):
            body = "• " + body[2:]

        lines.append(body + nl)

    s = "".join(lines)

    # Links: [label](https://url)
    s = re.sub(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', s)

    # Bold: **...**
    s = re.sub(r"\*\*([^\n]+?)\*\*", r"<b>\1</b>", s)

    # Italic: *...* but avoid **...**
    s = re.sub(r"(?<!\*)\*([^\n]+?)\*(?!\*)", r"<i>\1</i>", s)

    # Italic: _..._ (avoid matching within words like foo_bar_baz)
    s = re.sub(r"(?<![A-Za-z0-9])_([^\n]+?)_(?![A-Za-z0-9])", r"<i>\1</i>", s)

    return s


def _state_access_mode(cfg: "BotConfig", *, chat_id: int | None = None) -> str:
    """
    Returns one of: "", "default", "full".
    Stored in cfg.state_file as:
    - legacy/global: {"access_mode": "..."}
    - per-chat: {"access_mode_by_chat": {"<chat_id>": "default|full"}}
    """
    st = _get_state(cfg)
    if chat_id is not None:
        by_chat = st.get("access_mode_by_chat")
        if isinstance(by_chat, dict):
            v = by_chat.get(str(int(chat_id)))
            vv = v.strip().lower() if isinstance(v, str) else ""
            if vv in ("default", "full"):
                return vv
    v = st.get("access_mode")
    v = v.strip().lower() if isinstance(v, str) else ""
    return v if v in ("default", "full") else ""


def _effective_bypass_sandbox(cfg: "BotConfig", *, chat_id: int | None = None) -> bool:
    """
    True means we pass `--dangerously-bypass-approvals-and-sandbox`.
    State override (if set) wins over env config.
    """
    mode = _state_access_mode(cfg, chat_id=chat_id)
    if mode == "default":
        return False
    if mode == "full":
        return True
    return bool(cfg.codex_dangerous_bypass_sandbox)


def _set_access_mode(cfg: "BotConfig", mode: str | None, *, chat_id: int | None = None) -> None:
    """
    Persist access mode override to cfg.state_file.
    mode=None clears the override.
    """
    st = _get_state(cfg)
    if chat_id is None:
        if mode is None:
            st.pop("access_mode", None)
        else:
            st["access_mode"] = mode
    else:
        by_chat = st.get("access_mode_by_chat")
        if not isinstance(by_chat, dict):
            by_chat = {}
        key = str(int(chat_id))
        if mode is None:
            by_chat.pop(key, None)
        else:
            by_chat[key] = mode
        st["access_mode_by_chat"] = by_chat
    _atomic_write_json(cfg.state_file, st)


def _permissions_text(cfg: "BotConfig", *, chat_id: int | None = None) -> str:
    # Mirror the Codex CLI picker labels, but also show how to set it from Telegram.
    bypass = _effective_bypass_sandbox(cfg, chat_id=chat_id)
    default_line = "- Default (current)" if not bypass else "- Default"
    full_line = "- Full access (current)" if bypass else "- Full access"
    return "\n".join(
        [
            "Dos opciones:",
            default_line,
            full_line,
            "",
            "Uso:",
            "- /permissions default",
            "- /permissions full",
            "- /permissions clear",
        ]
    )


def _format_preview_text() -> str:
    # Intentionally uses markdown-ish backticks/fences; TelegramAPI converts to HTML when enabled.
    return "\n".join(
        [
            "# Ejemplo",
            "",
            "**Esto debe verse bien en Telegram.**",
            "",
            "**Titulo:** # Titulo",
            "",
            "**Estilos:** **negrita**, *cursiva*, y _cursiva_.",
            "",
            "**Lista:**",
            "- Item 1",
            "- Item 2",
            "",
            "**Link:** [OpenAI](https://openai.com)",
            "",
            "**Inline:** `CODEX_WORKDIR=/home/aponce` y `codex --version`",
            "",
            "**Bloque:**",
            "",
            "```bash",
            "cd /home/aponce",
            "codex --version",
            "ls -la codexbot",
            "```",
        ]
    )


def _status_text_for_chat(
    cfg: BotConfig,
    *,
    chat_id: int,
    tracker: "JobTracker",
    jobs: "queue.Queue[Job]",
    thread_mgr: "ThreadManager",
    orchestrator_queue: OrchestratorQueue | None = None,
) -> str:
    profile = _auth_effective_profile_name(cfg, chat_id=chat_id) if cfg.auth_enabled else ""
    eff_cfg = _apply_profile_to_cfg(cfg, profile_name=profile) if profile else cfg

    bypass = _effective_bypass_sandbox(eff_cfg, chat_id=chat_id)
    permissions = "full" if bypass else "default"
    sandbox_default = _threaded_sandbox_mode_label(eff_cfg) if not bypass else "(disabled)"

    tid = thread_mgr.get(chat_id) or ""

    model, effort = _job_model_label(eff_cfg, ["exec"], chat_id=chat_id)
    model_label = _format_model_for_display(model, effort)

    inflight = tracker.inflight(chat_id)
    queued = tracker.queued(chat_id)
    global_q = jobs.qsize()
    if orchestrator_queue is not None:
        orch_q = orchestrator_queue.get_queued_count()
        orch_r = orchestrator_queue.get_running_count()
        orch_paused = "paused" if orchestrator_queue.is_paused_globally() else "active"
        qmax = f"unbounded:{orch_q}" if cfg.queue_maxsize == 0 else f"{cfg.queue_maxsize}:{orch_q}"
    else:
        orch_q = 0
        orch_r = 0
        orch_paused = "disabled"
        qmax = "unbounded" if cfg.queue_maxsize == 0 else str(cfg.queue_maxsize)

    lines = [
        f"permissions: {permissions}",
        f"sandbox_default: {sandbox_default}",
        f"provider: {eff_cfg.codex_local_provider if eff_cfg.codex_use_oss else 'default (non-oss)'}",
        f"model: {model_label}",
        f"thread: {tid or '(none; send a message or use /reset)'}",
        f"queue: inflight={inflight} queued={queued} legacy_global={global_q} orch_queued={orch_q} orch_running={orch_r}",
        f"orchestrator: {orch_paused}",
        f"queue policy: max={qmax}",
        "",
        "Common commands:",
        "- /reset  (new thread)",
        "- /thread (show thread id)",
        "- /x      (cancel)",
        "- /agents (agent/role status)",
        "- /job <id> (job status)",
        "- /daily (daily digest)",
        "- /approve <id> (approve blocked job)",
        "- /pause <role> (pause role)",
        "- /resume <role> (resume role)",
        "- /cancel <id> (cancel orchestrator job)",
        "- /restart",
        "- /m      (model)",
        "- /p      (permissions)",
    ]
    return "\n".join(lines)


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def _parse_int_set(value: str | None) -> set[int]:
    if not value:
        return set()
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _chunk_text(text: str, limit: int = TELEGRAM_MSG_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    i = 0
    while i < len(text):
        # Prefer splitting on newlines near the limit.
        j = min(i + limit, len(text))
        if j < len(text):
            nl = text.rfind("\n", i, j)
            if nl != -1 and nl > i:
                j = nl + 1
        chunk = text[i:j]
        chunks.append(chunk)
        i = j
    return chunks


@dataclass(frozen=True)
class IncomingMessage:
    update_id: int
    chat_id: int
    user_id: int
    message_id: int
    username: str | None
    text: str


@dataclass(frozen=True)
class Job:
    chat_id: int
    reply_to_message_id: int
    user_text: str
    argv: list[str]
    mode_hint: str  # "ro" | "rw" (only used for defaults when argv implies exec)
    epoch: int  # increments on /cancel; jobs with stale epoch are dropped silently
    threaded: bool  # if enabled, reuse a Codex CLI session (exec resume) for this chat
    # Optional local image(s) to attach to the prompt (downloaded from Telegram).
    image_paths: list[Path]
    # Optional file(s) downloaded from Telegram and saved under CODEX_WORKDIR.
    # These are referenced by path in the prompt (Codex reads them from disk).
    upload_paths: list[Path]
    # If true, forces a brand-new thread even if a thread_id exists for this chat.
    force_new_thread: bool
    # If true, prefer replying as a Telegram voice note (if enabled/configured).
    prefer_voice_reply: bool = False


@dataclass(frozen=True)
class BotConfig:
    telegram_token: str
    allowed_chat_ids: set[int]
    allowed_user_ids: set[int]

    # If enabled, bypasses bot-side argv validation and treats (most) slash commands
    # as direct `codex ...` invocations. This is intentionally unsafe.
    unsafe_direct_codex: bool

    poll_timeout_seconds: int
    http_timeout_seconds: int
    http_max_retries: int
    http_retry_initial_seconds: float
    http_retry_max_seconds: float
    unauthorized_reply_cooldown_seconds: int
    drain_updates_on_start: bool
    worker_count: int
    queue_maxsize: int
    max_queued_per_chat: int
    heartbeat_seconds: int
    send_as_file_threshold_chars: int
    max_download_bytes: int

    # If enabled, act like a thin proxy: forward almost all text to Codex thread, avoiding bot parsing/validation.
    # Only a few bot-control commands remain (/new, /thread, /cancel).
    strict_proxy: bool

    # If enabled, voice/audio messages are downloaded, transcribed, and then processed as if they were text.
    transcribe_audio: bool
    # Transcription backend: "auto" | "whispercpp" | "openai"
    transcribe_backend: str
    transcribe_timeout_seconds: int
    ffmpeg_bin: str
    whispercpp_bin: str
    whispercpp_model_path: str
    whispercpp_threads: int
    openai_api_key: str
    openai_api_base_url: str
    transcribe_model: str
    transcribe_language: str
    transcribe_prompt: str
    transcribe_max_bytes: int

    state_file: Path
    notify_chat_id: int | None
    notify_on_start: bool

    codex_workdir: Path
    codex_timeout_seconds: int
    codex_use_oss: bool
    codex_local_provider: str
    codex_oss_model: str
    codex_openai_model: str
    codex_default_mode: str  # "ro" | "rw" | "full"

    # If enabled, force full access regardless of /ro /rw /full.
    codex_force_full_access: bool

    # If enabled, pass `--dangerously-bypass-approvals-and-sandbox` to Codex.
    # EXTREMELY DANGEROUS.
    codex_dangerous_bypass_sandbox: bool

    # Telegram formatting. Recommended: "HTML" (safe, escaped).
    telegram_parse_mode: str

    # Optional: application-level auth (username/password) + per-user profiles.
    # This is separate from TELEGRAM_ALLOWED_* allow-lists (which are still enforced).
    auth_enabled: bool
    auth_session_ttl_seconds: int
    auth_users_file: Path
    auth_profiles_file: Path

    # Orchestrator feature flags / policy defaults.
    orchestrator_db_path: Path = Path(__file__).with_name("data") / "jobs.sqlite"
    orchestrator_enabled: bool = True
    orchestrator_default_priority: int = 2
    orchestrator_default_max_cost_window_usd: float = 8.0
    orchestrator_default_role: str = "jarvis"
    orchestrator_daily_digest_seconds: int = 6 * 60 * 60
    orchestrator_agent_profiles: Path = Path(__file__).with_name("orchestrator") / "agents.yaml"
    orchestrator_worker_count: int = 3
    orchestrator_sessions_enabled: bool = True
    orchestrator_live_update_seconds: int = 8
    # Notification policy for Telegram chat: "verbose" | "minimal".
    orchestrator_notify_mode: str = "minimal"
    worktree_root: Path = Path(__file__).with_name("data") / "worktrees"
    artifacts_root: Path = Path(__file__).with_name("data") / "artifacts"
    runbooks_enabled: bool = True
    runbooks_path: Path = Path(__file__).with_name("orchestrator") / "runbooks.yaml"
    screenshot_enabled: bool = False
    screenshot_allowed_hosts: frozenset[str] = frozenset()
    transcribe_async: bool = True
    # Display/name used in prompts and documentation for the CEO user.
    ceo_name: str = "Alejandro Ponce"
    # Optional extra guardrails for destructive bot actions (e.g. /job del).
    # If empty, the bot falls back to profile-based permissions (when BOT_AUTH_ENABLED=1).
    admin_user_ids: frozenset[int] = frozenset()
    admin_chat_ids: frozenset[int] = frozenset()
    # Voice-out (reply as Telegram voice note).
    voice_out_enabled: bool = False
    # TTS backend: "none" | "piper" | "openai" | "tone"
    tts_backend: str = "none"
    # Max characters to speak (caption can still include text).
    tts_max_chars: int = 600
    # Optional global pitch shift for voice notes (in semitones). Negative = lower (more masculine).
    tts_voice_pitch_semitones: float = 0.0
    # OpenAI TTS settings.
    tts_openai_model: str = "tts-1"
    tts_openai_voice: str = "alloy"
    tts_openai_response_format: str = "mp3"
    # Piper (local/free) TTS settings.
    tts_piper_bin: str = "piper"
    tts_piper_model_path: str = ""
    # Optional (multi-speaker models). Empty = default.
    tts_piper_speaker: str = ""
    # Optional Piper tunables (empty/0 = defaults).
    tts_piper_noise_scale: float = 0.0
    tts_piper_length_scale: float = 0.0
    tts_piper_noise_w: float = 0.0
    tts_piper_sentence_silence: float = 0.0


class TelegramAPI:
    def __init__(
        self,
        token: str,
        *,
        http_timeout_seconds: int,
        http_max_retries: int,
        http_retry_initial_seconds: float,
        http_retry_max_seconds: float,
        parse_mode: str = "",
    ) -> None:
        self._base_url = f"https://api.telegram.org/bot{token}/"
        self._file_base_url = f"https://api.telegram.org/file/bot{token}/"
        self._http_timeout_seconds = http_timeout_seconds
        self._http_max_retries = max(0, int(http_max_retries))
        self._http_retry_initial_seconds = max(0.0, float(http_retry_initial_seconds))
        self._http_retry_max_seconds = max(0.0, float(http_retry_max_seconds))
        self._parse_mode = (parse_mode or "").strip()
        self._ssl_context = ssl.create_default_context()

    def _request(self, method: str, payload: dict[str, Any] | None) -> Any:
        url = self._base_url + method
        data = None
        # Telegram occasionally closes keep-alive connections without a response; prefer short-lived connections.
        headers = {"Connection": "close"}
        if payload is not None:
            data = urllib.parse.urlencode(payload).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        last_err: Exception | None = None

        for attempt in range(self._http_max_retries + 1):
            try:
                # Use the same SSL context as multipart uploads for consistency.
                with urllib.request.urlopen(req, timeout=self._http_timeout_seconds, context=self._ssl_context) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                retry_after = _extract_retry_after_seconds(body)
                if e.code in (429, 500, 502, 503, 504) and attempt < self._http_max_retries:
                    _sleep_retry(attempt, self._http_retry_initial_seconds, self._http_retry_max_seconds, retry_after)
                    continue
                raise RuntimeError(f"Telegram HTTP error calling {method}: {e.code} {body}") from e
            except urllib.error.URLError as e:
                last_err = e
                if attempt < self._http_max_retries:
                    _sleep_retry(attempt, self._http_retry_initial_seconds, self._http_retry_max_seconds, None)
                    continue
                raise RuntimeError(f"Telegram URL error calling {method}: {e}") from e
            except Exception as e:
                # e.g. http.client.RemoteDisconnected, ConnectionResetError, SSL errors, etc.
                last_err = e
                if attempt < self._http_max_retries:
                    _sleep_retry(attempt, self._http_retry_initial_seconds, self._http_retry_max_seconds, None)
                    continue
                raise RuntimeError(f"Telegram request error calling {method}: {e}") from e

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                last_err = e
                if attempt < self._http_max_retries:
                    _sleep_retry(attempt, self._http_retry_initial_seconds, self._http_retry_max_seconds, None)
                    continue
                raise RuntimeError(f"Telegram returned non-JSON for {method}: {raw[:2000]}") from e

            if not parsed.get("ok", False):
                retry_after = None
                try:
                    params = parsed.get("parameters") or {}
                    if isinstance(params, dict) and isinstance(params.get("retry_after"), (int, float)):
                        retry_after = float(params["retry_after"])
                except Exception:
                    retry_after = None
                err_code = parsed.get("error_code")
                if err_code in (429, 500, 502, 503, 504) and attempt < self._http_max_retries:
                    _sleep_retry(attempt, self._http_retry_initial_seconds, self._http_retry_max_seconds, retry_after)
                    continue
                raise RuntimeError(f"Telegram API error calling {method}: {raw[:2000]}")

            return parsed["result"]

        raise RuntimeError(f"Telegram request failed calling {method}: {last_err}")

    def get_updates(self, *, offset: int, timeout_seconds: int) -> list[dict[str, Any]]:
        return self._request(
            "getUpdates",
            {
                "timeout": str(timeout_seconds),
                "offset": str(offset),
                "allowed_updates": json.dumps(["message"]),
            },
        )

    def get_file(self, file_id: str) -> dict[str, Any]:
        """
        Telegram getFile -> returns a dict including file_path, file_size, etc.
        """
        return self._request("getFile", {"file_id": file_id})

    def download_file_to(self, *, file_path: str, dest: Path, max_bytes: int = 0) -> None:
        """
        Download a Telegram file (from getFile.file_path) to dest.
        If max_bytes > 0, abort if download exceeds that size.
        """
        fp = (file_path or "").lstrip("/")
        if not fp:
            raise RuntimeError("Empty Telegram file_path")
        url = self._file_base_url + fp
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self._http_timeout_seconds, context=self._ssl_context) as resp:
            total = 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 64)
                    if not chunk:
                        break
                    total += len(chunk)
                    if max_bytes and max_bytes > 0 and total > max_bytes:
                        raise RuntimeError(f"Download too large (>{max_bytes} bytes)")
                    f.write(chunk)

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        disable_web_page_preview: bool = True,
    ) -> int | None:
        parse_mode = self._parse_mode
        payload_text = text
        if parse_mode.lower() == "html":
            payload_text = _markdownish_to_telegram_html(payload_text)
            parse_mode = "HTML"
            if len(payload_text) > TELEGRAM_MSG_LIMIT:
                # Best-effort fallback: keep it valid and under Telegram's limit.
                payload_text = _html_escape(text)[:TELEGRAM_MSG_LIMIT]
        elif parse_mode.lower() in ("markdown", "markdownv2"):
            # Not recommended unless you implement proper escaping.
            parse_mode = "MarkdownV2" if parse_mode.lower() == "markdownv2" else "Markdown"
            if len(payload_text) > TELEGRAM_MSG_LIMIT:
                payload_text = payload_text[:TELEGRAM_MSG_LIMIT]
        else:
            parse_mode = ""
            if len(payload_text) > TELEGRAM_MSG_LIMIT:
                payload_text = payload_text[:TELEGRAM_MSG_LIMIT]

        payload: dict[str, Any] = {
            "chat_id": str(chat_id),
            "text": payload_text,
            "disable_web_page_preview": "1" if disable_web_page_preview else "0",
        }
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = str(reply_to_message_id)
        if parse_mode:
            payload["parse_mode"] = parse_mode
        res = self._request("sendMessage", payload)
        try:
            if isinstance(res, dict) and res.get("message_id") is not None:
                return int(res["message_id"])
        except Exception:
            pass
        return None

    def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        disable_web_page_preview: bool = True,
    ) -> None:
        """
        Edit a previously-sent bot message. Used to keep "ticket cards" and /watch status to one message (no spam).
        """
        parse_mode = self._parse_mode
        payload_text = text
        if parse_mode.lower() == "html":
            payload_text = _markdownish_to_telegram_html(payload_text)
            parse_mode = "HTML"
            if len(payload_text) > TELEGRAM_MSG_LIMIT:
                payload_text = _html_escape(text)[:TELEGRAM_MSG_LIMIT]
        elif parse_mode.lower() in ("markdown", "markdownv2"):
            parse_mode = "MarkdownV2" if parse_mode.lower() == "markdownv2" else "Markdown"
            if len(payload_text) > TELEGRAM_MSG_LIMIT:
                payload_text = payload_text[:TELEGRAM_MSG_LIMIT]
        else:
            parse_mode = ""
            if len(payload_text) > TELEGRAM_MSG_LIMIT:
                payload_text = payload_text[:TELEGRAM_MSG_LIMIT]

        payload: dict[str, Any] = {
            "chat_id": str(chat_id),
            "message_id": str(int(message_id)),
            "text": payload_text,
            "disable_web_page_preview": "1" if disable_web_page_preview else "0",
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        self._request("editMessageText", payload)

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """
        Ephemeral UX hint (e.g. "typing") without sending a message.
        """
        self._request("sendChatAction", {"chat_id": str(chat_id), "action": str(action)})

    def set_my_commands(self, commands: list[tuple[str, str]], *, scope_type: str = "", language_code: str = "") -> None:
        """
        Configure Telegram slash-command suggestions shown in the client when user types "/".
        """
        payload_cmds: list[dict[str, str]] = []
        for cmd, desc in commands:
            c = (cmd or "").strip().lstrip("/")
            d = (desc or "").strip()
            if not c or not d:
                continue
            payload_cmds.append({"command": c, "description": d[:256]})
        if not payload_cmds:
            return
        payload: dict[str, str] = {"commands": json.dumps(payload_cmds, ensure_ascii=False)}
        scope_type = (scope_type or "").strip()
        if scope_type:
            payload["scope"] = json.dumps({"type": scope_type}, ensure_ascii=False)
        language_code = (language_code or "").strip()
        if language_code:
            payload["language_code"] = language_code
        self._request(
            "setMyCommands",
            payload,
        )

    def set_chat_menu_button_commands(self) -> None:
        """
        Ensure Telegram clients show the slash command menu button.
        """
        self._request(
            "setChatMenuButton",
            {"menu_button": json.dumps({"type": "commands"}, ensure_ascii=False)},
        )

    def _request_multipart(
        self,
        method: str,
        *,
        fields: dict[str, str],
        file_field: str,
        file_path: Path,
        filename: str,
        content_type: str,
    ) -> Any:
        """
        Minimal multipart/form-data uploader using stdlib only.
        Streams the file content to avoid reading the whole file into memory.
        """
        url = self._base_url + method
        u = urllib.parse.urlparse(url)
        if not u.hostname or not u.path:
            raise RuntimeError(f"Bad Telegram URL: {url}")

        boundary = f"----codexbot_{int(time.time()*1000)}_{os.getpid()}"

        def _part_field(name: str, value: str) -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"\r\n"
                f"{value}\r\n"
            ).encode("utf-8")

        def _part_file_header() -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n"
                f"\r\n"
            ).encode("utf-8")

        closing = f"\r\n--{boundary}--\r\n".encode("utf-8")
        header = _part_file_header()
        field_parts = b"".join(_part_field(k, v) for k, v in fields.items())

        try:
            file_size = file_path.stat().st_size
        except OSError as e:
            raise RuntimeError(f"Failed to stat upload file: {file_path} ({e})") from e

        content_length = len(field_parts) + len(header) + file_size + len(closing)

        last_err: Exception | None = None
        for attempt in range(self._http_max_retries + 1):
            conn: http.client.HTTPSConnection | None = None
            try:
                conn = http.client.HTTPSConnection(u.hostname, timeout=self._http_timeout_seconds, context=self._ssl_context)
                conn.putrequest("POST", u.path)
                conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
                conn.putheader("Content-Length", str(content_length))
                conn.endheaders()

                conn.send(field_parts)
                conn.send(header)
                with file_path.open("rb") as f:
                    while True:
                        chunk = f.read(1024 * 64)
                        if not chunk:
                            break
                        conn.send(chunk)
                conn.send(closing)

                resp = conn.getresponse()
                raw_bytes = resp.read()
                raw = raw_bytes.decode("utf-8", errors="replace")

                if resp.status >= 400:
                    retry_after = _extract_retry_after_seconds(raw)
                    if resp.status in (429, 500, 502, 503, 504) and attempt < self._http_max_retries:
                        _sleep_retry(attempt, self._http_retry_initial_seconds, self._http_retry_max_seconds, retry_after)
                        continue
                    raise RuntimeError(f"Telegram HTTP error calling {method}: {resp.status} {raw[:2000]}")

                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    last_err = e
                    if attempt < self._http_max_retries:
                        _sleep_retry(attempt, self._http_retry_initial_seconds, self._http_retry_max_seconds, None)
                        continue
                    raise RuntimeError(f"Telegram returned non-JSON for {method}: {raw[:2000]}") from e

                if not parsed.get("ok", False):
                    retry_after = None
                    try:
                        params = parsed.get("parameters") or {}
                        if isinstance(params, dict) and isinstance(params.get("retry_after"), (int, float)):
                            retry_after = float(params["retry_after"])
                    except Exception:
                        retry_after = None
                    err_code = parsed.get("error_code")
                    if err_code in (429, 500, 502, 503, 504) and attempt < self._http_max_retries:
                        _sleep_retry(attempt, self._http_retry_initial_seconds, self._http_retry_max_seconds, retry_after)
                        continue
                    raise RuntimeError(f"Telegram API error calling {method}: {raw[:2000]}")

                return parsed["result"]
            except Exception as e:
                last_err = e
                if attempt < self._http_max_retries:
                    _sleep_retry(attempt, self._http_retry_initial_seconds, self._http_retry_max_seconds, None)
                    continue
                raise
            finally:
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass

        raise RuntimeError(f"Telegram request failed calling {method}: {last_err}")

    def send_document(
        self,
        chat_id: int,
        file_path: Path,
        *,
        filename: str | None = None,
        caption: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> None:
        fields: dict[str, str] = {"chat_id": str(chat_id)}
        if caption:
            fields["caption"] = caption
        if reply_to_message_id is not None:
            fields["reply_to_message_id"] = str(reply_to_message_id)
        fn = filename or file_path.name
        ctype = mimetypes.guess_type(fn)[0] or "application/octet-stream"
        if ctype.startswith("text/"):
            ctype = f"{ctype}; charset=utf-8"
        self._request_multipart(
            "sendDocument",
            fields=fields,
            file_field="document",
            file_path=file_path,
            filename=fn,
            content_type=ctype,
        )

    def send_photo(
        self,
        chat_id: int,
        file_path: Path,
        *,
        filename: str | None = None,
        caption: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> None:
        """
        Send an image with inline preview in Telegram clients.
        """
        fields: dict[str, str] = {"chat_id": str(chat_id)}
        if caption:
            fields["caption"] = caption
        if reply_to_message_id is not None:
            fields["reply_to_message_id"] = str(reply_to_message_id)
        fn = filename or file_path.name
        ctype = mimetypes.guess_type(fn)[0] or "application/octet-stream"
        self._request_multipart(
            "sendPhoto",
            fields=fields,
            file_field="photo",
            file_path=file_path,
            filename=fn,
            content_type=ctype,
        )

    def send_voice(
        self,
        chat_id: int,
        file_path: Path,
        *,
        filename: str | None = None,
        caption: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> int | None:
        """
        Send a Telegram voice note (OGG/Opus).
        """
        fields: dict[str, str] = {"chat_id": str(chat_id)}
        if caption:
            # Telegram captions have a smaller limit than messages; keep it conservative.
            fields["caption"] = caption[:900]
        if reply_to_message_id is not None:
            fields["reply_to_message_id"] = str(reply_to_message_id)
        fn = filename or file_path.name or "voice.ogg"
        res = self._request_multipart(
            "sendVoice",
            fields=fields,
            file_field="voice",
            file_path=file_path,
            filename=fn,
            content_type="audio/ogg",
        )
        try:
            if isinstance(res, dict) and res.get("message_id") is not None:
                return int(res["message_id"])
        except Exception:
            pass
        return None


class OpenAITranscriber:
    """
    Minimal OpenAI speech-to-text client (stdlib-only).
    Uses multipart/form-data to POST to /v1/audio/transcriptions.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_base_url: str,
        timeout_seconds: int,
        max_retries: int,
        retry_initial_seconds: float,
        retry_max_seconds: float,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base = (api_base_url or "https://api.openai.com").strip().rstrip("/")
        self._timeout_seconds = int(timeout_seconds)
        self._max_retries = max(0, int(max_retries))
        self._retry_initial_seconds = max(0.0, float(retry_initial_seconds))
        self._retry_max_seconds = max(0.0, float(retry_max_seconds))
        self._ssl_context = ssl.create_default_context()

    def _post_multipart(
        self,
        *,
        url: str,
        fields: dict[str, str],
        file_field: str,
        file_path: Path,
        filename: str,
        content_type: str,
    ) -> dict[str, Any]:
        u = urllib.parse.urlparse(url)
        if u.scheme != "https" or not u.hostname or not u.path:
            raise RuntimeError(f"Bad OpenAI URL: {url}")

        boundary = f"----codexbot_openai_{int(time.time()*1000)}_{os.getpid()}"

        def _part_field(name: str, value: str) -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"\r\n"
                f"{value}\r\n"
            ).encode("utf-8")

        def _part_file_header() -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n"
                f"\r\n"
            ).encode("utf-8")

        closing = f"\r\n--{boundary}--\r\n".encode("utf-8")
        header = _part_file_header()
        field_parts = b"".join(_part_field(k, v) for k, v in fields.items())

        try:
            file_size = file_path.stat().st_size
        except OSError as e:
            raise RuntimeError(f"Failed to stat audio file: {file_path} ({e})") from e

        content_length = len(field_parts) + len(header) + file_size + len(closing)

        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            conn: http.client.HTTPSConnection | None = None
            try:
                conn = http.client.HTTPSConnection(u.hostname, timeout=self._timeout_seconds, context=self._ssl_context)
                conn.putrequest("POST", u.path)
                conn.putheader("Authorization", f"Bearer {self._api_key}")
                conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
                conn.putheader("Content-Length", str(content_length))
                conn.putheader("Connection", "close")
                conn.endheaders()

                conn.send(field_parts)
                conn.send(header)
                with file_path.open("rb") as f:
                    while True:
                        chunk = f.read(1024 * 64)
                        if not chunk:
                            break
                        conn.send(chunk)
                conn.send(closing)

                resp = conn.getresponse()
                raw_bytes = resp.read()
                raw = raw_bytes.decode("utf-8", errors="replace")
                if resp.status >= 400:
                    if resp.status in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                        _sleep_retry(attempt, self._retry_initial_seconds, self._retry_max_seconds, None)
                        continue
                    raise RuntimeError(f"OpenAI HTTP error: {resp.status} {raw[:2000]}")

                try:
                    parsed = json.loads(raw)
                except Exception as e:
                    raise RuntimeError(f"OpenAI returned non-JSON: {raw[:2000]}") from e
                if not isinstance(parsed, dict):
                    raise RuntimeError("OpenAI returned unexpected JSON")
                return parsed
            except Exception as e:
                last_err = e
                if attempt < self._max_retries:
                    _sleep_retry(attempt, self._retry_initial_seconds, self._retry_max_seconds, None)
                    continue
                raise
            finally:
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass

        raise RuntimeError(f"OpenAI request failed: {last_err}")

    def transcribe(
        self,
        *,
        audio_path: Path,
        model: str,
        language: str = "",
        prompt: str = "",
    ) -> str:
        if not self._api_key:
            raise RuntimeError("Missing OpenAI API key")
        model = (model or "").strip()
        if not model:
            raise RuntimeError("Missing transcription model")

        fields: dict[str, str] = {
            "model": model,
            "response_format": "json",
        }
        lang = (language or "").strip()
        if lang:
            fields["language"] = lang
        pr = (prompt or "").strip()
        if pr:
            fields["prompt"] = pr

        fn = audio_path.name
        ctype = mimetypes.guess_type(fn)[0] or "application/octet-stream"
        url = self._base + "/v1/audio/transcriptions"
        parsed = self._post_multipart(
            url=url,
            fields=fields,
            file_field="file",
            file_path=audio_path,
            filename=fn,
            content_type=ctype,
        )
        txt = parsed.get("text")
        return txt.strip() if isinstance(txt, str) else ""


class OpenAITTS:
    """
    Minimal OpenAI text-to-speech client (stdlib-only).
    Uses JSON POST to /v1/audio/speech and returns raw audio bytes.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_base_url: str,
        timeout_seconds: int,
        max_retries: int,
        retry_initial_seconds: float,
        retry_max_seconds: float,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base = (api_base_url or "https://api.openai.com").strip().rstrip("/")
        self._timeout_seconds = int(timeout_seconds)
        self._max_retries = max(0, int(max_retries))
        self._retry_initial_seconds = max(0.0, float(retry_initial_seconds))
        self._retry_max_seconds = max(0.0, float(retry_max_seconds))
        self._ssl_context = ssl.create_default_context()

    def synthesize(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        response_format: str,
    ) -> bytes:
        if not self._api_key:
            raise RuntimeError("Missing OpenAI API key")
        t = (text or "").strip()
        if not t:
            raise RuntimeError("Empty TTS input")
        m = (model or "").strip() or "tts-1"
        v = (voice or "").strip() or "alloy"
        fmt = (response_format or "").strip().lower() or "mp3"

        url = self._base + "/v1/audio/speech"
        u = urllib.parse.urlparse(url)
        if u.scheme != "https" or not u.hostname or not u.path:
            raise RuntimeError(f"Bad OpenAI URL: {url}")

        body = json.dumps(
            {
                "model": m,
                "input": t,
                "voice": v,
                "response_format": fmt,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            conn: http.client.HTTPSConnection | None = None
            try:
                conn = http.client.HTTPSConnection(u.hostname, timeout=self._timeout_seconds, context=self._ssl_context)
                conn.putrequest("POST", u.path)
                conn.putheader("Authorization", f"Bearer {self._api_key}")
                conn.putheader("Content-Type", "application/json; charset=utf-8")
                conn.putheader("Content-Length", str(len(body)))
                conn.putheader("Connection", "close")
                conn.endheaders()
                conn.send(body)

                resp = conn.getresponse()
                raw = resp.read()
                if resp.status >= 400:
                    msg = raw.decode("utf-8", errors="replace")
                    if resp.status in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                        _sleep_retry(attempt, self._retry_initial_seconds, self._retry_max_seconds, None)
                        continue
                    raise RuntimeError(f"OpenAI TTS HTTP error: {resp.status} {msg[:2000]}")
                return raw
            except Exception as e:
                last_err = e
                if attempt < self._max_retries:
                    _sleep_retry(attempt, self._retry_initial_seconds, self._retry_max_seconds, None)
                    continue
                raise
            finally:
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass

        raise RuntimeError(f"OpenAI TTS request failed: {last_err}")


class WhisperCppTranscriber:
    """
    Local/offline transcriber using whisper.cpp + ffmpeg.
    """

    def __init__(
        self,
        *,
        ffmpeg_bin: str,
        whisper_bin: str,
        model_path: str,
        threads: int,
        timeout_seconds: int,
        language: str,
        prompt: str,
    ) -> None:
        self._ffmpeg = (ffmpeg_bin or "").strip()
        self._whisper = (whisper_bin or "").strip()
        self._model_path = (model_path or "").strip()
        self._threads = int(threads) if int(threads) > 0 else 1
        self._timeout_seconds = int(timeout_seconds) if int(timeout_seconds) > 0 else 300
        self._language = (language or "").strip()
        self._prompt = (prompt or "").strip()

    @staticmethod
    def _is_exec_available(cmd: str) -> bool:
        if not cmd:
            return False
        if Path(cmd).expanduser().exists():
            return True
        return shutil.which(cmd) is not None

    def is_available(self) -> tuple[bool, str]:
        if not self._is_exec_available(self._ffmpeg):
            return False, f"ffmpeg no encontrado: {self._ffmpeg or '(empty)'}"
        if not self._is_exec_available(self._whisper):
            return False, f"whisper.cpp bin no encontrado: {self._whisper or '(empty)'}"
        if not self._model_path or not Path(self._model_path).expanduser().exists():
            return False, f"modelo whisper.cpp no encontrado: {self._model_path or '(empty)'}"
        return True, ""

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds,
        )

    def transcribe_file(self, *, input_path: Path) -> str:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(reason)

        input_path = input_path.expanduser().resolve()
        if not input_path.exists():
            raise RuntimeError(f"audio no existe: {input_path}")

        tmp_dir = Path(tempfile.mkdtemp(prefix="codexbot_whisper_", dir=str(input_path.parent)))
        wav_path = tmp_dir / "audio.wav"
        out_prefix = tmp_dir / "out"
        out_txt = Path(str(out_prefix) + ".txt")

        try:
            ffmpeg = str(Path(self._ffmpeg).expanduser()) if Path(self._ffmpeg).expanduser().exists() else self._ffmpeg
            whisper = str(Path(self._whisper).expanduser()) if Path(self._whisper).expanduser().exists() else self._whisper
            model = str(Path(self._model_path).expanduser())

            # Convert to 16kHz mono WAV (whisper.cpp expects PCM WAV).
            p1 = self._run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(input_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "wav",
                    str(wav_path),
                ]
            )
            if p1.returncode != 0 or not wav_path.exists():
                raise RuntimeError(f"ffmpeg fallo: {(p1.stderr or p1.stdout or '').strip()[:2000]}")

            # whisper.cpp: write plain text output to out_prefix.txt
            cmd = [
                whisper,
                "-m",
                model,
                "-f",
                str(wav_path),
                "-otxt",
                "-of",
                str(out_prefix),
                "-nt",  # no timestamps
                "-t",
                str(self._threads),
            ]
            if self._language:
                cmd += ["-l", self._language]
            if self._prompt:
                cmd += ["--prompt", self._prompt]

            p2 = self._run(cmd)
            if p2.returncode != 0:
                raise RuntimeError(f"whisper.cpp fallo: {(p2.stderr or p2.stdout or '').strip()[:2000]}")
            try:
                txt = out_txt.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                raise RuntimeError(f"no pude leer transcripcion: {e}") from e
            return txt
        finally:
            try:
                for p in tmp_dir.glob("*"):
                    p.unlink(missing_ok=True)
                tmp_dir.rmdir()
            except Exception:
                pass


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a unique temp file in the same directory to keep replace() atomic and
    # to avoid collisions if multiple processes write state concurrently.
    tmp_f = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
        delete=False,
    )
    try:
        tmp_f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        tmp_f.flush()
    finally:
        tmp_f.close()
    Path(tmp_f.name).replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except FileNotFoundError:
        return {}
    except Exception:
        LOG.exception("Failed to read json: %s", path)
    return {}


def _get_state(cfg: "BotConfig") -> dict[str, Any]:
    return _read_json(cfg.state_file)


def _get_voice_state(cfg: "BotConfig") -> dict[str, Any]:
    st = _get_state(cfg)
    raw = st.get("voice")
    return raw if isinstance(raw, dict) else {}


def _set_voice_state(cfg: "BotConfig", voice_state: dict[str, Any]) -> None:
    st = _get_state(cfg)
    st["voice"] = voice_state
    _atomic_write_json(cfg.state_file, st)


def _clear_voice_state(cfg: "BotConfig") -> None:
    st = _get_state(cfg)
    st.pop("voice", None)
    _atomic_write_json(cfg.state_file, st)


def _get_qa_state(cfg: "BotConfig") -> dict[str, Any]:
    st = _get_state(cfg)
    raw = st.get("qa")
    return raw if isinstance(raw, dict) else {}


def _set_qa_state(cfg: "BotConfig", qa_state: dict[str, Any]) -> None:
    st = _get_state(cfg)
    st["qa"] = qa_state
    _atomic_write_json(cfg.state_file, st)


def _qa_chat_key(chat_id: int) -> str:
    return str(int(chat_id))


def _qa_is_safe_artifact_id(s: str) -> bool:
    """
    Defensive: artifacts dir ids should be plain UUID-ish tokens (no slashes, no traversal).
    """
    s = (s or "").strip()
    if not s:
        return False
    if len(s) > 128:
        return False
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            continue
        return False
    # Common case: UUID with hyphens. Don't require exact UUID format to stay flexible.
    if "/" in s or "\\" in s or ".." in s:
        return False
    return True


def _qa_get_evidence_artifact_id(cfg: "BotConfig", *, chat_id: int) -> str:
    qa = _get_qa_state(cfg)
    by_chat = qa.get("evidence_artifact_by_chat")
    if not isinstance(by_chat, dict):
        return ""
    raw = by_chat.get(_qa_chat_key(chat_id))
    return raw.strip() if isinstance(raw, str) else ""


def _qa_set_evidence_artifact_id(cfg: "BotConfig", *, chat_id: int, artifact_id: str) -> None:
    qa = _get_qa_state(cfg)
    by_chat = qa.get("evidence_artifact_by_chat")
    if not isinstance(by_chat, dict):
        by_chat = {}
    artifact_id = (artifact_id or "").strip()
    if artifact_id:
        by_chat[_qa_chat_key(chat_id)] = artifact_id
    else:
        by_chat.pop(_qa_chat_key(chat_id), None)
    qa["evidence_artifact_by_chat"] = by_chat
    _set_qa_state(cfg, qa)


def _qa_evidence_dir(cfg: "BotConfig", *, chat_id: int) -> Path | None:
    artifact_id = _qa_get_evidence_artifact_id(cfg, chat_id=chat_id)
    if not artifact_id:
        return None
    if not _qa_is_safe_artifact_id(artifact_id):
        return None
    return (cfg.artifacts_root / artifact_id).resolve()


def _qa_append_evidence(cfg: "BotConfig", *, chat_id: int, event: dict[str, Any]) -> None:
    """
    Best-effort JSONL evidence capture for human-in-the-loop Telegram QA.
    Stored under: cfg.artifacts_root/<artifact_id>/telegram_qa.jsonl
    """
    out_dir = _qa_evidence_dir(cfg, chat_id=chat_id)
    if out_dir is None:
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "telegram_qa.jsonl"
        payload = dict(event or {})
        payload.setdefault("ts", time.time())
        payload.setdefault("chat_id", int(chat_id))
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Evidence capture should never break the bot.
        LOG.exception("QA evidence append failed. chat_id=%s", chat_id)


def _voice_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
    return None


def _voice_int(v: Any, *, min_value: int, max_value: int) -> int | None:
    try:
        i = int(v)
    except Exception:
        return None
    if i < min_value or i > max_value:
        return None
    return i


def _voice_str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _effective_transcribe_enabled(cfg: "BotConfig") -> bool:
    raw = _get_voice_state(cfg).get("enabled")
    vb = _voice_bool(raw)
    return vb if vb is not None else bool(cfg.transcribe_audio)


def _effective_transcribe_backend(cfg: "BotConfig") -> str:
    raw = _voice_str(_get_voice_state(cfg).get("backend"))
    if raw in ("auto", "openai", "whispercpp"):
        return raw
    return (cfg.transcribe_backend or "auto").strip().lower() or "auto"


def _effective_whisper_model_path(cfg: "BotConfig") -> str:
    raw = _voice_str(_get_voice_state(cfg).get("whisper_model_path"))
    return raw or cfg.whispercpp_model_path


def _effective_whisper_threads(cfg: "BotConfig") -> int:
    raw = _get_voice_state(cfg).get("whisper_threads")
    vi = _voice_int(raw, min_value=1, max_value=64)
    return vi if vi is not None else int(cfg.whispercpp_threads)


def _effective_transcribe_timeout(cfg: "BotConfig") -> int:
    raw = _get_voice_state(cfg).get("timeout_seconds")
    vi = _voice_int(raw, min_value=5, max_value=3600)
    return vi if vi is not None else int(cfg.transcribe_timeout_seconds)


def _effective_transcribe_language(cfg: "BotConfig") -> str:
    raw = _voice_str(_get_voice_state(cfg).get("language"))
    # Allow empty for auto-detect.
    return raw if raw or raw == "" else cfg.transcribe_language


def _get_auth_state(cfg: "BotConfig") -> dict[str, Any]:
    st = _get_state(cfg)
    raw = st.get("auth")
    return raw if isinstance(raw, dict) else {}


def _set_auth_state(cfg: "BotConfig", auth_state: dict[str, Any]) -> None:
    st = _get_state(cfg)
    st["auth"] = auth_state
    _atomic_write_json(cfg.state_file, st)


def _get_auth_sessions(cfg: "BotConfig") -> dict[str, Any]:
    auth = _get_auth_state(cfg)
    raw = auth.get("sessions")
    return raw if isinstance(raw, dict) else {}


def _set_auth_sessions(cfg: "BotConfig", sessions: dict[str, Any]) -> None:
    auth = _get_auth_state(cfg)
    auth["sessions"] = sessions
    _set_auth_state(cfg, auth)


def _auth_now() -> float:
    return time.time()


def _session_key(chat_id: int) -> str:
    return str(int(chat_id))


def _auth_is_session_active(cfg: "BotConfig", *, chat_id: int) -> tuple[bool, dict[str, Any]]:
    """
    Returns (active, session_dict). Session is considered active if now <= expires_at.
    """
    sessions = _get_auth_sessions(cfg)
    s = sessions.get(_session_key(chat_id))
    if not isinstance(s, dict):
        return False, {}
    exp = s.get("expires_at")
    try:
        exp_f = float(exp)
    except Exception:
        exp_f = 0.0
    now = _auth_now()
    if exp_f and now <= exp_f:
        return True, s
    return False, {}


def _auth_touch_session(cfg: "BotConfig", *, chat_id: int) -> None:
    sessions = _get_auth_sessions(cfg)
    key = _session_key(chat_id)
    s = sessions.get(key)
    if not isinstance(s, dict):
        return
    now = _auth_now()
    ttl = int(cfg.auth_session_ttl_seconds) if int(cfg.auth_session_ttl_seconds) > 0 else 0
    if ttl <= 0:
        return
    s["last_active_at"] = now
    s["expires_at"] = now + ttl
    sessions[key] = s
    _set_auth_sessions(cfg, sessions)


def _auth_logout(cfg: "BotConfig", *, chat_id: int) -> None:
    sessions = _get_auth_sessions(cfg)
    sessions.pop(_session_key(chat_id), None)
    _set_auth_sessions(cfg, sessions)

def _auth_clear_all_sessions(cfg: "BotConfig") -> None:
    """
    Clears all auth sessions from state.
    Use this if you want every bot restart to require /login again.
    """
    auth = _get_auth_state(cfg)
    auth["sessions"] = {}
    _set_auth_state(cfg, auth)


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {}
    except Exception:
        LOG.exception("Failed to read json file: %s", path)
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        LOG.exception("Failed to parse json file: %s", path)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_profiles(cfg: "BotConfig") -> dict[str, Any]:
    raw = _load_json_file(cfg.auth_profiles_file)
    profiles = raw.get("profiles") if isinstance(raw, dict) else None
    return profiles if isinstance(profiles, dict) else {}


def _load_users(cfg: "BotConfig") -> dict[str, Any]:
    raw = _load_json_file(cfg.auth_users_file)
    users = raw.get("users") if isinstance(raw, dict) else None
    return users if isinstance(users, dict) else {}


def _normalize_username(u: str) -> str:
    return (u or "").strip()


def _pbkdf2_hash_password(*, password: str, salt_b64: str, iterations: int) -> bytes:
    salt = base64.b64decode(salt_b64.encode("ascii"), validate=True)
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))


def _auth_verify_password(user_rec: dict[str, Any], password: str) -> bool:
    try:
        salt_b64 = user_rec.get("salt_b64")
        hash_b64 = user_rec.get("hash_b64")
        iters = user_rec.get("iterations")
        if not isinstance(salt_b64, str) or not isinstance(hash_b64, str):
            return False
        iterations = int(iters) if isinstance(iters, int) or (isinstance(iters, str) and str(iters).isdigit()) else 200_000
        expected = base64.b64decode(hash_b64.encode("ascii"), validate=True)
        got = _pbkdf2_hash_password(password=password, salt_b64=salt_b64, iterations=iterations)
        return hmac.compare_digest(expected, got)
    except Exception:
        return False


def _auth_login(cfg: "BotConfig", *, chat_id: int, username: str, password: str) -> tuple[bool, str]:
    users = _load_users(cfg)
    u = _normalize_username(username)
    rec = users.get(u)
    if not isinstance(rec, dict):
        return False, "Usuario o password incorrectos."
    if not _auth_verify_password(rec, password):
        return False, "Usuario o password incorrectos."

    profile = rec.get("profile")
    profile_s = profile.strip() if isinstance(profile, str) else ""
    profiles = _load_profiles(cfg)
    if profile_s and profile_s not in profiles:
        return False, "Usuario valido pero su perfil no existe en profiles.json."

    ttl = int(cfg.auth_session_ttl_seconds) if int(cfg.auth_session_ttl_seconds) > 0 else 0
    now = _auth_now()
    sessions = _get_auth_sessions(cfg)
    sessions[_session_key(chat_id)] = {
        "username": u,
        "profile": profile_s,
        "logged_in_at": now,
        "last_active_at": now,
        "expires_at": (now + ttl) if ttl else 0.0,
    }
    _set_auth_sessions(cfg, sessions)
    return True, f"OK. Login: {u} (perfil={profile_s or 'default'})"


def _auth_effective_profile_name(cfg: "BotConfig", *, chat_id: int) -> str:
    active, s = _auth_is_session_active(cfg, chat_id=chat_id)
    if not active:
        return ""
    p = s.get("profile")
    return p.strip() if isinstance(p, str) else ""


def _profile_value(profiles: dict[str, Any], profile_name: str, key: str, default: Any) -> Any:
    p = profiles.get(profile_name)
    if not isinstance(p, dict):
        return default
    v = p.get(key)
    return v if v is not None else default


def _apply_profile_to_cfg(cfg: "BotConfig", *, profile_name: str) -> BotConfig:
    if not profile_name:
        return cfg
    profiles = _load_profiles(cfg)
    if profile_name not in profiles:
        return cfg

    # Minimal surface area: only override the risky bits. Everything else remains from env config.
    overrides: dict[str, Any] = {}
    for k in ("codex_default_mode", "codex_force_full_access", "codex_dangerous_bypass_sandbox", "unsafe_direct_codex", "codex_workdir"):
        if k in ("codex_default_mode",):
            v = _profile_value(profiles, profile_name, k, None)
            if isinstance(v, str) and v.strip().lower() in ("ro", "rw", "full"):
                overrides[k] = v.strip().lower()
        elif k == "codex_workdir":
            v = _profile_value(profiles, profile_name, k, None)
            if isinstance(v, str) and v.strip():
                try:
                    p = Path(v).expanduser().resolve()
                    p.mkdir(parents=True, exist_ok=True)
                    if p.exists() and p.is_dir():
                        overrides[k] = p
                except Exception:
                    LOG.exception("Invalid codex_workdir for profile=%s: %r", profile_name, v)
        elif k in ("codex_force_full_access", "codex_dangerous_bypass_sandbox", "unsafe_direct_codex"):
            v = _profile_value(profiles, profile_name, k, None)
            if isinstance(v, bool):
                overrides[k] = v

    if not overrides:
        return cfg
    return BotConfig(**{**cfg.__dict__, **overrides})


def _profile_max_mode(cfg: "BotConfig", *, profile_name: str) -> str:
    if not profile_name:
        return "full"
    profiles = _load_profiles(cfg)
    v = _profile_value(profiles, profile_name, "max_mode", "full")
    if isinstance(v, str) and v.strip().lower() in ("ro", "rw", "full"):
        return v.strip().lower()
    return "full"


def _mode_rank(m: str) -> int:
    if m == "ro":
        return 0
    if m == "rw":
        return 1
    return 2


def _profile_allows_mode(cfg: "BotConfig", *, profile_name: str, requested: str) -> bool:
    return _mode_rank(requested) <= _mode_rank(_profile_max_mode(cfg, profile_name=profile_name))


def _profile_can_set_permissions(cfg: "BotConfig", *, profile_name: str) -> bool:
    if not profile_name:
        return True
    profiles = _load_profiles(cfg)
    v = _profile_value(profiles, profile_name, "can_set_permissions", True)
    return bool(v) if isinstance(v, bool) else True


def _profile_can_manage_bot(cfg: "BotConfig", *, profile_name: str) -> bool:
    if not profile_name:
        return True
    profiles = _load_profiles(cfg)
    v = _profile_value(profiles, profile_name, "can_manage_bot", True)
    return bool(v) if isinstance(v, bool) else True


def _auth_required_text() -> str:
    return "\n".join(
        [
            "Hola, soy PonceBot. Bienvenido.",
            "",
            "Para empezar, inicia sesion con:",
            "- /login <usuario> <password>",
        ]
    )

def _get_threads_state(cfg: "BotConfig") -> dict[str, str]:
    """
    Returns mapping {chat_id_str: thread_id}.
    Stored in cfg.state_file as {"threads": {...}}.
    """
    st = _get_state(cfg)
    raw = st.get("threads")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        kk = k.strip()
        vv = v.strip()
        if kk and vv:
            out[kk] = vv
    return out


def _persist_thread_id(cfg: "BotConfig", *, chat_id: int, thread_id: str) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    st = _get_state(cfg)
    raw = st.get("threads")
    threads: dict[str, Any] = raw if isinstance(raw, dict) else {}
    threads[str(int(chat_id))] = tid
    st["threads"] = threads
    _atomic_write_json(cfg.state_file, st)


def _clear_persisted_thread_id(cfg: "BotConfig", *, chat_id: int) -> None:
    st = _get_state(cfg)
    raw = st.get("threads")
    if not isinstance(raw, dict):
        return
    key = str(int(chat_id))
    if key not in raw:
        return
    raw.pop(key, None)
    st["threads"] = raw
    _atomic_write_json(cfg.state_file, st)


def _get_model_overrides(cfg: "BotConfig", *, chat_id: int | None = None) -> tuple[str, str]:
    """
    Returns (openai_model_override, oss_model_override) as strings (possibly empty).
    """
    st = _get_state(cfg)
    openai_model = ""
    oss_model = ""
    if chat_id is not None:
        by_chat = st.get("model_overrides_by_chat")
        if isinstance(by_chat, dict):
            rec = by_chat.get(str(int(chat_id)))
            if isinstance(rec, dict):
                openai_model = rec.get("openai_model") or ""
                oss_model = rec.get("oss_model") or ""
    if not openai_model and not oss_model:
        # Back-compat: fall back to legacy global keys.
        openai_model = st.get("openai_model") or ""
        oss_model = st.get("oss_model") or ""
    if not isinstance(openai_model, str):
        openai_model = ""
    if not isinstance(oss_model, str):
        oss_model = ""
    return _sanitize_model_id(openai_model), _sanitize_model_id(oss_model)


def _sanitize_model_id(model: str) -> str:
    """
    Model ids passed to `codex --model` should be a single token (no whitespace).
    Be conservative and drop obviously-invalid values rather than breaking all future runs.
    """
    m = (model or "").strip()
    if not m:
        return ""
    if any(ch.isspace() for ch in m):
        return ""
    # Allow typical OpenAI + local provider ids.
    if not re.fullmatch(r"[A-Za-z0-9_.:/+-]+", m):
        return ""
    return m


def _effective_models(cfg: "BotConfig", *, chat_id: int | None = None) -> tuple[str, str]:
    """
    Returns (openai_model, oss_model) after applying state overrides.
    """
    openai_override, oss_override = _get_model_overrides(cfg, chat_id=chat_id)
    openai_model = openai_override or cfg.codex_openai_model
    oss_model = oss_override or cfg.codex_oss_model
    return openai_model, oss_model


def _codex_config_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def _codex_models_cache_path() -> Path:
    return Path.home() / ".codex" / "models_cache.json"


def _codex_models_from_cache(*, max_models: int = 50) -> list[dict[str, Any]]:
    """
    Reads ~/.codex/models_cache.json (written by Codex CLI) and returns a list of model dicts.
    This is the closest approximation to what the interactive /model picker shows.
    """
    p = _codex_models_cache_path()
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    except Exception:
        LOG.exception("Failed to read models cache: %s", p)
        return []

    try:
        parsed = json.loads(raw)
    except Exception:
        LOG.exception("Failed to parse models cache JSON: %s", p)
        return []

    models = parsed.get("models") if isinstance(parsed, dict) else None
    if not isinstance(models, list):
        return []

    # Sort by "priority" (interactive UI often puts the newest/best first), then slug.
    def key_fn(m: Any) -> tuple[int, str]:
        if not isinstance(m, dict):
            return (10_000, "")
        pr = m.get("priority")
        pr_i = int(pr) if isinstance(pr, int) else 10_000
        slug = m.get("slug")
        slug_s = slug if isinstance(slug, str) else ""
        return (pr_i, slug_s)

    out: list[dict[str, Any]] = []
    for m in sorted(models, key=key_fn):
        if isinstance(m, dict):
            out.append(m)
            if max_models > 0 and len(out) >= max_models:
                break
    return out


def _model_choices_for_display() -> list[tuple[str, str, str, list[str]]]:
    """
    Returns list of (slug, display_name, default_effort, supported_efforts).
    """
    out: list[tuple[str, str, str, list[str]]] = []
    for m in _codex_models_from_cache(max_models=50):
        slug = m.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        slug = slug.strip()
        dn = m.get("display_name")
        display = (dn.strip() if isinstance(dn, str) and dn.strip() else slug)
        de = m.get("default_reasoning_level")
        default_effort = de.strip() if isinstance(de, str) else ""
        effs: list[str] = []
        srl = m.get("supported_reasoning_levels")
        if isinstance(srl, list):
            for e in srl:
                if isinstance(e, dict) and isinstance(e.get("effort"), str):
                    eff = e["effort"].strip()
                    if eff:
                        effs.append(eff)
        # De-dupe while preserving order.
        seen: set[str] = set()
        effs2: list[str] = []
        for e in effs:
            if e not in seen:
                seen.add(e)
                effs2.append(e)
        out.append((slug, display, default_effort, effs2))
    return out


def _get_effort_overrides(cfg: "BotConfig", *, chat_id: int | None = None) -> tuple[str, str]:
    """
    Returns (openai_effort_override, oss_effort_override) as strings (possibly empty).
    """
    st = _get_state(cfg)
    openai_effort = ""
    oss_effort = ""
    if chat_id is not None:
        by_chat = st.get("effort_overrides_by_chat")
        if isinstance(by_chat, dict):
            rec = by_chat.get(str(int(chat_id)))
            if isinstance(rec, dict):
                openai_effort = rec.get("openai_effort") or ""
                oss_effort = rec.get("oss_effort") or ""
    if not openai_effort and not oss_effort:
        # Back-compat: fall back to legacy global keys.
        openai_effort = st.get("openai_effort") or ""
        oss_effort = st.get("oss_effort") or ""
    if not isinstance(openai_effort, str):
        openai_effort = ""
    if not isinstance(oss_effort, str):
        oss_effort = ""
    return _sanitize_effort(openai_effort), _sanitize_effort(oss_effort)


def _normalize_effort_token(tok: str) -> str:
    """
    Accept effort in forms like: xhigh, [xhigh], <xhigh>, (xhigh), "xhigh"
    """
    t = (tok or "").strip().lower()
    if not t:
        return ""
    # Strip common wrappers.
    t = t.strip("[](){}<>\"'`")
    return t


def _sanitize_effort(effort: str) -> str:
    e = _normalize_effort_token(effort)
    if e in ("low", "medium", "high", "xhigh"):
        return e
    return ""


def _effective_efforts(cfg: "BotConfig", *, chat_id: int | None = None) -> tuple[str, str]:
    """
    Returns (openai_effort, oss_effort) after applying state overrides.
    For OpenAI mode, the config.toml default is used when not overridden.
    """
    openai_override, oss_override = _get_effort_overrides(cfg, chat_id=chat_id)
    _, cfg_effort = _codex_defaults_from_config()
    openai_effort = openai_override or cfg_effort
    oss_effort = oss_override or cfg_effort
    return openai_effort, oss_effort


def _codex_defaults_from_config(path: Path | None = None) -> tuple[str, str]:
    """
    Best-effort read of Codex CLI config to surface the *actual* default model + reasoning effort
    when the bot isn't explicitly passing `--model` for OpenAI mode.

    Returns: (model, model_reasoning_effort) as strings (possibly empty).
    """
    if tomllib is None:
        return "", ""
    p = path or _codex_config_path()
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return "", ""
    except Exception:
        LOG.exception("Failed to read codex config: %s", p)
        return "", ""

    try:
        cfg = tomllib.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        LOG.exception("Failed to parse codex config TOML: %s", p)
        return "", ""

    model = cfg.get("model") if isinstance(cfg, dict) else None
    effort = cfg.get("model_reasoning_effort") if isinstance(cfg, dict) else None
    model = model.strip() if isinstance(model, str) else ""
    effort = effort.strip() if isinstance(effort, str) else ""
    return model, effort


def _extract_model_from_argv(argv: list[str]) -> str:
    for i, a in enumerate(argv):
        if a in ("-m", "--model"):
            if i + 1 < len(argv):
                v = argv[i + 1]
                return v.strip() if isinstance(v, str) else ""
            return ""
        if a.startswith("--model="):
            return a.split("=", 1)[1].strip()
    return ""


def _extract_config_override_from_argv(argv: list[str], *, key: str) -> str:
    def _normalize_val(v: str) -> str:
        v = v.strip()
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1].strip()
        return v

    for i, a in enumerate(argv):
        if a in ("-c", "--config"):
            if i + 1 >= len(argv):
                continue
            kv = argv[i + 1]
            if not isinstance(kv, str) or "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            if k.strip() == key:
                return _normalize_val(v)
        if isinstance(a, str) and a.startswith(("-c", "--config=")):
            kv = a.split("=", 1)[1] if a.startswith("--config=") else a[2:]
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            if k.strip() == key:
                return _normalize_val(v)
    return ""


def _extract_effort_override_from_argv(argv: list[str]) -> str:
    """
    Parse `-c key=value` overrides looking for `model_reasoning_effort`.
    We don't try to fully interpret TOML; we just extract common string forms.
    """
    return _extract_config_override_from_argv(argv, key="model_reasoning_effort")


def _job_model_label(cfg: "BotConfig", argv: list[str], *, chat_id: int | None = None) -> tuple[str, str]:
    """
    Returns (model, reasoning_effort) for UX display.
    """
    argv = list(argv or [])
    cmd = argv[0] if argv else "exec"
    if cmd.startswith("-"):
        return "", ""

    # Common across commands: allow `-c model_reasoning_effort=...`
    effort = _extract_effort_override_from_argv(argv)
    if not effort:
        openai_effort, oss_effort = _effective_efforts(cfg, chat_id=chat_id)
        effort = oss_effort if cfg.codex_use_oss else openai_effort

    if cmd == "review":
        # Top-level `codex review` doesn't expose `--model`/`--oss`; use config overrides instead.
        model = _extract_config_override_from_argv(argv, key="model")
        if not model:
            model, _ = _codex_defaults_from_config()
        return model, effort

    if cmd != "exec":
        return "", ""

    model = _extract_model_from_argv(argv) or _extract_config_override_from_argv(argv, key="model")

    if cfg.codex_use_oss:
        # OSS: if user didn't pass --model, we pass/choose cfg defaults (or state overrides).
        if not model:
            _, model = _effective_models(cfg, chat_id=chat_id)
        return model, effort

    # OpenAI/default provider: only pass --model when explicitly configured. Otherwise codex uses ~/.codex/config.toml.
    if not model:
        model, _ = _effective_models(cfg, chat_id=chat_id)
        if not model:
            model, _ = _codex_defaults_from_config()
    return model, effort


def _pretty_model_name(model: str) -> str:
    """
    Human-friendly model name for UX. Keep the raw model id for exactness.
    """
    m = (model or "").strip()
    if not m:
        return ""
    # Common Codex CLI model ids look like: gpt-5.3-codex
    mm = re.match(r"^gpt-(\d+(?:\.\d+)+)-codex(?:$|-)", m)
    if mm:
        return f"codex {mm.group(1)}"
    return ""


def _format_model_for_display(model: str, effort: str) -> str:
    if not (model or "").strip() and not (effort or "").strip():
        return "n/a"
    raw = (model or "").strip() or "(unknown)"
    eff = (effort or "").strip()
    pretty = _pretty_model_name(raw)
    parts: list[str] = [raw]
    if pretty and pretty != raw:
        parts.append(f"({pretty})")
    if eff:
        parts.append(f"effort={eff}")
    return " ".join(parts)


def _redact_codex_cmd_for_log(cmd: list[str]) -> list[str]:
    """
    Best-effort redaction for logs: keep flags and structure, but avoid logging user prompts/transcripts.
    """
    out = list(cmd or [])
    if not out:
        return out

    def _redact_from(i: int) -> None:
        for j in range(i, len(out)):
            a = out[j]
            if not isinstance(a, str):
                continue
            if a.startswith("-"):
                continue
            out[j] = "<redacted>"

    try:
        if "exec" in out:
            i = out.index("exec")
            if i + 1 < len(out) and out[i + 1] == "resume":
                # exec resume <thread_id> <prompt>
                _redact_from(i + 3)
                return out
            # exec <prompt>
            _redact_from(i + 1)
            return out
        if "review" in out:
            i = out.index("review")
            _redact_from(i + 1)
            return out
    except Exception:
        return ["codex", "<redacted>"]
    return out


class CodexRunner:
    def __init__(self, cfg: BotConfig, *, chat_id: int | None = None) -> None:
        self._cfg = cfg
        self._chat_id = chat_id
    
    def _bypass_sandbox(self) -> bool:
        return _effective_bypass_sandbox(self._cfg, chat_id=self._chat_id)

    @dataclass(frozen=True)
    class Running:
        proc: subprocess.Popen[object]
        start_time: float
        cmd: list[str]
        last_msg_path: Path | None
        stdout_path: Path
        stderr_path: Path

    def _start_with_cmd(self, *, cmd: list[str], last_msg_path: Path | None) -> "CodexRunner.Running":
        start_time = time.time()
        LOG.info("Running: %s", " ".join(_redact_codex_cmd_for_log(cmd)))

        # Keep environment small-ish; codex still needs PATH.
        env = os.environ.copy()
        # Avoid leaking Telegram secrets to subprocesses unless explicitly needed.
        env.pop("TELEGRAM_BOT_TOKEN", None)
        env.pop("TELEGRAM_ALLOWED_CHAT_IDS", None)
        env.pop("TELEGRAM_ALLOWED_USER_IDS", None)

        # Codex sandboxing may block writes outside the workdir (including /tmp). Force a per-workdir temp dir
        # so QA/tests and other tools relying on tempfile.* keep working in workspace-write mode.
        try:
            tmp_root = (self._cfg.codex_workdir / ".codexbot_tmp").resolve()
            tmp_root.mkdir(parents=True, exist_ok=True)
            env["TMPDIR"] = str(tmp_root)
            env["TMP"] = str(tmp_root)
            env["TEMP"] = str(tmp_root)
        except Exception:
            pass
        # Frontend evidence dir: agents can drop `.codexbot_preview/preview.html` and the bot will screenshot it.
        try:
            preview_root = (self._cfg.codex_workdir / ".codexbot_preview").resolve()
            preview_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        out_f = tempfile.NamedTemporaryFile(prefix="codexbot_stdout_", suffix=".log", delete=False)
        err_f = tempfile.NamedTemporaryFile(prefix="codexbot_stderr_", suffix=".log", delete=False)
        stdout_path = Path(out_f.name)
        stderr_path = Path(err_f.name)

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self._cfg.codex_workdir),
                env=env,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            # Clean up temp files if we couldn't start at all.
            try:
                out_f.close()
                err_f.close()
            finally:
                stdout_path.unlink(missing_ok=True)
                stderr_path.unlink(missing_ok=True)
                if last_msg_path:
                    last_msg_path.unlink(missing_ok=True)
            raise e
        finally:
            try:
                out_f.close()
            except Exception:
                pass
            try:
                err_f.close()
            except Exception:
                pass

        return CodexRunner.Running(
            proc=proc,
            start_time=start_time,
            cmd=cmd,
            last_msg_path=last_msg_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    def start(self, *, argv: list[str], mode_hint: str) -> "CodexRunner.Running":
        cmd, last_msg_path = self._build_cmd(argv=argv, mode_hint=mode_hint)
        return self._start_with_cmd(cmd=cmd, last_msg_path=last_msg_path)

    def start_threaded_new(
        self,
        *,
        prompt: str,
        mode_hint: str,
        image_paths: list[Path] | None = None,
        model_override: str | None = None,
        effort_override: str | None = None,
    ) -> "CodexRunner.Running":
        """
        Starts a brand-new Codex thread (session). Uses `--json` so the caller can extract `thread_id`
        from stdout, and `--output-last-message` to capture the final assistant message.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Empty prompt")

        # Global options must appear before the subcommand.
        cmd: list[str] = ["codex"]
        if self._bypass_sandbox():
            # No sandboxing at all (host must be externally sandboxed).
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            sandbox = self._threaded_sandbox_mode(mode_hint)
            cmd += ["-a", "never", "--sandbox", sandbox]

        # Apply reasoning effort override (if any). Mirrors the interactive /model effort picker.
        eff = _sanitize_effort(effort_override or "")
        if not eff:
            openai_effort, oss_effort = _effective_efforts(self._cfg, chat_id=self._chat_id)
            eff = oss_effort if self._cfg.codex_use_oss else openai_effort
        if eff:
            cmd += ["-c", f'model_reasoning_effort="{eff}"']
        cmd += ["-C", str(self._cfg.codex_workdir)]

        last_msg_file = tempfile.NamedTemporaryFile(prefix="codexbot_codex_last_", suffix=".txt", delete=False)
        last_msg_path = Path(last_msg_file.name)
        last_msg_file.close()

        # `--oss` is safest after `exec` (Codex quirk).
        argv: list[str] = ["exec"]
        if image_paths:
            for p in image_paths:
                argv += ["--image", str(p)]
        if self._cfg.codex_use_oss:
            argv += ["--oss", "--local-provider", self._cfg.codex_local_provider]

        # Apply model defaults/overrides.
        active_model = _sanitize_model_id(model_override or "")
        if not active_model:
            openai_model, oss_model = _effective_models(self._cfg, chat_id=self._chat_id)
            active_model = oss_model if self._cfg.codex_use_oss else openai_model
        if active_model:
            argv += ["--model", active_model]

        # Keep stdout machine-readable; the human output comes from `--output-last-message`.
        argv += ["--json", "--output-last-message", str(last_msg_path)]

        if not (self._cfg.codex_workdir / ".git").exists():
            argv.append("--skip-git-repo-check")

        argv.append(prompt)
        cmd += argv
        return self._start_with_cmd(cmd=cmd, last_msg_path=last_msg_path)

    def start_threaded_resume(
        self,
        *,
        thread_id: str,
        prompt: str,
        mode_hint: str,
        image_paths: list[Path] | None = None,
        model_override: str | None = None,
        effort_override: str | None = None,
    ) -> "CodexRunner.Running":
        """
        Resumes an existing Codex thread (session) using `codex exec resume <thread_id>`.
        This does not use `--output-last-message` (not supported by resume); the final response is read from stdout.
        """
        tid = (thread_id or "").strip()
        prompt = (prompt or "").strip()
        if not tid:
            raise ValueError("Empty thread_id")
        if not prompt:
            raise ValueError("Empty prompt")

        cmd: list[str] = ["codex"]
        if self._bypass_sandbox():
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            sandbox = self._threaded_sandbox_mode(mode_hint)
            cmd += ["-a", "never", "--sandbox", sandbox]

        eff = _sanitize_effort(effort_override or "")
        if not eff:
            openai_effort, oss_effort = _effective_efforts(self._cfg, chat_id=self._chat_id)
            eff = oss_effort if self._cfg.codex_use_oss else openai_effort
        if eff:
            cmd += ["-c", f'model_reasoning_effort="{eff}"']
        cmd += ["-C", str(self._cfg.codex_workdir), "exec", "resume", tid]

        if not (self._cfg.codex_workdir / ".git").exists():
            cmd.append("--skip-git-repo-check")

        # Apply model defaults/overrides (resume supports --model).
        # If empty, Codex will use config defaults for the resumed session.
        active_model = _sanitize_model_id(model_override or "")
        if not active_model:
            openai_model, oss_model = _effective_models(self._cfg, chat_id=self._chat_id)
            active_model = oss_model if self._cfg.codex_use_oss else openai_model
        if active_model:
            cmd += ["--model", active_model]

        if image_paths:
            for p in image_paths:
                cmd += ["--image", str(p)]

        cmd.append(prompt)
        return self._start_with_cmd(cmd=cmd, last_msg_path=None)

    def _threaded_sandbox_mode(self, mode_hint: str) -> str:
        if mode_hint not in ("ro", "rw", "full"):
            raise ValueError(f"Invalid mode hint: {mode_hint}")
        effective_mode = "full" if self._cfg.codex_force_full_access else mode_hint
        if effective_mode == "ro":
            return "read-only"
        if effective_mode == "rw":
            return "workspace-write"
        return "danger-full-access"

    def _build_cmd(self, *, argv: list[str], mode_hint: str) -> tuple[list[str], Path | None]:
        if mode_hint not in ("ro", "rw", "full"):
            raise ValueError(f"Invalid mode hint: {mode_hint}")

        # Note: `--oss` must be placed after the `exec` subcommand, otherwise Codex may ignore it.
        # `--dangerously-bypass-approvals-and-sandbox` is incompatible with `-a/--ask-for-approval`.
        cmd: list[str] = ["codex"]

        last_msg_path: Path | None = None

        # If caller didn't specify a subcommand, assume exec.
        argv = list(argv)
        if not argv:
            argv = ["exec"]

        # Unsafe global bypass (applies to any command). This skips Codex sandboxing entirely.
        if self._bypass_sandbox() and "--dangerously-bypass-approvals-and-sandbox" not in argv:
            argv = ["--dangerously-bypass-approvals-and-sandbox"] + argv
        if "--dangerously-bypass-approvals-and-sandbox" not in argv:
            cmd += ["-a", "never"]

        # Defaults are applied only to `exec` calls.
        if argv[0] == "exec":
            # Ensure codex uses the intended provider mode if user didn't specify otherwise.
            if self._cfg.codex_use_oss and "--oss" not in argv:
                argv[1:1] = ["--oss", "--local-provider", self._cfg.codex_local_provider]

            # Apply reasoning effort defaults/overrides: only if user didn't already pass `-c model_reasoning_effort=...`.
            if not _extract_effort_override_from_argv(argv):
                openai_effort, oss_effort = _effective_efforts(self._cfg, chat_id=self._chat_id)
                eff = oss_effort if self._cfg.codex_use_oss else openai_effort
                if eff:
                    argv[1:1] = ["-c", f'model_reasoning_effort="{eff}"']

            # Apply model defaults/overrides: only if user didn't already pass `--model`.
            if "--model" not in argv and "-m" not in argv:
                if self._cfg.codex_use_oss:
                    _, oss_model = _effective_models(self._cfg, chat_id=self._chat_id)
                    if oss_model:
                        argv[1:1] = ["--model", oss_model]
                else:
                    openai_model, _ = _effective_models(self._cfg, chat_id=self._chat_id)
                    if openai_model:
                        argv[1:1] = ["--model", openai_model]

            # Make output parseable.
            if "--color" not in argv:
                argv[1:1] = ["--color", "never"]

            # Set working dir unless user passed -C/--cd.
            if "-C" not in argv and "--cd" not in argv:
                argv[1:1] = ["-C", str(self._cfg.codex_workdir)]

            # Only bypass git repo check when needed.
            if "--skip-git-repo-check" not in argv and not (self._cfg.codex_workdir / ".git").exists():
                argv.append("--skip-git-repo-check")

            # Provide a last-message file unless user asked for JSON events.
            if "--json" not in argv and "--output-last-message" not in argv and "-o" not in argv:
                last_msg_file = tempfile.NamedTemporaryFile(prefix="codexbot_codex_last_", suffix=".txt", delete=False)
                last_msg_path = Path(last_msg_file.name)
                last_msg_file.close()
                argv += ["--output-last-message", str(last_msg_path)]

            # Sandbox default (only if user didn't choose one).
            if "--sandbox" not in argv and "--full-auto" not in argv and "--dangerously-bypass-approvals-and-sandbox" not in argv:
                effective_mode = "full" if self._cfg.codex_force_full_access else mode_hint
                if effective_mode == "ro":
                    sandbox = "read-only"
                elif effective_mode == "rw":
                    sandbox = "workspace-write"
                else:
                    sandbox = "danger-full-access"
                argv += ["--sandbox", sandbox]

        cmd += argv
        return cmd, last_msg_path


def _tail_text(s: str, *, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def _read_text_file(path: Path, *, max_bytes: int | None = None) -> str:
    try:
        b = path.read_bytes()
    except FileNotFoundError:
        return ""
    except Exception:
        LOG.exception("Failed to read file: %s", path)
        return ""
    if max_bytes is not None and max_bytes >= 0 and len(b) > max_bytes:
        b = b[-max_bytes:]
    return b.decode("utf-8", errors="replace")


def _tail_file_text(path: Path, *, max_chars: int) -> str:
    """
    Returns the last max_chars characters from a potentially large file.
    """
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return ""
            # Read a bit more than max_chars to reduce the chance of cutting multi-byte chars.
            max_bytes = min(size, max(4096, max_chars * 4))
            f.seek(-max_bytes, os.SEEK_END)
            b = f.read()
    except FileNotFoundError:
        return ""
    except Exception:
        LOG.exception("Failed to tail file: %s", path)
        return ""
    s = b.decode("utf-8", errors="replace")
    s = _strip_ansi(s)
    return _tail_text(s, max_chars=max_chars).strip()


def _render_placeholders_text(text: str, *, ceo_name: str) -> str:
    """
    Very small templating layer for prompts/docs.

    Ground truth: we only support the placeholders we explicitly use in this repo.
    """
    if not isinstance(text, str) or not text:
        return ""
    out = text
    out = out.replace("{CEO_NAME}", ceo_name)
    return out


def _render_placeholders_obj(obj: Any, *, ceo_name: str) -> Any:
    """
    Recursively render placeholders in nested dict/list structures (agent profiles, etc.).
    """
    if isinstance(obj, str):
        return _render_placeholders_text(obj, ceo_name=ceo_name)
    if isinstance(obj, list):
        return [_render_placeholders_obj(x, ceo_name=ceo_name) for x in obj]
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            out[k] = _render_placeholders_obj(v, ceo_name=ceo_name)
        return out
    return obj


def _extract_retry_after_seconds(body: str) -> float | None:
    try:
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            return None
        params = parsed.get("parameters") or {}
        if isinstance(params, dict) and isinstance(params.get("retry_after"), (int, float)):
            return float(params["retry_after"])
    except Exception:
        return None
    return None


def _sleep_retry(attempt: int, initial: float, max_s: float, retry_after: float | None) -> None:
    base = initial if initial > 0 else 1.0
    cap = max_s if max_s > 0 else base
    delay = min(cap, base * (2.0**attempt))
    if retry_after is not None:
        delay = max(delay, max(0.0, retry_after))
    time.sleep(delay)


def _ollama_status() -> tuple[bool, str]:
    # Minimal local probe (no deps): hits the default Ollama API endpoint.
    url = "http://127.0.0.1:11434/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return False, f"ollama: not reachable at {url} ({e})"

    try:
        parsed = json.loads(raw)
        models = parsed.get("models") or []
        names = [m.get("name") for m in models if isinstance(m, dict)]
        names = [n for n in names if isinstance(n, str)]
        if names:
            return True, "ollama: OK (" + ", ".join(names[:10]) + ("" if len(names) <= 10 else ", ...") + ")"
        return True, "ollama: OK (no models listed)"
    except json.JSONDecodeError:
        return True, "ollama: reachable (unexpected response)"


def _codex_version() -> str:
    try:
        proc = subprocess.run(["codex", "--version"], capture_output=True, text=True, timeout=5)
        out = (proc.stdout or proc.stderr or "").strip()
        return out or "codex: (no version output)"
    except Exception as e:
        return f"codex: not runnable ({e})"


def _help_text(cfg: BotConfig) -> str:
    model_line = ""
    if cfg.codex_use_oss:
        _, oss_model = _effective_models(cfg)
        _, effort = _codex_defaults_from_config()
        model_line = "Model: " + _format_model_for_display(oss_model or "(default)", effort)
    else:
        openai_model, _ = _effective_models(cfg)
        if openai_model:
            model = openai_model
            effort = _codex_defaults_from_config()[1]
        else:
            model, effort = _codex_defaults_from_config()
        if model:
            model_line = "Model: " + _format_model_for_display(model, effort)
        else:
            model_line = "Model: (unknown; check ~/.codex/config.toml or set CODEX_OPENAI_MODEL)"
    qmax = "unbounded" if cfg.queue_maxsize == 0 else str(cfg.queue_maxsize)

    lines: list[str] = [
        "codexbot commands:",
        "- /help               Show this help",
        "- /whoami             Show your ids (chat_id, user_id)",
    ]
    if cfg.auth_enabled:
        lines += [
            "- /login <u> <p>      Login",
            "- /logout             Logout",
        ]

    lines += [
        "- /status             Show legacy/system status and orchestrator queue",
        "- /agents             Show orchestrator role status and queue per role",
        "- /dashboard          Visual dashboard snapshot (PNG)",
        "- /watch              Live company status (single message, auto-updated)",
        "- /unwatch            Disable /watch",
        "- /orders             List CEO orders (autopilot scope)",
        "- /order show|pause|done <id>   Manage an order",
        "- /job <id>           Show task/job status by id",
        "- /daily              Show orchestrator digest now",
        "- /brief              Executive brief (short digest)",
        "- /approve <id>       Approve a blocked task",
        "- /emergency_stop     Stop all orchestrator tasks and pause all roles",
        "- /pause <role>       Pause role in orchestrator",
        "- /resume <role>      Resume role in orchestrator",
        "- /cancel <id>        Cancel orchestrator task by id",
        "- /purge [chat|global]  Purge queued+blocked orchestrator jobs (keeps running)",
        "- /emergency_resume   Resume orchestrator after emergency stop",
        "- /cancel             Cancel the running job (and drop queued jobs) for this chat",
        "- /new                Start a new Codex conversation thread for this chat",
        "- /restart            Restart the bot service (systemd will bring it back)",
        "- /thread             Show the current Codex thread id for this chat",
        f"- Strict proxy mode:  {'ON' if cfg.strict_proxy else 'off'} (forwards most text directly to Codex)",
        "- /setnotify          Save this chat as the notify target",
        "- /notify <text>      Send a message to the notify target chat",
        "- /synccommands       Re-sync Telegram slash command suggestions",
        "- /model              Show current model selection",
        "- /model <name>       Set model for current provider mode",
        "- /model openai <n>   Set model for OpenAI mode (CODEX_USE_OSS=0)",
        "- /model oss <n>      Set model for OSS mode (CODEX_USE_OSS=1)",
        "- /model clear        Clear model override for current mode",
        "- /m                  Alias for /model",
        "- /voice              Show/set voice transcription settings",
        "- /v                  Alias for /voice",
        "- /snapshot           Request frontend snapshot task (screenshot-oriented)",
        "- /effort             Show current reasoning effort",
        "- /effort <level>     Set effort: low|medium|high|xhigh",
        "- /effort clear       Clear effort override for current mode",
        "- /skills             List installed skills (local + disabled + .system)",
        "- /skills catalog     List installable curated skills (from openai/skills)",
        "- /skills install <s> Install a curated skill to ~/.codex/skills/<s>",
        "- /skills enable <s>  Re-enable a previously disabled local skill",
        "- /skills disable <s> Disable a local skill (moves it under ~/.codex/skills/.disabled/)",
        "- /permissions        Show/set Codex CLI permission options",
        "- /p                  Alias for /permissions",
        "- /botpermissions     Show bot + codex execution policy",
        "- /format             Show Telegram formatting preview",
        "- /example            Show a pretty formatted example",
        "- /reset              Alias for /new (new thread)",
        "- /x                  Alias for /cancel",
        "",
        "Codex passthrough:",
        "- Plain text runs: codex exec (threaded per chat, using codex exec resume)",
        "- /exec ... runs:   codex exec ...",
        "- /review ... runs: codex review ...",
        "- /codex ... runs:  codex ...",
        "",
        "Sandbox shortcuts:",
        "- /ro <text> runs exec with default read-only sandbox",
        "- /rw <text> runs exec with default workspace-write sandbox",
        "- /full <text> runs exec with danger-full-access sandbox (unsafe)",
        "",
        "Attachments:",
        "- If Codex creates or references *.png files inside the workdir, they will be sent as images automatically.",
        "- If you send a Telegram document (file), it will be saved under .codexbot_uploads/ and Codex will be told the path.",
        "- If you send a voice note / audio and transcription is enabled (BOT_TRANSCRIBE_AUDIO=1), it will be transcribed and treated as normal text.",
        "",
        f"Default mode for plain text: {cfg.codex_default_mode}",
        f"Workdir: {cfg.codex_workdir}",
        f"Provider: {cfg.codex_local_provider if cfg.codex_use_oss else 'default (non-oss)'}",
        model_line,
        f"Queue maxsize: {qmax}",
    ]

    return "\n".join(lines)


def _telegram_commands_for_suggestions(cfg: BotConfig) -> list[tuple[str, str]]:
    """
    Command list shown by Telegram UI when user types "/".
    Keep descriptions short and action-oriented.
    """
    cmds: list[tuple[str, str]] = [
        ("help", "Show help"),
        ("agents", "Orchestrator status"),
        ("dashboard", "Visual dashboard (PNG)"),
        ("watch", "Live company status"),
        ("orders", "List CEO orders"),
        ("status", "Bot/model status"),
        ("s", "Alias for /status"),
        ("whoami", "Show your IDs"),
        ("purge", "Clear queue (queued+blocked)"),
    ]
    if cfg.auth_enabled:
        cmds += [
            ("login", "Login"),
            ("logout", "Logout"),
        ]
    cmds += [
        ("order", "Show/pause/done order"),
        ("job", "Show job status"),
        ("ticket", "Show ticket tree"),
        ("inbox", "Inbox by role"),
        ("runbooks", "Show runbooks"),
        ("reset_role", "Reset role memory"),
        ("emergency_stop", "Stop orchestrator"),
        ("emergency_resume", "Resume orchestrator"),
        ("daily", "Digest now"),
        ("brief", "Executive brief"),
        ("snapshot", "Request UI snapshot"),
        ("approve", "Approve blocked job"),
        ("pause", "Pause role"),
        ("resume", "Resume role"),
        ("new", "New Codex thread"),
        ("thread", "Show current thread"),
        ("cancel", "Cancel job(s)"),
        ("model", "Show/set model"),
        ("m", "Alias for /model"),
        ("voice", "Voice transcription"),
        ("v", "Alias for /voice"),
        ("permissions", "Codex permissions"),
        ("p", "Alias for /permissions"),
        ("skills", "Manage skills"),
        ("synccommands", "Re-sync commands"),
    ]
    return cmds


def _telegram_command_scopes_for_suggestions() -> tuple[str, ...]:
    """
    Command scopes so suggestions appear in private chats and groups.
    """
    return ("default", "all_private_chats", "all_group_chats", "all_chat_administrators")


def _sync_telegram_command_suggestions(api: TelegramAPI, cfg: BotConfig) -> None:
    cmds = _telegram_commands_for_suggestions(cfg)
    scopes = _telegram_command_scopes_for_suggestions()
    synced = 0
    errors: list[str] = []
    for scope in scopes:
        try:
            api.set_my_commands(cmds, scope_type=scope)
            synced += 1
        except Exception as e:
            errors.append(f"{scope}: {e}")

    if synced == 0:
        detail = "; ".join(errors[:2]) if errors else "unknown error"
        raise RuntimeError(f"setMyCommands failed for all scopes ({detail})")

    if errors:
        LOG.warning(
            "Telegram command suggestions synced partially (%d/%d): %s",
            synced,
            len(scopes),
            "; ".join(errors[:2]),
        )

    try:
        api.set_chat_menu_button_commands()
    except Exception:
        LOG.exception("Failed to set Telegram chat menu button")


def _whoami_text(msg: IncomingMessage) -> str:
    uname = msg.username or "(none)"
    return f"chat_id={msg.chat_id}\nuser_id={msg.user_id}\nusername={uname}"


def _maybe_handle_ceo_query(
    *,
    api: TelegramAPI,
    cfg: BotConfig,
    msg: IncomingMessage,
    orchestrator_profiles: dict[str, dict[str, Any]] | None,
    orchestrator_queue: OrchestratorQueue | None,
) -> bool:
    """
    Deterministic "CEO queries" that should not go to Codex and should not delegate.

    Grounded motivation: these are questions the bot can answer locally (Telegram metadata + agent config).
    """
    raw = (msg.text or "").strip()
    if not raw:
        return False
    if raw.startswith("/"):
        return False

    t = raw.lower()
    # If the CEO explicitly targets an employee (@backend/@frontend/...), don't steal it.
    if "@" in t and any(m in t for m in ("@jarvis", "@frontend", "@backend", "@qa", "@sre", "@orchestrator", "@ceo")):
        return False

    req_type = detect_request_type(t)
    if req_type not in ("query", "status"):
        return False

    # Helpers (best-effort; no secrets).
    display_user = " ".join(
        [str(x).strip() for x in (getattr(msg, "username", None),) if isinstance(x, str) and x.strip()]
    ).strip()
    if display_user:
        display_user = "@" + display_user.lstrip("@")
    else:
        display_user = "(no username)"

    # Status requests: answer from the orchestrator directly (no Codex, no ticket).
    if req_type == "status":
        if orchestrator_queue is None:
            return False
        try:
            health = orchestrator_queue.get_role_health()
            queued = int(orchestrator_queue.get_queued_count())
            running_n = int(orchestrator_queue.get_running_count())
            blocked = 0
            for rec in (health or {}).values():
                try:
                    blocked += int((rec or {}).get("blocked", 0))
                except Exception:
                    pass
            running = orchestrator_queue.jobs_by_state(state="running", limit=5)
            queued_samples = orchestrator_queue.peek(state="queued", limit=30)
            blocked_samples = orchestrator_queue.peek(state="blocked", limit=30)
        except Exception:
            health = {}
            queued = 0
            running_n = 0
            blocked = 0
            running = []
            queued_samples = []
            blocked_samples = []

        lines: list[str] = []
        lines.append("Jarvis: team status")
        lines.append(f"- queue: queued={queued} running={running_n} blocked={blocked}")
        if running:
            parts: list[str] = []
            for r in running[:5]:
                role_h = _humanize_orchestrator_role(r.role)
                snippet = (r.input_text or "").strip().replace("\n", " ")
                if len(snippet) > 70:
                    snippet = snippet[:70] + "..."
                parts.append(f"{role_h}: {snippet}")
            lines.append("- running: " + " | ".join(parts))

        def _sample_line(label: str, items: list[Task]) -> str | None:
            by_role: dict[str, Task] = {}
            for it in items:
                r = str(it.role or "").strip().lower()
                if not r or r in by_role:
                    continue
                by_role[r] = it
                if len(by_role) >= 5:
                    break
            if not by_role:
                return None
            parts: list[str] = []
            for r in sorted(by_role.keys()):
                it = by_role[r]
                role_h = _humanize_orchestrator_role(it.role)
                snippet = (it.input_text or "").strip().replace("\n", " ")
                if len(snippet) > 70:
                    snippet = snippet[:70] + "..."
                parts.append(f"{role_h}: {snippet}")
            return f"- {label}: " + " | ".join(parts)

        q_line = _sample_line("queued (examples)", queued_samples)
        if q_line:
            lines.append(q_line)
        b_line = _sample_line("blocked (examples)", blocked_samples)
        if b_line:
            lines.append(b_line)

        lines.append("Links: /agents  /dashboard")
        api.send_message(
            msg.chat_id,
            "\n".join(lines),
            reply_to_message_id=msg.message_id if msg.message_id else None,
        )
        return True

    if any(k in t for k in ("who am i", "quien soy", "quién soy")):
        api.send_message(
            msg.chat_id,
            "\n".join(
                [
                    f"Jarvis: CEO = {cfg.ceo_name}",
                    f"Jarvis: Telegram user_id={msg.user_id} chat_id={msg.chat_id} username={display_user}",
                ]
            ),
            reply_to_message_id=msg.message_id if msg.message_id else None,
        )
        return True

    if any(
        k in t
        for k in (
            "que tienes pendiente",
            "qué tienes pendiente",
            "pendientes",
            "pending",
            "backlog",
            "que falta",
            "qué falta",
            "what's next",
            "whats next",
        )
    ):
        if orchestrator_queue is None:
            api.send_message(
                msg.chat_id,
                "Jarvis: I can't read the orchestrator queue right now.",
                reply_to_message_id=msg.message_id if msg.message_id else None,
            )
            return True
        try:
            active = orchestrator_queue.list_orders(chat_id=int(msg.chat_id), status="active", limit=6)
            queued = int(orchestrator_queue.get_queued_count())
            running_n = int(orchestrator_queue.get_running_count())
            health = orchestrator_queue.get_role_health()
            blocked = 0
            for rec in (health or {}).values():
                try:
                    blocked += int((rec or {}).get("blocked", 0))
                except Exception:
                    pass
        except Exception:
            active = []
            queued = 0
            running_n = 0
            blocked = 0

        lines: list[str] = []
        lines.append("Jarvis: pending")
        if active:
            parts: list[str] = []
            for o in active[:6]:
                oid = str(o.get("order_id") or "")[:8]
                title = str(o.get("title") or "").strip()
                if len(title) > 60:
                    title = title[:60] + "..."
                parts.append(f"{oid} {title}".strip())
            lines.append("- active_orders: " + " | ".join(parts))
        else:
            lines.append("- active_orders: (none)")
        lines.append(f"- queue: queued={queued} running={running_n} blocked={blocked}")
        api.send_message(
            msg.chat_id,
            "\n".join(lines),
            reply_to_message_id=msg.message_id if msg.message_id else None,
        )
        return True

    if any(k in t for k in ("cuantos empleados", "cuántos empleados", "cuantos trabajadores", "how many employees", "how many agents")):
        profs = orchestrator_profiles or {}
        roles = sorted({str(k).strip().lower() for k in profs.keys() if str(k).strip()})
        if not roles:
            api.send_message(
                msg.chat_id,
                "I don't have an agent roster loaded right now (agents.yaml missing/unreadable).",
                reply_to_message_id=msg.message_id if msg.message_id else None,
            )
            return True
        api.send_message(
            msg.chat_id,
            "\n".join(
                [
                    f"Jarvis: employees (agents) = {len(roles)}",
                    "Jarvis: roles = " + ", ".join(roles),
                ]
            ),
            reply_to_message_id=msg.message_id if msg.message_id else None,
        )
        return True

    if any(k in t for k in ("qué modelos", "que modelos", "modelos usan", "modelo usan", "what models")):
        profs = orchestrator_profiles or {}
        if not profs:
            api.send_message(
                msg.chat_id,
                "I don't have agent profiles loaded right now (agents.yaml missing/unreadable).",
                reply_to_message_id=msg.message_id if msg.message_id else None,
            )
            return True
        lines = ["Default models by role:"]
        for role in sorted(profs.keys()):
            rec = profs.get(role) or {}
            model = str(rec.get("model") or "").strip() or "(default)"
            effort = str(rec.get("effort") or "").strip() or "(default)"
            lines.append(f"- {role}: model={model} effort={effort}")
        api.send_message(
            msg.chat_id,
            "Jarvis: models by role\n" + "\n".join(lines[1:40]),
            reply_to_message_id=msg.message_id if msg.message_id else None,
        )
        return True

    if any(k in t for k in ("what is sre", "que es sre", "qué es sre")):
        api.send_message(
            msg.chat_id,
            "\n".join(
                [
                    "Jarvis: SRE = Site Reliability Engineering.",
                    "Jarvis: They keep services reliable (monitoring, incidents, deploy safety, performance, uptime).",
                ]
            ),
            reply_to_message_id=msg.message_id if msg.message_id else None,
        )
        return True

    if any(k in t for k in ("equipo tenemos", "a quien tenemos en el equipo", "who is on the team", "team do we have")):
        profs = orchestrator_profiles or {}
        roles = sorted({str(k).strip().lower() for k in profs.keys() if str(k).strip()})
        if not roles:
            api.send_message(
                msg.chat_id,
                "No agent roster loaded right now.",
                reply_to_message_id=msg.message_id if msg.message_id else None,
            )
            return True
        api.send_message(
            msg.chat_id,
            "Jarvis: team roles = " + ", ".join(roles),
            reply_to_message_id=msg.message_id if msg.message_id else None,
        )
        return True

    # Query type but not one we can answer deterministically: let Jarvis handle it.
    return False


def _skills_status_text() -> str:
    skills_root = _skills_root_dir()
    sys_root = skills_root / ".system"
    disabled_root = skills_root / ".disabled"

    enabled: list[str] = []
    disabled: list[str] = []
    system: list[str] = []

    try:
        # Enabled local skills live directly under ~/.codex/skills/<skill>/SKILL.md
        for p in sorted(skills_root.glob("*/SKILL.md")):
            enabled.append(p.parent.name)
    except Exception:
        LOG.exception("Failed to list enabled skills under: %s", skills_root)

    try:
        for p in sorted(disabled_root.glob("*/SKILL.md")):
            disabled.append(p.parent.name)
    except Exception:
        # Disabled dir may not exist; ignore.
        pass

    try:
        for p in sorted(sys_root.glob("*/SKILL.md")):
            system.append(p.parent.name)
    except Exception:
        # System skills may not exist; ignore.
        pass

    lines: list[str] = []
    lines.append(f"skills_root: {skills_root}")
    lines.append("")
    lines.append("Enabled:")
    lines.extend(["- " + s for s in enabled] or ["- (none)"])
    lines.append("")
    lines.append("Disabled:")
    lines.extend(["- " + s for s in disabled] or ["- (none)"])
    lines.append("")
    lines.append("System:")
    lines.extend(["- .system/" + s for s in system] or ["- (none)"])
    lines.append("")
    lines.append("Usage:")
    lines.append("- /skills catalog [filter]")
    lines.append("- /skills install <skill>  (or: /skills install experimental/<skill>)")
    lines.append("- /skills disable <skill>")
    lines.append("- /skills enable <skill>")
    return "\n".join(lines)


def _move_skill_dir(*, src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(str(src))
    if dst.exists():
        raise FileExistsError(str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


_PUNCT_TRIM = "`\"'()[]{}<>.,;:"
_PNG_TOKEN_RE = re.compile(r"(?i)(?:^|\\s)([^\\s\"'<>]+\\.png)")


def _path_is_within_dir(path: Path, base_dir: Path) -> bool:
    try:
        path.resolve().relative_to(base_dir.resolve())
        return True
    except Exception:
        return False


def _collect_png_artifacts(cfg: BotConfig, *, start_time: float, text: str) -> list[Path]:
    """
    Collect candidate PNGs to send back to Telegram.

    Rules:
    - Only send files within cfg.codex_workdir (no absolute path exfil).
    - Include explicitly referenced .png tokens in the output (even if older).
    - Also include recently modified PNGs in workdir (mtime >= start_time - small slack).
    - Hard caps avoid scanning/sending too much.
    """
    workdir = cfg.codex_workdir
    workdir_resolved = workdir.resolve()

    out: list[Path] = []
    seen: set[Path] = set()

    def _maybe_add(p: Path) -> None:
        try:
            rp = p.resolve()
        except Exception:
            return
        if rp in seen:
            return
        if not _path_is_within_dir(rp, workdir_resolved):
            return
        try:
            if not rp.exists() or not rp.is_file():
                return
            # Keep this conservative; Telegram limits vary and we don't want to block on huge uploads.
            if rp.stat().st_size > 10 * 1024 * 1024:
                return
        except Exception:
            return
        seen.add(rp)
        out.append(rp)

    # 1) Explicit references in output.
    if text:
        for m in _PNG_TOKEN_RE.finditer(text):
            tok = (m.group(1) or "").strip().strip(_PUNCT_TRIM)
            if not tok:
                continue
            p = Path(tok)
            if not p.is_absolute():
                p = workdir / tok
            _maybe_add(p)

    # 2) Recent files written to workdir.
    slack = 2.0
    scanned = 0
    try:
        for p in workdir.rglob("*.png"):
            scanned += 1
            if scanned > 2000:
                break
            try:
                st = p.stat()
            except Exception:
                continue
            if st.st_mtime < (start_time - slack):
                continue
            _maybe_add(p)
    except Exception:
        # Workdir might become inaccessible; sending the main text output is still useful.
        return out

    # Prefer most-recent first for the implicit scan.
    try:
        out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        pass
    return out


def _is_authorized(cfg: BotConfig, msg: IncomingMessage) -> bool:
    # No restrictions mode: if BOT_UNSAFE_DIRECT_CODEX=1, allow anyone.
    # This is intentionally dangerous and should only be used when the host/bot is otherwise isolated.
    if cfg.unsafe_direct_codex:
        return True
    # Safety by default: if no allow-list is configured, treat all chats as unauthorized.
    if not cfg.allowed_chat_ids and not cfg.allowed_user_ids:
        return False
    if cfg.allowed_chat_ids and msg.chat_id not in cfg.allowed_chat_ids:
        return False
    if cfg.allowed_user_ids and msg.user_id not in cfg.allowed_user_ids:
        return False
    return True


_ALLOWED_PASSTHROUGH_SLASH = {"exec", "review", "codex"}
_BANNED_CODEX_FLAGS = {
    "--dangerously-bypass-approvals-and-sandbox",
    "--full-auto",
}

_ALLOWED_CODEX_COMMANDS = {"exec", "review"}
# Allow only safe config overrides. Codex supports powerful overrides like:
# -c 'sandbox_permissions=["disk-full-read-access"]'
# -c shell_environment_policy.inherit=all
_ALLOWED_CONFIG_KEYS = {"model", "model_reasoning_effort"}
_BANNED_FLAGS_ALWAYS = {
    # Expands writable scope outside CODEX_WORKDIR.
    "--add-dir",
    # Enables live web search tooling.
    "--search",
    # Feature toggles map to config overrides (-c features.*=...).
    "--enable",
    "--disable",
    # Attach local files (exfil risk).
    "--image",
    "-i",
    # Avoid selecting arbitrary profiles (may change sandbox/shell env policy).
    "--profile",
    "-p",
    # Keep approval policy stable; wrapper forces `-a never`.
    "--ask-for-approval",
    "-a",
}


def _iter_config_override_keys(argv: list[str]) -> list[str]:
    keys: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-c", "--config"):
            kv = argv[i + 1] if i + 1 < len(argv) else ""
            if isinstance(kv, str) and "=" in kv:
                k = kv.split("=", 1)[0].strip()
                if k:
                    keys.append(k)
            i += 2
            continue
        if isinstance(a, str) and a.startswith("--config="):
            kv = a.split("=", 1)[1]
            if "=" in kv:
                k = kv.split("=", 1)[0].strip()
                if k:
                    keys.append(k)
            i += 1
            continue
        # `-cfoo=bar` form (when user doesn't add a space).
        if isinstance(a, str) and a.startswith("-c") and len(a) > 2:
            kv = a[2:]
            if "=" in kv:
                k = kv.split("=", 1)[0].strip()
                if k:
                    keys.append(k)
            i += 1
            continue
        i += 1
    return keys


def _extract_codex_command(argv: list[str]) -> str:
    """
    Best-effort extract the Codex subcommand from argv, accounting for global options
    appearing before the command (e.g. `codex --oss exec ...`).

    Returns an empty string if no subcommand is present (e.g. `codex --version`).
    """
    i = 0
    # Global options that take a value. Keep this list tight; it's only used for bot-side validation.
    opts_with_value = {
        "-c",
        "--config",
        "-m",
        "--model",
        "--local-provider",
        "-s",
        "--sandbox",
        "-a",
        "--ask-for-approval",
        "-C",
        "--cd",
        "--add-dir",
        "-p",
        "--profile",
        "--enable",
        "--disable",
        "-i",
        "--image",
    }
    while i < len(argv):
        a = argv[i]
        if not isinstance(a, str):
            i += 1
            continue
        if a == "--":
            nxt = argv[i + 1] if i + 1 < len(argv) else ""
            return nxt if isinstance(nxt, str) else ""
        if a.startswith("-"):
            if a in opts_with_value:
                i += 2
                continue
            i += 1
            continue
        return a
    return ""


def _validate_codex_argv(cfg: BotConfig, argv: list[str], mode_hint: str) -> str | None:
    if cfg.unsafe_direct_codex:
        return None

    # Enforce the "writes require /rw" policy by not allowing callers to smuggle their own sandbox/workdir.
    allow_custom_sandbox = os.environ.get("BOT_ALLOW_CUSTOM_SANDBOX", "0").strip().lower() in ("1", "true", "yes", "on")
    lower = [a.lower() for a in argv]

    # Restrict the Codex subcommand surface area for bot-invoked runs.
    # `codex apply`, `login`, etc. can mutate state without going through the exec sandbox.
    cmd = _extract_codex_command(argv)
    if cmd and cmd.lower() not in _ALLOWED_CODEX_COMMANDS:
        return f"Not allowed: codex {cmd} (allowed: exec, review)."

    for a in lower:
        if a in _BANNED_CODEX_FLAGS:
            return f"Not allowed: {a}"
        if a.startswith("--dangerously-bypass-approvals-and-sandbox="):
            return f"Not allowed: {a}"

    for a in lower:
        if a in _BANNED_FLAGS_ALWAYS:
            return f"Not allowed: {a}"
        if a.startswith("--add-dir="):
            return f"Not allowed: {a}"
        if a.startswith("--profile="):
            return f"Not allowed: {a}"
        if a.startswith("--enable=") or a.startswith("--disable="):
            return f"Not allowed: {a}"
        if a.startswith("--image="):
            return f"Not allowed: {a}"

    # Disallow config overrides except for a tight allow-list of safe keys.
    keys = _iter_config_override_keys(argv)
    for k in keys:
        if k not in _ALLOWED_CONFIG_KEYS:
            return f"Not allowed: -c/--config {k}=... (allowed keys: {', '.join(sorted(_ALLOWED_CONFIG_KEYS))})."

    # Disallow changing directories via argv (keep runs scoped to CODEX_WORKDIR).
    if "--cd" in lower:
        return "Not allowed: -C/--cd in bot commands."
    for a in lower:
        if a.startswith("--cd="):
            return "Not allowed: -C/--cd in bot commands."
    for a in argv:
        if a == "-C" or (a.startswith("-C") and len(a) > 2):
            return "Not allowed: -C/--cd in bot commands."

    # Disallow choosing arbitrary sandboxes unless explicitly enabled; even then, restrict values.
    if "--sandbox" in lower:
        if not allow_custom_sandbox:
            return "Custom --sandbox is not allowed. Use /ro or /rw."
        try:
            idx = lower.index("--sandbox")
            val = lower[idx + 1] if idx + 1 < len(lower) else ""
        except Exception:
            val = ""
        if val not in ("read-only", "workspace-write", "danger-full-access"):
            return "Not allowed: --sandbox must be read-only, workspace-write, or danger-full-access."

    for a in lower:
        if a.startswith("--sandbox="):
            if not allow_custom_sandbox:
                return "Custom --sandbox is not allowed. Use /ro or /rw."
            val = a.split("=", 1)[1].strip()
            if val not in ("read-only", "workspace-write", "danger-full-access"):
                return "Not allowed: --sandbox must be read-only, workspace-write, or danger-full-access."

    # In /ro, don't allow callers to opt into workspace-write by sneaking it into args.
    if mode_hint == "ro":
        if "workspace-write" in lower or "danger-full-access" in lower:
            return "Not allowed in /ro. Use /rw for workspace-write."

    return None


def _threaded_sandbox_mode_label(cfg: BotConfig) -> str:
    effective_mode = "full" if cfg.codex_force_full_access else cfg.codex_default_mode
    if effective_mode == "ro":
        return "read-only"
    if effective_mode == "rw":
        return "workspace-write"
    return "danger-full-access"


def _orch_marker(kind: str, payload: str = "") -> str:
    """
    Internal marker for orchestration commands handled in poll_loop.
    """
    if payload:
        return f"__orch_{kind}:{payload}"
    return f"__orch_{kind}__"


def _orch_job_id(raw: str) -> str:
    return (raw or "").strip()


_ORCHESTRATOR_ROLES = (
    "jarvis",
    "frontend",
    "backend",
    "qa",
    "sre",
    "product_ops",
    "security",
    "research",
    "release_mgr",
)


def _coerce_orchestrator_role(value: str) -> str:
    role = (value or "").strip().lower()
    # Legacy aliases.
    if role in ("ceo", "orchestrator"):
        return "jarvis"
    return role if role in _ORCHESTRATOR_ROLES else "backend"


def _orchestrator_known_roles(profiles: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    keys = sorted({*_ORCHESTRATOR_ROLES, *(str(k).strip().lower() for k in profiles.keys())})
    return tuple(keys)


def _orchestrator_role_is_valid(role: str, profiles: dict[str, dict[str, Any]]) -> bool:
    normalized = (role or "").strip().lower()
    return normalized in _orchestrator_known_roles(profiles)


def _coerce_orchestrator_mode(value: str) -> str:
    mode = (value or "").strip().lower()
    return mode if mode in ("ro", "rw", "full") else "ro"


def _default_orchestrator_profile(role: str) -> dict[str, Any]:
    return {
        "name": role.title(),
        "role": role,
        "system_prompt": "",
        "model": "",
        "effort": "medium",
        "mode_hint": "ro",
        "allowed_tools": [],
        "max_parallel_jobs": 1,
        "max_runtime_seconds": 900,
        "approval_required": False,
    }


def _orchestrator_profile(
    profiles: dict[str, dict[str, Any]] | None,
    role: str,
) -> dict[str, Any]:
    normalized = _coerce_orchestrator_role(role)
    if profiles is None:
        return _default_orchestrator_profile(normalized)
    profile = profiles.get(normalized)
    if profile is None:
        return _default_orchestrator_profile(normalized)
    if not isinstance(profile, dict):
        return _default_orchestrator_profile(normalized)
    out: dict[str, Any] = _default_orchestrator_profile(normalized)
    out.update(profile)
    return out


def _orchestrator_model_for_profile(cfg: BotConfig, profile: dict[str, Any]) -> str:
    model = str(profile.get("model") or "").strip()
    model = _sanitize_model_id(model)
    if model:
        return model
    return _sanitize_model_id(cfg.codex_openai_model if not cfg.codex_use_oss else cfg.codex_oss_model)


def _orchestrator_effort_for_profile(profile: dict[str, Any], cfg: BotConfig) -> str:
    effort = str(profile.get("effort") or "").strip().lower()
    effort = _sanitize_effort(effort)
    if effort:
        return effort
    _, cfg_effort = _codex_defaults_from_config()
    return cfg_effort or "medium"


def _orchestrator_task_from_job(
    cfg: BotConfig,
    job: Job,
    *,
    profiles: dict[str, dict[str, Any]] | None,
    user_id: int | None = None,
) -> Task:
    employee_name, user_text = _parse_employee_forward(job.user_text)

    # CEO UX rule (grounded): Jarvis is the single front door.
    # Only explicit @role markers can override the default role for top-level messages.
    base_role = _coerce_orchestrator_role(cfg.orchestrator_default_role)
    pre = to_task(
        user_text,
        context={
            "chat_id": job.chat_id,
            "default_role": base_role,
            "role": base_role,
        },
    )
    role = _coerce_orchestrator_role(pre.role)
    profile = _orchestrator_profile(profiles, role)

    mode_hint = _coerce_orchestrator_mode(str(profile.get("mode_hint") or ""))
    mode_hint = _coerce_orchestrator_mode(job.mode_hint or mode_hint)
    model = _orchestrator_model_for_profile(cfg, profile)
    effort = _orchestrator_effort_for_profile(profile, cfg)

    requires_approval = bool(profile.get("approval_required", False))
    if mode_hint == "full":
        requires_approval = True

    trace: dict[str, str | int | float | bool | list[str]] = {
        "source": "telegram",
        "legacy_mode_hint": job.mode_hint,
        "profile_name": str(profile.get("name") or role),
        "profile_role": role,
        "max_runtime_seconds": int(profile.get("max_runtime_seconds") or 0),
    }
    trace["prefer_voice_reply"] = bool(getattr(job, "prefer_voice_reply", False))
    if employee_name:
        trace["employee_name"] = employee_name
    if job.reply_to_message_id is not None:
        trace["reply_to_message_id"] = int(job.reply_to_message_id)

    raw_priority = cfg.orchestrator_default_priority
    try:
        priority = int(raw_priority)
    except Exception:
        priority = 2
    if priority < 1:
        priority = 1
    # CEO preemption: for manual top-level work requests, default to priority=1 so it
    # doesn't sit behind an old backlog.
    if not (user_text or "").lstrip().startswith("/") and str(pre.request_type or "task") == "task":
        priority = 1

    context = {
        "source": "telegram",
        "chat_id": job.chat_id,
        "user_id": user_id,
        "reply_to_message_id": job.reply_to_message_id,
        "model": model,
        "effort": effort,
        "role": role,
        "default_role": base_role,
        "priority": priority,
        "due_at": None,
        "mode_hint": mode_hint,
        # Let dispatcher detect request_type (query/status/task), but keep our role fixed unless @role was explicit.
        "request_type": pre.request_type,
        "requires_approval": requires_approval,
        "max_cost_window_usd": float(cfg.orchestrator_default_max_cost_window_usd),
        "trace": trace,
    }

    # role can still be overridden by explicit @role markers in the input text (dispatcher handles that).
    task = to_task(user_text, context=context)
    if task.role not in _ORCHESTRATOR_ROLES:
        task = task.with_updates(role=role)
    task = task.with_updates(
        model=task.model or model,
        effort=task.effort or effort,
        mode_hint=_coerce_orchestrator_mode(task.mode_hint),
        requires_approval=bool(requires_approval),
        trace=trace,
        priority=int(task.priority or priority),
        state="queued",
        max_cost_window_usd=float(task.max_cost_window_usd or cfg.orchestrator_default_max_cost_window_usd),
    )
    # Snapshot hint: allow bot-side screenshot capture for frontend tasks.
    try:
        txt = (user_text or "").strip()
        prefix = "@frontend Solicitud de snapshot:"
        if txt.lower().startswith(prefix.lower()):
            url = txt[len(prefix) :].strip().split(None, 1)[0].strip().strip(_PUNCT_TRIM)
            if url:
                trace["needs_screenshot"] = True
                trace["screenshot_url"] = url
                # Snapshot is a bot-side operation; no need to run Codex afterwards.
                trace["screenshot_only"] = True
                task = task.with_updates(trace=trace)
    except Exception:
        pass
    if task.max_cost_window_usd <= 0:
        task = task.with_updates(max_cost_window_usd=float(cfg.orchestrator_default_max_cost_window_usd))
    if not (task.artifacts_dir or "").strip():
        task = task.with_updates(artifacts_dir=str((cfg.artifacts_root / task.job_id).resolve()))
    return task


def _parse_orchestrator_marker(text: str) -> tuple[str, str] | None:
    """
    Decode internal command markers produced by _orch_marker into (kind, payload).
    """
    if not text.startswith("__orch_"):
        return None
    body = text[len("__orch_") :]
    if not body:
        return None
    if ":" in body:
        kind, payload = body.split(":", 1)
        return kind, payload.strip()
    return body, ""


def _can_manage_orchestrator(cfg: BotConfig, *, chat_id: int) -> bool:
    if not cfg.auth_enabled:
        return True
    profile = _auth_effective_profile_name(cfg, chat_id=chat_id)
    if not profile:
        return True
    return _profile_can_manage_bot(cfg, profile_name=profile)

def _can_delete_jobs(cfg: BotConfig, *, chat_id: int, user_id: int | None) -> bool:
    """
    Destructive actions (delete) should be stricter than "manage orchestrator".
    Preference order:
    1) If BOT_ADMIN_* allow-lists are configured, require a match.
    2) Else, if BOT_AUTH_ENABLED=1, require profile permission.
    3) Else, deny.
    """
    if cfg.admin_chat_ids or cfg.admin_user_ids:
        if cfg.admin_chat_ids and int(chat_id) in cfg.admin_chat_ids:
            return True
        if cfg.admin_user_ids and user_id is not None and int(user_id) in cfg.admin_user_ids:
            return True
        return False
    if cfg.auth_enabled:
        return _can_manage_orchestrator(cfg, chat_id=chat_id)
    return False


def _send_chunked_text(
    api: "TelegramAPI",
    *,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None,
    limit: int = TELEGRAM_MSG_LIMIT - 64,
) -> None:
    chunks = _chunk_text(text, limit=max(512, int(limit)))
    for idx, ch in enumerate(chunks, start=1):
        payload = ch if len(chunks) == 1 else f"[{idx}/{len(chunks)}]\n{ch}"
        try:
            api.send_message(chat_id, payload, reply_to_message_id=reply_to_message_id)
        except Exception:
            LOG.exception("Failed to send chunked message. chat_id=%s", chat_id)


def _send_orchestrator_marker_response(
    kind: str,
    payload: str,
    cfg: BotConfig,
    api: "TelegramAPI",
    chat_id: int,
    user_id: int | None,
    reply_to_message_id: int | None,
    orch_q: OrchestratorQueue | None,
    profiles: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """
    Returns True if marker was handled.
    """
    if kind == "agents":
        if orch_q is None:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        _send_chunked_text(api, chat_id=chat_id, text=_orchestrator_status_text(orch_q), reply_to_message_id=reply_to_message_id)
        return True

    if kind == "dashboard":
        if orch_q is None:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not cfg.screenshot_enabled:
            api.send_message(chat_id, _orchestrator_status_text(orch_q), reply_to_message_id=reply_to_message_id)
            return True
        now = time.time()
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        health = orch_q.get_role_health() or {}
        running = orch_q.jobs_by_state(state="running", limit=16)

        def _pill(label: str, val: str) -> str:
            return (
                '<span class="pill">'
                + _html_escape(label)
                + ': <b>'
                + _html_escape(val)
                + "</b></span>"
            )

        role_rows: list[str] = []
        for role in sorted(health.keys()):
            vals = health.get(role, {}) or {}
            queued = int(vals.get("queued", 0) or 0)
            running_n = int(vals.get("running", 0) or 0)
            blocked = int(vals.get("blocked", 0) or 0)
            failed = int(vals.get("failed", 0) or 0)
            role_rows.append(
                "<tr>"
                + f"<td>{_html_escape(role)}</td>"
                + f"<td>{queued}</td>"
                + f"<td>{running_n}</td>"
                + f"<td>{blocked}</td>"
                + f"<td>{failed}</td>"
                + "</tr>"
            )
        if not role_rows:
            role_rows.append("<tr><td colspan='5' class='muted'>(no data yet)</td></tr>")

        run_rows: list[str] = []
        for t in running[:16]:
            trace = t.trace or {}
            phase = str(trace.get("live_phase") or "").strip() or "running"
            tail = str(trace.get("live_stdout_tail") or "").strip()
            if len(tail) > 600:
                tail = tail[-600:]
            title = (t.input_text or "").strip().replace("\n", " ")
            if len(title) > 120:
                title = title[:120] + "..."
            run_rows.append(
                "<div class='run'>"
                + f"<div class='run-h'><b>{_html_escape(t.job_id[:8])}</b> <span class='muted'>role={_html_escape(t.role)} phase={_html_escape(phase)}</span></div>"
                + f"<div class='run-t'>{_html_escape(title)}</div>"
                + (f"<pre class='tail'>{_html_escape(tail)}</pre>" if tail else "")
                + "</div>"
            )
        if not run_rows:
            run_rows.append("<div class='muted'>(no running jobs)</div>")

        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<style>"
            "body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Arial; margin:0; background:#0b0f14; color:#e8eef6}"
            ".wrap{padding:18px}"
            ".h{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:14px}"
            ".title{font-size:18px;font-weight:700}"
            ".sub{font-size:12px;color:#9fb0c3}"
            ".grid{display:grid;grid-template-columns:1fr 1.2fr;gap:14px}"
            ".card{background:#121a24;border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:14px;box-shadow:0 10px 30px rgba(0,0,0,.35)}"
            ".pill{display:inline-flex;gap:6px;align-items:baseline;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);margin-right:8px;font-size:12px}"
            "table{width:100%;border-collapse:collapse;font-size:12px}"
            "th,td{padding:8px 6px;border-bottom:1px solid rgba(255,255,255,.06);text-align:left}"
            "th{color:#9fb0c3;font-weight:600}"
            ".muted{color:#9fb0c3}"
            ".run{padding:10px 0;border-bottom:1px solid rgba(255,255,255,.06)}"
            ".run:last-child{border-bottom:none}"
            ".run-h{font-size:12px}"
            ".run-t{font-size:12px;margin-top:4px;color:#cfe0f3}"
            "pre.tail{margin:8px 0 0; padding:10px; border-radius:10px; background:#0b0f14; border:1px solid rgba(255,255,255,.06); font-size:11px; white-space:pre-wrap; max-height:160px; overflow:hidden}"
            "</style></head><body><div class='wrap'>"
            "<div class='h'>"
            "<div><div class='title'>PonceBot Dashboard</div><div class='sub'>"
            + _html_escape(ts)
            + "</div></div>"
            "<div class='sub'>"
            + _pill("queued", str(orch_q.get_queued_count()))
            + _pill("running", str(orch_q.get_running_count()))
            + "</div></div>"
            "<div class='grid'>"
            "<div class='card'><div class='title' style='font-size:14px'>Roles</div>"
            "<table><thead><tr><th>role</th><th>queued</th><th>running</th><th>blocked</th><th>failed</th></tr></thead><tbody>"
            + "".join(role_rows)
            + "</tbody></table></div>"
            "<div class='card'><div class='title' style='font-size:14px'>Running Now</div>"
            + "".join(run_rows)
            + "</div>"
            "</div></div></body></html>"
        )

        out_dir = (cfg.artifacts_root / "dashboard").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"dashboard_{int(now)}.png"
        try:
            capture_screenshot_html(
                html,
                out_png,
                viewport=Viewport(width=1280, height=720),
                allowed_hosts=cfg.screenshot_allowed_hosts,
                allow_private=False,
                block_network=True,
            )
            api.send_photo(chat_id, out_png, caption="dashboard", reply_to_message_id=reply_to_message_id)
        except Exception as e:
            api.send_message(chat_id, f"Dashboard failed: {e}", reply_to_message_id=reply_to_message_id)
        return True

    if kind == "watch":
        if orch_q is None:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        mode = (payload or "").strip().lower() or "on"
        st = _get_state(cfg)
        watch_by_chat = st.get("watch_by_chat")
        if not isinstance(watch_by_chat, dict):
            watch_by_chat = {}
        key = str(int(chat_id))

        if mode in ("off", "0", "false", "stop", "disable", "disabled"):
            watch_by_chat.pop(key, None)
            st["watch_by_chat"] = watch_by_chat
            _atomic_write_json(cfg.state_file, st)
            api.send_message(chat_id, "OK. Watch disabled.", reply_to_message_id=reply_to_message_id)
            return True

        # Enable: create/update the stored message id and start updating every scheduler tick.
        msg_txt = _watch_status_text(orch_q)
        mid = api.send_message(chat_id, msg_txt, reply_to_message_id=reply_to_message_id)
        if mid is not None:
            watch_by_chat[key] = int(mid)
            st["watch_by_chat"] = watch_by_chat
            _atomic_write_json(cfg.state_file, st)
        return True

    if kind == "orders":
        if orch_q is None:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        _send_chunked_text(api, chat_id=chat_id, text=_orders_text(orch_q, chat_id=chat_id), reply_to_message_id=reply_to_message_id)
        return True

    if kind == "order":
        if orch_q is None:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        _send_chunked_text(
            api,
            chat_id=chat_id,
            text=_order_command_text(orch_q, chat_id=chat_id, payload=payload),
            reply_to_message_id=reply_to_message_id,
        )
        return True

    if kind == "job":
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        raw = (payload or "").strip()
        if not raw or raw.lower() == "list":
            items = orch_q.peek(limit=25)
            if not items:
                api.send_message(chat_id, "Jobs: empty.", reply_to_message_id=reply_to_message_id)
                return True

            def _title(t: Task) -> str:
                s = (t.input_text or "").strip().replace("\n", " ")
                if len(s) > 80:
                    s = s[:80] + "..."
                return s

            def _owner(t: Task) -> str:
                if t.owner and str(t.owner).strip():
                    return str(t.owner).strip()
                if bool(t.is_autonomous):
                    return "scheduler"
                if t.user_id is not None:
                    return f"user:{int(t.user_id)}"
                return f"chat:{int(t.chat_id)}"

            lines = ["Jobs (latest 25)", "================", ""]
            for t in items:
                created = time.strftime("%Y-%m-%d %H:%M", time.localtime(t.created_at))
                lines.append(f"- {t.job_id[:8]}  {t.state}  role={t.role} owner={_owner(t)}  {created}  {_title(t)}")
            lines.append("")
            lines.append("Use: /job show <id>  |  /job del <id>")
            _send_chunked_text(api, chat_id=chat_id, text="\n".join(lines), reply_to_message_id=reply_to_message_id)
            return True

        try:
            parts = shlex.split(raw)
        except Exception:
            parts = raw.split()
        if not parts:
            api.send_message(chat_id, "Uso: /job [list|show <id>|del <id>]", reply_to_message_id=reply_to_message_id)
            return True

        cmd = parts[0].strip().lower()
        if cmd in ("show", "get"):
            if len(parts) < 2 or not parts[1].strip():
                api.send_message(chat_id, "Uso: /job show <id>", reply_to_message_id=reply_to_message_id)
                return True
            jid = parts[1].strip()
            task = orch_q.get_job(jid)
            _send_chunked_text(api, chat_id=chat_id, text=_orchestrator_job_text(task), reply_to_message_id=reply_to_message_id)
            return True

        if cmd in ("del", "delete", "rm"):
            if len(parts) < 2 or not parts[1].strip():
                api.send_message(chat_id, "Uso: /job del <id>", reply_to_message_id=reply_to_message_id)
                return True
            jid = parts[1].strip()
            confirm = len(parts) >= 3 and parts[2].strip().lower() == "confirm"
            if not confirm:
                api.send_message(chat_id, f"Confirm delete:\n/job del {jid} confirm", reply_to_message_id=reply_to_message_id)
                return True
            if not _can_delete_jobs(cfg, chat_id=chat_id, user_id=user_id):
                api.send_message(
                    chat_id,
                    "No permitido: delete requiere admin allowlist (BOT_ADMIN_USER_IDS/BOT_ADMIN_CHAT_IDS) o permisos via auth profile.",
                    reply_to_message_id=reply_to_message_id,
                )
                return True
            task = orch_q.get_job(jid)
            if task is None:
                api.send_message(chat_id, f"No existe job: {jid}", reply_to_message_id=reply_to_message_id)
                return True
            if str(task.state) == "running":
                api.send_message(chat_id, f"No borro jobs en running. Usa: /cancel {task.job_id[:8]}", reply_to_message_id=reply_to_message_id)
                return True
            ok = orch_q.delete_job(jid)
            msg_txt = f"Deleted: {task.job_id[:8]}" if ok else f"No pude borrar: {jid}"
            api.send_message(chat_id, msg_txt, reply_to_message_id=reply_to_message_id)
            return True

        # Back-compat: /job <id>
        jid = parts[0].strip()
        task = orch_q.get_job(jid)
        _send_chunked_text(api, chat_id=chat_id, text=_orchestrator_job_text(task), reply_to_message_id=reply_to_message_id)
        return True

    if kind == "ticket":
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not payload:
            api.send_message(chat_id, "Uso: /ticket <id>", reply_to_message_id=reply_to_message_id)
            return True
        _send_chunked_text(
            api,
            chat_id=chat_id,
            text=_orchestrator_ticket_text(orch_q, payload),
            reply_to_message_id=reply_to_message_id,
        )
        return True

    if kind == "inbox":
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        role = (payload or "").strip().lower() or None
        if role is not None and role != "all" and not _orchestrator_role_is_valid(role, profiles or {}):
            roles = ", ".join(_orchestrator_known_roles(profiles or {}))
            api.send_message(chat_id, f"Rol invalido: {role}. Roles: {roles}", reply_to_message_id=reply_to_message_id)
            return True
        _send_chunked_text(
            api,
            chat_id=chat_id,
            text=_orchestrator_inbox_text(orch_q, role=None if role in (None, "all") else role),
            reply_to_message_id=reply_to_message_id,
        )
        return True

    if kind == "runbooks":
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not _can_manage_orchestrator(cfg, chat_id=chat_id):
            api.send_message(chat_id, "No permitido: necesitas permisos de gestor.", reply_to_message_id=reply_to_message_id)
            return True
        _send_chunked_text(api, chat_id=chat_id, text=_orchestrator_runbooks_text(cfg, orch_q), reply_to_message_id=reply_to_message_id)
        return True

    if kind == "reset_role":
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not _can_manage_orchestrator(cfg, chat_id=chat_id):
            api.send_message(chat_id, "No permitido: necesitas permisos de gestor.", reply_to_message_id=reply_to_message_id)
            return True
        role = (payload or "").strip().lower()
        if not role:
            api.send_message(chat_id, "Uso: /reset_role <role|all>", reply_to_message_id=reply_to_message_id)
            return True
        if role == "all":
            n = orch_q.clear_agent_threads(chat_id=chat_id)
            api.send_message(chat_id, f"OK. Reset sessions: {n}", reply_to_message_id=reply_to_message_id)
            return True
        if not _orchestrator_role_is_valid(role, profiles or {}):
            roles = ", ".join(_orchestrator_known_roles(profiles or {}))
            api.send_message(chat_id, f"Rol invalido: {role}. Roles: {roles}", reply_to_message_id=reply_to_message_id)
            return True
        ok = orch_q.clear_agent_thread(chat_id=chat_id, role=role)
        api.send_message(chat_id, f"OK. Reset {role}: {'cleared' if ok else 'not set'}", reply_to_message_id=reply_to_message_id)
        return True

    if kind in ("daily", "brief"):
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        _send_chunked_text(
            api,
            chat_id=chat_id,
            text=_orchestrator_daily_digest_text(orch_q),
            reply_to_message_id=reply_to_message_id,
        )
        return True

    if kind in ("pause", "resume"):
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not _can_manage_orchestrator(cfg, chat_id=chat_id):
            api.send_message(
                chat_id,
                "No permitido: necesitas permisos de gestor para acciones de orquestador.",
                reply_to_message_id=reply_to_message_id,
            )
            return True
        if not payload:
            api.send_message(chat_id, f"Uso: /{kind} <role>", reply_to_message_id=reply_to_message_id)
            return True
        role = (payload or "").strip().lower()
        if not _orchestrator_role_is_valid(role, profiles or {}):
            roles = ", ".join(_orchestrator_known_roles(profiles or {}))
            api.send_message(chat_id, f"Rol invalido: {role}. Roles: {roles}", reply_to_message_id=reply_to_message_id)
            return True
        if kind == "pause":
            orch_q.pause_role(role)
            api.send_message(chat_id, f"Pausado: {role}", reply_to_message_id=reply_to_message_id)
        else:
            orch_q.resume_role(role)
            api.send_message(chat_id, f"Reanudado: {role}", reply_to_message_id=reply_to_message_id)
        return True

    if kind in ("emergency_stop", "emergency_resume"):
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not _can_manage_orchestrator(cfg, chat_id=chat_id):
            api.send_message(
                chat_id,
                "No permitido: necesitas permisos de gestor para controlar el orquestador.",
                reply_to_message_id=reply_to_message_id,
            )
            return True
        if kind == "emergency_stop":
            orch_q.pause_all_roles()
            canceled = orch_q.cancel_running_jobs()
            api.send_message(
                chat_id,
                f"Emergency stop activo. Roles pausados y tareas en ejecución canceladas: {canceled}.",
                reply_to_message_id=reply_to_message_id,
            )
        else:
            orch_q.resume_all_roles()
            api.send_message(chat_id, "Emergency stop liberado. Roles reanudados.", reply_to_message_id=reply_to_message_id)
        return True

    if kind == "approve":
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not _can_manage_orchestrator(cfg, chat_id=chat_id):
            api.send_message(
                chat_id,
                "No permitido: necesitas permisos de gestor para aprobar tareas.",
                reply_to_message_id=reply_to_message_id,
            )
            return True
        if not payload:
            api.send_message(chat_id, "Uso: /approve <id>", reply_to_message_id=reply_to_message_id)
            return True
        ok = orch_q.set_job_approved(payload)
        if ok:
            api.send_message(chat_id, f"Aprobado: {payload}", reply_to_message_id=reply_to_message_id)
            return True
        api.send_message(chat_id, f"No existe tarea: {payload}", reply_to_message_id=reply_to_message_id)
        return True

    if kind == "cancel_job":
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not _can_manage_orchestrator(cfg, chat_id=chat_id):
            api.send_message(
                chat_id,
                "No permitido: necesitas permisos de gestor para cancelar tareas.",
                reply_to_message_id=reply_to_message_id,
            )
            return True
        if not payload:
            api.send_message(chat_id, "Uso: /cancel <id>", reply_to_message_id=reply_to_message_id)
            return True
        ok = orch_q.cancel(payload)
        if ok:
            api.send_message(chat_id, f"Cancelado: {payload}", reply_to_message_id=reply_to_message_id)
            return True
        api.send_message(chat_id, f"No existe o ya finalizado: {payload}", reply_to_message_id=reply_to_message_id)
        return True

    if kind == "purge_queue":
        if not orch_q:
            api.send_message(chat_id, "Jarvis disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not _can_manage_orchestrator(cfg, chat_id=chat_id):
            api.send_message(
                chat_id,
                "No permitido: necesitas permisos de gestor para purgar la cola.",
                reply_to_message_id=reply_to_message_id,
            )
            return True

        scope = (payload or "").strip().lower()
        target_chat_id: int | None = None
        if scope in ("chat", "here", "aqui", "aquí", "this"):
            target_chat_id = int(chat_id)
        cancelled = orch_q.cancel_by_states(
            states=("queued", "blocked"),
            reason="purge_queue",
            chat_id=target_chat_id,
        )
        try:
            health = orch_q.get_role_health()
            queued = int(orch_q.get_queued_count())
            running_n = int(orch_q.get_running_count())
            blocked = 0
            for rec in (health or {}).values():
                try:
                    blocked += int((rec or {}).get("blocked", 0))
                except Exception:
                    pass
            running = orch_q.jobs_by_state(state="running", limit=5)
        except Exception:
            queued = 0
            running_n = 0
            blocked = 0
            running = []

        lines: list[str] = []
        lines.append("Jarvis: queue purged")
        if target_chat_id is None:
            lines.append(f"- scope: global")
        else:
            lines.append(f"- scope: chat_id={target_chat_id}")
        lines.append(f"- cancelled: {int(cancelled)} (queued+blocked)")
        lines.append(f"- now: queued={queued} running={running_n} blocked={blocked}")
        if running:
            parts: list[str] = []
            for r in running[:5]:
                role_h = _humanize_orchestrator_role(r.role)
                snippet = (r.input_text or "").strip().replace("\n", " ")
                if len(snippet) > 70:
                    snippet = snippet[:70] + "..."
                parts.append(f"{role_h}: {snippet}")
            lines.append("- running: " + " | ".join(parts))

        api.send_message(chat_id, "\n".join(lines), reply_to_message_id=reply_to_message_id)
        return True

    return False


def _orchestrator_status_text(orch_q: OrchestratorQueue) -> str:
    health = orch_q.get_role_health()
    if not health:
        return "No orchestrator jobs yet."

    states = ("queued", "running", "blocked", "done", "failed", "cancelled")
    system_state = "paused" if orch_q.is_paused_globally() else "active"
    lines = ["Jarvis role health:", f"system: {system_state}", ""]
    for role in sorted(health.keys()):
        vals = health.get(role, {})
        state_parts = [f"{s}={int(vals.get(s, 0))}" for s in states if vals.get(s) is not None]
        if not state_parts:
            state_parts = [f"{s}=0" for s in states]
        paused = int(vals.get("paused", 0))
        lines.append(f"- {role} ({'paused' if paused else 'active'}): " + ", ".join(state_parts))

    running = orch_q.jobs_by_state(state="running", limit=12)
    if running:
        lines.append("")
        lines.append("Running (live):")
        for t in running[:12]:
            trace = t.trace or {}
            phase = str(trace.get("live_phase") or "").strip() or "running"
            slot = trace.get("live_workspace_slot")
            try:
                slot_s = str(int(slot)) if slot is not None else "n/a"
            except Exception:
                slot_s = "n/a"
            tail = str(trace.get("live_stdout_tail") or "").strip().replace("\n", " ")
            if len(tail) > 160:
                tail = tail[-160:]
            snippet = (t.input_text or "").strip().replace("\n", " ")[:120]
            extra = f" tail={tail}" if tail else ""
            lines.append(f"- {t.job_id[:8]} role={t.role} phase={phase} slot={slot_s} text={snippet}{extra}")

    return "\n".join(lines)


def _orchestrator_daily_digest_text(orch_q: OrchestratorQueue) -> str:
    lines = ["Jarvis digest", "=" * 12]
    lines.append(_orchestrator_status_text(orch_q))

    running = orch_q.jobs_by_state(state="running", limit=8)
    if running:
        lines.extend(["", "Running jobs:"])
        for t in running[:8]:
            summary = (t.input_text or "").strip().replace("\n", " ")[:160]
            lines.append(f"- {t.job_id[:8]} role={t.role} state={t.state} text={summary}")
        if len(running) > 8:
            lines.append(f"- ... +{len(running) - 8} more")

    return "\n".join(lines)


def _watch_status_text(orch_q: OrchestratorQueue) -> str:
    """
    Single-message "company status" snapshot intended to be edited in-place (no spam).
    """
    now = time.time()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    health = orch_q.get_role_health() or {}
    system_state = "paused" if orch_q.is_paused_globally() else "active"

    lines: list[str] = [
        "Company Status (Jarvis)",
        f"updated_at: {ts}",
        f"system: {system_state}",
        "",
    ]

    if health:
        lines.append("roles:")
        for role in sorted(health.keys()):
            vals = health.get(role, {}) or {}
            queued = int(vals.get("queued", 0) or 0)
            running = int(vals.get("running", 0) or 0)
            blocked = int(vals.get("blocked", 0) or 0)
            failed = int(vals.get("failed", 0) or 0)
            lines.append(f"- {role}: queued={queued} running={running} blocked={blocked} failed={failed}")
    else:
        lines.append("roles: (no data yet)")

    running_jobs = orch_q.jobs_by_state(state="running", limit=8)
    lines.append("")
    lines.append("running:")
    if not running_jobs:
        lines.append("- (none)")
    else:
        for t in running_jobs[:8]:
            tr = t.trace or {}
            phase = str(tr.get("live_phase") or "").strip() or "running"
            tail = str(tr.get("live_stdout_tail") or "").strip().replace("\n", " ")
            if len(tail) > 160:
                tail = tail[-160:]
            snippet = (t.input_text or "").strip().replace("\n", " ")
            if len(snippet) > 90:
                snippet = snippet[:90] + "..."
            extra = f" tail={tail}" if tail else ""
            lines.append(f"- {t.job_id[:8]} role={t.role} phase={phase} text={snippet}{extra}")

    lines.append("")
    lines.append("commands: /agents  /dashboard  /orders  /ticket <id>  /job <id>  /unwatch")
    return "\n".join(lines)


def _tick_watch_messages(*, cfg: BotConfig, api: TelegramAPI, orch_q: OrchestratorQueue) -> None:
    """
    Periodically edit the stored /watch message(s) (one per chat).
    Best-effort: if editing fails (message deleted, permissions), disable watch for that chat.
    """
    st = _get_state(cfg)
    watch_by_chat = st.get("watch_by_chat")
    if not isinstance(watch_by_chat, dict) or not watch_by_chat:
        return

    changed = False
    for chat_id_str, msg_id in list(watch_by_chat.items()):
        try:
            chat_id = int(chat_id_str)
            mid = int(msg_id)
        except Exception:
            watch_by_chat.pop(chat_id_str, None)
            changed = True
            continue

        try:
            api.edit_message_text(chat_id, mid, _watch_status_text(orch_q))
        except Exception:
            # Disable watch to avoid retry loops/spam.
            watch_by_chat.pop(chat_id_str, None)
            changed = True

    if changed:
        st["watch_by_chat"] = watch_by_chat
        _atomic_write_json(cfg.state_file, st)


def _orders_text(orch_q: OrchestratorQueue, *, chat_id: int) -> str:
    items = orch_q.list_orders(chat_id=int(chat_id), status=None, limit=50)
    if not items:
        return "Orders: (none)\n\nTip: any top-level Jarvis ticket becomes an active order (except pure queries)."
    lines: list[str] = ["Orders", "=" * 6]
    for it in items[:50]:
        oid = str(it.get("order_id") or "").strip()
        status = str(it.get("status") or "").strip() or "active"
        pr = it.get("priority")
        try:
            pr_s = str(int(pr)) if pr is not None else "n/a"
        except Exception:
            pr_s = "n/a"
        title = str(it.get("title") or "").strip().replace("\n", " ")
        if len(title) > 80:
            title = title[:80] + "..."
        lines.append(f"- {oid[:8]} status={status} priority={pr_s} title={title}")
    lines.append("")
    lines.append("Usage:")
    lines.append("- /order show <id>")
    lines.append("- /order pause <id>")
    lines.append("- /order done <id>")
    return "\n".join(lines)


def _order_command_text(orch_q: OrchestratorQueue, *, chat_id: int, payload: str) -> str:
    parts = (payload or "").strip().split()
    if len(parts) < 2:
        return "Usage: /order show|pause|done <id>"
    cmd = parts[0].strip().lower()
    oid = parts[1].strip()
    if cmd == "show":
        it = orch_q.get_order(oid, chat_id=int(chat_id))
        if not it:
            return f"No such order: {oid}"
        title = str(it.get("title") or "").strip()
        body = str(it.get("body") or "").strip()
        status = str(it.get("status") or "").strip()
        pr = it.get("priority")
        try:
            pr_s = str(int(pr)) if pr is not None else "n/a"
        except Exception:
            pr_s = "n/a"
        return "\n".join(
            [
                "Order",
                "=" * 5,
                f"id: {str(it.get('order_id') or '')}",
                f"status: {status}",
                f"priority: {pr_s}",
                "",
                "title:",
                title,
                "",
                "body:",
                body[:2400],
            ]
        )
    if cmd in ("pause", "paused"):
        ok = orch_q.set_order_status(oid, chat_id=int(chat_id), status="paused")
        return "OK." if ok else f"No such order: {oid}"
    if cmd in ("done", "complete", "completed"):
        ok = orch_q.set_order_status(oid, chat_id=int(chat_id), status="done")
        return "OK." if ok else f"No such order: {oid}"
    return "Usage: /order show|pause|done <id>"


def _autopilot_tick(
    *,
    cfg: BotConfig,
    orch_q: OrchestratorQueue,
    profiles: dict[str, dict[str, Any]] | None,
    chat_id: int,
    now: float,
) -> int:
    """
    24/7 Autopilot: keep agents working only on existing CEO orders (no random new projects).

    Grounded design:
    - Orders are persisted in SQLite (`ceo_orders`).
    - Autopilot enqueues an autonomous Jarvis job per active order only when the order is idle.
    - The Jarvis job is labeled `kind=autopilot` and is allowed to delegate.
    """
    created = 0
    orders = orch_q.list_orders(chat_id=int(chat_id), status="active", limit=30)
    if not orders:
        return 0

    # Avoid growing backlog: if there is already a meaningful queue, do not enqueue new autopilot jobs.
    # Autopilot should only "fill idle time", not compete with CEO-driven work.
    try:
        if int(orch_q.get_queued_count()) >= 10:
            return 0
    except Exception:
        pass

    for o in orders:
        oid = str(o.get("order_id") or "").strip()
        if not oid:
            continue

        # Don't enqueue another autopilot job if one is already queued/running.
        children = orch_q.jobs_by_parent(parent_job_id=oid, limit=200)
        if any(
            str((c.labels or {}).get("kind") or "").strip().lower() == "autopilot"
            and c.state in ("queued", "running")
            for c in children
        ):
            continue

        # If there is any real work already queued/running for this order, do nothing.
        has_active_work = any(
            c.state in ("queued", "running")
            and str((c.labels or {}).get("kind") or "").strip().lower() not in ("autopilot", "wrapup", "evidence")
            for c in children
        )
        if has_active_work:
            continue

        title = str(o.get("title") or "").strip()
        body = str(o.get("body") or "").strip()
        pr = o.get("priority")
        try:
            pr_i = int(pr) if pr is not None else 2
        except Exception:
            pr_i = 2
        pr_i = max(1, min(3, pr_i))
        # Autopilot should always be lowest priority.
        pr_i = 3

        ap_id = str(uuid.uuid4())
        trace: dict[str, Any] = {
            "source": "scheduler",
            "autopilot": True,
            "allow_delegation": True,
            "order_id": oid,
        }
        t = Task.new(
            source="telegram",
            role="jarvis",
            input_text=(
                "AUTOPILOT TICK\n"
                f"CEO: {cfg.ceo_name}\n"
                f"Order ID: {oid}\n"
                f"Order title: {title}\n\n"
                f"Order body:\n{body}\n\n"
                "Rules:\n"
                "- Only propose work that advances this order.\n"
                "- If the order looks complete, propose marking it DONE (do not create new projects).\n"
                "- Keep outputs short.\n"
            ),
            request_type="maintenance",
            priority=int(pr_i),
            model="",
            effort="medium",
            mode_hint="ro",
            requires_approval=False,
            max_cost_window_usd=float(cfg.orchestrator_default_max_cost_window_usd),
            chat_id=int(chat_id),
            is_autonomous=True,
            parent_job_id=oid,
            owner="scheduler",
            labels={"ticket": oid, "kind": "autopilot", "order": oid, "runbook": "jarvis_autopilot"},
            artifacts_dir=str((cfg.artifacts_root / ap_id).resolve()),
            trace=trace,
            job_id=ap_id,
        )

        # Apply profile defaults for Jarvis so the autopilot behaves like the same agent.
        try:
            rb_role = _coerce_orchestrator_role(t.role)
            rb_profile = _orchestrator_profile(profiles, rb_role)
            rb_model = _orchestrator_model_for_profile(cfg, rb_profile)
            rb_effort = _orchestrator_effort_for_profile(rb_profile, cfg)
            rb_requires_approval = bool(rb_profile.get("approval_required", False)) or (t.mode_hint == "full")
            rb_trace = dict(t.trace)
            rb_trace["profile_name"] = str(rb_profile.get("name") or rb_role)
            rb_trace["profile_role"] = rb_role
            rb_trace["max_runtime_seconds"] = int(rb_profile.get("max_runtime_seconds") or 0)
            t = t.with_updates(
                role=rb_role,
                model=rb_model,
                effort=rb_effort,
                requires_approval=rb_requires_approval,
                trace=rb_trace,
            )
        except Exception:
            pass

        orch_q.submit_task(t)
        created += 1
        # Cap per tick to avoid spam / runaway job creation.
        if created >= 1:
            break

    return created


def _orchestrator_job_text(task: Task | None) -> str:
    if task is None:
        return "No such job found."

    def _as_int(v: Any) -> str:
        try:
            return str(int(v))
        except Exception:
            return "n/a"

    created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task.created_at))
    updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task.updated_at))
    due = "n/a" if task.due_at in (None, 0) else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task.due_at))

    trace = dict(task.trace or {})
    result_status = str(trace.get("result_status") or "").strip()
    result_summary = str(trace.get("result_summary") or "").strip()
    if len(result_summary) > 1200:
        result_summary = result_summary[:1200] + "..."
    result_next_action = trace.get("result_next_action")
    result_duration_s = trace.get("result_duration_s")
    result_thread_id = str(trace.get("result_thread_id") or "").strip()
    result_workspace_slot = trace.get("result_workspace_slot")
    result_artifacts = trace.get("result_artifacts")
    if not isinstance(result_artifacts, list):
        result_artifacts = []
    artifacts_preview = [str(x) for x in result_artifacts if str(x).strip()][:3]

    live_phase = str(trace.get("live_phase") or "").strip()
    live_at = trace.get("live_at")
    live_pid = trace.get("live_pid")
    live_workdir = str(trace.get("live_workdir") or "").strip()
    live_workspace_slot = trace.get("live_workspace_slot")
    live_stdout_tail = str(trace.get("live_stdout_tail") or "").strip()
    live_stderr_tail = str(trace.get("live_stderr_tail") or "").strip()

    # Avoid dumping large trace payloads (result_summary can be big); keep it compact.
    for k in (
        "result_summary",
        "result_artifacts",
        "result_next_action",
        "result_duration_s",
        "result_thread_id",
        "result_workspace_slot",
    ):
        trace.pop(k, None)
    for k in (
        "live_phase",
        "live_at",
        "live_pid",
        "live_workdir",
        "live_workspace_slot",
        "live_stdout_tail",
        "live_stderr_tail",
    ):
        trace.pop(k, None)

    live_lines: list[str] = []
    if task.state == "running" and (live_phase or live_at or live_pid or live_stdout_tail or live_stderr_tail):
        live_lines.append("")
        live_lines.append("live:")
        live_lines.append(f"- phase: {live_phase or 'running'}")
        if live_at:
            try:
                live_lines.append(
                    f"- updated_at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(live_at)))}"
                )
            except Exception:
                pass
        if live_pid:
            live_lines.append(f"- pid: {live_pid}")
        if live_workdir:
            live_lines.append(f"- workdir: {live_workdir}")
        if live_workspace_slot is not None:
            try:
                live_lines.append(f"- workspace_slot: {int(live_workspace_slot)}")
            except Exception:
                pass
        if live_stdout_tail:
            if len(live_stdout_tail) > 1400:
                live_stdout_tail = live_stdout_tail[-1400:]
            live_lines.append("")
            live_lines.append("stdout_tail:")
            live_lines.append(live_stdout_tail)
        if live_stderr_tail:
            if len(live_stderr_tail) > 1400:
                live_stderr_tail = live_stderr_tail[-1400:]
            live_lines.append("")
            live_lines.append("stderr_tail:")
            live_lines.append(live_stderr_tail)

    return "\n".join(
        [
            f"Job: {task.job_id}",
            f"state: {task.state}",
            f"role: {task.role}",
            f"request_type: {task.request_type}",
            f"priority: {_as_int(task.priority)}",
            f"mode_hint: {task.mode_hint}",
            f"model: {task.model}",
            f"effort: {task.effort}",
            f"requires_approval: {task.requires_approval}",
            f"max_cost_window_usd: {task.max_cost_window_usd}",
            f"created_at: {created}",
            f"updated_at: {updated}",
            f"due_at: {due}",
            "",
            "result:",
            f"- status: {result_status or '(none)'}",
            f"- duration_s: {result_duration_s if result_duration_s is not None else 'n/a'}",
            f"- next_action: {result_next_action if result_next_action else 'n/a'}",
            f"- thread_id: {result_thread_id if result_thread_id else 'n/a'}",
            f"- workspace_slot: {result_workspace_slot if result_workspace_slot is not None else 'n/a'}",
            f"- artifacts: {', '.join(artifacts_preview) if artifacts_preview else '(none)'}",
            "input_text:",
            (task.input_text or "")[:900],
            *live_lines,
            "",
            f"trace: {trace}",
        ]
    )


def _orchestrator_ticket_text(orch_q: OrchestratorQueue, job_id: str) -> str:
    t = orch_q.get_job(job_id)
    if t is None:
        return "No such job found."

    root = t.parent_job_id or t.job_id
    parent = orch_q.get_job(root)
    children = orch_q.jobs_by_parent(parent_job_id=root, limit=200)

    lines: list[str] = ["Ticket", "=" * 6]
    if parent is not None:
        lines.append(f"id: {parent.job_id}")
        lines.append(f"state: {parent.state} role={parent.role} mode={parent.mode_hint}")
        summary = (parent.input_text or "").strip().replace("\n", " ")[:200]
        lines.append(f"text: {summary}")
        parent_res = str((parent.trace or {}).get("result_summary") or "").strip().replace("\n", " ")
        if parent_res:
            if len(parent_res) > 260:
                parent_res = parent_res[:260] + "..."
            lines.append(f"result: {parent_res}")
    else:
        lines.append(f"id: {root} (missing parent row)")

    if not children:
        lines.append("")
        lines.append("(no subtasks yet)")
        return "\n".join(lines)

    counts: dict[str, int] = {}
    for c in children:
        counts[c.state] = counts.get(c.state, 0) + 1
    counts_part = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    lines.append("")
    lines.append(f"subtasks: {len(children)} ({counts_part})")
    lines.append("")
    for c in children[:60]:
        snippet = (c.input_text or "").strip().replace("\n", " ")[:120]
        kind = str((c.labels or {}).get("kind") or "").strip()
        tag = f"[{kind}] " if kind else ""
        res = str((c.trace or {}).get("result_summary") or "").strip().replace("\n", " ")
        if len(res) > 140:
            res = res[:140] + "..."
        res_part = f" result={res}" if res else ""
        lines.append(f"- {c.job_id[:8]} {tag}role={c.role} state={c.state} text={snippet}{res_part}")
    if len(children) > 60:
        lines.append(f"- ... +{len(children) - 60} more")
    return "\n".join(lines)


def _ticket_card_text(orch_q: OrchestratorQueue, *, ticket_id: str) -> str:
    """
    Short, editable "ticket card" for CEO chat.
    """
    t = orch_q.get_job(ticket_id)
    if t is None:
        return f"Ticket {ticket_id[:8]}: (missing)"

    root_id = (t.parent_job_id or t.job_id or "").strip() or t.job_id
    parent = orch_q.get_job(root_id)
    children = orch_q.jobs_by_parent(parent_job_id=root_id, limit=200)

    head = parent or t
    created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(head.created_at))
    goal = (head.input_text or "").strip().replace("\n", " ")
    if len(goal) > 160:
        goal = goal[:160] + "..."

    order = None
    try:
        order = orch_q.get_order(root_id, chat_id=int(head.chat_id))
    except Exception:
        order = None

    counts: dict[str, int] = {}
    for c in children:
        counts[c.state] = counts.get(c.state, 0) + 1
    progress = " ".join(f"{k}={v}" for k, v in sorted(counts.items())) if counts else "(no subtasks)"

    running = [c for c in children if c.state == "running"][:5]

    lines: list[str] = [
        f"Jarvis: ticket {root_id[:8]} ({head.state})  created={created}",
    ]
    if order:
        lines.append(f"Order status: {str(order.get('status') or 'active')}")
    lines.append(f"Goal: {goal or '(empty)'}")
    lines.append(f"Progress: {progress}")

    # Show the latest outcome for the ticket itself so the CEO doesn't have to open /job.
    try:
        head_trace = head.trace or {}
        head_res = str(head_trace.get("result_summary") or "").strip()
        if head_res:
            first = (head_res.splitlines()[0] if head_res else "").strip()
            if len(first) > 220:
                first = first[:220] + "..."
            label = "Result" if head.state in ("done", "failed", "cancelled", "blocked") else "Latest"
            lines.append(f"{label}: {first}")
        head_next = str(head_trace.get("result_next_action") or "").strip()
        if head_next:
            if len(head_next) > 140:
                head_next = head_next[:140] + "..."
            lines.append(f"Next: {head_next}")
    except Exception:
        pass

    if running:
        lines.append("Running:")
        for r in running:
            tr = r.trace or {}
            phase = str(tr.get("live_phase") or "").strip() or "running"
            snippet = (r.input_text or "").strip().replace("\n", " ")
            if len(snippet) > 90:
                snippet = snippet[:90] + "..."
            role_h = _humanize_orchestrator_role(r.role)
            lines.append(f"- {role_h} ({phase}): {snippet}")
    lines.append(f"Links: /ticket {root_id[:8]}  /job {root_id[:8]}  /agents  /dashboard")
    return "\n".join(lines)


def _maybe_update_ticket_card(
    *,
    cfg: BotConfig,
    api: TelegramAPI,
    orch_q: OrchestratorQueue,
    ticket_id: str,
) -> None:
    """
    Best-effort edit of the single ticket card message (no spam).
    """
    root_id = (ticket_id or "").strip()
    if not root_id:
        return
    root = orch_q.get_job(root_id)
    if root is None:
        return
    chat_id = int(root.chat_id)
    trace = dict(root.trace or {})
    mid = trace.get("ticket_card_message_id")
    try:
        message_id = int(mid) if mid is not None else None
    except Exception:
        message_id = None
    if message_id is None:
        return

    now = time.time()
    with _TICKET_CARD_LOCK:
        k = (chat_id, root_id)
        last = float(_TICKET_CARD_LAST_EDIT.get(k, 0.0) or 0.0)
        if (now - last) < float(_TICKET_CARD_MIN_EDIT_INTERVAL_S):
            return
        _TICKET_CARD_LAST_EDIT[k] = now

    try:
        api.edit_message_text(chat_id, int(message_id), _ticket_card_text(orch_q, ticket_id=root_id))
    except Exception:
        # If edit fails, fall back to sending a fresh card and updating trace.
        try:
            new_mid = api.send_message(chat_id, _ticket_card_text(orch_q, ticket_id=root_id))
            if new_mid is not None:
                orch_q.update_trace(root_id, ticket_card_message_id=int(new_mid), ticket_card_replaced_at=time.time())
        except Exception:
            pass


def _orchestrator_inbox_text(orch_q: OrchestratorQueue, role: str | None) -> str:
    items = orch_q.inbox(role=role, limit=25)
    if not items:
        return "Inbox: empty."
    title = f"Inbox ({role})" if role else "Inbox"
    lines = [title, "=" * len(title)]
    for t in items:
        snippet = (t.input_text or "").strip().replace("\n", " ")[:160]
        lines.append(f"- {t.job_id[:8]} role={t.role} state={t.state} text={snippet}")
    return "\n".join(lines)


def _orchestrator_runbooks_text(cfg: BotConfig, orch_q: OrchestratorQueue) -> str:
    if not cfg.runbooks_enabled:
        return "Runbooks: disabled."
    rbs = load_runbooks(cfg.runbooks_path)
    if not rbs:
        return f"Runbooks: none found at {cfg.runbooks_path}"
    lines = ["Runbooks", "=" * 8]
    now = time.time()
    for rb in rbs:
        last = orch_q.get_runbook_last_run(runbook_id=rb.runbook_id)
        due = "DUE" if runbook_due(rb, last_run_at=last, now=now) else "ok"
        last_s = "never" if not last else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last))
        lines.append(f"- {rb.runbook_id} role={rb.role} every={rb.interval_seconds}s enabled={rb.enabled} last={last_s} {due}")
    return "\n".join(lines)


def _parse_job(cfg: BotConfig, msg: IncomingMessage) -> tuple[str, Job | None]:
    """
    Returns (response_text, job)
    If response_text is non-empty, caller should send it immediately.
    """
    text = _normalize_slash_aliases((msg.text or "").strip())
    if not text:
        return "", None

    # Friendly local handling for greetings: avoid creating a ticket card for
    # tiny social messages.
    if not text.startswith("/") and _is_greeting(text):
        return "Jarvis: ¡Hola! Soy tu mano derecha ejecutiva. ¿En qué puedo ayudar?", None

    # CEO control-plane: clear queue/backlog should be immediate and deterministic.
    if not text.startswith("/") and _is_purge_queue_request(text):
        return _orch_marker("purge_queue", "global"), None

    if text in ("/start", "/help"):
        profile = _auth_effective_profile_name(cfg, chat_id=msg.chat_id) if cfg.auth_enabled else ""
        eff_cfg = _apply_profile_to_cfg(cfg, profile_name=profile) if profile else cfg
        return _help_text(eff_cfg), None

    if text == "/whoami":
        return _whoami_text(msg), None

    if text == "/logout":
        if not cfg.auth_enabled:
            return "Auth disabled on this bot.", None
        return "__logout__", None

    if text.startswith("/login "):
        if not cfg.auth_enabled:
            return "Auth disabled on this bot.", None
        # Handled in poll_loop (needs access to chat_id + updates auth state).
        return "__login__:" + text[len("/login ") :].strip(), None

    if text == "/login":
        if not cfg.auth_enabled:
            return "Auth disabled on this bot.", None
        return "Uso: /login <usuario> <password>", None

    if text == "/cancel":
        return "__cancel__", None

    if text == "/purge":
        return _orch_marker("purge_queue", "global"), None

    if text.startswith("/purge "):
        scope = (text[len("/purge ") :] or "").strip().lower()
        if scope in ("chat", "here", "aqui", "aquí", "this"):
            return _orch_marker("purge_queue", "chat"), None
        if scope in ("all", "global", "company"):
            return _orch_marker("purge_queue", "global"), None
        return "Usage: /purge [chat|global]", None

    if text == "/synccommands":
        return "__synccommands__", None

    if text == "/restart":
        if cfg.auth_enabled:
            profile = _auth_effective_profile_name(cfg, chat_id=msg.chat_id)
            if profile and not _profile_can_manage_bot(cfg, profile_name=profile):
                return f"No permitido por tu perfil ({profile}).", None
        return "__restart__", None

    if text == "/status":
        profile = _auth_effective_profile_name(cfg, chat_id=msg.chat_id) if cfg.auth_enabled else ""
        eff_cfg = _apply_profile_to_cfg(cfg, profile_name=profile) if profile else cfg
        oll = ""
        if eff_cfg.codex_use_oss and eff_cfg.codex_local_provider == "ollama":
            _, oll = _ollama_status()
        codex = _codex_version()
        lines = [codex]
        if oll:
            lines.append(oll)
        model, effort = _job_model_label(eff_cfg, ["exec"], chat_id=msg.chat_id)
        model_label = _format_model_for_display(model, effort)
        qmax = "unbounded" if eff_cfg.queue_maxsize == 0 else str(eff_cfg.queue_maxsize)
        lines += [
            f"workdir: {eff_cfg.codex_workdir}",
            f"mode: default={eff_cfg.codex_default_mode}",
            f"provider: {eff_cfg.codex_local_provider if eff_cfg.codex_use_oss else 'default (non-oss)'}",
            f"model: {model_label}",
            f"workers: {eff_cfg.worker_count}",
            f"queue_maxsize: {qmax}",
            f"max_queued_per_chat: {eff_cfg.max_queued_per_chat}",
            f"heartbeat_seconds: {eff_cfg.heartbeat_seconds}",
            f"send_as_file_threshold_chars: {eff_cfg.send_as_file_threshold_chars}",
        ]
        return "\n".join(lines), None

    if text == "/agents":
        return _orch_marker("agents"), None

    if text == "/dashboard":
        return _orch_marker("dashboard"), None

    if text == "/watch":
        return _orch_marker("watch", "on"), None

    if text in ("/unwatch", "/watch off", "/watch stop"):
        return _orch_marker("watch", "off"), None

    if text == "/orders":
        return _orch_marker("orders"), None

    if text.startswith("/order "):
        payload = text[len("/order ") :].strip()
        if not payload:
            return "Usage: /order show|pause|done <id>", None
        return _orch_marker("order", payload), None

    if text.startswith("/job "):
        raw = text[len("/job ") :].strip()
        if not raw:
            return "Uso: /job [list|show <id>|del <id>]", None
        try:
            parts = shlex.split(raw)
        except Exception:
            parts = raw.split()
        if not parts:
            return "Uso: /job [list|show <id>|del <id>]", None
        cmd = parts[0].strip().lower()
        if cmd in ("list",):
            return _orch_marker("job", "list"), None
        if cmd in ("show", "get"):
            if len(parts) < 2 or not parts[1].strip():
                return "Uso: /job show <id>", None
            return _orch_marker("job", f"show {parts[1].strip()}"), None
        if cmd in ("del", "delete", "rm"):
            if len(parts) < 2 or not parts[1].strip():
                return "Uso: /job del <id>", None
            tail = " ".join(parts[2:]).strip()
            payload = f"del {parts[1].strip()}" + (f" {tail}" if tail else "")
            return _orch_marker("job", payload), None
        # Back-compat: /job <id>
        job_id = _orch_job_id(raw)
        if not job_id:
            return "Uso: /job [list|show <id>|del <id>]", None
        return _orch_marker("job", job_id), None

    if text == "/job":
        return _orch_marker("job", "list"), None

    if text.startswith("/ticket "):
        job_id = _orch_job_id(text[len("/ticket ") :])
        if not job_id:
            return "Uso: /ticket <id>", None
        return _orch_marker("ticket", job_id), None

    if text == "/ticket":
        return "Uso: /ticket <id>", None

    if text == "/inbox":
        return _orch_marker("inbox"), None

    if text.startswith("/inbox "):
        role = _orch_job_id(text[len("/inbox ") :]).lower()
        if not role:
            return "Uso: /inbox [role]", None
        return _orch_marker("inbox", role), None

    if text == "/runbooks":
        return _orch_marker("runbooks"), None

    if text == "/reset_role":
        return "Uso: /reset_role <role|all>", None

    if text.startswith("/reset_role "):
        role = _orch_job_id(text[len("/reset_role ") :]).lower()
        if not role:
            return "Uso: /reset_role <role|all>", None
        return _orch_marker("reset_role", role), None

    if text == "/daily":
        return _orch_marker("daily"), None

    if text == "/brief":
        return _orch_marker("brief"), None

    if text == "/pause":
        return "Uso: /pause <role>", None

    if text.startswith("/pause "):
        role = _orch_job_id(text[len("/pause ") :]).lower()
        if not role:
            return "Uso: /pause <role>", None
        return _orch_marker("pause", role), None

    if text == "/emergency_stop":
        return _orch_marker("emergency_stop"), None

    if text == "/emergency_resume":
        return _orch_marker("emergency_resume"), None

    if text == "/resume":
        return "Uso: /resume <role>", None

    if text.startswith("/resume "):
        role = _orch_job_id(text[len("/resume ") :]).lower()
        if not role:
            return "Uso: /resume <role>", None
        return _orch_marker("resume", role), None

    if text.startswith("/approve "):
        job_id = _orch_job_id(text[len("/approve ") :])
        if not job_id:
            return "Uso: /approve <id>", None
        return _orch_marker("approve", job_id), None

    if text.startswith("/cancel "):
        job_id = _orch_job_id(text[len("/cancel ") :])
        if not job_id:
            # Preserve legacy behavior for "/cancel" without args.
            return "__cancel__", None
        return _orch_marker("cancel_job", job_id), None

    if text == "/snapshot":
        return "Uso: /snapshot <url|objetivo>", None

    if text.startswith("/snapshot "):
        target = text[len("/snapshot") :].strip()
        if not target:
            return "Uso: /snapshot <url|objetivo>", None
        # Force frontend role explicitly.
        snapshot_text = f"@frontend Solicitud de snapshot: {target}. Genera captura visual y devuelve ruta/descripcion."
        return (
            "",
            Job(
                chat_id=msg.chat_id,
                reply_to_message_id=msg.message_id,
                user_text=snapshot_text,
                argv=["exec", snapshot_text],
                mode_hint="rw",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            ),
        )

    if text == "/permissions":
        profile = _auth_effective_profile_name(cfg, chat_id=msg.chat_id) if cfg.auth_enabled else ""
        eff_cfg = _apply_profile_to_cfg(cfg, profile_name=profile) if profile else cfg
        return _permissions_text(eff_cfg, chat_id=msg.chat_id), None

    if text.startswith("/permissions "):
        arg = text[len("/permissions ") :].strip().lower()
        if cfg.auth_enabled:
            profile = _auth_effective_profile_name(cfg, chat_id=msg.chat_id)
            if profile and not _profile_can_set_permissions(cfg, profile_name=profile):
                return f"No permitido por tu perfil ({profile}).", None
            if profile and _profile_max_mode(cfg, profile_name=profile) != "full":
                return f"No permitido por tu perfil ({profile}).", None
        if arg in ("default", "full"):
            _set_access_mode(cfg, arg, chat_id=msg.chat_id)
            return f"OK. permissions={arg}", None
        if arg == "clear":
            _set_access_mode(cfg, None, chat_id=msg.chat_id)
            return "OK. permissions cleared (using env defaults).", None
        return "Usage: /permissions default|full|clear", None

    if text == "/botpermissions":
        # Bot + Codex CLI execution policy (not OS permissions).
        profile = _auth_effective_profile_name(cfg, chat_id=msg.chat_id) if cfg.auth_enabled else ""
        eff_cfg = _apply_profile_to_cfg(cfg, profile_name=profile) if profile else cfg
        bypass = _effective_bypass_sandbox(eff_cfg, chat_id=msg.chat_id)
        lines = [
            f"permissions: {'full' if bypass else 'default'}",
            f"profile: {profile or '(none)'}",
            f"strict_proxy: {'ON' if eff_cfg.strict_proxy else 'off'}",
            f"unsafe_direct_codex: {'ON' if eff_cfg.unsafe_direct_codex else 'off'}",
            f"telegram_parse_mode: {eff_cfg.telegram_parse_mode or '(empty)'}",
            f"workdir: {eff_cfg.codex_workdir}",
            f"default_mode: {eff_cfg.codex_default_mode}",
            f"force_full_access: {'ON' if eff_cfg.codex_force_full_access else 'off'}",
        ]
        if bypass:
            lines.append("codex: --dangerously-bypass-approvals-and-sandbox (no approvals, no sandbox)")
        else:
            lines.append("codex: -a never (no approval prompts)")
            lines.append(f"codex sandbox default: {_threaded_sandbox_mode_label(eff_cfg)}")
        return "\n".join(lines), None

    if text == "/format":
        return _format_preview_text(), None
    
    if text == "/example":
        return _format_preview_text(), None

    if text == "/skills":
        return _skills_status_text(), None

    if text.startswith("/skills "):
        argline = text[len("/skills ") :].strip()
        if not argline:
            return _skills_status_text(), None

        parts = argline.split()
        sub = (parts[0] or "").strip().lower()
        rest = parts[1:]

        if sub in ("help", "?"):
            return _skills_status_text(), None

        if sub in ("catalog", "list", "available"):
            flt = " ".join(rest).strip()
            # Run in worker to avoid blocking the polling loop (network call to GitHub API).
            return "", Job(
                chat_id=msg.chat_id,
                reply_to_message_id=msg.message_id,
                user_text=text,
                argv=["__skills__", "catalog", flt],
                mode_hint="ro",
                epoch=0,
                threaded=False,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            )

        if sub == "install":
            if not rest:
                return "Uso: /skills install <skill>  (o: /skills install experimental/<skill>)", None

            spec = rest[0].strip()
            scope = "curated"
            name = spec
            if "/" in spec:
                a, b = spec.split("/", 1)
                a = a.strip().lower()
                b = b.strip()
                if a in ("experimental", "exp"):
                    scope = "experimental"
                    name = b
                else:
                    return "Solo soporto: <skill> o experimental/<skill>.", None

            if not _skill_segment_ok(name):
                return "Nombre de skill invalido. Ejemplos: imagegen, gh-fix-ci", None

            if cfg.auth_enabled:
                profile = _auth_effective_profile_name(cfg, chat_id=msg.chat_id)
                if profile and not _profile_can_manage_bot(cfg, profile_name=profile):
                    return f"No permitido por tu perfil ({profile}).", None

            # Install can take time; run in worker.
            return "", Job(
                chat_id=msg.chat_id,
                reply_to_message_id=msg.message_id,
                user_text=text,
                argv=["__skills__", "install", scope, name],
                mode_hint="ro",
                epoch=0,
                threaded=False,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            )

        if sub in ("enable", "disable"):
            if not rest:
                return f"Uso: /skills {sub} <skill>", None
            name = rest[0].strip()
            if not _skill_segment_ok(name):
                return "Nombre de skill invalido. Ejemplos: imagegen, gh-fix-ci", None

            if cfg.auth_enabled:
                profile = _auth_effective_profile_name(cfg, chat_id=msg.chat_id)
                if profile and not _profile_can_manage_bot(cfg, profile_name=profile):
                    return f"No permitido por tu perfil ({profile}).", None

            skills_root = _skills_root_dir()
            disabled_root = skills_root / ".disabled"
            src = skills_root / name if sub == "disable" else (disabled_root / name)
            dst = (disabled_root / name) if sub == "disable" else (skills_root / name)

            if src.parts and ".system" in src.parts:
                return "No puedo activar/desactivar skills de .system.", None

            try:
                _move_skill_dir(src=src, dst=dst)
                return f"OK. {sub} {name}", None
            except FileNotFoundError:
                where = "~/.codex/skills" if sub == "disable" else "~/.codex/skills/.disabled"
                return f"No encontre {name} en {where}.", None
            except FileExistsError:
                return f"Ya existe el destino para {name}.", None
            except Exception as e:
                LOG.exception("Failed to %s skill: %s", sub, name)
                return f"Error al {sub} {name}: {e}", None

        return _help_text(cfg), None

    if text == "/model":
        openai_override, oss_override = _get_model_overrides(cfg, chat_id=msg.chat_id)
        openai_eff_override, oss_eff_override = _get_effort_overrides(cfg, chat_id=msg.chat_id)
        cfg_model, cfg_effort = _codex_defaults_from_config()

        openai_model = openai_override or cfg.codex_openai_model or cfg_model
        oss_model = oss_override or cfg.codex_oss_model
        openai_eff = openai_eff_override or cfg_effort
        oss_eff = oss_eff_override or cfg_effort

        active_model = oss_model if cfg.codex_use_oss else openai_model
        active_eff = oss_eff if cfg.codex_use_oss else openai_eff
        active_label = _format_model_for_display(active_model or "(unknown)", active_eff)

        choices = _model_choices_for_display()
        choice_lines: list[str] = []
        if choices:
            for i, (slug, _display, default_effort, effs) in enumerate(choices, start=1):
                eff_part = f"default={default_effort}" if default_effort else ""
                if effs:
                    eff_part = (eff_part + " " if eff_part else "") + "efforts=" + ",".join(effs)
                suffix = f" ({eff_part})" if eff_part else ""
                choice_lines.append(f"{i}. {slug}{suffix}")

        return (
            "\n".join(
                [
                    f"mode: {'oss' if cfg.codex_use_oss else 'openai'}",
                    f"active: {active_label}",
                    f"openai model: {openai_model or '(unknown)'}",
                    f"openai effort: {openai_eff or '(unknown)'}",
                    f"oss model: {oss_model or '(default)'}",
                    f"oss effort: {oss_eff or '(unknown)'}",
                    "",
                    "Available (from ~/.codex/models_cache.json):" if choice_lines else "Available: (no local models cache found)",
                    *choice_lines,
                    "",
                    "Usage:",
                    "- /model <name> [low|medium|high|xhigh]",
                    "- /model <number> [low|medium|high|xhigh]",
                    "- /model openai <name> [effort]",
                    "- /model oss <name> [effort]",
                    "- /model clear",
                    "- /effort <low|medium|high|xhigh>",
                    "- /effort clear",
                ]
            ),
            None,
        )

    if text.startswith("/model "):
        args = text.split()
        choices = _model_choices_for_display()
        efforts = {"low", "medium", "high", "xhigh"}
        maybe_eff = _normalize_effort_token(args[-1]) if len(args) >= 2 else ""
        eff: str = maybe_eff if maybe_eff in efforts else ""

        def _resolve_model_token(tok: str) -> tuple[str, str]:
            t = (tok or "").strip()
            if not t:
                return "", ""
            if t.isdigit() and choices:
                idx = int(t)
                if 1 <= idx <= len(choices):
                    slug, _display, default_effort, _effs = choices[idx - 1]
                    return slug, default_effort
            for slug, _display, default_effort, _effs in choices:
                if slug == t:
                    return slug, default_effort
            return t, ""

        # /model clear
        if len(args) == 2 and args[1].lower() == "clear":
            st = _get_state(cfg)
            by_chat_models = st.get("model_overrides_by_chat")
            if not isinstance(by_chat_models, dict):
                by_chat_models = {}
            by_chat_efforts = st.get("effort_overrides_by_chat")
            if not isinstance(by_chat_efforts, dict):
                by_chat_efforts = {}
            key = str(int(msg.chat_id))
            rec_m = by_chat_models.get(key) if isinstance(by_chat_models.get(key), dict) else {}
            rec_e = by_chat_efforts.get(key) if isinstance(by_chat_efforts.get(key), dict) else {}
            if cfg.codex_use_oss:
                rec_m.pop("oss_model", None)
                rec_e.pop("oss_effort", None)
            else:
                rec_m.pop("openai_model", None)
                rec_e.pop("openai_effort", None)
            if rec_m:
                by_chat_models[key] = rec_m
            else:
                by_chat_models.pop(key, None)
            if rec_e:
                by_chat_efforts[key] = rec_e
            else:
                by_chat_efforts.pop(key, None)
            st["model_overrides_by_chat"] = by_chat_models
            st["effort_overrides_by_chat"] = by_chat_efforts
            _atomic_write_json(cfg.state_file, st)
            return "OK. Cleared model/effort override for current mode.", None

        # /model openai <name> OR /model oss <name>
        if len(args) >= 3 and args[1].lower() in ("openai", "oss"):
            scope = args[1].lower()
            name_tokens = args[2:-1] if eff else args[2:]
            name_raw = " ".join(name_tokens).strip()
            name, default_eff = _resolve_model_token(name_raw)
            if not name:
                return "Usage: /model openai <name> [effort] OR /model oss <name> [effort]", None
            st = _get_state(cfg)
            by_chat_models = st.get("model_overrides_by_chat")
            if not isinstance(by_chat_models, dict):
                by_chat_models = {}
            by_chat_efforts = st.get("effort_overrides_by_chat")
            if not isinstance(by_chat_efforts, dict):
                by_chat_efforts = {}
            key = str(int(msg.chat_id))
            rec_m = by_chat_models.get(key) if isinstance(by_chat_models.get(key), dict) else {}
            rec_e = by_chat_efforts.get(key) if isinstance(by_chat_efforts.get(key), dict) else {}
            if scope == "openai":
                rec_m["openai_model"] = name
                if eff:
                    rec_e["openai_effort"] = eff
                elif default_eff:
                    rec_e["openai_effort"] = default_eff
            else:
                rec_m["oss_model"] = name
                if eff:
                    rec_e["oss_effort"] = eff
                elif default_eff:
                    rec_e["oss_effort"] = default_eff
            by_chat_models[key] = rec_m
            by_chat_efforts[key] = rec_e
            st["model_overrides_by_chat"] = by_chat_models
            st["effort_overrides_by_chat"] = by_chat_efforts
            _atomic_write_json(cfg.state_file, st)
            eff_set = eff or default_eff
            if eff_set:
                return f"OK. Set {scope} model to: {name} (effort={eff_set})", None
            return f"OK. Set {scope} model to: {name}", None

        # /model <name> (set for current mode)
        rest_tokens = args[1:-1] if eff else args[1:]
        name_raw = " ".join(rest_tokens).strip()
        name, default_eff = _resolve_model_token(name_raw)
        if not name:
            return "Usage: /model <name> [effort]", None
        st = _get_state(cfg)
        by_chat_models = st.get("model_overrides_by_chat")
        if not isinstance(by_chat_models, dict):
            by_chat_models = {}
        by_chat_efforts = st.get("effort_overrides_by_chat")
        if not isinstance(by_chat_efforts, dict):
            by_chat_efforts = {}
        key = str(int(msg.chat_id))
        rec_m = by_chat_models.get(key) if isinstance(by_chat_models.get(key), dict) else {}
        rec_e = by_chat_efforts.get(key) if isinstance(by_chat_efforts.get(key), dict) else {}
        if cfg.codex_use_oss:
            rec_m["oss_model"] = name
            if eff:
                rec_e["oss_effort"] = eff
            elif default_eff:
                rec_e["oss_effort"] = default_eff
        else:
            rec_m["openai_model"] = name
            if eff:
                rec_e["openai_effort"] = eff
            elif default_eff:
                rec_e["openai_effort"] = default_eff
        by_chat_models[key] = rec_m
        by_chat_efforts[key] = rec_e
        st["model_overrides_by_chat"] = by_chat_models
        st["effort_overrides_by_chat"] = by_chat_efforts
        _atomic_write_json(cfg.state_file, st)
        eff_set = eff or default_eff
        if eff_set:
            return f"OK. Set model for current mode to: {name} (effort={eff_set})", None
        return f"OK. Set model for current mode to: {name}", None

    if text == "/voice":
        enabled = _effective_transcribe_enabled(cfg)
        backend = _effective_transcribe_backend(cfg)
        model_path = _effective_whisper_model_path(cfg)
        threads = _effective_whisper_threads(cfg)
        timeout_s = _effective_transcribe_timeout(cfg)
        lang = _effective_transcribe_language(cfg)
        return (
            "\n".join(
                [
                    f"voice transcription: {'ON' if enabled else 'off'}",
                    f"backend: {backend}",
                    f"whisper.cpp model: {model_path}",
                    f"whisper.cpp threads: {threads}",
                    f"timeout_seconds: {timeout_s}",
                    f"language: {lang or '(auto)'}",
                    "",
                    "Uso:",
                    "- /voice on|off",
                    "- /voice accuracy   (whisper.cpp + medium)",
                    "- /voice speed      (whisper.cpp + small)",
                    "- /voice backend auto|whispercpp|openai",
                    "- /voice language <es|en|...> | /voice language clear",
                    "- /voice threads <1-64>",
                    "- /voice timeout <5-3600>",
                    "- /voice clear",
                ]
            ),
            None,
        )

    if text.startswith("/voice "):
        arg = text[len("/voice ") :].strip()
        if not arg:
            return _parse_job(cfg, IncomingMessage(**{**msg.__dict__, "text": "/voice"}))

        toks = arg.split()
        head = toks[0].strip().lower()

        if head in ("on", "off"):
            vs = _get_voice_state(cfg)
            vs["enabled"] = (head == "on")
            _set_voice_state(cfg, vs)
            return f"OK. voice transcription={'ON' if head == 'on' else 'off'}", None

        if head == "clear":
            _clear_voice_state(cfg)
            return "OK. voice settings cleared (using env defaults).", None

        if head == "backend":
            val = toks[1].strip().lower() if len(toks) >= 2 else ""
            if val not in ("auto", "whispercpp", "openai"):
                return "Uso: /voice backend auto|whispercpp|openai", None
            vs = _get_voice_state(cfg)
            vs["backend"] = val
            _set_voice_state(cfg, vs)
            return f"OK. backend={val}", None

        if head == "language":
            val = toks[1].strip() if len(toks) >= 2 else ""
            if not val:
                return "Uso: /voice language <es|en|...> | /voice language clear", None
            vs = _get_voice_state(cfg)
            if val.lower() == "clear":
                vs.pop("language", None)
                _set_voice_state(cfg, vs)
                return "OK. language cleared (auto-detect).", None
            vs["language"] = val
            _set_voice_state(cfg, vs)
            return f"OK. language={val}", None

        if head == "threads":
            if len(toks) < 2:
                return "Uso: /voice threads <1-64>", None
            n = _voice_int(toks[1], min_value=1, max_value=64)
            if n is None:
                return "Uso: /voice threads <1-64>", None
            vs = _get_voice_state(cfg)
            vs["whisper_threads"] = n
            _set_voice_state(cfg, vs)
            return f"OK. whisper_threads={n}", None

        if head == "timeout":
            if len(toks) < 2:
                return "Uso: /voice timeout <5-3600>", None
            n = _voice_int(toks[1], min_value=5, max_value=3600)
            if n is None:
                return "Uso: /voice timeout <5-3600>", None
            vs = _get_voice_state(cfg)
            vs["timeout_seconds"] = n
            _set_voice_state(cfg, vs)
            return f"OK. timeout_seconds={n}", None

        if head in ("accuracy", "speed") or head == "model":
            # Resolve model token to a local ggml file under codexbot/models by default.
            if head == "accuracy":
                name = "medium"
            elif head == "speed":
                name = "small"
            else:
                name = toks[1].strip().lower() if len(toks) >= 2 else ""
                if not name:
                    return "Uso: /voice model tiny|base|small|medium|large", None
            if name not in ("tiny", "base", "small", "medium", "large"):
                return "Uso: /voice model tiny|base|small|medium|large", None
            models_dir = Path(__file__).resolve().parent / "models"
            model_path = str((models_dir / f"ggml-{name}.bin").resolve())
            vs = _get_voice_state(cfg)
            vs["enabled"] = True
            vs["backend"] = "whispercpp"
            vs["whisper_model_path"] = model_path
            # Keep existing threads unless unset; but accuracy defaults want more parallelism.
            if "whisper_threads" not in vs:
                vs["whisper_threads"] = 8
            _set_voice_state(cfg, vs)
            extra = "" if Path(model_path).exists() else " (nota: el archivo no existe aun)"
            return f"OK. whisper model={name}{extra}", None

        return "Uso: /voice (manda /voice para ver ayuda)", None

    if text == "/effort":
        openai_eff_override, oss_eff_override = _get_effort_overrides(cfg, chat_id=msg.chat_id)
        _, cfg_effort = _codex_defaults_from_config()
        openai_eff = openai_eff_override or cfg_effort or "(unknown)"
        oss_eff = oss_eff_override or cfg_effort or "(unknown)"
        active_eff = oss_eff if cfg.codex_use_oss else openai_eff
        return (
            "\n".join(
                [
                    f"mode: {'oss' if cfg.codex_use_oss else 'openai'}",
                    f"active effort: {active_eff}",
                    f"openai effort: {openai_eff}",
                    f"oss effort: {oss_eff}",
                    "",
                    "Usage:",
                    "- /effort low|medium|high|xhigh",
                    "- /effort clear",
                ]
            ),
            None,
        )

    if text.startswith("/effort "):
        val = _normalize_effort_token(text[len("/effort ") :])
        if not val:
            return "Usage: /effort low|medium|high|xhigh OR /effort clear", None
        if val == "clear":
            st = _get_state(cfg)
            by_chat_efforts = st.get("effort_overrides_by_chat")
            if not isinstance(by_chat_efforts, dict):
                by_chat_efforts = {}
            key = str(int(msg.chat_id))
            rec_e = by_chat_efforts.get(key) if isinstance(by_chat_efforts.get(key), dict) else {}
            if cfg.codex_use_oss:
                rec_e.pop("oss_effort", None)
            else:
                rec_e.pop("openai_effort", None)
            if rec_e:
                by_chat_efforts[key] = rec_e
            else:
                by_chat_efforts.pop(key, None)
            st["effort_overrides_by_chat"] = by_chat_efforts
            _atomic_write_json(cfg.state_file, st)
            return "OK. Cleared effort override for current mode.", None
        if val not in ("low", "medium", "high", "xhigh"):
            return "Invalid effort. Use: low, medium, high, xhigh.", None
        st = _get_state(cfg)
        by_chat_efforts = st.get("effort_overrides_by_chat")
        if not isinstance(by_chat_efforts, dict):
            by_chat_efforts = {}
        key = str(int(msg.chat_id))
        rec_e = by_chat_efforts.get(key) if isinstance(by_chat_efforts.get(key), dict) else {}
        if cfg.codex_use_oss:
            rec_e["oss_effort"] = val
        else:
            rec_e["openai_effort"] = val
        by_chat_efforts[key] = rec_e
        st["effort_overrides_by_chat"] = by_chat_efforts
        _atomic_write_json(cfg.state_file, st)
        return f"OK. Set effort for current mode to: {val}", None

    if text == "/setnotify":
        # Preserve any existing state (e.g. model overrides).
        st = _get_state(cfg)
        st["notify_chat_id"] = msg.chat_id
        _atomic_write_json(cfg.state_file, st)
        return f"OK. notify_chat_id={msg.chat_id}", None

    if text.startswith("/notify "):
        payload = text[len("/notify ") :].strip()
        if not payload:
            return "Usage: /notify <text>", None
        state = _read_json(cfg.state_file)
        chat_id = cfg.notify_chat_id or state.get("notify_chat_id")
        try:
            chat_id = int(chat_id) if chat_id is not None else None
        except Exception:
            chat_id = None
        if not chat_id:
            return "No notify chat set. Run /setnotify in the chat you want me to message.", None
        # Poll loop will send this response; the actual notify send happens in poll loop using api.
        return f"__notify__:{chat_id}:{payload}", None

    # Explicit modes.
    if text.startswith("/ro "):
        prompt = text[len("/ro ") :].strip()
        if not prompt:
            return "Usage: /ro <prompt>", None
        return (
            "",
            Job(
                chat_id=msg.chat_id,
                reply_to_message_id=msg.message_id,
                user_text=prompt,
                argv=["exec", prompt],
                mode_hint="ro",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            ),
        )

    if text.startswith("/rw "):
        prompt = text[len("/rw ") :].strip()
        if not prompt:
            return "Usage: /rw <prompt>", None
        return (
            "",
            Job(
                chat_id=msg.chat_id,
                reply_to_message_id=msg.message_id,
                user_text=prompt,
                argv=["exec", prompt],
                mode_hint="rw",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            ),
        )

    if text.startswith("/full "):
        prompt = text[len("/full ") :].strip()
        if not prompt:
            return "Usage: /full <prompt>", None
        return (
            "",
            Job(
                chat_id=msg.chat_id,
                reply_to_message_id=msg.message_id,
                user_text=prompt,
                argv=["exec", prompt],
                mode_hint="full",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            ),
        )

    if text.startswith("/"):
        # Default: only allow a small set of passthrough commands (avoid accidental triggers).
        # Unsafe mode: treat unknown /<anything> as a prompt to `codex exec <anything>`.
        #
        # Examples (safe default):
        # - `/exec --help` => `codex exec --help`
        # - `/codex --version` => `codex --version`
        #
        # Examples (unsafe mode):
        # - `/ls -la` => `codex exec "ls -la"`
        # - `/git status` => `codex exec "git status"`
        raw = text[1:].strip()
        if not raw:
            return _help_text(cfg), None
        first = (raw.split(None, 1)[0] if raw else "").lower()
        if first not in _ALLOWED_PASSTHROUGH_SLASH:
            if cfg.unsafe_direct_codex:
                # Treat unknown slash commands as a plain prompt.
                return (
                    "",
                    Job(
                        chat_id=msg.chat_id,
                        reply_to_message_id=msg.message_id,
                        user_text=raw,
                        argv=["exec", raw],
                        mode_hint=cfg.codex_default_mode,
                        epoch=0,
                        threaded=True,
                        image_paths=[],
                        upload_paths=[],
                        force_new_thread=False,
                    ),
                )
            return _help_text(cfg), None
        if first == "codex":
            raw = raw[len("codex") :].strip()
            if not raw:
                return _help_text(cfg), None
        try:
            argv = shlex.split(raw)
        except ValueError as e:
            return f"Parse error: {e}", None
        err = _validate_codex_argv(cfg, argv, cfg.codex_default_mode)
        if err:
            return err, None
        return (
            "",
            Job(
                chat_id=msg.chat_id,
                reply_to_message_id=msg.message_id,
                user_text=text,
                argv=argv,
                mode_hint=cfg.codex_default_mode,
                epoch=0,
                threaded=False,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            ),
        )

    # Plain text: default mode.
    return (
        "",
        Job(
            chat_id=msg.chat_id,
            reply_to_message_id=msg.message_id,
            user_text=text,
            argv=["exec", text],
            # For orchestrator mode, keep legacy default out of the job so role profiles can
            # decide (and explicit /ro|/rw still overrides).
            mode_hint=("" if cfg.orchestrator_enabled else cfg.codex_default_mode),
            epoch=0,
            threaded=True,
            image_paths=[],
            upload_paths=[],
            force_new_thread=False,
        ),
    )


def _terminate_process(proc: subprocess.Popen[object]) -> None:
    """
    Best-effort terminate the process (and its children) on POSIX.
    """
    try:
        # We start Codex with start_new_session=True, so it has its own process group.
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


class JobTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._epoch_by_chat: dict[int, int] = {}
        self._queued_by_chat: dict[int, int] = {}
        self._inflight_by_chat: dict[int, int] = {}
        self._proc_by_chat: dict[int, subprocess.Popen[object]] = {}

    def current_epoch(self, chat_id: int) -> int:
        return self._epoch_by_chat.get(chat_id, 0)

    def queued(self, chat_id: int) -> int:
        return self._queued_by_chat.get(chat_id, 0)

    def inflight(self, chat_id: int) -> int:
        return self._inflight_by_chat.get(chat_id, 0)

    def try_mark_enqueued(self, chat_id: int, *, max_queued_per_chat: int) -> tuple[bool, str, int, int]:
        """
        Returns (ok, reason, epoch, queued_after).
        """
        with self._lock:
            epoch = self._epoch_by_chat.get(chat_id, 0)
            q = self._queued_by_chat.get(chat_id, 0)
            if max_queued_per_chat < 0:
                max_queued_per_chat = 0
            if q >= max_queued_per_chat and max_queued_per_chat != 0:
                return False, f"Too many queued jobs for this chat (max={max_queued_per_chat}).", epoch, q
            if max_queued_per_chat == 0 and (q > 0 or self._inflight_by_chat.get(chat_id, 0) > 0):
                return False, "A job is already running/queued for this chat.", epoch, q
            self._queued_by_chat[chat_id] = q + 1
            return True, "", epoch, q + 1

    def on_dequeue(self, chat_id: int) -> None:
        with self._lock:
            q = self._queued_by_chat.get(chat_id, 0)
            self._queued_by_chat[chat_id] = max(0, q - 1)

    def wait_turn_and_mark_inflight(self, job: Job, stop_event: threading.Event) -> bool:
        """
        Serialize execution per-chat across multiple workers.
        Returns False if job is stale (canceled) or stop requested.
        """
        chat_id = job.chat_id
        with self._cond:
            while True:
                if stop_event.is_set():
                    return False
                epoch = self._epoch_by_chat.get(chat_id, 0)
                if job.epoch != epoch:
                    return False
                if self._inflight_by_chat.get(chat_id, 0) == 0:
                    self._inflight_by_chat[chat_id] = 1
                    return True
                self._cond.wait(timeout=0.5)

    def set_running_proc(self, chat_id: int, proc: subprocess.Popen[object]) -> None:
        with self._cond:
            self._proc_by_chat[chat_id] = proc

    def clear_running(self, chat_id: int) -> None:
        with self._cond:
            self._proc_by_chat.pop(chat_id, None)
            self._inflight_by_chat[chat_id] = 0
            self._cond.notify_all()

    def cancel(self, chat_id: int) -> bool:
        """
        Increments epoch (dropping queued jobs) and terminates running proc if any.
        Returns True if there was a running proc.
        """
        with self._cond:
            self._epoch_by_chat[chat_id] = self._epoch_by_chat.get(chat_id, 0) + 1
            # Drop queued count; stale jobs will be skipped when dequeued.
            self._queued_by_chat[chat_id] = 0
            proc = self._proc_by_chat.get(chat_id)
            self._cond.notify_all()

        if proc is None:
            return False
        _terminate_process(proc)
        return True


class ThreadManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread_by_chat: dict[int, str] = {}
        self._model_by_chat: dict[int, str] = {}

    def get(self, chat_id: int) -> str | None:
        with self._lock:
            return self._thread_by_chat.get(int(chat_id))

    def set(self, chat_id: int, thread_id: str) -> None:
        tid = (thread_id or "").strip()
        if not tid:
            return
        with self._lock:
            self._thread_by_chat[int(chat_id)] = tid

    def clear(self, chat_id: int) -> None:
        with self._lock:
            self._thread_by_chat.pop(int(chat_id), None)

    def get_model(self, chat_id: int) -> str | None:
        with self._lock:
            m = self._model_by_chat.get(int(chat_id), "").strip()
            return m or None

    def set_model(self, chat_id: int, model: str) -> None:
        m = (model or "").strip()
        if not m:
            return
        with self._lock:
            self._model_by_chat[int(chat_id)] = m

    def clear_model(self, chat_id: int) -> None:
        with self._lock:
            self._model_by_chat.pop(int(chat_id), None)


def _extract_thread_id_from_jsonl(text: str) -> str:
    """
    Extract Codex thread id from `codex exec --json` output.
    """
    tid = ""
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        t = obj.get("type")
        if t == "thread.started" and isinstance(obj.get("thread_id"), str):
            tid = obj["thread_id"].strip()
        elif isinstance(obj.get("thread_id"), str) and not tid:
            # Best-effort fallback for future event shapes.
            tid = obj["thread_id"].strip()
    return tid


def _extract_thread_id_from_jsonl_file(path: Path, *, max_bytes: int = 1_000_000) -> str:
    """
    Stream-parse JSONL from the beginning so we reliably catch `thread.started` even when output is large.
    """
    try:
        with path.open("rb") as f:
            read = 0
            while True:
                line_b = f.readline()
                if not line_b:
                    break
                read += len(line_b)
                if max_bytes > 0 and read > max_bytes:
                    break
                try:
                    line = line_b.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "thread.started" and isinstance(obj.get("thread_id"), str):
                    return obj["thread_id"].strip()
    except FileNotFoundError:
        return ""
    except Exception:
        LOG.exception("Failed to extract thread id from: %s", path)
        return ""
    return ""


def _skill_installer_scripts_dir() -> Path:
    # skill-installer is a system skill that ships with this deployment.
    return _skills_root_dir() / ".system" / "skill-installer" / "scripts"


def _run_internal_skills_job(
    *,
    cfg: BotConfig,
    api: TelegramAPI,
    tracker: JobTracker,
    stop_event: threading.Event,
    job: Job,
    eff_cfg: BotConfig,
) -> None:
    """
    Runs skill management actions in a worker (so the poll loop stays responsive).
    """
    argv = list(job.argv or [])
    sub = argv[1] if len(argv) > 1 else ""
    started = time.time()

    scripts = _skill_installer_scripts_dir()
    list_py = scripts / "list-skills.py"
    install_py = scripts / "install-skill-from-github.py"

    if sub == "catalog":
        raw = (argv[2] if len(argv) > 2 else "").strip()
        scope = "curated"
        flt = raw
        if raw.lower().startswith("experimental "):
            scope = "experimental"
            flt = raw[len("experimental ") :].strip()
        elif raw.lower().startswith("exp "):
            scope = "experimental"
            flt = raw[len("exp ") :].strip()
        elif raw.lower().startswith("curated "):
            scope = "curated"
            flt = raw[len("curated ") :].strip()

        if not list_py.exists():
            api.send_message(
                job.chat_id,
                f"No encontre list-skills.py en: {list_py}",
                reply_to_message_id=job.reply_to_message_id,
            )
            return

        repo_path = "skills/.curated" if scope == "curated" else "skills/.experimental"
        cmd = ["python3", str(list_py), "--format", "json", "--path", repo_path]
        api.send_message(
            job.chat_id,
            f"Listing skills ({scope})...",
            reply_to_message_id=job.reply_to_message_id,
        )
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        tracker.set_running_proc(job.chat_id, proc)
        last_typing = 0.0
        canceled = False
        while proc.poll() is None:
            if stop_event.is_set():
                _terminate_process(proc)
                canceled = True
                break
            if tracker.current_epoch(job.chat_id) != job.epoch:
                _terminate_process(proc)
                canceled = True
                break
            if eff_cfg.strict_proxy:
                now = time.time()
                if now - last_typing >= 4.0:
                    try:
                        api.send_chat_action(job.chat_id, "typing")
                    except Exception:
                        pass
                    last_typing = now
            time.sleep(0.25)

        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            out, err = "", ""
        code = int(proc.returncode or 0)
        if canceled:
            api.send_message(job.chat_id, "Canceled.", reply_to_message_id=job.reply_to_message_id)
            return

        if code != 0:
            msg = (err or out or "").strip() or "Unknown error."
            api.send_message(
                job.chat_id,
                f"Failed to list skills (exit={code}).\n\n{_tail_text(msg, max_chars=3500)}",
                reply_to_message_id=job.reply_to_message_id,
            )
            return

        try:
            payload = json.loads(out or "[]")
        except Exception:
            payload = []
        items: list[tuple[str, bool]] = []
        if isinstance(payload, list):
            for it in payload:
                if not isinstance(it, dict):
                    continue
                name = it.get("name")
                installed = it.get("installed")
                if isinstance(name, str):
                    items.append((name, bool(installed) if isinstance(installed, bool) else False))

        if flt:
            f = flt.lower()
            items = [(n, ins) for (n, ins) in items if f in n.lower()]

        if not items:
            api.send_message(
                job.chat_id,
                "No matches." if flt else "No skills found.",
                reply_to_message_id=job.reply_to_message_id,
            )
            return

        # Keep output small for Telegram.
        items = items[:60]
        lines = [f"Installable skills ({scope}):"]
        for idx, (n, ins) in enumerate(items, start=1):
            suffix = " (installed)" if ins else ""
            lines.append(f"{idx}. {n}{suffix}")
        if flt:
            lines.append("")
            lines.append(f"filter: {flt}")
        lines.append("")
        lines.append("Usage: /skills install <skill>")
        out_msg = "\n".join(lines)
        for idx, ch in enumerate(_chunk_text(out_msg, limit=TELEGRAM_MSG_LIMIT - 64), start=1):
            prefix = "" if idx == 1 and len(out_msg) <= (TELEGRAM_MSG_LIMIT - 64) else f"[{idx}]\n"
            api.send_message(job.chat_id, prefix + ch, reply_to_message_id=job.reply_to_message_id)
        return

    if sub == "install":
        scope = (argv[2] if len(argv) > 2 else "curated").strip().lower()
        name = (argv[3] if len(argv) > 3 else "").strip()
        if scope not in ("curated", "experimental"):
            scope = "curated"
        if not _skill_segment_ok(name):
            api.send_message(
                job.chat_id,
                "Nombre de skill invalido.",
                reply_to_message_id=job.reply_to_message_id,
            )
            return
        if not install_py.exists():
            api.send_message(
                job.chat_id,
                f"No encontre install-skill-from-github.py en: {install_py}",
                reply_to_message_id=job.reply_to_message_id,
            )
            return

        repo_path = ("skills/.curated/" if scope == "curated" else "skills/.experimental/") + name
        cmd = ["python3", str(install_py), "--repo", "openai/skills", "--path", repo_path]

        api.send_message(
            job.chat_id,
            f"Installing skill {name} ({scope})...",
            reply_to_message_id=job.reply_to_message_id,
        )
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        tracker.set_running_proc(job.chat_id, proc)
        last_typing = 0.0
        canceled = False
        while proc.poll() is None:
            if stop_event.is_set():
                _terminate_process(proc)
                canceled = True
                break
            if tracker.current_epoch(job.chat_id) != job.epoch:
                _terminate_process(proc)
                canceled = True
                break
            if eff_cfg.strict_proxy:
                now = time.time()
                if now - last_typing >= 4.0:
                    try:
                        api.send_chat_action(job.chat_id, "typing")
                    except Exception:
                        pass
                    last_typing = now
            time.sleep(0.25)

        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            out, err = "", ""
        code = int(proc.returncode or 0)
        secs = time.time() - started
        if canceled:
            api.send_message(job.chat_id, "Canceled.", reply_to_message_id=job.reply_to_message_id)
            return

        if code == 0:
            api.send_message(
                job.chat_id,
                f"OK. Installed {name} ({scope}). secs={secs:.1f}\n\nTip: /skills para ver estado.",
                reply_to_message_id=job.reply_to_message_id,
            )
            return

        detail = (err or out or "").strip() or "(no output)"
        api.send_message(
            job.chat_id,
            f"Install failed (exit={code}, secs={secs:.1f}).\n\n{_tail_text(detail, max_chars=3500)}",
            reply_to_message_id=job.reply_to_message_id,
        )
        return

    api.send_message(job.chat_id, "Unknown skills command.", reply_to_message_id=job.reply_to_message_id)


def _should_route_to_orchestrator(cfg: BotConfig, job: Job | None) -> bool:
    if not cfg.orchestrator_enabled or job is None:
        return False
    if not job.argv:
        return False
    if job.image_paths or job.upload_paths:
        return False
    cmd = job.argv[0]
    if not cmd:
        return False
    if cmd.startswith("__"):
        return False
    return True


def _submit_orchestrator_task(
    cfg: BotConfig,
    orch_q: OrchestratorQueue | None,
    profiles: dict[str, dict[str, Any]] | None,
    job: Job,
    *,
    user_id: int | None = None,
) -> tuple[bool, str]:
    if not cfg.orchestrator_enabled or orch_q is None:
        return False, ""
    if not _should_route_to_orchestrator(cfg, job):
        return False, ""
    try:
        task = _orchestrator_task_from_job(cfg, job, profiles=profiles, user_id=user_id)
        job_id = orch_q.submit_task(task)

        # Autopilot scope: any top-level Jarvis ticket (except pure queries) becomes an "active order".
        try:
            if (
                _coerce_orchestrator_role(task.role) == "jarvis"
                and not task.is_autonomous
                and not (task.parent_job_id or "").strip()
                and (task.request_type or "task") != "query"
            ):
                title = (task.input_text or "").strip().splitlines()[0].strip()
                if len(title) > 120:
                    title = title[:120] + "..."
                orch_q.upsert_order(
                    order_id=task.job_id,
                    chat_id=int(task.chat_id),
                    title=title or f"Order {task.job_id[:8]}",
                    body=(task.input_text or "").strip(),
                    status="active",
                    priority=int(task.priority or cfg.orchestrator_default_priority),
                )
        except Exception:
            pass

        return True, job_id
    except Exception as e:
        LOG.exception("Failed to submit orchestrator task")
        raise RuntimeError(f"Failed to submit orchestrator task: {e}") from e


def _orchestrator_apply_task_flags(task: Task, argv: list[str]) -> list[str]:
    args = list(argv)
    if task.model:
        model = _sanitize_model_id(task.model)
        if model and "--model" not in args and "-m" not in args:
            args[1:1] = ["--model", model]
    if task.effort:
        effort = _sanitize_effort(task.effort)
        if effort and not _extract_effort_override_from_argv(args):
            args[1:1] = ["-c", f"model_reasoning_effort=\"{effort}\""]
    return args


def _orchestrator_run_codex(
    cfg: BotConfig,
    task: Task,
    *,
    stop_event: threading.Event,
    orch_q: OrchestratorQueue | None,
    profiles: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    Execute a single orchestrator task using Codex.

    Changes vs legacy behavior (grounded in this repository's "Jarvis v1" plan):
    - Apply role profiles via `build_agent_prompt(...)` (system_prompt becomes real agent behavior).
    - Optional per-(chat, role) Codex sessions via `codex exec resume` (memory per role).
    - Optional git worktree pool per role/slot for isolation.
    - Optional Playwright screenshot capture for frontend snapshot tasks.
    - Always emit git diff/status artifacts when running in a worktree.
    """
    started = time.time()
    role = _coerce_orchestrator_role(task.role)
    profile = _orchestrator_profile(profiles, role)

    mode = _coerce_orchestrator_mode(task.mode_hint)
    timeout_seconds = cfg.codex_timeout_seconds
    try:
        profile_timeout = int(task.trace.get("max_runtime_seconds", 0) or 0)
    except Exception:
        profile_timeout = 0
    if profile_timeout > 0:
        timeout_seconds = min(timeout_seconds, profile_timeout) if timeout_seconds > 0 else profile_timeout

    artifacts_dir = Path((task.artifacts_dir or str(cfg.artifacts_root / task.job_id))).expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Fast-path: status requests should not spend a Codex run. Return an in-bot view of the orchestrator.
    if task.request_type == "status" and orch_q is not None:
        try:
            running = orch_q.jobs_by_state(state="running", limit=6)
        except Exception:
            running = []
        lines: list[str] = []
        if not running:
            lines.append("No hay jobs corriendo ahora.")
        else:
            lines.append(f"Jobs corriendo: {len(running)}")
            for t in running[:6]:
                phase = str((t.trace or {}).get("live_phase") or "").strip() or "running"
                snippet = (t.input_text or "").strip().replace("\n", " ")[:110]
                role_h = _humanize_orchestrator_role(t.role)
                lines.append(f"- {role_h} ({phase}): {snippet}")
        return {
            "status": "ok",
            "summary": "\n".join(lines),
            "artifacts": [],
            "logs": "",
            "next_action": None,
            "structured_digest": {"role": role},
        }

    # Snapshot-only tasks: do bot-side capture and finish without running Codex.
    needs_shot = bool(task.trace.get("needs_screenshot", False))
    screenshot_only = bool(task.trace.get("screenshot_only", False))
    if needs_shot and screenshot_only:
        url = str(task.trace.get("screenshot_url") or "").strip()
        if not url:
            return {
                "status": "error",
                "summary": "Snapshot requested but no URL/target provided.",
                "artifacts": [],
                "logs": "",
                "next_action": None,
                "structured_digest": {"role": role, "artifacts_dir": str(artifacts_dir)},
            }
        if not cfg.screenshot_enabled:
            return {
                "status": "error",
                "summary": "Screenshots are disabled. Set BOT_SCREENSHOT_ENABLED=1 and install Playwright.",
                "artifacts": [],
                "logs": "",
                "next_action": "enable_screenshots",
                "structured_digest": {"role": role, "artifacts_dir": str(artifacts_dir)},
            }
        approved = bool(task.trace.get("approved", False))
        check = validate_screenshot_url(
            url,
            allowed_hosts=cfg.screenshot_allowed_hosts,
            allow_private=approved,
        )
        if not check.ok:
            if check.overrideable and not approved:
                return {
                    "status": "blocked",
                    "summary": f"Snapshot blocked by guardrails: {check.reason}\nApprove with: /approve {task.job_id}",
                    "artifacts": [],
                    "logs": "",
                    "next_action": "approve_screenshot",
                    "structured_digest": {"role": role, "artifacts_dir": str(artifacts_dir)},
                }
            return {
                "status": "error",
                "summary": f"Snapshot blocked: {check.reason}",
                "artifacts": [],
                "logs": "",
                "next_action": None,
                "structured_digest": {"role": role, "artifacts_dir": str(artifacts_dir)},
            }
        url = check.normalized_url
        out_png = artifacts_dir / "snapshot.png"
        if orch_q is not None:
            try:
                orch_q.update_trace(task.job_id, live_phase="snapshot", live_at=time.time())
            except Exception:
                pass
        try:
            capture_screenshot(
                url,
                out_png,
                viewport=Viewport(width=1280, height=720),
                allowed_hosts=cfg.screenshot_allowed_hosts,
                allow_private=approved,
            )
        except Exception as e:
            return {
                "status": "error",
                "summary": f"Screenshot failed: {e}",
                "artifacts": [],
                "logs": str(e),
                "next_action": "install_playwright",
                "structured_digest": {"role": role, "artifacts_dir": str(artifacts_dir)},
            }
        return {
            "status": "ok",
            "summary": f"Snapshot captured: {url}",
            "artifacts": [str(out_png)],
            "logs": "",
            "next_action": None,
            "structured_digest": {"role": role, "artifacts_dir": str(artifacts_dir)},
        }

    # Worktree isolation (best-effort). If configured incorrectly, fail safe (no writes) by falling back.
    eff_cfg = cfg
    worktree_dir: Path | None = None
    leased_slot: int | None = None
    lease_enabled = orch_q is not None and (cfg.codex_workdir / ".git").exists()
    try:
        slots = max(1, int(profile.get("max_parallel_jobs") or 1))
    except Exception:
        slots = 1

    if lease_enabled:
        try:
            leased_slot = orch_q.lease_workspace(role=role, job_id=task.job_id, slots=slots)
            if leased_slot is None:
                return {
                    "status": "error",
                    "summary": "No workspace slot available for this role; try again.",
                    "artifacts": [],
                    "logs": f"role={role} slots={slots}",
                    "next_action": "retry",
                    "structured_digest": {"role": role, "workspace": "unavailable"},
                }
            ensure_worktree_pool(base_repo=cfg.codex_workdir, root=cfg.worktree_root, role=role, slots=slots)
            worktree_dir = (cfg.worktree_root / role / f"slot{leased_slot}").resolve()
            prepare_clean_workspace(worktree_dir)
            eff_cfg = dataclasses.replace(cfg, codex_workdir=worktree_dir)
            try:
                if orch_q is not None:
                    orch_q.update_trace(
                        task.job_id,
                        live_phase="workspace_ready",
                        live_workdir=str(worktree_dir),
                        live_workspace_slot=int(leased_slot),
                        live_at=time.time(),
                    )
            except Exception:
                pass
        except Exception as e:
            # Release the lease and fall back to the base repo in read-only mode.
            try:
                if orch_q is not None:
                    orch_q.release_workspace(job_id=task.job_id)
            except Exception:
                pass
            leased_slot = None
            worktree_dir = None
            eff_cfg = cfg
            mode = "ro"
            LOG.exception("Worktree setup failed; falling back to base workdir read-only. job=%s role=%s", task.job_id, role)
            artifacts_f = artifacts_dir / "worktree_error.txt"
            artifacts_f.write_text(str(e) + "\n", encoding="utf-8", errors="replace")
            try:
                if orch_q is not None:
                    orch_q.update_trace(task.job_id, live_phase="workspace_fallback_ro", live_at=time.time())
            except Exception:
                pass

    # Optional screenshot capture (Playwright).
    image_paths: list[Path] = []
    needs_shot = bool(task.trace.get("needs_screenshot", False))
    if needs_shot:
        url = str(task.trace.get("screenshot_url") or "").strip()
        if not url:
            try:
                if orch_q is not None and leased_slot is not None:
                    orch_q.release_workspace(job_id=task.job_id)
            except Exception:
                LOG.exception("Failed to release workspace lease (snapshot preflight). job=%s role=%s", task.job_id, role)
            return {
                "status": "error",
                "summary": "Snapshot requested but no URL/target provided.",
                "artifacts": [],
                "logs": "",
                "next_action": None,
            }
        if not cfg.screenshot_enabled:
            try:
                if orch_q is not None and leased_slot is not None:
                    orch_q.release_workspace(job_id=task.job_id)
            except Exception:
                LOG.exception("Failed to release workspace lease (snapshot disabled). job=%s role=%s", task.job_id, role)
            return {
                "status": "error",
                "summary": "Screenshots are disabled. Set BOT_SCREENSHOT_ENABLED=1 and install Playwright.",
                "artifacts": [],
                "logs": "",
                "next_action": "enable_screenshots",
            }
        approved = bool(task.trace.get("approved", False))
        check = validate_screenshot_url(
            url,
            allowed_hosts=cfg.screenshot_allowed_hosts,
            allow_private=approved,
        )
        if not check.ok:
            try:
                if orch_q is not None and leased_slot is not None:
                    orch_q.release_workspace(job_id=task.job_id)
            except Exception:
                LOG.exception("Failed to release workspace lease (snapshot blocked). job=%s role=%s", task.job_id, role)
            if check.overrideable and not approved:
                return {
                    "status": "blocked",
                    "summary": f"Snapshot blocked by guardrails: {check.reason}\nApprove with: /approve {task.job_id}",
                    "artifacts": [],
                    "logs": "",
                    "next_action": "approve_screenshot",
                }
            return {
                "status": "error",
                "summary": f"Snapshot blocked: {check.reason}",
                "artifacts": [],
                "logs": "",
                "next_action": None,
            }
        url = check.normalized_url
        out_png = artifacts_dir / "snapshot.png"
        try:
            capture_screenshot(
                url,
                out_png,
                viewport=Viewport(width=1280, height=720),
                allowed_hosts=cfg.screenshot_allowed_hosts,
                allow_private=approved,
            )
            image_paths.append(out_png)
        except Exception as e:
            try:
                if orch_q is not None and leased_slot is not None:
                    orch_q.release_workspace(job_id=task.job_id)
            except Exception:
                LOG.exception("Failed to release workspace lease (snapshot failure). job=%s role=%s", task.job_id, role)
            return {
                "status": "error",
                "summary": f"Screenshot failed: {e}",
                "artifacts": [],
                "logs": str(e),
                "next_action": "install_playwright",
            }

    # Wrap-up jobs: inject child results into the USER_REQUEST so the orchestrator can synthesize a final brief.
    wrapup_for = str(task.trace.get("wrapup_for") or "").strip()
    if wrapup_for and orch_q is not None:
        try:
            items = orch_q.jobs_by_parent(parent_job_id=wrapup_for, limit=200)
            lines: list[str] = []
            for it in items:
                if it.job_id == task.job_id:
                    continue
                res_summary = str(it.trace.get("result_summary") or "").strip().replace("\n", " ")
                if len(res_summary) > 260:
                    res_summary = res_summary[:260] + "..."
                lines.append(f"- {it.job_id[:8]} role={it.role} state={it.state} result={res_summary or '(no result)'}")
            if lines:
                extra = "SUBTASK_RESULTS:\n" + "\n".join(lines)
                task = task.with_updates(input_text=(task.input_text or "").rstrip() + "\n\n" + extra)
        except Exception:
            LOG.exception("Failed to build wrap-up context. job=%s wrapup_for=%s", task.job_id, wrapup_for)

    # Autopilot jobs: inject order + current queue context so Jarvis can decide next best work.
    if bool(task.trace.get("autopilot", False)) and orch_q is not None:
        try:
            oid = str(task.trace.get("order_id") or task.parent_job_id or "").strip()
            order = orch_q.get_order(oid, chat_id=int(task.chat_id)) if oid else None
            children = orch_q.jobs_by_parent(parent_job_id=oid, limit=120) if oid else []
            health = orch_q.get_role_health() or {}
            running = orch_q.jobs_by_state(state="running", limit=8)

            ctx_lines: list[str] = ["AUTOPILOT_CONTEXT", "================"]
            if order:
                ctx_lines.append(f"order_status={order.get('status')}")
                ctx_lines.append(f"order_priority={order.get('priority')}")
            if children:
                counts: dict[str, int] = {}
                for c in children:
                    counts[c.state] = counts.get(c.state, 0) + 1
                ctx_lines.append("order_children_counts=" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
                # Show a few most recent failures/blocked items as unblock targets.
                blockers = [c for c in children if c.state in ("blocked", "failed")][:8]
                if blockers:
                    ctx_lines.append("blockers:")
                    for c in blockers[:8]:
                        res = str((c.trace or {}).get("result_summary") or "").strip().replace("\n", " ")
                        if len(res) > 160:
                            res = res[:160] + "..."
                        ctx_lines.append(f"- {c.job_id[:8]} role={c.role} state={c.state} result={res or '(no result)'}")
            if health:
                ctx_lines.append("role_health:")
                for r in sorted(health.keys()):
                    vals = health.get(r, {}) or {}
                    ctx_lines.append(
                        f"- {r}: queued={int(vals.get('queued',0) or 0)} running={int(vals.get('running',0) or 0)} blocked={int(vals.get('blocked',0) or 0)}"
                    )
            if running:
                ctx_lines.append("running_jobs:")
                for t in running[:8]:
                    tr = t.trace or {}
                    phase = str(tr.get("live_phase") or "").strip() or "running"
                    snippet = (t.input_text or "").strip().replace("\n", " ")[:120]
                    ctx_lines.append(f"- {t.job_id[:8]} role={t.role} phase={phase} text={snippet}")

            extra = "\n".join(ctx_lines)
            task = task.with_updates(input_text=(task.input_text or "").rstrip() + "\n\n" + extra)
        except Exception:
            LOG.exception("Failed to build autopilot context. job=%s", task.job_id)

    prompt = build_agent_prompt(task, profile=profile)
    try:
        # Safe logging: lengths only (never log prompt contents).
        LOG.debug(
            "Built codex prompt. job=%s role=%s input_chars=%s prompt_chars=%s",
            task.job_id[:8],
            role,
            len(task.input_text or ""),
            len(prompt or ""),
        )
    except Exception:
        pass

    # Execution: optionally keep per-(chat, role) memory via `codex exec resume`.
    runner = CodexRunner(eff_cfg, chat_id=task.chat_id)
    proc: CodexRunner.Running
    used_thread_id: str | None = None
    started_new_thread = False
    try:
        if cfg.orchestrator_sessions_enabled and orch_q is not None:
            tid = orch_q.get_agent_thread(chat_id=task.chat_id, role=role) or ""
            if tid:
                used_thread_id = tid
                proc = runner.start_threaded_resume(
                    thread_id=tid,
                    prompt=prompt,
                    mode_hint=mode,
                    image_paths=image_paths or None,
                    model_override=task.model or None,
                    effort_override=task.effort or None,
                )
            else:
                started_new_thread = True
                proc = runner.start_threaded_new(
                    prompt=prompt,
                    mode_hint=mode,
                    image_paths=image_paths or None,
                    model_override=task.model or None,
                    effort_override=task.effort or None,
                )
        else:
            argv = ["exec"]
            for p in image_paths:
                argv += ["--image", str(p)]
            argv.append(prompt)
            argv = _orchestrator_apply_task_flags(task, argv)
            proc = runner.start(argv=argv, mode_hint=mode)
    except Exception as e:
        try:
            if orch_q is not None and leased_slot is not None:
                orch_q.release_workspace(job_id=task.job_id)
        except Exception:
            LOG.exception("Failed to release workspace lease (codex start error). job=%s role=%s", task.job_id, role)
        return {
            "status": "error",
            "summary": f"Failed to start codex for job={task.job_id}",
            "artifacts": [],
            "logs": str(e),
            "next_action": None,
        }

    timed_out = False
    canceled = False
    last_live_update = 0.0
    try:
        try:
            if orch_q is not None:
                orch_q.update_trace(
                    task.job_id,
                    live_phase="codex_start",
                    live_pid=int(getattr(proc.proc, "pid", 0) or 0) or None,
                    live_workdir=str(eff_cfg.codex_workdir),
                    live_workspace_slot=int(leased_slot) if leased_slot is not None else None,
                    live_at=time.time(),
                )
        except Exception:
            pass
        while proc.proc.poll() is None:
            if stop_event.is_set():
                _terminate_process(proc.proc)
                canceled = True
                break
            if orch_q is not None and _poll_orchestrator_job_state(orch_q, task.job_id) == "cancelled":
                _terminate_process(proc.proc)
                canceled = True
                break
            if timeout_seconds > 0 and (time.time() - proc.start_time) >= timeout_seconds:
                _terminate_process(proc.proc)
                timed_out = True
                break

            # Live progress for CEO visibility: update stdout/stderr tails periodically.
            if orch_q is not None and cfg.orchestrator_live_update_seconds > 0:
                now = time.time()
                if now - last_live_update >= float(cfg.orchestrator_live_update_seconds):
                    try:
                        stdout_tail = _strip_ansi(_tail_file_text(proc.stdout_path, max_chars=1200)).strip()
                    except Exception:
                        stdout_tail = ""
                    try:
                        stderr_tail = _strip_ansi(_tail_file_text(proc.stderr_path, max_chars=1200)).strip()
                    except Exception:
                        stderr_tail = ""
                    try:
                        orch_q.update_trace(
                            task.job_id,
                            live_phase="running",
                            live_pid=int(getattr(proc.proc, "pid", 0) or 0) or None,
                            live_workdir=str(eff_cfg.codex_workdir),
                            live_workspace_slot=int(leased_slot) if leased_slot is not None else None,
                            live_stdout_tail=stdout_tail,
                            live_stderr_tail=stderr_tail,
                            live_at=now,
                        )
                    except Exception:
                        pass
                    last_live_update = now
            time.sleep(0.25)

        try:
            proc.proc.wait(timeout=5)
        except Exception:
            pass

        if canceled:
            return {
                "status": "cancelled",
                "summary": "Task canceled by operator.",
                "artifacts": [],
                "logs": "",
                "next_action": None,
            }
        if timed_out:
            return {
                "status": "error",
                "summary": f"Task timed out after {timeout_seconds}s.",
                "artifacts": [],
                "logs": _tail_file_text(proc.stderr_path, max_chars=6000),
                "next_action": None,
            }

        code = int(proc.proc.returncode) if proc.proc.returncode is not None else 1

        body = ""
        if proc.last_msg_path is not None:
            body = _read_text_file(proc.last_msg_path).strip()
        if not body:
            # If stdout is small-ish, read it; otherwise show a tail.
            try:
                sz = proc.stdout_path.stat().st_size
            except OSError:
                sz = 0
            if sz <= 256_000:
                body = _strip_ansi(_read_text_file(proc.stdout_path)).strip()
            else:
                body = _tail_file_text(proc.stdout_path, max_chars=6000).strip()
        if not body:
            body = "(no output)"

        # If we started a new session, extract and persist the thread id.
        if started_new_thread and orch_q is not None:
            try:
                tid = _extract_thread_id_from_jsonl_file(proc.stdout_path)
                if tid:
                    used_thread_id = tid
                    orch_q.set_agent_thread(chat_id=task.chat_id, role=role, thread_id=tid)
            except Exception:
                LOG.exception("Failed to extract/persist orchestrator thread_id. job=%s role=%s", task.job_id, role)

        logs = _tail_file_text(proc.stderr_path, max_chars=6000)

        artifacts: list[Path] = []
        # Screenshot output is outside workdir; include explicitly.
        for p in image_paths:
            if p.exists():
                artifacts.append(p)
        # Collect PNGs created in the Codex workdir.
        artifacts.extend(_collect_png_artifacts(eff_cfg, start_time=proc.start_time, text=body))
        # Collect git diff/status whenever we ran inside a managed worktree.
        if worktree_dir is not None:
            try:
                artifacts.extend(collect_git_artifacts(repo_dir=worktree_dir, artifacts_dir=artifacts_dir))
            except Exception:
                LOG.exception("Failed to collect git artifacts. job=%s role=%s", task.job_id, role)

        # Frontend evidence: allow the agent to drop a self-contained preview HTML inside the workspace.
        # The bot screenshots it (headless) and sends `preview.png` to Telegram as an artifact.
        if role == "frontend" and cfg.screenshot_enabled:
            try:
                preview_html = Path(eff_cfg.codex_workdir) / ".codexbot_preview" / "preview.html"
                if preview_html.exists() and preview_html.is_file():
                    dest_html = artifacts_dir / "preview.html"
                    shutil.copyfile(preview_html, dest_html)
                    out_png = artifacts_dir / "preview.png"
                    capture_screenshot_html_file(
                        dest_html,
                        out_png,
                        viewport=Viewport(width=1280, height=720),
                        allowed_hosts=cfg.screenshot_allowed_hosts,
                        allow_private=False,
                        block_network=True,
                    )
                    artifacts.append(dest_html)
                    artifacts.append(out_png)
            except Exception as e:
                LOG.exception("Failed to generate frontend preview screenshot. job=%s", task.job_id)
                try:
                    err_f = artifacts_dir / "preview_screenshot_error.txt"
                    err_f.write_text(str(e) + "\n", encoding="utf-8", errors="replace")
                    artifacts.append(err_f)
                except Exception:
                    pass

        artifacts_text = [str(p) for p in artifacts]
        structured: dict[str, Any] = {
            "role": role,
            "workdir": str(eff_cfg.codex_workdir),
            "artifacts_dir": str(artifacts_dir),
        }
        if leased_slot is not None:
            structured["workspace_slot"] = int(leased_slot)
        if used_thread_id:
            structured["thread_id"] = used_thread_id

        if code == 0:
            return {
                "status": "ok",
                "summary": body,
                "artifacts": artifacts_text,
                "logs": logs,
                "next_action": None,
                "structured_digest": structured,
            }
        return {
            "status": "error",
            "summary": f"Codex returned code {code}.",
            "artifacts": artifacts_text,
            "logs": logs or body,
            "next_action": None,
            "structured_digest": structured,
        }
    finally:
        elapsed = time.time() - started
        try:
            if proc.last_msg_path is not None:
                proc.last_msg_path.unlink(missing_ok=True)
            proc.stdout_path.unlink(missing_ok=True)
            proc.stderr_path.unlink(missing_ok=True)
        except Exception:
            LOG.exception("Failed to cleanup orchestrator proc temp files")
        try:
            if orch_q is not None and leased_slot is not None:
                orch_q.release_workspace(job_id=task.job_id)
        except Exception:
            LOG.exception("Failed to release workspace lease. job=%s role=%s", task.job_id, role)
        LOG.info("Orchestrator task %s finished in %.2fs", task.job_id, elapsed)


def _send_orchestrator_result(
    api: TelegramAPI,
    task: Task,
    result: Any,
    *,
    cfg: BotConfig,
) -> None:
    try:
        if isinstance(result, dict):
            status = str(result.get("status", "error"))
            summary = str(result.get("summary", "")).strip() or "(no summary)"
            logs = str(result.get("logs", ""))
            next_action = result.get("next_action", None)
            artifacts = list(result.get("artifacts", []) or [])
        else:
            status = str(getattr(result, "status", "error"))
            summary = str(getattr(result, "summary", "")).strip() or "(no summary)"
            logs = str(getattr(result, "logs", ""))
            next_action = getattr(result, "next_action", None)
            artifacts = list(getattr(result, "artifacts", []) or [])

        labels = task.labels or {}
        kind = str(labels.get("kind") or "").strip().lower()
        mode = (cfg.orchestrator_notify_mode or "minimal").strip().lower()

        def _should_notify() -> bool:
            if mode == "verbose":
                return True
            if kind == "wrapup":
                return True
            # Ticket-card UX: if a top-level ticket has an editable card message, do not send a separate
            # "job=... status=ok" message on success. The card is updated in-place by the worker loop.
            try:
                if (
                    status == "ok"
                    and not (task.parent_job_id or "").strip()
                    and str((task.trace or {}).get("ticket_card_message_id") or "").strip()
                ):
                    return False
            except Exception:
                pass
            # Autonomous runbooks: only notify on non-ok outcomes.
            if bool(task.is_autonomous):
                return status != "ok"
            # Subtasks: only notify on non-ok outcomes (details are visible via /ticket and /job).
            if (task.parent_job_id or "").strip():
                return status != "ok"
            return True

        # Evidence jobs: on success, send only the artifact(s) and skip the text message.
        artifacts_only = (kind == "evidence") and status == "ok" and mode != "verbose"

        prefer_voice = bool((task.trace or {}).get("prefer_voice_reply", False))
        force_notify = prefer_voice and bool(cfg.voice_out_enabled)
        notify = _should_notify() or force_notify

        if notify:
            # Keep chat updates short; details are always available via /job and /ticket.
            max_summary = 900
            if len(summary) > max_summary:
                summary = summary[:max_summary] + "...\n(details: /job %s)" % task.job_id[:8]

            role_name = _humanize_orchestrator_role(task.role)
            summary_lines = (summary or "").splitlines() or ["(no summary)"]
            summary_lines[0] = f"{role_name}: {summary_lines[0]}".strip()
            payload = list(summary_lines)
            if next_action:
                payload.append(f"Next: {next_action}")

            # If there's no ticket card to click, provide /job for errors only.
            try:
                has_ticket_card = (
                    (not (task.parent_job_id or "").strip())
                    and bool(str((task.trace or {}).get("ticket_card_message_id") or "").strip())
                )
            except Exception:
                has_ticket_card = False
            if status != "ok" and (not has_ticket_card):
                payload.append(f"Details: /job {task.job_id[:8]}")

            # Wrap-ups: include a short pointer back to the ticket tree.
            if kind == "wrapup":
                root = (task.parent_job_id or "").strip()
                if root:
                    payload.append(f"Ticket: /ticket {root[:8]}")

            msg = "\n".join(payload)
            sent_voice = False
            if prefer_voice and cfg.voice_out_enabled:
                try:
                    tts_client: OpenAITTS | None = None
                    if cfg.openai_api_key and (cfg.tts_backend or "").strip().lower() == "openai":
                        tts_client = OpenAITTS(
                            api_key=cfg.openai_api_key,
                            api_base_url=cfg.openai_api_base_url,
                            timeout_seconds=cfg.http_timeout_seconds,
                            max_retries=cfg.http_max_retries,
                            retry_initial_seconds=cfg.http_retry_initial_seconds,
                            retry_max_seconds=cfg.http_retry_max_seconds,
                        )
                    speak_text = (summary_lines[0] if summary_lines else summary) or "(no summary)"
                    _send_voice_note(
                        api=api,
                        cfg=cfg,
                        chat_id=int(task.chat_id),
                        reply_to_message_id=int(task.reply_to_message_id or 0),
                        tts=tts_client,
                        speak_text=str(speak_text),
                        caption_text=msg,
                    )
                    sent_voice = True
                except Exception:
                    sent_voice = False
                    LOG.exception("Failed to send orchestrator voice reply. job=%s", task.job_id)

            if (not sent_voice) and (not artifacts_only):
                _send_chunked_text(api, chat_id=task.chat_id, text=msg, reply_to_message_id=task.reply_to_message_id)

            # In minimal mode, don't spam stderr tails. Operators can inspect via /job and artifacts.
            if mode == "verbose" and status != "ok" and logs:
                log_chunks = _chunk_text(logs, limit=TELEGRAM_MSG_LIMIT - 64)
                for idx, ch in enumerate(log_chunks, start=1):
                    prefix = f"log[{idx}/{len(log_chunks)}]\n"
                    try:
                        api.send_message(task.chat_id, f"{prefix}{ch}", reply_to_message_id=task.reply_to_message_id)
                    except Exception:
                        LOG.exception("Failed to send orchestrator logs chunk. job=%s", task.job_id)

        if (not notify) and (not artifacts_only):
            return

        for raw in artifacts[:3]:
            p = Path(str(raw))
            try:
                if not p.exists() or p.is_dir():
                    continue
                if p.stat().st_size <= 0:
                    continue
            except OSError:
                continue

            ext = p.suffix.lower()
            is_img = ext in (".png", ".jpg", ".jpeg", ".webp")
            if is_img:
                try:
                    api.send_photo(task.chat_id, p, caption=p.name, reply_to_message_id=task.reply_to_message_id)
                    continue
                except Exception:
                    LOG.exception("Failed to send orchestrator image artifact. job=%s file=%s", task.job_id, p)

            try:
                api.send_document(task.chat_id, p, filename=p.name, reply_to_message_id=task.reply_to_message_id)
            except Exception:
                LOG.exception("Failed to send orchestrator artifact. job=%s file=%s", task.job_id, p)
    except Exception:
        LOG.exception("Failed to send orchestrator result. job=%s", task.job_id)


def _poll_orchestrator_job_state(orch_q: OrchestratorQueue | None, job_id: str) -> str:
    if not orch_q:
        return ""
    try:
        t = orch_q.get_job(job_id)
    except Exception:
        return ""
    if t is None:
        return ""
    return t.state


class _OrchestratorExecutor:
    def __init__(
        self,
        cfg: BotConfig,
        stop_event: threading.Event,
        orch_q: OrchestratorQueue | None,
        *,
        profiles: dict[str, dict[str, Any]] | None,
    ) -> None:
        self._cfg = cfg
        self._stop_event = stop_event
        self._orch_q = orch_q
        self._profiles = profiles

    def run_task(self, task: Task) -> dict[str, Any]:
        return _orchestrator_run_codex(
            self._cfg,
            task,
            stop_event=self._stop_event,
            orch_q=self._orch_q,
            profiles=self._profiles,
        )


def orchestrator_worker_loop(
    *,
    cfg: BotConfig,
    api: TelegramAPI,
    orch_q: OrchestratorQueue,
    stop_event: threading.Event,
    profiles: dict[str, dict[str, Any]] | None,
) -> None:
    executor = _OrchestratorExecutor(cfg=cfg, stop_event=stop_event, orch_q=orch_q, profiles=profiles)

    while not stop_event.is_set():
        task = orch_q.take_next()
        if task is None:
            stop_event.wait(0.25)
            continue

        if profiles and task.role in profiles:
            trace = dict(task.trace)
            trace["profile"] = task.role
            task = task.with_updates(trace=trace)
        try:
            started = time.time()
            orch_q.update_state(task.job_id, "running")
            # Update the single ticket card message (no spam) when work starts.
            try:
                root_ticket = (task.parent_job_id or task.job_id or "").strip() or task.job_id
                _maybe_update_ticket_card(cfg=cfg, api=api, orch_q=orch_q, ticket_id=root_ticket)
            except Exception:
                pass
            result = run_orchestrator_task(task, executor=executor, cfg=cfg)

            raw_status = str(getattr(result, "status", "") or "")
            orch_state = raw_status or "failed"
            if orch_state not in {"ok", "blocked", "done", "failed", "cancelled"}:
                orch_state = "failed"
            if orch_state == "ok":
                orch_state = "done"

            # Persist a minimal structured result into trace so /job and /ticket have "memory".
            summary = str(getattr(result, "summary", "") or "").strip()
            if len(summary) > 4000:
                summary = summary[:4000] + "..."
            artifacts = list(getattr(result, "artifacts", []) or [])
            artifacts = [str(a) for a in artifacts if str(a).strip()][:20]
            next_action = getattr(result, "next_action", None)
            if next_action is not None:
                next_action = str(next_action).strip() or None

            structured_digest: Any = getattr(result, "structured_digest", None)
            if structured_digest is None and isinstance(result, dict):
                structured_digest = result.get("structured_digest")

            duration_s = max(0.0, float(time.time() - started))
            result_meta: dict[str, Any] = {
                "result_status": raw_status or orch_state,
                "result_summary": summary,
                "result_artifacts": artifacts,
                "result_next_action": next_action,
                "result_duration_s": duration_s,
            }
            if isinstance(structured_digest, dict):
                tid = structured_digest.get("thread_id")
                if isinstance(tid, str) and tid.strip():
                    result_meta["result_thread_id"] = tid.strip()
                slot = structured_digest.get("workspace_slot")
                try:
                    if slot is not None:
                        result_meta["result_workspace_slot"] = int(slot)
                except Exception:
                    pass

            # Retry/backoff: if task fails and retries remain, requeue with due_at.
                if orch_state == "failed" and int(task.retry_count or 0) < int(task.max_retries or 0):
                    retry_n = int(task.retry_count or 0) + 1
                    base = 30.0
                    delay = min(15 * 60.0, base * (2.0 ** max(0, retry_n - 1)))
                    due_at = time.time() + delay
                    err_for_trace = summary or str(getattr(result, "logs", "") or "").strip()
                    scheduled = orch_q.bump_retry(task.job_id, due_at=due_at, error=err_for_trace)
                    if scheduled:
                        try:
                            orch_q.update_trace(
                                task.job_id,
                                retry_scheduled_at=time.time(),
                                retry_due_at=float(due_at),
                                retry_delay_s=float(delay),
                            )
                        except Exception:
                            pass
                        try:
                            root_ticket = (task.parent_job_id or task.job_id or "").strip() or task.job_id
                            _maybe_update_ticket_card(cfg=cfg, api=api, orch_q=orch_q, ticket_id=root_ticket)
                        except Exception:
                            pass
                        # In minimal mode, avoid extra spam; the ticket card reflects the state.
                        if (cfg.orchestrator_notify_mode or "minimal").strip().lower() == "verbose":
                            api.send_message(
                                task.chat_id,
                                f"Retry scheduled: job={task.job_id[:8]} retry={retry_n}/{int(task.max_retries or 0)} in {int(delay)}s",
                                reply_to_message_id=task.reply_to_message_id,
                            )
                        continue

            orch_q.update_state(task.job_id, orch_state, **result_meta)
            # Update ticket card after the state transition so it reflects latest progress/result.
            try:
                root_ticket = (task.parent_job_id or task.job_id or "").strip() or task.job_id
                _maybe_update_ticket_card(cfg=cfg, api=api, orch_q=orch_q, ticket_id=root_ticket)
            except Exception:
                pass
            _send_orchestrator_result(api, task, result, cfg=cfg)

            # Frontend evidence: if the frontend agent outputs a snapshot_url in its structured JSON,
            # queue a snapshot-only job so Telegram receives a real screenshot.
            try:
                if (
                    orch_state == "done"
                    and _coerce_orchestrator_role(task.role) == "frontend"
                    and isinstance(structured_digest, dict)
                    and not bool(task.trace.get("needs_screenshot", False))
                ):
                    snap_url = str(structured_digest.get("snapshot_url") or "").strip()
                    if snap_url:
                        root_ticket = task.parent_job_id or task.job_id
                        shot_id = str(uuid.uuid4())
                        shot = Task.new(
                            source="telegram",
                            role="frontend",
                            input_text=f"Snapshot evidence for job={task.job_id}: {snap_url}",
                            request_type="maintenance",
                            priority=int(task.priority or 2),
                            model=str(task.model or ""),
                            effort=str(task.effort or "medium"),
                            mode_hint="ro",
                            requires_approval=False,
                            max_cost_window_usd=float(task.max_cost_window_usd or cfg.orchestrator_default_max_cost_window_usd),
                            chat_id=int(task.chat_id),
                            user_id=task.user_id,
                            reply_to_message_id=task.reply_to_message_id,
                            parent_job_id=root_ticket,
                            depends_on=[],
                            labels={"ticket": root_ticket, "kind": "evidence", "for": task.job_id},
                            artifacts_dir=str((cfg.artifacts_root / shot_id).resolve()),
                            trace={
                                "source": "telegram",
                                "needs_screenshot": True,
                                "screenshot_url": snap_url,
                                "screenshot_only": True,
                                "delegated_by": task.job_id,
                            },
                            job_id=shot_id,
                        )
                        orch_q.submit_task(shot)
                        try:
                            orch_q.append_delegation_edge(
                                root_ticket_id=root_ticket,
                                from_job_id=task.job_id,
                                to_job_id=shot.job_id,
                                edge_type="evidence",
                                to_role=shot.role,
                                to_key=None,
                                details={"snapshot_url": snap_url},
                            )
                        except Exception:
                            pass
                        if (cfg.orchestrator_notify_mode or "minimal").strip().lower() == "verbose":
                            api.send_message(
                                task.chat_id,
                                f"Snapshot queued: {shot.job_id[:8]} for job={task.job_id[:8]}",
                                reply_to_message_id=task.reply_to_message_id,
                            )
            except Exception:
                LOG.exception("Failed to enqueue frontend snapshot evidence job. job=%s", task.job_id)

            # Jarvis delegation: enqueue child jobs when Jarvis outputs structured subtasks.
            try:
                allow_delegation = bool(task.trace.get("allow_delegation", False))
                is_jarvis = _coerce_orchestrator_role(task.role) == "jarvis"
                is_query = (task.request_type or "task") == "query"
                is_top_level_manual = (not task.is_autonomous) and (not task.parent_job_id) and ((task.request_type or "task") == "task")

                if orch_state == "done" and is_jarvis and (not is_query) and (is_top_level_manual or allow_delegation):
                    root_ticket = (task.parent_job_id or task.job_id or "").strip() or task.job_id
                    existing = orch_q.jobs_by_parent(parent_job_id=root_ticket, limit=400)
                    existing_wrapup = any(str((c.labels or {}).get("kind") or "") == "wrapup" for c in existing)
                    existing_subtasks = [c for c in existing if str((c.labels or {}).get("kind") or "") != "wrapup"]
                    existing_keys = {
                        str((c.labels or {}).get("key") or "").strip()
                        for c in existing_subtasks
                        if str((c.labels or {}).get("key") or "").strip()
                    }

                    specs = parse_orchestrator_subtasks(getattr(result, "structured_digest", None))
                    if specs:
                        # Cap to avoid runaway delegation.
                        specs = specs[:12]
                        # Key-based dedupe: allows multiple waves (autopilot) without duplicating keys.
                        specs = [s for s in specs if s.key not in existing_keys]

                    if specs:
                        key_to_job: dict[str, str] = {s.key: str(uuid.uuid4()) for s in specs}
                        children: list[Task] = []
                        for spec in specs:
                            child_role = _coerce_orchestrator_role(spec.role)
                            child_profile = _orchestrator_profile(profiles, child_role)
                            model = _orchestrator_model_for_profile(cfg, child_profile)
                            effort = _orchestrator_effort_for_profile(child_profile, cfg)
                            mode_hint = _coerce_orchestrator_mode(spec.mode_hint or str(child_profile.get("mode_hint") or "ro"))
                            requires_approval = bool(
                                spec.requires_approval
                                or bool(child_profile.get("approval_required", False))
                                or mode_hint == "full"
                            )
                            deps = [key_to_job[k] for k in spec.depends_on if k in key_to_job]
                            trace: dict[str, str | int | float | bool | list[str]] = {
                                "source": "telegram",
                                "delegated_by": task.job_id,
                                "delegated_key": spec.key,
                                "profile_name": str(child_profile.get("name") or child_role),
                                "profile_role": child_role,
                                "max_runtime_seconds": int(child_profile.get("max_runtime_seconds") or 0),
                            }
                            child = Task.new(
                                source="telegram",
                                role=child_role,
                                input_text=spec.text,
                                request_type="task",
                                priority=int(spec.priority),
                                model=model,
                                effort=effort,
                                mode_hint=mode_hint,
                                requires_approval=requires_approval,
                                max_cost_window_usd=float(cfg.orchestrator_default_max_cost_window_usd),
                                chat_id=int(task.chat_id),
                                user_id=task.user_id,
                                reply_to_message_id=task.reply_to_message_id,
                                parent_job_id=root_ticket,
                                depends_on=deps,
                                labels={"ticket": root_ticket, "kind": "subtask", "key": spec.key},
                                artifacts_dir=str((cfg.artifacts_root / key_to_job[spec.key]).resolve()),
                                trace=trace,
                                job_id=key_to_job[spec.key],
                            )
                            children.append(child)

                        if children:
                            # Persist delegation graph (delegated + depends_on edges).
                            try:
                                for c in children:
                                    ckey = str((c.labels or {}).get("key") or "").strip() or None
                                    orch_q.append_delegation_edge(
                                        root_ticket_id=root_ticket,
                                        from_job_id=task.job_id,
                                        to_job_id=c.job_id,
                                        edge_type="delegated",
                                        to_role=c.role,
                                        to_key=ckey,
                                        details={"priority": int(c.priority or 2), "text": (c.input_text or "")[:400]},
                                    )
                                    for dep in (c.depends_on or []):
                                        if dep:
                                            orch_q.append_delegation_edge(
                                                root_ticket_id=root_ticket,
                                                from_job_id=c.job_id,
                                                to_job_id=str(dep),
                                                edge_type="depends_on",
                                                to_role=None,
                                                to_key=None,
                                                details={},
                                            )
                            except Exception:
                                pass
                            orch_q.submit_batch(children)
                            try:
                                orch_q.update_trace(root_ticket, delegated_count=int(len(children)), live_at=time.time())
                            except Exception:
                                pass

                    # Wrap-up is only scheduled for manual top-level tickets (avoid autopilot spam).
                    if is_top_level_manual:
                        wrapup_deps: list[str] = []
                        if existing_subtasks:
                            wrapup_deps.extend([c.job_id for c in existing_subtasks if c.job_id != task.job_id])
                        if specs:
                            wrapup_deps.extend([str(v) for v in key_to_job.values()])
                        wrapup_deps = [d for d in wrapup_deps if d and d != task.job_id]
                        if wrapup_deps and not existing_wrapup:
                            orch_profile = _orchestrator_profile(profiles, "jarvis")
                            orch_model = _orchestrator_model_for_profile(cfg, orch_profile)
                            orch_effort = _orchestrator_effort_for_profile(orch_profile, cfg)
                            wrap_id = str(uuid.uuid4())
                            wrap_trace: dict[str, str | int | float | bool | list[str]] = {
                                "source": "telegram",
                                "wrapup_for": root_ticket,
                                "profile_name": str(orch_profile.get("name") or "jarvis"),
                                "profile_role": "jarvis",
                                "max_runtime_seconds": int(orch_profile.get("max_runtime_seconds") or 0),
                            }
                            wrap = Task.new(
                                source="telegram",
                                role="jarvis",
                                input_text=f"Wrap up ticket {root_ticket}. Provide an executive summary and next actions.",
                                request_type="review",
                                priority=int(task.priority or 2),
                                model=orch_model,
                                effort=orch_effort,
                                mode_hint="ro",
                                requires_approval=False,
                                max_cost_window_usd=float(cfg.orchestrator_default_max_cost_window_usd),
                                chat_id=int(task.chat_id),
                                user_id=task.user_id,
                                reply_to_message_id=task.reply_to_message_id,
                                parent_job_id=root_ticket,
                                depends_on=wrapup_deps,
                                labels={"ticket": root_ticket, "kind": "wrapup"},
                                artifacts_dir=str((cfg.artifacts_root / wrap_id).resolve()),
                                trace=wrap_trace,
                                job_id=wrap_id,
                            )
                            orch_q.submit_task(wrap)
                            try:
                                orch_q.update_trace(root_ticket, wrapup_job_id=wrap.job_id, live_at=time.time())
                            except Exception:
                                pass
            except Exception:
                LOG.exception("Failed to delegate orchestrator subtasks. job=%s", task.job_id)
        except Exception as e:
            LOG.exception("Orchestrator worker failed for task=%s", task.job_id)
            try:
                orch_q.update_state(task.job_id, "failed", error=str(e))
                _send_orchestrator_result(
                    api,
                    task,
                    {
                        "status": "error",
                        "summary": f"Worker failed: {e}",
                        "artifacts": [],
                        "logs": str(e),
                        "next_action": None,
                    },
                    cfg=cfg,
                )
            except Exception:
                LOG.exception("Failed to report orchestrator worker failure for task=%s", task.job_id)



@dataclass(frozen=True)
class _TranscribeRequest:
    chat_id: int
    user_id: int
    message_id: int
    username: str | None
    file_id: str
    orig_name: str


def _transcribe_worker_loop(
    *,
    cfg: BotConfig,
    api: TelegramAPI,
    stop_event: threading.Event,
    requests: "queue.Queue[_TranscribeRequest]",
    openai_transcriber: OpenAITranscriber | None,
    orchestrator_queue: OrchestratorQueue | None,
    orchestrator_profiles: dict[str, dict[str, Any]] | None,
    jobs: "queue.Queue[Job]",
    tracker: JobTracker,
) -> None:
    """
    Background voice/audio transcriber so poll_loop can ACK quickly.

    Grounded behavior (in-code): mirrors the synchronous transcription path but runs off-thread.
    """
    while not stop_event.is_set():
        try:
            req = requests.get(timeout=0.5)
        except queue.Empty:
            continue
        except Exception:
            continue

        chat_id = int(req.chat_id)
        message_id = int(req.message_id)
        file_id = (req.file_id or "").strip()
        if not file_id:
            continue

        backend = _effective_transcribe_backend(cfg)
        eff_lang = _effective_transcribe_language(cfg)
        eff_timeout = _effective_transcribe_timeout(cfg)
        eff_threads = _effective_whisper_threads(cfg)
        eff_model_path = _effective_whisper_model_path(cfg)

        def _pick_backend() -> str:
            if backend == "openai":
                return "openai"
            if backend == "whispercpp":
                return "whispercpp"
            # auto: prefer local if available, else OpenAI.
            w = WhisperCppTranscriber(
                ffmpeg_bin=cfg.ffmpeg_bin,
                whisper_bin=cfg.whispercpp_bin,
                model_path=eff_model_path,
                threads=eff_threads,
                timeout_seconds=eff_timeout,
                language=eff_lang,
                prompt=cfg.transcribe_prompt,
            )
            ok, _reason = w.is_available()
            if ok:
                return "whispercpp"
            if openai_transcriber is not None:
                return "openai"
            return ""

        chosen = _pick_backend()
        if not chosen:
            api.send_message(
                chat_id,
                "Transcripción habilitada pero no hay backend disponible. Configura whisper.cpp (recomendado) o OPENAI_API_KEY.",
                reply_to_message_id=message_id if message_id else None,
            )
            continue

        upload_dir = cfg.codex_workdir / ".codexbot_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe = _safe_filename(req.orig_name, fallback="audio.bin")
        dest_path = upload_dir / f"tg_audio_{chat_id}_{message_id}_{safe}"
        if dest_path.exists():
            dest_path = upload_dir / f"tg_audio_{chat_id}_{message_id}_{int(time.time())}_{safe}"

        incoming_text = ""
        try:
            info = api.get_file(file_id)
            fp = info.get("file_path") if isinstance(info, dict) else None
            if not isinstance(fp, str) or not fp:
                raise RuntimeError("Telegram getFile did not return file_path")
            api.download_file_to(file_path=fp, dest=dest_path, max_bytes=cfg.max_download_bytes)

            try:
                sz = dest_path.stat().st_size
            except Exception:
                sz = 0
            if cfg.transcribe_max_bytes > 0 and sz > cfg.transcribe_max_bytes:
                raise RuntimeError(f"Audio demasiado grande para transcribir (>{cfg.transcribe_max_bytes} bytes)")

            if chosen == "whispercpp":
                w = WhisperCppTranscriber(
                    ffmpeg_bin=cfg.ffmpeg_bin,
                    whisper_bin=cfg.whispercpp_bin,
                    model_path=eff_model_path,
                    threads=eff_threads,
                    timeout_seconds=eff_timeout,
                    language=eff_lang,
                    prompt=cfg.transcribe_prompt,
                )
                incoming_text = w.transcribe_file(input_path=dest_path)
            else:
                if openai_transcriber is None:
                    raise RuntimeError("OPENAI_API_KEY faltante o OpenAI transcriber no inicializado")
                incoming_text = openai_transcriber.transcribe(
                    audio_path=dest_path,
                    model=cfg.transcribe_model,
                    language=eff_lang,
                    prompt=cfg.transcribe_prompt,
                )
            incoming_text = (incoming_text or "").strip()
            if not incoming_text:
                raise RuntimeError("Transcripción vacía")
        except Exception as e:
            api.send_message(
                chat_id,
                f"No pude transcribir el audio: {e}",
                reply_to_message_id=message_id if message_id else None,
            )
            continue
        finally:
            try:
                dest_path.unlink(missing_ok=True)
            except Exception:
                pass

        prompt = _normalize_slash_aliases(incoming_text)
        job = Job(
            chat_id=chat_id,
            reply_to_message_id=message_id,
            user_text=prompt,
            argv=["exec", prompt],
            # Same rule as plain text: let orchestrator role profiles pick defaults.
            mode_hint=("" if cfg.orchestrator_enabled else cfg.codex_default_mode),
            epoch=0,
            threaded=True,
            image_paths=[],
            upload_paths=[],
            force_new_thread=False,
            prefer_voice_reply=True,
        )

        # Prefer orchestrator if enabled; otherwise fall back to legacy queue.
        if orchestrator_queue is not None and cfg.orchestrator_enabled:
            try:
                did_submit, orch_job_id = _submit_orchestrator_task(
                    cfg=cfg,
                    orch_q=orchestrator_queue,
                    profiles=orchestrator_profiles,
                    job=job,
                    user_id=req.user_id,
                )
            except Exception:
                did_submit = False
                orch_job_id = ""
                LOG.exception("Failed to submit transcribed task to orchestrator")
            if did_submit:
                api.send_message(
                    chat_id,
                    f"Transcrito y encolado: task={orch_job_id[:8]}",
                    reply_to_message_id=message_id if message_id else None,
                )
                continue

        ok, reason, epoch, q_after = tracker.try_mark_enqueued(chat_id, max_queued_per_chat=cfg.max_queued_per_chat)
        if not ok:
            api.send_message(chat_id, reason, reply_to_message_id=message_id if message_id else None)
            continue
        job = Job(
            chat_id=job.chat_id,
            reply_to_message_id=job.reply_to_message_id,
            user_text=job.user_text,
            argv=job.argv,
            mode_hint=job.mode_hint,
            epoch=epoch,
            threaded=job.threaded,
            image_paths=job.image_paths,
            upload_paths=job.upload_paths,
            force_new_thread=job.force_new_thread,
            prefer_voice_reply=job.prefer_voice_reply,
        )
        try:
            jobs.put(job, block=False)
            if q_after > 1 or tracker.inflight(chat_id) > 0:
                api.send_message(
                    chat_id,
                    f"Queued (voice) (mode={job.mode_hint or cfg.codex_default_mode}, queue_len={jobs.qsize()}).",
                    reply_to_message_id=message_id if message_id else None,
                )
        except queue.Full:
            tracker.on_dequeue(chat_id)
            api.send_message(chat_id, "Queue is full; try again in a bit.", reply_to_message_id=message_id if message_id else None)


def _is_exec_available(cmd: str) -> bool:
    if not cmd:
        return False
    p = Path(cmd).expanduser()
    if p.exists():
        return True
    return shutil.which(cmd) is not None


def _normalize_tts_speak_text(text: str, *, backend: str) -> str:
    """
    Light normalization to improve pronunciation for local TTS.
    Keep it conservative: the caption is still sent as text, this only affects spoken audio.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # Remove very long hex-like tokens (job ids, commit hashes) that sound bad.
    t = re.sub(r"\b[0-9a-f]{8,}\b", "", t, flags=re.IGNORECASE)

    # Slash commands: "/ticket 123" -> "ticket 123"
    t = re.sub(r"\b/(ticket|job|agents|dashboard|restart|say)\b", r"\1", t, flags=re.IGNORECASE)

    # Common acronyms and terms. Keep this short to avoid "spelled out" / staccato speech.
    # CEO preference: E2E is read in English ("end to end").
    repl: list[tuple[str, str]] = [
        (r"\bE2E\b", "end to end"),
        (r"\bPR\b", "pull request"),
        (r"\bCI/CD\b", "ci cd"),
    ]
    for pat, rep in repl:
        t = re.sub(pat, rep, t)

    # Reduce awkward pauses caused by punctuation / weird sequences.
    t = t.replace("...", ".")
    t = re.sub(r"[,:;]+", " ", t)
    t = re.sub(r"\s*[|/]\s*", " ", t)
    t = re.sub(r"\s+\.\s+", ". ", t)

    # Common English product/engineering terms: make them easier for Spanish TTS
    # without fully translating (keeps CEO preference of mixing ES + EN).
    eng_terms: list[tuple[str, str]] = [
        (r"\bbackend\b", "back end"),
        (r"\bfrontend\b", "front end"),
        (r"\bdashboard\b", "dash board"),
        (r"\bdeploy\b", "de ploy"),
        (r"\brelease\b", "re lease"),
        (r"\brollback\b", "roll back"),
        (r"\bhotfix\b", "hot fix"),
    ]
    for pat, rep in eng_terms:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)

    # Piper (Spanish) tends to sound better without excessive punctuation.
    if (backend or "").strip().lower() == "piper":
        t = t.replace("_", " ").replace("|", " ")

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _piper_to_wav(
    *,
    piper_bin: str,
    model_path: str,
    speaker: str,
    noise_scale: float,
    length_scale: float,
    noise_w: float,
    sentence_silence: float,
    text: str,
    output_wav_path: Path,
) -> None:
    """
    Run Piper locally to synthesize `text` into a WAV file.
    Uses stdin (no shell) and sets env vars so the bundled Piper release works reliably.
    """
    if not _is_exec_available(piper_bin):
        raise RuntimeError(f"piper no encontrado: {piper_bin or '(empty)'}")
    mp = (model_path or "").strip()
    if not mp or not Path(mp).expanduser().exists():
        raise RuntimeError(f"modelo piper no encontrado: {model_path or '(empty)'}")
    t = (text or "").strip()
    if not t:
        raise RuntimeError("Empty TTS input")

    out = output_wav_path
    out.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        piper_bin,
        "--model",
        str(Path(mp).expanduser()),
        "--output_file",
        str(out),
    ]
    spk = (speaker or "").strip()
    if spk:
        argv.extend(["--speaker", spk])
    # Tunables: only pass if explicitly set to a positive value.
    if float(noise_scale) > 0:
        argv.extend(["--noise_scale", str(float(noise_scale))])
    if float(length_scale) > 0:
        argv.extend(["--length_scale", str(float(length_scale))])
    if float(noise_w) > 0:
        argv.extend(["--noise_w", str(float(noise_w))])
    if float(sentence_silence) > 0:
        argv.extend(["--sentence_silence", str(float(sentence_silence))])

    env = dict(os.environ)
    try:
        # When running from the bundled tarball, shared libs and espeak data live next to the binary.
        piper_dir = Path(piper_bin).expanduser().resolve().parent
        if piper_dir.exists():
            env["LD_LIBRARY_PATH"] = str(piper_dir) + (
                (":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else ""
            )
            espeak_data = piper_dir / "espeak-ng-data"
            if espeak_data.exists():
                env["ESPEAK_DATA_PATH"] = str(espeak_data)
    except Exception:
        pass

    p = subprocess.run(
        argv,
        input=(t + "\n"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(Path(piper_bin).expanduser().resolve().parent)
        if Path(piper_bin).expanduser().exists()
        else None,
    )
    if p.returncode != 0:
        raise RuntimeError(f"piper fallo: {(p.stderr or p.stdout or '').strip()[:2000]}")


def _ffmpeg_to_ogg_opus_voip(
    *,
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    pitch_ratio: float = 1.0,
) -> None:
    if not _is_exec_available(ffmpeg_bin):
        raise RuntimeError(f"ffmpeg no encontrado: {ffmpeg_bin or '(empty)'}")
    pr = float(pitch_ratio) if float(pitch_ratio) > 0 else 1.0
    use_pitch = abs(pr - 1.0) >= 0.001
    argv = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
    ]
    if use_pitch:
        # Use rubberband filter (time-stretch + pitch shift) to keep duration while lowering pitch.
        argv.extend(["-filter:a", f"rubberband=pitch={pr:.6f}"])
    argv.extend(
        [
        "-ac",
        "1",
        "-ar",
        "48000",
        "-c:a",
        "libopus",
        "-b:a",
        "32k",
        "-application",
        "voip",
        str(output_path),
        ]
    )
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg fallo: {(p.stderr or p.stdout or '').strip()[:2000]}")


def _make_tone_ogg(*, ffmpeg_bin: str, output_path: Path, seconds: float = 1.2, freq_hz: int = 660) -> None:
    if not _is_exec_available(ffmpeg_bin):
        raise RuntimeError(f"ffmpeg no encontrado: {ffmpeg_bin or '(empty)'}")
    # Generate an OGG/Opus voice-note friendly tone (works as a minimal voice-out fallback).
    argv = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={int(freq_hz)}:duration={float(seconds)}",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-c:a",
        "libopus",
        "-b:a",
        "32k",
        "-application",
        "voip",
        str(output_path),
    ]
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg fallo: {(p.stderr or p.stdout or '').strip()[:2000]}")


def _synthesize_voice_note_ogg(
    *,
    cfg: BotConfig,
    tts: OpenAITTS | None,
    text: str,
) -> Path:
    """
    Returns a temp .ogg path suitable for Telegram sendVoice.
    Falls back to a short tone if TTS isn't available.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="poncebot_tts_"))
    out_ogg = tmp_dir / "voice.ogg"

    backend = (cfg.tts_backend or "none").strip().lower()
    speak = (text or "").strip()
    if cfg.tts_max_chars > 0 and len(speak) > cfg.tts_max_chars:
        speak = speak[: cfg.tts_max_chars].rstrip() + "..."

    # Normalize for clearer pronunciation (especially acronyms like E2E, CI/CD, etc).
    speak = _normalize_tts_speak_text(speak, backend=backend)
    semis = float(getattr(cfg, "tts_voice_pitch_semitones", 0.0) or 0.0)
    if semis > 12.0:
        semis = 12.0
    if semis < -12.0:
        semis = -12.0
    pitch_ratio = pow(2.0, semis / 12.0) if abs(semis) >= 0.001 else 1.0

    try:
        if cfg.voice_out_enabled and backend == "piper":
            # Local/free speech via Piper, then transcode to Telegram-voice friendly OGG/Opus.
            in_path = tmp_dir / "voice_in.wav"
            _piper_to_wav(
                piper_bin=cfg.tts_piper_bin,
                model_path=cfg.tts_piper_model_path,
                speaker=cfg.tts_piper_speaker,
                noise_scale=cfg.tts_piper_noise_scale,
                length_scale=cfg.tts_piper_length_scale,
                noise_w=cfg.tts_piper_noise_w,
                sentence_silence=cfg.tts_piper_sentence_silence,
                text=speak,
                output_wav_path=in_path,
            )
            _ffmpeg_to_ogg_opus_voip(
                ffmpeg_bin=cfg.ffmpeg_bin,
                input_path=in_path,
                output_path=out_ogg,
                pitch_ratio=pitch_ratio,
            )
            return out_ogg

        if cfg.voice_out_enabled and backend == "openai" and tts is not None and cfg.openai_api_key:
            # Ask OpenAI for audio, then transcode to Telegram-voice friendly OGG/Opus.
            raw = tts.synthesize(
                text=speak,
                model=cfg.tts_openai_model,
                voice=cfg.tts_openai_voice,
                response_format=cfg.tts_openai_response_format,
            )
            in_path = tmp_dir / f"voice_in.{cfg.tts_openai_response_format}"
            in_path.write_bytes(raw)
            _ffmpeg_to_ogg_opus_voip(
                ffmpeg_bin=cfg.ffmpeg_bin,
                input_path=in_path,
                output_path=out_ogg,
                pitch_ratio=pitch_ratio,
            )
            return out_ogg

        if cfg.voice_out_enabled and backend == "tone":
            _make_tone_ogg(ffmpeg_bin=cfg.ffmpeg_bin, output_path=out_ogg)
            return out_ogg

        # Default fallback: tone.
        _make_tone_ogg(ffmpeg_bin=cfg.ffmpeg_bin, output_path=out_ogg)
        return out_ogg
    except Exception:
        # Last resort: try a tone even if TTS failed.
        try:
            _make_tone_ogg(ffmpeg_bin=cfg.ffmpeg_bin, output_path=out_ogg)
            return out_ogg
        except Exception:
            raise


def _send_voice_note(
    *,
    api: TelegramAPI,
    cfg: BotConfig,
    chat_id: int,
    reply_to_message_id: int,
    tts: OpenAITTS | None,
    speak_text: str,
    caption_text: str,
) -> int | None:
    ogg_path = _synthesize_voice_note_ogg(cfg=cfg, tts=tts, text=speak_text)
    try:
        mid = api.send_voice(
            chat_id,
            ogg_path,
            filename="jarvis.ogg",
            caption=(caption_text or "").strip()[:900],
            reply_to_message_id=reply_to_message_id if reply_to_message_id else None,
        )
        try:
            _qa_append_evidence(
                cfg,
                chat_id=chat_id,
                event={
                    "event_type": "voice_out",
                    "reply_to_message_id": int(reply_to_message_id or 0),
                    "voice_message_id": int(mid) if mid is not None else None,
                    "tts_backend": (cfg.tts_backend or "").strip().lower(),
                    "caption_len": len((caption_text or "").strip()),
                    "speak_len": len((speak_text or "").strip()),
                },
            )
        except Exception:
            pass
        return mid
    finally:
        try:
            ogg_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            # Remove the temp dir as well.
            if ogg_path.parent and ogg_path.parent.name.startswith("poncebot_tts_"):
                shutil.rmtree(ogg_path.parent, ignore_errors=True)
        except Exception:
            pass


def worker_loop(
    *,
    cfg: BotConfig,
    api: TelegramAPI,
    jobs: "queue.Queue[Job]",
    tracker: JobTracker,
    stop_event: threading.Event,
    thread_mgr: ThreadManager,
) -> None:
    tts_client: OpenAITTS | None = None
    if cfg.openai_api_key and (cfg.tts_backend or "").strip().lower() == "openai":
        tts_client = OpenAITTS(
            api_key=cfg.openai_api_key,
            api_base_url=cfg.openai_api_base_url,
            timeout_seconds=cfg.http_timeout_seconds,
            max_retries=cfg.http_max_retries,
            retry_initial_seconds=cfg.http_retry_initial_seconds,
            retry_max_seconds=cfg.http_retry_max_seconds,
        )
    while not stop_event.is_set():
        try:
            job = jobs.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            tracker.on_dequeue(job.chat_id)
            if not tracker.wait_turn_and_mark_inflight(job, stop_event):
                # Stale (canceled) or shutting down.
                continue

            profile = _auth_effective_profile_name(cfg, chat_id=job.chat_id) if cfg.auth_enabled else ""
            eff_cfg = _apply_profile_to_cfg(cfg, profile_name=profile) if profile else cfg

            # Treat empty mode_hint as "use effective defaults" (allows orchestrator to omit legacy defaults).
            mode_hint = (job.mode_hint or "").strip().lower()
            if not mode_hint:
                mode_hint = eff_cfg.codex_default_mode
            if mode_hint not in ("ro", "rw", "full"):
                mode_hint = eff_cfg.codex_default_mode
            if profile and not _profile_allows_mode(cfg, profile_name=profile, requested=mode_hint):
                api.send_message(
                    job.chat_id,
                    f"No permitido por tu perfil ({profile}). Modo solicitado={mode_hint}.",
                    reply_to_message_id=job.reply_to_message_id,
                )
                continue

            if job.argv and job.argv[0] == "__skills__":
                _run_internal_skills_job(
                    cfg=cfg,
                    api=api,
                    tracker=tracker,
                    stop_event=stop_event,
                    job=job,
                    eff_cfg=eff_cfg,
                )
                continue

            runner = CodexRunner(eff_cfg, chat_id=job.chat_id)

            model, effort = _job_model_label(eff_cfg, job.argv, chat_id=job.chat_id)
            model_part = _format_model_for_display(model, effort)
            if not eff_cfg.strict_proxy:
                api.send_message(
                    job.chat_id,
                    "Running codex (mode=%s, provider=%s, model=%s, workdir=%s)..."
                    % (
                        mode_hint,
                        eff_cfg.codex_local_provider if eff_cfg.codex_use_oss else "default",
                        model_part,
                        eff_cfg.codex_workdir,
                    ),
                    reply_to_message_id=job.reply_to_message_id,
                )

            started_new_thread = False
            if job.threaded and job.argv and job.argv[0] == "exec" and len(job.argv) == 2:
                tid = thread_mgr.get(job.chat_id)
                if tid and not job.force_new_thread:
                    running = runner.start_threaded_resume(
                        thread_id=tid,
                        prompt=job.user_text,
                        mode_hint=mode_hint,
                        image_paths=job.image_paths,
                    )
                else:
                    started_new_thread = True
                    running = runner.start_threaded_new(
                        prompt=job.user_text,
                        mode_hint=mode_hint,
                        image_paths=job.image_paths or None,
                    )
            else:
                running = runner.start(argv=job.argv, mode_hint=mode_hint)
            tracker.set_running_proc(job.chat_id, running.proc)

            last_beat = time.time()
            timed_out = False
            last_typing = 0.0
            while running.proc.poll() is None:
                if stop_event.is_set():
                    tracker.cancel(job.chat_id)
                    break
                # If user canceled, kill the running proc.
                if tracker.current_epoch(job.chat_id) != job.epoch:
                    _terminate_process(running.proc)
                    break
                if cfg.codex_timeout_seconds > 0 and (time.time() - running.start_time) >= cfg.codex_timeout_seconds:
                    timed_out = True
                    _terminate_process(running.proc)
                    break
                # In strict proxy mode, avoid noisy heartbeat messages; show a typing indicator instead.
                if eff_cfg.strict_proxy:
                    now = time.time()
                    if now - last_typing >= 4.0:
                        try:
                            api.send_chat_action(job.chat_id, "typing")
                        except Exception:
                            # Best-effort only.
                            pass
                        last_typing = now
                if cfg.heartbeat_seconds > 0 and (time.time() - last_beat) >= cfg.heartbeat_seconds:
                    elapsed = int(time.time() - running.start_time)
                    try:
                        if not eff_cfg.strict_proxy:
                            api.send_message(
                                job.chat_id,
                                f"Still running... elapsed={elapsed}s",
                                reply_to_message_id=job.reply_to_message_id,
                            )
                    except Exception:
                        LOG.exception("Failed to send heartbeat")
                    last_beat = time.time()
                time.sleep(0.5)

            try:
                running.proc.wait(timeout=5)
            except Exception:
                pass

            canceled = tracker.current_epoch(job.chat_id) != job.epoch
            secs = time.time() - running.start_time
            code = int(running.proc.returncode) if running.proc.returncode is not None else 1

            # Prefer Codex last-message output, fall back to stdout.
            final_msg = ""
            if running.last_msg_path:
                final_msg = _read_text_file(running.last_msg_path).strip()
            if not final_msg:
                # If stdout is small-ish, read it; otherwise show a tail and rely on send-as-file if enabled.
                try:
                    sz = running.stdout_path.stat().st_size
                except OSError:
                    sz = 0
                if sz <= 256_000:
                    final_msg = _strip_ansi(_read_text_file(running.stdout_path)).strip()
                else:
                    final_msg = _tail_file_text(running.stdout_path, max_chars=6000).strip() or "(no output)"
            if not final_msg:
                final_msg = "(no output)"

            tail_stdout = _tail_file_text(running.stdout_path, max_chars=3500)
            tail_stderr = _tail_file_text(running.stderr_path, max_chars=3500)
            debug_tail = _tail_text((tail_stdout + "\n" + tail_stderr).strip(), max_chars=6000)

            header = f"exit={code} secs={secs:.1f} mode={mode_hint}"

            if canceled:
                api.send_message(job.chat_id, "Canceled.", reply_to_message_id=job.reply_to_message_id)
                continue

            if timed_out:
                api.send_message(
                    job.chat_id,
                    f"Codex timed out after {cfg.codex_timeout_seconds}s.",
                    reply_to_message_id=job.reply_to_message_id,
                )
                if debug_tail:
                    dbg = "debug tail:\n" + debug_tail
                    for idx, ch in enumerate(_chunk_text(dbg, limit=TELEGRAM_MSG_LIMIT - 64), start=1):
                        api.send_message(job.chat_id, f"[debug {idx}]\n{ch}", reply_to_message_id=job.reply_to_message_id)
                continue

            body = final_msg.strip() or "(empty)"
            if job.threaded and code == 0:
                # Chat UX: for threaded conversations, keep the output "just the assistant message".
                out = body
            else:
                out = header + "\n\n" + body

            if started_new_thread:
                try:
                    tid = _extract_thread_id_from_jsonl_file(running.stdout_path)
                    if tid:
                        thread_mgr.set(job.chat_id, tid)
                        _persist_thread_id(cfg, chat_id=job.chat_id, thread_id=tid)
                        LOG.info("Set thread_id for chat_id=%s: %s", job.chat_id, tid)
                except Exception:
                    LOG.exception("Failed to extract/set new thread id")

            sent_voice = False
            if job.prefer_voice_reply and cfg.voice_out_enabled:
                try:
                    caption = (body or "").strip()
                    if len(caption) > 850:
                        caption = caption[:850].rstrip() + "..."
                    speak_text = (body or "").strip() or "(empty)"
                    if (cfg.tts_backend or "").strip().lower() == "openai" and not cfg.openai_api_key:
                        caption = "TTS no configurado (OPENAI_API_KEY faltante). Enviando tono de prueba.\n\n" + caption
                    _send_voice_note(
                        api=api,
                        cfg=cfg,
                        chat_id=job.chat_id,
                        reply_to_message_id=job.reply_to_message_id,
                        tts=tts_client,
                        speak_text=speak_text,
                        caption_text=caption,
                    )
                    sent_voice = True
                except Exception:
                    sent_voice = False
                    LOG.exception("Voice reply failed; falling back to text")

            if not sent_voice:
                if cfg.send_as_file_threshold_chars > 0 and len(out) > cfg.send_as_file_threshold_chars:
                    tmp_f = tempfile.NamedTemporaryFile(prefix="codexbot_output_", suffix=".txt", delete=False)
                    tmp = Path(tmp_f.name)
                    tmp_f.close()
                    try:
                        tmp.write_text(out + "\n", encoding="utf-8", errors="replace")
                        try:
                            api.send_message(
                                job.chat_id,
                                f"{header}\n\n(Output too large; sent as file.)",
                                reply_to_message_id=job.reply_to_message_id,
                            )
                            api.send_document(
                                job.chat_id,
                                tmp,
                                filename="codex_output.txt",
                                reply_to_message_id=job.reply_to_message_id,
                            )
                        except Exception:
                            LOG.exception("Failed to send as file; falling back to chunked messages")
                            chunks = _chunk_text(out, limit=TELEGRAM_MSG_LIMIT - 64)
                            for idx, ch in enumerate(chunks, start=1):
                                prefix = "" if len(chunks) == 1 else f"[{idx}/{len(chunks)}]\n"
                                api.send_message(job.chat_id, prefix + ch, reply_to_message_id=job.reply_to_message_id)
                    finally:
                        tmp.unlink(missing_ok=True)
                else:
                    # Leave headroom for chunk headers like "[1/3]\n".
                    chunks = _chunk_text(out, limit=TELEGRAM_MSG_LIMIT - 64)
                    if len(chunks) == 1:
                        api.send_message(job.chat_id, chunks[0], reply_to_message_id=job.reply_to_message_id)
                    else:
                        for idx, ch in enumerate(chunks, start=1):
                            api.send_message(
                                job.chat_id,
                                f"[{idx}/{len(chunks)}]\n{ch}",
                                reply_to_message_id=job.reply_to_message_id,
                            )

            # If the run produced images in the workdir, attach them (previewable) after the main output.
            try:
                pngs = _collect_png_artifacts(cfg, start_time=running.start_time, text=out)
                for p in pngs[:4]:
                    try:
                        api.send_photo(job.chat_id, p, caption=p.name, reply_to_message_id=job.reply_to_message_id)
                    except Exception:
                        # Fall back to sendDocument if Telegram rejects the image upload.
                        api.send_document(job.chat_id, p, filename=p.name, reply_to_message_id=job.reply_to_message_id)
            except Exception:
                LOG.exception("Failed to send PNG artifacts")

            if code != 0 and debug_tail:
                dbg = "debug tail:\n" + debug_tail
                for idx, ch in enumerate(_chunk_text(dbg, limit=TELEGRAM_MSG_LIMIT - 64), start=1):
                    api.send_message(job.chat_id, f"[debug {idx}]\n{ch}", reply_to_message_id=job.reply_to_message_id)
        except Exception:
            LOG.exception("Worker error")
            try:
                api.send_message(job.chat_id, "Internal error. Check server logs.", reply_to_message_id=job.reply_to_message_id)
            except Exception:
                LOG.exception("Failed to send Internal error message to Telegram")
        finally:
            try:
                tracker.clear_running(job.chat_id)
            except Exception:
                LOG.exception("Failed to clear running state")
            # Best-effort cleanup of downloaded image files.
            try:
                for p in (job.image_paths or []):
                    Path(p).unlink(missing_ok=True)
            except Exception:
                LOG.exception("Failed to clean up image uploads")
            # Best-effort cleanup of temp files.
            try:
                if "running" in locals():
                    r = locals()["running"]
                    if isinstance(r, CodexRunner.Running):
                        if r.last_msg_path:
                            r.last_msg_path.unlink(missing_ok=True)
                        r.stdout_path.unlink(missing_ok=True)
                        r.stderr_path.unlink(missing_ok=True)
            except Exception:
                LOG.exception("Failed to clean up temp files")
            jobs.task_done()


def poll_loop(
    *,
    cfg: BotConfig,
    api: TelegramAPI,
    jobs: "queue.Queue[Job]",
    tracker: JobTracker,
    stop_event: threading.Event,
    thread_mgr: ThreadManager,
    orchestrator_queue: OrchestratorQueue | None = None,
    orchestrator_profiles: dict[str, dict[str, Any]] | None = None,
    offset: int = 0,
    command_suggestions_synced: bool = False,
) -> None:
    backoff = 1.0
    last_unauth_reply_at: dict[int, float] = {}
    next_command_sync_at = 0.0
    openai_transcriber: OpenAITranscriber | None = None
    openai_tts: OpenAITTS | None = None
    if cfg.openai_api_key:
        # Initialize once; per-message selection is controlled by /voice state + cfg defaults.
        if cfg.transcribe_backend in ("openai", "auto", "whispercpp"):
            openai_transcriber = OpenAITranscriber(
                api_key=cfg.openai_api_key,
                api_base_url=cfg.openai_api_base_url,
                timeout_seconds=cfg.http_timeout_seconds,
                max_retries=cfg.http_max_retries,
                retry_initial_seconds=cfg.http_retry_initial_seconds,
                retry_max_seconds=cfg.http_retry_max_seconds,
            )
        if (cfg.tts_backend or "").strip().lower() == "openai":
            openai_tts = OpenAITTS(
                api_key=cfg.openai_api_key,
                api_base_url=cfg.openai_api_base_url,
                timeout_seconds=cfg.http_timeout_seconds,
                max_retries=cfg.http_max_retries,
                retry_initial_seconds=cfg.http_retry_initial_seconds,
                retry_max_seconds=cfg.http_retry_max_seconds,
            )

    transcribe_requests: "queue.Queue[_TranscribeRequest]" | None = None
    if cfg.transcribe_async and _effective_transcribe_enabled(cfg):
        transcribe_requests = queue.Queue(maxsize=16)
        t = threading.Thread(
            target=_transcribe_worker_loop,
            kwargs={
                "cfg": cfg,
                "api": api,
                "stop_event": stop_event,
                "requests": transcribe_requests,
                "openai_transcriber": openai_transcriber,
                "orchestrator_queue": orchestrator_queue,
                "orchestrator_profiles": orchestrator_profiles,
                "jobs": jobs,
                "tracker": tracker,
            },
            daemon=True,
            name="transcribe-worker",
        )
        t.start()

    while not stop_event.is_set():
        try:
            now = time.time()
            if not command_suggestions_synced and now >= next_command_sync_at:
                try:
                    _sync_telegram_command_suggestions(api, cfg)
                    command_suggestions_synced = True
                    LOG.info("Telegram command suggestions synced.")
                except Exception:
                    LOG.exception("Failed to sync Telegram command suggestions; retrying in 60s")
                    next_command_sync_at = now + 60.0

            updates = api.get_updates(offset=offset, timeout_seconds=cfg.poll_timeout_seconds)
            backoff = 1.0

            for upd in updates:
                update_id = int(upd.get("update_id", -1))
                if update_id >= 0:
                    offset = max(offset, update_id + 1)

                msg = upd.get("message") or {}
                text = msg.get("text")
                incoming_prefer_voice_reply = False

                chat = msg.get("chat") or {}
                from_user = msg.get("from") or {}
                chat_id = int(chat.get("id", 0))
                user_id = int(from_user.get("id", 0))
                message_id = int(msg.get("message_id", 0))
                username = from_user.get("username")
                if username is not None and not isinstance(username, str):
                    username = None

                # Text messages or media messages (photo/document).
                is_media = not isinstance(text, str)
                incoming_text = text if isinstance(text, str) else ""

                # Optional: voice/audio -> transcribe into text, then handle as a normal text message.
                if is_media and _effective_transcribe_enabled(cfg):
                    voice = msg.get("voice")
                    audio = msg.get("audio")

                    file_id = ""
                    orig_name = ""
                    if isinstance(voice, dict) and isinstance(voice.get("file_id"), str):
                        file_id = voice["file_id"]
                        orig_name = "voice.ogg"
                    elif isinstance(audio, dict) and isinstance(audio.get("file_id"), str):
                        file_id = audio["file_id"]
                        fn = audio.get("file_name")
                        orig_name = fn if isinstance(fn, str) and fn else "audio.bin"

                    if file_id:
                        incoming_prefer_voice_reply = True
                        try:
                            dur = 0
                            if isinstance(voice, dict) and isinstance(voice.get("duration"), (int, float)):
                                dur = int(voice.get("duration") or 0)
                            elif isinstance(audio, dict) and isinstance(audio.get("duration"), (int, float)):
                                dur = int(audio.get("duration") or 0)
                            _qa_append_evidence(
                                cfg,
                                chat_id=chat_id,
                                event={
                                    "event_type": "voice_in",
                                    "message_id": int(message_id or 0),
                                    "file_id": str(file_id),
                                    "kind": "voice" if isinstance(voice, dict) else "audio",
                                    "duration_s": dur,
                                    "user_id": int(user_id or 0),
                                },
                            )
                        except Exception:
                            pass
                        if cfg.transcribe_async and transcribe_requests is not None:
                            # Auth preflight: avoid downloading/transcribing for unauthorized chats.
                            incoming_stub = IncomingMessage(
                                update_id=update_id,
                                chat_id=chat_id,
                                user_id=user_id,
                                message_id=message_id,
                                username=username,
                                text="",
                            )
                            if not _is_authorized(cfg, incoming_stub):
                                now = time.time()
                                last = last_unauth_reply_at.get(chat_id, 0.0)
                                if now - last >= cfg.unauthorized_reply_cooldown_seconds:
                                    last_unauth_reply_at[chat_id] = now
                                    api.send_message(
                                        chat_id,
                                        "Unauthorized. Ask the admin to add your chat_id/user_id.\n\n" + _whoami_text(incoming_stub),
                                        reply_to_message_id=message_id if message_id else None,
                                    )
                                continue

                            if cfg.auth_enabled:
                                active, _sess = _auth_is_session_active(cfg, chat_id=chat_id)
                                if not active:
                                    api.send_message(
                                        chat_id,
                                        _auth_required_text(),
                                        reply_to_message_id=message_id if message_id else None,
                                    )
                                    continue
                                _auth_touch_session(cfg, chat_id=chat_id)

                            api.send_message(
                                chat_id,
                                "Recibido, transcribiendo y encolando...",
                                reply_to_message_id=message_id if message_id else None,
                            )
                            try:
                                transcribe_requests.put_nowait(
                                    _TranscribeRequest(
                                        chat_id=chat_id,
                                        user_id=user_id,
                                        message_id=message_id,
                                        username=username,
                                        file_id=file_id,
                                        orig_name=orig_name or "audio.bin",
                                    )
                                )
                            except queue.Full:
                                api.send_message(
                                    chat_id,
                                    "Cola de transcripción llena; intenta de nuevo en un momento.",
                                    reply_to_message_id=message_id if message_id else None,
                                )
                            continue

                        backend = _effective_transcribe_backend(cfg)
                        eff_lang = _effective_transcribe_language(cfg)
                        eff_timeout = _effective_transcribe_timeout(cfg)
                        eff_threads = _effective_whisper_threads(cfg)
                        eff_model_path = _effective_whisper_model_path(cfg)

                        def _pick_backend() -> str:
                            if backend == "openai":
                                return "openai"
                            if backend == "whispercpp":
                                return "whispercpp"
                            # auto: prefer local if available, else OpenAI.
                            w = WhisperCppTranscriber(
                                ffmpeg_bin=cfg.ffmpeg_bin,
                                whisper_bin=cfg.whispercpp_bin,
                                model_path=eff_model_path,
                                threads=eff_threads,
                                timeout_seconds=eff_timeout,
                                language=eff_lang,
                                prompt=cfg.transcribe_prompt,
                            )
                            ok, _reason = w.is_available()
                            if ok:
                                return "whispercpp"
                            if openai_transcriber is not None:
                                return "openai"
                            return ""

                        chosen = _pick_backend()
                        if not chosen:
                            api.send_message(
                                chat_id,
                                "Transcripción habilitada pero no hay backend disponible. "
                                "Configura whisper.cpp (recomendado) o OPENAI_API_KEY.",
                                reply_to_message_id=message_id if message_id else None,
                            )
                            continue

                        upload_dir = cfg.codex_workdir / ".codexbot_uploads"
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        safe = _safe_filename(orig_name, fallback="audio.bin")
                        dest_path = upload_dir / f"tg_audio_{chat_id}_{message_id}_{safe}"
                        if dest_path.exists():
                            dest_path = upload_dir / f"tg_audio_{chat_id}_{message_id}_{int(time.time())}_{safe}"

                        try:
                            info = api.get_file(file_id)
                            fp = info.get("file_path") if isinstance(info, dict) else None
                            if not isinstance(fp, str) or not fp:
                                raise RuntimeError("Telegram getFile did not return file_path")
                            api.download_file_to(file_path=fp, dest=dest_path, max_bytes=cfg.max_download_bytes)

                            try:
                                sz = dest_path.stat().st_size
                            except Exception:
                                sz = 0
                            if cfg.transcribe_max_bytes > 0 and sz > cfg.transcribe_max_bytes:
                                raise RuntimeError(f"Audio demasiado grande para transcribir (>{cfg.transcribe_max_bytes} bytes)")

                            if chosen == "whispercpp":
                                w = WhisperCppTranscriber(
                                    ffmpeg_bin=cfg.ffmpeg_bin,
                                    whisper_bin=cfg.whispercpp_bin,
                                    model_path=eff_model_path,
                                    threads=eff_threads,
                                    timeout_seconds=eff_timeout,
                                    language=eff_lang,
                                    prompt=cfg.transcribe_prompt,
                                )
                                incoming_text = w.transcribe_file(input_path=dest_path)
                            else:
                                if openai_transcriber is None:
                                    raise RuntimeError("OPENAI_API_KEY faltante o OpenAI transcriber no inicializado")
                                incoming_text = openai_transcriber.transcribe(
                                    audio_path=dest_path,
                                    model=cfg.transcribe_model,
                                    language=eff_lang,
                                    prompt=cfg.transcribe_prompt,
                                )
                            if not incoming_text:
                                raise RuntimeError("Transcripción vacía")
                        except Exception as e:
                            api.send_message(
                                chat_id,
                                f"No pude transcribir el audio: {e}",
                                reply_to_message_id=message_id if message_id else None,
                            )
                            continue
                        finally:
                            try:
                                dest_path.unlink(missing_ok=True)
                            except Exception:
                                pass

                incoming_text = _normalize_slash_aliases(incoming_text)
                incoming = IncomingMessage(
                    update_id=update_id,
                    chat_id=chat_id,
                    user_id=user_id,
                    message_id=message_id,
                    username=username,
                    text=incoming_text,
                )
                LOG.debug(
                    "Incoming update_id=%s chat_id=%s user_id=%s message_id=%s kind=%s text_chars=%s text_lines=%s",
                    update_id,
                    chat_id,
                    user_id,
                    message_id,
                    "text" if incoming_text.strip() else "media/empty",
                    len(incoming_text or ""),
                    (incoming_text or "").count("\n") + 1 if (incoming_text or "") else 0,
                )

                if not _is_authorized(cfg, incoming):
                    now = time.time()
                    last = last_unauth_reply_at.get(chat_id, 0.0)
                    if now - last >= cfg.unauthorized_reply_cooldown_seconds:
                        last_unauth_reply_at[chat_id] = now
                        api.send_message(
                            chat_id,
                            "Unauthorized. Ask the admin to add your chat_id/user_id.\n\n"
                            + _whoami_text(incoming),
                            reply_to_message_id=message_id if message_id else None,
                        )
                    continue

                # App-level auth: require /login per chat session (expires on inactivity).
                if cfg.auth_enabled:
                    # Allow a small set of commands before login.
                    preauth_ok = incoming_text.strip() in ("/start", "/help", "/whoami", "/login") or incoming_text.strip().startswith("/login ")

                    active, _sess = _auth_is_session_active(cfg, chat_id=chat_id)
                    if not active and not preauth_ok:
                        api.send_message(
                            chat_id,
                            _auth_required_text(),
                            reply_to_message_id=message_id if message_id else None,
                        )
                        continue
                    if active:
                        # Sliding session TTL: any message keeps the session alive.
                        _auth_touch_session(cfg, chat_id=chat_id)

                # QA evidence capture (human-in-the-loop). Stores JSONL under cfg.artifacts_root/<id>/telegram_qa.jsonl.
                # Safe: only affects authorized chats, and evidence writes are best-effort.
                qa_txt = incoming_text.strip()
                if qa_txt.startswith("/qa_evidence"):
                    parts = qa_txt.split(None, 1)
                    arg = parts[1].strip() if len(parts) > 1 else "status"
                    if arg.lower() in ("status", "s"):
                        cur = _qa_get_evidence_artifact_id(cfg, chat_id=chat_id)
                        if cur:
                            api.send_message(
                                chat_id,
                                f"QA evidence: ON ({cur})",
                                reply_to_message_id=message_id if message_id else None,
                            )
                        else:
                            api.send_message(
                                chat_id,
                                "QA evidence: OFF",
                                reply_to_message_id=message_id if message_id else None,
                            )
                        continue
                    if arg.lower() in ("off", "disable", "0"):
                        _qa_set_evidence_artifact_id(cfg, chat_id=chat_id, artifact_id="")
                        api.send_message(
                            chat_id,
                            "OK. QA evidence disabled.",
                            reply_to_message_id=message_id if message_id else None,
                        )
                        continue
                    if not _qa_is_safe_artifact_id(arg):
                        api.send_message(
                            chat_id,
                            "Uso: /qa_evidence <artifact_id> | /qa_evidence status | /qa_evidence off",
                            reply_to_message_id=message_id if message_id else None,
                        )
                        continue
                    _qa_set_evidence_artifact_id(cfg, chat_id=chat_id, artifact_id=arg)
                    _qa_append_evidence(
                        cfg,
                        chat_id=chat_id,
                        event={
                            "event_type": "qa_evidence_enabled",
                            "artifact_id": arg,
                            "tts_backend": (cfg.tts_backend or "").strip().lower(),
                            "voice_out_enabled": bool(cfg.voice_out_enabled),
                            "piper_model_path": str(cfg.tts_piper_model_path or ""),
                            "piper_bin": str(cfg.tts_piper_bin or ""),
                        },
                    )
                    api.send_message(
                        chat_id,
                        f"OK. QA evidence enabled: {arg}",
                        reply_to_message_id=message_id if message_id else None,
                    )
                    continue

                if qa_txt.startswith("/qa_feedback") or qa_txt.startswith("qa_ack_"):
                    fb = ""
                    if qa_txt.startswith("/qa_feedback"):
                        parts = qa_txt.split(None, 1)
                        fb = parts[1].strip() if len(parts) > 1 else ""
                    tag = qa_txt if qa_txt.startswith("qa_ack_") else ""

                    r = msg.get("reply_to_message") or {}
                    rmid = int(r.get("message_id", 0)) if isinstance(r, dict) else 0
                    r_has_voice = bool(isinstance(r, dict) and isinstance(r.get("voice"), dict))
                    r_has_document = bool(isinstance(r, dict) and isinstance(r.get("document"), dict))
                    r_caption = (r.get("caption") if isinstance(r, dict) else "") or ""
                    if not isinstance(r_caption, str):
                        r_caption = ""

                    _qa_append_evidence(
                        cfg,
                        chat_id=chat_id,
                        event={
                            "event_type": "qa_feedback",
                            "message_id": int(message_id or 0),
                            "reply_to_message_id": int(rmid or 0),
                            "reply_to_has_voice": bool(r_has_voice),
                            "reply_to_has_document": bool(r_has_document),
                            "reply_to_caption_preview": r_caption[:200],
                            "tag": tag,
                            "feedback": fb,
                            "user_id": int(user_id or 0),
                        },
                    )
                    api.send_message(
                        chat_id,
                        "OK. Feedback recorded.",
                        reply_to_message_id=message_id if message_id else None,
                    )
                    continue

                # Deterministic CEO queries (no Codex, no delegation).
                # This runs after authorization/auth checks, so we don't leak info to unauthorized chats.
                try:
                    if _maybe_handle_ceo_query(
                        api=api,
                        cfg=cfg,
                        msg=incoming,
                        orchestrator_profiles=orchestrator_profiles,
                        orchestrator_queue=orchestrator_queue,
                    ):
                        continue
                except Exception:
                    # Best-effort only; fall back to normal routing.
                    pass

                if incoming_text.strip().startswith("/say "):
                    to_say = incoming_text.strip()[5:].strip()
                    if not to_say:
                        api.send_message(
                            chat_id,
                            "Uso: /say <texto>",
                            reply_to_message_id=message_id if message_id else None,
                        )
                        continue
                    if not cfg.voice_out_enabled:
                        api.send_message(
                            chat_id,
                            "Voice-out deshabilitado (BOT_VOICE_OUT_ENABLED=0).",
                            reply_to_message_id=message_id if message_id else None,
                        )
                        continue
                    caption = to_say
                    if (cfg.tts_backend or "").strip().lower() == "openai" and not cfg.openai_api_key:
                        caption = "TTS no configurado (OPENAI_API_KEY faltante). Enviando tono de prueba.\n\n" + caption
                    _send_voice_note(
                        api=api,
                        cfg=cfg,
                        chat_id=chat_id,
                        reply_to_message_id=message_id,
                        tts=openai_tts,
                        speak_text=to_say,
                        caption_text=caption,
                    )
                    continue

                if incoming_text.strip() in ("/new", "/reset"):
                    thread_mgr.clear(chat_id)
                    _clear_persisted_thread_id(cfg, chat_id=chat_id)
                    api.send_message(
                        chat_id,
                        "OK. Next message will start a new Codex thread for this chat.",
                        reply_to_message_id=message_id if message_id else None,
                    )
                    continue

                if incoming_text.strip() == "/thread":
                    tid = thread_mgr.get(chat_id)
                    api.send_message(
                        chat_id,
                        f"thread_id={tid}" if tid else "No active thread yet. Send a message first.",
                        reply_to_message_id=message_id if message_id else None,
                    )
                    continue

                if incoming_text.strip() in ("/cancel", "/x") and cfg.strict_proxy:
                    had_running = tracker.cancel(chat_id)
                    msg_txt = "Cancel requested." if had_running else "Canceled queued jobs (no running job)."
                    api.send_message(chat_id, msg_txt, reply_to_message_id=message_id if message_id else None)
                    continue

                if incoming_text.strip() == "/status":
                    api.send_message(
                        chat_id,
                        _status_text_for_chat(
                            cfg,
                            chat_id=chat_id,
                            tracker=tracker,
                            jobs=jobs,
                            thread_mgr=thread_mgr,
                            orchestrator_queue=orchestrator_queue,
                        ),
                        reply_to_message_id=message_id if message_id else None,
                    )
                    continue

                # Image messages: photo or image document. Use caption as the prompt, or a default.
                if is_media and not incoming_text.strip():
                    photo = msg.get("photo")
                    document = msg.get("document")
                    caption = msg.get("caption")
                    if caption is not None and not isinstance(caption, str):
                        caption = None

                    kind = ""
                    file_id = ""
                    orig_name = ""
                    suffix = ""
                    mime = ""

                    if isinstance(photo, list) and photo:
                        # Pick the last entry (typically highest resolution).
                        best = photo[-1] if isinstance(photo[-1], dict) else None
                        if isinstance(best, dict) and isinstance(best.get("file_id"), str):
                            kind = "image"
                            file_id = best["file_id"]
                            orig_name = "photo.jpg"
                            suffix = ".jpg"
                    elif isinstance(document, dict) and isinstance(document.get("file_id"), str):
                        file_id = document["file_id"]
                        fn = document.get("file_name")
                        orig_name = fn if isinstance(fn, str) and fn else "document.bin"
                        mime = document.get("mime_type") if isinstance(document.get("mime_type"), str) else ""
                        if (mime and mime.startswith("image/")) or (
                            isinstance(fn, str) and fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                        ):
                            kind = "image"
                            ext = Path(orig_name).suffix
                            suffix = ext if ext else ".img"
                        else:
                            kind = "document"

                    if file_id and kind:
                        upload_dir = cfg.codex_workdir / ".codexbot_uploads"
                        upload_dir.mkdir(parents=True, exist_ok=True)

                        # Download to a stable, readable filename so Codex can open it from disk.
                        if kind == "image":
                            tmp_f = tempfile.NamedTemporaryFile(prefix="tg_", suffix=suffix or ".img", dir=str(upload_dir), delete=False)
                            img_path = Path(tmp_f.name)
                            tmp_f.close()
                            dest_path = img_path
                        else:
                            safe = _safe_filename(orig_name, fallback="document.bin")
                            dest_path = upload_dir / f"tg_doc_{chat_id}_{message_id}_{safe}"
                            if dest_path.exists():
                                # Avoid clobbering existing files; keep the original too.
                                dest_path = upload_dir / f"tg_doc_{chat_id}_{message_id}_{int(time.time())}_{safe}"

                        try:
                            info = api.get_file(file_id)
                            fp = info.get("file_path") if isinstance(info, dict) else None
                            if not isinstance(fp, str) or not fp:
                                raise RuntimeError("Telegram getFile did not return file_path")
                            api.download_file_to(file_path=fp, dest=dest_path, max_bytes=cfg.max_download_bytes)
                        except Exception as e:
                            dest_path.unlink(missing_ok=True)
                            api.send_message(
                                chat_id,
                                f"Failed to download {kind} from Telegram: {e}",
                                reply_to_message_id=message_id if message_id else None,
                            )
                            continue

                        if kind == "image":
                            prompt = (caption or "").strip() or "Describe esta imagen."
                            job = Job(
                                chat_id=chat_id,
                                reply_to_message_id=message_id,
                                user_text=prompt,
                                argv=["exec", prompt],
                                mode_hint=cfg.codex_default_mode,
                                epoch=0,
                                threaded=True,
                                image_paths=[dest_path],
                                upload_paths=[],
                                force_new_thread=False,
                            )
                        else:
                            safe = _safe_filename(orig_name, fallback="document.bin")
                            base = (caption or "").strip()
                            if base:
                                prompt = f"{base}\n\nArchivo: {safe}\nPath: {dest_path}"
                            else:
                                prompt = f"Lee este archivo y dime que contiene.\n\nArchivo: {safe}\nPath: {dest_path}"
                            job = Job(
                                chat_id=chat_id,
                                reply_to_message_id=message_id,
                                user_text=prompt,
                                argv=["exec", prompt],
                                mode_hint=cfg.codex_default_mode,
                                epoch=0,
                                threaded=True,
                                image_paths=[],
                                upload_paths=[dest_path],
                                force_new_thread=False,
                            )
                        ok, reason, epoch, q_after = tracker.try_mark_enqueued(chat_id, max_queued_per_chat=cfg.max_queued_per_chat)
                        if not ok:
                            if kind == "image":
                                dest_path.unlink(missing_ok=True)
                            api.send_message(chat_id, reason, reply_to_message_id=message_id if message_id else None)
                            continue

                        job = Job(
                            chat_id=job.chat_id,
                            reply_to_message_id=job.reply_to_message_id,
                            user_text=job.user_text,
                            argv=job.argv,
                            mode_hint=job.mode_hint,
                            epoch=epoch,
                            threaded=job.threaded,
                            image_paths=job.image_paths,
                            upload_paths=job.upload_paths,
                            force_new_thread=job.force_new_thread,
                        )
                        try:
                            jobs.put(job, block=False)
                            # Only send a queued message when the request will actually wait.
                            # This keeps chats clean for the common "run immediately" case.
                            if q_after > 1 or tracker.inflight(chat_id) > 0:
                                model, effort = _job_model_label(cfg, job.argv, chat_id=chat_id)
                                model_part = _format_model_for_display(model, effort)
                                api.send_message(
                                    chat_id,
                                    "Queued (%s, mode=%s, provider=%s, model=%s, queue_len=%d)."
                                    % (
                                        kind,
                                        job.mode_hint,
                                        cfg.codex_local_provider if cfg.codex_use_oss else "default",
                                        model_part,
                                        jobs.qsize(),
                                    ),
                                    reply_to_message_id=message_id if message_id else None,
                                )
                        except queue.Full:
                            tracker.on_dequeue(chat_id)
                            if kind == "image":
                                dest_path.unlink(missing_ok=True)
                            api.send_message(
                                chat_id,
                                "Queue is full; try again in a bit.",
                                reply_to_message_id=message_id if message_id else None,
                            )
                        continue

                if cfg.strict_proxy:
                    # Forward almost everything to Codex thread directly, but keep a few local commands.
                    raw = incoming_text.strip()
                    if not raw:
                        continue

                    local_exact = {
                        "/start",
                        "/help",
                        "/whoami",
                        "/login",
                        "/logout",
                        "/status",
                        "/restart",
                        "/permissions",
                        "/skills",
                        "/model",
                        "/voice",
                        "/effort",
                        "/setnotify",
                        "/agents",
                        "/daily",
                        "/brief",
                        "/snapshot",
                        "/approve",
                        "/emergency_stop",
                        "/emergency_resume",
                        "/pause",
                        "/resume",
                        "/ticket",
                        "/inbox",
                        "/runbooks",
                        "/reset_role",
                        "/synccommands",
                        "/cancel",
                        "/purge",
                        "/job",
                        "/botpermissions",
                        "/format",
                        "/example",
                    }
                    local_prefixes = (
                        "/model ",
                        "/effort ",
                        "/notify ",
                        "/login ",
                        "/skills ",
                        "/ro ",
                        "/rw ",
                        "/full ",
                        "/exec ",
                        "/review ",
                        "/codex ",
                        "/job ",
                        "/ticket ",
                        "/inbox ",
                        "/reset_role ",
                        "/snapshot ",
                        "/daily",
                        "/approve ",
                        "/pause ",
                        "/resume ",
                        "/cancel ",
                        "/purge ",
                    )

                    # CEO UX rule: plain text should still go through our parser so we can
                    # handle greetings and keep orchestrator role profiles in control.
                    if not raw.startswith("/"):
                        response, job = _parse_job(cfg, incoming)
                    elif raw in local_exact or any(raw.startswith(p) for p in local_prefixes):
                        response, job = _parse_job(cfg, incoming)
                    else:
                        # Unknown slash commands are forwarded to Codex (strict-proxy behavior).
                        # Keep legacy default out of the job when orchestrator is enabled so role
                        # profiles decide the effective mode.
                        response, job = "", Job(
                            chat_id=chat_id,
                            reply_to_message_id=message_id,
                            user_text=raw,
                            argv=["exec", raw],
                            mode_hint=("" if cfg.orchestrator_enabled else cfg.codex_default_mode),
                            epoch=0,
                            threaded=True,
                            image_paths=[],
                            upload_paths=[],
                            force_new_thread=False,
                        )
                else:
                    response, job = _parse_job(cfg, incoming)
                if response:
                    marker = _parse_orchestrator_marker(response)
                    if marker:
                        kind, payload = marker
                        if _send_orchestrator_marker_response(
                            kind=kind,
                            payload=payload,
                            cfg=cfg,
                            api=api,
                            chat_id=chat_id,
                            user_id=incoming.user_id if incoming else None,
                            reply_to_message_id=message_id if message_id else None,
                            orch_q=orchestrator_queue,
                            profiles=orchestrator_profiles,
                        ):
                            continue

                    if response.startswith("__login__:"):
                        payload = response[len("__login__:") :].strip()
                        # Expected: "<user> <pass...>"
                        try:
                            parts = shlex.split(payload)
                        except Exception:
                            parts = payload.split()
                        if len(parts) < 2:
                            api.send_message(chat_id, "Uso: /login <usuario> <password>", reply_to_message_id=message_id if message_id else None)
                            continue
                        user = parts[0]
                        pw = " ".join(parts[1:])
                        ok, msg_txt = _auth_login(cfg, chat_id=chat_id, username=user, password=pw)
                        api.send_message(chat_id, msg_txt, reply_to_message_id=message_id if message_id else None)
                        continue

                    if response == "__logout__":
                        _auth_logout(cfg, chat_id=chat_id)
                        api.send_message(chat_id, "OK. Logout.", reply_to_message_id=message_id if message_id else None)
                        continue

                    if response.startswith("__notify__:"):
                        try:
                            _, raw_chat_id, payload = response.split(":", 2)
                            target_chat_id = int(raw_chat_id)
                            api.send_message(target_chat_id, payload)
                            api.send_message(
                                chat_id,
                                f"Sent to notify_chat_id={target_chat_id}.",
                                reply_to_message_id=message_id if message_id else None,
                            )
                        except Exception:
                            LOG.exception("Failed to send notify message")
                            api.send_message(
                                chat_id,
                                "Failed to send notify message. Check logs.",
                                reply_to_message_id=message_id if message_id else None,
                            )
                        continue

                    if response == "__cancel__":
                        had_running = tracker.cancel(chat_id)
                        msg_txt = "Cancel requested." if had_running else "Canceled queued jobs (no running job)."
                        api.send_message(chat_id, msg_txt, reply_to_message_id=message_id if message_id else None)
                        continue

                    if response == "__synccommands__":
                        try:
                            _sync_telegram_command_suggestions(api, cfg)
                            command_suggestions_synced = True
                            api.send_message(
                                chat_id,
                                "OK. Comandos de Telegram sincronizados.",
                                reply_to_message_id=message_id if message_id else None,
                            )
                        except Exception as e:
                            LOG.exception("Failed to sync Telegram command suggestions from chat")
                            api.send_message(
                                chat_id,
                                f"No pude sincronizar comandos: {e}",
                                reply_to_message_id=message_id if message_id else None,
                            )
                        continue

                    if response == "__restart__":
                        api.send_message(
                            chat_id,
                            "Restarting Poncebot... (systemd should bring it back in a few seconds)",
                            reply_to_message_id=message_id if message_id else None,
                        )
                        stop_event.set()
                        return

                    api.send_message(chat_id, response, reply_to_message_id=message_id if message_id else None)

                if job is not None:
                    if incoming_prefer_voice_reply and not job.prefer_voice_reply:
                        job = Job(
                            chat_id=job.chat_id,
                            reply_to_message_id=job.reply_to_message_id,
                            user_text=job.user_text,
                            argv=job.argv,
                            mode_hint=job.mode_hint,
                            epoch=job.epoch,
                            threaded=job.threaded,
                            image_paths=job.image_paths,
                            upload_paths=job.upload_paths,
                            force_new_thread=job.force_new_thread,
                            prefer_voice_reply=True,
                        )
                    profile = _auth_effective_profile_name(cfg, chat_id=chat_id) if cfg.auth_enabled else ""
                    if profile:
                        # Enforce max_mode for /ro,/rw,/full and default messages.
                        if not _profile_allows_mode(cfg, profile_name=profile, requested=job.mode_hint):
                            api.send_message(
                                chat_id,
                                f"No permitido por tu perfil ({profile}). Modo solicitado={job.mode_hint}.",
                                reply_to_message_id=message_id if message_id else None,
                            )
                            continue

                    enqueued_to_orchestrator = False
                    if orchestrator_queue is not None and cfg.orchestrator_enabled:
                        try:
                            did_submit, orch_job_id = _submit_orchestrator_task(
                                cfg=cfg,
                                orch_q=orchestrator_queue,
                                profiles=orchestrator_profiles,
                                job=job,
                                user_id=user_id,
                            )
                        except Exception:
                            did_submit = False
                            orch_job_id = ""
                            LOG.exception("Failed to submit orchestrator task")
                        if did_submit:
                            enqueued_to_orchestrator = True
                            try:
                                card = _ticket_card_text(orchestrator_queue, ticket_id=orch_job_id)
                            except Exception:
                                card = f"Ticket {orch_job_id[:8]} queued. Track: /ticket {orch_job_id[:8]} | /agents"
                            mid = api.send_message(
                                chat_id,
                                card,
                                reply_to_message_id=message_id if message_id else None,
                            )
                            try:
                                if mid is not None and orchestrator_queue is not None:
                                    orchestrator_queue.update_trace(
                                        orch_job_id,
                                        ticket_card_message_id=int(mid),
                                        ticket_card_created_at=time.time(),
                                    )
                            except Exception:
                                pass

                    if enqueued_to_orchestrator:
                        continue

                    ok, reason, epoch, q_after = tracker.try_mark_enqueued(chat_id, max_queued_per_chat=cfg.max_queued_per_chat)
                    if not ok:
                        api.send_message(chat_id, reason, reply_to_message_id=message_id if message_id else None)
                        continue

                    job = Job(
                        chat_id=job.chat_id,
                        reply_to_message_id=job.reply_to_message_id,
                        user_text=job.user_text,
                        argv=job.argv,
                        mode_hint=job.mode_hint,
                        epoch=epoch,
                        threaded=job.threaded,
                        image_paths=job.image_paths,
                        upload_paths=job.upload_paths,
                        force_new_thread=job.force_new_thread,
                        prefer_voice_reply=job.prefer_voice_reply,
                    )
                    # Attach profile to job via a lightweight hack: embed in upload_paths? No.
                    # Instead, persist per-chat profile in auth session and re-resolve in worker.
                    try:
                        jobs.put(job, block=False)
                        LOG.debug("Enqueued job chat_id=%s epoch=%s mode=%s", chat_id, epoch, job.mode_hint)
                        # Only send a queued message when the request will actually wait.
                        if q_after > 1 or tracker.inflight(chat_id) > 0:
                            model, effort = _job_model_label(cfg, job.argv, chat_id=chat_id)
                            model_part = _format_model_for_display(model, effort)
                            api.send_message(
                                chat_id,
                                "Queued (mode=%s, provider=%s, model=%s, queue_len=%d)."
                                % (
                                    job.mode_hint,
                                    cfg.codex_local_provider if cfg.codex_use_oss else "default",
                                    model_part,
                                    jobs.qsize(),
                                ),
                                reply_to_message_id=message_id if message_id else None,
                            )
                    except queue.Full:
                        # Roll back per-chat queued count.
                        tracker.on_dequeue(chat_id)
                        api.send_message(
                            chat_id,
                            "Queue is full; try again in a bit.",
                            reply_to_message_id=message_id if message_id else None,
                        )

        except Exception:
            LOG.exception("Polling error; backing off")
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


def _load_config() -> BotConfig:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN")

    allowed_chat_ids = _parse_int_set(os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS"))
    allowed_user_ids = _parse_int_set(os.environ.get("TELEGRAM_ALLOWED_USER_IDS"))
    admin_user_ids = frozenset(_parse_int_set(os.environ.get("BOT_ADMIN_USER_IDS")))
    admin_chat_ids = frozenset(_parse_int_set(os.environ.get("BOT_ADMIN_CHAT_IDS")))

    unsafe_direct_codex = os.environ.get("BOT_UNSAFE_DIRECT_CODEX", "0").strip().lower() in ("1", "true", "yes", "on")

    poll_timeout_seconds = int(os.environ.get("BOT_POLL_TIMEOUT_SECONDS", "30"))
    http_timeout_seconds = int(os.environ.get("BOT_HTTP_TIMEOUT_SECONDS", "60"))
    http_max_retries = int(os.environ.get("BOT_HTTP_MAX_RETRIES", "3"))
    http_retry_initial_seconds = float(os.environ.get("BOT_HTTP_RETRY_INITIAL_SECONDS", "1"))
    http_retry_max_seconds = float(os.environ.get("BOT_HTTP_RETRY_MAX_SECONDS", "10"))
    unauthorized_reply_cooldown_seconds = int(os.environ.get("BOT_UNAUTHORIZED_REPLY_COOLDOWN_SECONDS", "600"))
    drain_updates_on_start = os.environ.get("BOT_DRAIN_UPDATES_ON_START", "1").strip().lower() in ("1", "true", "yes", "on")
    worker_count = int(os.environ.get("BOT_WORKERS", "1"))
    if worker_count < 1:
        worker_count = 1
    queue_maxsize = int(os.environ.get("BOT_QUEUE_MAXSIZE", "0"))
    if queue_maxsize < 0:
        queue_maxsize = 0
    max_queued_per_chat = int(os.environ.get("BOT_MAX_QUEUED_PER_CHAT", "1"))
    if max_queued_per_chat < 0:
        max_queued_per_chat = 0
    heartbeat_seconds = int(os.environ.get("BOT_HEARTBEAT_SECONDS", "60"))
    if heartbeat_seconds < 0:
        heartbeat_seconds = 0
    send_as_file_threshold_chars = int(os.environ.get("BOT_SEND_AS_FILE_THRESHOLD_CHARS", "12000"))
    if send_as_file_threshold_chars < 0:
        send_as_file_threshold_chars = 0
    strict_proxy = os.environ.get("BOT_STRICT_PROXY", "0").strip().lower() in ("1", "true", "yes", "on")
    max_download_bytes = int(os.environ.get("BOT_MAX_DOWNLOAD_BYTES", str(50 * 1024 * 1024)))
    if max_download_bytes < 0:
        max_download_bytes = 0

    transcribe_audio = os.environ.get("BOT_TRANSCRIBE_AUDIO", "0").strip().lower() in ("1", "true", "yes", "on")
    transcribe_backend = os.environ.get("BOT_TRANSCRIBE_BACKEND", "auto").strip().lower() or "auto"
    if transcribe_backend not in ("auto", "openai", "whispercpp"):
        transcribe_backend = "auto"
    transcribe_timeout_seconds = int(os.environ.get("BOT_TRANSCRIBE_TIMEOUT_SECONDS", "300"))
    if transcribe_timeout_seconds < 1:
        transcribe_timeout_seconds = 300

    here = Path(__file__).resolve().parent
    bin_dir = here / "bin"
    models_dir = here / "models"
    ffmpeg_default = str(bin_dir / "ffmpeg") if (bin_dir / "ffmpeg").exists() else "ffmpeg"
    whisper_default = (
        str(bin_dir / "whisper-cli")
        if (bin_dir / "whisper-cli").exists()
        else (str(bin_dir / "main") if (bin_dir / "main").exists() else "whisper-cli")
    )
    model_default = str(models_dir / "ggml-medium.bin")

    ffmpeg_bin = os.environ.get("BOT_TRANSCRIBE_FFMPEG_BIN", ffmpeg_default).strip() or ffmpeg_default
    whispercpp_bin = os.environ.get("BOT_TRANSCRIBE_WHISPERCPP_BIN", whisper_default).strip() or whisper_default
    whispercpp_model_path = os.environ.get("BOT_TRANSCRIBE_WHISPERCPP_MODEL_PATH", model_default).strip() or model_default
    whispercpp_threads = int(os.environ.get("BOT_TRANSCRIBE_WHISPERCPP_THREADS", "8"))
    if whispercpp_threads < 1:
        whispercpp_threads = 1

    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    openai_api_base_url = os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com").strip() or "https://api.openai.com"
    transcribe_model = os.environ.get("BOT_TRANSCRIBE_MODEL", "gpt-4o-transcribe").strip() or "gpt-4o-transcribe"
    transcribe_language = os.environ.get("BOT_TRANSCRIBE_LANGUAGE", "es").strip()
    transcribe_prompt = os.environ.get("BOT_TRANSCRIBE_PROMPT", "").strip()
    transcribe_max_bytes = int(os.environ.get("BOT_TRANSCRIBE_MAX_BYTES", str(25 * 1024 * 1024)))
    if transcribe_max_bytes < 0:
        transcribe_max_bytes = 0

    telegram_parse_mode = os.environ.get("BOT_TELEGRAM_PARSE_MODE", "HTML").strip()

    state_file = Path(os.environ.get("BOT_STATE_FILE", str(Path(__file__).with_name("state.json")))).expanduser().resolve()
    ceo_name = os.environ.get("BOT_CEO_NAME", "Alejandro Ponce").strip() or "Alejandro Ponce"
    auth_enabled = os.environ.get("BOT_AUTH_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
    auth_session_ttl_seconds = int(os.environ.get("BOT_AUTH_SESSION_TTL_SECONDS", str(12 * 60 * 60)))
    if auth_session_ttl_seconds < 60:
        auth_session_ttl_seconds = 12 * 60 * 60
    auth_users_file = Path(os.environ.get("BOT_AUTH_USERS_FILE", str(Path(__file__).with_name("users.json")))).expanduser().resolve()
    auth_profiles_file = Path(os.environ.get("BOT_AUTH_PROFILES_FILE", str(Path(__file__).with_name("profiles.json")))).expanduser().resolve()
    notify_on_start = os.environ.get("TELEGRAM_NOTIFY_ON_START", "0").strip().lower() in ("1", "true", "yes", "on")
    notify_chat_id_raw = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "").strip()
    notify_chat_id: int | None
    if notify_chat_id_raw:
        notify_chat_id = int(notify_chat_id_raw)
    else:
        notify_chat_id = None

    orchestrator_enabled = os.environ.get("BOT_ORCHESTRATOR_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
    orchestrator_db_path = Path(
        os.environ.get(
            "BOT_ORCHESTRATOR_DB_PATH",
            str(Path(__file__).with_name("data") / "jobs.sqlite"),
        )
    ).expanduser().resolve()
    orchestrator_default_priority = int(os.environ.get("BOT_ORCHESTRATOR_DEFAULT_PRIORITY", "2"))
    if orchestrator_default_priority < 1:
        orchestrator_default_priority = 1
    if orchestrator_default_priority > 3:
        orchestrator_default_priority = 3
    orchestrator_default_max_cost_window_usd = float(os.environ.get("BOT_ORCHESTRATOR_DEFAULT_MAX_COST_WINDOW_USD", "8.0"))
    if orchestrator_default_max_cost_window_usd <= 0:
        orchestrator_default_max_cost_window_usd = 8.0
    orchestrator_default_role = os.environ.get("BOT_ORCHESTRATOR_DEFAULT_ROLE", "jarvis").strip() or "jarvis"
    orchestrator_daily_digest_seconds = int(
        os.environ.get("BOT_ORCHESTRATOR_DAILY_DIGEST_SECONDS", str(6 * 60 * 60))
    )
    if orchestrator_daily_digest_seconds < 60:
        orchestrator_daily_digest_seconds = 0
    orchestrator_agent_profiles = Path(
        os.environ.get(
            "BOT_ORCHESTRATOR_AGENT_PROFILES",
            str(Path(__file__).with_name("orchestrator") / "agents.yaml"),
        )
    ).expanduser().resolve()
    orchestrator_worker_count = int(os.environ.get("BOT_ORCHESTRATOR_WORKERS", str(worker_count)))
    if orchestrator_worker_count < 1:
        orchestrator_worker_count = max(1, int(worker_count))
    orchestrator_sessions_enabled = os.environ.get("BOT_ORCH_SESSIONS_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
    orchestrator_live_update_seconds = int(os.environ.get("BOT_ORCHESTRATOR_LIVE_UPDATE_SECONDS", "8"))
    if orchestrator_live_update_seconds < 2:
        orchestrator_live_update_seconds = 2
    if orchestrator_live_update_seconds > 60:
        orchestrator_live_update_seconds = 60
    orchestrator_notify_mode = os.environ.get("BOT_ORCHESTRATOR_NOTIFY_MODE", "minimal").strip().lower() or "minimal"
    if orchestrator_notify_mode not in ("minimal", "verbose"):
        orchestrator_notify_mode = "minimal"

    worktree_root = Path(
        os.environ.get(
            "BOT_WORKTREE_ROOT",
            str(Path(__file__).with_name("data") / "worktrees"),
        )
    ).expanduser().resolve()
    artifacts_root = Path(
        os.environ.get(
            "BOT_ARTIFACTS_ROOT",
            str(Path(__file__).with_name("data") / "artifacts"),
        )
    ).expanduser().resolve()
    runbooks_enabled = os.environ.get("BOT_RUNBOOKS_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
    runbooks_path = Path(
        os.environ.get(
            "BOT_RUNBOOKS_PATH",
            str(Path(__file__).with_name("orchestrator") / "runbooks.yaml"),
        )
    ).expanduser().resolve()
    screenshot_enabled = os.environ.get("BOT_SCREENSHOT_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
    screenshot_allowed_hosts_raw = os.environ.get("BOT_SCREENSHOT_ALLOWED_HOSTS", "").strip()
    screenshot_allowed_hosts = frozenset(
        {h.strip().lower() for h in screenshot_allowed_hosts_raw.split(",") if h.strip()}
    )
    transcribe_async = os.environ.get("BOT_TRANSCRIBE_ASYNC", "1").strip().lower() in ("1", "true", "yes", "on")

    voice_out_enabled = os.environ.get("BOT_VOICE_OUT_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
    tts_backend = os.environ.get("BOT_TTS_BACKEND", "none").strip().lower() or "none"
    if tts_backend not in ("none", "piper", "openai", "tone"):
        tts_backend = "none"
    tts_max_chars = int(os.environ.get("BOT_TTS_MAX_CHARS", "600"))
    if tts_max_chars < 0:
        tts_max_chars = 0
    try:
        tts_voice_pitch_semitones = float(os.environ.get("BOT_TTS_VOICE_PITCH_SEMITONES", "0").strip() or "0")
    except Exception:
        tts_voice_pitch_semitones = 0.0
    if tts_voice_pitch_semitones > 12.0:
        tts_voice_pitch_semitones = 12.0
    if tts_voice_pitch_semitones < -12.0:
        tts_voice_pitch_semitones = -12.0
    tts_openai_model = os.environ.get("BOT_TTS_OPENAI_MODEL", "tts-1").strip() or "tts-1"
    tts_openai_voice = os.environ.get("BOT_TTS_OPENAI_VOICE", "alloy").strip() or "alloy"
    tts_openai_response_format = os.environ.get("BOT_TTS_OPENAI_RESPONSE_FORMAT", "mp3").strip().lower() or "mp3"
    piper_default = str(bin_dir / "piper" / "piper") if (bin_dir / "piper" / "piper").exists() else "piper"
    piper_model_default = ""
    # Prefer Mexico/US-neutral Spanish voices if present.
    if (models_dir / "piper" / "es_MX-claude-high.onnx").exists():
        piper_model_default = str(models_dir / "piper" / "es_MX-claude-high.onnx")
    elif (models_dir / "piper" / "es_MX-ald-medium.onnx").exists():
        piper_model_default = str(models_dir / "piper" / "es_MX-ald-medium.onnx")
    # Fallback: Spain voices (often sound more "male" to some users).
    elif (models_dir / "piper" / "es_ES-carlfm-x_low.onnx").exists():
        piper_model_default = str(models_dir / "piper" / "es_ES-carlfm-x_low.onnx")
    elif (models_dir / "piper" / "es_ES-davefx-medium.onnx").exists():
        piper_model_default = str(models_dir / "piper" / "es_ES-davefx-medium.onnx")
    tts_piper_bin = os.environ.get("BOT_TTS_PIPER_BIN", piper_default).strip() or piper_default
    tts_piper_model_path = os.environ.get("BOT_TTS_PIPER_MODEL_PATH", piper_model_default).strip() or piper_model_default
    tts_piper_speaker = os.environ.get("BOT_TTS_PIPER_SPEAKER", "").strip()
    try:
        tts_piper_noise_scale = float(os.environ.get("BOT_TTS_PIPER_NOISE_SCALE", "0").strip() or "0")
    except Exception:
        tts_piper_noise_scale = 0.0
    try:
        tts_piper_length_scale = float(os.environ.get("BOT_TTS_PIPER_LENGTH_SCALE", "0").strip() or "0")
    except Exception:
        tts_piper_length_scale = 0.0
    try:
        tts_piper_noise_w = float(os.environ.get("BOT_TTS_PIPER_NOISE_W", "0").strip() or "0")
    except Exception:
        tts_piper_noise_w = 0.0
    try:
        tts_piper_sentence_silence = float(os.environ.get("BOT_TTS_PIPER_SENTENCE_SILENCE", "0").strip() or "0")
    except Exception:
        tts_piper_sentence_silence = 0.0

    codex_workdir = Path(os.environ.get("CODEX_WORKDIR", os.getcwd())).expanduser().resolve()
    if not codex_workdir.exists() or not codex_workdir.is_dir():
        raise SystemExit(f"CODEX_WORKDIR must be an existing directory: {codex_workdir}")
    codex_timeout_seconds = int(os.environ.get("CODEX_TIMEOUT_SECONDS", "900"))
    codex_use_oss = os.environ.get("CODEX_USE_OSS", "0").strip().lower() not in ("0", "false", "no", "off")
    codex_local_provider = os.environ.get("CODEX_LOCAL_PROVIDER", "ollama").strip() or "ollama"
    # Back-compat: CODEX_MODEL refers to OSS model when CODEX_USE_OSS=1.
    codex_oss_model = os.environ.get("CODEX_OSS_MODEL", "").strip() or os.environ.get("CODEX_MODEL", "").strip()
    codex_openai_model = os.environ.get("CODEX_OPENAI_MODEL", "").strip()

    if codex_use_oss and not codex_oss_model:
        # Reasonable default for local coding tasks. Adjust to your hardware/preferences.
        codex_oss_model = "qwen2.5-coder:7b"
    if not codex_use_oss:
        # Avoid accidentally passing a local model name to the OpenAI provider.
        codex_oss_model = ""
    codex_default_mode = os.environ.get("CODEX_DEFAULT_MODE", "ro").strip().lower() or "ro"
    if codex_default_mode not in ("ro", "rw", "full"):
        raise SystemExit("CODEX_DEFAULT_MODE must be 'ro', 'rw', or 'full'")

    codex_force_full_access = os.environ.get("CODEX_FORCE_FULL_ACCESS", "0").strip().lower() in ("1", "true", "yes", "on")
    codex_dangerous_bypass_sandbox = os.environ.get("CODEX_DANGEROUS_BYPASS_SANDBOX", "0").strip().lower() in ("1", "true", "yes", "on")

    return BotConfig(
        telegram_token=token,
        allowed_chat_ids=allowed_chat_ids,
        allowed_user_ids=allowed_user_ids,
        unsafe_direct_codex=unsafe_direct_codex,
        poll_timeout_seconds=poll_timeout_seconds,
        http_timeout_seconds=http_timeout_seconds,
        http_max_retries=http_max_retries,
        http_retry_initial_seconds=http_retry_initial_seconds,
        http_retry_max_seconds=http_retry_max_seconds,
        unauthorized_reply_cooldown_seconds=unauthorized_reply_cooldown_seconds,
        drain_updates_on_start=drain_updates_on_start,
        worker_count=worker_count,
        queue_maxsize=queue_maxsize,
        max_queued_per_chat=max_queued_per_chat,
        heartbeat_seconds=heartbeat_seconds,
        send_as_file_threshold_chars=send_as_file_threshold_chars,
        max_download_bytes=max_download_bytes,
        strict_proxy=strict_proxy,
        transcribe_audio=transcribe_audio,
        transcribe_backend=transcribe_backend,
        transcribe_timeout_seconds=transcribe_timeout_seconds,
        ffmpeg_bin=ffmpeg_bin,
        whispercpp_bin=whispercpp_bin,
        whispercpp_model_path=whispercpp_model_path,
        whispercpp_threads=whispercpp_threads,
        openai_api_key=openai_api_key,
        openai_api_base_url=openai_api_base_url,
        transcribe_model=transcribe_model,
        transcribe_language=transcribe_language,
        transcribe_prompt=transcribe_prompt,
        transcribe_max_bytes=transcribe_max_bytes,
        state_file=state_file,
        notify_chat_id=notify_chat_id,
        notify_on_start=notify_on_start,
        codex_workdir=codex_workdir,
        codex_timeout_seconds=codex_timeout_seconds,
        codex_use_oss=codex_use_oss,
        codex_local_provider=codex_local_provider,
        codex_oss_model=codex_oss_model,
        codex_openai_model=codex_openai_model,
        codex_default_mode=codex_default_mode,
        codex_force_full_access=codex_force_full_access,
        codex_dangerous_bypass_sandbox=codex_dangerous_bypass_sandbox,
        telegram_parse_mode=telegram_parse_mode,
        auth_enabled=auth_enabled,
        auth_session_ttl_seconds=auth_session_ttl_seconds,
        auth_users_file=auth_users_file,
        auth_profiles_file=auth_profiles_file,
        orchestrator_db_path=orchestrator_db_path,
        orchestrator_enabled=orchestrator_enabled,
        orchestrator_default_priority=orchestrator_default_priority,
        orchestrator_default_max_cost_window_usd=orchestrator_default_max_cost_window_usd,
        orchestrator_default_role=orchestrator_default_role,
        orchestrator_daily_digest_seconds=orchestrator_daily_digest_seconds,
        orchestrator_agent_profiles=orchestrator_agent_profiles,
        orchestrator_worker_count=orchestrator_worker_count,
        orchestrator_sessions_enabled=orchestrator_sessions_enabled,
        orchestrator_live_update_seconds=orchestrator_live_update_seconds,
        orchestrator_notify_mode=orchestrator_notify_mode,
        worktree_root=worktree_root,
        artifacts_root=artifacts_root,
        runbooks_enabled=runbooks_enabled,
        runbooks_path=runbooks_path,
        screenshot_enabled=screenshot_enabled,
        screenshot_allowed_hosts=screenshot_allowed_hosts,
        transcribe_async=transcribe_async,
        voice_out_enabled=voice_out_enabled,
        tts_backend=tts_backend,
        tts_max_chars=tts_max_chars,
        tts_voice_pitch_semitones=tts_voice_pitch_semitones,
        tts_openai_model=tts_openai_model,
        tts_openai_voice=tts_openai_voice,
        tts_openai_response_format=tts_openai_response_format,
        tts_piper_bin=tts_piper_bin,
        tts_piper_model_path=tts_piper_model_path,
        tts_piper_speaker=tts_piper_speaker,
        tts_piper_noise_scale=tts_piper_noise_scale,
        tts_piper_length_scale=tts_piper_length_scale,
        tts_piper_noise_w=tts_piper_noise_w,
        tts_piper_sentence_silence=tts_piper_sentence_silence,
        ceo_name=ceo_name,
        admin_user_ids=admin_user_ids,
        admin_chat_ids=admin_chat_ids,
    )


def _drain_pending_updates(cfg: BotConfig, api: TelegramAPI) -> int:
    """
    Returns the next offset to use.
    This intentionally discards any pending updates so a restart doesn't re-run old commands.
    """
    offset = 0
    drained = 0
    for _ in range(20):
        try:
            updates = api.get_updates(offset=offset, timeout_seconds=0)
        except Exception:
            # On boot, DNS/network can be temporarily unavailable. Draining is a convenience, not a requirement.
            # If we crash here, systemd may hit StartLimit* and leave the bot "off" until manually restarted.
            LOG.exception("Failed to drain pending Telegram updates; continuing without drain")
            break
        if not updates:
            break
        for upd in updates:
            update_id = int(upd.get("update_id", -1))
            if update_id >= 0:
                offset = max(offset, update_id + 1)
        drained += len(updates)
    if drained:
        LOG.info("Drained %d pending Telegram updates; next offset=%d", drained, offset)
    return offset


def _configured_notify_chat_id(cfg: BotConfig) -> int | None:
    if cfg.notify_chat_id is not None:
        return cfg.notify_chat_id
    state = _read_json(cfg.state_file)
    try:
        raw = state.get("notify_chat_id")
        if raw is None:
            return None
        return int(raw)
    except Exception:
        return None


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = _load_config()
    api = TelegramAPI(
        cfg.telegram_token,
        http_timeout_seconds=cfg.http_timeout_seconds,
        http_max_retries=cfg.http_max_retries,
        http_retry_initial_seconds=cfg.http_retry_initial_seconds,
        http_retry_max_seconds=cfg.http_retry_max_seconds,
        parse_mode=cfg.telegram_parse_mode,
    )
    orchestrator_queue: OrchestratorQueue | None = None
    orchestrator_profiles: dict[str, dict[str, Any]] | None = None
    orchestrator_scheduler: OrchestratorScheduler | None = None
    runbooks_scheduler: OrchestratorScheduler | None = None
    jobs: "queue.Queue[Job]" = queue.Queue(maxsize=cfg.queue_maxsize)
    stop_event = threading.Event()
    tracker = JobTracker()
    thread_mgr = ThreadManager()

    if cfg.orchestrator_enabled:
        try:
            orchestrator_profiles = load_agent_profiles(cfg.orchestrator_agent_profiles)
            # Render small placeholders for prompts (e.g. {CEO_NAME}).
            try:
                orchestrator_profiles = _render_placeholders_obj(orchestrator_profiles, ceo_name=cfg.ceo_name)
            except Exception:
                pass
            orch_storage = SQLiteTaskStorage(cfg.orchestrator_db_path)
            orchestrator_queue = OrchestratorQueue(storage=orch_storage, role_profiles=orchestrator_profiles)
            recovered = orchestrator_queue.recover_stale_running()
            if recovered:
                LOG.info("Recovered %d stale orchestrator jobs to queued state.", recovered)
        except Exception:
            LOG.exception("Failed to initialize orchestrator storage/queue; disabling orchestrator for this session.")
            orchestrator_queue = None

    if orchestrator_queue is not None:
        # Optional localhost status API (snapshot + SSE stream). Disabled by default.
        if os.environ.get("BOT_STATUS_HTTP_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on"):
            listen = os.environ.get("BOT_STATUS_HTTP_LISTEN", "127.0.0.1:8090").strip() or "127.0.0.1:8090"
            host = "127.0.0.1"
            port = 8090
            if ":" in listen:
                h, p = listen.rsplit(":", 1)
                if h.strip():
                    host = h.strip()
                try:
                    port = int(p.strip())
                except Exception:
                    port = 8090
            else:
                try:
                    port = int(listen)
                except Exception:
                    port = 8090
            try:
                ttl_s = int(os.environ.get("BOT_STATUS_CACHE_TTL_SECONDS", "2"))
            except Exception:
                ttl_s = 2
            try:
                interval_s = float(os.environ.get("BOT_STATUS_STREAM_INTERVAL_SECONDS", "1.0"))
            except Exception:
                interval_s = 1.0

            try:
                svc = StatusService(orch_q=orchestrator_queue, role_profiles=orchestrator_profiles, cache_ttl_seconds=ttl_s)
                auth_token = os.environ.get("BOT_STATUS_HTTP_TOKEN", "").strip()
                try:
                    snap_rps = float(os.environ.get("BOT_STATUS_HTTP_SNAPSHOT_RPS", "2.0"))
                except Exception:
                    snap_rps = 2.0
                try:
                    snap_burst = float(os.environ.get("BOT_STATUS_HTTP_SNAPSHOT_BURST", "4.0"))
                except Exception:
                    snap_burst = 4.0
                try:
                    max_sse = int(os.environ.get("BOT_STATUS_HTTP_MAX_SSE_PER_IP", "2"))
                except Exception:
                    max_sse = 2
                http_srv = start_status_http_server(
                    host=host,
                    port=port,
                    status_service=svc,
                    stream_interval_s=interval_s,
                    auth_token=auth_token,
                    snapshot_rate_per_s=snap_rps,
                    snapshot_burst=snap_burst,
                    max_sse_per_ip=max_sse,
                )
                th = threading.Thread(target=http_srv.serve_forever, daemon=True, name="status-http")
                th.start()
                LOG.info("Status HTTP API enabled on http://%s:%s (snapshot=/api/status/snapshot stream=/api/status/stream)", host, port)
            except Exception:
                LOG.exception("Failed to start status HTTP server")

        for i in range(max(1, cfg.orchestrator_worker_count)):
            t = threading.Thread(
                target=orchestrator_worker_loop,
                kwargs={
                    "cfg": cfg,
                    "api": api,
                    "orch_q": orchestrator_queue,
                    "stop_event": stop_event,
                    "profiles": orchestrator_profiles,
                },
                daemon=True,
                name=f"orch-worker-{i+1}",
            )
            t.start()

        if cfg.orchestrator_daily_digest_seconds >= 60 and _configured_notify_chat_id(cfg):
            notify_chat_id = _configured_notify_chat_id(cfg)

            def _send_orchestrator_digest() -> None:
                if orchestrator_queue is None or notify_chat_id is None:
                    return
                try:
                    api.send_message(notify_chat_id, _orchestrator_daily_digest_text(orchestrator_queue))
                except Exception:
                    LOG.exception("Failed to send scheduled orchestrator digest")

            orchestrator_scheduler = OrchestratorScheduler(interval_seconds=cfg.orchestrator_daily_digest_seconds, enabled=True)
            orchestrator_scheduler.add_tick(_send_orchestrator_digest)
            orchestrator_scheduler.start()

        # Runbooks scheduler: enqueues autonomous tasks periodically to the notify chat.
        if cfg.runbooks_enabled and _configured_notify_chat_id(cfg):
            notify_chat_id = _configured_notify_chat_id(cfg)

            def _tick_runbooks() -> None:
                if orchestrator_queue is None:
                    return

                # /watch updates (every scheduler tick). Independent of runbooks and notify chat.
                try:
                    _tick_watch_messages(cfg=cfg, api=api, orch_q=orchestrator_queue)
                except Exception:
                    pass

                if notify_chat_id is None:
                    return
                if orchestrator_queue.is_paused_globally():
                    return
                try:
                    rbs = load_runbooks(cfg.runbooks_path)
                except Exception:
                    LOG.exception("Failed to load runbooks: %s", cfg.runbooks_path)
                    return
                if not rbs:
                    return
                # Render placeholders for runbook prompts (e.g. {CEO_NAME}).
                try:
                    rbs = [
                        Runbook(
                            runbook_id=rb.runbook_id,
                            role=rb.role,
                            interval_seconds=rb.interval_seconds,
                            prompt=_render_placeholders_text(rb.prompt, ceo_name=cfg.ceo_name),
                            mode_hint=rb.mode_hint,
                            priority=rb.priority,
                            enabled=rb.enabled,
                        )
                        for rb in rbs
                    ]
                except Exception:
                    pass
                now = time.time()
                for rb in rbs:
                    try:
                        last = orchestrator_queue.get_runbook_last_run(runbook_id=rb.runbook_id)
                        if not runbook_due(rb, last_run_at=last, now=now):
                            continue
                        # Autopilot is implemented in-process so it can read persisted orders and current queue state.
                        if rb.runbook_id == "jarvis_autopilot":
                            try:
                                _autopilot_tick(
                                    cfg=cfg,
                                    orch_q=orchestrator_queue,
                                    profiles=orchestrator_profiles,
                                    chat_id=int(notify_chat_id),
                                    now=now,
                                )
                            except Exception:
                                LOG.exception("Autopilot tick failed")
                            orchestrator_queue.set_runbook_last_run(runbook_id=rb.runbook_id, ts=now)
                            continue
                        t = runbook_to_task(rb, chat_id=int(notify_chat_id))
                        # Apply role profile defaults so autonomous tasks behave like the same "agents".
                        try:
                            rb_role = _coerce_orchestrator_role(t.role)
                            rb_profile = _orchestrator_profile(orchestrator_profiles, rb_role)
                            rb_model = _orchestrator_model_for_profile(cfg, rb_profile)
                            rb_effort = _orchestrator_effort_for_profile(rb_profile, cfg)
                            rb_requires_approval = bool(rb_profile.get("approval_required", False)) or (t.mode_hint == "full")
                            rb_trace = dict(t.trace)
                            rb_trace["profile_name"] = str(rb_profile.get("name") or rb_role)
                            rb_trace["profile_role"] = rb_role
                            rb_trace["max_runtime_seconds"] = int(rb_profile.get("max_runtime_seconds") or 0)
                            t = t.with_updates(
                                role=rb_role,
                                model=rb_model,
                                effort=rb_effort,
                                requires_approval=rb_requires_approval,
                                trace=rb_trace,
                            )
                        except Exception:
                            pass
                        if not (t.artifacts_dir or "").strip():
                            t = t.with_updates(artifacts_dir=str((cfg.artifacts_root / t.job_id).resolve()))
                        orchestrator_queue.submit_task(t)
                        orchestrator_queue.set_runbook_last_run(runbook_id=rb.runbook_id, ts=now)
                    except Exception:
                        LOG.exception("Failed to enqueue runbook=%s", rb.runbook_id)

            runbooks_scheduler = OrchestratorScheduler(interval_seconds=60, enabled=True)
            runbooks_scheduler.add_tick(_tick_runbooks)
            runbooks_scheduler.start(name="orchestrator-runbooks")
    try:
        for chat_id_str, tid in _get_threads_state(cfg).items():
            try:
                cid = int(chat_id_str)
            except Exception:
                continue
            thread_mgr.set(cid, tid)
    except Exception:
        LOG.exception("Failed to load persisted threads from state file")

    if cfg.auth_enabled:
        # Security: if the bot restarts (deploy, crash, manual restart), require /login again.
        # This matches the "session lasts 12h of inactivity" model but does not persist across restarts.
        try:
            _auth_clear_all_sessions(cfg)
        except Exception:
            LOG.exception("Failed to clear auth sessions on startup")

    LOG.info(
        "Starting codexbot. workdir=%s default_mode=%s provider=%s allowed_chat_ids=%s allowed_user_ids=%s",
        cfg.codex_workdir,
        cfg.codex_default_mode,
        cfg.codex_local_provider if cfg.codex_use_oss else "default",
        sorted(cfg.allowed_chat_ids),
        sorted(cfg.allowed_user_ids),
    )

    command_suggestions_synced = False
    try:
        _sync_telegram_command_suggestions(api, cfg)
        command_suggestions_synced = True
        LOG.info("Telegram command suggestions synced.")
    except Exception:
        LOG.exception("Failed to set Telegram command suggestions at startup; will retry in poll loop")

    for i in range(cfg.worker_count):
        t = threading.Thread(
            target=worker_loop,
            kwargs={"cfg": cfg, "api": api, "jobs": jobs, "tracker": tracker, "stop_event": stop_event, "thread_mgr": thread_mgr},
            daemon=True,
            name=f"worker-{i+1}",
        )
        t.start()

    if cfg.notify_on_start:
        target = _configured_notify_chat_id(cfg)
        if target:
            try:
                api.send_message(int(target), "Poncebot is online.")
            except Exception:
                LOG.exception("Failed to send startup notification")

    try:
        start_offset = _drain_pending_updates(cfg, api) if cfg.drain_updates_on_start else 0
        poll_loop(
            cfg=cfg,
            api=api,
            jobs=jobs,
            tracker=tracker,
            stop_event=stop_event,
            thread_mgr=thread_mgr,
            orchestrator_queue=orchestrator_queue,
            orchestrator_profiles=orchestrator_profiles,
            offset=start_offset,
            command_suggestions_synced=command_suggestions_synced,
        )
    except KeyboardInterrupt:
        LOG.info("Stopping (KeyboardInterrupt)")
    finally:
        stop_event.set()
        if orchestrator_scheduler is not None:
            orchestrator_scheduler.stop()
        if runbooks_scheduler is not None:
            runbooks_scheduler.stop()


if __name__ == "__main__":
    main()
