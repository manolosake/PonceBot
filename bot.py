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
from orchestrator.dispatcher import to_task
from orchestrator.delegation import parse_ceo_subtasks
from orchestrator.prompting import build_agent_prompt
from orchestrator.queue import OrchestratorQueue
from orchestrator.runbooks import Runbook, due as runbook_due, load_runbooks, to_task as runbook_to_task
from orchestrator.schemas.task import Task
from orchestrator.screenshot import Viewport, capture as capture_screenshot
from orchestrator.storage import SQLiteTaskStorage
from orchestrator.scheduler import OrchestratorScheduler
from orchestrator.runner import run_task as run_orchestrator_task
from orchestrator.workspaces import WorktreeLease, collect_git_artifacts, ensure_worktree_pool, prepare_clean_workspace

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


LOG = logging.getLogger("codexbot")


TELEGRAM_MSG_LIMIT = 4096


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SKILL_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


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
    orchestrator_default_role: str = "ceo"
    orchestrator_daily_digest_seconds: int = 6 * 60 * 60
    orchestrator_agent_profiles: Path = Path(__file__).with_name("orchestrator") / "agents.yaml"
    orchestrator_worker_count: int = 3
    orchestrator_sessions_enabled: bool = True
    worktree_root: Path = Path(__file__).with_name("data") / "worktrees"
    artifacts_root: Path = Path(__file__).with_name("data") / "artifacts"
    runbooks_enabled: bool = True
    runbooks_path: Path = Path(__file__).with_name("orchestrator") / "runbooks.yaml"
    screenshot_enabled: bool = False
    transcribe_async: bool = True


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
    ) -> None:
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
        self._request("sendMessage", payload)

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
        "- /job <id>           Show task/job status by id",
        "- /daily              Show orchestrator digest now",
        "- /brief              Alias rápido de resumen de estado ejecutivo",
        "- /approve <id>       Approve a blocked task",
        "- /emergency_stop     Stop all orchestrator tasks and pause all roles",
        "- /pause <role>       Pause role in orchestrator",
        "- /resume <role>      Resume role in orchestrator",
        "- /cancel <id>        Cancel orchestrator task by id",
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
        ("help", "Mostrar ayuda"),
        ("agents", "Estado del orquestador"),
        ("status", "Estado del bot/modelo"),
        ("s", "Alias de /status"),
        ("whoami", "Ver tus IDs"),
    ]
    if cfg.auth_enabled:
        cmds += [
            ("login", "Iniciar sesion"),
            ("logout", "Cerrar sesion"),
        ]
    cmds += [
        ("job", "Ver estado de tarea"),
        ("ticket", "Ver ticket/subtareas"),
        ("inbox", "Ver backlog por rol"),
        ("runbooks", "Ver runbooks"),
        ("reset_role", "Reset memoria de rol"),
        ("emergency_stop", "Parar orquestador"),
        ("emergency_resume", "Reanudar orquestador"),
        ("daily", "Resumen automático de estado"),
        ("brief", "Resumen ejecutivo corto"),
        ("snapshot", "Solicitar captura de UI"),
        ("approve", "Aprobar tarea bloqueada"),
        ("pause", "Pausar rol"),
        ("resume", "Reanudar rol"),
        ("new", "Nuevo hilo"),
        ("thread", "Ver thread actual"),
        ("cancel", "Cancelar (chat o por id)"),
        ("model", "Ver/cambiar modelo"),
        ("m", "Alias de /model"),
        ("voice", "Configurar voice"),
        ("v", "Alias de /voice"),
        ("permissions", "Permisos de Codex"),
        ("p", "Alias de /permissions"),
        ("skills", "Ver/administrar skills"),
        ("synccommands", "Re-sincronizar comandos"),
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


_ORCHESTRATOR_ROLES = ("ceo", "frontend", "backend", "qa", "sre")


def _coerce_orchestrator_role(value: str) -> str:
    role = (value or "").strip().lower()
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
    detected_role = to_task(job.user_text, context={"chat_id": job.chat_id, "default_role": cfg.orchestrator_default_role}).role
    role = _coerce_orchestrator_role(detected_role)
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
    if job.reply_to_message_id is not None:
        trace["reply_to_message_id"] = int(job.reply_to_message_id)

    raw_priority = cfg.orchestrator_default_priority
    try:
        priority = int(raw_priority)
    except Exception:
        priority = 2
    if priority < 1:
        priority = 1

    context = {
        "source": "telegram",
        "chat_id": job.chat_id,
        "user_id": user_id,
        "reply_to_message_id": job.reply_to_message_id,
        "model": model,
        "effort": effort,
        "role": role,
        "default_role": cfg.orchestrator_default_role,
        "priority": priority,
        "due_at": None,
        "mode_hint": mode_hint,
        "request_type": "task",
        "requires_approval": requires_approval,
        "max_cost_window_usd": float(cfg.orchestrator_default_max_cost_window_usd),
        "trace": trace,
    }

    # role can be overridden by explicit @role markers in the input text.
    task = to_task(job.user_text, context=context)
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
        txt = (job.user_text or "").strip()
        prefix = "@frontend Solicitud de snapshot:"
        if txt.lower().startswith(prefix.lower()):
            url = txt[len(prefix) :].strip().split(None, 1)[0].strip().strip(_PUNCT_TRIM)
            if url:
                trace["needs_screenshot"] = True
                trace["screenshot_url"] = url
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


def _send_orchestrator_marker_response(
    kind: str,
    payload: str,
    cfg: BotConfig,
    api: "TelegramAPI",
    chat_id: int,
    reply_to_message_id: int | None,
    orch_q: OrchestratorQueue | None,
    profiles: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """
    Returns True if marker was handled.
    """
    if kind == "agents":
        if orch_q is None:
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
            return True
        api.send_message(chat_id, _orchestrator_status_text(orch_q), reply_to_message_id=reply_to_message_id)
        return True

    if kind == "job":
        if not orch_q:
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not payload:
            api.send_message(chat_id, "Uso: /job <id>", reply_to_message_id=reply_to_message_id)
            return True
        task = orch_q.get_job(payload)
        api.send_message(chat_id, _orchestrator_job_text(task), reply_to_message_id=reply_to_message_id)
        return True

    if kind == "ticket":
        if not orch_q:
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not payload:
            api.send_message(chat_id, "Uso: /ticket <id>", reply_to_message_id=reply_to_message_id)
            return True
        api.send_message(chat_id, _orchestrator_ticket_text(orch_q, payload), reply_to_message_id=reply_to_message_id)
        return True

    if kind == "inbox":
        if not orch_q:
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
            return True
        role = (payload or "").strip().lower() or None
        if role is not None and role != "all" and not _orchestrator_role_is_valid(role, profiles or {}):
            roles = ", ".join(_orchestrator_known_roles(profiles or {}))
            api.send_message(chat_id, f"Rol invalido: {role}. Roles: {roles}", reply_to_message_id=reply_to_message_id)
            return True
        api.send_message(chat_id, _orchestrator_inbox_text(orch_q, role=None if role in (None, "all") else role), reply_to_message_id=reply_to_message_id)
        return True

    if kind == "runbooks":
        if not orch_q:
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
            return True
        if not _can_manage_orchestrator(cfg, chat_id=chat_id):
            api.send_message(chat_id, "No permitido: necesitas permisos de gestor.", reply_to_message_id=reply_to_message_id)
            return True
        api.send_message(chat_id, _orchestrator_runbooks_text(cfg, orch_q), reply_to_message_id=reply_to_message_id)
        return True

    if kind == "reset_role":
        if not orch_q:
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
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
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
            return True
        api.send_message(chat_id, _orchestrator_daily_digest_text(orch_q), reply_to_message_id=reply_to_message_id)
        return True

    if kind in ("pause", "resume"):
        if not orch_q:
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
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
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
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
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
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
            api.send_message(chat_id, "Orchestrator disabled.", reply_to_message_id=reply_to_message_id)
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

    return False


def _orchestrator_status_text(orch_q: OrchestratorQueue) -> str:
    health = orch_q.get_role_health()
    if not health:
        return "No orchestrator jobs yet."

    states = ("queued", "running", "blocked", "done", "failed", "cancelled")
    system_state = "paused" if orch_q.is_paused_globally() else "active"
    lines = ["Orchestrator role health:", f"system: {system_state}", ""]
    for role in sorted(health.keys()):
        vals = health.get(role, {})
        state_parts = [f"{s}={int(vals.get(s, 0))}" for s in states if vals.get(s) is not None]
        if not state_parts:
            state_parts = [f"{s}=0" for s in states]
        paused = int(vals.get("paused", 0))
        lines.append(f"- {role} ({'paused' if paused else 'active'}): " + ", ".join(state_parts))

    return "\n".join(lines)


def _orchestrator_daily_digest_text(orch_q: OrchestratorQueue) -> str:
    lines = ["Orchestrator digest", "=" * 19]
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
            "input_text:",
            task.input_text[:1200],
            f"trace: {task.trace}",
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
        lines.append(f"- {c.job_id[:8]} role={c.role} state={c.state} text={snippet}")
    if len(children) > 60:
        lines.append(f"- ... +{len(children) - 60} more")
    return "\n".join(lines)


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

    if text.startswith("/job "):
        job_id = _orch_job_id(text[len("/job ") :])
        if not job_id:
            return "Uso: /job <id>", None
        return _orch_marker("job", job_id), None

    if text == "/job":
        return "Uso: /job <id>", None

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
            mode_hint=cfg.codex_default_mode,
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
        return True, orch_q.submit_task(task)
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
        out_png = artifacts_dir / "snapshot.png"
        try:
            capture_screenshot(url, out_png, viewport=Viewport(width=1280, height=720))
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

    prompt = build_agent_prompt(task, profile=profile)

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
    try:
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
            time.sleep(0.25)

        try:
            proc.proc.wait(timeout=5)
        except Exception:
            pass

        if canceled:
            return {
                "status": "error",
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
) -> None:
    status = str(getattr(result, "status", "error"))
    summary = str(getattr(result, "summary", "")).strip() or "(no summary)"
    logs = str(getattr(result, "logs", ""))
    next_action = getattr(result, "next_action", None)
    artifacts = list(getattr(result, "artifacts", []) or [])

    header = f"Orchestrator task={task.job_id[:8]} role={task.role} status={status}"
    payload = [header, summary]
    if next_action:
        payload.append(f"next_action={next_action}")
    msg = "\n".join(payload)
    msg_chunks = _chunk_text(msg, limit=TELEGRAM_MSG_LIMIT - 64)
    for idx, ch in enumerate(msg_chunks, start=1):
        text = ch if len(msg_chunks) == 1 else f"[{idx}/{len(msg_chunks)}]\n{ch}"
        api.send_message(task.chat_id, text, reply_to_message_id=task.reply_to_message_id)

    if status != "ok" and logs:
        log_chunks = _chunk_text(logs, limit=TELEGRAM_MSG_LIMIT - 64)
        for idx, ch in enumerate(log_chunks, start=1):
            prefix = f"log[{idx}/{len(log_chunks)}]\n"
            api.send_message(task.chat_id, f"{prefix}{ch}", reply_to_message_id=task.reply_to_message_id)

    for raw in artifacts[:3]:
        p = Path(str(raw))
        if not p.exists():
            continue
        ext = p.suffix.lower()
        is_img = ext in (".png", ".jpg", ".jpeg", ".webp")
        if is_img:
            try:
                api.send_photo(task.chat_id, p, caption=p.name, reply_to_message_id=task.reply_to_message_id)
                continue
            except Exception:
                pass
        api.send_document(task.chat_id, p, filename=p.name, reply_to_message_id=task.reply_to_message_id)


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
            orch_q.update_state(task.job_id, "running")
            result = run_orchestrator_task(task, executor=executor, cfg=cfg)
            orch_state = str(result.status or "failed")
            if orch_state not in {"ok", "blocked", "done", "failed", "cancelled"}:
                orch_state = "failed"
            if orch_state == "ok":
                orch_state = "done"
            orch_q.update_state(task.job_id, orch_state)
            _send_orchestrator_result(api, task, result)

            # CEO delegation: when a top-level CEO ticket finishes, enqueue child jobs from structured output.
            try:
                if (
                    orch_state == "done"
                    and _coerce_orchestrator_role(task.role) == "ceo"
                    and not task.is_autonomous
                    and not task.parent_job_id
                ):
                    if orch_q.jobs_by_parent(parent_job_id=task.job_id, limit=1):
                        # Avoid duplicate delegation on retries/restarts.
                        specs = []
                    else:
                        specs = parse_ceo_subtasks(getattr(result, "structured_digest", None))
                    if specs:
                        # Cap to avoid runaway delegation.
                        specs = specs[:12]
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
                                parent_job_id=task.job_id,
                                depends_on=deps,
                                labels={"ticket": task.job_id, "kind": "subtask", "key": spec.key},
                                artifacts_dir=str((cfg.artifacts_root / key_to_job[spec.key]).resolve()),
                                trace=trace,
                                job_id=key_to_job[spec.key],
                            )
                            children.append(child)
                        if children:
                            orch_q.submit_batch(children)
                            lines = ["CEO delegation:", f"ticket={task.job_id[:8]} subtasks={len(children)}"]
                            for c in children[:12]:
                                lines.append(f"- {c.job_id[:8]} role={c.role} mode={c.mode_hint} deps={len(c.depends_on)}")
                            api.send_message(task.chat_id, "\n".join(lines), reply_to_message_id=task.reply_to_message_id)
            except Exception:
                LOG.exception("Failed to delegate CEO subtasks. job=%s", task.job_id)
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
            mode_hint=cfg.codex_default_mode,
            epoch=0,
            threaded=True,
            image_paths=[],
            upload_paths=[],
            force_new_thread=False,
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
        )
        try:
            jobs.put(job, block=False)
            if q_after > 1 or tracker.inflight(chat_id) > 0:
                api.send_message(
                    chat_id,
                    f"Queued (voice) (mode={job.mode_hint}, queue_len={jobs.qsize()}).",
                    reply_to_message_id=message_id if message_id else None,
                )
        except queue.Full:
            tracker.on_dequeue(chat_id)
            api.send_message(chat_id, "Queue is full; try again in a bit.", reply_to_message_id=message_id if message_id else None)


def worker_loop(
    *,
    cfg: BotConfig,
    api: TelegramAPI,
    jobs: "queue.Queue[Job]",
    tracker: JobTracker,
    stop_event: threading.Event,
    thread_mgr: ThreadManager,
) -> None:
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

            # If job used the global default mode, apply per-profile default mode.
            mode_hint = job.mode_hint
            if profile and mode_hint == cfg.codex_default_mode:
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

            header = f"exit={code} secs={secs:.1f} mode={job.mode_hint}"

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
                        api.send_message(job.chat_id, f"[{idx}/{len(chunks)}]\n{ch}", reply_to_message_id=job.reply_to_message_id)

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
                    "Incoming update_id=%s chat_id=%s user_id=%s message_id=%s kind=%s",
                    update_id,
                    chat_id,
                    user_id,
                    message_id,
                    "text" if incoming_text.strip() else "media/empty",
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
                    )

                    if raw in local_exact or any(raw.startswith(p) for p in local_prefixes):
                        response, job = _parse_job(cfg, incoming)
                    else:
                        response, job = "", Job(
                            chat_id=chat_id,
                            reply_to_message_id=message_id,
                            user_text=raw,
                            argv=["exec", raw],
                            mode_hint=cfg.codex_default_mode,
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
                            api.send_message(
                                chat_id,
                                (
                                    f"Queued to orchestrator: task={orch_job_id[:8]} "
                                    f"(mode={job.mode_hint}, queue_queued={orchestrator_queue.get_queued_count()})."
                                ),
                                reply_to_message_id=message_id if message_id else None,
                            )

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
    orchestrator_default_role = os.environ.get("BOT_ORCHESTRATOR_DEFAULT_ROLE", "ceo").strip() or "ceo"
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
    transcribe_async = os.environ.get("BOT_TRANSCRIBE_ASYNC", "1").strip().lower() in ("1", "true", "yes", "on")

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
        telegram_parse_mode=telegram_parse_mode,
        state_file=state_file,
        notify_chat_id=notify_chat_id,
        notify_on_start=notify_on_start,
        orchestrator_enabled=orchestrator_enabled,
        orchestrator_db_path=orchestrator_db_path,
        orchestrator_default_priority=orchestrator_default_priority,
        orchestrator_default_max_cost_window_usd=orchestrator_default_max_cost_window_usd,
        orchestrator_default_role=orchestrator_default_role,
        orchestrator_daily_digest_seconds=orchestrator_daily_digest_seconds,
        orchestrator_agent_profiles=orchestrator_agent_profiles,
        orchestrator_worker_count=orchestrator_worker_count,
        orchestrator_sessions_enabled=orchestrator_sessions_enabled,
        worktree_root=worktree_root,
        artifacts_root=artifacts_root,
        runbooks_enabled=runbooks_enabled,
        runbooks_path=runbooks_path,
        screenshot_enabled=screenshot_enabled,
        transcribe_async=transcribe_async,
        codex_workdir=codex_workdir,
        codex_timeout_seconds=codex_timeout_seconds,
        codex_use_oss=codex_use_oss,
        codex_local_provider=codex_local_provider,
        codex_oss_model=codex_oss_model,
        codex_openai_model=codex_openai_model,
        codex_default_mode=codex_default_mode,
        codex_force_full_access=codex_force_full_access,
        codex_dangerous_bypass_sandbox=codex_dangerous_bypass_sandbox,
        auth_enabled=auth_enabled,
        auth_session_ttl_seconds=auth_session_ttl_seconds,
        auth_users_file=auth_users_file,
        auth_profiles_file=auth_profiles_file,
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
            orch_storage = SQLiteTaskStorage(cfg.orchestrator_db_path)
            orchestrator_queue = OrchestratorQueue(storage=orch_storage, role_profiles=orchestrator_profiles)
            recovered = orchestrator_queue.recover_stale_running()
            if recovered:
                LOG.info("Recovered %d stale orchestrator jobs to queued state.", recovered)
        except Exception:
            LOG.exception("Failed to initialize orchestrator storage/queue; disabling orchestrator for this session.")
            orchestrator_queue = None

    if orchestrator_queue is not None:
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
                if orchestrator_queue is None or notify_chat_id is None:
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
                now = time.time()
                for rb in rbs:
                    try:
                        last = orchestrator_queue.get_runbook_last_run(runbook_id=rb.runbook_id)
                        if not runbook_due(rb, last_run_at=last, now=now):
                            continue
                        t = runbook_to_task(rb, chat_id=int(notify_chat_id))
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
