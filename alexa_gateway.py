#!/usr/bin/env python3
"""
Alexa Custom Skill gateway for PonceBot.

This service receives Alexa JSON requests, extracts the spoken text, submits it
to the existing PonceBot orchestrator queue, and returns an Alexa response fast.
It intentionally does not try to read Echo audio directly; Alexa owns the mic and
speaker path and sends us text.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html import escape as html_escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

import bot as poncebot
from orchestrator.agents import load_agent_profiles
from orchestrator.queue import OrchestratorQueue
from orchestrator.storage import SQLiteTaskStorage


LOG = logging.getLogger("poncebot.alexa")
_TERMINAL_STATES = {"done", "failed", "cancelled"}
_CERT_CACHE_LOCK = threading.Lock()
_CERT_CACHE: dict[str, tuple[float, str]] = {}


@dataclass(frozen=True)
class AlexaGatewayConfig:
    host: str
    port: int
    endpoint_path: str
    path_secret: str
    verify_signature: bool
    skill_id: str
    allowed_skill_ids: tuple[str, ...]
    timestamp_tolerance_seconds: int
    chat_id: int
    user_id: int
    wait_seconds: float
    max_speech_chars: int


class AlexaRequestError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = message


class AlexaGateway:
    def __init__(
        self,
        *,
        cfg: poncebot.BotConfig,
        orch_q: OrchestratorQueue,
        profiles: dict[str, dict[str, Any]] | None,
        gateway_cfg: AlexaGatewayConfig,
    ) -> None:
        self.cfg = cfg
        self.orch_q = orch_q
        self.profiles = profiles
        self.gateway_cfg = gateway_cfg

    def handle(self, envelope: dict[str, Any]) -> dict[str, Any]:
        req = envelope.get("request") if isinstance(envelope, dict) else None
        if not isinstance(req, dict):
            raise AlexaRequestError(400, "missing request")

        req_type = str(req.get("type") or "").strip()
        if req_type == "LaunchRequest":
            return _alexa_response(
                "Aqui servidor remoto, conectado a r530. Que necesitas?",
                reprompt="Dime que quieres revisar o pedir.",
                should_end_session=False,
            )

        if req_type == "SessionEndedRequest":
            return {"version": "1.0", "response": {"shouldEndSession": True}}

        if req_type != "IntentRequest":
            return _alexa_response(
                "No pude entender esa solicitud. Puedes decir: pregunta a servidor remoto.",
                reprompt="Que quieres revisar o pedir?",
                should_end_session=False,
            )

        intent = req.get("intent") if isinstance(req.get("intent"), dict) else {}
        intent_name = str(intent.get("name") or "").strip()

        if intent_name in ("AMAZON.CancelIntent", "AMAZON.StopIntent"):
            return _alexa_response("Listo.", should_end_session=True)

        if intent_name in ("AMAZON.HelpIntent", "AMAZON.NavigateHomeIntent"):
            return _alexa_response(
                "Puedes decir: pregunta a servidor remoto que tengo pendiente hoy.",
                reprompt="Que quieres revisar o pedir?",
                should_end_session=False,
            )

        if intent_name in ("AMAZON.FallbackIntent", ""):
            return _alexa_response(
                "No cache la instruccion. Intenta con: pregunta a servidor remoto.",
                reprompt="Dime que quieres revisar o pedir.",
                should_end_session=False,
            )

        if intent_name != "AskPonceIntent":
            return _alexa_response(
                "Ese intent todavia no esta conectado al servidor.",
                reprompt="Que quieres revisar o pedir?",
                should_end_session=False,
            )

        query = _extract_query(intent).strip()
        if not query:
            return _alexa_response(
                "Que quieres revisar o pedir?",
                reprompt="Dime que quieres revisar o pedir.",
                should_end_session=False,
            )

        job_id = self._submit_query(envelope=envelope, query=query)
        task = self._wait_for_terminal(job_id)
        if task is not None:
            trace = task.trace or {}
            result_status = str(trace.get("result_status") or "").strip().lower()
            summary = str(trace.get("result_summary") or "").strip()
            if task.state == "done" and result_status in ("", "ok") and summary:
                return _alexa_response(
                    _speech_text(summary, max_chars=self.gateway_cfg.max_speech_chars),
                    reprompt="Quieres pedirle algo mas al servidor?",
                    should_end_session=False,
                    card_title="Servidor Remoto",
                    card_text=summary,
                )
            if task.state in ("failed", "cancelled"):
                return _alexa_response(
                    "El servidor no pudo completar eso. Ya quedo registrado para revisarlo.",
                    should_end_session=False,
                )

        return _alexa_response(
            f"Listo, se lo pase al servidor. Ticket {job_id[:8]}.",
            reprompt="Puedes pedirme otra cosa para el servidor.",
            should_end_session=False,
        )

    def _submit_query(self, *, envelope: dict[str, Any], query: str) -> str:
        request = envelope.get("request") if isinstance(envelope.get("request"), dict) else {}
        session = envelope.get("session") if isinstance(envelope.get("session"), dict) else {}
        context = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
        system = context.get("System") if isinstance(context.get("System"), dict) else {}
        user = system.get("user") if isinstance(system.get("user"), dict) else {}

        request_id = str(request.get("requestId") or "").strip()
        source_message_id = _stable_int(request_id or str(time.time()))
        alexa_user_id = str(user.get("userId") or "").strip()
        alexa_session_id = str(session.get("sessionId") or "").strip()
        application_id = _application_id(envelope)

        job = poncebot.Job(
            chat_id=int(self.gateway_cfg.chat_id),
            reply_to_message_id=0,
            user_text=query,
            argv=["exec", query],
            mode_hint="",
            epoch=0,
            threaded=True,
            image_paths=[],
            upload_paths=[],
            force_new_thread=False,
            prefer_voice_reply=False,
        )

        submitted, job_id = poncebot._submit_orchestrator_task(
            cfg=self.cfg,
            orch_q=self.orch_q,
            profiles=self.profiles,
            job=job,
            user_id=int(self.gateway_cfg.user_id),
            source_message_id=source_message_id,
        )
        if not submitted or not job_id:
            raise AlexaRequestError(503, "orchestrator not available")

        try:
            self.orch_q.update_trace(
                job_id,
                source="alexa",
                alexa_request_id=request_id,
                alexa_session_id=alexa_session_id,
                alexa_user_id=alexa_user_id[:256],
                alexa_application_id=application_id,
                alexa_submitted_at=time.time(),
            )
            self.orch_q.append_audit_event(
                event_type="alexa.request_submitted",
                actor="alexa_gateway",
                details={
                    "job_id": job_id,
                    "request_id": request_id,
                    "application_id": application_id,
                },
            )
        except Exception:
            LOG.exception("Failed to annotate Alexa job %s", job_id)
        return job_id

    def _wait_for_terminal(self, job_id: str) -> Any | None:
        deadline = time.monotonic() + max(0.0, float(self.gateway_cfg.wait_seconds))
        while time.monotonic() < deadline:
            try:
                task = self.orch_q.get_job(job_id)
            except Exception:
                LOG.exception("Failed to read Alexa job %s", job_id)
                return None
            if task is not None and str(task.state or "").strip().lower() in _TERMINAL_STATES:
                return task
            time.sleep(0.25)
        return None


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        gateway: AlexaGateway,
        gateway_cfg: AlexaGatewayConfig,
    ) -> None:
        super().__init__(server_address, handler_cls)
        self.gateway = gateway
        self.gateway_cfg = gateway_cfg


class AlexaHandler(BaseHTTPRequestHandler):
    server: _ThreadingHTTPServer  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("", "/", "/health", "/healthz"):
            self._send_json(200, {"ok": True, "service": "poncebot-alexa"})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._check_path()
            raw = self._read_body()
            try:
                envelope = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise AlexaRequestError(400, f"invalid json: {exc}") from exc
            if not isinstance(envelope, dict):
                raise AlexaRequestError(400, "request body must be an object")

            self._check_application_id(envelope)
            if self.server.gateway_cfg.verify_signature:
                _verify_alexa_signature(
                    raw_body=raw,
                    headers=self.headers,
                    envelope=envelope,
                    tolerance_seconds=self.server.gateway_cfg.timestamp_tolerance_seconds,
                )

            response = self.server.gateway.handle(envelope)
            self._send_json(200, response)
        except AlexaRequestError as exc:
            LOG.warning("Alexa request rejected: %s", exc.message)
            self._send_json(exc.code, {"error": exc.message})
        except Exception as exc:
            LOG.exception("Alexa request failed")
            self._send_json(500, {"error": f"internal_error: {exc}"})

    def _read_body(self) -> bytes:
        raw_len = (self.headers.get("Content-Length") or "").strip()
        try:
            n = int(raw_len)
        except Exception:
            n = 0
        if n <= 0:
            raise AlexaRequestError(400, "empty request")
        if n > 256 * 1024:
            raise AlexaRequestError(413, "request too large")
        return self.rfile.read(n)

    def _check_path(self) -> None:
        cfg = self.server.gateway_cfg
        parsed = urllib.parse.urlparse(self.path)
        expected = cfg.endpoint_path.rstrip("/") or "/alexa"
        if cfg.path_secret:
            expected = expected.rstrip("/") + "/" + urllib.parse.quote(cfg.path_secret, safe="")
        if parsed.path.rstrip("/") != expected.rstrip("/"):
            raise AlexaRequestError(404, "not found")

    def _check_application_id(self, envelope: dict[str, Any]) -> None:
        allowed = tuple(
            sid.strip()
            for sid in self.server.gateway_cfg.allowed_skill_ids
            if str(sid or "").strip()
        )
        if not allowed:
            return
        actual = _application_id(envelope)
        if actual not in allowed:
            LOG.warning(
                "Alexa request applicationId mismatch actual=%s actual_sha=%s allowed=%s",
                actual or "(missing)",
                _short_sha(actual),
                ",".join(allowed),
            )
            raise AlexaRequestError(403, "invalid skill id")

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(code))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _application_id(envelope: dict[str, Any]) -> str:
    session = envelope.get("session") if isinstance(envelope.get("session"), dict) else {}
    app = session.get("application") if isinstance(session.get("application"), dict) else {}
    value = str(app.get("applicationId") or "").strip()
    if value:
        return value
    context = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    system = context.get("System") if isinstance(context.get("System"), dict) else {}
    app = system.get("application") if isinstance(system.get("application"), dict) else {}
    return str(app.get("applicationId") or "").strip()


def _extract_query(intent: dict[str, Any]) -> str:
    slots = intent.get("slots") if isinstance(intent.get("slots"), dict) else {}
    for name in ("query", "utterance", "text"):
        slot = slots.get(name) if isinstance(slots.get(name), dict) else {}
        value = slot.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _stable_int(value: str) -> int:
    digest = hashlib.sha1(str(value or "").encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:8], 16)


def _alexa_response(
    speech: str,
    *,
    reprompt: str | None = None,
    should_end_session: bool = False,
    card_title: str | None = None,
    card_text: str | None = None,
) -> dict[str, Any]:
    safe_speech = _ssml(_speech_text(speech, max_chars=900))
    response: dict[str, Any] = {
        "outputSpeech": {"type": "SSML", "ssml": safe_speech},
        "shouldEndSession": bool(should_end_session),
    }
    if reprompt:
        response["reprompt"] = {
            "outputSpeech": {"type": "SSML", "ssml": _ssml(_speech_text(reprompt, max_chars=500))}
        }
    if card_title and card_text:
        response["card"] = {
            "type": "Simple",
            "title": str(card_title)[:128],
            "content": str(card_text)[:7500],
        }
    return {"version": "1.0", "response": response}


def _ssml(text: str) -> str:
    return "<speak>" + html_escape(text, quote=False) + "</speak>"


def _speech_text(text: str, *, max_chars: int) -> str:
    s = str(text or "").strip()
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", s)
    s = re.sub(r"https?://\S+", " enlace ", s)
    s = re.sub(r"[*_#>~|]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = "Listo."
    if len(s) > max_chars:
        s = s[: max(0, max_chars - 20)].rstrip() + ". Te deje el resto por escrito."
    return s


def _verify_alexa_signature(
    *,
    raw_body: bytes,
    headers: Any,
    envelope: dict[str, Any],
    tolerance_seconds: int,
) -> None:
    _verify_timestamp(envelope, tolerance_seconds=tolerance_seconds)
    cert_url = str(headers.get("SignatureCertChainUrl") or "").strip()
    sig_b64 = str(headers.get("Signature") or "").strip()
    if not cert_url or not sig_b64:
        raise AlexaRequestError(401, "missing Alexa signature headers")
    if not _valid_alexa_cert_url(cert_url):
        raise AlexaRequestError(401, "invalid Alexa cert URL")
    pem = _download_cert(cert_url)
    if not _verify_cert_and_signature(pem=pem, signature_b64=sig_b64, raw_body=raw_body):
        raise AlexaRequestError(401, "invalid Alexa signature")


def _verify_timestamp(envelope: dict[str, Any], *, tolerance_seconds: int) -> None:
    request = envelope.get("request") if isinstance(envelope.get("request"), dict) else {}
    raw = str(request.get("timestamp") or "").strip()
    if not raw:
        raise AlexaRequestError(401, "missing Alexa timestamp")
    try:
        dt = _dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception as exc:
        raise AlexaRequestError(401, f"invalid Alexa timestamp: {exc}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    now = _dt.datetime.now(_dt.timezone.utc)
    delta = abs((now - dt.astimezone(_dt.timezone.utc)).total_seconds())
    if delta > max(1, int(tolerance_seconds)):
        raise AlexaRequestError(401, "stale Alexa request")


def _valid_alexa_cert_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    if parsed.scheme.lower() != "https":
        return False
    if (parsed.hostname or "").lower() != "s3.amazonaws.com":
        return False
    if parsed.port not in (None, 443):
        return False
    path = urllib.parse.unquote(parsed.path or "")
    return path.startswith("/echo.api/")


def _download_cert(url: str) -> str:
    now = time.time()
    with _CERT_CACHE_LOCK:
        cached = _CERT_CACHE.get(url)
        if cached and now - cached[0] < 3600:
            return cached[1]
    with urllib.request.urlopen(url, timeout=5) as resp:
        raw = resp.read(128 * 1024)
    pem = raw.decode("utf-8", errors="replace")
    if "BEGIN CERTIFICATE" not in pem:
        raise AlexaRequestError(401, "Alexa cert download did not return PEM")
    with _CERT_CACHE_LOCK:
        _CERT_CACHE[url] = (now, pem)
    return pem


def _verify_cert_and_signature(*, pem: str, signature_b64: str, raw_body: bytes) -> bool:
    certs = re.findall(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
        pem,
        flags=re.S,
    )
    if not certs:
        return False
    try:
        signature = base64.b64decode(signature_b64, validate=True)
    except Exception:
        return False

    ca_file = Path("/etc/ssl/certs/ca-certificates.crt")
    with tempfile.TemporaryDirectory(prefix="poncebot_alexa_verify_") as td:
        root = Path(td)
        leaf = root / "leaf.pem"
        chain = root / "chain.pem"
        pubkey = root / "pubkey.pem"
        sig = root / "signature.bin"
        body = root / "body.json"
        leaf.write_text(certs[0] + "\n", encoding="utf-8")
        chain.write_text("\n".join(certs[1:]) + "\n", encoding="utf-8")
        sig.write_bytes(signature)
        body.write_bytes(raw_body)

        verify_cmd = ["openssl", "verify"]
        if ca_file.exists():
            verify_cmd += ["-CAfile", str(ca_file)]
        if certs[1:]:
            verify_cmd += ["-untrusted", str(chain)]
        verify_cmd += ["-verify_hostname", "echo-api.amazon.com", str(leaf)]
        if not _run_ok(verify_cmd):
            return False
        if not _run_ok(["openssl", "x509", "-in", str(leaf), "-checkend", "0", "-noout"]):
            return False
        pub = subprocess.run(
            ["openssl", "x509", "-in", str(leaf), "-pubkey", "-noout"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if pub.returncode != 0 or not pub.stdout.strip():
            return False
        pubkey.write_text(pub.stdout, encoding="utf-8")
        return _run_ok(
            [
                "openssl",
                "dgst",
                "-sha1",
                "-verify",
                str(pubkey),
                "-signature",
                str(sig),
                str(body),
            ]
        )


def _run_ok(cmd: list[str]) -> bool:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        LOG.exception("Verification command failed: %s", cmd)
        return False
    if proc.returncode != 0:
        LOG.warning("Verification command rejected: %s stderr=%s stdout=%s", cmd, proc.stderr[:500], proc.stdout[:500])
        return False
    return True


def _parse_listen(raw: str) -> tuple[str, int]:
    listen = (raw or "").strip() or "127.0.0.1:8095"
    host = "127.0.0.1"
    port = 8095
    if ":" in listen:
        h, p = listen.rsplit(":", 1)
        host = h.strip() or host
        try:
            port = int(p.strip())
        except Exception:
            port = 8095
    else:
        try:
            port = int(listen)
        except Exception:
            port = 8095
    return host, port


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _parse_skill_ids(*values: str) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for part in re.split(r"[\s,;]+", str(value or "").strip()):
            sid = part.strip()
            if not sid or sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
    return tuple(out)


def _short_sha(value: str) -> str:
    if not value:
        return "missing"
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]


def _first_int(values: set[int]) -> int | None:
    if not values:
        return None
    return sorted(int(v) for v in values)[0]


def _gateway_config(cfg: poncebot.BotConfig) -> AlexaGatewayConfig:
    host, port = _parse_listen(os.environ.get("PONCEBOT_ALEXA_LISTEN", "127.0.0.1:8095"))
    endpoint_path = os.environ.get("PONCEBOT_ALEXA_ENDPOINT_PATH", "/alexa").strip() or "/alexa"
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path
    raw_chat = os.environ.get("PONCEBOT_ALEXA_CHAT_ID", "").strip()
    if raw_chat:
        chat_id = int(raw_chat)
    else:
        chat_id = cfg.notify_chat_id or _first_int(cfg.allowed_chat_ids) or _first_int(cfg.allowed_user_ids)
        if chat_id is None:
            raise SystemExit("Set PONCEBOT_ALEXA_CHAT_ID or TELEGRAM_NOTIFY_CHAT_ID for Alexa routing.")
    raw_user = os.environ.get("PONCEBOT_ALEXA_USER_ID", "").strip()
    user_id = int(raw_user) if raw_user else int(chat_id)
    wait_seconds = float(os.environ.get("PONCEBOT_ALEXA_WAIT_SECONDS", "5.5"))
    max_speech_chars = int(os.environ.get("PONCEBOT_ALEXA_MAX_SPEECH_CHARS", "900"))
    tolerance = int(os.environ.get("PONCEBOT_ALEXA_TIMESTAMP_TOLERANCE_SECONDS", "150"))
    primary_skill_id = os.environ.get("PONCEBOT_ALEXA_SKILL_ID", "").strip()
    allowed_skill_ids = _parse_skill_ids(
        primary_skill_id,
        os.environ.get("PONCEBOT_ALEXA_ALLOWED_SKILL_IDS", ""),
        os.environ.get("PONCEBOT_ALEXA_SKILL_IDS", ""),
    )
    return AlexaGatewayConfig(
        host=host,
        port=port,
        endpoint_path=endpoint_path,
        path_secret=os.environ.get("PONCEBOT_ALEXA_PATH_SECRET", "").strip(),
        verify_signature=_env_bool("PONCEBOT_ALEXA_VERIFY_SIGNATURE", True),
        skill_id=primary_skill_id,
        allowed_skill_ids=allowed_skill_ids,
        timestamp_tolerance_seconds=tolerance,
        chat_id=int(chat_id),
        user_id=int(user_id),
        wait_seconds=max(0.0, wait_seconds),
        max_speech_chars=max(120, max_speech_chars),
    )


def _load_gateway() -> tuple[AlexaGateway, AlexaGatewayConfig]:
    cfg = poncebot._load_config()
    if not cfg.orchestrator_enabled:
        raise SystemExit("BOT_ORCHESTRATOR_ENABLED=1 is required for Alexa gateway.")
    profiles = load_agent_profiles(cfg.orchestrator_agent_profiles)
    try:
        profiles = poncebot._render_placeholders_obj(profiles, ceo_name=cfg.ceo_name)
    except Exception:
        LOG.exception("Failed to render agent profile placeholders; continuing")
    storage = SQLiteTaskStorage(cfg.orchestrator_db_path)
    orch_q = OrchestratorQueue(storage=storage, role_profiles=profiles)
    gateway_cfg = _gateway_config(cfg)
    gateway = AlexaGateway(cfg=cfg, orch_q=orch_q, profiles=profiles, gateway_cfg=gateway_cfg)
    return gateway, gateway_cfg


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    gateway, gateway_cfg = _load_gateway()
    httpd = _ThreadingHTTPServer(
        (gateway_cfg.host, gateway_cfg.port),
        AlexaHandler,
        gateway=gateway,
        gateway_cfg=gateway_cfg,
    )
    path = gateway_cfg.endpoint_path.rstrip("/") or "/alexa"
    log_path = path
    if gateway_cfg.path_secret:
        path = path.rstrip("/") + "/" + urllib.parse.quote(gateway_cfg.path_secret, safe="")
        log_path = log_path.rstrip("/") + "/<secret>"
    LOG.info(
        "PonceBot Alexa gateway listening on http://%s:%s%s verify_signature=%s skill_ids=%s",
        gateway_cfg.host,
        gateway_cfg.port,
        log_path,
        gateway_cfg.verify_signature,
        ",".join(gateway_cfg.allowed_skill_ids) or "(not pinned)",
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
