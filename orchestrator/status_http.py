from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
import json
import threading
import time
import urllib.parse

from .status_service import StatusService


_API_VERSION = "v1"


def _parse_chat_id(qs: dict[str, list[str]]) -> int | None:
    raw = (qs.get("chat_id") or [""])[0].strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _parse_limit(qs: dict[str, list[str]], default: int, *, lo: int = 1, hi: int = 500) -> int:
    raw = (qs.get("limit") or [""])[0].strip()
    if not raw:
        return int(default)
    try:
        v = int(raw)
    except Exception:
        return int(default)
    return max(int(lo), min(int(hi), v))


def _parse_since_ts(qs: dict[str, list[str]]) -> float | None:
    raw = (qs.get("since_ts") or [""])[0].strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _extract_bearer_token(handler: BaseHTTPRequestHandler, qs: dict[str, list[str]]) -> str:
    # Prefer Authorization header: "Bearer <token>"
    auth = handler.headers.get("Authorization") or ""
    if isinstance(auth, str):
        a = auth.strip()
        if a.lower().startswith("bearer "):
            return a.split(None, 1)[1].strip()
    # Fallback: query param for constrained clients.
    raw = (qs.get("token") or [""])[0]
    return str(raw or "").strip()


class _TokenBucket:
    def __init__(self, *, rate_per_s: float, burst: float) -> None:
        self.rate_per_s = max(0.0, float(rate_per_s))
        self.burst = max(0.0, float(burst))
        self.tokens = self.burst
        self.last = time.time()

    def allow(self, *, cost: float = 1.0) -> bool:
        now = time.time()
        dt = max(0.0, now - self.last)
        self.last = now
        if self.rate_per_s > 0:
            self.tokens = min(self.burst, self.tokens + dt * self.rate_per_s)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        *,
        svc: StatusService,
        stream_interval_s: float,
        auth_token: str,
        snapshot_rate_per_s: float,
        snapshot_burst: float,
        max_sse_per_ip: int,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.status_service = svc
        self.stream_interval_s = max(0.25, float(stream_interval_s))
        self.auth_token = (auth_token or "").strip()

        self._rl_lock = threading.Lock()
        self._snap_buckets: dict[str, _TokenBucket] = {}
        self._snapshot_rate_per_s = max(0.0, float(snapshot_rate_per_s))
        self._snapshot_burst = max(1.0, float(snapshot_burst))

        self._sse_lock = threading.Lock()
        self._sse_by_ip: dict[str, int] = {}
        self._max_sse_per_ip = max(1, int(max_sse_per_ip))

    def allow_snapshot(self, ip: str) -> bool:
        if self._snapshot_rate_per_s <= 0:
            return True
        key = (ip or "").strip() or "unknown"
        with self._rl_lock:
            b = self._snap_buckets.get(key)
            if b is None:
                b = _TokenBucket(rate_per_s=self._snapshot_rate_per_s, burst=self._snapshot_burst)
                self._snap_buckets[key] = b
            return b.allow(cost=1.0)

    def sse_acquire(self, ip: str) -> bool:
        key = (ip or "").strip() or "unknown"
        with self._sse_lock:
            n = int(self._sse_by_ip.get(key, 0))
            if n >= self._max_sse_per_ip:
                return False
            self._sse_by_ip[key] = n + 1
            return True

    def sse_release(self, ip: str) -> None:
        key = (ip or "").strip() or "unknown"
        with self._sse_lock:
            n = int(self._sse_by_ip.get(key, 0))
            if n <= 1:
                self._sse_by_ip.pop(key, None)
            else:
                self._sse_by_ip[key] = n - 1


class StatusAPIHandler(BaseHTTPRequestHandler):
    server: _ThreadingHTTPServer  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
        # Keep server quiet by default; integrate with app logging if needed.
        return

    def _send_json(self, code: int, payload: dict[str, Any], *, extra_headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-ED-API-Version", _API_VERSION)
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(str(k), str(v))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # pragma: no cover
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("X-ED-API-Version", _API_VERSION)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path or "/"
        qs = urllib.parse.parse_qs(parsed.query or "")
        chat_id = _parse_chat_id(qs)
        ip = str(getattr(self, "client_address", ("unknown", 0))[0] or "unknown")

        # Lightweight token auth (optional, but recommended for Tailscale/mobile).
        if self.server.auth_token:
            tok = _extract_bearer_token(self, qs)
            if not tok or tok != self.server.auth_token:
                self._send_json(401, {"error": "unauthorized"})
                return

        if path == "/healthz":
            self._send_json(200, {"ok": True})
            return

        # Structured logs (dashboard/audit).
        if path in ("/api/v1/logs/decisions", "/api/logs/decisions"):
            if not self.server.allow_snapshot(ip):
                self._send_json(429, {"error": "rate_limited"}, extra_headers={"Retry-After": "1"})
                return
            order_id = (qs.get("order_id") or [""])[0].strip()
            if not order_id:
                self._send_json(400, {"error": "missing_order_id"})
                return
            limit = _parse_limit(qs, 50, hi=500)
            items = self.server.status_service.orch_q.list_decision_log(order_id=order_id, limit=limit)
            self._send_json(200, {"api_version": _API_VERSION, "order_id": order_id, "items": items})
            return

        if path in ("/api/v1/logs/delegation", "/api/logs/delegation"):
            if not self.server.allow_snapshot(ip):
                self._send_json(429, {"error": "rate_limited"}, extra_headers={"Retry-After": "1"})
                return
            root = (qs.get("root_ticket_id") or qs.get("ticket_id") or [""])[0].strip()
            if not root:
                self._send_json(400, {"error": "missing_root_ticket_id"})
                return
            limit = _parse_limit(qs, 200, hi=2000)
            items = self.server.status_service.orch_q.list_delegation_log(root_ticket_id=root, limit=limit)
            self._send_json(200, {"api_version": _API_VERSION, "root_ticket_id": root, "items": items})
            return

        if path in ("/api/v1/logs/activity", "/api/logs/activity"):
            if not self.server.allow_snapshot(ip):
                self._send_json(429, {"error": "rate_limited"}, extra_headers={"Retry-After": "1"})
                return
            role = (qs.get("role") or [""])[0].strip().lower() or None
            since_ts = _parse_since_ts(qs)
            limit = _parse_limit(qs, 200, hi=5000)
            items = self.server.status_service.orch_q.list_worker_activity(role=role, since_ts=since_ts, limit=limit)
            self._send_json(200, {"api_version": _API_VERSION, "role": role, "since_ts": since_ts, "items": items})
            return

        if path in ("/api/status/snapshot", f"/api/{_API_VERSION}/status/snapshot"):
            if not self.server.allow_snapshot(ip):
                self._send_json(429, {"error": "rate_limited"}, extra_headers={"Retry-After": "1"})
                return
            snap = self.server.status_service.snapshot(chat_id=chat_id)
            self._send_json(200, snap)
            return

        if path in ("/api/status/stream", f"/api/{_API_VERSION}/status/stream"):
            if not self.server.sse_acquire(ip):
                self._send_json(429, {"error": "too_many_streams"}, extra_headers={"Retry-After": "5"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-ED-API-Version", _API_VERSION)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            last_hash = ""
            try:
                snap = self.server.status_service.snapshot(chat_id=chat_id)
                last_hash = str(snap.get("snapshot_hash") or "")
                data = json.dumps(snap, ensure_ascii=False)
                eid = str(int(float(snap.get("generated_at") or time.time()) * 1000.0))
                self.wfile.write(f"event: snapshot\nid: {eid}\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                self.server.sse_release(ip)
                return

            try:
                while True:
                    try:
                        time.sleep(self.server.stream_interval_s)
                        snap = self.server.status_service.snapshot(chat_id=chat_id)
                        h = str(snap.get("snapshot_hash") or "")
                        if h and h == last_hash:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                            continue
                        last_hash = h
                        data = json.dumps(snap, ensure_ascii=False)
                        eid = str(int(float(snap.get("generated_at") or time.time()) * 1000.0))
                        self.wfile.write(f"event: snapshot\nid: {eid}\ndata: {data}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    except Exception:
                        return
            finally:
                self.server.sse_release(ip)

        self._send_json(404, {"error": "not_found", "path": path})


@dataclass(frozen=True)
class StatusHTTPServer:
    host: str
    port: int
    httpd: _ThreadingHTTPServer

    def serve_forever(self) -> None:
        self.httpd.serve_forever(poll_interval=0.25)

    def shutdown(self) -> None:
        try:
            self.httpd.shutdown()
        except Exception:
            pass
        try:
            self.httpd.server_close()
        except Exception:
            pass


def start_status_http_server(
    *,
    host: str,
    port: int,
    status_service: StatusService,
    stream_interval_s: float = 1.0,
    auth_token: str = "",
    snapshot_rate_per_s: float = 2.0,
    snapshot_burst: float = 4.0,
    max_sse_per_ip: int = 2,
) -> StatusHTTPServer:
    httpd = _ThreadingHTTPServer(
        (host, int(port)),
        StatusAPIHandler,
        svc=status_service,
        stream_interval_s=stream_interval_s,
        auth_token=auth_token,
        snapshot_rate_per_s=snapshot_rate_per_s,
        snapshot_burst=snapshot_burst,
        max_sse_per_ip=max_sse_per_ip,
    )
    actual_host, actual_port = httpd.server_address[0], int(httpd.server_address[1])
    return StatusHTTPServer(host=str(actual_host), port=actual_port, httpd=httpd)
