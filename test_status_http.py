from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.status_http import StatusAPIHandler, start_status_http_server
from orchestrator.status_service import StatusService
from orchestrator.storage import SQLiteTaskStorage


def _wait_for_server(base_url: str, headers: dict[str, str] | None = None, timeout_s: float = 2.0) -> None:
    url = base_url + "/api/v1/status/snapshot"
    deadline = time.monotonic() + timeout_s
    delay = 0.01
    while True:
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=0.2) as resp:
                if resp.status in (200, 401, 403, 404):
                    return
        except urllib.error.HTTPError as exc:
            if exc.code in (200, 401, 403, 404):
                return
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"status http server not ready within {timeout_s}s")
        time.sleep(delay)
        delay = min(delay * 1.5, 0.2)


@contextmanager
def _running_status_http_server(*, wait_headers: dict[str, str] | None = None, **server_kwargs: object):
    http_srv = start_status_http_server(**server_kwargs)
    thread = threading.Thread(target=http_srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://{http_srv.host}:{http_srv.port}"
    _wait_for_server(base, headers=wait_headers)
    try:
        yield http_srv, base
    finally:
        http_srv.shutdown()
        thread.join(timeout=2.0)
        server_close = getattr(http_srv, "server_close", None)
        if callable(server_close):
            server_close()
        if thread.is_alive():
            raise AssertionError("status http server thread did not terminate within 2.0s after shutdown")


class TestStatusHTTP(unittest.TestCase):
    def test_send_json_ignores_client_disconnect_errors(self) -> None:
        class _DisconnectingWriter:
            def __init__(self, exc: BaseException) -> None:
                self._exc = exc

            def write(self, _data: bytes) -> None:
                raise self._exc

        for exc in (BrokenPipeError(), ConnectionResetError()):
            handler = StatusAPIHandler.__new__(StatusAPIHandler)
            handler.send_response = mock.Mock()
            handler.send_header = mock.Mock()
            handler.end_headers = mock.Mock()
            handler.wfile = _DisconnectingWriter(exc)
            handler._cors_headers = {}
            handler._send_json(200, {"ok": True})

    def test_token_bucket_rate_limits_using_monotonic(self) -> None:
        from orchestrator import status_http

        with mock.patch("orchestrator.status_http.time.monotonic") as mono:
            mono.side_effect = [100.0, 100.0, 100.0, 101.0]
            bucket = status_http._TokenBucket(rate_per_s=1.0, burst=1.0)
            self.assertTrue(bucket.allow())
            self.assertFalse(bucket.allow())
            self.assertTrue(bucket.allow())

    def test_snapshot_requires_token_when_configured_and_supports_v1_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend"}}, cache_ttl_seconds=0)

            with _running_status_http_server(
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            ) as (_http_srv, base):
                url = base + "/api/v1/status/snapshot"

                # No token => 401
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(url, timeout=2).read()
                self.assertEqual(ctx.exception.code, 401)

                # Malformed Bearer header (no token) => 401 (no crash)
                bad_bearer = urllib.request.Request(url, headers={"Authorization": "Bearer "})
                with self.assertRaises(urllib.error.HTTPError) as bad_ctx:
                    urllib.request.urlopen(bad_bearer, timeout=2).read()
                self.assertEqual(bad_ctx.exception.code, 401)

                # Query token is deprecated and must be rejected
                with self.assertRaises(urllib.error.HTTPError) as ctx_q:
                    urllib.request.urlopen(url + "?token=secret", timeout=2).read()
                self.assertEqual(ctx_q.exception.code, 401)

                # With Bearer token => 200 and version header
                req = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(resp.headers.get("X-ED-API-Version"), "v1")
                    payload = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(payload.get("api_version"), "v1")
                    self.assertEqual(payload.get("schema_version"), 2)

    def test_snapshot_supports_legacy_query_token_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend"}}, cache_ttl_seconds=0)

            with _running_status_http_server(
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                allow_query_token=True,
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            ) as (_http_srv, base):
                url = base + "/api/v1/status/snapshot?token=secret"

                with urllib.request.urlopen(url, timeout=2) as resp:
                    self.assertEqual(resp.status, 200)
                    payload = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(payload.get("api_version"), "v1")

    def test_cors_allowlist_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend"}}, cache_ttl_seconds=0)

            with _running_status_http_server(
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                allowed_origins=["https://allowed.example"],
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            ) as (_http_srv, base):
                url = base + "/api/v1/status/snapshot"

                # Disallowed origin => 403
                bad_req = urllib.request.Request(
                    url,
                    headers={"Authorization": "Bearer secret", "Origin": "https://evil.example"},
                )
                with self.assertRaises(urllib.error.HTTPError) as bad_ctx:
                    urllib.request.urlopen(bad_req, timeout=2).read()
                self.assertEqual(bad_ctx.exception.code, 403)

                # Allowed origin => 200 + reflected CORS header
                good_req = urllib.request.Request(
                    url,
                    headers={"Authorization": "Bearer secret", "Origin": "https://allowed.example"},
                )
                with urllib.request.urlopen(good_req, timeout=2) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "https://allowed.example")

    def test_stream_sends_snapshot_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend"}}, cache_ttl_seconds=0)

            with _running_status_http_server(
                wait_headers={"Authorization": "Bearer secret"},
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            ) as (_http_srv, base):
                url = base + "/api/v1/status/stream"
                req = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    # Read a small chunk; should include "event: snapshot"
                    data = resp.read(512).decode("utf-8", errors="replace")
                    self.assertIn("event: snapshot", data)

    def test_alerts_risks_decisions_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})
            order_id = "44444444-4444-4444-4444-444444444444"
            q.upsert_order(order_id=order_id, chat_id=7, title="Order", body="Body", status="active", priority=2)
            q.append_decision_log(
                order_id=order_id,
                job_id="ffffffff-ffff-ffff-ffff-ffffffffffff",
                kind="manager_review",
                state="blocked",
                summary="Review requerida",
                next_action="Resolver decisión de alcance",
                details=None,
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend"}}, cache_ttl_seconds=0)

            with _running_status_http_server(
                wait_headers={"Authorization": "Bearer secret"},
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            ) as (_http_srv, base):
                headers = {"Authorization": "Bearer secret"}
                for suffix in ("/api/v1/status/alerts", "/api/v1/status/risks", "/api/v1/status/decisions"):
                    req = urllib.request.Request(base + suffix, headers=headers)
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        self.assertEqual(resp.status, 200)
                        payload = json.loads(resp.read().decode("utf-8"))
                        self.assertEqual(payload.get("api_version"), "v1")
                        self.assertIn("items", payload)
                        self.assertIsInstance(payload["items"], list)

    def test_overview_includes_factory_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})
            svc = StatusService(
                orch_q=q,
                role_profiles={"backend": {"role": "backend"}},
                cache_ttl_seconds=0,
                factory_snapshot_fn=lambda chat_id: {"factory": {"status": "soft_pause", "mode": "ceo-bounded"}},
            )

            with _running_status_http_server(
                wait_headers={"Authorization": "Bearer secret"},
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            ) as (_http_srv, base):
                req = urllib.request.Request(base + "/api/v1/orchestration/overview", headers={"Authorization": "Bearer secret"})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    self.assertEqual(resp.status, 200)
                    payload = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual((payload.get("factory") or {}).get("status"), "soft_pause")

    def test_order_evidence_endpoint_success_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = SQLiteTaskStorage(root / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"skynet": {"role": "skynet"}, "backend": {"role": "backend"}})
            order_id = "77777777-7777-7777-7777-777777777777"
            child_id = "88888888-8888-8888-8888-888888888888"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Evidence endpoint",
                body="Expose evidence packet.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="skynet",
                    input_text="Root order",
                    request_type="maintenance",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    is_autonomous=True,
                    trace={"proactive_slices_applied": 1},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="backend",
                    input_text="Child job",
                    request_type="task",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    is_autonomous=True,
                    parent_job_id=order_id,
                    artifacts_dir=str(root / "artifacts" / child_id),
                    trace={"result_artifacts": ["child-result.txt"]},
                    job_id=child_id,
                )
            )
            q.append_trace_event(
                order_id=order_id,
                job_id=child_id,
                agent_role="backend",
                event_type="job.done",
                severity="info",
                message="done",
            )
            q.append_decision_log(
                order_id=order_id,
                job_id=child_id,
                kind="qa",
                state="done",
                summary="Accepted",
                next_action=None,
                details=None,
            )
            svc = StatusService(orch_q=q, role_profiles={"skynet": {"role": "skynet"}, "backend": {"role": "backend"}}, cache_ttl_seconds=0)

            with _running_status_http_server(
                wait_headers={"Authorization": "Bearer secret"},
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            ) as (_http_srv, base):
                url = base + "/api/v1/orchestration/orders/evidence"

                with self.assertRaises(urllib.error.HTTPError) as unauth_ctx:
                    urllib.request.urlopen(url + f"?order_id={order_id}", timeout=2).read()
                self.assertEqual(unauth_ctx.exception.code, 401)

                headers = {"Authorization": "Bearer secret"}
                missing_req = urllib.request.Request(url, headers=headers)
                with self.assertRaises(urllib.error.HTTPError) as missing_ctx:
                    urllib.request.urlopen(missing_req, timeout=2).read()
                self.assertEqual(missing_ctx.exception.code, 400)
                self.assertEqual(json.loads(missing_ctx.exception.read().decode("utf-8"))["error"], "missing_order_id")

                not_found_req = urllib.request.Request(url + "?order_id=missing-order", headers=headers)
                with self.assertRaises(urllib.error.HTTPError) as not_found_ctx:
                    urllib.request.urlopen(not_found_req, timeout=2).read()
                self.assertEqual(not_found_ctx.exception.code, 404)
                self.assertEqual(json.loads(not_found_ctx.exception.read().decode("utf-8"))["error"], "order_not_found")

                req = urllib.request.Request(url + f"?order_id={order_id}", headers=headers)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    self.assertEqual(resp.status, 200)
                    payload = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(payload["api_version"], "v1")
                    self.assertEqual(payload["schema_version"], 1)
                    self.assertEqual(payload["order_id"], order_id)
                    self.assertEqual(payload["workflow"]["order_id"], order_id)
                    self.assertEqual(payload["children"][0]["job_id"], child_id)
                    self.assertEqual(payload["counts"]["traces"], 1)
                    self.assertEqual(payload["counts"]["decision_log"], 1)
                    artifact_paths = {str(a.get("path")) for a in payload["artifacts"] if a.get("path")}
                    self.assertIn(str(root / "artifacts" / child_id), artifact_paths)
                    self.assertIn("child-result.txt", artifact_paths)
