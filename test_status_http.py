from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
import tempfile
from pathlib import Path

from orchestrator.queue import OrchestratorQueue
from orchestrator.status_http import start_status_http_server
from orchestrator.status_service import StatusService
from orchestrator.storage import SQLiteTaskStorage


class TestStatusHTTP(unittest.TestCase):
    def test_snapshot_requires_token_when_configured_and_supports_v1_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend"}}, cache_ttl_seconds=0)

            http_srv = start_status_http_server(
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            )
            t = threading.Thread(target=http_srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)

            base = f"http://{http_srv.host}:{http_srv.port}"
            url = base + "/api/v1/status/snapshot"

            # No token => 401
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(url, timeout=2).read()
            self.assertEqual(ctx.exception.code, 401)

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
                self.assertEqual(payload.get("schema_version"), 1)

            http_srv.shutdown()

    def test_snapshot_supports_legacy_query_token_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend"}}, cache_ttl_seconds=0)

            http_srv = start_status_http_server(
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                allow_query_token=True,
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            )
            t = threading.Thread(target=http_srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)

            base = f"http://{http_srv.host}:{http_srv.port}"
            url = base + "/api/v1/status/snapshot?token=secret"

            with urllib.request.urlopen(url, timeout=2) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(payload.get("api_version"), "v1")

            http_srv.shutdown()

    def test_cors_allowlist_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend"}}, cache_ttl_seconds=0)

            http_srv = start_status_http_server(
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                allowed_origins=["https://allowed.example"],
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            )
            t = threading.Thread(target=http_srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)

            base = f"http://{http_srv.host}:{http_srv.port}"
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

            http_srv.shutdown()

    def test_stream_sends_snapshot_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend"}}, cache_ttl_seconds=0)

            http_srv = start_status_http_server(
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            )
            t = threading.Thread(target=http_srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)

            base = f"http://{http_srv.host}:{http_srv.port}"
            url = base + "/api/v1/status/stream"
            req = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                # Read a small chunk; should include "event: snapshot"
                data = resp.read(512).decode("utf-8", errors="replace")
                self.assertIn("event: snapshot", data)

            http_srv.shutdown()

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

            http_srv = start_status_http_server(
                host="127.0.0.1",
                port=0,
                status_service=svc,
                stream_interval_s=0.5,
                auth_token="secret",
                snapshot_rate_per_s=0.0,
                snapshot_burst=1.0,
                max_sse_per_ip=2,
            )
            t = threading.Thread(target=http_srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)

            base = f"http://{http_srv.host}:{http_srv.port}"
            headers = {"Authorization": "Bearer secret"}
            for suffix in ("/api/v1/status/alerts", "/api/v1/status/risks", "/api/v1/status/decisions"):
                req = urllib.request.Request(base + suffix, headers=headers)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    self.assertEqual(resp.status, 200)
                    payload = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(payload.get("api_version"), "v1")
                    self.assertIn("items", payload)
                    self.assertIsInstance(payload["items"], list)

            http_srv.shutdown()
