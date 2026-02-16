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

            # With token => 200 and version header
            req = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("X-ED-API-Version"), "v1")
                payload = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(payload.get("api_version"), "v1")
                self.assertEqual(payload.get("schema_version"), 1)

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

