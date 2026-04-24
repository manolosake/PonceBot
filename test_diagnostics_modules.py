from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from orchestrator.diagnostics.close_gate import evaluate_gate, main as close_gate_main
from orchestrator.diagnostics.job_liveness import build_liveness_payload
from orchestrator.diagnostics.status_snapshot import build_snapshot


def _make_db(path: Path) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                state TEXT NOT NULL,
                parent_job_id TEXT,
                owner TEXT,
                input_text TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                due_at REAL,
                stalled_since REAL,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 0
            )
            """
        )
        rows = [
            ("queued-1", "skynet", "queued", None, "scheduler", "queued work", 100.0, 110.0, None, None),
            ("running-1", "sre", "running", None, "worker", "running work", 101.0, 111.0, None, None),
            ("done-1", "sre", "done", None, "worker", "closed work", 90.0, 99.0, None, None),
            ("paused-1", "skynet", "paused", None, "scheduler", "manual pause", 80.0, 81.0, None, None),
        ]
        con.executemany(
            """
            INSERT INTO jobs (
                job_id, role, state, parent_job_id, owner, input_text,
                created_at, updated_at, due_at, stalled_since
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


class DiagnosticsModuleTests(unittest.TestCase):
    def test_status_snapshot_counts_open_jobs_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            _make_db(db)

            payload = build_snapshot(db, now=200.0)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["open_jobs"], 2)
        self.assertEqual(set(payload["open_job_ids"]), {"queued-1", "running-1"})
        self.assertEqual(payload["state_counts"]["done"], 1)
        self.assertEqual(payload["state_counts"]["paused"], 1)
        self.assertEqual(payload["role_state_counts"]["skynet"]["queued"], 1)

    def test_job_liveness_returns_actionable_open_job_details(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            _make_db(db)

            payload = build_liveness_payload(db, now=200.0)

        self.assertEqual(payload["open_jobs_count"], 2)
        first = payload["open_jobs"][0]
        self.assertIn(first["state"], {"queued", "running"})
        self.assertIn("input_preview", first)
        self.assertIsInstance(first["idle_seconds"], int)

    def test_close_gate_passes_only_when_status_and_liveness_are_zero(self) -> None:
        ok, reasons = evaluate_gate(
            {"ok": True, "open_jobs": 0},
            {"ok": True, "open_jobs_count": 0, "open_jobs": []},
        )
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

        ok, reasons = evaluate_gate(
            {"ok": True, "open_jobs": 1},
            {"ok": True, "open_jobs_count": 1, "open_jobs": [{"job_id": "x"}]},
        )
        self.assertFalse(ok)
        self.assertIn("status snapshot has open_jobs=1", reasons)

    def test_close_gate_cli_writes_non_empty_result_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "status.json"
            jobs = root / "jobs.json"
            out = root / "close_gate.txt"
            status.write_text(json.dumps({"ok": True, "open_jobs": 0}), encoding="utf-8")
            jobs.write_text(json.dumps({"ok": True, "open_jobs_count": 0, "open_jobs": []}), encoding="utf-8")

            rc = close_gate_main(["--status", str(status), "--jobs", str(jobs), "--out", str(out)])

            self.assertEqual(rc, 0)
            self.assertGreater(out.stat().st_size, 0)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["gate"], "PASS")


if __name__ == "__main__":

    unittest.main()
