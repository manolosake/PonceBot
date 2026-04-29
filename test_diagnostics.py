import sqlite3
import tempfile
import unittest
from pathlib import Path

from orchestrator.diagnostics.common import build_liveness_payload, fetch_open_jobs


_IDLE_PROMPT = "Preflight guard and closure protocol for idle scheduler state (no open jobs)."


class TestDiagnosticsOpenJobs(unittest.TestCase):
    def _init_db(self, path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                parent_job_id TEXT,
                role TEXT,
                state TEXT,
                owner TEXT,
                input_text TEXT,
                created_at REAL,
                updated_at REAL,
                due_at REAL,
                stalled_since REAL,
                retry_count INTEGER,
                max_retries INTEGER
            )
            """
        )
        return conn

    def test_idle_no_open_jobs_self_check_is_not_counted_as_open_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            conn = self._init_db(db_path)
            conn.execute(
                "INSERT INTO jobs(job_id, role, state, input_text, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("idle-self", "sre", "running", _IDLE_PROMPT + "\nRequired commands:", 10.0, 10.0),
            )
            conn.commit()
            conn.close()

            payload = build_liveness_payload(db_path, now=20.0)

            self.assertEqual(payload["open_jobs_count"], 0)
            self.assertEqual(payload["open_jobs"], [])

    def test_real_open_jobs_are_still_reported(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                role TEXT,
                state TEXT,
                input_text TEXT,
                created_at REAL,
                updated_at REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO jobs(job_id, role, state, input_text, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("idle-self", "sre", "running", _IDLE_PROMPT, 10.0, 10.0),
        )
        conn.execute(
            "INSERT INTO jobs(job_id, role, state, input_text, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("real-job", "backend", "queued", "Implement a real feature.", 11.0, 11.0),
        )

        jobs = fetch_open_jobs(conn, now=20.0)

        self.assertEqual([job["job_id"] for job in jobs], ["real-job"])


if __name__ == "__main__":
    unittest.main()
