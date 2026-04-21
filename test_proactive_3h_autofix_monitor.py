import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import proactive_3h_autofix_monitor as monitor


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE ceo_orders (
            order_id TEXT PRIMARY KEY,
            chat_id INTEGER NOT NULL,
            title TEXT,
            body TEXT,
            status TEXT NOT NULL,
            phase TEXT NOT NULL,
            updated_at REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            role TEXT NOT NULL,
            input_text TEXT NOT NULL,
            request_type TEXT NOT NULL,
            priority INTEGER NOT NULL,
            model TEXT NOT NULL,
            effort TEXT NOT NULL,
            mode_hint TEXT NOT NULL,
            requires_approval INTEGER NOT NULL DEFAULT 0,
            max_cost_window_usd REAL NOT NULL DEFAULT 0.0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            due_at REAL,
            state TEXT NOT NULL,
            chat_id INTEGER NOT NULL,
            user_id INTEGER,
            reply_to_message_id INTEGER,
            is_autonomous INTEGER NOT NULL DEFAULT 0,
            parent_job_id TEXT,
            owner TEXT,
            depends_on TEXT NOT NULL DEFAULT '[]',
            ttl_seconds INTEGER,
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 0,
            labels TEXT NOT NULL DEFAULT '{}',
            requires_review INTEGER NOT NULL DEFAULT 0,
            artifacts_dir TEXT,
            blocked_reason TEXT,
            plan_revision INTEGER NOT NULL DEFAULT 0,
            stalled_since REAL,
            trace TEXT NOT NULL DEFAULT '{}'
        )
        """
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.2
        return self.now


class TestIdleNoOpenGuardrail(unittest.TestCase):
    def test_closes_order_when_no_open_jobs_and_terminal_children(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            _create_schema(conn)
            order_id = "209b02f5-a5cf-409d-a912-5873446ee9f0"
            conn.execute(
                "INSERT INTO ceo_orders(order_id, chat_id, title, body, status, phase, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (order_id, 10, "Proactive Sprint", "[proactive: lane]", "active", "executing", 1.0),
            )
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, source, role, input_text, request_type, priority, model, effort, mode_hint,
                    requires_approval, max_cost_window_usd, created_at, updated_at, due_at, state, chat_id,
                    user_id, reply_to_message_id, is_autonomous, parent_job_id, owner, depends_on, ttl_seconds,
                    retry_count, max_retries, labels, requires_review, artifacts_dir, blocked_reason, plan_revision,
                    stalled_since, trace
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "done-1",
                    "autopilot",
                    "backend",
                    "done child",
                    "task",
                    2,
                    "gpt-5",
                    "medium",
                    "rw",
                    0,
                    2.0,
                    -500.0,
                    -500.0,
                    None,
                    "done",
                    10,
                    None,
                    None,
                    1,
                    order_id,
                    "autofix_monitor",
                    "[]",
                    None,
                    0,
                    0,
                    "{}",
                    0,
                    None,
                    None,
                    0,
                    None,
                    "{}",
                ),
            )
            conn.commit()
            conn.close()

            with (
                mock.patch.object(monitor, "DB_PATH", db_path),
                mock.patch.object(monitor, "_trigger_autopilot", return_value=0),
                mock.patch.object(monitor.time, "time", side_effect=_Clock()),
                mock.patch.object(monitor.time, "sleep", return_value=None),
            ):
                rc = monitor.run_monitor(duration_s=1, interval_s=0, log_path=Path(td) / "monitor.log")
            self.assertEqual(rc, 0)

            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT status, phase FROM ceo_orders WHERE order_id=?", (order_id,)).fetchone()
            conn.close()
            self.assertEqual(str(row[0]), "completed")
            self.assertEqual(str(row[1]), "completed")

    def test_enqueues_fallback_when_no_open_jobs_and_no_terminal_children(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            _create_schema(conn)
            order_id = "209b02f5-a5cf-409d-a912-5873446ee9f0"
            conn.execute(
                "INSERT INTO ceo_orders(order_id, chat_id, title, body, status, phase, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (order_id, 10, "Proactive Sprint", "[proactive: lane]", "active", "executing", 1.0),
            )
            conn.commit()
            conn.close()

            with (
                mock.patch.object(monitor, "DB_PATH", db_path),
                mock.patch.object(monitor, "_trigger_autopilot", return_value=0),
                mock.patch.object(monitor.time, "time", side_effect=_Clock()),
                mock.patch.object(monitor.time, "sleep", return_value=None),
            ):
                rc = monitor.run_monitor(duration_s=1, interval_s=0, log_path=Path(td) / "monitor.log")
            self.assertEqual(rc, 0)

            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                "SELECT role, state FROM jobs WHERE parent_job_id=? ORDER BY created_at DESC LIMIT 1",
                (order_id,),
            ).fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(str(row[0]), "architect_local")
            self.assertEqual(str(row[1]), "queued")


if __name__ == "__main__":
    unittest.main()
