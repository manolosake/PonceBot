import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools import proactive_3h_autofix_monitor as monitor


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            parent_job_id TEXT,
            role TEXT,
            state TEXT,
            mode_hint TEXT,
            blocked_reason TEXT,
            updated_at REAL,
            trace TEXT
        )
        """
    )
    conn.commit()
    return conn


class TestProactive3hAutofixMonitor(unittest.TestCase):
    def test_log_appends_lines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "monitor.log"
            monitor._log("first", path=log_path, a=1)
            monitor._log("second", path=log_path, b=2)
            content = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(content), 2)
            self.assertIn('"msg": "first"', content[0])
            self.assertIn('"msg": "second"', content[1])

    def test_promote_waiting_qa_to_rw(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            log_path = Path(td) / "monitor.log"
            conn = _make_db(db)
            order_id = "ord-promote"
            conn.execute(
                """
                INSERT INTO jobs(job_id, parent_job_id, role, state, mode_hint, blocked_reason, updated_at, trace)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                ("qa1", order_id, "qa", "waiting_deps", "ro", "dependencies_pending:x", 1.0, "{}"),
            )
            conn.execute(
                """
                INSERT INTO jobs(job_id, parent_job_id, role, state, mode_hint, blocked_reason, updated_at, trace)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                ("qa2", order_id, "qa", "waiting_deps", "rw", "dependencies_pending:y", 1.0, "{}"),
            )
            conn.commit()

            changed = monitor._promote_waiting_qa_to_rw(conn, order_id=order_id, log_path=log_path)
            conn.commit()
            self.assertEqual(changed, 1)

            row1 = conn.execute("SELECT mode_hint FROM jobs WHERE job_id='qa1'").fetchone()
            row2 = conn.execute("SELECT mode_hint FROM jobs WHERE job_id='qa2'").fetchone()
            self.assertEqual(str(row1["mode_hint"]), "rw")
            self.assertEqual(str(row2["mode_hint"]), "rw")
            conn.close()


if __name__ == "__main__":
    unittest.main()
