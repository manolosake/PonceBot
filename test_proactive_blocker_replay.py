import sqlite3
import time
import unittest

from tools import proactive_blocker_replay as pbr


def _mk_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            role TEXT,
            state TEXT,
            depends_on TEXT,
            blocked_reason TEXT,
            updated_at REAL,
            created_at REAL,
            parent_job_id TEXT,
            labels TEXT
        )
        """
    )
    return con


class TestProactiveBlockerReplay(unittest.TestCase):
    def test_select_rows_prefers_parent_job_id(self) -> None:
        con = _mk_db()
        now = time.time()
        con.execute(
            """
            INSERT INTO jobs(job_id, role, state, depends_on, blocked_reason, updated_at, created_at, parent_job_id, labels)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                "j-parent",
                "backend",
                "running",
                "[]",
                "",
                now,
                now,
                "t-1",
                '{"ticket": "t-1"}',
            ),
        )
        rows, strategy = pbr._select_rows_for_ticket(con, "t-1")
        self.assertEqual(strategy, "parent_job_id")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["job_id"], "j-parent")

    def test_select_rows_falls_back_to_labels_when_parent_missing(self) -> None:
        con = _mk_db()
        now = time.time()
        con.execute(
            """
            INSERT INTO jobs(job_id, role, state, depends_on, blocked_reason, updated_at, created_at, parent_job_id, labels)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                "j-label",
                "qa",
                "waiting_deps",
                '["dep-1"]',
                "dependencies_pending:dep-1",
                now,
                now,
                None,
                '{"ticket": "t-2", "kind": "subtask"}',
            ),
        )
        rows, strategy = pbr._select_rows_for_ticket(con, "t-2")
        self.assertEqual(strategy, "labels_ticket_fallback")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["job_id"], "j-label")

    def test_select_rows_falls_back_to_compact_label_json(self) -> None:
        con = _mk_db()
        now = time.time()
        con.execute(
            """
            INSERT INTO jobs(job_id, role, state, depends_on, blocked_reason, updated_at, created_at, parent_job_id, labels)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                "j-compact-label",
                "qa",
                "queued",
                "[]",
                "",
                now,
                now,
                None,
                '{"ticket":"t-compact","kind":"subtask"}',
            ),
        )
        rows, strategy = pbr._select_rows_for_ticket(con, "t-compact")
        self.assertEqual(strategy, "labels_ticket_fallback")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["job_id"], "j-compact-label")


if __name__ == "__main__":
    unittest.main()
