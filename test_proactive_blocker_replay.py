import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
import time
import unittest
from unittest import mock

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


def _run_replay_for_wait_dep(depends_on: str, ticket_id: str, job_id: str) -> tuple[int, dict]:
    with tempfile.NamedTemporaryFile(delete=False) as db_file:
        db_path = db_file.name

    try:
        con = sqlite3.connect(db_path)
        try:
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
            now = time.time()
            con.execute(
                """
                INSERT INTO jobs(job_id, role, state, depends_on, blocked_reason, updated_at, created_at, parent_job_id, labels)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    "backend",
                    "waiting_deps",
                    depends_on,
                    "",
                    now,
                    now,
                    ticket_id,
                    "{}",
                ),
            )
            con.commit()
        finally:
            con.close()

        out = io.StringIO()
        argv = [
            "proactive_blocker_replay.py",
            "--db",
            db_path,
            "--ticket-id",
            ticket_id,
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            rc = pbr.main()

        return rc, json.loads(out.getvalue())
    finally:
        os.unlink(db_path)


class TestProactiveBlockerReplay(unittest.TestCase):
    def test_coerce_depends_on_list_reports_malformed_json(self) -> None:
        deps, malformed = pbr._coerce_depends_on_list("[")

        self.assertEqual(deps, [])
        self.assertTrue(malformed)

    def test_coerce_depends_on_list_reports_non_list_json(self) -> None:
        for payload in ('{"dep": "job-a"}', '"job-a"', "42"):
            with self.subTest(payload=payload):
                deps, malformed = pbr._coerce_depends_on_list(payload)

                self.assertEqual(deps, [])
                self.assertTrue(malformed)

    def test_coerce_depends_on_list_preserves_valid_string_lists(self) -> None:
        deps, malformed = pbr._coerce_depends_on_list('[" job-a ", "", 42, "job-b"]')

        self.assertEqual(deps, ["job-a", "job-b"])
        self.assertFalse(malformed)

    def test_updated_age_seconds_accepts_epoch_and_iso_timestamps(self) -> None:
        now = 1_700_000_100.0
        iso = datetime.fromtimestamp(now - 42.0, UTC).isoformat().replace("+00:00", "Z")

        self.assertEqual(pbr._updated_age_seconds(now - 5.0, now), 5.0)
        self.assertEqual(pbr._updated_age_seconds((now - 7.0) * 1000.0, now), 7.0)
        self.assertEqual(round(pbr._updated_age_seconds(iso, now), 1), 42.0)
        self.assertEqual(pbr._updated_age_seconds("not-a-date", now), 0.0)

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

    def test_select_rows_escapes_like_wildcards_in_ticket_id(self) -> None:
        con = _mk_db()
        now = time.time()
        for job_id, ticket in (
            ("j-literal", "ticket_%"),
            ("j-wildcard-decoy", "ticket-xx"),
        ):
            con.execute(
                """
                INSERT INTO jobs(job_id, role, state, depends_on, blocked_reason, updated_at, created_at, parent_job_id, labels)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    "qa",
                    "queued",
                    "[]",
                    "",
                    now,
                    now,
                    None,
                    f'{{"ticket":"{ticket}","kind":"subtask"}}',
                ),
            )

        rows, strategy = pbr._select_rows_for_ticket(con, "ticket_%")
        self.assertEqual(strategy, "labels_ticket_fallback")
        self.assertEqual([row["job_id"] for row in rows], ["j-literal"])

    def test_main_reports_malformed_wait_dependency_job_id(self) -> None:
        rc, payload = _run_replay_for_wait_dep("[", "t-bad-json", "waiting-bad-json")

        self.assertEqual(rc, 2)
        self.assertEqual(payload["invalid_wait_job_ids"], ["waiting-bad-json"])

    def test_main_reports_non_list_wait_dependency_job_id(self) -> None:
        rc, payload = _run_replay_for_wait_dep(
            '{"dep": "job-a"}',
            "t-object-json",
            "waiting-object-json",
        )

        self.assertEqual(rc, 2)
        self.assertEqual(payload["invalid_wait_job_ids"], ["waiting-object-json"])


if __name__ == "__main__":
    unittest.main()
