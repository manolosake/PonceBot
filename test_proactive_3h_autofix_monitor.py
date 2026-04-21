from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.proactive_3h_autofix_monitor as monitor
from tools.proactive_3h_autofix_monitor import _log


def _mp_writer(log_path: str, start_event: mp.synchronize.Event, count: int, worker_id: int) -> None:
    start_event.wait(timeout=10)
    for i in range(count):
        _log("mp_event", path=Path(log_path), worker=worker_id, i=i)


class TestProactive3hAutofixMonitorLog(unittest.TestCase):
    def test_log_appends_without_read_modify_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "autofix.log"
            log_path.write_text('{"ts":"old","msg":"seed"}\n', encoding="utf-8")

            with mock.patch.object(Path, "read_text", side_effect=AssertionError("read_text should not be used by _log")):
                _log("new_event", path=log_path, order_id="o-1")

            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"msg":"seed"', lines[0])
            payload = json.loads(lines[1])
            self.assertEqual(payload["msg"], "new_event")
            self.assertEqual(payload["order_id"], "o-1")

    def test_log_keeps_all_lines_under_multiprocess_writers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "autofix.log"
            workers = 4
            writes_per_worker = 75
            ctx = mp.get_context("spawn")
            start_event = ctx.Event()
            procs = [
                ctx.Process(target=_mp_writer, args=(str(log_path), start_event, writes_per_worker, worker_id))
                for worker_id in range(workers)
            ]

            for proc in procs:
                proc.start()
            start_event.set()
            for proc in procs:
                proc.join(timeout=20)
                self.assertEqual(proc.exitcode, 0)

            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), workers * writes_per_worker)
            payloads = [json.loads(line) for line in lines]
            self.assertTrue(all(p["msg"] == "mp_event" for p in payloads))


class TestProactive3hAutofixMonitorSweep(unittest.TestCase):
    def test_closes_active_proactive_order_when_idle_and_reseed_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
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
                    parent_job_id TEXT,
                    role TEXT,
                    state TEXT,
                    model TEXT,
                    created_at REAL,
                    updated_at REAL,
                    trace TEXT
                )
                """
            )
            order_id = "65b0b3ec-f69f-4c06-94df-572debf167a9"
            conn.execute(
                "INSERT INTO ceo_orders(order_id, chat_id, title, body, status, phase, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (order_id, 10, "Proactive Sprint", "[proactive: reliability]", "active", "review", 1.0),
            )
            conn.execute(
                "INSERT INTO jobs(job_id, parent_job_id, role, state, model, created_at, updated_at, trace) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("job-1", order_id, "backend", "done", "gpt-5", 1.0, 1.0, "{}"),
            )
            conn.commit()
            conn.close()

            class _Clock:
                def __init__(self) -> None:
                    self.now = 0.0

                def __call__(self) -> float:
                    self.now += 0.2
                    return self.now

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
            self.assertIsNotNone(row)
            self.assertEqual(str(row[0]), "done")
            self.assertEqual(str(row[1]), "done")

    def test_enqueues_fallback_job_when_idle_and_reseed_fails_without_terminal_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
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
            order_id = "65b0b3ec-f69f-4c06-94df-572debf167a9"
            conn.execute(
                "INSERT INTO ceo_orders(order_id, chat_id, title, body, status, phase, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (order_id, 10, "Proactive Sprint", "[proactive: reliability]", "active", "review", 1.0),
            )
            conn.commit()
            conn.close()

            class _Clock:
                def __init__(self) -> None:
                    self.now = 0.0

                def __call__(self) -> float:
                    self.now += 0.2
                    return self.now

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
                "SELECT parent_job_id, role, state, source FROM jobs WHERE parent_job_id=? ORDER BY created_at DESC LIMIT 1",
                (order_id,),
            ).fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(str(row[0]), order_id)
            self.assertEqual(str(row[1]), "architect_local")
            self.assertEqual(str(row[2]), "queued")
            self.assertEqual(str(row[3]), "autopilot")

    def test_cancels_stale_blocked_only_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
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
            order_id = "65b0b3ec-f69f-4c06-94df-572debf167a9"
            conn.execute(
                "INSERT INTO ceo_orders(order_id, chat_id, title, body, status, phase, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (order_id, 10, "Proactive Sprint", "[proactive: reliability]", "active", "review", 1.0),
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
                    "blocked-1",
                    "autopilot",
                    "backend",
                    "stuck",
                    "task",
                    2,
                    "gpt-5",
                    "medium",
                    "rw",
                    0,
                    2.0,
                    -2000.0,
                    -2000.0,
                    None,
                    "blocked",
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
                    "legacy_blocker",
                    0,
                    None,
                    "{}",
                ),
            )
            conn.commit()
            conn.close()

            class _Clock:
                def __init__(self) -> None:
                    self.now = 0.0

                def __call__(self) -> float:
                    self.now += 0.2
                    return self.now

            with (
                mock.patch.object(monitor, "DB_PATH", db_path),
                mock.patch.object(monitor, "_trigger_autopilot", return_value=0),
                mock.patch.object(monitor.time, "time", side_effect=_Clock()),
                mock.patch.object(monitor.time, "sleep", return_value=None),
            ):
                rc = monitor.run_monitor(duration_s=1, interval_s=0, log_path=Path(td) / "monitor.log")
            self.assertEqual(rc, 0)

            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT state, blocked_reason FROM jobs WHERE job_id='blocked-1'").fetchone()
            followup = conn.execute(
                "SELECT role, state, source FROM jobs WHERE parent_job_id=? AND job_id<>'blocked-1' ORDER BY created_at DESC LIMIT 1",
                (order_id,),
            ).fetchone()
            order = conn.execute("SELECT status FROM ceo_orders WHERE order_id=?", (order_id,)).fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(str(row[0]), "cancelled")
            self.assertEqual(str(row[1]), "autofix_stale_blocked_only")
            self.assertIsNotNone(followup)
            self.assertEqual(str(followup[0]), "architect_local")
            self.assertEqual(str(followup[1]), "queued")
            self.assertEqual(str(followup[2]), "autopilot")
            self.assertIsNotNone(order)
            self.assertEqual(str(order[0]), "active")

    def test_cancels_superseded_blocker_when_dependency_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
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
            order_id = "65b0b3ec-f69f-4c06-94df-572debf167a9"
            conn.execute(
                "INSERT INTO ceo_orders(order_id, chat_id, title, body, status, phase, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (order_id, 10, "Proactive Sprint", "[proactive: reliability]", "active", "review", 1.0),
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
                    "dep-1",
                    "autopilot",
                    "backend",
                    "dependency done",
                    "task",
                    2,
                    "gpt-5",
                    "medium",
                    "rw",
                    0,
                    2.0,
                    1.0,
                    1.0,
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
                    "blocked-2",
                    "autopilot",
                    "backend",
                    "blocked on done dep",
                    "task",
                    2,
                    "gpt-5",
                    "medium",
                    "rw",
                    0,
                    2.0,
                    1.0,
                    1.0,
                    None,
                    "blocked",
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
                    "waiting_for_job:dep-1",
                    0,
                    None,
                    "{}",
                ),
            )
            conn.commit()
            conn.close()

            class _Clock:
                def __init__(self) -> None:
                    self.now = 0.0

                def __call__(self) -> float:
                    self.now += 0.2
                    return self.now

            with (
                mock.patch.object(monitor, "DB_PATH", db_path),
                mock.patch.object(monitor, "_trigger_autopilot", return_value=0),
                mock.patch.object(monitor.time, "time", side_effect=_Clock()),
                mock.patch.object(monitor.time, "sleep", return_value=None),
            ):
                rc = monitor.run_monitor(duration_s=1, interval_s=0, log_path=Path(td) / "monitor.log")
            self.assertEqual(rc, 0)

            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT state, blocked_reason FROM jobs WHERE job_id='blocked-2'").fetchone()
            followup = conn.execute(
                "SELECT role, state FROM jobs WHERE parent_job_id=? AND job_id NOT IN ('dep-1', 'blocked-2') ORDER BY created_at DESC LIMIT 1",
                (order_id,),
            ).fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(str(row[0]), "cancelled")
            self.assertEqual(str(row[1]), "autofix_superseded_blocker")
            self.assertIsNotNone(followup)
            self.assertEqual(str(followup[0]), "architect_local")
            self.assertEqual(str(followup[1]), "queued")

    def test_cancels_stale_blocked_approval_only_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
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
            order_id = "65b0b3ec-f69f-4c06-94df-572debf167a9"
            conn.execute(
                "INSERT INTO ceo_orders(order_id, chat_id, title, body, status, phase, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (order_id, 10, "Proactive Sprint", "[proactive: reliability]", "active", "review", 1.0),
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
                    "blocked-appr-1",
                    "autopilot",
                    "backend",
                    "waiting approval",
                    "task",
                    2,
                    "gpt-5",
                    "medium",
                    "rw",
                    1,
                    2.0,
                    -2000.0,
                    -2000.0,
                    None,
                    "blocked_approval",
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
                    "approval_timeout",
                    0,
                    None,
                    "{}",
                ),
            )
            conn.commit()
            conn.close()

            class _Clock:
                def __init__(self) -> None:
                    self.now = 0.0

                def __call__(self) -> float:
                    self.now += 0.2
                    return self.now

            with (
                mock.patch.object(monitor, "DB_PATH", db_path),
                mock.patch.object(monitor, "_trigger_autopilot", return_value=0),
                mock.patch.object(monitor.time, "time", side_effect=_Clock()),
                mock.patch.object(monitor.time, "sleep", return_value=None),
            ):
                rc = monitor.run_monitor(duration_s=1, interval_s=0, log_path=Path(td) / "monitor.log")
            self.assertEqual(rc, 0)

            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT state, blocked_reason FROM jobs WHERE job_id='blocked-appr-1'").fetchone()
            followup = conn.execute(
                "SELECT role, state FROM jobs WHERE parent_job_id=? AND job_id<>'blocked-appr-1' ORDER BY created_at DESC LIMIT 1",
                (order_id,),
            ).fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(str(row[0]), "cancelled")
            self.assertEqual(str(row[1]), "autofix_stale_blocked_only")
            self.assertIsNotNone(followup)
            self.assertEqual(str(followup[0]), "architect_local")
            self.assertEqual(str(followup[1]), "queued")


if __name__ == "__main__":
    unittest.main()
