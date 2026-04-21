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


if __name__ == "__main__":
    unittest.main()
