from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
