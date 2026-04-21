from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.proactive_3h_autofix_monitor import _log


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


if __name__ == "__main__":
    unittest.main()
