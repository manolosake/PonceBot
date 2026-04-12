from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestProactiveHealthReport(unittest.TestCase):
    def test_missing_db_writes_error_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            env = {**dict(os.environ)}
            env["CODEXBOT_ORCH_DB"] = str(root / "missing.sqlite")
            env["CODEXBOT_STATE_FILE"] = str(root / "state.json")
            env["CODEXBOT_PROACTIVE_HEALTH_OUT_DIR"] = str(out_dir)
            proc = subprocess.run(
                ["python3", "tools/proactive_health_report.py"],
                cwd=str(Path(__file__).resolve().parent),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            latest = out_dir / "latest.json"
            self.assertTrue(latest.exists())
            payload = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("operational_status"), "CRITICAL")
            self.assertEqual(payload.get("error"), "db_missing")


if __name__ == "__main__":
    unittest.main()
