from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.proactive_health_report import order_autonomy_funnel
from tools.proactive_health_report import classify_open_job_mode, is_blocked_without_open_jobs


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

    def test_order_autonomy_funnel_no_code_change_slice_can_close(self) -> None:
        order_id = "order-nochange-1234"
        now = 2000.0
        since = 0.0
        children = [
            {
                "role": "implementer_local",
                "state": "done",
                "created_at": 1100.0,
                "updated_at": 1200.0,
                "trace": json.dumps(
                    {
                        "slice_id": "slice_a",
                        "slice_no_code_change": True,
                        "result_summary": "No code changes required; existing behavior already satisfied.",
                    }
                ),
                "labels": "{}",
            },
            {
                "role": "reviewer_local",
                "state": "done",
                "created_at": 1210.0,
                "updated_at": 1300.0,
                "trace": json.dumps(
                    {
                        "slice_id": "slice_a",
                        "result_summary": "READY: verified expected behavior and evidence.",
                    }
                ),
                "labels": "{}",
            },
            {
                "role": "skynet",
                "state": "done",
                "created_at": 1310.0,
                "updated_at": 1400.0,
                "trace": json.dumps(
                    {
                        "slice_id": "slice_a",
                        "improvement_verified": True,
                        "result_summary": "improvement_verified",
                    }
                ),
                "labels": "{}",
            },
        ]

        funnel = order_autonomy_funnel(children, order_id=order_id, now=now, since=since)
        self.assertEqual(funnel["slices_validated"], 1)
        self.assertEqual(funnel["slices_closed"], 1)
        self.assertEqual(funnel["quality_gate_status"], "closed")
        self.assertTrue(funnel["improvement_verified"])
    def test_classify_open_job_mode_respects_open_jobs_semantics(self) -> None:
        self.assertEqual(classify_open_job_mode({}, 0), "idle")
        self.assertEqual(classify_open_job_mode({"skynet": 1}, 1), "controller")
        self.assertEqual(classify_open_job_mode({"architect_local": 1}, 1), "local")
        self.assertEqual(classify_open_job_mode({"backend": 1}, 1), "cli")
        self.assertEqual(classify_open_job_mode({"backend": 1, "reviewer_local": 1}, 2), "mixed")

    def test_is_blocked_without_open_jobs_detects_stall_condition(self) -> None:
        self.assertTrue(is_blocked_without_open_jobs("blocked", 0))
        self.assertTrue(is_blocked_without_open_jobs("blocked_waiting_only", 0))
        self.assertTrue(is_blocked_without_open_jobs("blocked_approval", 0))
        self.assertFalse(is_blocked_without_open_jobs("blocked", 1))
        self.assertFalse(is_blocked_without_open_jobs("review", 0))


if __name__ == "__main__":
    unittest.main()
