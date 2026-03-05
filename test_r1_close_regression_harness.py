import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestR1CloseRegressionHarness(unittest.TestCase):
    def test_outputs_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rc = subprocess.call(
                [
                    "python3",
                    "tools/r1_close_regression_harness.py",
                    "--repo-root",
                    ".",
                    "--artifacts-dir",
                    str(root),
                    "--probe-interval-seconds",
                    "0.1",
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads((root / "r1_close_regression_harness_report.json").read_text(encoding="utf-8"))
            self.assertIn("pre_summary", report)
            self.assertIn("post_guard", report)
            self.assertIn("live_final", report)
            self.assertIn("invariant_diffs", report)
            self.assertEqual(report["status"], "PASS")

    def test_detects_late_mutation_as_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rc = subprocess.call(
                [
                    "python3",
                    "tools/r1_close_regression_harness.py",
                    "--repo-root",
                    ".",
                    "--artifacts-dir",
                    str(root),
                    "--probe-interval-seconds",
                    "0.1",
                    "--simulate-late-mutation",
                ]
            )
            self.assertNotEqual(rc, 0)
            report = json.loads((root / "r1_close_regression_harness_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")

    def test_detects_post_guard_mutation_as_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rc = subprocess.call(
                [
                    "python3",
                    "tools/r1_close_regression_harness.py",
                    "--repo-root",
                    ".",
                    "--artifacts-dir",
                    str(root),
                    "--probe-interval-seconds",
                    "0.1",
                    "--simulate-post-guard-mutation",
                ]
            )
            self.assertNotEqual(rc, 0)
            report = json.loads((root / "r1_close_regression_harness_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(report["invariant_diffs"]["post_guard_to_live_final"] or report["invariant_diffs"]["mtime_late_write_after_summary"])


if __name__ == "__main__":
    unittest.main()
