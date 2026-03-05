import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tools.postfinal_root_smoke_guard import compare


class TestPostfinalRootSmokeGuard(unittest.TestCase):
    def _seed(self, root: Path) -> None:
        smoke = root / "smoke_stable"
        smoke.mkdir(parents=True, exist_ok=True)
        status = "M\tMakefile\n"
        patch = "diff --git a/Makefile b/Makefile\n--- a/Makefile\n+++ b/Makefile\n"
        (smoke / "git_status.txt").write_text(status, encoding="utf-8")
        (smoke / "changes.patch").write_text(patch, encoding="utf-8")
        (root / "git_status.txt").write_text(status, encoding="utf-8")
        (root / "changes.patch").write_text(patch, encoding="utf-8")

    def test_compare_ignores_mtime_when_content_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "root"
            smoke = base / "smoke_stable"
            root.mkdir(parents=True, exist_ok=True)
            smoke.mkdir(parents=True, exist_ok=True)

            (root / "changes.patch").write_text("diff --git a/a b/a\n", encoding="utf-8")
            (root / "git_status.txt").write_text("M\ta\n", encoding="utf-8")
            time.sleep(0.01)
            (smoke / "changes.patch").write_text((root / "changes.patch").read_text(encoding="utf-8"), encoding="utf-8")
            (smoke / "git_status.txt").write_text((root / "git_status.txt").read_text(encoding="utf-8"), encoding="utf-8")

            mismatches, _, _ = compare(root, smoke, ["changes.patch", "git_status.txt"])
            self.assertEqual(mismatches, [])

    def test_pass_when_root_smoke_match_and_no_late_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed(root)
            time.sleep(0.01)
            (root / "sre_close_summary.json").write_text(
                json.dumps({"status": "PASS", "generated_at_utc": "2026-03-05T00:00:00Z"}, indent=2) + "\n",
                encoding="utf-8",
            )
            report = root / "postfinal_close_guard_report.json"
            rc = subprocess.call(
                [
                    "python3",
                    "tools/postfinal_root_smoke_guard.py",
                    "--root-dir",
                    str(root),
                    "--smoke-stable-dir",
                    str(root / "smoke_stable"),
                    "--files",
                    "changes.patch,git_status.txt",
                    "--summary",
                    str(root / "sre_close_summary.json"),
                    "--report",
                    str(report),
                    "--sleep-seconds",
                    "0.1",
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(obj["status"], "PASS")
            self.assertTrue(obj["root_contract_files_non_empty_terminal"])
            self.assertTrue(obj["root_smoke_consistency_terminal"])
            self.assertTrue(obj["root_mtime_not_newer_than_summary"])

    def test_fail_on_post_summary_truncation_regression(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed(root)
            time.sleep(0.01)
            (root / "sre_close_summary.json").write_text(
                json.dumps({"status": "PASS", "generated_at_utc": "2026-03-05T00:00:00Z"}, indent=2) + "\n",
                encoding="utf-8",
            )
            report = root / "postfinal_close_guard_report.json"

            proc = subprocess.Popen(
                [
                    "python3",
                    "tools/postfinal_root_smoke_guard.py",
                    "--root-dir",
                    str(root),
                    "--smoke-stable-dir",
                    str(root / "smoke_stable"),
                    "--files",
                    "changes.patch,git_status.txt",
                    "--summary",
                    str(root / "sre_close_summary.json"),
                    "--report",
                    str(report),
                    "--sleep-seconds",
                    "0.3",
                ]
            )
            time.sleep(0.1)
            (root / "changes.patch").write_text("", encoding="utf-8")
            rc = proc.wait(timeout=5)
            self.assertNotEqual(rc, 0)
            obj = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(obj["status"], "FAIL")
            error_types = {e.get("type") for e in obj.get("errors", [])}
            self.assertTrue(
                "root_contract_file_empty_or_missing_terminal" in error_types
                or "late_root_mutation_after_summary" in error_types
                or "root_smoke_mismatch_terminal" in error_types
            )


if __name__ == "__main__":
    unittest.main()
