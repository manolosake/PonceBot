from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.publish_atomic_guard import run_guard


class TestPublishAtomicGuard(unittest.TestCase):
    def test_pass_when_status_patch_and_declared_align(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "git_status.txt").write_text(" M Makefile\n M tools/x.py\n", encoding="utf-8")
            (d / "changes.patch").write_text(
                (
                    "diff --git a/Makefile b/Makefile\n"
                    "index 1..2 100644\n"
                    "--- a/Makefile\n"
                    "+++ b/Makefile\n"
                    "diff --git a/tools/x.py b/tools/x.py\n"
                    "index 1..2 100644\n"
                    "--- a/tools/x.py\n"
                    "+++ b/tools/x.py\n"
                ),
                encoding="utf-8",
            )
            (d / "patch_apply_check.json").write_text(
                '{ "declared_files": ["Makefile", "tools/x.py"] }\n',
                encoding="utf-8",
            )
            out = run_guard(d)
            self.assertEqual(out["status"], "PASS")
            self.assertEqual(out["exit_code"], 0)

    def test_fail_when_patch_empty_with_declared_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "git_status.txt").write_text(" M Makefile\n", encoding="utf-8")
            (d / "changes.patch").write_text("", encoding="utf-8")
            (d / "patch_apply_check.json").write_text(
                '{ "declared_files": ["Makefile"] }\n',
                encoding="utf-8",
            )
            out = run_guard(d)
            self.assertEqual(out["status"], "FAIL")
            rules = {i["rule"] for i in out["issues"]}
            self.assertIn("changes_patch_non_empty_when_declared_paths_exist", rules)
            self.assertIn("declared_paths_covered_by_patch", rules)


if __name__ == "__main__":
    unittest.main()
