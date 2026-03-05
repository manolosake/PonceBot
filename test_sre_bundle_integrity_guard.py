from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestSreBundleIntegrityGuard(unittest.TestCase):
    def test_fails_when_summary_is_not_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            (art / "summary.json").write_text("{bad-json", encoding="utf-8")
            (art / "merge_tree.log").write_text("ok\n", encoding="utf-8")
            (art / "changes.patch").write_text("ok\n", encoding="utf-8")
            (art / "git_status.txt").write_text("ok\n", encoding="utf-8")
            (art / "git_status").write_text("ok\n", encoding="utf-8")

            rc = subprocess.call(
                [
                    "python3",
                    "tools/sre_bundle_integrity_guard.py",
                    "--artifacts-dir",
                    str(art),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads((art / "bundle_integrity_guard_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report.get("status"), "FAIL")


if __name__ == "__main__":
    unittest.main()

