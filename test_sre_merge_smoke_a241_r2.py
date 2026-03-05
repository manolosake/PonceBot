from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestSreMergeSmokeA241R2(unittest.TestCase):
    def test_generates_parseable_summary_and_non_empty_contract_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
            ).strip()
            rc = subprocess.call(
                [
                    "python3",
                    "tools/sre_merge_smoke_a241_r2.py",
                    "--repo-root",
                    ".",
                    "--artifacts-dir",
                    str(art),
                    "--ticket-id",
                    "a24101c9-f0ce-43f5-9298-79562357cc3f",
                    "--expected-branch",
                    branch,
                ]
            )
            self.assertIn(rc, (0, 1))

            summary_path = art / "summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary.get("ticket_id"), "a24101c9-f0ce-43f5-9298-79562357cc3f")
            self.assertEqual(summary.get("order_branch"), branch)

            for name in ("merge_tree.log", "changes.patch", "git_status.txt"):
                p = art / name
                self.assertTrue(p.exists(), name)
                self.assertGreater(p.stat().st_size, 0, name)


if __name__ == "__main__":
    unittest.main()
