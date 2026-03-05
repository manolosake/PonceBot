import tempfile
import unittest
from pathlib import Path

from tools.postfinal_root_smoke_guard import compare


class TestPostfinalRootSmokeGuard(unittest.TestCase):
    def test_compare_ignores_mtime_when_content_matches(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "root"
            smoke = base / "smoke_stable"
            root.mkdir(parents=True, exist_ok=True)
            smoke.mkdir(parents=True, exist_ok=True)

            (root / "changes.patch").write_text("diff --git a/a b/a\n", encoding="utf-8")
            (root / "git_status.txt").write_text("M\ta\n", encoding="utf-8")
            # Copy through write to force a distinct mtime while preserving bytes.
            (smoke / "changes.patch").write_text((root / "changes.patch").read_text(encoding="utf-8"), encoding="utf-8")
            (smoke / "git_status.txt").write_text((root / "git_status.txt").read_text(encoding="utf-8"), encoding="utf-8")

            mismatches, _, _ = compare(root, smoke, ["changes.patch", "git_status.txt"])
            self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()
