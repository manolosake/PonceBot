import tempfile
import unittest
from pathlib import Path

from tools.wormhole_bundle_guard import evaluate_bundle


class TestWormholeBundleGuard(unittest.TestCase):
    def test_fail_when_patch_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            (art / "changes.patch").write_text("", encoding="utf-8")
            (art / "git_status.txt").write_text(" M tools/file.py\n", encoding="utf-8")
            (art / "patch_apply_check.json").write_text(
                '{"status":"PASS","declared_files":["tools/file.py"]}\n',
                encoding="utf-8",
            )
            result = evaluate_bundle(art, Path('.').resolve(), run_apply_check=False)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("changes.patch is empty", result["errors"])

    def test_fail_when_status_path_missing_in_patch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            (art / "changes.patch").write_text(
                "diff --git a/Makefile b/Makefile\n--- a/Makefile\n+++ b/Makefile\n",
                encoding="utf-8",
            )
            (art / "git_status.txt").write_text(" M tools/wormhole_scene_contract.py\n", encoding="utf-8")
            (art / "patch_apply_check.json").write_text(
                '{"status":"PASS","declared_files":["tools/wormhole_scene_contract.py"]}\n',
                encoding="utf-8",
            )
            result = evaluate_bundle(art, Path('.').resolve(), run_apply_check=False)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["checks"]["status_paths_covered_by_patch"])


if __name__ == "__main__":
    unittest.main()
