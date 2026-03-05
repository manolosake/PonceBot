import base64
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6pJ8kAAAAASUVORK5CYII="
)


class TestBackendTraceabilityRuntimeExport(unittest.TestCase):
    def _current_branch(self) -> str:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        )
        return out.strip()

    def _seed_bundle(self, root: Path) -> None:
        (root / "git_status.txt").write_text("M\tMakefile\n", encoding="utf-8")
        (root / "changes.patch").write_text(
            "diff --git a/Makefile b/Makefile\n--- a/Makefile\n+++ b/Makefile\n",
            encoding="utf-8",
        )
        (root / "patch_apply_check.json").write_text(
            json.dumps({"status": "PASS", "files_in_patch": ["Makefile"]}) + "\n",
            encoding="utf-8",
        )

    def test_export_includes_visual_metadata_per_viewport(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed_bundle(root)
            (root / "desktop_capture.png").write_bytes(PNG_1X1)
            (root / "tablet_capture.png").write_bytes(PNG_1X1)
            (root / "mobile_capture.png").write_bytes(PNG_1X1)

            rc = subprocess.call(
                [
                    "python3",
                    "tools/backend_traceability_runtime_export.py",
                    "--repo-root",
                    ".",
                    "--artifacts-dir",
                    str(root),
                    "--ticket-id",
                    "01b4d09a-665c-4d53-b1fb-8cea3509d4b5",
                    "--expected-branch",
                    self._current_branch(),
                    "--frontend-job-id",
                    "frontend_rc_bundle_republish_v2",
                    "--target-artifact-dir",
                    str(root),
                    "--execution-id",
                    "exec-visual-1",
                ]
            )
            self.assertEqual(rc, 0)

            summary = json.loads((root / "backend_traceability_runtime_summary.json").read_text(encoding="utf-8"))
            by_vp = summary["visual_metadata"]["by_viewport"]
            for vp in ("desktop", "tablet", "mobile"):
                self.assertTrue(by_vp[vp]["present"])
                self.assertGreater(by_vp[vp]["bytes"], 0)
                self.assertEqual(len(by_vp[vp]["sha256"]), 64)
                self.assertGreater(by_vp[vp]["width"], 0)
                self.assertGreater(by_vp[vp]["height"], 0)

    def test_export_fails_when_viewport_metadata_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._seed_bundle(root)
            (root / "desktop_capture.png").write_bytes(PNG_1X1)
            (root / "tablet_capture.png").write_bytes(PNG_1X1)

            rc = subprocess.call(
                [
                    "python3",
                    "tools/backend_traceability_runtime_export.py",
                    "--repo-root",
                    ".",
                    "--artifacts-dir",
                    str(root),
                    "--ticket-id",
                    "01b4d09a-665c-4d53-b1fb-8cea3509d4b5",
                    "--expected-branch",
                    self._current_branch(),
                    "--frontend-job-id",
                    "frontend_rc_bundle_republish_v2",
                    "--target-artifact-dir",
                    str(root),
                    "--execution-id",
                    "exec-visual-2",
                ]
            )
            self.assertNotEqual(rc, 0)

            summary = json.loads((root / "backend_traceability_runtime_summary.json").read_text(encoding="utf-8"))
            self.assertIn("mobile", summary["visual_metadata"]["missing_viewports"])


if __name__ == "__main__":
    unittest.main()
