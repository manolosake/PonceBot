import tempfile
import subprocess
import unittest
from pathlib import Path


class TestBackendTraceabilityRuntimeExport(unittest.TestCase):
    def test_cli_exports_contractual_ids_and_viewports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            png_stub = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 40)
            (root / "desktop_capture.png").write_bytes(png_stub)
            (root / "tablet_capture.png").write_bytes(png_stub)
            (root / "mobile_capture.png").write_bytes(png_stub)
            (root / "git_status.txt").write_text("M\tMakefile\n", encoding="utf-8")
            (root / "changes.patch").write_text(
                "diff --git a/Makefile b/Makefile\n--- a/Makefile\n+++ b/Makefile\n",
                encoding="utf-8",
            )
            (root / "patch_apply_check.json").write_text(
                '{"status":"PASS","files_in_patch":["Makefile"]}\n',
                encoding="utf-8",
            )
            rc = subprocess.call(
                [
                    "python3",
                    "tools/backend_traceability_runtime_export.py",
                    "--repo-root",
                    ".",
                    "--artifacts-dir",
                    str(root),
                    "--ticket-id",
                    "cdcb0d10-86d8-4086-8d7a-f256dc6bec30",
                    "--expected-branch",
                    "feature/order-cdcb0d10-nuevo-proyecto-ceo-iteracion-wormhole-",
                    "--frontend-job-id",
                    "frontend_rc_bundle_republish_v2",
                    "--target-artifact-dir",
                    str(root),
                    "--execution-id",
                    "exec-12345",
                ]
            )
            self.assertEqual(rc, 0)
            import json

            trace = json.loads((root / "wormhole_scene_trace.json").read_text(encoding="utf-8"))
            summary = json.loads((root / "backend_traceability_runtime_summary.json").read_text(encoding="utf-8"))
            runtime = json.loads((root / "backend_runtime_telemetry_report.json").read_text(encoding="utf-8"))

            self.assertEqual(trace["execution_id"], "exec-12345")
            self.assertEqual(summary["execution_id"], "exec-12345")
            self.assertIn("cdcb0d10-86d8-4086-8d7a-f256dc6bec30", trace["telegram_correlation_id"])
            self.assertIn("desktop", runtime["viewports"])
            self.assertIn("tablet", runtime["viewports"])
            self.assertIn("mobile", runtime["viewports"])

    def test_branch_mismatch_sets_fail_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "git_status.txt").write_text("M\tMakefile\n", encoding="utf-8")
            (root / "changes.patch").write_text(
                "diff --git a/Makefile b/Makefile\n--- a/Makefile\n+++ b/Makefile\n",
                encoding="utf-8",
            )
            (root / "patch_apply_check.json").write_text(
                '{"status":"PASS","files_in_patch":["Makefile"]}\n',
                encoding="utf-8",
            )
            rc = subprocess.call(
                [
                    "python3",
                    "tools/backend_traceability_runtime_export.py",
                    "--repo-root",
                    ".",
                    "--artifacts-dir",
                    str(root),
                    "--ticket-id",
                    "cdcb0d10-86d8-4086-8d7a-f256dc6bec30",
                    "--expected-branch",
                    "feature/does-not-match",
                    "--frontend-job-id",
                    "frontend_rc_bundle_republish_v2",
                    "--target-artifact-dir",
                    str(root),
                    "--execution-id",
                    "exec-999",
                ]
            )
            self.assertNotEqual(rc, 0)
            import json

            trace = json.loads((root / "wormhole_scene_trace.json").read_text(encoding="utf-8"))
            summary = json.loads((root / "backend_traceability_runtime_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(trace["branch_matches_expected"])
            self.assertEqual(summary["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
