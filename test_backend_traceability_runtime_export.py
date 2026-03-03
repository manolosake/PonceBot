import tempfile
import unittest
from pathlib import Path

from tools.backend_traceability_runtime_export import export_payload


class TestBackendTraceabilityRuntimeExport(unittest.TestCase):
    def test_payload_contains_contractual_ids_and_viewports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = export_payload(
                repo_root=Path(".").resolve(),
                artifacts_dir=root,
                ticket_id="cdcb0d10-86d8-4086-8d7a-f256dc6bec30",
                expected_branch="feature/order-cdcb0d10-nuevo-proyecto-ceo-iteracion-wormhole-",
                frontend_job_id="frontend_rc_bundle_republish_v2",
                target_artifact_dir=str(root),
                execution_id="exec-12345",
            )
            trace = payload["trace"]
            summary = payload["summary"]
            runtime = payload["runtime_report"]

            self.assertEqual(trace["execution_id"], "exec-12345")
            self.assertEqual(summary["execution_id"], "exec-12345")
            self.assertIn("cdcb0d10-86d8-4086-8d7a-f256dc6bec30", trace["telegram_correlation_id"])
            self.assertIn("desktop", runtime["viewports"])
            self.assertIn("tablet", runtime["viewports"])
            self.assertIn("mobile", runtime["viewports"])

    def test_branch_mismatch_sets_fail_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = export_payload(
                repo_root=Path(".").resolve(),
                artifacts_dir=root,
                ticket_id="cdcb0d10-86d8-4086-8d7a-f256dc6bec30",
                expected_branch="feature/does-not-match",
                frontend_job_id="frontend_rc_bundle_republish_v2",
                target_artifact_dir=str(root),
                execution_id="exec-999",
            )
            self.assertFalse(payload["trace"]["branch_matches_expected"])
            self.assertEqual(payload["summary"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
