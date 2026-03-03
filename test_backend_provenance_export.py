from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.backend_provenance_export import build_trace, _default_runtime_telemetry, _validate_runtime_telemetry


class BackendProvenanceExportTests(unittest.TestCase):
    def test_trace_locks_reported_branch_to_expected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = build_trace(
                repo_root=root,
                artifacts_dir=root,
                ticket_id="821284e8-6786-407d-8c65-7f64e9bfcee8",
                expected_branch="feature/order-821284e8-nuevo-proyecto-ceo-iteracion-final-wor",
                execution_id="exec-test-1",
            )
            self.assertEqual(
                trace["reported_branch"],
                "feature/order-821284e8-nuevo-proyecto-ceo-iteracion-final-wor",
            )
            self.assertTrue(trace["branch_matches_expected"])
            self.assertTrue(trace["telegram_correlation_id"].endswith(":exec-test-1"))

    def test_default_runtime_has_all_viewports(self) -> None:
        rt = _default_runtime_telemetry()
        self.assertEqual(sorted(rt.keys()), ["desktop", "mobile", "tablet"])
        errs = _validate_runtime_telemetry(rt)
        self.assertEqual(errs, [])

    def test_runtime_validation_fails_missing_viewport(self) -> None:
        rt = _default_runtime_telemetry()
        del rt["mobile"]
        errs = _validate_runtime_telemetry(rt)
        self.assertTrue(any("missing viewport telemetry: mobile" in e for e in errs))


if __name__ == "__main__":
    unittest.main()

