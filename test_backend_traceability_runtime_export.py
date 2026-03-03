from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.backend_traceability_runtime_export import (
    _default_runtime_metrics,
    _validate_runtime_metrics,
    export_payload,
)


class BackendTraceabilityRuntimeExportTests(unittest.TestCase):
    def test_export_contains_trace_ids_and_branch_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace, runtime_report, events = export_payload(
                repo_root=root,
                artifacts_dir=root,
                ticket_id="470f8b7f-6f39-4fa9-9a30-75dae0a10bad",
                expected_branch="feature/order-470f8b7f-nuevo-proyecto-ceo-wormhole-vnext-qual",
                execution_id="exec-test-470f",
                runtime_metrics=_default_runtime_metrics(),
            )
            self.assertEqual(trace["reported_branch"], "feature/order-470f8b7f-nuevo-proyecto-ceo-wormhole-vnext-qual")
            self.assertEqual(trace["observed_branch"], "feature/order-470f8b7f-nuevo-proyecto-ceo-wormhole-vnext-qual")
            self.assertTrue(trace["branch_matches_expected"])
            self.assertEqual(trace["execution_id"], "exec-test-470f")
            self.assertEqual(trace["telegram_correlation_id"], "470f8b7f-6f39-4fa9-9a30-75dae0a10bad:exec-test-470f")
            self.assertEqual(sorted(runtime_report["viewports"].keys()), ["desktop", "mobile", "tablet"])
            self.assertGreaterEqual(len(events), 5)

    def test_runtime_metrics_validation(self) -> None:
        metrics = _default_runtime_metrics()
        self.assertEqual(_validate_runtime_metrics(metrics), [])
        del metrics["mobile"]
        errs = _validate_runtime_metrics(metrics)
        self.assertTrue(any("missing viewport: mobile" in e for e in errs))


if __name__ == "__main__":
    unittest.main()

