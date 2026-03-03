from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.wormhole_trace_export import build_trace


class WormholeTraceExportTests(unittest.TestCase):
    def test_expected_mode_locks_reported_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = build_trace(
                repo_root=root,
                artifacts_dir=root,
                ticket_id="4084914d-ac6e-4c97-9060-bb025e9eb12a",
                expected_branch="feature/order-4084914d-nuevo-proyecto-ceo-wormhole-vnext-hipe",
                reported_branch_mode="expected",
                execution_id="exec-test",
            )
            self.assertEqual(
                trace["reported_branch"],
                "feature/order-4084914d-nuevo-proyecto-ceo-wormhole-vnext-hipe",
            )
            self.assertTrue(trace["branch_matches_expected"])

    def test_observed_mode_uses_git_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = build_trace(
                repo_root=root,
                artifacts_dir=root,
                ticket_id="t",
                expected_branch="feature/order-4084914d-nuevo-proyecto-ceo-wormhole-vnext-hipe",
                reported_branch_mode="observed",
                execution_id="exec-test",
            )
            self.assertEqual(trace["reported_branch"], trace["observed_git_branch"])
            self.assertFalse(trace["branch_matches_expected"])


if __name__ == "__main__":
    unittest.main()
