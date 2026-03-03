from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.backend_traceability_runtime_export import (
    _binding_errors,
    _default_runtime_metrics,
    _patch_status_coverage,
    _validate_runtime_metrics,
    export_payload,
)


class BackendTraceabilityRuntimeExportTests(unittest.TestCase):
    def _init_git_repo(self, root: Path, branch: str) -> None:
        subprocess.run(["git", "init"], cwd=str(root), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        subprocess.run(
            ["git", "config", "user.email", "qa@example.test"],
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "QA Bot"],
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        (root / "README.md").write_text("x\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=str(root), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "checkout", "-B", branch],
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_export_contains_trace_ids_and_branch_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            branch = "feature/order-470f8b7f-nuevo-proyecto-ceo-wormhole-vnext-qual"
            self._init_git_repo(root, branch)
            trace, runtime_report, events = export_payload(
                repo_root=root,
                artifacts_dir=root,
                ticket_id="470f8b7f-6f39-4fa9-9a30-75dae0a10bad",
                expected_branch=branch,
                execution_id="exec-test-470f",
                frontend_job_id="fe-job-001",
                target_artifact_dir=str(root),
                runtime_metrics=_default_runtime_metrics(),
            )
            self.assertEqual(trace["reported_branch"], branch)
            self.assertEqual(trace["observed_branch"], branch)
            self.assertTrue(trace["branch_matches_expected"])
            self.assertEqual(trace["execution_id"], "exec-test-470f")
            self.assertEqual(trace["telegram_correlation_id"], "470f8b7f-6f39-4fa9-9a30-75dae0a10bad:exec-test-470f")
            self.assertEqual(trace["frontend_job_id"], "fe-job-001")
            self.assertEqual(trace["target_artifact_dir"], str(root))
            self.assertEqual(sorted(runtime_report["viewports"].keys()), ["desktop", "mobile", "tablet"])
            self.assertGreaterEqual(len(events), 5)

    def test_runtime_metrics_validation(self) -> None:
        metrics = _default_runtime_metrics()
        self.assertEqual(_validate_runtime_metrics(metrics), [])
        del metrics["mobile"]
        errs = _validate_runtime_metrics(metrics)
        self.assertTrue(any("missing viewport: mobile" in e for e in errs))

    def test_patch_status_coverage_detects_missing_and_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "git_status.txt").write_text(" M README.md\n", encoding="utf-8")
            (root / "changes.patch").write_text(
                "diff --git a/other.txt b/other.txt\n--- a/other.txt\n+++ b/other.txt\n@@ -1 +1 @@\n-a\n+b\n",
                encoding="utf-8",
            )
            cov = _patch_status_coverage(root)
            self.assertIn("README.md", cov["missing_in_patch"])
            self.assertIn("other.txt", cov["orphan_in_patch"])

    def test_branch_mismatch_fails_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._init_git_repo(root, "feature/actual-branch")
            trace, _runtime_report, _events = export_payload(
                repo_root=root,
                artifacts_dir=root,
                ticket_id="470f8b7f-6f39-4fa9-9a30-75dae0a10bad",
                expected_branch="feature/expected-branch",
                execution_id="exec-test-mismatch",
                frontend_job_id="fe-job-001",
                target_artifact_dir=str(root),
                runtime_metrics=_default_runtime_metrics(),
            )
            self.assertFalse(trace["branch_matches_expected"])
            self.assertEqual(trace["observed_git_branch_actual"], "feature/actual-branch")

    def test_binding_errors_correct_and_incorrect(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            branch = "feature/order-470f8b7f-nuevo-proyecto-ceo-wormhole-vnext-qual"
            self._init_git_repo(root, branch)
            trace, runtime_report, _events = export_payload(
                repo_root=root,
                artifacts_dir=root,
                ticket_id="470f8b7f-6f39-4fa9-9a30-75dae0a10bad",
                expected_branch=branch,
                execution_id="exec-bind",
                frontend_job_id="fe-job-bind",
                target_artifact_dir=str(root),
                runtime_metrics=_default_runtime_metrics(),
            )
            self.assertEqual(_binding_errors(trace=trace, runtime_report=runtime_report, artifacts_dir=root), [])

            bad_trace = dict(trace)
            bad_trace["target_artifact_dir"] = str(root / "other")
            errs = _binding_errors(trace=bad_trace, runtime_report=runtime_report, artifacts_dir=root)
            self.assertTrue(any("target_artifact_dir mismatch" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
