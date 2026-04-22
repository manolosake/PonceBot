import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools import release_governance as rg


class TestReleaseGovernanceProvenance(unittest.TestCase):
    def test_build_run_provenance_serialization(self) -> None:
        payload = rg._build_run_provenance(
            generated_at="2026-04-22T00:00:00Z",
            branch="feature/order-aaa28bd3-proactive-sprint-codexbot-reliability-",
            commit_sha="abc123",
            role="backend",
            job_id="060a361f-6bd1-41f5-b94f-0b3245dd8724",
            ticket_id="aaa28bd3-3cdd-402c-917d-881544b08927",
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["branch"], "feature/order-aaa28bd3-proactive-sprint-codexbot-reliability-")
        self.assertEqual(payload["commit_sha"], "abc123")
        self.assertEqual(payload["role"], "backend")
        self.assertEqual(payload["job_id"], "060a361f-6bd1-41f5-b94f-0b3245dd8724")
        self.assertEqual(payload["ticket_id"], "aaa28bd3-3cdd-402c-917d-881544b08927")

    def test_traceability_ids_check_fails_when_missing(self) -> None:
        chk = rg._traceability_ids_check(job_id="", ticket_id="aaa28bd3-3cdd-402c-917d-881544b08927")
        self.assertEqual(chk.key, "traceability_ids_present")
        self.assertFalse(chk.ok)
        self.assertIn("job_id=<missing>", chk.details)

    def test_final_exit_code_hard_fails_on_manifest_mismatch(self) -> None:
        self.assertEqual(rg._final_exit_code(checks_ok=True, manifest_mismatch_count=1), 2)
        self.assertEqual(rg._final_exit_code(checks_ok=False, manifest_mismatch_count=0), 2)
        self.assertEqual(rg._final_exit_code(checks_ok=True, manifest_mismatch_count=0), 0)

    def test_build_final_validation_marks_not_ok_on_manifest_mismatch(self) -> None:
        fv = rg._build_final_validation(
            base_checks={
                "checks_ok": True,
                "pr_url_targets_head": True,
                "required_artifacts_non_empty": True,
            },
            manifest_mismatch_count=2,
        )
        self.assertFalse(fv["ok"])
        self.assertFalse(fv["checks"]["manifest_integrity_ok"])
        self.assertEqual(fv["checks"]["manifest_mismatch_count"], 2)

    def test_artifact_provenance_gate_fails_on_empty_required_files_for_implementation(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "changes.patch").write_text("", encoding="utf-8")
            (root / "git_status.txt").write_text("", encoding="utf-8")
            ok, reasons = rg._artifact_provenance_gate_check(artifacts_dir=root, implementation_claim=True)
            self.assertFalse(ok)
            self.assertIn("artifact_empty_changes_patch", reasons)
            self.assertIn("artifact_empty_git_status", reasons)

    def test_artifact_provenance_gate_passes_on_non_empty_required_files(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "changes.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
            (root / "git_status.txt").write_text("## branch\n", encoding="utf-8")
            ok, reasons = rg._artifact_provenance_gate_check(artifacts_dir=root, implementation_claim=True)
            self.assertTrue(ok)
            self.assertEqual(reasons, [])

    def test_build_final_validation_respects_artifact_provenance_gate(self) -> None:
        fv = rg._build_final_validation(
            base_checks={
                "checks_ok": True,
                "pr_url_targets_head": True,
                "required_artifacts_non_empty": True,
                "artifact_provenance_gate_ok": False,
            },
            manifest_mismatch_count=0,
        )
        self.assertFalse(fv["ok"])
        self.assertFalse(fv["checks"]["artifact_provenance_gate_ok"])


if __name__ == "__main__":
    unittest.main()
