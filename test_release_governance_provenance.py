import unittest

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


if __name__ == "__main__":
    unittest.main()
