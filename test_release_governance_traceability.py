import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools import release_governance as rg


class TestReleaseGovernanceTraceability(unittest.TestCase):
    def test_traceability_mismatch_seed_vs_reseed_not_counted(self) -> None:
        log_text = (
            "abc1234 order:019abfcf key:proactive_cli_seed_r1\n"
            "def5678 order:019abfcf key:proactive_cli_seed_r1_2\n"
        )
        count = rg._traceability_count_from_log(
            log_text,
            order_token="order:019abfcf",
            key_token="key:proactive_cli_reseed_r1",
        )
        self.assertEqual(count, 0)

    def test_traceability_cross_order_false_positive_not_counted(self) -> None:
        log_text = (
            "1111111 order:deadbeef key:proactive_cli_reseed_r1\n"
            "2222222 order:019abfcf key:proactive_cli_reseed_r1\n"
        )
        count = rg._traceability_count_from_log(
            log_text,
            order_token="order:deadbeef",
            key_token="key:proactive_cli_reseed_r1",
        )
        self.assertEqual(count, 1)

    def test_manifest_mismatch_count_zero_for_stable_files(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "command_transcript.txt").write_text("cmd\n", encoding="utf-8")
            (root / "release_governance.exit_code.txt").write_text("0\n", encoding="utf-8")
            (root / "release_governance.stdout.json").write_text("{\"ok\":true}\n", encoding="utf-8")
            (root / "release_governance_run.stdout.json").write_text("{\"ok\":true}\n", encoding="utf-8")
            manifest = {
                "files": rg._manifest_entries(root, exclude_names=set()),
            }
            mismatches = rg._manifest_mismatches(manifest, root)
            self.assertEqual(mismatches, [])

    def test_manifest_mismatch_detects_drift_after_capture(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            p = root / "release_governance.stdout.json"
            p.write_text("{\"ok\":true}\n", encoding="utf-8")
            manifest = {
                "files": rg._manifest_entries(root, exclude_names=set()),
            }
            p.write_text("{\"ok\":false}\n", encoding="utf-8")
            mismatches = rg._manifest_mismatches(manifest, root)
            self.assertEqual(len(mismatches), 1)
            self.assertEqual(mismatches[0]["name"], "release_governance.stdout.json")

    def test_load_traceability_keys_prefers_depends_on_and_authoritative_fields(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "qa_result.json"
            p.write_text(
                (
                    "{\n"
                    '  "depends_on": ["proactive_cli_seed_r1_3", "proactive_cli_seed_r1"],\n'
                    '  "delegated_key": "proactive_cli_seed_r1_3",\n'
                    '  "active_key": "proactive_cli_reseed_r1",\n'
                    '  "key": "auto_qa_ignored_key"\n'
                    "}\n"
                ),
                encoding="utf-8",
            )
            keys = rg._load_traceability_keys(p)
            self.assertEqual(
                keys,
                [
                    "proactive_cli_seed_r1_3",
                    "proactive_cli_seed_r1",
                    "proactive_cli_reseed_r1",
                ],
            )

    def test_traceability_count_can_match_any_authoritative_key(self) -> None:
        log_text = (
            "aaaa111 order:019abfcf key:proactive_cli_seed_r1_3\n"
            "bbbb222 order:019abfcf key:proactive_cli_reseed_r1\n"
        )
        counts = [
            rg._traceability_count_from_log(
                log_text,
                order_token="order:019abfcf",
                key_token=kt,
            )
            for kt in ("key:proactive_cli_seed_r1_3", "key:proactive_cli_seed_r1")
        ]
        self.assertEqual(max(counts), 1)

    def test_required_final_artifacts_includes_run_outputs(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            checklist_path = root / "RELEASE_CHECKLIST.json"
            required = rg._required_final_artifacts(artifacts_dir=root, checklist_path=checklist_path)
            names = [p.name for p in required]
            self.assertIn("RELEASE_CHECKLIST.json", names)
            self.assertIn("RUN_PROVENANCE.json", names)
            self.assertIn("release_governance_run.stdout.json", names)
            self.assertIn("release_governance_run.exit_code.txt", names)


if __name__ == "__main__":
    unittest.main()
