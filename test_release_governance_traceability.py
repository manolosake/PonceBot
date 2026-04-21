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


if __name__ == "__main__":
    unittest.main()
