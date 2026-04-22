import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

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

    def test_manifest_mismatch_handles_malformed_size_metadata(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            p = root / "release_governance.stdout.json"
            p.write_text("{\"ok\":true}\n", encoding="utf-8")
            mismatches = rg._manifest_mismatches(
                {"files": [{"name": "release_governance.stdout.json", "size": "not-an-int", "sha256": "abc"}]},
                root,
            )
            self.assertEqual(len(mismatches), 1)
            self.assertEqual(mismatches[0]["name"], "release_governance.stdout.json")
            self.assertEqual(mismatches[0]["reason"], "malformed_metadata")
            self.assertEqual(mismatches[0]["field"], "size")

    def test_manifest_mismatch_handles_malformed_sha_metadata(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            p = root / "release_governance.stdout.json"
            p.write_text("{\"ok\":true}\n", encoding="utf-8")
            mismatches = rg._manifest_mismatches(
                {"files": [{"name": "release_governance.stdout.json", "size": p.stat().st_size, "sha256": "xyz"}]},
                root,
            )
            self.assertEqual(len(mismatches), 1)
            self.assertEqual(mismatches[0]["name"], "release_governance.stdout.json")
            self.assertEqual(mismatches[0]["reason"], "malformed_metadata")
            self.assertEqual(mismatches[0]["field"], "sha256")

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
            self.assertIn("command_transcript.jsonl", names)
            self.assertIn("test_logs.txt", names)
            self.assertIn("release_governance_run.stdout.json", names)
            self.assertIn("release_governance_run.exit_code.txt", names)

    def test_finalize_manifest_detects_post_capture_drift(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            lock_path = root / ".finalize.lock"
            lock_path.write_text("lock\n", encoding="utf-8")
            (root / "test_logs.txt").write_text("ok\n", encoding="utf-8")
            with patch.object(
                rg,
                "_manifest_mismatches",
                side_effect=[[], [{"name": "test_logs.txt", "reason": "hash_or_size_mismatch"}]],
            ):
                manifest = rg._finalize_manifest(artifacts_dir=root, lock_path=lock_path)
            self.assertEqual(manifest["pre_capture_mismatch_count"], 0)
            self.assertEqual(manifest["mismatch_count"], 1)
            self.assertEqual(manifest["mismatches"][0]["name"], "test_logs.txt")

    def test_manifest_post_write_violations_detects_mtime_drift(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            covered = root / "command_transcript.jsonl"
            covered.write_text("initial\n", encoding="utf-8")
            manifest_path = root / "FINAL_MANIFEST.json"
            manifest_path.write_text(
                '{"files":[{"name":"command_transcript.jsonl","size":8,"sha256":"x"}]}\n',
                encoding="utf-8",
            )
            # Ensure mtime drift after manifest by rewriting covered file.
            time.sleep(0.01)
            covered.write_text("modified\n", encoding="utf-8")
            violations = rg._manifest_post_write_violations(
                manifest={"files": [{"name": "command_transcript.jsonl"}]},
                artifacts_dir=root,
                manifest_path=manifest_path,
            )
            self.assertEqual(len(violations), 1)
            self.assertEqual(violations[0]["name"], "command_transcript.jsonl")

    def test_finalize_manifest_ignores_non_covered_side_logs(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            lock_path = root / ".finalize.lock"
            lock_path.write_text("lock\n", encoding="utf-8")
            (root / "CHANGED_FILES.txt").write_text("a.py\n", encoding="utf-8")
            (root / "command_transcript.jsonl").write_text('{"k":"v"}\n', encoding="utf-8")
            (root / "release_governance_run.invocation.stdout.log").write_text("volatile\n", encoding="utf-8")
            manifest = rg._finalize_manifest(artifacts_dir=root, lock_path=lock_path)
            names = [f["name"] for f in manifest.get("files", [])]
            self.assertIn("CHANGED_FILES.txt", names)
            self.assertIn("command_transcript.jsonl", names)
            self.assertNotIn("release_governance_run.invocation.stdout.log", names)


if __name__ == "__main__":
    unittest.main()
