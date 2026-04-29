from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.backend_done_evidence_guard import validate_evidence


class TestBackendDoneEvidenceGuard(unittest.TestCase):
    def test_validate_evidence_passes_with_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            proof = artifacts_dir / "proof.log"
            proof.write_text("ok\n", encoding="utf-8")
            (artifacts_dir / "final_evidence.json").write_text(
                json.dumps(
                    {
                        "summary": "verified improvement",
                        "artifacts": ["proof.log"],
                        "next_action": None,
                    }
                ),
                encoding="utf-8",
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertTrue(ok)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["artifact_count"], 1)

    def test_validate_evidence_fails_when_summary_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ok, payload = validate_evidence(artifacts_dir=Path(td))

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "summary_missing")

    def test_validate_evidence_fails_when_summary_escapes_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir, summary_name="../escape.json"
            )

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "summary_outside_dir")

    def test_validate_evidence_fails_when_referenced_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            (artifacts_dir / "final_evidence.json").write_text(
                json.dumps(
                    {
                        "summary": "verified improvement",
                        "artifacts": ["missing.log"],
                        "next_action": None,
                    }
                ),
                encoding="utf-8",
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "artifact_missing")
            self.assertEqual(len(payload["missing_artifacts"]), 1)

    def test_validate_evidence_fails_when_referenced_artifact_is_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            proof_dir = artifacts_dir / "proof"
            proof_dir.mkdir()
            (artifacts_dir / "final_evidence.json").write_text(
                json.dumps(
                    {
                        "summary": "verified improvement",
                        "artifacts": ["proof"],
                        "next_action": None,
                    }
                ),
                encoding="utf-8",
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "artifact_not_file")
            self.assertEqual(payload["not_file_artifacts"], [str(proof_dir.resolve())])

    def test_validate_evidence_fails_when_referenced_artifact_is_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            proof = artifacts_dir / "proof.log"
            proof.write_text("", encoding="utf-8")
            (artifacts_dir / "final_evidence.json").write_text(
                json.dumps(
                    {
                        "summary": "verified improvement",
                        "artifacts": ["proof.log"],
                        "next_action": None,
                    }
                ),
                encoding="utf-8",
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "artifact_empty")
            self.assertEqual(payload["empty_artifacts"], [str(proof.resolve())])

    def test_validate_evidence_fails_when_artifacts_list_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            (artifacts_dir / "final_evidence.json").write_text(
                json.dumps(
                    {
                        "summary": "verified improvement",
                        "artifacts": [],
                        "next_action": None,
                    }
                ),
                encoding="utf-8",
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "artifacts_empty")

    def test_validate_evidence_fails_when_relative_artifact_escapes_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            (artifacts_dir / "final_evidence.json").write_text(
                json.dumps(
                    {
                        "summary": "verified improvement",
                        "artifacts": ["../escape.log"],
                        "next_action": None,
                    }
                ),
                encoding="utf-8",
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "artifact_outside_dir")

    def test_validate_evidence_fails_when_windows_artifact_path_escapes_dir(self) -> None:
        unsafe_paths = [
            "..\\escape.log",
            "C:/temp/proof.log",
            "C:\\temp\\proof.log",
            "\\\\server\\share\\proof.log",
        ]
        for unsafe_path in unsafe_paths:
            with self.subTest(unsafe_path=unsafe_path), tempfile.TemporaryDirectory() as td:
                artifacts_dir = Path(td)
                (artifacts_dir / "final_evidence.json").write_text(
                    json.dumps(
                        {
                            "summary": "verified improvement",
                            "artifacts": [unsafe_path],
                            "next_action": None,
                        }
                    ),
                    encoding="utf-8",
                )

                ok, payload = validate_evidence(
                    artifacts_dir=artifacts_dir,
                    allow_missing_files=True,
                )

                self.assertFalse(ok)
                self.assertEqual(payload["reason"], "artifact_outside_dir")

    def test_validate_evidence_fails_when_absolute_artifact_escapes_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            artifacts_dir = Path(td)
            outside_artifact = Path(outside_td) / "escape.log"
            outside_artifact.write_text("nope\n", encoding="utf-8")
            (artifacts_dir / "final_evidence.json").write_text(
                json.dumps(
                    {
                        "summary": "verified improvement",
                        "artifacts": [str(outside_artifact)],
                        "next_action": None,
                    }
                ),
                encoding="utf-8",
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "artifact_outside_dir")


if __name__ == "__main__":
    unittest.main()
