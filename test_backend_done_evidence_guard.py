from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from tools.backend_done_evidence_guard import main, validate_evidence


class TestBackendDoneEvidenceGuard(unittest.TestCase):
    def _write_artifact(self, artifacts_dir: Path, name: str, content: str = "ok\n") -> None:
        (artifacts_dir / name).write_text(content, encoding="utf-8")

    def _write_evidence(self, artifacts_dir: Path, payload: dict) -> None:
        (artifacts_dir / "final_evidence.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def _write_shipped_artifacts(self, artifacts_dir: Path) -> None:
        for artifact_name in ["proof.log", "changes.patch", "tests.log", "release.txt"]:
            self._write_artifact(artifacts_dir, artifact_name)

    def _shipped_payload(self, **overrides: object) -> dict:
        payload = {
            "summary": "verified improvement",
            "artifacts": ["proof.log", "changes.patch", "tests.log", "release.txt"],
            "next_action": None,
            "outcome": "shipped_to_main",
            "delivery_contract": {
                "branch": "main",
                "diff_artifact": "changes.patch",
                "validation_artifacts": ["tests.log"],
                "ship_artifacts": ["release.txt"],
            },
        }
        payload.update(overrides)
        return payload

    def _root_caused_payload(self, **overrides: object) -> dict:
        payload = {
            "summary": "verified improvement",
            "artifacts": ["proof.log"],
            "next_action": None,
            "outcome": "failed_root_caused",
            "root_cause": "the requested delivery cannot be completed without operator input",
        }
        payload.update(overrides)
        return payload

    def _commercial_evidence(self, **overrides: object) -> dict:
        commercial_evidence = {
            "monetization_or_savings_path": (
                "fewer ghost closes improve selection quality and delivery reliability"
            ),
            "evidence": "guard evidence saves review time by blocking low-value proactive work",
        }
        commercial_evidence.update(overrides)
        return commercial_evidence

    def test_validate_evidence_passes_with_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(artifacts_dir, self._shipped_payload())

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertTrue(ok)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["artifact_count"], 4)
            self.assertEqual(payload["outcome"], "shipped_to_main")

    def test_validate_evidence_default_allows_missing_factory_delta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(artifacts_dir, self._shipped_payload())

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertTrue(ok)
            self.assertTrue(payload["ok"])

    def test_validate_evidence_default_allows_missing_commercial_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(artifacts_dir, self._shipped_payload())

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertTrue(ok)
            self.assertTrue(payload["ok"])

    def test_validate_evidence_requires_commercial_evidence_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(artifacts_dir, self._shipped_payload())

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir,
                require_commercial_evidence=True,
            )

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "commercial_evidence_missing")

    def test_main_require_commercial_evidence_flag_rejects_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(artifacts_dir, self._shipped_payload())
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--artifacts-dir",
                        str(artifacts_dir),
                        "--require-commercial-evidence",
                    ]
                )

            self.assertEqual(exit_code, 2)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["reason"], "commercial_evidence_missing")

    def test_main_require_commercial_evidence_false_allows_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(artifacts_dir, self._shipped_payload())
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--artifacts-dir",
                        str(artifacts_dir),
                        "--require-commercial-evidence=false",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["ok"])

    def test_validate_evidence_requires_commercial_evidence_fields_when_enabled(self) -> None:
        cases = [
            (
                "monetization_or_savings_path",
                {
                    "evidence": (
                        "guard evidence saves review time by blocking low-value proactive work"
                    )
                },
                "commercial_evidence_monetization_path_missing",
            ),
            (
                "evidence",
                {
                    "monetization_or_savings_path": (
                        "fewer ghost closes improve selection quality and delivery reliability"
                    )
                },
                "commercial_evidence_evidence_missing",
            ),
        ]
        for name, commercial_evidence, reason in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                artifacts_dir = Path(td)
                self._write_shipped_artifacts(artifacts_dir)
                self._write_evidence(
                    artifacts_dir,
                    self._shipped_payload(commercial_evidence=commercial_evidence),
                )

                ok, payload = validate_evidence(
                    artifacts_dir=artifacts_dir,
                    require_commercial_evidence=True,
                )

                self.assertFalse(ok)
                self.assertEqual(payload["reason"], reason)

    def test_validate_evidence_rejects_generic_commercial_evidence_fields(self) -> None:
        cases = [
            (
                "monetization_or_savings_path",
                {
                    "monetization_or_savings_path": "done",
                    "evidence": (
                        "guard evidence saves review time by blocking low-value proactive work"
                    ),
                },
                "commercial_evidence_monetization_path_too_generic",
            ),
            (
                "evidence",
                {
                    "monetization_or_savings_path": (
                        "fewer ghost closes improve selection quality and delivery reliability"
                    ),
                    "evidence": "proof.log",
                },
                "commercial_evidence_evidence_too_generic",
            ),
        ]
        for name, commercial_evidence, reason in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                artifacts_dir = Path(td)
                self._write_shipped_artifacts(artifacts_dir)
                self._write_evidence(
                    artifacts_dir,
                    self._shipped_payload(commercial_evidence=commercial_evidence),
                )

                ok, payload = validate_evidence(
                    artifacts_dir=artifacts_dir,
                    require_commercial_evidence=True,
                )

                self.assertFalse(ok)
                self.assertEqual(payload["reason"], reason)

    def test_validate_evidence_rejects_non_commercial_commercial_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(
                artifacts_dir,
                self._shipped_payload(
                    commercial_evidence={
                        "monetization_or_savings_path": (
                            "updated backend docs for internal maintainers"
                        ),
                        "evidence": "documented backend cleanup notes in the project summary",
                    },
                ),
            )

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir,
                require_commercial_evidence=True,
            )

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "commercial_evidence_not_commercially_relevant")

    def test_validate_evidence_rejects_current_as_rent_false_positive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(
                artifacts_dir,
                self._shipped_payload(
                    commercial_evidence={
                        "monetization_or_savings_path": (
                            "updated current backend docs for maintainers"
                        ),
                        "evidence": "documented current backend cleanup notes",
                    },
                ),
            )

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir,
                require_commercial_evidence=True,
            )

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "commercial_evidence_not_commercially_relevant")

    def test_validate_evidence_accepts_substantive_commercial_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(
                artifacts_dir,
                self._shipped_payload(commercial_evidence=self._commercial_evidence()),
            )

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir,
                require_commercial_evidence=True,
            )

            self.assertTrue(ok)
            self.assertTrue(payload["ok"])

    def test_validate_evidence_commercial_gate_applies_to_root_caused_terminal_outcome(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(artifacts_dir, self._root_caused_payload())

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir,
                require_commercial_evidence=True,
            )

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "commercial_evidence_missing")

    def test_validate_evidence_requires_factory_delta_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(artifacts_dir, self._shipped_payload())

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir,
                require_factory_delta=True,
            )

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "factory_delta_missing")

    def test_validate_evidence_requires_factory_delta_fields_when_enabled(self) -> None:
        cases = [
            (
                "capability_changed",
                {"measurable_delta": "blocked generic terminal bundles", "evidence": "guard test"},
                "factory_delta_capability_changed_missing",
            ),
            (
                "measurable_delta",
                {
                    "capability_changed": "validation requires internal capability delta",
                    "evidence": "guard test",
                },
                "factory_delta_measurable_delta_missing",
            ),
            (
                "evidence",
                {
                    "capability_changed": "validation requires internal capability delta",
                    "measurable_delta": "blocked generic terminal bundles",
                },
                "factory_delta_evidence_missing",
            ),
        ]
        for name, factory_delta, reason in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                artifacts_dir = Path(td)
                self._write_shipped_artifacts(artifacts_dir)
                self._write_evidence(
                    artifacts_dir,
                    self._shipped_payload(factory_delta=factory_delta),
                )

                ok, payload = validate_evidence(
                    artifacts_dir=artifacts_dir,
                    require_factory_delta=True,
                )

                self.assertFalse(ok)
                self.assertEqual(payload["reason"], reason)

    def test_validate_evidence_rejects_generic_factory_delta_capability(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(
                artifacts_dir,
                self._shipped_payload(
                    factory_delta={
                        "capability_changed": "updated backend docs",
                        "measurable_delta": "blocked generic terminal bundles",
                        "evidence": "guard test",
                    },
                ),
            )

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir,
                require_factory_delta=True,
            )

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "factory_delta_capability_not_factory_relevant")

    def test_validate_evidence_rejects_reviewer_probe_generic_factory_delta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(
                artifacts_dir,
                self._shipped_payload(
                    factory_delta={
                        "capability_changed": "validation",
                        "measurable_delta": "done",
                        "evidence": "proof.log",
                    },
                ),
            )

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir,
                require_factory_delta=True,
            )

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "factory_delta_capability_changed_too_generic")

    def test_validate_evidence_rejects_generic_factory_delta_fields(self) -> None:
        cases = [
            (
                "capability_changed",
                {
                    "capability_changed": "validation",
                    "measurable_delta": "blocked generic terminal bundles",
                    "evidence": "test_backend_done_evidence_guard.py",
                },
                "factory_delta_capability_changed_too_generic",
            ),
            (
                "measurable_delta",
                {
                    "capability_changed": "validation evidence now gates delivery",
                    "measurable_delta": "done",
                    "evidence": "test_backend_done_evidence_guard.py",
                },
                "factory_delta_measurable_delta_too_generic",
            ),
            (
                "evidence",
                {
                    "capability_changed": "validation evidence now gates delivery",
                    "measurable_delta": "blocked generic terminal bundles",
                    "evidence": "proof.log",
                },
                "factory_delta_evidence_too_generic",
            ),
        ]
        for name, factory_delta, reason in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                artifacts_dir = Path(td)
                self._write_shipped_artifacts(artifacts_dir)
                self._write_evidence(
                    artifacts_dir,
                    self._shipped_payload(factory_delta=factory_delta),
                )

                ok, payload = validate_evidence(
                    artifacts_dir=artifacts_dir,
                    require_factory_delta=True,
                )

                self.assertFalse(ok)
                self.assertEqual(payload["reason"], reason)

    def test_validate_evidence_accepts_relevant_factory_delta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_shipped_artifacts(artifacts_dir)
            self._write_evidence(
                artifacts_dir,
                self._shipped_payload(
                    factory_delta={
                        "capability_changed": "validation and learning evidence now gates delivery",
                        "measurable_delta": "generic terminal bundles fail without explicit delta",
                        "evidence": "test_backend_done_evidence_guard.py",
                    },
                ),
            )

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir,
                require_factory_delta=True,
            )

            self.assertTrue(ok)
            self.assertTrue(payload["ok"])

    def test_validate_evidence_fails_when_shipped_outcome_missing_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                {
                    "summary": "verified improvement",
                    "artifacts": ["proof.log"],
                    "next_action": None,
                    "outcome": "shipped_to_main",
                },
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "delivery_contract_missing")

    def test_validate_evidence_fails_when_shipped_outcome_missing_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                {
                    "summary": "verified improvement",
                    "artifacts": ["proof.log"],
                    "next_action": None,
                    "outcome": "shipped_to_main",
                    "delivery_contract": {
                        "diffstat": "1 file changed",
                        "validation": "pytest PASS",
                        "ship_evidence": "merged to main",
                    },
                },
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "branch_missing")

    def test_validate_evidence_fails_when_shipped_outcome_missing_diff_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                {
                    "summary": "verified improvement",
                    "artifacts": ["proof.log"],
                    "next_action": None,
                    "outcome": "shipped_to_main",
                    "delivery_contract": {
                        "branch": "main",
                        "validation": "pytest PASS",
                        "ship_evidence": "merged to main",
                    },
                },
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "diff_evidence_missing")

    def test_validate_evidence_fails_when_shipped_outcome_missing_validation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                {
                    "summary": "verified improvement",
                    "artifacts": ["proof.log"],
                    "next_action": None,
                    "outcome": "shipped_to_main",
                    "delivery_contract": {
                        "branch": "main",
                        "diffstat": "1 file changed",
                        "ship_evidence": "merged to main",
                    },
                },
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "validation_evidence_missing")

    def test_validate_evidence_fails_when_shipped_outcome_missing_ship_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                {
                    "summary": "verified improvement",
                    "artifacts": ["proof.log"],
                    "next_action": None,
                    "outcome": "shipped_to_main",
                    "delivery_contract": {
                        "branch": "main",
                        "diffstat": "1 file changed",
                        "validation": "pytest PASS",
                    },
                },
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "ship_evidence_missing")

    def test_validate_evidence_fails_when_nested_delivery_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                {
                    "summary": "verified improvement",
                    "artifacts": ["proof.log"],
                    "next_action": None,
                    "outcome": "shipped_to_main",
                    "delivery_contract": {
                        "branch": "main",
                        "diff_artifact": "missing.patch",
                        "validation": "pytest PASS",
                        "ship_evidence": "merged to main",
                    },
                },
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "delivery_contract_artifact_missing")
            self.assertEqual(len(payload["missing_artifacts"]), 1)

    def test_validate_evidence_fails_when_nested_delivery_artifact_is_missing_with_allow_missing_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                {
                    "summary": "verified improvement",
                    "artifacts": ["proof.log"],
                    "next_action": None,
                    "outcome": "shipped_to_main",
                    "delivery_contract": {
                        "branch": "main",
                        "diff_artifact": "missing.patch",
                        "validation": "pytest PASS",
                        "ship_evidence": "merged to main",
                    },
                },
            )

            ok, payload = validate_evidence(
                artifacts_dir=artifacts_dir,
                allow_missing_files=True,
            )

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "delivery_contract_artifact_missing")
            self.assertEqual(len(payload["missing_artifacts"]), 1)

    def test_validate_evidence_fails_when_nested_delivery_artifact_list_item_is_missing(
        self,
    ) -> None:
        cases = [
            (
                "validation_artifacts",
                {
                    "branch": "main",
                    "diffstat": "1 file changed",
                    "validation_artifacts": ["missing-tests.log"],
                    "ship_evidence": "merged to main",
                },
            ),
            (
                "ship_artifacts",
                {
                    "branch": "main",
                    "diffstat": "1 file changed",
                    "validation": "pytest PASS",
                    "ship_artifacts": ["missing-release.json"],
                },
            ),
        ]
        for name, delivery_contract in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                artifacts_dir = Path(td)
                self._write_artifact(artifacts_dir, "proof.log")
                self._write_evidence(
                    artifacts_dir,
                    {
                        "summary": "verified improvement",
                        "artifacts": ["proof.log"],
                        "next_action": None,
                        "outcome": "shipped_to_main",
                        "delivery_contract": delivery_contract,
                    },
                )

                ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

                self.assertFalse(ok)
                self.assertEqual(payload["reason"], "delivery_contract_artifact_missing")

    def test_validate_evidence_fails_when_nested_delivery_artifact_escapes_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                {
                    "summary": "verified improvement",
                    "artifacts": ["proof.log"],
                    "next_action": None,
                    "outcome": "shipped_to_main",
                    "delivery_contract": {
                        "branch": "main",
                        "diff_artifact": "../escape.patch",
                        "validation": "pytest PASS",
                        "ship_evidence": "merged to main",
                    },
                },
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "delivery_contract_artifact_outside_dir")

    def test_validate_evidence_passes_with_root_caused_terminal_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(artifacts_dir, self._root_caused_payload())

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertTrue(ok)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["outcome"], "failed_root_caused")

    def test_validate_evidence_fails_when_root_caused_outcome_missing_root_cause(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(artifacts_dir, self._root_caused_payload(root_cause=""))

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "root_cause_missing")

    def test_validate_evidence_fails_when_outcome_is_missing_or_invalid(self) -> None:
        cases = [
            ("missing", {}, "outcome_missing"),
            ("blank", {"outcome": " "}, "outcome_missing"),
            ("invalid", {"outcome": "done"}, "outcome_invalid"),
        ]
        for name, overrides, reason in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                artifacts_dir = Path(td)
                self._write_artifact(artifacts_dir, "proof.log")
                payload = self._root_caused_payload(**overrides)
                if name == "missing":
                    payload.pop("outcome")
                self._write_evidence(artifacts_dir, payload)

                ok, result = validate_evidence(artifacts_dir=artifacts_dir)

                self.assertFalse(ok)
                self.assertEqual(result["reason"], reason)

    def test_validate_evidence_fails_when_next_action_is_string(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                self._root_caused_payload(next_action="follow up with reviewer"),
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "next_action_open")

    def test_validate_evidence_fails_when_next_action_is_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                self._root_caused_payload(next_action={"step": "follow up"}),
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "next_action_open")

    def test_validate_evidence_fails_when_next_action_is_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_artifact(artifacts_dir, "proof.log")
            self._write_evidence(
                artifacts_dir,
                self._root_caused_payload(next_action=["follow up"]),
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "next_action_open")

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
            self._write_evidence(
                artifacts_dir,
                self._root_caused_payload(artifacts=["missing.log"]),
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
            self._write_evidence(artifacts_dir, self._root_caused_payload(artifacts=["proof"]))

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "artifact_not_file")
            self.assertEqual(payload["not_file_artifacts"], [str(proof_dir.resolve())])

    def test_validate_evidence_fails_when_referenced_artifact_is_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            proof = artifacts_dir / "proof.log"
            self._write_artifact(artifacts_dir, "proof.log", "")
            self._write_evidence(artifacts_dir, self._root_caused_payload())

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "artifact_empty")
            self.assertEqual(payload["empty_artifacts"], [str(proof.resolve())])

    def test_validate_evidence_fails_when_artifacts_list_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_evidence(artifacts_dir, self._root_caused_payload(artifacts=[]))

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "artifacts_empty")

    def test_validate_evidence_fails_when_relative_artifact_escapes_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            self._write_evidence(
                artifacts_dir,
                self._root_caused_payload(artifacts=["../escape.log"]),
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
                self._write_evidence(
                    artifacts_dir,
                    self._root_caused_payload(artifacts=[unsafe_path]),
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
            self._write_evidence(
                artifacts_dir,
                self._root_caused_payload(artifacts=[str(outside_artifact)]),
            )

            ok, payload = validate_evidence(artifacts_dir=artifacts_dir)

            self.assertFalse(ok)
            self.assertEqual(payload["reason"], "artifact_outside_dir")


if __name__ == "__main__":
    unittest.main()
