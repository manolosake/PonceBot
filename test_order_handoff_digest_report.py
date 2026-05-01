from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from tools import order_handoff_digest_report as report_tool


class TestOrderHandoffDigestReport(unittest.TestCase):
    def test_render_markdown_includes_checks_blockers_and_actions(self) -> None:
        report = {
            "order_id": "order-123",
            "phase": "handoff",
            "current_stage": "qa",
            "state": "blocked",
            "verdict": "needs_attention",
            "summary": "Release checks are blocked.",
            "next_action": "Clear the blocker and re-run the handoff.",
            "order": {
                "title": "Release handoff",
                "status": "active",
            },
            "readiness": {
                "applies": True,
                "scope": "release",
                "merge_ready": False,
                "merged_to_main": False,
                "deploy_status": "blocked",
                "deploy_summary": "Waiting on QA",
                "deployed_commit": None,
                "checks_total": 1,
                "checks_by_status": {"blocked": 1},
            },
            "evidence_counts": {
                "children": 1,
                "traces": 2,
                "decision_log": 1,
                "delegation_log": 0,
                "artifacts": 1,
                "handoff_refs": 1,
            },
            "checks": [
                {
                    "key": "qa",
                    "status": "blocked",
                    "evidence_count": 1,
                    "summary": "Waiting on QA signoff.",
                }
            ],
            "blockers": [
                {
                    "stage": "qa",
                    "summary": "QA signoff missing",
                    "job": {"job_id_short": "deadbeef"},
                }
            ],
            "recent_jobs": [
                {
                    "job_id_short": "deadbeef",
                    "role": "reviewer_local",
                    "state": "blocked",
                    "title": "QA signoff",
                    "result_summary": "Blocked pending evidence",
                    "result_next_action": "Review latest artifacts",
                }
            ],
            "recent_artifacts": [
                {
                    "kind": "file",
                    "role": "backend",
                    "job_id_short": "cafebabe",
                    "path": "/tmp/evidence.txt",
                    "artifact_id": "artifact-1",
                }
            ],
            "evidence_refs": [
                {
                    "kind": "artifact",
                    "path": "/tmp/evidence.txt",
                    "summary": "Evidence file",
                }
            ],
            "operator_actions": ["Inspect the blocker details."],
            "release_manager_actions": ["Decide whether to hold the release."],
        }

        rendered = report_tool.render_markdown(report)

        self.assertIn("# Order Handoff Digest", rendered)
        self.assertIn("| qa | blocked | 1 | Waiting on QA signoff. |", rendered)
        self.assertIn("QA signoff missing", rendered)
        self.assertIn("Inspect the blocker details.", rendered)
        self.assertIn("Decide whether to hold the release.", rendered)

    def test_main_writes_json_output_and_missing_order_returns_1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jobs.sqlite"
            db.write_text("", encoding="utf-8")
            output = root / "report.json"

            report = {
                "order_id": "order-123",
                "summary": "Ready",
                "checks": [],
                "blockers": [],
                "recent_jobs": [],
                "recent_artifacts": [],
                "evidence_refs": [],
                "operator_actions": [],
                "release_manager_actions": [],
            }

            with mock.patch.object(report_tool, "build_report", return_value=report):
                rc = report_tool.main(
                    [
                        "--db",
                        str(db),
                        "--order-id",
                        "order-123",
                        "--format",
                        "json",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertTrue(output.exists())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["order_id"], "order-123")

            stderr = StringIO()
            with mock.patch.object(report_tool, "build_report", return_value=None):
                missing_rc = report_tool.main(
                    [
                        "--db",
                        str(db),
                        "--order-id",
                        "missing-order",
                    ],
                    stderr=stderr,
                )

            self.assertEqual(missing_rc, 1)
            self.assertIn("order not found: missing-order", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
