from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from tools import operator_focus_report as report_tool


class TestOperatorFocusReport(unittest.TestCase):
    def test_render_handoff_markdown_includes_delegate_contract_sections(self) -> None:
        handoff = {
            "generated_at": "2026-05-01T00:00:00Z",
            "chat_id": 7,
            "selection": {"action_id": "focus:first", "rank": 1, "matched_by": "action_id"},
            "summary": {
                "returned": 1,
                "health_level": "attention",
                "top_action_id": "focus:first",
                "top_category": "release",
                "filtered_out": 0,
                "filters": {"categories": ["release"], "urgencies": ["high"], "sources": ["control_room"]},
                "available": {"categories": ["release"], "urgencies": ["high"], "sources": ["control_room"], "total": 1},
            },
            "item": {
                "rank": 1,
                "action_id": "focus:first",
                "urgency": "high",
                "category": "release",
                "label": "Unblock release handoff",
                "next_action": "Review the handoff packet.",
                "target": "backend",
                "inspect_path": "/inspect/path",
                "inspect_target": "artifact",
                "action_target": "job",
                "order_id": "order-123",
                "job_id": "job-123",
                "repo_id": "repo-123",
                "source": "control_room",
                "source_signals": ["blocked", "waiting"],
                "reason": "The release handoff is blocked.",
                "delegate_contract": {
                    "delegate_role": "implementer_local",
                    "task_title": "Unblock release handoff",
                    "source_action_id": "focus:first",
                    "handoff_endpoint": "/api/v1/orchestration/control-room?chat_id=7",
                    "inspect_endpoint": "/api/v1/orchestration/control-room/inspect?chat_id=7",
                    "acceptance_criteria": ["Targeted fix is implemented."],
                    "definition_of_done": ["Patch is validated."],
                    "evidence_required": ["validation.log"],
                    "suggested_tests": ["python3 -m unittest test_operator_focus_report.py"],
                    "risk_notes": ["Covers CLI/report rendering only."],
                    "task_prompt": "Implement the smallest handoff-safe recovery slice.",
                },
            },
        }

        rendered = report_tool.render_handoff_markdown(handoff)

        self.assertIn("# Operator Focus Handoff", rendered)
        self.assertIn("- Action id: focus:first", rendered)
        self.assertIn("## Delegate Contract", rendered)
        self.assertIn("## Acceptance Criteria", rendered)
        self.assertIn("## Definition Of Done", rendered)
        self.assertIn("## Evidence Required", rendered)
        self.assertIn("## Suggested Tests", rendered)
        self.assertIn("## Risk Notes", rendered)
        self.assertIn("## Task Prompt", rendered)
        self.assertIn("```text", rendered)
        self.assertIn("Implement the smallest handoff-safe recovery slice.", rendered)

    def test_main_handoff_writes_json_output_and_handles_missing_item(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jobs.sqlite"
            db.write_text("", encoding="utf-8")
            output = root / "handoff.json"

            selected = {
                "generated_at": "2026-05-01T00:00:00Z",
                "chat_id": 7,
                "selection": {"action_id": "focus:first", "rank": 1, "matched_by": "action_id"},
                "summary": {"returned": 1, "health_level": "attention", "top_action_id": "focus:first", "top_category": "release", "filtered_out": 0, "filters": {}, "available": {"total": 1}},
                "item": {"action_id": "focus:first", "label": "Unblock release handoff"},
            }

            with mock.patch.object(report_tool, "build_handoff", return_value=selected):
                rc = report_tool.main(
                    [
                        "--db",
                        str(db),
                        "--handoff",
                        "--action-id",
                        "focus:first",
                        "--format",
                        "json",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["item"]["action_id"], "focus:first")

            missing = {
                "generated_at": "2026-05-01T00:00:00Z",
                "chat_id": 7,
                "selection": {"action_id": "missing", "rank": None, "matched_by": "action_id"},
                "summary": {"returned": 0, "health_level": "ok", "top_action_id": None, "top_category": None, "filtered_out": 0, "filters": {}, "available": {"total": 0}},
                "item": None,
            }
            stdout = StringIO()
            with mock.patch.object(report_tool, "build_handoff", return_value=missing):
                missing_rc = report_tool.main(
                    [
                        "--db",
                        str(db),
                        "--handoff",
                        "--action-id",
                        "missing",
                        "--format",
                        "md",
                    ],
                    stdout=stdout,
                )

            self.assertEqual(missing_rc, 0)
            self.assertIn("No handoff item matched the requested selector.", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

class TestOperatorFocusReportBriefing(unittest.TestCase):
    def test_render_briefing_markdown_includes_packet_sections(self) -> None:
        briefing = {
            "generated_at": "2026-05-02T00:00:00Z",
            "chat_id": 7,
            "selection": {"action_id": "focus:first", "rank": 1, "matched_by": "action_id"},
            "summary": {
                "returned": 1,
                "health_level": "attention",
                "top_action_id": "focus:first",
                "top_category": "release",
                "filtered_out": 0,
                "filters": {"categories": ["release"], "urgencies": ["high"], "sources": ["control_room"]},
                "available": {"categories": ["release"], "urgencies": ["high"], "sources": ["control_room"], "total": 1},
            },
            "item_identity": {
                "rank": 1,
                "action_id": "focus:first",
                "urgency": "high",
                "category": "release",
                "label": "Briefing item",
                "source": "control_room",
                "order_id": "order-123",
                "job_id": "job-123",
                "repo_id": "repo-123",
            },
            "briefing_packet": {
                "owner_role": "reviewer_local",
                "action": "Review the operator focus item.",
                "inspect_endpoint": "/inspect/path",
                "handoff_endpoint": "/handoff/path",
                "evidence_required": ["validation.log"],
                "suggested_validation": ["python3 -m unittest test_operator_focus_report.py"],
                "definition_of_done": ["Packet reviewed."],
                "assignment_prompt": "Review the selected operator focus briefing.",
            },
        }

        rendered = report_tool.render_briefing_markdown(briefing)

        self.assertIn("# Operator Focus Briefing", rendered)
        self.assertIn("## Selected Item Identity", rendered)
        self.assertIn("## Briefing Packet", rendered)
        self.assertIn("## Evidence Required", rendered)
        self.assertIn("## Suggested Validation", rendered)
        self.assertIn("## Definition Of Done", rendered)
        self.assertIn("## Assignment Prompt", rendered)
        self.assertIn("```text", rendered)
        self.assertIn("Review the selected operator focus briefing.", rendered)

    def test_main_briefing_writes_json_output_and_handles_missing_packet(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jobs.sqlite"
            db.write_text("", encoding="utf-8")
            output = root / "briefing.json"

            selected = {
                "generated_at": "2026-05-02T00:00:00Z",
                "chat_id": 7,
                "selection": {"action_id": "focus:first", "rank": 1, "matched_by": "action_id"},
                "summary": {"returned": 1, "health_level": "attention", "top_action_id": "focus:first", "top_category": "release", "filtered_out": 0, "filters": {}, "available": {"total": 1}},
                "item_identity": {"action_id": "focus:first", "label": "Briefing item"},
                "briefing_packet": {"owner_role": "reviewer_local", "assignment_prompt": "Use the briefing packet."},
            }

            with mock.patch.object(report_tool, "build_briefing", return_value=selected):
                rc = report_tool.main(
                    [
                        "--db",
                        str(db),
                        "--briefing",
                        "--action-id",
                        "focus:first",
                        "--format",
                        "json",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["item_identity"]["action_id"], "focus:first")

            missing = {
                "generated_at": "2026-05-02T00:00:00Z",
                "chat_id": 7,
                "selection": {"action_id": "missing", "rank": None, "matched_by": "action_id"},
                "summary": {"returned": 0, "health_level": "ok", "top_action_id": None, "top_category": None, "filtered_out": 0, "filters": {}, "available": {"total": 0}},
                "item_identity": None,
                "briefing_packet": None,
            }
            stdout = StringIO()
            with mock.patch.object(report_tool, "build_briefing", return_value=missing):
                missing_rc = report_tool.main(
                    [
                        "--db",
                        str(db),
                        "--briefing",
                        "--action-id",
                        "missing",
                        "--format",
                        "md",
                    ],
                    stdout=stdout,
                )

            self.assertEqual(missing_rc, 0)
            self.assertIn("No operator focus briefing packet matched the requested selector.", stdout.getvalue())
