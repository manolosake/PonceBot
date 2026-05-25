from __future__ import annotations

import unittest

from tools import proactive_action_plan_report as report_tool


class TestProactiveActionPlanReport(unittest.TestCase):
    def _base_report(self) -> dict:
        return {
            "generated_at": 123.0,
            "chat_id": 7,
            "limit": 10,
            "summary": {
                "active_proactive_orders": 1,
                "returned": 1,
                "lanes": {"advance": 1},
                "selection_quality": {"ok": 1},
                "top_lane": "advance",
                "top_action": "Implement the bounded slice.",
                "next_delegate": {
                    "owner_role": "implementer_local",
                    "order_id": "advance-order",
                    "lane": "advance",
                    "action": "Implement the bounded slice.",
                    "inspect_endpoint": "/inspect/advance-order",
                    "handoff_endpoint": "/handoff/advance-order",
                    "acceptance_criteria": ["Open the order."],
                    "evidence_required": ["pytest output"],
                    "suggested_validation": ["Run focused tests."],
                    "definition_of_done": ["Tests pass."],
                    "assignment_prompt": "ROLE: implementer_local.\nAction: Implement the bounded slice.",
                    "studio_decision_evidence_contract": {
                        "required_fields": [
                            "studio_decision_evidence.candidate_bets",
                            "studio_decision_evidence.killed_bets",
                            "studio_decision_evidence.selected_bet",
                            "studio_decision_evidence.critic_answers",
                            "studio_decision_evidence.debate_summary",
                        ],
                        "suggested_validation": [
                            "python3 tools/backend_done_evidence_guard.py --artifacts-dir <dir> --require-studio-decision-evidence"
                        ],
                    },
                },
            },
            "top_execution_packet": {
                "owner_role": "implementer_local",
                "order_id": "advance-order",
                "lane": "advance",
                "action": "Implement the bounded slice.",
                "inspect_endpoint": "/inspect/advance-order",
                "handoff_endpoint": "/handoff/advance-order",
                "acceptance_criteria": ["Open the order."],
                "evidence_required": ["pytest output"],
                "suggested_validation": ["Run focused tests."],
                "definition_of_done": ["Tests pass."],
                "assignment_prompt": "ROLE: implementer_local.\nAction: Implement the bounded slice.",
                "studio_decision_evidence_contract": {
                    "required_fields": [
                        "studio_decision_evidence.candidate_bets",
                        "studio_decision_evidence.killed_bets",
                        "studio_decision_evidence.selected_bet",
                        "studio_decision_evidence.critic_answers",
                        "studio_decision_evidence.debate_summary",
                    ],
                    "suggested_validation": [
                        "python3 tools/backend_done_evidence_guard.py --artifacts-dir <dir> --require-studio-decision-evidence"
                    ],
                },
            },
            "lanes": [
                {
                    "lane": "advance",
                    "label": "Advance",
                    "count": 1,
                    "recommended_next_action": "Implement the bounded slice.",
                    "orders": [
                        {
                            "rank": 1,
                            "order_id": "advance-order",
                            "order_id_short": "advance",
                            "current_stage": "delivery",
                            "readiness_verdict": "wait",
                            "next_action": "Implement the bounded slice.",
                            "selection_quality": {
                                "status": "ok",
                                "flags": ["selection_evidence_present"],
                                "summary": "Commercial, factory-value, or studio-decision evidence is present.",
                            },
                        }
                    ],
                }
            ],
        }

    def test_render_markdown_includes_top_execution_packet_contract_and_delegate(self) -> None:
        rendered = report_tool.render_markdown(self._base_report())

        self.assertIn("## Top Execution Packet", rendered)
        self.assertIn("- owner_role: implementer_local", rendered)
        self.assertIn("- lane: advance", rendered)
        self.assertIn("- order_id: advance-order", rendered)
        self.assertIn("- action: Implement the bounded slice.", rendered)
        self.assertIn("- inspect_endpoint: /inspect/advance-order", rendered)
        self.assertIn("- handoff_endpoint: /handoff/advance-order", rendered)
        self.assertIn("- acceptance_criteria:", rendered)
        self.assertIn("  - Open the order.", rendered)
        self.assertIn("- evidence_required:", rendered)
        self.assertIn("  - pytest output", rendered)
        self.assertIn("- suggested_validation:", rendered)
        self.assertIn("  - Run focused tests.", rendered)
        self.assertIn("- definition_of_done:", rendered)
        self.assertIn("  - Tests pass.", rendered)
        self.assertIn("- studio_decision_evidence_contract:", rendered)
        self.assertIn("    - studio_decision_evidence.candidate_bets", rendered)
        self.assertIn("    - studio_decision_evidence.debate_summary", rendered)
        self.assertIn("    - python3 tools/backend_done_evidence_guard.py --artifacts-dir <dir> --require-studio-decision-evidence", rendered)
        self.assertIn("## Next Delegate", rendered)
        next_delegate_section = rendered.split("## Next Delegate", 1)[1].split("## Advance", 1)[0]
        self.assertIn("- acceptance_criteria:", next_delegate_section)
        self.assertIn("  - Open the order.", next_delegate_section)
        self.assertIn("- evidence_required:", next_delegate_section)
        self.assertIn("  - pytest output", next_delegate_section)
        self.assertIn("- suggested_validation:", next_delegate_section)
        self.assertIn("  - Run focused tests.", next_delegate_section)
        self.assertIn("- definition_of_done:", next_delegate_section)
        self.assertIn("  - Tests pass.", next_delegate_section)
        self.assertIn("- assignment_prompt: ROLE: implementer_local. Action: Implement the bounded slice.", next_delegate_section)

        self.assertIn("| Rank | Order | Stage | Verdict | Selection | Next action |", rendered)
        self.assertIn("| 1 | advance | delivery | wait | ok | Implement the bounded slice. |", rendered)

    def test_render_markdown_includes_needs_review_selection_quality_evidence_sources(self) -> None:
        report = self._base_report()
        report["summary"]["selection_quality"] = {"needs_review": 1}
        report["summary"]["lanes"] = {"selection_review": 1, "advance": 0}
        report["summary"]["top_lane"] = "selection_review"
        report["summary"]["next_delegate"]["owner_role"] = "architect_local"
        report["summary"]["next_delegate"]["lane"] = "selection_review"
        report["top_execution_packet"]["owner_role"] = "architect_local"
        report["top_execution_packet"]["lane"] = "selection_review"
        report["lanes"][0]["lane"] = "selection_review"
        report["lanes"][0]["label"] = "Selection Review"
        order = report["lanes"][0]["orders"][0]
        order["decision"] = "selection_review"
        order["selection_quality"] = {
            "status": "needs_review",
            "flags": [
                "weak_selection_evidence",
                "advance_without_commercial_factory_or_studio_evidence",
                "implementation_churn_without_validation",
            ],
            "summary": "Advance work needs a selection review before delivery continues.",
            "evidence_sources": [
                {
                    "kind": "trace",
                    "key": "factory_value",
                    "summary": "Factory-value evidence was too weak to continue.",
                }
            ],
            "churn_risk": {
                "status": "needs_review",
                "flags": ["started_slices_exceed_validated_or_closed"],
                "counters": {
                    "delivery_children": 3,
                    "proactive_slices_closed": 0,
                    "proactive_slices_started": 4,
                    "proactive_slices_validated": 0,
                },
            },
        }

        rendered = report_tool.render_markdown(report)

        self.assertIn("### Top Order Selection Quality", rendered)
        self.assertIn("- Top lane: selection_review", rendered)
        self.assertIn("- Selection Review: 1", rendered)
        self.assertIn("- owner_role: architect_local", rendered)
        self.assertIn("- lane: selection_review", rendered)
        self.assertIn("## Selection Review", rendered)
        self.assertIn("- status: needs_review", rendered)
        self.assertIn("- flags:", rendered)
        self.assertIn("  - weak_selection_evidence", rendered)
        self.assertIn("  - advance_without_commercial_factory_or_studio_evidence", rendered)
        self.assertIn("  - implementation_churn_without_validation", rendered)
        self.assertIn("- summary: Advance work needs a selection review before delivery continues.", rendered)
        self.assertIn("- evidence_sources:", rendered)
        self.assertIn("  - trace/factory_value: Factory-value evidence was too weak to continue.", rendered)
        self.assertIn("- churn_risk: needs_review", rendered)
        self.assertIn("  - started_slices_exceed_validated_or_closed", rendered)
        self.assertIn("proactive_slices_started=4", rendered)


if __name__ == "__main__":
    unittest.main()
