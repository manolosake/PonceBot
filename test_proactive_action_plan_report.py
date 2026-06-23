from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def _base_receipt_report(self) -> dict:
        return {
            "generated_at": 123.0,
            "chat_id": 7,
            "selection": {"order_id": "advance-order", "rank": 1, "matched_by": "order_id"},
            "summary": {
                "active_proactive_orders": 1,
                "returned": 1,
                "top_lane": "advance",
                "top_action": "Implement the bounded slice.",
            },
            "order_identity": {
                "rank": 1,
                "order_id": "advance-order",
                "order_id_short": "advance",
                "title": "Advance order",
                "priority": 1,
                "phase": "executing",
                "current_stage": "delivery",
                "readiness_state": "not_ready",
                "readiness_verdict": "wait",
                "decision": "advance",
                "next_action": "Implement the bounded slice.",
                "selection_quality": {
                    "status": "needs_review",
                    "summary": "Advance work needs a selection review before delivery continues.",
                    "flags": ["weak_selection_evidence"],
                    "evidence_sources": [
                        {
                            "kind": "trace",
                            "key": "factory_value",
                            "summary": "Factory-value evidence was too weak to continue.",
                        }
                    ],
                    "recommended_owner_role": "architect_local",
                    "delegation_reason": "weak_selection_evidence",
                    "delegation_focus": "deep_improvement_selection_quality",
                },
                "handoff": {
                    "suggested_role": "implementer_local",
                    "suggested_endpoint": "/handoff/advance-order",
                    "inspect_path": "/inspect/advance-order",
                    "assignment_prompt": "ROLE: implementer_local.\nAction: Implement the bounded slice.",
                    "evidence_expectations": ["pytest output"],
                    "suggested_validation": ["Run focused tests."],
                    "definition_of_done": ["Tests pass."],
                },
            },
            "receipt": {
                "event_type": "proactive_action_plan_receipt",
                "state": "acknowledged",
                "summary": "Delegated the bounded slice.",
                "next_action": "Track implementation receipt.",
                "actor": "implementer_local",
                "details": {"note": "delegate now"},
                "persisted": True,
                "persistence_reason": "decision_log_appended",
                "order_id": "advance-order",
                "job_id": "advance-order",
            },
            "persisted_receipt": {
                "state": "acknowledged",
                "summary": "Delegated the bounded slice.",
                "next_action": "Track implementation receipt.",
                "actor": "implementer_local",
                "recorded_at": 200.0,
                "order_id": "advance-order",
                "job_id": "advance-order",
                "details": {"note": "delegate now"},
            },
            "receipt_count": 1,
            "receipt_counts_by_state": {"acknowledged": 1},
            "receipt_history": [
                {
                    "state": "acknowledged",
                    "summary": "Delegated the bounded slice.",
                    "next_action": "Track implementation receipt.",
                    "actor": "implementer_local",
                    "recorded_at": 200.0,
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

    def test_render_markdown_includes_publication_recovery_section(self) -> None:
        report = self._base_report()
        report["publication_recovery"] = {
            "count": 2,
            "truncated": True,
            "items": [
                {
                    "project_name": "SignalDeck",
                    "project_path": "/home/aponce/signaldeck",
                    "github_repo": "manolosake/signaldeck",
                    "required_action": "resolve_publication_contract",
                    "reason": "Private visibility confirmation is still missing.",
                    "missing_json": '["private"]',
                    "missing_fields": ["private"],
                    "status": "open",
                    "github_url": "https://github.com/manolosake/signaldeck",
                    "latest_head": "abc1234def5678",
                    "source_order_id": "order-17",
                },
                {
                    "project_name": "Local Only Project",
                    "project_path": "/home/aponce/local-only-project",
                    "required_action": "archive_or_reject_missing_path",
                    "reason": "No GitHub publication target was found.",
                    "missing_json": '["github_repo", "github_url"]',
                    "missing_fields": ["github_repo", "github_url"],
                    "status": "open",
                },
            ],
        }

        rendered = report_tool.render_markdown(report)

        self.assertIn("## Publication Recovery", rendered)
        self.assertIn("- Open items: 2", rendered)
        self.assertIn("- Truncated: True", rendered)
        self.assertIn("| Project | Target | Evidence | Required action | Missing | Status |", rendered)
        self.assertIn("| SignalDeck | manolosake/signaldeck | url=https://github.com/manolosake/signaldeck; head=abc1234def5678; order=order-17 | resolve_publication_contract | private | open: Private visibility confirmation is still missing. |", rendered)
        self.assertIn("| Local Only Project | /home/aponce/local-only-project | - | archive_or_reject_missing_path | github_repo, github_url | open: No GitHub publication target was found. |", rendered)

    def test_render_markdown_surfaces_publication_recovery_delegate_without_proactive_orders(self) -> None:
        report = {
            "generated_at": 123.0,
            "chat_id": 7,
            "limit": 10,
            "summary": {
                "active_proactive_orders": 0,
                "returned": 1,
                "lanes": {"publication_recovery": 1},
                "selection_quality": {},
                "top_lane": "publication_recovery",
                "top_action": "Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof.",
                "next_delegate": {
                    "owner_role": "release_mgr",
                    "order_id": "order-17",
                    "lane": "publication_recovery",
                    "action": "Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof.",
                    "inspect_endpoint": "/api/v1/orchestration/proactive-action-plan",
                    "handoff_endpoint": "/api/v1/orchestration/proactive-action-plan",
                    "acceptance_criteria": ["Inspect the open publication recovery entry for SignalDeck in /api/v1/orchestration/proactive-action-plan."],
                    "evidence_required": ["publication recovery decision summary", "github_url", "private"],
                    "suggested_validation": ["Re-open the proactive action plan report and confirm the publication recovery item no longer needs follow-up."],
                    "definition_of_done": ["The publication recovery item is either resolved with publication evidence or explicitly archived/rejected with rationale."],
                    "assignment_prompt": "ROLE: release_mgr.\nAction: Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof.",
                },
            },
            "top_execution_packet": {
                "owner_role": "release_mgr",
                "order_id": "order-17",
                "lane": "publication_recovery",
                "action": "Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof.",
                "inspect_endpoint": "/api/v1/orchestration/proactive-action-plan",
                "handoff_endpoint": "/api/v1/orchestration/proactive-action-plan",
                "acceptance_criteria": ["Inspect the open publication recovery entry for SignalDeck in /api/v1/orchestration/proactive-action-plan."],
                "evidence_required": ["publication recovery decision summary", "github_url", "private"],
                "suggested_validation": ["Re-open the proactive action plan report and confirm the publication recovery item no longer needs follow-up."],
                "definition_of_done": ["The publication recovery item is either resolved with publication evidence or explicitly archived/rejected with rationale."],
                "assignment_prompt": "ROLE: release_mgr.\nAction: Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof.",
            },
            "lanes": [
                {
                    "lane": "publication_recovery",
                    "label": "Publication Recovery",
                    "count": 1,
                    "recommended_next_action": "Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof.",
                    "execution_packet": {
                        "owner_role": "release_mgr",
                        "order_id": "order-17",
                        "lane": "publication_recovery",
                        "action": "Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof.",
                    },
                    "orders": [
                        {
                            "rank": 1,
                            "order_id": "order-17",
                            "order_id_short": "pub-recov",
                            "current_stage": "deploy",
                            "readiness_verdict": "wait",
                            "next_action": "Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof.",
                        }
                    ],
                }
            ],
            "publication_recovery": {
                "count": 1,
                "truncated": False,
                "items": [
                    {
                        "project_name": "SignalDeck",
                        "project_path": "/home/aponce/signaldeck",
                        "github_repo": "manolosake/signaldeck",
                        "required_action": "resolve_publication_contract",
                        "reason": "Private visibility confirmation is still missing.",
                        "missing_json": '["github_url", "private"]',
                        "missing_fields": ["github_url", "private"],
                        "status": "open",
                        "github_url": "https://github.com/manolosake/signaldeck",
                        "latest_head": "abc1234def5678",
                        "source_order_id": "order-17",
                    }
                ],
            },
        }

        rendered = report_tool.render_markdown(report)

        self.assertIn("- Active proactive orders: 0", rendered)
        self.assertIn("- Top lane: publication_recovery", rendered)
        self.assertIn("## Top Execution Packet", rendered)
        self.assertIn("- owner_role: release_mgr", rendered)
        self.assertIn("- lane: publication_recovery", rendered)
        self.assertIn("- order_id: order-17", rendered)
        self.assertIn("## Next Delegate", rendered)
        self.assertIn("- action: Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof.", rendered)
        self.assertIn("## Publication Recovery", rendered)
        self.assertIn("## Publication Recovery\n\n- Recommended next action: Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof.", rendered)
        self.assertIn("| 1 | pub-recov | deploy | wait | - | Recover publication evidence for SignalDeck by confirming the GitHub target, head, and publication visibility proof. |", rendered)

    def test_render_markdown_includes_latest_head_backfill_publication_recovery_contract(self) -> None:
        report = {
            "generated_at": 123.0,
            "chat_id": 7,
            "limit": 10,
            "summary": {
                "active_proactive_orders": 0,
                "returned": 1,
                "lanes": {"publication_recovery": 1},
                "selection_quality": {},
                "top_lane": "publication_recovery",
                "top_action": "Backfill the latest published head for SignalDeck or validate the private GitHub target if the head cannot be confirmed.",
                "next_delegate": {
                    "owner_role": "release_mgr",
                    "order_id": "order-head",
                    "lane": "publication_recovery",
                    "action": "Backfill the latest published head for SignalDeck or validate the private GitHub target if the head cannot be confirmed.",
                    "inspect_endpoint": "/api/v1/orchestration/proactive-action-plan",
                    "handoff_endpoint": "/api/v1/orchestration/proactive-action-plan",
                    "acceptance_criteria": [
                        "Inspect the open publication recovery entry for SignalDeck in /api/v1/orchestration/proactive-action-plan."
                    ],
                    "evidence_required": [
                        "latest local Git head backfill evidence or proof that the private GitHub target was validated without a new head",
                        "recorded head or visibility verification summary tied to the recovered publication target",
                    ],
                    "suggested_validation": [
                        "Confirm the recovery row now records latest_head or an explicit private GitHub validation blocker."
                    ],
                    "definition_of_done": [
                        "The latest_head field is backfilled from current publication evidence or the private GitHub target is explicitly validated with the remaining blocker recorded."
                    ],
                    "assignment_prompt": "ROLE: release_mgr.\nAction: Backfill the latest published head for SignalDeck or validate the private GitHub target if the head cannot be confirmed.",
                },
            },
            "top_execution_packet": {
                "owner_role": "release_mgr",
                "order_id": "order-head",
                "lane": "publication_recovery",
                "action": "Backfill the latest published head for SignalDeck or validate the private GitHub target if the head cannot be confirmed.",
                "inspect_endpoint": "/api/v1/orchestration/proactive-action-plan",
                "handoff_endpoint": "/api/v1/orchestration/proactive-action-plan",
                "acceptance_criteria": [
                    "Inspect the open publication recovery entry for SignalDeck in /api/v1/orchestration/proactive-action-plan."
                ],
                "evidence_required": [
                    "latest local Git head backfill evidence or proof that the private GitHub target was validated without a new head",
                    "recorded head or visibility verification summary tied to the recovered publication target",
                ],
                "suggested_validation": [
                    "Confirm the recovery row now records latest_head or an explicit private GitHub validation blocker."
                ],
                "definition_of_done": [
                    "The latest_head field is backfilled from current publication evidence or the private GitHub target is explicitly validated with the remaining blocker recorded."
                ],
                "assignment_prompt": "ROLE: release_mgr.\nAction: Backfill the latest published head for SignalDeck or validate the private GitHub target if the head cannot be confirmed.",
            },
            "lanes": [
                {
                    "lane": "publication_recovery",
                    "label": "Publication Recovery",
                    "count": 1,
                    "recommended_next_action": "Backfill the latest published head for SignalDeck or validate the private GitHub target if the head cannot be confirmed.",
                    "execution_packet": {
                        "owner_role": "release_mgr",
                        "order_id": "order-head",
                        "lane": "publication_recovery",
                        "action": "Backfill the latest published head for SignalDeck or validate the private GitHub target if the head cannot be confirmed.",
                    },
                    "orders": [
                        {
                            "rank": 1,
                            "order_id": "order-head",
                            "order_id_short": "pub-head",
                            "current_stage": "deploy",
                            "readiness_verdict": "wait",
                            "next_action": "Backfill the latest published head for SignalDeck or validate the private GitHub target if the head cannot be confirmed.",
                        }
                    ],
                }
            ],
            "publication_recovery": {
                "count": 1,
                "truncated": False,
                "items": [
                    {
                        "project_name": "SignalDeck",
                        "project_path": "/home/aponce/signaldeck",
                        "github_repo": "manolosake/signaldeck",
                        "required_action": "backfill_latest_head_or_validate_private_github",
                        "reason": "Private GitHub target is known but latest_head evidence is missing.",
                        "missing_json": '["latest_head"]',
                        "missing_fields": ["latest_head"],
                        "status": "open",
                        "github_url": "https://github.com/manolosake/signaldeck",
                        "source_order_id": "order-head",
                    }
                ],
            },
        }

        rendered = report_tool.render_markdown(report)

        self.assertIn(
            "latest local Git head backfill evidence or proof that the private GitHub target was validated without a new head",
            rendered,
        )
        self.assertIn(
            "The latest_head field is backfilled from current publication evidence or the private GitHub target is explicitly validated with the remaining blocker recorded.",
            rendered,
        )
        self.assertIn(
            "Confirm the recovery row now records latest_head or an explicit private GitHub validation blocker.",
            rendered,
        )
        self.assertIn(
            "| SignalDeck | manolosake/signaldeck | url=https://github.com/manolosake/signaldeck; order=order-head | backfill_latest_head_or_validate_private_github | latest_head | open: Private GitHub target is known but latest_head evidence is missing. |",
            rendered,
        )

    def test_render_markdown_escapes_publication_recovery_pipe_characters(self) -> None:
        report = self._base_report()
        report["publication_recovery"] = {
            "count": 1,
            "truncated": False,
            "items": [
                {
                    "project_name": "Signal|Deck",
                    "project_path": "/home/aponce/signal|deck",
                    "required_action": "resolve|publication_contract",
                    "reason": "Confirm private|visibility before publish.",
                    "missing_json": '["github_repo|github_url"]',
                    "missing_fields": ["github_repo|github_url"],
                    "status": "open|queued",
                    "github_url": "https://github.com/org/signal|deck",
                    "latest_head": "abc|123",
                    "source_order_id": "order|42",
                }
            ],
        }

        rendered = report_tool.render_markdown(report)

        self.assertIn("| Signal\\|Deck | /home/aponce/signal\\|deck | url=https://github.com/org/signal\\|deck; head=abc\\|123; order=order\\|42 | resolve\\|publication_contract | github_repo\\|github_url | open\\|queued: Confirm private\\|visibility before publish. |", rendered)

    def test_render_markdown_degrades_when_publication_recovery_evidence_fields_are_missing_or_malformed(self) -> None:
        report = self._base_report()
        report["publication_recovery"] = {
            "count": 1,
            "truncated": False,
            "items": [
                {
                    "project_name": "Malformed Recovery",
                    "project_path": "/home/aponce/malformed",
                    "required_action": "resolve_publication_contract",
                    "reason": None,
                    "missing_json": "{not-json",
                    "missing_fields": [],
                    "status": 17,
                    "github_url": "",
                    "latest_head": None,
                    "source_order_id": [],
                }
            ],
        }

        rendered = report_tool.render_markdown(report)

        self.assertIn("| Malformed Recovery | /home/aponce/malformed | - | resolve_publication_contract | {not-json | 17 |", rendered)

    def test_render_receipt_markdown_includes_selection_order_and_persisted_sections(self) -> None:
        rendered = report_tool.render_receipt_markdown(self._base_receipt_report())

        self.assertIn("# Proactive Action Plan Receipt", rendered)
        self.assertIn("## Selection", rendered)
        self.assertIn("- Order id: advance-order", rendered)
        self.assertIn("- Rank: 1", rendered)
        self.assertIn("## Selected Order", rendered)
        self.assertIn("- title: Advance order", rendered)
        self.assertIn("### Selection Context", rendered)
        self.assertIn("- status: needs_review", rendered)
        self.assertIn("- recommended_owner_role: architect_local", rendered)
        self.assertIn("- delegation_reason: weak_selection_evidence", rendered)
        self.assertIn("- delegation_focus: deep_improvement_selection_quality", rendered)
        self.assertIn("  - weak_selection_evidence", rendered)
        self.assertIn("  - trace/factory_value: Factory-value evidence was too weak to continue.", rendered)
        self.assertIn("### Delegation", rendered)
        self.assertIn("- suggested_role: implementer_local", rendered)
        self.assertIn("- evidence_expectations:", rendered)
        self.assertIn("  - pytest output", rendered)
        self.assertIn("- suggested_validation:", rendered)
        self.assertIn("  - Run focused tests.", rendered)
        self.assertIn("- definition_of_done:", rendered)
        self.assertIn("  - Tests pass.", rendered)
        self.assertIn("## Receipt", rendered)
        self.assertIn("- event_type: proactive_action_plan_receipt", rendered)
        self.assertIn("- persisted: True", rendered)
        self.assertIn("## Persisted Receipt", rendered)
        self.assertIn("- recorded_at: 200.0", rendered)
        self.assertIn("## Receipt Counts", rendered)
        self.assertIn("- Total receipts: 1", rendered)
        self.assertIn("- acknowledged: 1", rendered)
        self.assertIn("## Receipt History", rendered)
        self.assertIn("| 1 | acknowledged | implementer_local | 200.0 | Delegated the bounded slice. | Track implementation receipt. |", rendered)

    def test_render_receipt_markdown_remains_compatible_when_context_fields_absent(self) -> None:
        receipt_report = self._base_receipt_report()
        receipt_report["order_identity"].pop("selection_quality", None)
        receipt_report["order_identity"]["handoff"] = {
            "suggested_role": "implementer_local",
            "suggested_endpoint": "/handoff/advance-order",
        }

        rendered = report_tool.render_receipt_markdown(receipt_report)

        self.assertNotIn("### Selection Context", rendered)
        self.assertIn("### Delegation", rendered)
        self.assertIn("- suggested_role: implementer_local", rendered)
        self.assertIn("- suggested_endpoint: /handoff/advance-order", rendered)

    def test_main_receipt_mode_renders_json_payload(self) -> None:
        receipt_report = self._base_receipt_report()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            db_path.write_text("", encoding="utf-8")

            with mock.patch.object(report_tool, "build_receipt", return_value=receipt_report):
                exit_code = report_tool.main(
                    [
                        "--db",
                        str(db_path),
                        "--receipt",
                        "--order-id",
                        "advance-order",
                        "--state",
                        "acknowledged",
                        "--format",
                        "json",
                    ],
                    stdout=stdout,
                    stderr=stderr,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        rendered = stdout.getvalue()
        self.assertIn('"event_type": "proactive_action_plan_receipt"', rendered)
        self.assertIn('"order_id": "advance-order"', rendered)
        self.assertIn('"persisted": true', rendered)


if __name__ == "__main__":
    unittest.main()
