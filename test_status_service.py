from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.status_service import StatusService, _structured_studio_decision_evidence_flags
from orchestrator.storage import SQLiteTaskStorage
from tools.backend_done_evidence_guard import _studio_decision_evidence_error


class TestStatusService(unittest.TestCase):
    def _valid_studio_decision_evidence(self) -> dict:
        return {
            "candidate_bets": [
                {
                    "id": "bet-a",
                    "summary": "Tighten proactive selection evidence before delegating implementation slices.",
                },
                {
                    "id": "bet-b",
                    "summary": "Add broad dashboard polish that does not change autonomy decisions.",
                },
                {
                    "id": "bet-c",
                    "summary": "Refactor unrelated queue plumbing without a measurable factory outcome.",
                },
            ],
            "killed_bets": [
                {
                    "id": "bet-b",
                    "reason": "Killed because dashboard polish would not reduce mistaken delegation or improve factory quality.",
                },
                {
                    "id": "bet-c",
                    "reason": "Killed because unrelated queue plumbing is wider than this bounded autonomy improvement slice.",
                },
            ],
            "selected_bet": {
                "id": "bet-a",
                "summary": "Tighten proactive selection evidence before delegating implementation slices.",
                "reason": "Selected because it directly prevents weak narrative evidence from causing more implementation churn.",
            },
            "critic_answers": [
                {
                    "answer": "The scope is bounded to the evidence gate and preserves existing commercial and factory-value signals.",
                },
                {
                    "answer": "The test plan covers both rejected narrative evidence and accepted structured decision evidence.",
                },
                {
                    "answer": "The change is reversible and does not alter downstream terminal completion guard behavior.",
                },
            ],
            "debate_summary": "Three candidate bets were compared, two were killed with reasons, and the structured evidence gate was selected.",
        }

    def _guard_compatible_studio_decision_evidence(self) -> dict:
        return {
            "candidate_bets": [
                {
                    "id": "bet-a",
                    "thesis": "Require structured Studio decision evidence before allowing proactive implementation to continue.",
                    "impact": "Reduces mistaken delegation by routing weak evidence back to selection review.",
                },
                {
                    "id": "bet-b",
                    "thesis": "Tune execution packet copy without changing the autonomy decision path.",
                    "impact": "Small presentation improvement with no measurable factory selection benefit.",
                },
                {
                    "id": "bet-c",
                    "thesis": "Broaden unrelated dashboard telemetry outside the bounded factory improvement slice.",
                    "impact": "Useful eventually, but it does not prove this proactive selection should continue.",
                },
            ],
            "killed_bets": [
                {
                    "id": "bet-b",
                    "explanation": "Killed because copy tuning does not address weak selection evidence or delegation quality.",
                },
                {
                    "id": "bet-c",
                    "explanation": "Killed because dashboard telemetry is too broad for this bounded autonomy improvement.",
                },
            ],
            "selected_bet": {
                "id": "bet-a",
                "summary": "Require structured Studio decision evidence before allowing proactive implementation to continue.",
                "explanation": "Selected because it directly aligns the early selection gate with final guard-compatible evidence.",
            },
            "critic_answers": {
                "scope": "The change is local to selection evidence and does not alter commercial or factory-value evidence.",
                "risk": "The regression test uses guard-compatible fields so valid structured evidence keeps advancing.",
                "validation": "The previous narrative-only and selected-is-killed reject paths continue to require review.",
            },
            "debate_summary": "The guard-compatible decision compared three bets, killed two alternatives, and selected the bounded evidence gate.",
        }

    def _title_only_studio_decision_evidence(self) -> dict:
        evidence = self._valid_studio_decision_evidence()
        for candidate in evidence["candidate_bets"]:
            candidate["title"] = candidate.pop("summary")
        return evidence

    def _nested_summary_studio_decision_evidence(self) -> dict:
        evidence = self._valid_studio_decision_evidence()
        evidence["candidate_bets"][0]["summary"] = {
            "text": "Nested candidate summary should not count because the final guard only accepts string values."
        }
        return evidence

    def _selected_bet_summary_studio_decision_evidence(self, summary: object = None) -> dict:
        evidence = self._valid_studio_decision_evidence()
        if summary is None:
            evidence["selected_bet"].pop("summary")
        else:
            evidence["selected_bet"]["summary"] = summary
        return evidence

    def _valid_factory_delta(self) -> dict:
        return {
            "capability_changed": "selection validation now gates factory delivery delegation",
            "measurable_delta": (
                "Before generic evidence could advance; after structured factory_delta and studio evidence are required."
            ),
            "evidence": "test_status_service.py validates missing structured fields and valid structured fields.",
        }

    def test_proactive_action_plan_adds_execution_packets_and_next_delegate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            orders = [
                {
                    "rank": 2,
                    "order_id": "release-order",
                    "order_id_short": "release",
                    "title": "Release order",
                    "priority": 1,
                    "phase": "review",
                    "current_stage": "deploy",
                    "readiness_state": "ready",
                    "readiness_verdict": "go",
                    "decision": "release",
                    "why": "Ready to release.",
                    "next_action": "Release the branch.",
                    "handoff": {
                        "suggested_role": "release_mgr",
                        "suggested_endpoint": "/handoff/release-order",
                        "inspect_path": "/inspect/release-order",
                        "checklist": ["Confirm release target."],
                        "definition_of_done": ["Release evidence recorded."],
                        "evidence_expectations": ["merge result"],
                        "suggested_validation": ["Check deploy status."],
                    },
                    "updated_at": 20.0,
                    "merge_ready": True,
                    "merged_to_main": False,
                },
                {
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
                    "why": "Needs delivery evidence.",
                    "next_action": "Implement the bounded slice.",
                    "handoff": {
                        "suggested_role": "implementer_local",
                        "suggested_endpoint": "/handoff/advance-order",
                        "inspect_path": "/inspect/advance-order",
                        "checklist": ["Open the order."],
                        "definition_of_done": ["Tests pass."],
                        "evidence_expectations": ["pytest output"],
                        "suggested_validation": ["Run focused tests."],
                        "assignment_prompt": "ROLE: implementer_local.\nAction: Implement the bounded slice.",
                    },
                    "updated_at": 10.0,
                    "merge_ready": False,
                    "merged_to_main": False,
                },
                {
                    "rank": 3,
                    "order_id": "unblock-order",
                    "order_id_short": "unblock",
                    "title": "Unblock order",
                    "priority": 2,
                    "phase": "review",
                    "current_stage": "skynet_review",
                    "readiness_state": "blocked",
                    "readiness_verdict": "no_go",
                    "decision": "unblock",
                    "why": "Reviewer blocker.",
                    "primary_blocker": {"stage": "skynet_review", "job": {"role": "reviewer_local"}},
                    "next_action": "Clear the review blocker.",
                    "handoff": {
                        "suggested_role": "reviewer_local",
                        "suggested_endpoint": "/handoff/unblock-order",
                        "inspect_path": "/inspect/unblock-order",
                        "checklist": ["Review blocker."],
                        "definition_of_done": ["Blocker cleared."],
                        "evidence_expectations": ["review note"],
                        "suggested_validation": ["Recheck readiness."],
                    },
                    "updated_at": 30.0,
                    "merge_ready": False,
                    "merged_to_main": False,
                },
            ]
            priorities = {
                "api_version": "v1",
                "schema_version": 1,
                "generated_at": 123.0,
                "chat_id": 7,
                "limit": 10,
                "summary": {"active_proactive_orders": 3},
                "orders": orders,
            }

            with mock.patch.object(svc, "proactive_priorities", return_value=priorities):
                plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["release"]["execution_packet"]["order_id"], "release-order")
            self.assertEqual(lanes["release"]["execution_packet"]["owner_role"], "release_mgr")
            self.assertEqual(lanes["unblock"]["execution_packet"]["owner_role"], "reviewer_local")
            self.assertIsNone(lanes["monitor"]["execution_packet"])

            top_packet = plan["top_execution_packet"]
            expected_outcome_contract = {
                "allowed_outcomes": [
                    "shipped_to_main",
                    "published_project",
                    "blocked_need_operator",
                    "rejected_low_value",
                    "failed_root_caused",
                ],
                "required_fields": ["outcome", "evidence", "residual_risk"],
                "definition": (
                    "Branch-only or done-only work is not completion without merge, publish, block, reject, "
                    "or root-cause evidence."
                ),
            }
            self.assertEqual(top_packet, lanes["advance"]["execution_packet"])
            self.assertEqual(top_packet["order_id"], "advance-order")
            self.assertEqual(top_packet["owner_role"], "implementer_local")
            self.assertEqual(top_packet["lane"], "advance")
            self.assertEqual(top_packet["action"], "Implement the bounded slice.")
            self.assertEqual(top_packet["inspect_endpoint"], "/inspect/advance-order")
            self.assertEqual(top_packet["handoff_endpoint"], "/handoff/advance-order")
            self.assertEqual(top_packet["acceptance_criteria"], ["Open the order."])
            self.assertEqual(
                top_packet["definition_of_done"],
                ["Tests pass.", "Final evidence records one allowed terminal outcome with evidence and residual_risk."],
            )
            self.assertEqual(
                top_packet["evidence_required"],
                ["pytest output", "terminal outcome evidence with outcome, evidence, and residual_risk"],
            )
            self.assertEqual(top_packet["suggested_validation"], ["Run focused tests."])
            self.assertEqual(top_packet["outcome_contract"], expected_outcome_contract)
            self.assertIn("ROLE: implementer_local.", top_packet["assignment_prompt"])
            self.assertIn("Outcome contract:", top_packet["assignment_prompt"])

            release_packet = lanes["release"]["execution_packet"]
            self.assertEqual(release_packet["outcome_contract"], expected_outcome_contract)
            self.assertIn("Final evidence records one allowed terminal outcome with evidence and residual_risk.", release_packet["definition_of_done"])
            self.assertIn("Release completion includes merge or release evidence, not branch-only or done-only status.", release_packet["definition_of_done"])
            self.assertIn("terminal outcome evidence with outcome, evidence, and residual_risk", release_packet["evidence_required"])
            self.assertIn("merge or release evidence for the terminal outcome", release_packet["evidence_required"])

            self.assertEqual(
                plan["summary"]["next_delegate"],
                {
                    "owner_role": "implementer_local",
                    "order_id": "advance-order",
                    "lane": "advance",
                    "action": "Implement the bounded slice.",
                    "inspect_endpoint": "/inspect/advance-order",
                    "handoff_endpoint": "/handoff/advance-order",
                    "acceptance_criteria": top_packet["acceptance_criteria"],
                    "evidence_required": top_packet["evidence_required"],
                    "suggested_validation": top_packet["suggested_validation"],
                    "definition_of_done": top_packet["definition_of_done"],
                    "assignment_prompt": top_packet["assignment_prompt"],
                    "outcome_contract": expected_outcome_contract,
                },
            )

    def test_proactive_action_plan_surfaces_deep_improvement_factory_delta_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            order_id = "deep-order"
            root = Task.new(
                job_id=order_id,
                source="test",
                role="skynet",
                input_text="[proactive: studio deep improvement]",
                request_type="task",
                priority=1,
                model="gpt-5.3-codex",
                effort="medium",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=0.0,
                chat_id=7,
                state="running",
                trace={
                    "proactive_lane": True,
                    "studio_cycle_id": "cycle-123",
                    "studio_selected_type": "DEEP_IMPROVEMENT",
                    "expected_measurable_delta": "Factory completion cannot pass without named capability delta.",
                },
            )
            q.submit_task(root)
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Studio deep improvement",
                body="[proactive: studio deep improvement]",
                status="active",
                priority=1,
            )
            board = {
                "orders": [
                    {
                        "order_id": order_id,
                        "order_id_short": "deep",
                        "title": "Studio deep improvement",
                        "priority": 1,
                        "phase": "executing",
                        "current_stage": "delivery",
                        "readiness_state": "not_ready",
                        "readiness_verdict": "wait",
                        "merge_ready": False,
                        "merged_to_main": False,
                        "children_total": 0,
                        "children_by_role": {},
                        "updated_at": 10.0,
                    }
                ]
            }

            with mock.patch.object(svc, "autonomy_board", return_value=board):
                plan = svc.proactive_action_plan(chat_id=7, limit=10)

            packet = plan["top_execution_packet"]
            contract = packet.get("factory_delta_contract")
            self.assertIsInstance(contract, dict)
            self.assertEqual(contract.get("studio_cycle_id"), "cycle-123")
            self.assertEqual(contract.get("required_fields"), ["capability_changed", "measurable_delta", "evidence"])
            self.assertIn("--require-factory-delta", " ".join(contract.get("suggested_validation") or []))
            self.assertIn("--require-factory-delta", " ".join(packet.get("suggested_validation") or []))
            self.assertIn("structured factory_delta.capability_changed", packet.get("evidence_required") or [])
            self.assertEqual(plan["summary"]["next_delegate"].get("factory_delta_contract"), contract)

            studio_contract = packet.get("studio_decision_evidence_contract")
            self.assertIsInstance(studio_contract, dict)
            self.assertEqual(studio_contract.get("studio_cycle_id"), "cycle-123")
            self.assertIn("summary", studio_contract.get("selected_bet") or "")
            self.assertEqual(
                studio_contract.get("required_fields"),
                [
                    "studio_decision_evidence.candidate_bets",
                    "studio_decision_evidence.killed_bets",
                    "studio_decision_evidence.selected_bet",
                    "studio_decision_evidence.selected_bet.summary",
                    "studio_decision_evidence.critic_answers",
                    "studio_decision_evidence.debate_summary",
                ],
            )
            self.assertIn("--require-studio-decision-evidence", " ".join(studio_contract.get("suggested_validation") or []))
            self.assertIn("--require-studio-decision-evidence", " ".join(packet.get("suggested_validation") or []))
            self.assertIn("structured studio_decision_evidence.candidate_bets", packet.get("evidence_required") or [])
            self.assertIn("structured studio_decision_evidence.killed_bets", packet.get("evidence_required") or [])
            self.assertIn("structured studio_decision_evidence.selected_bet with summary", packet.get("evidence_required") or [])
            self.assertIn("structured studio_decision_evidence.critic_answers", packet.get("evidence_required") or [])
            self.assertIn("structured studio_decision_evidence.debate_summary", packet.get("evidence_required") or [])
            self.assertIn("Studio decision evidence contract:", packet.get("assignment_prompt") or "")
            self.assertEqual(plan["summary"]["next_delegate"].get("studio_decision_evidence_contract"), studio_contract)

    def test_proactive_action_plan_omits_factory_delta_contract_for_non_deep_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            priorities = {
                "api_version": "v1",
                "schema_version": 1,
                "generated_at": 123.0,
                "chat_id": 7,
                "limit": 10,
                "summary": {"active_proactive_orders": 1},
                "orders": [
                    {
                        "rank": 1,
                        "order_id": "non-deep-order",
                        "order_id_short": "nondeep",
                        "title": "Non-deep order",
                        "priority": 1,
                        "phase": "executing",
                        "current_stage": "delivery",
                        "readiness_state": "not_ready",
                        "readiness_verdict": "wait",
                        "decision": "advance",
                        "why": "Needs delivery evidence.",
                        "next_action": "Implement the bounded slice.",
                        "handoff": {"suggested_role": "implementer_local"},
                        "updated_at": 10.0,
                    }
                ],
            }

            with mock.patch.object(svc, "proactive_priorities", return_value=priorities):
                plan = svc.proactive_action_plan(chat_id=7, limit=10)

            self.assertNotIn("factory_delta_contract", plan["top_execution_packet"])
            self.assertNotIn("factory_delta_contract", plan["summary"]["next_delegate"])
            self.assertNotIn("studio_decision_evidence_contract", plan["top_execution_packet"])
            self.assertNotIn("studio_decision_evidence_contract", plan["summary"]["next_delegate"])

    def test_proactive_action_plan_empty_packets_are_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            priorities = {
                "api_version": "v1",
                "schema_version": 1,
                "generated_at": 123.0,
                "chat_id": 7,
                "limit": 10,
                "summary": {"active_proactive_orders": 0},
                "orders": [],
            }

            with mock.patch.object(svc, "proactive_priorities", return_value=priorities):
                plan = svc.proactive_action_plan(chat_id=7, limit=10)

            self.assertIsNone(plan["top_execution_packet"])
            self.assertIsNone(plan["summary"]["next_delegate"])
            self.assertTrue(all(lane["execution_packet"] is None for lane in plan["lanes"]))

    def test_proactive_action_plan_reroutes_weak_advance_order_to_selection_review(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "weak-advance-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Weak advance order",
                body="Implement another bounded slice.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Weak advance order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={"proactive_lane": True, "result_summary": "Implementation requested."},
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertLess(
                [lane["lane"] for lane in plan["lanes"]].index("selection_review"),
                [lane["lane"] for lane in plan["lanes"]].index("advance"),
            )
            order = lanes["selection_review"]["orders"][0]
            self.assertEqual(order["decision"], "selection_review")
            self.assertEqual(order["selection_quality"]["status"], "needs_review")
            self.assertIn("weak_selection_evidence", order["selection_quality"]["flags"])
            self.assertEqual(plan["summary"]["lanes"]["selection_review"], 1)
            self.assertEqual(plan["summary"]["top_lane"], "selection_review")
            self.assertEqual(plan["summary"]["selection_quality"]["needs_review"], 1)

            top_packet = plan["top_execution_packet"]
            self.assertEqual(top_packet["order_id"], order_id)
            self.assertEqual(top_packet["owner_role"], "architect_local")
            self.assertEqual(top_packet["lane"], "selection_review")
            self.assertIn("Selection review", top_packet["action"])
            self.assertIn("kill/continue", top_packet["action"])
            self.assertIn("kill/continue/replan decision", top_packet["evidence_required"])
            self.assertIn("factory-value", " ".join(top_packet["evidence_required"]))
            self.assertNotIn(
                "Final evidence records one allowed terminal outcome with evidence and residual_risk.",
                top_packet["definition_of_done"],
            )
            self.assertNotIn(
                "terminal outcome evidence with outcome, evidence, and residual_risk",
                top_packet["evidence_required"],
            )
            self.assertIn(
                "If ending the order, record one allowed terminal outcome with evidence and residual_risk.",
                top_packet["definition_of_done"],
            )
            self.assertIn(
                "If ending the order, include terminal outcome evidence with outcome, evidence, and residual_risk.",
                top_packet["evidence_required"],
            )
            self.assertEqual(top_packet["outcome_contract"], plan["summary"]["next_delegate"]["outcome_contract"])

    def test_proactive_action_plan_ignores_narrative_studio_decision_evidence_for_selection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "narrative-studio-evidence-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Narrative studio evidence order",
                body="Implement another bounded slice.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Narrative studio evidence order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "studio_decision_evidence": {
                            "summary": "Studio considered this important and decided to continue with implementation.",
                            "notes": "Narrative-only evidence lacks killed bets, selected bet rationale, critic answers, and debate summary.",
                        },
                        "result_summary": "Implementation requested.",
                    },
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 1)
            self.assertEqual(lanes["advance"]["count"], 0)
            order = lanes["selection_review"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "selection_review")
            self.assertEqual(selection_quality["status"], "needs_review")
            self.assertIn("weak_selection_evidence", selection_quality["flags"])
            self.assertNotIn("selection_evidence_present", selection_quality["flags"])
            self.assertEqual(selection_quality["evidence_sources"], [])

            top_packet = plan["top_execution_packet"]
            self.assertEqual(top_packet["order_id"], order_id)
            self.assertEqual(top_packet["owner_role"], "architect_local")
            self.assertEqual(top_packet["lane"], "selection_review")

    def test_proactive_action_plan_accepts_structured_studio_decision_evidence_for_selection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "structured-studio-evidence-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Structured studio evidence order",
                body="Implement another bounded slice.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Structured studio evidence order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "studio_decision_evidence": self._valid_studio_decision_evidence(),
                    },
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 0)
            self.assertEqual(lanes["advance"]["count"], 1)
            order = lanes["advance"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "advance")
            self.assertEqual(selection_quality["status"], "ok")
            self.assertEqual(selection_quality["flags"], ["selection_evidence_present"])
            self.assertEqual(selection_quality["evidence_sources"][0]["key"], "studio_decision_evidence")
            self.assertEqual(plan["top_execution_packet"]["lane"], "advance")

    def test_deep_improvement_generic_evidence_requires_structured_selection_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "deep-generic-evidence-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Deep generic evidence order",
                body="Factory-value evidence says this selected revenue workflow should continue.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Deep generic evidence order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "studio_selected_type": "DEEP_IMPROVEMENT",
                        "commercial_evidence_target": "Revenue and factory selection evidence are described generically.",
                        "factory_value": {
                            "score": 88,
                            "dimensions": {"factory": {"ok": True}},
                            "summary": "Factory-value evidence says this selected workflow should advance.",
                        },
                    },
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 1)
            self.assertEqual(lanes["advance"]["count"], 0)
            order = lanes["selection_review"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "selection_review")
            self.assertEqual(selection_quality["status"], "needs_review")
            self.assertIn("factory_delta_missing", selection_quality["flags"])
            self.assertIn("studio_decision_evidence_missing", selection_quality["flags"])
            self.assertEqual(selection_quality["recommended_owner_role"], "architect_local")
            self.assertEqual(selection_quality["delegation_reason"], "deep_improvement_evidence_required")
            self.assertEqual(selection_quality["delegation_focus"], "deep_improvement_selection_quality")
            self.assertTrue(selection_quality["evidence_sources"])
            self.assertIn("DEEP_IMPROVEMENT needs structured factory_delta", selection_quality["summary"])
            self.assertEqual(plan["top_execution_packet"]["lane"], "selection_review")

    def test_deep_improvement_structured_factory_and_studio_evidence_allows_selection_ok(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "deep-structured-evidence-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Deep structured evidence order",
                body="Implement the selected factory selection evidence gate.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Deep structured evidence order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "studio_selected_type": "DEEP_IMPROVEMENT",
                        "factory_delta": self._valid_factory_delta(),
                        "studio_decision_evidence": self._valid_studio_decision_evidence(),
                    },
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 0)
            self.assertEqual(lanes["advance"]["count"], 1)
            order = lanes["advance"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "advance")
            self.assertEqual(selection_quality["status"], "ok")
            self.assertEqual(selection_quality["flags"], ["selection_evidence_present"])
            self.assertEqual(selection_quality["evidence_sources"][0]["key"], "studio_decision_evidence")
            self.assertEqual(plan["top_execution_packet"]["lane"], "advance")

    def test_studio_decision_flags_reject_selected_bet_without_substantive_summary(self) -> None:
        cases = [
            ("missing", self._selected_bet_summary_studio_decision_evidence()),
            ("generic", self._selected_bet_summary_studio_decision_evidence("unknown")),
        ]
        for name, evidence in cases:
            with self.subTest(name=name):
                guard_error = _studio_decision_evidence_error(
                    payload={"studio_decision_evidence": evidence},
                    summary_path=Path("summary.json"),
                )
                self.assertIsNotNone(guard_error)
                self.assertEqual(guard_error.get("reason"), "studio_decision_selected_bet_summary_missing")

                flags = _structured_studio_decision_evidence_flags(evidence)

                self.assertIn("studio_decision_evidence_selected_bet_missing_or_weak", flags)

    def test_proactive_action_plan_accepts_guard_compatible_dict_critic_studio_decision_evidence(self) -> None:
        guard_error = _studio_decision_evidence_error(
            payload={"studio_decision_evidence": self._guard_compatible_studio_decision_evidence()},
            summary_path=Path("summary.json"),
        )
        self.assertIsNone(guard_error)

        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "guard-compatible-studio-evidence-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Guard compatible studio evidence order",
                body="Implement another bounded slice.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Guard compatible studio evidence order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "studio_decision_evidence": self._guard_compatible_studio_decision_evidence(),
                    },
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 0)
            self.assertEqual(lanes["advance"]["count"], 1)
            order = lanes["advance"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "advance")
            self.assertEqual(selection_quality["status"], "ok")
            self.assertEqual(selection_quality["flags"], ["selection_evidence_present"])
            self.assertEqual(selection_quality["evidence_sources"][0]["key"], "studio_decision_evidence")

    def test_proactive_action_plan_rejects_title_only_studio_decision_candidate_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "title-only-studio-evidence-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Title only studio evidence order",
                body="Implement another bounded slice.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Title only studio evidence order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "studio_decision_evidence": self._title_only_studio_decision_evidence(),
                    },
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 1)
            self.assertEqual(lanes["advance"]["count"], 0)
            order = lanes["selection_review"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "selection_review")
            self.assertEqual(selection_quality["status"], "needs_review")
            self.assertIn("weak_selection_evidence", selection_quality["flags"])
            self.assertNotIn("selection_evidence_present", selection_quality["flags"])
            self.assertEqual(selection_quality["evidence_sources"], [])

    def test_proactive_action_plan_rejects_nested_studio_decision_candidate_text(self) -> None:
        evidence = self._nested_summary_studio_decision_evidence()
        guard_error = _studio_decision_evidence_error(
            payload={"studio_decision_evidence": evidence},
            summary_path=Path("summary.json"),
        )
        self.assertIsNotNone(guard_error)
        self.assertEqual(guard_error.get("reason"), "studio_decision_candidate_bet_text_missing")

        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "nested-studio-evidence-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Nested studio evidence order",
                body="Implement another bounded slice.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Nested studio evidence order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "studio_decision_evidence": evidence,
                    },
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 1)
            self.assertEqual(lanes["advance"]["count"], 0)
            order = lanes["selection_review"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "selection_review")
            self.assertEqual(selection_quality["status"], "needs_review")
            self.assertIn("weak_selection_evidence", selection_quality["flags"])
            self.assertNotIn("selection_evidence_present", selection_quality["flags"])
            self.assertEqual(selection_quality["evidence_sources"], [])

    def test_proactive_action_plan_rejects_studio_decision_evidence_when_selected_bet_is_killed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "killed-selected-studio-order"
            evidence = self._valid_studio_decision_evidence()
            evidence["killed_bets"].append(
                {
                    "id": "bet-a",
                    "reason": "Killed because this edge case proves selected bets cannot also be rejected.",
                }
            )
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Killed selected studio evidence order",
                body="Implement another bounded slice.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Killed selected studio evidence order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "studio_decision_evidence": evidence,
                    },
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 1)
            order = lanes["selection_review"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "selection_review")
            self.assertEqual(selection_quality["status"], "needs_review")
            self.assertIn("weak_selection_evidence", selection_quality["flags"])
            self.assertNotIn("selection_evidence_present", selection_quality["flags"])

    def test_proactive_action_plan_reroutes_evidence_backed_churn_without_validation_to_selection_review(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "churn-heavy-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Churn-heavy proactive order",
                body="Factory-value evidence says this selected workflow should continue.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Churn-heavy proactive order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "proactive_slices_started": 4,
                        "proactive_slices_applied": 3,
                        "proactive_slices_validated": 0,
                        "proactive_slices_closed": 0,
                        "factory_value": {
                            "score": 82,
                            "dimensions": {"factory": {"ok": True}},
                            "summary": "Factory-value evidence says this selected workflow should advance.",
                        },
                        "result_summary": "More implementation slices were started.",
                    },
                    job_id=order_id,
                )
            )
            for index in range(3):
                q.submit_task(
                    Task.new(
                        source="telegram",
                        role="implementer_local",
                        input_text=f"Implement slice {index + 1}",
                        request_type="task",
                        priority=1,
                        model="gpt-5.2",
                        effort="medium",
                        mode_hint="rw",
                        requires_approval=False,
                        max_cost_window_usd=1.0,
                        chat_id=7,
                        state="done",
                        parent_job_id=order_id,
                        trace={"result_summary": f"Implemented slice {index + 1}."},
                        job_id=f"churn-child-{index + 1}",
                    )
                )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 1)
            order = lanes["selection_review"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "selection_review")
            self.assertEqual(selection_quality["status"], "needs_review")
            self.assertIn("implementation_churn_without_validation", selection_quality["flags"])
            self.assertIn("selection_evidence_present", selection_quality["flags"])
            self.assertNotIn("weak_selection_evidence", selection_quality["flags"])
            self.assertNotIn("advance_without_commercial_factory_or_studio_evidence", selection_quality["flags"])
            self.assertIn("started_slices_exceed_validated_or_closed", selection_quality["flags"])
            self.assertIn("repeated_delivery_children_without_validation", selection_quality["flags"])
            self.assertTrue(selection_quality["evidence_sources"])
            self.assertEqual(selection_quality["evidence_sources"][0]["key"], "factory_value")
            self.assertEqual(selection_quality["churn_risk"]["status"], "needs_review")
            self.assertEqual(selection_quality["churn_risk"]["counters"]["proactive_slices_started"], 4)
            self.assertEqual(selection_quality["churn_risk"]["counters"]["delivery_children"], 3)

            top_packet = plan["top_execution_packet"]
            self.assertEqual(plan["summary"]["top_lane"], "selection_review")
            self.assertEqual(top_packet["order_id"], order_id)
            self.assertEqual(top_packet["owner_role"], "architect_local")
            self.assertEqual(plan["summary"]["next_delegate"]["owner_role"], "architect_local")
            self.assertEqual(plan["summary"]["next_delegate"]["lane"], "selection_review")
            self.assertIn("kill/continue/replan", top_packet["action"])

    def test_proactive_action_plan_reroutes_validated_closure_debt_to_selection_review(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "closure-debt-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Closure-debt proactive order",
                body="Validated slices exist, but no terminal closure was recorded.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Closure-debt proactive order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "proactive_slices_started": 2,
                        "proactive_slices_applied": 2,
                        "proactive_slices_validated": 2,
                        "proactive_slices_closed": 0,
                        "factory_value": {
                            "score": 80,
                            "dimensions": {"factory": {"ok": True}},
                            "summary": "Factory-value evidence supports the selected workflow.",
                        },
                        "result_summary": "Two slices validated without closure.",
                    },
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 1)
            self.assertEqual(lanes["advance"]["count"], 0)
            order = lanes["selection_review"]["orders"][0]
            selection_quality = order["selection_quality"]
            churn_risk = selection_quality["churn_risk"]
            self.assertEqual(order["decision"], "selection_review")
            self.assertEqual(selection_quality["status"], "needs_review")
            self.assertIn("implementation_churn_without_validation", selection_quality["flags"])
            self.assertIn("selection_evidence_present", selection_quality["flags"])
            self.assertIn("validated_slices_without_terminal_closure", selection_quality["flags"])
            self.assertEqual(churn_risk["status"], "needs_review")
            self.assertIn("validated_slices_without_terminal_closure", churn_risk["flags"])
            self.assertEqual(churn_risk["counters"]["proactive_slices_applied"], 2)
            self.assertEqual(churn_risk["counters"]["proactive_slices_validated"], 2)
            self.assertEqual(churn_risk["counters"]["proactive_slices_closed"], 0)
            self.assertEqual(selection_quality["recommended_owner_role"], "release_mgr")
            self.assertEqual(selection_quality["delegation_reason"], "validated_slices_without_terminal_closure")
            self.assertEqual(selection_quality["delegation_focus"], "terminal_outcome_closure")

            top_packet = plan["top_execution_packet"]
            self.assertEqual(top_packet["order_id"], order_id)
            self.assertEqual(top_packet["lane"], "selection_review")
            self.assertEqual(top_packet["owner_role"], "release_mgr")
            self.assertEqual(top_packet["delegation_reason"], "validated_slices_without_terminal_closure")
            self.assertEqual(top_packet["delegation_focus"], "terminal_outcome_closure")
            self.assertEqual(plan["summary"]["next_delegate"]["owner_role"], "release_mgr")
            self.assertEqual(
                plan["summary"]["next_delegate"]["delegation_reason"],
                "validated_slices_without_terminal_closure",
            )
            self.assertIn("close, release, or replan", top_packet["action"])
            self.assertIn("terminal outcome evidence", top_packet["action"])
            self.assertNotIn("kill/continue", top_packet["action"])
            self.assertIn("close/release/replan decision", top_packet["evidence_required"])
            self.assertIn("terminal outcome evidence", top_packet["evidence_required"])
            self.assertIn(
                "No new implementation slice is delegated from this packet.",
                top_packet["definition_of_done"],
            )

    def test_proactive_action_plan_reroutes_validation_child_closure_debt_to_release_mgr(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "qa-reviewed-closure-debt-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="QA-reviewed proactive order",
                body="QA reviewed the slice, but no terminal closure was recorded.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] QA-reviewed proactive order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "proactive_slices_started": 1,
                        "proactive_slices_applied": 1,
                        "proactive_slices_validated": 0,
                        "proactive_slices_closed": 0,
                        "factory_value": {
                            "score": 76,
                            "dimensions": {"factory": {"ok": True}},
                            "summary": "Factory-value evidence supports the reviewed improvement.",
                        },
                    },
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="qa",
                    input_text="Validate the proactive slice.",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    parent_job_id=order_id,
                    trace={
                        "quality_gate_status": "reviewed_ready",
                        "slice_validation_ok": True,
                        "result_summary": "QA reviewed the implemented slice and marked it ready.",
                    },
                    job_id="qa-reviewed-child",
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 1)
            self.assertEqual(lanes["advance"]["count"], 0)
            order = lanes["selection_review"]["orders"][0]
            selection_quality = order["selection_quality"]
            churn_risk = selection_quality["churn_risk"]
            self.assertEqual(order["decision"], "selection_review")
            self.assertEqual(selection_quality["status"], "needs_review")
            self.assertIn("validated_slices_without_terminal_closure", selection_quality["flags"])
            self.assertIn("validated_slices_without_terminal_closure", churn_risk["flags"])
            self.assertEqual(churn_risk["counters"]["proactive_slices_validated"], 0)
            self.assertEqual(churn_risk["counters"]["proactive_slices_closed"], 0)
            self.assertEqual(churn_risk["counters"]["validation_children_done"], 1)
            self.assertEqual(churn_risk["counters"]["qa_children_done"], 1)
            self.assertEqual(churn_risk["counters"]["reviewer_children_done"], 0)
            self.assertEqual(churn_risk["counters"]["accepted_validation_children"], 1)
            self.assertFalse(churn_risk["counters"]["terminal_closure_evidence"])
            self.assertEqual(selection_quality["recommended_owner_role"], "release_mgr")
            self.assertEqual(selection_quality["delegation_reason"], "validated_slices_without_terminal_closure")
            self.assertEqual(selection_quality["delegation_focus"], "terminal_outcome_closure")

            top_packet = plan["top_execution_packet"]
            self.assertEqual(plan["summary"]["top_lane"], "selection_review")
            self.assertEqual(top_packet["order_id"], order_id)
            self.assertEqual(top_packet["owner_role"], "release_mgr")
            self.assertEqual(top_packet["delegation_reason"], "validated_slices_without_terminal_closure")
            self.assertIn("Closure debt review", top_packet["action"])
            self.assertIn("terminal outcome evidence", top_packet["action"])
            self.assertNotIn("kill/continue", top_packet["action"])
            self.assertEqual(plan["summary"]["next_delegate"]["owner_role"], "release_mgr")

    def test_proactive_action_plan_no_go_validation_child_does_not_count_as_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "qa-no-go-validation-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="QA no-go proactive order",
                body="QA finished with no-go validation.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] QA no-go proactive order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "proactive_slices_started": 1,
                        "proactive_slices_applied": 1,
                        "proactive_slices_validated": 0,
                        "proactive_slices_closed": 0,
                        "factory_value": {
                            "score": 74,
                            "dimensions": {"factory": {"ok": True}},
                            "summary": "Factory-value evidence exists, but validation did not accept the slice.",
                        },
                    },
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="qa",
                    input_text="Validate the proactive slice.",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    parent_job_id=order_id,
                    trace={
                        "quality_gate_status": "no_go",
                        "slice_validation_ok": False,
                        "result_summary": "QA completed validation with a no-go because checks failed.",
                    },
                    job_id="qa-no-go-child",
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 0)
            self.assertEqual(lanes["advance"]["count"], 1)
            order = lanes["advance"]["orders"][0]
            churn_risk = order["selection_quality"]["churn_risk"]
            self.assertEqual(churn_risk["counters"]["validation_children_done"], 1)
            self.assertEqual(churn_risk["counters"]["qa_children_done"], 1)
            self.assertEqual(churn_risk["counters"]["accepted_validation_children"], 0)
            self.assertNotIn("validated_slices_without_terminal_closure", churn_risk["flags"])
            self.assertNotEqual(order["selection_quality"].get("recommended_owner_role"), "release_mgr")

    def test_proactive_action_plan_merge_ready_with_validation_child_still_routes_closure_debt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "merge-ready-reviewed-closure-debt-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Merge-ready reviewed proactive order",
                body="QA accepted the slice and it is merge-ready, but nothing terminal happened.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Merge-ready reviewed proactive order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "proactive_slices_started": 1,
                        "proactive_slices_applied": 1,
                        "proactive_slices_validated": 0,
                        "proactive_slices_closed": 0,
                        "merge_ready": True,
                        "factory_value": {
                            "score": 78,
                            "dimensions": {"factory": {"ok": True}},
                            "summary": "Factory-value evidence supports the accepted improvement.",
                        },
                    },
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="reviewer_local",
                    input_text="Review the proactive slice.",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    parent_job_id=order_id,
                    trace={
                        "review_ready": True,
                        "result_summary": "Reviewer accepted the slice as ready.",
                    },
                    job_id="reviewer-merge-ready-child",
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 1)
            self.assertEqual(lanes["advance"]["count"], 0)
            order = lanes["selection_review"]["orders"][0]
            selection_quality = order["selection_quality"]
            churn_risk = selection_quality["churn_risk"]
            self.assertTrue(order["merge_ready"])
            self.assertFalse(order["merged_to_main"])
            self.assertEqual(order["readiness_state"], "not_ready")
            self.assertEqual(order["readiness_verdict"], "wait")
            self.assertEqual(churn_risk["counters"]["accepted_validation_children"], 1)
            self.assertFalse(churn_risk["counters"]["terminal_closure_evidence"])
            self.assertIn("validated_slices_without_terminal_closure", churn_risk["flags"])
            self.assertEqual(order["decision"], "selection_review")
            self.assertEqual(selection_quality["recommended_owner_role"], "release_mgr")
            self.assertEqual(selection_quality["delegation_focus"], "terminal_outcome_closure")
            self.assertEqual(plan["top_execution_packet"]["owner_role"], "release_mgr")
            self.assertEqual(plan["top_execution_packet"]["delegation_focus"], "terminal_outcome_closure")

    def test_proactive_action_plan_validation_child_closure_debt_exempts_merged_orders(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "merged-reviewed-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Merged reviewed proactive order",
                body="Reviewer evidence exists and the order already merged.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Merged reviewed proactive order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "proactive_slices_started": 1,
                        "proactive_slices_applied": 1,
                        "proactive_slices_closed": 0,
                        "merged_to_main": True,
                        "deploy_status": "ok",
                        "result_summary": "The reviewed slice merged to main.",
                    },
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="reviewer_local",
                    input_text="Review the proactive slice.",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    parent_job_id=order_id,
                    trace={
                        "review_ready": True,
                        "result_summary": "Reviewer accepted the slice before merge.",
                    },
                    job_id="reviewer-merged-child",
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 0)
            self.assertEqual(lanes["monitor"]["count"], 1)
            order = lanes["monitor"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "monitor")
            self.assertEqual(selection_quality["status"], "ok")
            self.assertNotIn("validated_slices_without_terminal_closure", selection_quality["flags"])
            self.assertEqual(lanes["monitor"]["execution_packet"]["owner_role"], "release_mgr")

    def test_proactive_action_plan_allows_single_validated_slice_without_closure_to_advance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            order_id = "single-validated-slice-order"
            q.upsert_order(
                order_id=order_id,
                chat_id=7,
                title="Single validated slice order",
                body="One validated slice should continue through normal advance routing.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Single validated slice order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "proactive_slices_started": 1,
                        "proactive_slices_applied": 1,
                        "proactive_slices_validated": 1,
                        "proactive_slices_closed": 0,
                        "factory_value": {
                            "score": 72,
                            "dimensions": {"factory": {"ok": True}},
                            "summary": "Factory-value evidence supports one more normal advance step.",
                        },
                    },
                    job_id=order_id,
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["selection_review"]["count"], 0)
            self.assertEqual(lanes["advance"]["count"], 1)
            order = lanes["advance"]["orders"][0]
            selection_quality = order["selection_quality"]
            self.assertEqual(order["decision"], "advance")
            self.assertEqual(selection_quality["status"], "ok")
            self.assertIn("selection_evidence_present", selection_quality["flags"])
            self.assertEqual(selection_quality["churn_risk"]["status"], "ok")
            self.assertNotIn(
                "validated_slices_without_terminal_closure",
                selection_quality["churn_risk"]["flags"],
            )

    def test_proactive_priorities_promotes_needs_review_advance_to_selection_review(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})

            weak_id = "weak-selection-order"
            q.upsert_order(
                order_id=weak_id,
                chat_id=7,
                title="Weak selection order",
                body="Implement another bounded slice.",
                status="active",
                priority=2,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Weak selection order",
                    request_type="task",
                    priority=2,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={"proactive_lane": True, "result_summary": "Implementation requested."},
                    job_id=weak_id,
                )
            )

            supported_id = "supported-advance-order"
            q.upsert_order(
                order_id=supported_id,
                chat_id=7,
                title="Supported advance order",
                body="Advance the selected value slice.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Supported advance order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "factory_value": {
                            "score": 75,
                            "dimensions": {"factory": {"ok": True}},
                            "summary": "Factory-value evidence shows this user-facing workflow should advance.",
                        },
                    },
                    job_id=supported_id,
                )
            )

            blocked_id = "blocked-order"
            q.upsert_order(
                order_id=blocked_id,
                chat_id=7,
                title="Blocked order",
                body="Blocked delivery.",
                status="active",
                priority=1,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Blocked order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={"proactive_lane": True},
                    job_id=blocked_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="implementer_local",
                    input_text="Blocked task",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="blocked",
                    parent_job_id=blocked_id,
                    blocked_reason="Missing dependency.",
                    job_id="blocked-child-priority",
                )
            )

            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            priorities = svc.proactive_priorities(chat_id=7, limit=10)

            orders_by_id = {order["order_id"]: order for order in priorities["orders"]}
            weak_order = orders_by_id[weak_id]
            supported_order = orders_by_id[supported_id]
            blocked_order = orders_by_id[blocked_id]

            self.assertEqual(blocked_order["decision"], "unblock")
            self.assertEqual(weak_order["decision"], "selection_review")
            self.assertEqual(supported_order["decision"], "advance")
            self.assertGreater(weak_order["score_breakdown"]["decision_rank"], blocked_order["score_breakdown"]["decision_rank"])
            self.assertLess(weak_order["score_breakdown"]["decision_rank"], supported_order["score_breakdown"]["decision_rank"])
            self.assertEqual(weak_order["selection_quality"]["status"], "needs_review")
            self.assertIn("Selection review", weak_order["next_action"])
            self.assertEqual(weak_order["handoff"]["suggested_role"], "architect_local")
            self.assertIn("kill/continue/replan decision", weak_order["handoff"]["evidence_expectations"])
            self.assertEqual(priorities["summary"]["by_decision"]["selection_review"], 1)

    def test_proactive_action_plan_selection_review_does_not_reroute_release_or_blocked_orders(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})

            release_id = "release-ready-order"
            q.upsert_order(
                order_id=release_id,
                chat_id=7,
                title="Release ready order",
                body="Release it.",
                status="active",
                priority=1,
                phase="review",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Release ready order",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={
                        "proactive_lane": True,
                        "proactive_slices_applied": 1,
                        "proactive_slices_validated": 1,
                        "proactive_slices_closed": 1,
                        "proactive_improvement_verified": True,
                        "proactive_improvement_closed": True,
                        "merge_ready": True,
                        "order_branch": "orders/release-ready-order",
                        "result_summary": "Validated release evidence for a shippable value path.",
                    },
                    job_id=release_id,
                )
            )

            blocked_id = "blocked-advance-order"
            q.upsert_order(
                order_id=blocked_id,
                chat_id=7,
                title="Blocked advance order",
                body="Blocked delivery.",
                status="active",
                priority=2,
                phase="executing",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="[proactive: test] Blocked advance order",
                    request_type="task",
                    priority=2,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="done",
                    trace={"proactive_lane": True},
                    job_id=blocked_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="implementer_local",
                    input_text="Implement blocked slice",
                    request_type="task",
                    priority=2,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=7,
                    state="blocked",
                    parent_job_id=blocked_id,
                    blocked_reason="Delivery dependency is missing.",
                    trace={"result_summary": "Delivery dependency is missing."},
                    job_id="blocked-child",
                )
            )
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            plan = svc.proactive_action_plan(chat_id=7, limit=10)

            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            release_packet = lanes["release"]["execution_packet"]
            unblock_packet = lanes["unblock"]["execution_packet"]
            self.assertEqual(release_packet["order_id"], release_id)
            self.assertEqual(release_packet["owner_role"], "release_mgr")
            self.assertEqual(lanes["release"]["orders"][0]["selection_quality"]["status"], "ok")
            self.assertEqual(unblock_packet["order_id"], blocked_id)
            self.assertEqual(unblock_packet["owner_role"], "implementer_local")
            self.assertEqual(lanes["unblock"]["orders"][0]["selection_quality"]["status"], "ok")
            self.assertNotIn("Selection review", unblock_packet["action"])

    def test_proactive_action_plan_receipt_persists_selected_order_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            report = {
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
                                "title": "Advance order",
                                "priority": 1,
                                "phase": "executing",
                                "current_stage": "delivery",
                                "readiness_state": "not_ready",
                                "readiness_verdict": "wait",
                                "decision": "advance",
                                "next_action": "Implement the bounded slice.",
                                "handoff": {
                                    "suggested_role": "implementer_local",
                                    "suggested_endpoint": "/handoff/advance-order",
                                    "inspect_path": "/inspect/advance-order",
                                    "assignment_prompt": "ROLE: implementer_local.\nAction: Implement the bounded slice.",
                                },
                            }
                        ],
                    }
                ],
            }
            receipt_rows = [
                {
                    "kind": "proactive_action_plan_receipt",
                    "state": "acknowledged",
                    "summary": "Delegated the bounded slice.",
                    "next_action": "Track implementation receipt.",
                    "order_id": "advance-order",
                    "job_id": "advance-order",
                    "ts": 200.0,
                    "details": {
                        "actor": "implementer_local",
                        "selection": {"rank": 1, "matched_by": "rank"},
                        "order_identity": {"order_id": "advance-order", "title": "Advance order"},
                        "proactive_action_plan_details": {"note": "delegate now"},
                    },
                }
            ]

            with mock.patch.object(svc, "proactive_action_plan", return_value=report), \
                 mock.patch.object(q, "append_decision_log") as append_decision_log, \
                 mock.patch.object(q, "list_decision_log", return_value=receipt_rows):
                payload = svc.proactive_action_plan_receipt(
                    chat_id=7,
                    rank=1,
                    state="acknowledged",
                    summary="Delegated the bounded slice.",
                    next_action="Track implementation receipt.",
                    actor="implementer_local",
                    details={"note": "delegate now"},
                )

            append_decision_log.assert_called_once()
            self.assertEqual(append_decision_log.call_args.kwargs["order_id"], "advance-order")
            self.assertEqual(append_decision_log.call_args.kwargs["job_id"], "advance-order")
            self.assertEqual(append_decision_log.call_args.kwargs["kind"], "proactive_action_plan_receipt")
            self.assertEqual(append_decision_log.call_args.kwargs["state"], "acknowledged")
            persisted_details = append_decision_log.call_args.kwargs["details"]
            self.assertEqual(persisted_details["selection"]["rank"], 1)
            self.assertEqual(persisted_details["order_identity"]["order_id"], "advance-order")
            self.assertEqual(persisted_details["proactive_action_plan_details"], {"note": "delegate now"})

            self.assertEqual(payload.get("selection"), {"order_id": None, "rank": 1, "matched_by": "rank"})
            order_identity = payload.get("order_identity") or {}
            self.assertEqual(order_identity.get("order_id"), "advance-order")
            self.assertEqual(order_identity.get("title"), "Advance order")
            self.assertEqual((order_identity.get("handoff") or {}).get("suggested_role"), "implementer_local")
            receipt = payload.get("receipt") or {}
            self.assertTrue(receipt.get("persisted"))
            self.assertEqual(receipt.get("persistence_reason"), "decision_log_appended")
            self.assertEqual(receipt.get("order_id"), "advance-order")
            persisted_receipt = payload.get("persisted_receipt") or {}
            self.assertEqual(persisted_receipt.get("actor"), "implementer_local")
            self.assertEqual(persisted_receipt.get("recorded_at"), 200.0)
            self.assertEqual((persisted_receipt.get("details") or {}).get("note"), "delegate now")
            self.assertEqual(payload.get("receipt_count"), 1)
            self.assertEqual(payload.get("receipt_counts_by_state"), {"acknowledged": 1})
            self.assertEqual(len(payload.get("receipt_history") or []), 1)

    def test_proactive_action_plan_receipt_returns_missing_selection_without_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            report = {
                "generated_at": 123.0,
                "chat_id": 7,
                "limit": 10,
                "summary": {
                    "active_proactive_orders": 1,
                    "returned": 1,
                    "top_lane": "advance",
                    "top_action": "Implement the bounded slice.",
                },
                "lanes": [
                    {
                        "lane": "advance",
                        "label": "Advance",
                        "count": 1,
                        "recommended_next_action": "Implement the bounded slice.",
                        "orders": [{"rank": 1, "order_id": "advance-order", "title": "Advance order", "decision": "advance"}],
                    }
                ],
            }

            with mock.patch.object(svc, "proactive_action_plan", return_value=report), \
                 mock.patch.object(q, "append_decision_log") as append_decision_log, \
                 mock.patch.object(q, "list_decision_log") as list_decision_log:
                payload = svc.proactive_action_plan_receipt(
                    chat_id=7,
                    order_id="missing-order",
                    state="completed",
                    summary="Nothing matched.",
                )

            append_decision_log.assert_not_called()
            list_decision_log.assert_not_called()
            self.assertEqual(payload.get("selection"), {"order_id": "missing-order", "rank": None, "matched_by": None})
            self.assertIsNone(payload.get("order_identity"))
            receipt = payload.get("receipt") or {}
            self.assertFalse(receipt.get("persisted"))
            self.assertEqual(receipt.get("persistence_reason"), "order_not_selected")
            self.assertIsNone(payload.get("persisted_receipt"))
            self.assertEqual(payload.get("receipt_count"), 0)
            self.assertEqual(payload.get("receipt_counts_by_state"), {})
            self.assertEqual(payload.get("receipt_history"), [])

    def test_proactive_action_plan_receipt_rejects_invalid_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            with self.assertRaisesRegex(ValueError, "invalid_proactive_action_plan_receipt_state"):
                svc.proactive_action_plan_receipt(chat_id=7, rank=1, state="delegated")

    def test_operator_focus_delegate_contract_is_populated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            control_room = {
                "health": {"level": "attention"},
                "attention": {
                    "blocked_approvals": [
                        {
                            "order_id": "order-123",
                            "job_id": "job-123",
                            "updated_at": 111.0,
                        }
                    ],
                    "pending_decisions": [],
                    "stalled_tasks": [],
                },
                "recommended_actions": [
                    {
                        "action_id": "approve_blocked_jobs",
                        "count": 1,
                        "label": "Approve blocked jobs",
                        "reason": "A blocked approval needs review.",
                        "target": "/api/v1/orchestration/control-room",
                    }
                ],
                "workflow_bottleneck": {"score": 0},
            }

            with mock.patch.object(svc, "control_room", return_value=control_room), \
                 mock.patch.object(svc, "proactive_priorities", return_value={"orders": []}), \
                 mock.patch.object(svc, "proactive_health", return_value={"status": "not_configured"}):
                report = svc.operator_focus(chat_id=7, limit=5)

            item = (report.get("items") or [None])[0]
            self.assertIsInstance(item, dict)

            delegate_contract = item.get("delegate_contract") if isinstance(item, dict) else None
            self.assertIsInstance(delegate_contract, dict)
            self.assertEqual(delegate_contract.get("delegate_role"), "reviewer_local")
            self.assertEqual(delegate_contract.get("source_action_id"), "approve_blocked_jobs")
            self.assertEqual(delegate_contract.get("handoff_endpoint"), "/api/v1/orchestration/control-room?chat_id=7")
            self.assertEqual(delegate_contract.get("inspect_endpoint"), "/api/v1/orchestration/control-room?chat_id=7")
            self.assertTrue(delegate_contract.get("task_title"))
            self.assertTrue(delegate_contract.get("task_prompt"))
            self.assertIn("approve_blocked_jobs", str(delegate_contract.get("task_prompt")))
            self.assertTrue(delegate_contract.get("acceptance_criteria"))
            self.assertTrue(delegate_contract.get("definition_of_done"))
            self.assertTrue(delegate_contract.get("evidence_required"))
            self.assertTrue(delegate_contract.get("suggested_tests"))
            self.assertTrue(delegate_contract.get("risk_notes"))

    def test_operator_focus_handoff_selects_by_rank_and_action_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            report = {
                "api_version": "v1",
                "schema_version": 1,
                "generated_at": 123.0,
                "chat_id": 7,
                "summary": {"returned": 2},
                "items": [
                    {
                        "rank": 1,
                        "action_id": "focus:first",
                        "label": "First",
                        "next_action": "Do first",
                        "source": "control_room",
                    },
                    {
                        "rank": 2,
                        "action_id": "focus:second",
                        "label": "Second",
                        "next_action": "Do second",
                        "source": "proactive_health",
                    },
                ],
            }

            with mock.patch.object(svc, "operator_focus", return_value=report):
                by_rank = svc.operator_focus_handoff(chat_id=7, rank=2)
                self.assertEqual((by_rank.get("item") or {}).get("action_id"), "focus:second")
                self.assertEqual((by_rank.get("selection") or {}).get("matched_by"), "rank")

                by_action = svc.operator_focus_handoff(chat_id=7, action_id="focus:first", rank=99)
                self.assertEqual((by_action.get("item") or {}).get("action_id"), "focus:first")
                self.assertEqual((by_action.get("selection") or {}).get("matched_by"), "action_id")

                top = svc.operator_focus_handoff(chat_id=7)
                self.assertEqual((top.get("item") or {}).get("action_id"), "focus:first")
                self.assertEqual((top.get("selection") or {}).get("matched_by"), "top")

    def test_operator_focus_briefing_preserves_existing_packet_for_action_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            report = {
                "api_version": "v1",
                "schema_version": 1,
                "generated_at": 123.0,
                "chat_id": 7,
                "summary": {"returned": 2},
                "items": [
                    {
                        "rank": 1,
                        "action_id": "focus:first",
                        "label": "First",
                        "next_action": "Do first",
                        "source": "control_room",
                        "briefing_packet": {
                            "owner_role": "reviewer_local",
                            "action": "Review first",
                            "inspect_endpoint": "/inspect/first",
                            "handoff_endpoint": "/handoff/first",
                            "evidence_required": ["summary"],
                            "suggested_validation": ["check first"],
                            "definition_of_done": ["first done"],
                            "assignment_prompt": "Use the first packet.",
                        },
                    },
                    {
                        "rank": 2,
                        "action_id": "focus:second",
                        "label": "Second",
                        "next_action": "Do second",
                        "source": "proactive_health",
                    },
                ],
            }

            with mock.patch.object(svc, "operator_focus", return_value=report):
                briefing = svc.operator_focus_briefing(chat_id=7, action_id="focus:first")

            self.assertEqual((briefing.get("selection") or {}).get("matched_by"), "action_id")
            self.assertEqual((briefing.get("item_identity") or {}).get("action_id"), "focus:first")
            packet = briefing.get("briefing_packet") or {}
            self.assertEqual(packet.get("owner_role"), "reviewer_local")
            self.assertEqual(packet.get("assignment_prompt"), "Use the first packet.")

    def test_operator_focus_briefing_falls_back_when_packet_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            report = {
                "api_version": "v1",
                "schema_version": 1,
                "generated_at": 123.0,
                "chat_id": 7,
                "summary": {"returned": 2},
                "items": [
                    {
                        "rank": 1,
                        "action_id": "focus:first",
                        "label": "First",
                        "next_action": "Do first",
                        "source": "control_room",
                    },
                    {
                        "rank": 2,
                        "action_id": "focus:second",
                        "label": "Second",
                        "next_action": "Do second",
                        "source": "proactive_health",
                        "inspect_path": "/inspect/second",
                        "target": "/handoff/second",
                    },
                ],
            }

            with mock.patch.object(svc, "operator_focus", return_value=report):
                briefing = svc.operator_focus_briefing(chat_id=7, rank=2)

            self.assertEqual((briefing.get("selection") or {}).get("matched_by"), "rank")
            self.assertEqual((briefing.get("item_identity") or {}).get("action_id"), "focus:second")
            packet = briefing.get("briefing_packet") or {}
            self.assertEqual(packet.get("owner_role"), "operator")
            self.assertEqual(packet.get("inspect_endpoint"), "/inspect/second")
            self.assertEqual(packet.get("handoff_endpoint"), "/handoff/second")
            self.assertEqual(packet.get("evidence_required"), ["completion summary"])
            self.assertIn("focus:second", str(packet.get("assignment_prompt")))

    def test_operator_focus_briefing_bundle_preserves_and_falls_back_packets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            report = {
                "api_version": "v1",
                "schema_version": 1,
                "generated_at": 123.0,
                "chat_id": 7,
                "limit": 2,
                "summary": {"returned": 2},
                "items": [
                    {
                        "rank": 1,
                        "action_id": "focus:first",
                        "label": "First",
                        "next_action": "Do first",
                        "source": "control_room",
                        "briefing_packet": {
                            "owner_role": "reviewer_local",
                            "action": "Review first",
                            "inspect_endpoint": "/inspect/first",
                            "handoff_endpoint": "/handoff/first",
                            "evidence_required": ["summary"],
                            "suggested_validation": ["check first"],
                            "definition_of_done": ["first done"],
                            "assignment_prompt": "Use the first packet.",
                        },
                    },
                    {
                        "rank": 2,
                        "action_id": "focus:second",
                        "label": "Second",
                        "next_action": "Do second",
                        "source": "proactive_health",
                        "inspect_path": "/inspect/second",
                        "target": "/handoff/second",
                    },
                ],
            }

            with mock.patch.object(svc, "operator_focus", return_value=report):
                bundle = svc.operator_focus_briefing_bundle(chat_id=7, limit=2)

            self.assertEqual(bundle.get("api_version"), "v1")
            self.assertEqual(bundle.get("schema_version"), 1)
            self.assertEqual(bundle.get("chat_id"), 7)
            self.assertEqual(bundle.get("limit"), 2)
            self.assertEqual((bundle.get("summary") or {}).get("returned"), 2)

            briefings = bundle.get("briefings") or []
            self.assertEqual(len(briefings), 2)

            first = briefings[0]
            self.assertEqual((first.get("selection") or {}).get("matched_by"), "rank")
            self.assertEqual((first.get("item_identity") or {}).get("action_id"), "focus:first")
            first_packet = first.get("briefing_packet") or {}
            self.assertEqual(first_packet.get("owner_role"), "reviewer_local")
            self.assertEqual(first_packet.get("assignment_prompt"), "Use the first packet.")

            second = briefings[1]
            self.assertEqual((second.get("selection") or {}).get("matched_by"), "rank")
            self.assertEqual((second.get("item_identity") or {}).get("action_id"), "focus:second")
            second_packet = second.get("briefing_packet") or {}
            self.assertEqual(second_packet.get("owner_role"), "operator")
            self.assertEqual(second_packet.get("inspect_endpoint"), "/inspect/second")
            self.assertEqual(second_packet.get("handoff_endpoint"), "/handoff/second")
            self.assertEqual(second_packet.get("evidence_required"), ["completion summary"])
            self.assertIn("focus:second", str(second_packet.get("assignment_prompt")))

    def test_snapshot_assigns_current_and_next_by_worker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 2}})

            # Active order as source-of-truth.
            order_id = "11111111-1111-1111-1111-111111111111"
            q.upsert_order(order_id=order_id, chat_id=1, title="Order", body="Body", status="active", priority=2)

            # Running backend job on slot 1.
            run = Task.new(
                source="telegram",
                role="backend",
                input_text="Do work now",
                request_type="task",
                priority=2,
                model="gpt-5.2",
                effort="medium",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="running",
                parent_job_id=order_id,
                trace={"live_workspace_slot": 1, "live_phase": "codex_run"},
                job_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            )
            q.submit_task(run)
            q.update_state(run.job_id, "running", live_phase="codex_run", live_workspace_slot=1)

            # Queued backend job should become "next" for slot 2.
            nxt = Task.new(
                source="telegram",
                role="backend",
                input_text="Do next thing",
                request_type="task",
                priority=1,
                model="gpt-5.2",
                effort="medium",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
                parent_job_id=order_id,
                job_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            )
            q.submit_task(nxt)

            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 2}}, cache_ttl_seconds=0)
            snap = svc.snapshot(chat_id=1)

            self.assertEqual(snap["chat_id"], 1)
            self.assertTrue(len(snap["orders_active"]) >= 1)
            self.assertTrue(any(o["order_id"] == order_id for o in snap["orders_active"]))

            workers = [w for w in snap["workers"] if w["role"] == "backend"]
            self.assertEqual(len(workers), 2)

            w1 = next(w for w in workers if w["slot"] == 1)
            self.assertIsNotNone(w1["current"])
            self.assertEqual(w1["current"]["job_id_short"], "aaaaaaaa")

            w2 = next(w for w in workers if w["slot"] == 2)
            self.assertIsNone(w2["current"])
            self.assertIsNotNone(w2["next"])
            self.assertEqual(w2["next"]["job_id_short"], "bbbbbbbb"[:8])

            self.assertIsNotNone(snap["snapshot_hash"])
            self.assertIn("alerts", snap)
            self.assertIn("risks", snap)
            self.assertIn("decisions_pending", snap)
            self.assertIn("live_view", snap)
            self.assertEqual(str((snap.get("live_view") or {}).get("transport")), "sse")

    def test_control_room_includes_due_runbooks_and_recommended_action(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runbooks_path = root / "runbooks.yaml"
            runbooks_path.write_text(
                "\n".join(
                    [
                        "- id: due-a",
                        "  role: sre",
                        "  interval_seconds: 300",
                        "  enabled: true",
                        "  prompt: Check A.",
                        "- id: due-b",
                        "  role: qa",
                        "  interval_seconds: 300",
                        "  enabled: true",
                        "  prompt: Check B.",
                    ]
                ),
                encoding="utf-8",
            )
            storage = SQLiteTaskStorage(root / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"sre": {"role": "sre"}, "qa": {"role": "qa"}})
            svc = StatusService(
                orch_q=q,
                role_profiles={"sre": {"role": "sre"}, "qa": {"role": "qa"}},
                cache_ttl_seconds=0,
                runbooks_path=runbooks_path,
            )

            room = svc.control_room()

            runbooks = room.get("runbooks") or {}
            self.assertEqual((runbooks.get("summary") or {}).get("due"), 2)
            due_ids = {str(item.get("runbook_id")) for item in list(runbooks.get("due_items") or [])}
            self.assertEqual(due_ids, {"due-a", "due-b"})

            actions = list(room.get("recommended_actions") or [])
            action = next((item for item in actions if item.get("action_id") == "inspect_due_runbooks"), None)
            self.assertIsNotNone(action)
            self.assertEqual((action or {}).get("target"), "/api/v1/orchestration/runbooks")
            self.assertEqual((action or {}).get("count"), 2)

    def test_snapshot_filters_by_chat_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})

            order1 = "11111111-1111-1111-1111-111111111111"
            order2 = "22222222-2222-2222-2222-222222222222"
            q.upsert_order(order_id=order1, chat_id=1, title="Order1", body="Body", status="active", priority=2)
            q.upsert_order(order_id=order2, chat_id=2, title="Order2", body="Body", status="active", priority=2)

            t1 = Task.new(
                source="telegram",
                role="backend",
                input_text="Chat1 queued",
                request_type="task",
                priority=2,
                model="gpt-5.2",
                effort="medium",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
                parent_job_id=order1,
                job_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
            )
            t2 = Task.new(
                source="telegram",
                role="backend",
                input_text="Chat2 queued",
                request_type="task",
                priority=2,
                model="gpt-5.2",
                effort="medium",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=2,
                state="queued",
                parent_job_id=order2,
                job_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
            )
            q.submit_task(t1)
            q.submit_task(t2)

            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            snap1 = svc.snapshot(chat_id=1)
            self.assertEqual(snap1["chat_id"], 1)
            self.assertEqual(int(snap1["queued_total"]), 1)
            workers1 = [w for w in snap1["workers"] if w["role"] == "backend"]
            self.assertEqual(len(workers1), 1)
            self.assertIsNotNone(workers1[0]["next"])
            self.assertEqual(workers1[0]["next"]["job_id"], t1.job_id)

            snap2 = svc.snapshot(chat_id=2)
            self.assertEqual(snap2["chat_id"], 2)
            self.assertEqual(int(snap2["queued_total"]), 1)
            workers2 = [w for w in snap2["workers"] if w["role"] == "backend"]
            self.assertEqual(len(workers2), 1)
            self.assertIsNotNone(workers2[0]["next"])
            self.assertEqual(workers2[0]["next"]["job_id"], t2.job_id)

    def test_snapshot_worker_current_next_are_chat_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 2}})

            order1 = "aaaa1111-1111-1111-1111-111111111111"
            order2 = "bbbb2222-2222-2222-2222-222222222222"
            q.upsert_order(order_id=order1, chat_id=1, title="Order1", body="Body", status="active", priority=2)
            q.upsert_order(order_id=order2, chat_id=2, title="Order2", body="Body", status="active", priority=2)

            run_chat1 = Task.new(
                source="telegram",
                role="backend",
                input_text="Chat1 running",
                request_type="task",
                priority=2,
                model="gpt-5.2",
                effort="medium",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="running",
                parent_job_id=order1,
                trace={"live_workspace_slot": 1},
                job_id="11111111-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            )
            run_chat2 = Task.new(
                source="telegram",
                role="backend",
                input_text="Chat2 running",
                request_type="task",
                priority=2,
                model="gpt-5.2",
                effort="medium",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=2,
                state="running",
                parent_job_id=order2,
                trace={"live_workspace_slot": 1},
                job_id="22222222-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            )
            next_chat1 = Task.new(
                source="telegram",
                role="backend",
                input_text="Chat1 queued",
                request_type="task",
                priority=1,
                model="gpt-5.2",
                effort="medium",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
                parent_job_id=order1,
                job_id="33333333-cccc-cccc-cccc-cccccccccccc",
            )
            next_chat2 = Task.new(
                source="telegram",
                role="backend",
                input_text="Chat2 queued",
                request_type="task",
                priority=1,
                model="gpt-5.2",
                effort="medium",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=2,
                state="queued",
                parent_job_id=order2,
                job_id="44444444-dddd-dddd-dddd-dddddddddddd",
            )
            q.submit_task(run_chat1)
            q.submit_task(run_chat2)
            q.submit_task(next_chat1)
            q.submit_task(next_chat2)
            q.update_state(run_chat1.job_id, "running", live_workspace_slot=1)
            q.update_state(run_chat2.job_id, "running", live_workspace_slot=1)

            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 2}}, cache_ttl_seconds=0)

            snap1 = svc.snapshot(chat_id=1)
            workers1 = [w for w in snap1["workers"] if w["role"] == "backend"]
            self.assertEqual(len(workers1), 2)
            slot1_chat1 = next(w for w in workers1 if w["slot"] == 1)
            slot2_chat1 = next(w for w in workers1 if w["slot"] == 2)
            self.assertIsNotNone(slot1_chat1["current"])
            self.assertEqual(slot1_chat1["current"]["job_id"], run_chat1.job_id)
            self.assertIsNotNone(slot2_chat1["next"])
            self.assertEqual(slot2_chat1["next"]["job_id"], next_chat1.job_id)

            exposed_ids_chat1 = {
                (slot1_chat1.get("current") or {}).get("job_id"),
                (slot1_chat1.get("next") or {}).get("job_id"),
                (slot2_chat1.get("current") or {}).get("job_id"),
                (slot2_chat1.get("next") or {}).get("job_id"),
            }
            self.assertNotIn(run_chat2.job_id, exposed_ids_chat1)
            self.assertNotIn(next_chat2.job_id, exposed_ids_chat1)

            snap2 = svc.snapshot(chat_id=2)
            workers2 = [w for w in snap2["workers"] if w["role"] == "backend"]
            self.assertEqual(len(workers2), 2)
            slot1_chat2 = next(w for w in workers2 if w["slot"] == 1)
            slot2_chat2 = next(w for w in workers2 if w["slot"] == 2)
            self.assertIsNotNone(slot1_chat2["current"])
            self.assertEqual(slot1_chat2["current"]["job_id"], run_chat2.job_id)
            self.assertIsNotNone(slot2_chat2["next"])
            self.assertEqual(slot2_chat2["next"]["job_id"], next_chat2.job_id)

            exposed_ids_chat2 = {
                (slot1_chat2.get("current") or {}).get("job_id"),
                (slot1_chat2.get("next") or {}).get("job_id"),
                (slot2_chat2.get("current") or {}).get("job_id"),
                (slot2_chat2.get("next") or {}).get("job_id"),
            }
            self.assertNotIn(run_chat1.job_id, exposed_ids_chat2)
            self.assertNotIn(next_chat1.job_id, exposed_ids_chat2)

    def test_snapshot_includes_pending_decisions_from_decision_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})

            order_id = "33333333-3333-3333-3333-333333333333"
            q.upsert_order(order_id=order_id, chat_id=11, title="Order3", body="Body", status="active", priority=2)
            q.append_decision_log(
                order_id=order_id,
                job_id="eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                kind="qa_verdict",
                state="blocked",
                summary="Falta evidencia QA",
                next_action="Aprobar evidencia o pedir corrección",
                details={"gate": "qa"},
            )

            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)
            snap = svc.snapshot(chat_id=11)
            pending = list(snap.get("decisions_pending") or [])
            self.assertTrue(len(pending) >= 1)
            self.assertTrue(any(str(x.get("next_action") or "").startswith("Aprobar evidencia") for x in pending))

    def test_snapshot_merges_factory_callback_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})

            svc = StatusService(
                orch_q=q,
                role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}},
                cache_ttl_seconds=0,
                factory_snapshot_fn=lambda chat_id: {
                    "factory": {"status": "active", "mode": "ceo-bounded"},
                    "repos": [{"repo_id": "repo-a", "path": "/tmp/repo-a", "status": "active"}],
                    "heartbeats": [{"agent_key": "repo-a:skynet", "stale": False}],
                    "models": {"local_ollama": {"healthy": True}},
                    "alerts": [{"kind": "factory_test", "severity": "warning", "summary": "factory callback merged"}],
                },
            )
            snap = svc.snapshot(chat_id=1)
            self.assertEqual((snap.get("factory") or {}).get("status"), "active")
            self.assertEqual(len(list(snap.get("repos") or [])), 1)
            self.assertEqual(len(list(snap.get("heartbeats") or [])), 1)
            self.assertTrue(any(str(item.get("kind") or "") == "factory_test" for item in list(snap.get("alerts") or [])))

    def test_operator_focus_decorates_latest_receipt_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            control_room = {
                "health": {"level": "attention"},
                "attention": {
                    "blocked_approvals": [
                        {
                            "order_id": "order-123",
                            "job_id": "job-123",
                            "updated_at": 111.0,
                        }
                    ],
                    "pending_decisions": [],
                    "stalled_tasks": [],
                },
                "recommended_actions": [
                    {
                        "action_id": "approve_blocked_jobs",
                        "count": 1,
                        "label": "Approve blocked jobs",
                        "reason": "A blocked approval needs review.",
                        "target": "/api/v1/orchestration/control-room",
                    }
                ],
                "workflow_bottleneck": {"score": 0},
            }
            receipt_rows = [
                {
                    "kind": "operator_focus_receipt",
                    "state": "acknowledged",
                    "summary": "Old receipt",
                    "next_action": "Wait",
                    "order_id": "order-123",
                    "job_id": "job-123",
                    "ts": 100.0,
                    "details": {
                        "actor": "reviewer_local",
                        "selection": {"action_id": "approve_blocked_jobs", "rank": 1},
                        "item_identity": {"action_id": "approve_blocked_jobs", "label": "Approve blocked jobs"},
                        "operator_focus_details": {"note": "old"},
                    },
                },
                {
                    "kind": "operator_focus_receipt",
                    "state": "completed",
                    "summary": "Latest receipt",
                    "next_action": "Ship it",
                    "order_id": "order-123",
                    "job_id": "job-123",
                    "ts": 200.0,
                    "details": {
                        "actor": "implementer_local",
                        "selection": {"action_id": "approve_blocked_jobs", "rank": 1},
                        "item_identity": {"action_id": "approve_blocked_jobs", "label": "Approve blocked jobs"},
                        "operator_focus_details": {"note": "latest"},
                    },
                },
            ]

            with mock.patch.object(svc, "control_room", return_value=control_room), \
                 mock.patch.object(svc, "proactive_priorities", return_value={"orders": []}), \
                 mock.patch.object(svc, "proactive_health", return_value={"status": "not_configured"}), \
                 mock.patch.object(q, "list_decision_log", return_value=receipt_rows):
                report = svc.operator_focus(chat_id=7, limit=5)

            item = (report.get("items") or [None])[0]
            self.assertIsInstance(item, dict)
            self.assertEqual(item.get("receipt_state"), "completed")
            self.assertEqual(item.get("receipt_count"), 2)
            self.assertEqual(item.get("receipt_counts_by_state"), {"acknowledged": 1, "completed": 1})
            receipt_history = item.get("receipt_history") or []
            self.assertEqual(len(receipt_history), 2)
            self.assertEqual(receipt_history[0].get("state"), "completed")
            self.assertEqual(receipt_history[0].get("summary"), "Latest receipt")
            self.assertEqual(receipt_history[1].get("state"), "acknowledged")
            self.assertEqual(receipt_history[1].get("summary"), "Old receipt")
            latest_receipt = item.get("latest_receipt") or {}
            self.assertEqual(latest_receipt.get("state"), "completed")
            self.assertEqual(latest_receipt.get("summary"), "Latest receipt")
            self.assertEqual(latest_receipt.get("actor"), "implementer_local")
            self.assertEqual((latest_receipt.get("details") or {}).get("note"), "latest")
            self.assertEqual(latest_receipt, receipt_history[0])

    def test_operator_focus_receipt_trail_returns_limited_newest_first_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}})
            svc = StatusService(orch_q=q, role_profiles={"backend": {"role": "backend", "max_parallel_jobs": 1}}, cache_ttl_seconds=0)

            control_room = {
                "health": {"level": "attention"},
                "attention": {
                    "blocked_approvals": [
                        {
                            "order_id": "order-123",
                            "job_id": "job-123",
                            "updated_at": 111.0,
                        }
                    ],
                    "pending_decisions": [],
                    "stalled_tasks": [],
                },
                "recommended_actions": [
                    {
                        "action_id": "approve_blocked_jobs",
                        "count": 1,
                        "label": "Approve blocked jobs",
                        "reason": "A blocked approval needs review.",
                        "target": "/api/v1/orchestration/control-room",
                    }
                ],
                "workflow_bottleneck": {"score": 0},
            }
            receipt_rows = [
                {
                    "kind": "operator_focus_receipt",
                    "state": "acknowledged",
                    "summary": "Old receipt",
                    "next_action": "Wait",
                    "order_id": "order-123",
                    "job_id": "job-123",
                    "ts": 100.0,
                    "details": {
                        "actor": "reviewer_local",
                        "selection": {"action_id": "approve_blocked_jobs", "rank": 1},
                        "item_identity": {"action_id": "approve_blocked_jobs", "label": "Approve blocked jobs"},
                        "operator_focus_details": {"note": "old"},
                    },
                },
                {
                    "kind": "operator_focus_receipt",
                    "state": "completed",
                    "summary": "Latest receipt",
                    "next_action": "Ship it",
                    "order_id": "order-123",
                    "job_id": "job-123",
                    "ts": 200.0,
                    "details": {
                        "actor": "implementer_local",
                        "selection": {"action_id": "approve_blocked_jobs", "rank": 1},
                        "item_identity": {"action_id": "approve_blocked_jobs", "label": "Approve blocked jobs"},
                        "operator_focus_details": {"note": "latest"},
                    },
                },
            ]

            with mock.patch.object(svc, "control_room", return_value=control_room), \
                 mock.patch.object(svc, "proactive_priorities", return_value={"orders": []}), \
                 mock.patch.object(svc, "proactive_health", return_value={"status": "not_configured"}), \
                 mock.patch.object(q, "list_decision_log", return_value=receipt_rows):
                trail = svc.operator_focus_receipt_trail(chat_id=7, rank=1, limit=1)

            self.assertEqual(trail.get("receipt_count"), 2)
            self.assertEqual(trail.get("receipt_counts_by_state"), {"acknowledged": 1, "completed": 1})
            receipts = trail.get("receipts") or []
            self.assertEqual(len(receipts), 1)
            self.assertEqual(receipts[0].get("state"), "completed")
            self.assertEqual(receipts[0].get("summary"), "Latest receipt")
            latest_receipt = trail.get("latest_receipt") or {}
            self.assertEqual(latest_receipt, receipts[0])
            item_identity = trail.get("item_identity") or {}
            self.assertEqual(item_identity.get("action_id"), "approve_blocked_jobs")
            self.assertEqual(item_identity.get("receipt_state"), "completed")
