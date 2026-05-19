from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.status_service import StatusService
from orchestrator.storage import SQLiteTaskStorage


class TestStatusService(unittest.TestCase):
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
            self.assertEqual(top_packet, lanes["advance"]["execution_packet"])
            self.assertEqual(top_packet["order_id"], "advance-order")
            self.assertEqual(top_packet["owner_role"], "implementer_local")
            self.assertEqual(top_packet["lane"], "advance")
            self.assertEqual(top_packet["action"], "Implement the bounded slice.")
            self.assertEqual(top_packet["inspect_endpoint"], "/inspect/advance-order")
            self.assertEqual(top_packet["handoff_endpoint"], "/handoff/advance-order")
            self.assertEqual(top_packet["acceptance_criteria"], ["Open the order."])
            self.assertEqual(top_packet["definition_of_done"], ["Tests pass."])
            self.assertEqual(top_packet["evidence_required"], ["pytest output"])
            self.assertEqual(top_packet["suggested_validation"], ["Run focused tests."])
            self.assertIn("ROLE: implementer_local.", top_packet["assignment_prompt"])

            self.assertEqual(
                plan["summary"]["next_delegate"],
                {
                    "owner_role": "implementer_local",
                    "order_id": "advance-order",
                    "lane": "advance",
                    "action": "Implement the bounded slice.",
                    "inspect_endpoint": "/inspect/advance-order",
                    "handoff_endpoint": "/handoff/advance-order",
                },
            )

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
            order = lanes["advance"]["orders"][0]
            self.assertEqual(order["selection_quality"]["status"], "needs_review")
            self.assertIn("weak_selection_evidence", order["selection_quality"]["flags"])
            self.assertEqual(plan["summary"]["selection_quality"]["needs_review"], 1)

            top_packet = plan["top_execution_packet"]
            self.assertEqual(top_packet["order_id"], order_id)
            self.assertEqual(top_packet["owner_role"], "architect_local")
            self.assertIn("Selection review", top_packet["action"])
            self.assertIn("kill/continue", top_packet["action"])
            self.assertIn("factory-value", " ".join(top_packet["evidence_required"]))

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
