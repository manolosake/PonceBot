from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.status_service import StatusService
from orchestrator.storage import SQLiteTaskStorage


class TestStatusWorkflowSummary(unittest.TestCase):
    def test_order_evidence_packet_includes_workflow_logs_and_artifact_refs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = SQLiteTaskStorage(root / "jobs.sqlite")
            profiles = {
                "skynet": {"role": "skynet", "max_parallel_jobs": 1},
                "implementer_local": {"role": "implementer_local", "max_parallel_jobs": 1},
            }
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)

            order_id = "11111111-2222-3333-4444-666666666666"
            child_id = "aaaaaaaa-bbbb-cccc-dddd-ffffffffffff"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Evidence packet",
                body="Collect order evidence.",
                status="active",
                priority=1,
                phase="executing",
                project_id="codexbot",
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="skynet",
                    input_text="Root order",
                    request_type="maintenance",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    is_autonomous=True,
                    artifacts_dir=str(root / "artifacts" / order_id),
                    trace={"proactive_slices_applied": 1, "result_artifacts": ["root-summary.md"]},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="implementer_local",
                    input_text="Implement evidence endpoint",
                    request_type="task",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    is_autonomous=True,
                    parent_job_id=order_id,
                    artifacts_dir=str(root / "artifacts" / child_id),
                    labels={"ticket": order_id, "kind": "subtask"},
                    trace={"result_artifacts": ["tests/status.log", "diff.patch"]},
                    job_id=child_id,
                )
            )
            q.append_trace_event(
                order_id=order_id,
                job_id=child_id,
                agent_role="implementer_local",
                event_type="job.done",
                severity="info",
                message="child completed",
                artifact_id="artifact-1",
                payload={"result_artifacts": ["trace/final.json"]},
            )
            q.append_decision_log(
                order_id=order_id,
                job_id=child_id,
                kind="implementation",
                state="done",
                summary="Endpoint implemented",
                next_action=None,
                details={"ok": True},
            )
            q.append_delegation_edge(
                root_ticket_id=order_id,
                from_job_id=order_id,
                to_job_id=child_id,
                edge_type="delegated",
                to_role="implementer_local",
                to_key="evidence_packet",
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)

            self.assertEqual(packet["api_version"], "v1")
            self.assertEqual(packet["schema_version"], 1)
            self.assertEqual(packet["order_id"], order_id)
            self.assertEqual(packet["workflow"]["current_stage"], "validation")
            self.assertEqual(len(packet["children"]), 1)
            self.assertEqual(packet["children"][0]["job_id"], child_id)
            self.assertEqual(len(packet["traces"]), 1)
            self.assertEqual(packet["traces"][0]["artifact_id"], "artifact-1")
            self.assertEqual(len(packet["decision_log"]), 1)
            self.assertEqual(len(packet["delegation_log"]), 1)

            artifacts = packet["artifacts"]
            artifact_paths = {str(a.get("path")) for a in artifacts if a.get("path")}
            artifact_ids = {str(a.get("artifact_id")) for a in artifacts if a.get("artifact_id")}
            self.assertIn(str(root / "artifacts" / order_id), artifact_paths)
            self.assertIn(str(root / "artifacts" / child_id), artifact_paths)
            self.assertIn("root-summary.md", artifact_paths)
            self.assertIn("tests/status.log", artifact_paths)
            self.assertIn("diff.patch", artifact_paths)
            self.assertIn("trace/final.json", artifact_paths)
            self.assertIn("artifact-1", artifact_ids)
            self.assertEqual(packet["counts"]["children"], 1)
            self.assertEqual(packet["counts"]["artifacts"], len(artifacts))

    def test_snapshot_includes_end_to_end_order_workflow_and_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = SQLiteTaskStorage(root / "jobs.sqlite")
            profiles = {
                "skynet": {"role": "skynet", "max_parallel_jobs": 1},
                "backend": {"role": "backend", "max_parallel_jobs": 1},
                "qa": {"role": "qa", "max_parallel_jobs": 1},
            }
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)

            order_id = "11111111-2222-3333-4444-555555555555"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: codexbot Reliability + Delivery",
                body="Ship one bounded improvement.",
                status="active",
                priority=2,
                phase="review",
                project_id="codexbot-6fb8d5b9",
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="skynet",
                    input_text="AUTONOMOUS PROACTIVE SPRINT",
                    request_type="maintenance",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    is_autonomous=True,
                    trace={
                        "allow_delegation": True,
                        "proactive_lane": True,
                        "proactive_slices_started": 1,
                        "proactive_slices_applied": 1,
                        "proactive_slices_validated": 0,
                        "proactive_slices_closed": 0,
                        "proactive_quality_gate_status": "applied",
                    },
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="backend",
                    input_text="Implement one bounded reliability fix.",
                    request_type="task",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    is_autonomous=True,
                    parent_job_id=order_id,
                    labels={"ticket": order_id, "kind": "subtask", "key": "backend_fix"},
                    job_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                )
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="skynet",
                    input_text="FINAL SWEEP",
                    request_type="maintenance",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="failed",
                    is_autonomous=True,
                    parent_job_id=order_id,
                    labels={"ticket": order_id, "kind": "final_sweep"},
                    trace={
                        "result_summary": "Codex returned code 1.",
                        "live_stdout_tail": (
                            "{\"detail\":\"The 'gpt-5.2-codex' model is not supported when using Codex with a ChatGPT account.\"}"
                        ),
                    },
                    job_id="ffffffff-1111-2222-3333-444444444444",
                )
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            snap = svc.snapshot(chat_id=1)

            self.assertEqual(len(snap["orders_active"]), 1)
            workflow = snap["orders_active"][0]["workflow"]
            self.assertEqual(workflow["current_stage"], "skynet_review")

            stages = {stage["stage"]: stage for stage in workflow["stages"]}
            self.assertEqual(stages["skynet_plan"]["status"], "done")
            self.assertEqual(stages["delivery"]["status"], "done")
            self.assertEqual(stages["validation"]["status"], "pending")
            self.assertEqual(stages["skynet_review"]["status"], "failed")
            self.assertEqual(stages["deploy"]["status"], "pending")

            blockers = workflow["blockers"]
            self.assertEqual(len(blockers), 1)
            self.assertEqual(blockers[0]["stage"], "skynet_review")
            self.assertIn("unsupported", str(blockers[0]["summary"]).lower())
            self.assertEqual(snap["order_workflows"][0]["order_id"], order_id)


if __name__ == "__main__":
    unittest.main()
