from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.status_service import StatusService
from orchestrator.storage import SQLiteTaskStorage


class TestStatusWorkflowSummary(unittest.TestCase):
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
