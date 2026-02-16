from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.storage import SQLiteTaskStorage


class TestStructuredLogs(unittest.TestCase):
    def test_decision_log_is_written_for_jarvis_terminal_states(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            order_id = "11111111-1111-1111-1111-111111111111"
            t = Task.new(
                source="telegram",
                role="jarvis",
                input_text="AUTOPILOT TICK",
                request_type="maintenance",
                priority=2,
                model="gpt-5.2",
                effort="medium",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
                parent_job_id=order_id,
                labels={"kind": "autopilot", "order": order_id},
                trace={"order_id": order_id},
                job_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            )
            q.submit_task(t)

            q.update_state(
                t.job_id,
                "done",
                result_summary="Decision: delegate to backend",
                result_next_action="check progress",
                result_status="ok",
            )

            rows = q.list_decision_log(order_id=order_id, limit=10)
            self.assertTrue(rows)
            self.assertEqual(rows[0]["order_id"], order_id)
            self.assertEqual(rows[0]["job_id_short"], "aaaaaaaa")
            self.assertEqual(rows[0]["kind"], "autopilot")
            self.assertIn("Decision:", rows[0]["summary"])

    def test_worker_activity_log_written_on_state_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            t = Task.new(
                source="telegram",
                role="backend",
                input_text="work",
                request_type="task",
                priority=2,
                model="gpt-5.2",
                effort="medium",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
                job_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            )
            q.submit_task(t)
            q.update_state(t.job_id, "running", live_workspace_slot=1, live_phase="codex_run")

            items = q.list_worker_activity(role="backend", since_ts=None, limit=50)
            self.assertTrue(items)
            self.assertEqual(items[0]["role"], "backend")
            self.assertEqual(items[0]["state"], "running")
            self.assertEqual(items[0]["worker_slot"], 1)

    def test_delegation_edges_persist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            root = "cccccccc-cccc-cccc-cccc-cccccccccccc"
            q.append_delegation_edge(
                root_ticket_id=root,
                from_job_id="parent",
                to_job_id="child",
                edge_type="delegated",
                to_role="backend",
                to_key="a",
                details={"priority": 2},
            )
            q.append_delegation_edge(
                root_ticket_id=root,
                from_job_id="child",
                to_job_id="dep",
                edge_type="depends_on",
                to_role=None,
                to_key=None,
                details={},
            )

            items = q.list_delegation_log(root_ticket_id=root, limit=10)
            self.assertEqual(len(items), 2)
            kinds = {it["edge_type"] for it in items}
            self.assertIn("delegated", kinds)
            self.assertIn("depends_on", kinds)

