from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.status_service import StatusService
from orchestrator.storage import SQLiteTaskStorage


class TestStatusService(unittest.TestCase):
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

