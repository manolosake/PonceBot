from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from orchestrator.delegation import parse_jarvis_subtasks
from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.storage import SQLiteTaskStorage


class TestRunnablePartitionFlow(unittest.TestCase):
    def test_blocker_approval_isolated_and_runnable_claimed(self) -> None:
        specs = parse_jarvis_subtasks(
            {
                "subtasks": [
                    {"key": "07625372", "role": "backend", "text": "Approval-dependent blocker", "priority": 1},
                    {"key": "run_backend", "role": "backend", "text": "Runnable backend task", "depends_on": ["07625372"], "priority": 1},
                ]
            }
        )
        by_key = {s.key: s for s in specs}
        self.assertTrue(by_key["07625372"].requires_approval)
        self.assertEqual(by_key["run_backend"].depends_on, [])

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.sqlite"
            storage = SQLiteTaskStorage(db_path)
            queue = OrchestratorQueue(storage)

            key_to_job = {k: str(uuid.uuid4()) for k in by_key.keys()}
            for key, spec in by_key.items():
                deps = [key_to_job[d] for d in spec.depends_on if d in key_to_job]
                queue.submit_task(
                    Task.new(
                        job_id=key_to_job[key],
                        source="test",
                        role=spec.role,
                        input_text=spec.text,
                        request_type="task",
                        priority=1,
                        model="gpt-5.2",
                        effort="medium",
                        mode_hint="rw",
                        requires_approval=spec.requires_approval,
                        max_cost_window_usd=1.0,
                        chat_id=1,
                        depends_on=deps,
                    )
                )

            claimed = queue.take_next(role="backend")
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed.job_id, key_to_job["run_backend"])

            blocker = queue.get_job(key_to_job["07625372"])
            self.assertIsNotNone(blocker)
            self.assertEqual(blocker.state, "blocked_approval")


if __name__ == "__main__":
    unittest.main()
