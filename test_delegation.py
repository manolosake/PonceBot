from __future__ import annotations

import unittest

from orchestrator.delegation import parse_jarvis_subtasks


class TestDelegationPartition(unittest.TestCase):
    def test_isolated_blocker_becomes_approval_dependent(self) -> None:
        specs = parse_jarvis_subtasks(
            {
                "subtasks": [
                    {"key": "07625372", "role": "backend", "text": "approval path"},
                ]
            }
        )
        self.assertEqual(len(specs), 1)
        self.assertTrue(specs[0].requires_approval)

    def test_runnable_strips_dependency_on_isolated_blocker(self) -> None:
        specs = parse_jarvis_subtasks(
            {
                "subtasks": [
                    {"key": "07625372", "role": "backend", "text": "approval path"},
                    {"key": "run_01", "role": "backend", "text": "runnable", "depends_on": ["07625372"]},
                ]
            }
        )
        by_key = {s.key: s for s in specs}
        self.assertIn("run_01", by_key)
        self.assertEqual(by_key["run_01"].depends_on, [])
        self.assertFalse(by_key["run_01"].requires_approval)

    def test_runnable_keeps_non_approval_dependencies(self) -> None:
        specs = parse_jarvis_subtasks(
            {
                "subtasks": [
                    {"key": "prep", "role": "backend", "text": "prep"},
                    {"key": "run_02", "role": "backend", "text": "runnable", "depends_on": ["prep"]},
                ]
            }
        )
        by_key = {s.key: s for s in specs}
        self.assertEqual(by_key["run_02"].depends_on, ["prep"])


if __name__ == "__main__":
    unittest.main()
