from __future__ import annotations

import os
import unittest
from unittest import mock

from tools import ceo_plane_worker


class TestCEOPlaneWorkerDefaults(unittest.TestCase):
    def test_default_env_isolates_ceo_plane_paths(self) -> None:
        env = ceo_plane_worker._ceo_plane_default_env(
            home="/tmp/example-home",
            environ={},
        )

        self.assertEqual(env["BOT_CEO_PLANE_ONLY"], "1")
        self.assertEqual(env["BOT_ORCHESTRATOR_DB_PATH"], "/tmp/example-home/codexbot/data/ceo_jobs.sqlite")
        self.assertEqual(env["BOT_CEO_PLANE_DB_PATH"], env["BOT_ORCHESTRATOR_DB_PATH"])
        self.assertEqual(env["BOT_STATE_FILE"], "/tmp/example-home/codexbot/data/ceo_state.json")
        self.assertEqual(env["BOT_CEO_PLANE_STATE_FILE"], env["BOT_STATE_FILE"])
        self.assertEqual(env["BOT_WORKTREE_ROOT"], "/tmp/example-home/codexbot/data/ceo_worktrees")
        self.assertEqual(env["BOT_CEO_PLANE_WORKTREE_ROOT"], env["BOT_WORKTREE_ROOT"])
        self.assertEqual(env["BOT_ARTIFACTS_ROOT"], "/tmp/example-home/codexbot/data/ceo_artifacts")
        self.assertEqual(env["BOT_CEO_PLANE_ARTIFACTS_ROOT"], env["BOT_ARTIFACTS_ROOT"])
        self.assertEqual(env["BOT_CEO_PLANE_SERVICE_NAME"], "poncebot-ceo.service")

    def test_default_env_preserves_overrides_consistently(self) -> None:
        overrides = {
            "BOT_CEO_PLANE_DB_PATH": "/srv/ceo/custom.sqlite",
            "BOT_CEO_PLANE_STATE_FILE": "/srv/ceo/state.json",
            "BOT_CEO_PLANE_WORKTREE_ROOT": "/srv/ceo/worktrees",
            "BOT_CEO_PLANE_ARTIFACTS_ROOT": "/srv/ceo/artifacts",
            "BOT_CEO_PLANE_SERVICE_NAME": "custom-ceo.service",
        }

        env = ceo_plane_worker._ceo_plane_default_env(
            home="/tmp/example-home",
            environ=overrides,
        )

        self.assertEqual(env["BOT_ORCHESTRATOR_DB_PATH"], overrides["BOT_CEO_PLANE_DB_PATH"])
        self.assertEqual(env["BOT_STATE_FILE"], overrides["BOT_CEO_PLANE_STATE_FILE"])
        self.assertEqual(env["BOT_WORKTREE_ROOT"], overrides["BOT_CEO_PLANE_WORKTREE_ROOT"])
        self.assertEqual(env["BOT_ARTIFACTS_ROOT"], overrides["BOT_CEO_PLANE_ARTIFACTS_ROOT"])
        self.assertEqual(env["BOT_CEO_PLANE_SERVICE_NAME"], overrides["BOT_CEO_PLANE_SERVICE_NAME"])

    def test_set_defaults_updates_process_environment(self) -> None:
        with mock.patch.dict(os.environ, {"HOME": "/tmp/ceo-home"}, clear=True):
            ceo_plane_worker._set_ceo_plane_defaults()

            self.assertEqual(os.environ["BOT_CEO_PLANE_ONLY"], "1")
            self.assertEqual(os.environ["BOT_ORCHESTRATOR_DB_PATH"], "/tmp/ceo-home/codexbot/data/ceo_jobs.sqlite")
            self.assertEqual(os.environ["BOT_STATE_FILE"], "/tmp/ceo-home/codexbot/data/ceo_state.json")
            self.assertEqual(os.environ["BOT_WORKTREE_ROOT"], "/tmp/ceo-home/codexbot/data/ceo_worktrees")
            self.assertEqual(os.environ["BOT_ARTIFACTS_ROOT"], "/tmp/ceo-home/codexbot/data/ceo_artifacts")


if __name__ == "__main__":
    unittest.main()
