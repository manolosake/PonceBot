from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Allow running tests from either repo root (e.g. `python3 -m unittest codexbot/test_bot.py`)
# or from within `codexbot/` (e.g. `cd codexbot && python3 -m unittest test_bot.py`).
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bot  # noqa: E402


class TestStateHandling(unittest.TestCase):
    def _cfg(self, state_file: Path) -> bot.BotConfig:
        return bot.BotConfig(
            telegram_token="dummy",
            allowed_chat_ids={1},
            allowed_user_ids={2},
            unsafe_direct_codex=False,
            poll_timeout_seconds=1,
            http_timeout_seconds=1,
            http_max_retries=0,
            http_retry_initial_seconds=0.0,
            http_retry_max_seconds=0.0,
            unauthorized_reply_cooldown_seconds=1,
            drain_updates_on_start=False,
            worker_count=1,
            queue_maxsize=0,
            max_queued_per_chat=1,
            heartbeat_seconds=0,
            send_as_file_threshold_chars=0,
            max_download_bytes=0,
            strict_proxy=False,
            transcribe_audio=False,
            transcribe_backend="auto",
            transcribe_timeout_seconds=300,
            ffmpeg_bin="ffmpeg",
            whispercpp_bin="whisper-cli",
            whispercpp_model_path=str((Path(__file__).resolve().parent / "models" / "ggml-medium.bin")),
            whispercpp_threads=1,
            openai_api_key="",
            openai_api_base_url="https://api.openai.com",
            transcribe_model="gpt-4o-transcribe",
            transcribe_language="es",
            transcribe_prompt="",
            transcribe_max_bytes=25 * 1024 * 1024,
            telegram_parse_mode="HTML",
            auth_enabled=False,
            auth_session_ttl_seconds=12 * 60 * 60,
            auth_users_file=(Path(__file__).resolve().parent / "users.json"),
            auth_profiles_file=(Path(__file__).resolve().parent / "profiles.json"),
            state_file=state_file,
            notify_chat_id=None,
            notify_on_start=False,
            codex_workdir=Path(".").resolve(),
            codex_timeout_seconds=1,
            codex_use_oss=False,
            codex_local_provider="ollama",
            codex_oss_model="",
            codex_openai_model="",
            codex_default_mode="ro",
            codex_force_full_access=False,
            codex_dangerous_bypass_sandbox=False,
            admin_user_ids=frozenset(),
            admin_chat_ids=frozenset(),
        )

    def test_setnotify_preserves_existing_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            state_file.write_text(json.dumps({"openai_model": "gpt-4.1", "oss_model": "qwen2.5-coder:7b"}) + "\n")

            cfg = self._cfg(state_file)
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=123,
                user_id=2,
                message_id=99,
                username="u",
                text="/setnotify",
            )

            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "OK. notify_chat_id=123")
            self.assertIsNone(job)

            saved = json.loads(state_file.read_text())
            self.assertEqual(saved["notify_chat_id"], 123)
            self.assertEqual(saved["openai_model"], "gpt-4.1")
            self.assertEqual(saved["oss_model"], "qwen2.5-coder:7b")

    def test_manual_proactive_pause_does_not_auto_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            bot._set_proactive_lane_pause(
                cfg,
                paused=True,
                reason="manual_pause_command",
                manual=True,
            )
            resumed = bot._maybe_resume_proactive_lane_for_manual_ceo_request(cfg)
            self.assertFalse(resumed)
            state = bot._proactive_lane_state(cfg)
            self.assertTrue(bool(state.get("paused", False)))
            self.assertTrue(bool(state.get("manual_pause", False)))
            self.assertEqual(str(state.get("reason") or ""), "manual_pause_command")

    def test_stop_all_proactive_pause_can_auto_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            bot._set_proactive_lane_pause(
                cfg,
                paused=True,
                reason="ceo_stop_all",
                manual=False,
            )
            resumed = bot._maybe_resume_proactive_lane_for_manual_ceo_request(cfg)
            self.assertTrue(resumed)
            state = bot._proactive_lane_state(cfg)
            self.assertFalse(bool(state.get("paused", False)))
            self.assertFalse(bool(state.get("manual_pause", False)))

    def test_factory_soft_pause_auto_resumes_when_due(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            bot._set_factory_pause(
                cfg,
                hard_stop=False,
                reason="factory_soft_pause",
                ttl_seconds=60,
            )
            state_before = bot._factory_state(cfg)
            self.assertTrue(bool(state_before.get("soft_pause_active", False)))

            def _m(st: dict[str, object]) -> None:
                st["factory_soft_pause_until"] = 0.0

            bot._update_state(cfg, _m)
            resumed = bot._factory_auto_resume_if_due(cfg)
            self.assertTrue(resumed)
            state_after = bot._factory_state(cfg)
            self.assertFalse(bool(state_after.get("paused", False)))

    def test_discover_factory_repos_scans_git_dirs_under_configured_roots(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_a = root / "repo-a"
            repo_b = root / "nested" / "repo-b"
            (repo_a / ".git").mkdir(parents=True)
            (repo_b / ".git").mkdir(parents=True)
            (root / "node_modules" / "skip-me" / ".git").mkdir(parents=True)
            cfg = self._cfg(root / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "codex_workdir": repo_a})

            with patch.dict(os.environ, {"BOT_FACTORY_REPO_ROOTS": str(root)}):
                repos = bot._discover_factory_repos(cfg)

            repo_paths = {str(item.get("path") or "") for item in repos}
            self.assertIn(str(repo_a.resolve()), repo_paths)
            self.assertIn(str(repo_b.resolve()), repo_paths)
            self.assertFalse(any("node_modules" in path for path in repo_paths))

    def test_sync_github_pat_git_credentials_writes_store_and_global_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            home.mkdir(parents=True, exist_ok=True)

            with patch.dict(os.environ, {"HOME": str(home), "GITHUB_TOKEN": "ghp_test_token_123"}, clear=False):
                issues = bot._sync_github_pat_git_credentials()

            self.assertEqual(issues, [])
            cred_path = (home / ".config" / "omnicrew" / "git-credentials").resolve()
            self.assertTrue(cred_path.exists())
            cred_text = cred_path.read_text(encoding="utf-8")
            self.assertIn("https://x-access-token:", cred_text)
            self.assertIn("@github.com", cred_text)
            self.assertEqual(cred_path.stat().st_mode & 0o777, 0o600)

            git_env = {**dict(os.environ), "HOME": str(home)}
            helper = subprocess.run(
                ["git", "config", "--global", "--get", "credential.helper"],
                check=True,
                capture_output=True,
                text=True,
                env=git_env,
            )
            self.assertEqual(helper.stdout.strip(), f"store --file {cred_path}")

            use_http_path = subprocess.run(
                ["git", "config", "--global", "--get", "credential.useHttpPath"],
                check=True,
                capture_output=True,
                text=True,
                env=git_env,
            )
            self.assertEqual(use_http_path.stdout.strip().lower(), "false")

            filled = subprocess.run(
                ["git", "credential", "fill"],
                input="protocol=https\nhost=github.com\npath=manolosake/PonceBot.git\n\n",
                check=True,
                capture_output=True,
                text=True,
                env=git_env,
            )
            self.assertIn("username=x-access-token", filled.stdout)
            self.assertIn("password=ghp_test_token_123", filled.stdout)

            rewrites = subprocess.run(
                ["git", "config", "--global", "--get-all", "url.https://github.com/.insteadOf"],
                check=True,
                capture_output=True,
                text=True,
                env=git_env,
            )
            rewrite_values = {line.strip() for line in rewrites.stdout.splitlines() if line.strip()}
            self.assertIn("ssh://git@github.com/", rewrite_values)
            self.assertIn("git@github.com:", rewrite_values)

    def test_orchestrator_threaded_session_enabled_defaults_true(self) -> None:
        self.assertTrue(bot._orchestrator_threaded_session_enabled({}, role="architect_local"))

    def test_orchestrator_threaded_session_enabled_respects_profile_override(self) -> None:
        self.assertFalse(bot._orchestrator_threaded_session_enabled({"threaded_session": False}, role="architect_local"))
        self.assertFalse(bot._orchestrator_threaded_session_enabled({"threaded_session": "off"}, role="reviewer_local"))
        self.assertTrue(bot._orchestrator_threaded_session_enabled({"threaded_session": "on"}, role="implementer_local"))

    def test_task_requires_fresh_threaded_session_for_autonomous_skynet_controller_jobs(self) -> None:
        autopilot = SimpleNamespace(role="skynet", is_autonomous=True, labels={"kind": "autopilot"}, trace={})
        final_sweep = SimpleNamespace(role="skynet", is_autonomous=True, labels={"kind": "final_sweep"}, trace={})
        proactive_trace = SimpleNamespace(role="skynet", is_autonomous=True, labels={}, trace={"proactive_lane": True})
        architect = SimpleNamespace(role="architect_local", is_autonomous=True, labels={"kind": "autopilot"}, trace={})
        self.assertTrue(bot._task_requires_fresh_threaded_session(autopilot, role="skynet"))
        self.assertTrue(bot._task_requires_fresh_threaded_session(final_sweep, role="skynet"))
        self.assertTrue(bot._task_requires_fresh_threaded_session(proactive_trace, role="skynet"))
        self.assertFalse(bot._task_requires_fresh_threaded_session(architect, role="architect_local"))

    def test_only_controller_roles_are_forced_read_only(self) -> None:
        self.assertTrue(bot._role_requires_enforced_read_only("skynet"))
        self.assertTrue(bot._role_requires_enforced_read_only("jarvis"))
        self.assertFalse(bot._role_requires_enforced_read_only("architect_local"))
        self.assertFalse(bot._role_requires_enforced_read_only("implementer_local"))
        self.assertFalse(bot._role_requires_enforced_read_only("reviewer_local"))

    def test_jsonl_stream_has_terminal_completion_detects_response_completed(self) -> None:
        payload = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "abc"}),
                json.dumps({"type": "response.completed", "text": "done"}),
            ]
        )
        self.assertTrue(bot._jsonl_stream_has_terminal_completion(payload))
        self.assertFalse(bot._jsonl_stream_has_terminal_completion(json.dumps({"type": "thread.started"})))

    def test_should_salvage_local_codex_exit_only_for_local_roles_with_terminal_jsonl(self) -> None:
        payload = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "abc"}),
                json.dumps({"type": "response.completed", "text": "READY"}),
            ]
        )
        self.assertTrue(
            bot._should_salvage_local_codex_exit(
                role="reviewer_local",
                code=1,
                body="READY",
                stdout_text=payload,
            )
        )
        self.assertFalse(
            bot._should_salvage_local_codex_exit(
                role="skynet",
                code=1,
                body="READY",
                stdout_text=payload,
            )
        )
        self.assertFalse(
            bot._should_salvage_local_codex_exit(
                role="reviewer_local",
                code=1,
                body="",
                stdout_text=payload,
            )
        )
        self.assertFalse(
            bot._should_salvage_local_codex_exit(
                role="reviewer_local",
                code=1,
                body="READY",
                stdout_text=json.dumps({"type": "thread.started"}),
            )
        )


    def test_local_slice_transitions_allow_planned_to_reviewer_ready(self) -> None:
        self.assertIn("reviewed_ready", bot._LOCAL_SLICE_ALLOWED_TRANSITIONS["planned"])


    def test_response_signals_no_code_change_accepts_additional_change_wording(self) -> None:
        self.assertTrue(bot._response_signals_no_code_change("READY. No additional code change is required."))

    def test_orchestrator_session_thread_id_prefers_repo_runtime_state(self) -> None:
        class _FakeQueue:
            def get_agent_runtime_state(self, *, repo_id: str, role: str) -> dict[str, object] | None:
                if repo_id == "repo-1" and role == "skynet":
                    return {"session_thread_id": "repo-thread"}
                return None

            def get_agent_thread(self, *, chat_id: int, role: str) -> str | None:
                return "global-thread"

        self.assertEqual(
            bot._orchestrator_session_thread_id(
                orch_q=_FakeQueue(),
                chat_id=1,
                role="skynet",
                repo_id="repo-1",
            ),
            "repo-thread",
        )
        self.assertEqual(
            bot._orchestrator_session_thread_id(
                orch_q=_FakeQueue(),
                chat_id=1,
                role="skynet",
                repo_id="",
            ),
            "global-thread",
        )

    def test_factory_touch_runtime_agents_preserves_repo_session_thread(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            storage = bot.SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = bot.OrchestratorQueue(storage=storage, role_profiles=None)
            q.upsert_agent_runtime_state(
                repo_id="repo-1",
                role="skynet",
                chat_id=1,
                session_thread_id="repo-thread",
                lane="factory",
                metadata={"repo_path": "/tmp/repo-1"},
            )
            q.set_agent_thread(chat_id=1, role="skynet", thread_id="global-thread")
            bot._factory_touch_runtime_agents(
                cfg=self._cfg(td_path / "state.json"),
                orch_q=q,
                chat_id=1,
                now=123.0,
                repos=[{
                    "repo_id": "repo-1",
                    "path": "/tmp/repo-1",
                    "default_branch": "main",
                    "autonomy_enabled": True,
                    "status": "active",
                }],
            )
            runtime = q.get_agent_runtime_state(repo_id="repo-1", role="skynet")
            self.assertEqual(str((runtime or {}).get("session_thread_id") or ""), "repo-thread")

    def test_task_counts_as_order_phase_blocker_ignores_salvageable_failed_reviewer(self) -> None:
        task = SimpleNamespace(
            role="reviewer_local",
            state="failed",
            labels={"key": "local_review_guard_slice1"},
            trace={
                "result_summary": "READY.\n\nLooks good.",
                "review_ready": True,
                "structured_digest": {"summary": "READY: patch is safe to merge."},
            },
            updated_at=20.0,
            created_at=10.0,
        )
        self.assertFalse(bot._task_counts_as_order_phase_blocker(task))

    def test_task_counts_as_order_phase_blocker_ignores_failed_local_job_superseded_by_newer_done(self) -> None:
        failed = SimpleNamespace(
            role="reviewer_local",
            state="failed",
            labels={"key": "local_review_guard_slice1"},
            trace={"result_summary": "NEEDS_REWORK."},
            updated_at=10.0,
            created_at=9.0,
        )
        newer_done = SimpleNamespace(
            role="reviewer_local",
            state="done",
            labels={"key": "local_review_guard_slice1"},
            trace={"result_summary": "READY."},
            updated_at=20.0,
            created_at=19.0,
        )
        latest = bot._latest_local_done_at_by_identity([failed, newer_done])
        self.assertFalse(
            bot._task_counts_as_order_phase_blocker(
                failed,
                latest_local_done_at_by_identity=latest,
            )
        )

    def test_controller_verified_slice_for_closure_accepts_no_change_evidence(self) -> None:
        impl = SimpleNamespace(
            role="implementer_local",
            state="done",
            labels={"key": "local_impl_guard_slice1"},
            trace={
                "slice_id": "slice1",
                "slice_no_code_change": True,
                "local_patch_info": {"no_code_change_required": True},
            },
            updated_at=10.0,
            created_at=9.0,
        )
        reviewer = SimpleNamespace(
            role="reviewer_local",
            state="done",
            labels={"key": "local_review_guard_slice1"},
            trace={
                "slice_id": "slice1",
                "result_summary": "READY. Existing implementation is correct.",
            },
            updated_at=20.0,
            created_at=19.0,
        )

        class _FakeQueue:
            def jobs_by_parent(self, *, parent_job_id: str, limit: int = 700) -> list[object]:
                return [impl, reviewer]

        self.assertEqual(
            bot._controller_verified_slice_for_closure(
                orch_q=_FakeQueue(),
                root_ticket="ticket-1",
                trace={"slice_id": "slice1"},
            ),
            "slice1",
        )

    def test_order_has_verified_no_change_resolution_accepts_salvageable_failed_reviewer(self) -> None:
        reviewer = SimpleNamespace(
            role="reviewer_local",
            state="failed",
            labels={"key": "local_review_guard_slice1"},
            trace={
                "slice_id": "slice1",
                "result_summary": "READY. No code change is required and the implementation is correct.",
            },
            updated_at=20.0,
            created_at=19.0,
        )

        class _FakeQueue:
            def jobs_by_parent(self, *, parent_job_id: str, limit: int = 600) -> list[object]:
                return [reviewer]

        self.assertTrue(
            bot._order_has_verified_no_change_resolution(
                orch_q=_FakeQueue(),
                root_ticket="ticket-1",
                now=30.0,
            )
        )

    def test_collect_order_local_autonomy_funnel_counts_no_change_slice_as_closed(self) -> None:
        impl = SimpleNamespace(
            role="implementer_local",
            state="done",
            labels={"key": "local_impl_guard_slice1"},
            trace={
                "slice_id": "slice1",
                "slice_no_code_change": True,
                "local_patch_info": {"no_code_change_required": True},
            },
            updated_at=10.0,
            created_at=9.0,
        )
        reviewer = SimpleNamespace(
            role="reviewer_local",
            state="done",
            labels={"key": "local_review_guard_slice1"},
            trace={
                "slice_id": "slice1",
                "result_summary": "READY. No additional code change is required.",
            },
            updated_at=20.0,
            created_at=19.0,
        )
        controller = SimpleNamespace(
            role="skynet",
            state="done",
            labels={},
            trace={
                "slice_id": "slice1",
                "result_summary": "PASS: VERIFIED_IMPROVEMENT",
                "improvement_verified": True,
            },
            updated_at=30.0,
            created_at=29.0,
        )

        class _FakeQueue:
            def jobs_by_parent(self, *, parent_job_id: str, limit: int = 800) -> list[object]:
                return [impl, reviewer, controller]

        funnel = bot._collect_order_local_autonomy_funnel(
            orch_q=_FakeQueue(),
            root_ticket="ticket-1",
            now=40.0,
        )
        self.assertEqual(funnel["slices_validated"], 1)
        self.assertEqual(funnel["slices_closed"], 1)
        self.assertTrue(funnel["improvement_verified"])

    def test_controller_verified_slice_for_closure_accepts_salvageable_failed_reviewer(self) -> None:
        impl = SimpleNamespace(
            role="implementer_local",
            state="done",
            labels={"key": "local_impl_guard_slice1"},
            trace={
                "slice_id": "slice1",
                "slice_no_code_change": True,
                "local_patch_info": {"no_code_change_required": True},
            },
            updated_at=10.0,
            created_at=9.0,
        )
        reviewer = SimpleNamespace(
            role="reviewer_local",
            state="failed",
            labels={"key": "local_review_guard_slice1"},
            trace={
                "slice_id": "slice1",
                "result_summary": "READY. Existing implementation is correct.",
            },
            updated_at=20.0,
            created_at=19.0,
        )

        class _FakeQueue:
            def jobs_by_parent(self, *, parent_job_id: str, limit: int = 700) -> list[object]:
                return [impl, reviewer]

        self.assertEqual(
            bot._controller_verified_slice_for_closure(
                orch_q=_FakeQueue(),
                root_ticket="ticket-1",
                trace={"slice_id": "slice1"},
            ),
            "slice1",
        )

    def test_controller_verified_slice_for_closure_accepts_backend_qa_order_branch_evidence_without_slice_id(self) -> None:
        branch_bucket = bot._sanitize_slice_token(
            "feature/order-ticket-1-bounded-reliability-fix",
            fallback="",
        )
        backend = SimpleNamespace(
            role="backend",
            state="done",
            labels={"key": "auto_backend_fix1"},
            trace={
                "order_branch": "feature/order-ticket-1-bounded-reliability-fix",
                "result_summary": "Implemented one bounded reliability fix with attached diff and validation logs.",
                "result_artifacts": ["/tmp/backend-artifact.txt"],
            },
            updated_at=10.0,
            created_at=9.0,
        )
        qa = SimpleNamespace(
            role="qa",
            state="done",
            labels={"key": "auto_qa_fix1"},
            trace={
                "order_branch": "feature/order-ticket-1-bounded-reliability-fix",
                "result_summary": "PASS / GO. Validation succeeded with attached evidence bundle.",
                "result_artifacts": ["/tmp/qa-artifact.txt"],
            },
            updated_at=20.0,
            created_at=19.0,
        )

        class _FakeQueue:
            def jobs_by_parent(self, *, parent_job_id: str, limit: int = 700) -> list[object]:
                return [backend, qa]

        self.assertEqual(
            bot._controller_verified_slice_for_closure(
                orch_q=_FakeQueue(),
                root_ticket="ticket-1",
                trace={"slice_id": "unmatched_local_slice"},
            ),
            branch_bucket,
        )


    def test_controller_local_recovery_specs_synthesizes_local_patch_flow_from_write_policy(self) -> None:
        specs = bot._controller_local_recovery_specs(
            {
                "summary": "Prepared a minimal reliability improvement in test_status_http.py.",
                "order_branch": "feature/test-order",
                "write_policy_violation": {
                    "changed_paths": ["test_status_http.py"],
                },
            }
        )
        self.assertEqual([spec.role for spec in specs], ["implementer_local", "reviewer_local"])
        self.assertIn("test_status_http.py", specs[0].text)
        self.assertIn("python3 -m unittest -q test_status_http", specs[0].text)
        self.assertEqual(specs[1].depends_on, [specs[0].key])


class TestParseJob(unittest.TestCase):
    def _cfg(self, state_file: Path) -> bot.BotConfig:
        return TestStateHandling()._cfg(state_file)

    def test_unknown_slash_command_shows_help(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/foo bar")
            resp, job = bot._parse_job(cfg, msg)
            self.assertIn("codexbot commands:", resp)
            self.assertIsNone(job)

    def test_unknown_slash_command_falls_back_to_exec_in_unsafe_direct_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "unsafe_direct_codex": True})
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/ls -la")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "")
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(job.argv, ["exec", "ls -la"])

    def test_cancel_is_special(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/cancel")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "__cancel__")
            self.assertIsNone(job)

    def test_plain_greeting_routes_to_codex(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="Hola")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "")
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(job.argv, ["exec", "Hola"])


    def test_ack_routes_to_codex(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="Ok")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "")
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(job.argv, ["exec", "Ok"])

    def test_dashboard_all_returns_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/dashboard all")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "__orch_dashboard:all")
            self.assertIsNone(job)

    def test_approve_merge_returns_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/approve_merge abc12345")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "__orch_approve_merge:abc12345")
            self.assertIsNone(job)

    def test_proposal_returns_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/proposal approve abc12345")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "__orch_proposal:approve abc12345")
            self.assertIsNone(job)

    def test_rollback_returns_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/rollback abc12345")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "__orch_rollback_merge:abc12345")
            self.assertIsNone(job)

    def test_orch_marker_no_payload_is_parsed(self) -> None:
        m = bot._orch_marker("agents")
        parsed = bot._parse_orchestrator_marker(m)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        kind, payload = parsed
        self.assertEqual(kind, "agents")
        self.assertEqual(payload, "")


class TestQAEvidenceState(unittest.TestCase):
    def test_artifact_id_validation(self) -> None:
        self.assertTrue(bot._qa_is_safe_artifact_id("48b12907-bd85-4ce8-88af-2982a78ebcfc"))
        self.assertTrue(bot._qa_is_safe_artifact_id("abc_123-XYZ"))
        self.assertFalse(bot._qa_is_safe_artifact_id(""))
        self.assertFalse(bot._qa_is_safe_artifact_id("../x"))
        self.assertFalse(bot._qa_is_safe_artifact_id("a/b"))
        self.assertFalse(bot._qa_is_safe_artifact_id("a\\b"))

    def test_evidence_artifact_id_persists_per_chat(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            cfg = TestStateHandling()._cfg(state_file)
            cfg = bot.BotConfig(**{**cfg.__dict__, "artifacts_root": Path(td) / "artifacts"})

            self.assertEqual(bot._qa_get_evidence_artifact_id(cfg, chat_id=123), "")
            self.assertIsNone(bot._qa_evidence_dir(cfg, chat_id=123))

            bot._qa_set_evidence_artifact_id(cfg, chat_id=123, artifact_id="abc-123")
            self.assertEqual(bot._qa_get_evidence_artifact_id(cfg, chat_id=123), "abc-123")
            self.assertEqual(bot._qa_evidence_dir(cfg, chat_id=123), (cfg.artifacts_root / "abc-123").resolve())

            bot._qa_set_evidence_artifact_id(cfg, chat_id=123, artifact_id="")
            self.assertEqual(bot._qa_get_evidence_artifact_id(cfg, chat_id=123), "")
            self.assertIsNone(bot._qa_evidence_dir(cfg, chat_id=123))


class TestSkillsCommands(unittest.TestCase):
    def _cfg(self, state_file: Path) -> bot.BotConfig:
        return TestStateHandling()._cfg(state_file)

    def test_skills_lists_enabled_disabled_and_system(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            codex_home = home / ".codex"
            skills_root = codex_home / "skills"
            (skills_root / "foo").mkdir(parents=True)
            (skills_root / "foo" / "SKILL.md").write_text("name: foo\n", encoding="utf-8")
            (skills_root / ".disabled" / "bar").mkdir(parents=True)
            (skills_root / ".disabled" / "bar" / "SKILL.md").write_text("name: bar\n", encoding="utf-8")
            (skills_root / ".system" / "skill-installer").mkdir(parents=True)
            (skills_root / ".system" / "skill-installer" / "SKILL.md").write_text("name: skill-installer\n", encoding="utf-8")

            with patch.dict(os.environ, {"HOME": str(home), "CODEX_HOME": str(codex_home)}):
                cfg = self._cfg(home / "state.json")
                msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/skills")
                resp, job = bot._parse_job(cfg, msg)
                self.assertIsNone(job)
                self.assertIn("Enabled:", resp)
                self.assertIn("- foo", resp)
                self.assertIn("Disabled:", resp)
                self.assertIn("- bar", resp)
                self.assertIn("System:", resp)
                self.assertIn(".system/skill-installer", resp)

    def test_skills_enable_disable_moves_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            codex_home = home / ".codex"
            skills_root = codex_home / "skills"
            (skills_root / "foo").mkdir(parents=True)
            (skills_root / "foo" / "SKILL.md").write_text("name: foo\n", encoding="utf-8")

            with patch.dict(os.environ, {"HOME": str(home), "CODEX_HOME": str(codex_home)}):
                cfg = self._cfg(home / "state.json")
                msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/skills disable foo")
                resp, job = bot._parse_job(cfg, msg)
                self.assertIsNone(job)
                self.assertEqual(resp, "OK. disable foo")
                self.assertFalse((skills_root / "foo").exists())
                self.assertTrue((skills_root / ".disabled" / "foo").exists())

                msg2 = bot.IncomingMessage(update_id=2, chat_id=1, user_id=2, message_id=11, username="u", text="/skills enable foo")
                resp2, job2 = bot._parse_job(cfg, msg2)
                self.assertIsNone(job2)
                self.assertEqual(resp2, "OK. enable foo")
                self.assertTrue((skills_root / "foo").exists())
                self.assertFalse((skills_root / ".disabled" / "foo").exists())

    def test_skills_install_returns_internal_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            codex_home = home / ".codex"
            with patch.dict(os.environ, {"HOME": str(home), "CODEX_HOME": str(codex_home)}):
                cfg = self._cfg(home / "state.json")
                msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/skills install imagegen")
                resp, job = bot._parse_job(cfg, msg)
                self.assertEqual(resp, "")
                self.assertIsNotNone(job)
                assert job is not None
                self.assertEqual(job.argv, ["__skills__", "install", "curated", "imagegen"])


    def test_restart_is_special(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/restart")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "__restart__")
            self.assertIsNone(job)

    def test_exec_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/exec --help")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "")
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(job.argv, ["exec", "--help"])

    def test_alias_m_maps_to_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/m")
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("Usage:", resp)
            self.assertIn("/model", resp)


class TestPromptConstruction(unittest.TestCase):
    def test_2000_chars_reach_codex_prompt_without_truncation(self) -> None:
        # Include ':' and newlines to catch accidental split/regex truncation.
        prefix = "Riesgo: A:B:C\nDetalle: "
        text = prefix + ("x" * (2000 - len(prefix)))
        self.assertEqual(len(text), 2000)

        with tempfile.TemporaryDirectory() as td:
            cfg = TestStateHandling()._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text=text)
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "")
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(len(job.user_text), 2000)

            task = bot._orchestrator_task_from_job(cfg, job, profiles=None, user_id=2)
            prompt = bot.build_agent_prompt(task, profile={})
            self.assertIn(text, prompt)

    def test_parse_employee_forward_is_robust_to_newlines_and_extra_colons(self) -> None:
        raw = "Empleado: Juan Perez: Primer: linea\nSegunda: linea\nFin"
        name, msg = bot._parse_employee_forward(raw)
        self.assertEqual(name, "Juan Perez")
        self.assertEqual(msg, "Primer: linea\nSegunda: linea\nFin")

    def test_architect_prompt_uses_controller_workspace_snapshot_as_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            (repo / "orchestrator").mkdir(parents=True, exist_ok=True)
            (repo / "tools").mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "bot.py").write_text("print('hi')\n", encoding="utf-8")
            (repo / "orchestrator" / "agents.yaml").write_text("roles: []\n", encoding="utf-8")
            (repo / "tools" / "proactive_health_report.py").write_text("print('ok')\n", encoding="utf-8")
            subprocess.run(["git", "add", "bot.py", "orchestrator/agents.yaml", "tools/proactive_health_report.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            task = SimpleNamespace(input_text="codexbot reliability delivery", trace={})
            prompt = bot._augment_local_specialist_prompt_with_workspace_context(
                task=task,
                user_prompt="LOCAL_WORKSPACE_RULES:\n- base\n",
                worktree_dir=repo,
                role="architect_local",
            )

            self.assertIn("authoritative snapshots", prompt)
            self.assertIn("Workspace inventory", prompt)
            self.assertIn("orchestrator/agents.yaml", prompt)

    def test_build_agent_prompt_marks_allowed_tools_as_advisory(self) -> None:
        task = SimpleNamespace(
            role="architect_local",
            request_type="maintenance",
            mode_hint="ro",
            artifacts_dir="/tmp/artifacts",
            trace={},
            input_text="do work",
            parent_job_id="",
            is_autonomous=True,
        )
        prompt = bot.build_agent_prompt(task, profile={"allowed_tools": ["plan", "repo_read"]})
        self.assertIn("ALLOWED_TOOLS: plan, repo_read (advisory focus; controller-provided repo context remains allowed)", prompt)

    def test_build_local_specialist_user_prompt_adds_workspace_snapshot_for_codex_roles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            (repo / "orchestrator").mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "bot.py").write_text("print('hi')\n", encoding="utf-8")
            (repo / "orchestrator" / "agents.yaml").write_text("roles: []\n", encoding="utf-8")
            subprocess.run(["git", "add", "bot.py", "orchestrator/agents.yaml"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            task = SimpleNamespace(
                role="architect_local",
                request_type="maintenance",
                mode_hint="ro",
                artifacts_dir="/tmp/artifacts",
                trace={},
                input_text="codexbot reliability delivery",
                parent_job_id="759a2373-00a8-4fd1-8763-1ef40db8ed1d",
                is_autonomous=True,
            )
            prompt = bot._build_local_specialist_user_prompt(
                task=task,
                role_profile={"allowed_tools": ["plan", "repo_read"]},
                role="architect_local",
                mode="ro",
                worktree_dir=repo,
            )

            self.assertIn("LOCAL_WORKSPACE_RULES", prompt)
            self.assertIn("Controller note: the Workspace inventory and FILE excerpts below are authoritative snapshots", prompt)
            self.assertIn("Workspace inventory", prompt)

    def test_build_local_specialist_user_prompt_for_implementer_uses_controller_apply_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "bot.py").write_text("print('hi')\n", encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            task = SimpleNamespace(
                role="implementer_local",
                request_type="task",
                mode_hint="ro",
                artifacts_dir="/tmp/artifacts",
                trace={},
                input_text="Implement exactly one bounded improvement in bot.py",
                parent_job_id="759a2373-00a8-4fd1-8763-1ef40db8ed1d",
                is_autonomous=True,
            )
            prompt = bot._build_local_specialist_user_prompt(
                task=task,
                role_profile={"allowed_tools": ["repo_read"], "execution_backend": "codex"},
                role="implementer_local",
                mode="ro",
                worktree_dir=repo,
            )

            self.assertIn("IMPLEMENTER_LOCAL_STRICT_OVERRIDE", prompt)
            self.assertIn("Do not use exec_command, apply_patch, shell tools", prompt)
            self.assertIn("Read-only workspace access is expected for this role", prompt)
            self.assertIn("The controller will apply your diff/rewrite", prompt)
            self.assertIn("EXPECTED_VALIDATION", prompt)
            self.assertIn("NO CODE CHANGE REQUIRED", prompt)

    def test_build_local_specialist_user_prompt_for_architect_large_exact_target_includes_focused_symbol_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            body = "".join(f"pad_{i} = 0\n" for i in range(7000)) + "\n_ORCHESTRATOR_ROLES = (\n    'jarvis',\n    'skynet',\n)\n"
            (repo / "bot.py").write_text(body, encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            task = SimpleNamespace(
                role="architect_local",
                request_type="task",
                mode_hint="ro",
                artifacts_dir="/tmp/artifacts",
                trace={},
                input_text=(
                    "FILES:\n- bot.py\n\nCHANGE:\n- Update `_ORCHESTRATOR_ROLES` to include `architect_local`, `implementer_local`, and `reviewer_local`.\n\nVALIDATION:\n- python3 -m py_compile bot.py\n\nRISK:\n- low\n"
                ),
                parent_job_id="759a2373-00a8-4fd1-8763-1ef40db8ed1d",
                is_autonomous=True,
            )
            prompt = bot._build_local_specialist_user_prompt(
                task=task,
                role_profile={"allowed_tools": ["repo_read"], "execution_backend": "codex"},
                role="architect_local",
                mode="ro",
                worktree_dir=repo,
            )

            self.assertIn("_ORCHESTRATOR_ROLES = (", prompt)
            self.assertIn("FOCUSED_EXACT_TARGET_CONTEXT bot.py", prompt)

    def test_build_local_specialist_user_prompt_for_large_exact_target_includes_focused_symbol_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            body = "".join(f"pad_{i} = 0\n" for i in range(7000)) + "\n_ORCHESTRATOR_ROLES = (\n    'jarvis',\n    'skynet',\n)\n"
            (repo / "bot.py").write_text(body, encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            task = SimpleNamespace(
                role="implementer_local",
                request_type="task",
                mode_hint="ro",
                artifacts_dir="/tmp/artifacts",
                trace={},
                input_text="Implement exactly one bounded improvement in bot.py",
                parent_job_id="759a2373-00a8-4fd1-8763-1ef40db8ed1d",
                is_autonomous=True,
            )
            architect_handoff = (
                "FILES:\n- bot.py\n\nCHANGE:\n- Update `_ORCHESTRATOR_ROLES` to include `architect_local`, `implementer_local`, and `reviewer_local`.\n\nVALIDATION:\n- python3 -m py_compile bot.py\n\nRISK:\n- low\n"
            )
            with patch.object(bot, "_latest_local_specialist_response", return_value=architect_handoff):
                prompt = bot._build_local_specialist_user_prompt(
                    task=task,
                    role_profile={"allowed_tools": ["repo_read"], "execution_backend": "codex"},
                    role="implementer_local",
                    mode="ro",
                    worktree_dir=repo,
                )

            self.assertIn("_ORCHESTRATOR_ROLES = (", prompt)
            self.assertIn("FOCUSED_EXACT_TARGET_CONTEXT bot.py", prompt)

    def test_build_local_specialist_user_prompt_large_exact_target_truncation_includes_tail_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            body = "".join(f"pad_{i} = 0\n" for i in range(9000)) + "\n_ORCHESTRATOR_ROLES = (\n    'jarvis',\n    'skynet',\n)\n"
            (repo / "bot.py").write_text(body, encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            task = SimpleNamespace(
                role="implementer_local",
                request_type="task",
                mode_hint="ro",
                artifacts_dir="/tmp/artifacts",
                trace={},
                input_text=(
                    "FILES:\n- bot.py\n\nCHANGE:\n- Adjust startup logging.\n\nVALIDATION:\n- python3 -m py_compile bot.py\n"
                ),
                parent_job_id="759a2373-00a8-4fd1-8763-1ef40db8ed1d",
                is_autonomous=True,
            )
            with patch.object(bot, "_latest_local_specialist_response", return_value=""):
                prompt = bot._build_local_specialist_user_prompt(
                    task=task,
                    role_profile={"allowed_tools": ["repo_read"], "execution_backend": "codex"},
                    role="implementer_local",
                    mode="ro",
                    worktree_dir=repo,
                )

            self.assertIn("_ORCHESTRATOR_ROLES = (", prompt)
            self.assertIn("# [TRUNCATED_EXACT_TARGET_CONTEXT_TAIL]", prompt)

    def test_build_local_specialist_user_prompt_for_recovery_implementer_does_not_mix_latest_architect_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "README.md").write_text("hello\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            task = SimpleNamespace(
                role="implementer_local",
                request_type="task",
                mode_hint="ro",
                artifacts_dir="/tmp/artifacts",
                trace={"delegated_key": "local_impl_recover_deadbeef"},
                labels={"key": "local_impl_recover_deadbeef"},
                input_text="Recover the bounded README.md change only.",
                parent_job_id="759a2373-00a8-4fd1-8763-1ef40db8ed1d",
                is_autonomous=True,
            )
            with patch.object(bot, "_latest_local_specialist_response", return_value="FILES:\n- orchestrator/queue.py\nCHANGE:\n- unrelated architect handoff"):
                prompt = bot._build_local_specialist_user_prompt(
                    task=task,
                    role_profile={"allowed_tools": ["repo_read"], "execution_backend": "codex"},
                    role="implementer_local",
                    mode="ro",
                    worktree_dir=repo,
                )

            self.assertNotIn("LATEST_ARCHITECT_LOCAL_OUTPUT", prompt)
            self.assertNotIn("unrelated architect handoff", prompt)
            self.assertIn("IMPLEMENTER_LOCAL_STRICT_OVERRIDE", prompt)

    def test_response_signals_repo_access_blocker_detects_known_patterns(self) -> None:
        self.assertTrue(bot._response_signals_repo_access_blocker("BLOCKER: sandbox exec denied; I can't access the repo filesystem"))
        self.assertTrue(bot._response_signals_repo_access_blocker("every command fails with bwrap: loopback: Failed RTM_NEWADDR"))
        self.assertFalse(bot._response_signals_repo_access_blocker("FILES:\n- bot.py\nCHANGE:\n- tweak timeout"))


class TestLocalSpecialistResponseHelpers(unittest.TestCase):
    def test_parse_orchestrator_subtasks_accepts_next_action_nested_subtasks(self) -> None:
        specs = bot.parse_orchestrator_subtasks(
            {
                "summary": "blocked for now",
                "next_action": {
                    "type": "LOCAL_REPLAN_REQUEST",
                    "subtasks": [
                        {
                            "role": "architect_local",
                            "task": "Define one bounded retry slice for bot.py",
                            "mode_hint": "ro",
                            "priority": 1,
                            "acceptance_criteria": ["Return exact files and one validation command."],
                            "definition_of_done": ["Actionable implementer handoff is ready."],
                        }
                    ],
                },
            }
        )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].role, "architect_local")
        self.assertTrue(specs[0].key.startswith("auto_architect_local_"))

    def test_result_structured_digest_reads_dict_payload_for_delegation(self) -> None:
        result = {
            "status": "ok",
            "summary": "done",
            "structured_digest": {
                "next_action": {
                    "type": "LOCAL_REPLAN_REQUEST",
                    "subtasks": [
                        {
                            "role": "reviewer_local",
                            "task": "Validate bot.py and report PASS/NO-GO.",
                        }
                    ],
                }
            },
        }

        structured = bot._result_structured_digest(result)
        specs = bot.parse_orchestrator_subtasks(structured)

        self.assertIsInstance(structured, dict)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].role, "reviewer_local")

    def test_helpers_read_result_summary_from_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            data_dir = repo_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "create table jobs ("
                    " job_id text primary key,"
                    " parent_job_id text,"
                    " role text,"
                    " state text,"
                    " updated_at real,"
                    " created_at real,"
                    " artifacts_dir text,"
                    " labels text,"
                    " trace text"
                    ")"
                )
                trace = json.dumps(
                    {
                        "result_summary": (
                            "FILES:\n"
                            "- tools/proactive_health_report.py\n"
                            "CHANGE:\n"
                            "- tighten proactive signal parsing\n"
                            "VALIDATION:\n"
                            "- python3 -m py_compile tools/proactive_health_report.py\n"
                            "RISK:\n"
                            "- low"
                        )
                    }
                )
                labels = json.dumps({"key": "local_arch_guard_slice1"})
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 10.0, 9.0, ?, ?, ?)",
                    (
                        "job-1",
                        "ticket-1",
                        "architect_local",
                        str(repo_root / "artifacts" / "job-1"),
                        labels,
                        trace,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            globals_map = bot._latest_local_specialist_response.__globals__
            original_file = globals_map.get("__file__")
            globals_map["__file__"] = str(repo_root / "bot.py")
            try:
                latest = bot._latest_local_specialist_response(
                    root_ticket="ticket-1",
                    role="architect_local",
                    max_chars=4000,
                )
                by_key = bot._local_specialist_response_for_key(
                    role="architect_local",
                    delegated_key="local_arch_guard_slice1",
                    max_chars=4000,
                )
            finally:
                if original_file is not None:
                    globals_map["__file__"] = original_file
                else:
                    globals_map.pop("__file__", None)

            self.assertIn("FILES:", latest)
            self.assertIn("tools/proactive_health_report.py", latest)
            self.assertEqual(latest, by_key)

    def test_autonomous_local_first_synthesizes_implementer_from_architect_handoff_when_specs_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            data_dir = repo_root / "data"
            repo_id = "codexbot-12345678"
            data_dir.mkdir(parents=True, exist_ok=True)
            cfg = bot.BotConfig(
                **{
                    **TestStateHandling()._cfg(repo_root / "state.json").__dict__,
                    "worktree_root": data_dir / "worktrees",
                }
            )
            repo_worktree_root = bot._repo_worktree_root(cfg, repo_id=repo_id)
            worktree_dir = repo_worktree_root / "implementer_local" / "slot1" / "tools"
            worktree_dir.mkdir(parents=True, exist_ok=True)
            (worktree_dir / "proactive_health_report.py").write_text("def is_proactive(text):\n    return False\n", encoding="utf-8")

            db_path = data_dir / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "create table jobs ("
                    " job_id text primary key,"
                    " parent_job_id text,"
                    " role text,"
                    " state text,"
                    " updated_at real,"
                    " created_at real,"
                    " artifacts_dir text,"
                    " labels text,"
                    " trace text"
                    ")"
                )
                handoff = (
                    "FILES:\n"
                    "- tools/proactive_health_report.py\n"
                    "CHANGE:\n"
                    "- expand proactive title matching\n"
                    "VALIDATION:\n"
                    "- python3 -m py_compile tools/proactive_health_report.py\n"
                    "RISK:\n"
                    "- low"
                )
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 50.0, 49.0, ?, ?, ?)",
                    (
                        "arch-job-1",
                        "ticket-1",
                        "architect_local",
                        str(repo_root / "artifacts" / "arch-job-1"),
                        json.dumps({"key": "local_arch_guard_slice1"}),
                        json.dumps({"result_summary": handoff}),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            class _FakeQueue:
                def jobs_by_parent(self, *, parent_job_id: str, limit: int = 2000) -> list[object]:
                    return []

                def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                    return {
                        "order_id": order_id,
                        "chat_id": 8355547734,
                        "title": "Proactive Sprint: codexbot Reliability + Delivery",
                        "body": f"Improve proactive health report reliability. [repo:{repo_id}]",
                    }

                def list_orders_global(self, status: str = "active", limit: int = 400) -> list[dict[str, object]]:
                    return []

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
                    if repo_id != "codexbot-12345678":
                        return None
                    return {
                        "repo_id": repo_id,
                        "path": str(repo_root / "repo"),
                        "default_branch": "main",
                        "autonomy_enabled": True,
                        "priority": 2,
                        "runtime_mode": "ceo-bounded",
                        "daily_budget": 0.0,
                        "status": "active",
                        "metadata": {},
                    }

                def get_job(self, job_id: str) -> object | None:
                    return SimpleNamespace(
                        trace={},
                        labels={},
                        parent_job_id="",
                        job_id=job_id,
                        chat_id=8355547734,
                    )

            globals_map = bot._apply_autonomous_local_first_policy.__globals__
            original_file = globals_map.get("__file__")
            globals_map["__file__"] = str(repo_root / "bot.py")
            try:
                with patch.dict(os.environ, {"BOT_AUTONOMOUS_LOCAL_FIRST": "1"}, clear=False):
                    specs = bot._apply_autonomous_local_first_policy(
                        cfg=cfg,
                        specs=[],
                        orch_q=_FakeQueue(),
                        root_ticket="ticket-1",
                        now=100.0,
                    )
            finally:
                if original_file is not None:
                    globals_map["__file__"] = original_file
                else:
                    globals_map.pop("__file__", None)

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].role, "implementer_local")
            self.assertEqual(specs[0].mode_hint, "ro")
            self.assertIn("PRIMARY_ARCHITECT_HANDOFF", specs[0].text)
            self.assertIn("tools/proactive_health_report.py", specs[0].text)
            self.assertIn("EXPECTED_VALIDATION", specs[0].text)

    def test_autonomous_local_first_synthesizes_implementer_rework_from_reviewer_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            data_dir = repo_root / "data"
            repo_id = "codexbot-12345678"
            data_dir.mkdir(parents=True, exist_ok=True)
            cfg = bot.BotConfig(
                **{
                    **TestStateHandling()._cfg(repo_root / "state.json").__dict__,
                    "worktree_root": data_dir / "worktrees",
                }
            )
            repo_worktree_root = bot._repo_worktree_root(cfg, repo_id=repo_id)
            worktree_dir = repo_worktree_root / "implementer_local" / "slot1" / "orchestrator"
            worktree_dir.mkdir(parents=True, exist_ok=True)
            (worktree_dir / "runbooks.py").write_text("def to_task(runbook, chat_id):\n    return None\n", encoding="utf-8")

            db_path = data_dir / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "create table jobs ("
                    " job_id text primary key,"
                    " parent_job_id text,"
                    " role text,"
                    " state text,"
                    " updated_at real,"
                    " created_at real,"
                    " artifacts_dir text,"
                    " labels text,"
                    " trace text"
                    ")"
                )
                impl_trace = {
                    "result_summary": (
                        "Bounded runbook patch prepared.\n"
                        "EXPECTED_VALIDATION: `python3 -m unittest -q test_orchestrator`"
                    ),
                    "slice_patch_applied": True,
                    "slice_validation_ok": True,
                    "patch_info": {
                        "changed_files": ["orchestrator/runbooks.py"],
                        "validation_ok": True,
                    },
                }
                review_trace = {
                    "result_summary": (
                        "Verdict: NEEDS_REWORK. Blocking issue: `Task.new(...)` does not accept `ticket`. "
                        "Remove the invalid kwarg and keep the slice bounded."
                    )
                }
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 90.0, 89.0, ?, ?, ?)",
                    (
                        "impl-job-1",
                        "ticket-1",
                        "implementer_local",
                        str(repo_root / "artifacts" / "impl-job-1"),
                        json.dumps({"key": "local_impl_guard_slice1"}),
                        json.dumps(impl_trace),
                    ),
                )
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 100.0, 99.0, ?, ?, ?)",
                    (
                        "review-job-1",
                        "ticket-1",
                        "reviewer_local",
                        str(repo_root / "artifacts" / "review-job-1"),
                        json.dumps({"key": "local_review_guard_slice1"}),
                        json.dumps(review_trace),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            class _FakeQueue:
                def jobs_by_parent(self, *, parent_job_id: str, limit: int = 2000) -> list[object]:
                    return []

                def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                    return {
                        "order_id": order_id,
                        "chat_id": 8355547734,
                        "title": "Proactive Sprint: codexbot Reliability + Delivery",
                        "body": f"Improve proactive local delegation. [repo:{repo_id}]",
                    }

                def list_orders_global(self, status: str = "active", limit: int = 400) -> list[dict[str, object]]:
                    return []

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
                    if repo_id != "codexbot-12345678":
                        return None
                    return {
                        "repo_id": repo_id,
                        "path": str(repo_root / "repo"),
                        "default_branch": "main",
                        "autonomy_enabled": True,
                        "priority": 2,
                        "runtime_mode": "ceo-bounded",
                        "daily_budget": 0.0,
                        "status": "active",
                        "metadata": {},
                    }

                def get_job(self, job_id: str) -> object | None:
                    return SimpleNamespace(
                        trace={},
                        labels={},
                        parent_job_id="",
                        job_id=job_id,
                        chat_id=8355547734,
                    )

            globals_map = bot._apply_autonomous_local_first_policy.__globals__
            original_file = globals_map.get("__file__")
            globals_map["__file__"] = str(repo_root / "bot.py")
            try:
                with patch.dict(os.environ, {"BOT_AUTONOMOUS_LOCAL_FIRST": "1"}, clear=False):
                    specs = bot._apply_autonomous_local_first_policy(
                        cfg=cfg,
                        specs=[],
                        orch_q=_FakeQueue(),
                        root_ticket="ticket-1",
                        now=120.0,
                    )
            finally:
                if original_file is not None:
                    globals_map["__file__"] = original_file
                else:
                    globals_map.pop("__file__", None)

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].role, "implementer_local")
            self.assertIn("LATEST_REVIEWER_FINDINGS", specs[0].text)
            self.assertIn("`Task.new(...)` does not accept `ticket`", specs[0].text)
            self.assertIn("orchestrator/runbooks.py", specs[0].text)
            self.assertIn("python3 -m unittest -q test_orchestrator", specs[0].text)

    def test_dedupe_specs_by_signature_allows_retry_after_recent_failed_local_guard(self) -> None:
        spec = bot.TaskSpec(
            key="local_impl_guard_retry_new",
            role="implementer_local",
            text="Implement one bounded retry slice in bot.py",
            mode_hint="ro",
            priority=1,
            depends_on=[],
            requires_approval=False,
            acceptance_criteria=["Return one diff."],
            definition_of_done=["Controller can apply retry patch."],
            eta_minutes=30,
            sla_tier="high",
        )
        failed = bot.Task.new(
            job_id="failed-impl",
            source="test",
            role="implementer_local",
            input_text="Older failed local guard slice",
            request_type="task",
            priority=1,
            model="gpt-5.3-codex",
            effort="medium",
            mode_hint="ro",
            requires_approval=False,
            max_cost_window_usd=0.0,
            chat_id=1,
            state="failed",
            labels={"key": "local_impl_guard_retry_old"},
            created_at=100.0,
        ).with_updates(updated_at=120.0)

        deduped = bot._dedupe_specs_by_signature(
            specs=[spec],
            existing_subtasks=[failed],
            now=180.0,
            recent_window_s=3600.0,
        )

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].key, "local_impl_guard_retry_new")

    def test_dedupe_specs_by_signature_keeps_recent_done_local_guard_suppressed(self) -> None:
        spec = bot.TaskSpec(
            key="local_impl_guard_retry_new",
            role="implementer_local",
            text="Implement one bounded retry slice in bot.py",
            mode_hint="ro",
            priority=1,
            depends_on=[],
            requires_approval=False,
            acceptance_criteria=["Return one diff."],
            definition_of_done=["Controller can apply retry patch."],
            eta_minutes=30,
            sla_tier="high",
        )
        done = bot.Task.new(
            job_id="done-impl",
            source="test",
            role="implementer_local",
            input_text="Older successful local guard slice",
            request_type="task",
            priority=1,
            model="gpt-5.3-codex",
            effort="medium",
            mode_hint="ro",
            requires_approval=False,
            max_cost_window_usd=0.0,
            chat_id=1,
            state="done",
            labels={"key": "local_impl_guard_retry_old"},
            created_at=100.0,
        ).with_updates(updated_at=120.0)

        deduped = bot._dedupe_specs_by_signature(
            specs=[spec],
            existing_subtasks=[done],
            now=180.0,
            recent_window_s=3600.0,
        )

        self.assertEqual(deduped, [])

    def test_autonomous_local_first_keeps_direct_implementer_recovery_when_architect_claims_no_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            data_dir = repo_root / "data"
            repo_id = "codexbot-12345678"
            data_dir.mkdir(parents=True, exist_ok=True)
            cfg = bot.BotConfig(
                **{
                    **TestStateHandling()._cfg(repo_root / "state.json").__dict__,
                    "worktree_root": data_dir / "worktrees",
                }
            )
            repo_worktree_root = bot._repo_worktree_root(cfg, repo_id=repo_id)
            worktree_dir = repo_worktree_root / "implementer_local" / "slot1"
            worktree_dir.mkdir(parents=True, exist_ok=True)
            (worktree_dir / "bot.py").write_text("VALUE = 1\n", encoding="utf-8")

            db_path = data_dir / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "create table jobs ("
                    " job_id text primary key,"
                    " parent_job_id text,"
                    " role text,"
                    " state text,"
                    " updated_at real,"
                    " created_at real,"
                    " artifacts_dir text,"
                    " labels text,"
                    " trace text"
                    ")"
                )
                arch_summary = (
                    "FILES:\n"
                    "- bot.py\n"
                    "CHANGE:\n"
                    "- no concrete code change to apply because behavior is already correct\n"
                    "VALIDATION:\n"
                    "- python3 -m py_compile bot.py\n"
                    "RISK:\n"
                    "- low\n"
                )
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 50.0, 49.0, ?, ?, ?)",
                    (
                        "arch-job-1",
                        "ticket-1",
                        "architect_local",
                        str(repo_root / "artifacts" / "arch-job-1"),
                        json.dumps({"key": "local_arch_guard_slice1"}),
                        json.dumps({"result_summary": arch_summary}),
                    ),
                )
                impl_summary = (
                    "BLOCKER: workspace is read-only, so I can't modify `bot.py` or apply the required diff. "
                    "Smallest next action: grant write permissions (RW) for `REAL_WORKTREE_DIR`."
                )
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'failed', 60.0, 59.0, ?, ?, ?)",
                    (
                        "impl-job-1",
                        "ticket-1",
                        "implementer_local",
                        str(repo_root / "artifacts" / "impl-job-1"),
                        json.dumps({"key": "local_impl_guard_slice1"}),
                        json.dumps({"result_summary": impl_summary, "slice_id": "slice1", "attempt_n": 1, "failure_class": "blocked"}),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            class _FakeQueue:
                def jobs_by_parent(self, *, parent_job_id: str, limit: int = 2000) -> list[object]:
                    return []

                def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                    return {
                        "order_id": order_id,
                        "chat_id": 8355547734,
                        "title": "Proactive Sprint: codexbot Reliability + Delivery",
                        "body": f"Improve reliability. [repo:{repo_id}]",
                    }

                def list_orders_global(self, status: str = "active", limit: int = 400) -> list[dict[str, object]]:
                    return []

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
                    if repo_id != "codexbot-12345678":
                        return None
                    return {
                        "repo_id": repo_id,
                        "path": str(repo_root / "repo"),
                        "default_branch": "main",
                        "autonomy_enabled": True,
                        "priority": 2,
                        "runtime_mode": "ceo-bounded",
                        "daily_budget": 0.0,
                        "status": "active",
                        "metadata": {},
                    }

                def get_job(self, job_id: str) -> object | None:
                    return SimpleNamespace(
                        trace={},
                        labels={},
                        parent_job_id="",
                        job_id=job_id,
                        chat_id=8355547734,
                    )

            globals_map = bot._apply_autonomous_local_first_policy.__globals__
            original_file = globals_map.get("__file__")
            globals_map["__file__"] = str(repo_root / "bot.py")
            try:
                specs = bot._apply_autonomous_local_first_policy(
                    cfg=cfg,
                    specs=[
                        bot.TaskSpec(
                            key="local_impl_guard_slice1_retry",
                            role="implementer_local",
                            text="Direct unblock implementation for ticket ticket-1 in bot.py",
                            mode_hint="ro",
                            priority=1,
                            depends_on=[],
                            requires_approval=False,
                            acceptance_criteria=["Return a concrete diff."],
                            definition_of_done=["Controller can apply the patch."],
                            eta_minutes=30,
                            sla_tier="high",
                        )
                    ],
                    orch_q=_FakeQueue(),
                    root_ticket="ticket-1",
                    now=100.0,
                )
            finally:
                if original_file is not None:
                    globals_map["__file__"] = original_file
                else:
                    globals_map.pop("__file__", None)

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].role, "implementer_local")

    def test_autonomous_local_first_redirects_validation_only_replan_to_reviewer_after_no_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            data_dir = repo_root / "data"
            repo_id = "codexbot-12345678"
            cfg = bot.BotConfig(
                **{
                    **TestStateHandling()._cfg(repo_root / "state.json").__dict__,
                    "worktree_root": data_dir / "worktrees",
                }
            )
            repo_worktree_root = bot._repo_worktree_root(cfg, repo_id=repo_id)
            worktree_dir = repo_worktree_root / "implementer_local" / "slot1"
            worktree_dir.mkdir(parents=True, exist_ok=True)
            (worktree_dir / "bot.py").write_text("VALUE = 1\n", encoding="utf-8")

            db_path = data_dir / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "create table jobs ("
                    " job_id text primary key,"
                    " parent_job_id text,"
                    " role text,"
                    " state text,"
                    " updated_at real,"
                    " created_at real,"
                    " artifacts_dir text,"
                    " labels text,"
                    " trace text"
                    ")"
                )
                arch_summary = (
                    "FILES:\n"
                    "- bot.py\n"
                    "CHANGE:\n"
                    "- remove redundant anchors from _SKILL_SEGMENT_RE because fullmatch already enforces full-string validation\n"
                    "VALIDATION:\n"
                    "- python3 -m py_compile bot.py\n"
                    "RISK:\n"
                    "- low\n"
                )
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 50.0, 49.0, ?, ?, ?)",
                    (
                        "arch-job-1",
                        "ticket-1",
                        "architect_local",
                        str(repo_root / "artifacts" / "arch-job-1"),
                        json.dumps({"key": "local_arch_guard_slice1"}),
                        json.dumps({"result_summary": arch_summary}),
                    ),
                )
                impl_trace = {
                    "result_summary": "No code change required because the current implementation already matches the requested fix.",
                    "slice_no_code_change": True,
                    "local_patch_info": {"no_code_change_required": True},
                }
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 60.0, 59.0, ?, ?, ?)",
                    (
                        "impl-job-1",
                        "ticket-1",
                        "implementer_local",
                        str(repo_root / "artifacts" / "impl-job-1"),
                        json.dumps({"key": "local_impl_guard_slice1"}),
                        json.dumps(impl_trace),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            class _FakeQueue:
                def jobs_by_parent(self, *, parent_job_id: str, limit: int = 2000) -> list[object]:
                    return []

                def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                    return {
                        "order_id": order_id,
                        "chat_id": 8355547734,
                        "title": "Proactive Sprint: codexbot Reliability + Delivery",
                        "body": f"Improve reliability. [repo:{repo_id}]",
                    }

                def list_orders_global(self, status: str = "active", limit: int = 400) -> list[dict[str, object]]:
                    return []

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
                    if repo_id != "codexbot-12345678":
                        return None
                    return {
                        "repo_id": repo_id,
                        "path": str(repo_root / "repo"),
                        "default_branch": "main",
                        "autonomy_enabled": True,
                        "priority": 2,
                        "runtime_mode": "ceo-bounded",
                        "daily_budget": 0.0,
                        "status": "active",
                        "metadata": {},
                    }

                def get_job(self, job_id: str) -> object | None:
                    return SimpleNamespace(
                        trace={},
                        labels={},
                        parent_job_id="",
                        job_id=job_id,
                        chat_id=8355547734,
                    )

            globals_map = bot._apply_autonomous_local_first_policy.__globals__
            original_file = globals_map.get("__file__")
            globals_map["__file__"] = str(repo_root / "bot.py")
            try:
                specs = bot._apply_autonomous_local_first_policy(
                    cfg=cfg,
                    specs=[
                        bot.TaskSpec(
                            key="auto_architect_local_slice1",
                            role="architect_local",
                            text="Define a minimal verification checklist for the current slice.",
                            mode_hint="ro",
                            priority=2,
                            depends_on=[],
                            requires_approval=False,
                            acceptance_criteria=["Return one checklist."],
                            definition_of_done=["Checklist returned."],
                            eta_minutes=5,
                            sla_tier="normal",
                        ),
                        bot.TaskSpec(
                            key="auto_implementer_local_slice1",
                            role="implementer_local",
                            text=(
                                "Run verify-only slice: capture line evidence for _skill_segment_ok and execute one targeted parser test; no code changes.\n\n"
                                "PRIMARY_ARCHITECT_HANDOFF:\n"
                                f"{arch_summary}\n"
                            ),
                            mode_hint="ro",
                            priority=2,
                            depends_on=[],
                            requires_approval=False,
                            acceptance_criteria=["Return evidence only."],
                            definition_of_done=["Validation completed."],
                            eta_minutes=10,
                            sla_tier="normal",
                        ),
                    ],
                    orch_q=_FakeQueue(),
                    root_ticket="ticket-1",
                    now=100.0,
                )
            finally:
                if original_file is not None:
                    globals_map["__file__"] = original_file
                else:
                    globals_map.pop("__file__", None)

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].role, "reviewer_local")
            self.assertIn("Validate implementer_local no-change claim", specs[0].text)

    def test_autonomous_local_first_prefers_reviewer_over_direct_retry_when_no_change_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            data_dir = repo_root / "data"
            repo_id = "codexbot-12345678"
            cfg = bot.BotConfig(
                **{
                    **TestStateHandling()._cfg(repo_root / "state.json").__dict__,
                    "worktree_root": data_dir / "worktrees",
                }
            )
            repo_worktree_root = bot._repo_worktree_root(cfg, repo_id=repo_id)
            worktree_dir = repo_worktree_root / "implementer_local" / "slot1"
            worktree_dir.mkdir(parents=True, exist_ok=True)
            (worktree_dir / "bot.py").write_text("VALUE = 1\n", encoding="utf-8")

            db_path = data_dir / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "create table jobs ("
                    " job_id text primary key,"
                    " parent_job_id text,"
                    " role text,"
                    " state text,"
                    " updated_at real,"
                    " created_at real,"
                    " artifacts_dir text,"
                    " labels text,"
                    " trace text"
                    ")"
                )
                arch_summary = (
                    "FILES:\n"
                    "- bot.py\n"
                    "CHANGE:\n"
                    "- no concrete code change to apply because behavior is already correct\n"
                    "VALIDATION:\n"
                    "- python3 -m py_compile bot.py\n"
                    "RISK:\n"
                    "- low\n"
                )
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 40.0, 39.0, ?, ?, ?)",
                    (
                        "arch-job-1",
                        "ticket-1",
                        "architect_local",
                        str(repo_root / "artifacts" / "arch-job-1"),
                        json.dumps({"key": "local_arch_guard_slice1"}),
                        json.dumps({"result_summary": arch_summary}),
                    ),
                )
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 60.0, 59.0, ?, ?, ?)",
                    (
                        "impl-job-1",
                        "ticket-1",
                        "implementer_local",
                        str(repo_root / "artifacts" / "impl-job-1"),
                        json.dumps({"key": "local_impl_guard_slice1"}),
                        json.dumps(
                            {
                                "result_summary": "No code change required because the requested behavior is already present.",
                                "slice_no_code_change": True,
                                "local_patch_info": {"no_code_change_required": True},
                            }
                        ),
                    ),
                )
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'failed', 70.0, 69.0, ?, ?, ?)",
                    (
                        "impl-job-2",
                        "ticket-1",
                        "implementer_local",
                        str(repo_root / "artifacts" / "impl-job-2"),
                        json.dumps({"key": "local_impl_guard_slice1_retry"}),
                        json.dumps(
                            {
                                "result_summary": "BLOCKER: Missing FAILED_VALIDATION_OUTPUT content.",
                                "slice_id": "slice1_retry",
                                "attempt_n": 2,
                                "failure_class": "blocked",
                            }
                        ),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            class _FakeQueue:
                def jobs_by_parent(self, *, parent_job_id: str, limit: int = 2000) -> list[object]:
                    return []

                def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                    return {
                        "order_id": order_id,
                        "chat_id": 8355547734,
                        "title": "Proactive Sprint: codexbot Reliability + Delivery",
                        "body": f"Improve reliability. [repo:{repo_id}]",
                    }

                def list_orders_global(self, status: str = "active", limit: int = 400) -> list[dict[str, object]]:
                    return []

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
                    if repo_id != "codexbot-12345678":
                        return None
                    return {
                        "repo_id": repo_id,
                        "path": str(repo_root / "repo"),
                        "default_branch": "main",
                        "autonomy_enabled": True,
                        "priority": 2,
                        "runtime_mode": "ceo-bounded",
                        "daily_budget": 0.0,
                        "status": "active",
                        "metadata": {},
                    }

                def get_job(self, job_id: str) -> object | None:
                    return SimpleNamespace(
                        trace={},
                        labels={},
                        parent_job_id="",
                        job_id=job_id,
                        chat_id=8355547734,
                    )

            globals_map = bot._apply_autonomous_local_first_policy.__globals__
            original_file = globals_map.get("__file__")
            globals_map["__file__"] = str(repo_root / "bot.py")
            try:
                specs = bot._apply_autonomous_local_first_policy(
                    cfg=cfg,
                    specs=[],
                    orch_q=_FakeQueue(),
                    root_ticket="ticket-1",
                    now=100.0,
                )
            finally:
                if original_file is not None:
                    globals_map["__file__"] = original_file
                else:
                    globals_map.pop("__file__", None)

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].role, "reviewer_local")
            self.assertIn("Validate implementer_local no-change claim", specs[0].text)

    def test_autonomous_local_first_prefers_reviewer_after_validated_implementer_even_if_planner_offers_architect(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            data_dir = repo_root / "data"
            repo_id = "codexbot-12345678"
            data_dir.mkdir(parents=True, exist_ok=True)
            cfg = bot.BotConfig(
                **{
                    **TestStateHandling()._cfg(repo_root / "state.json").__dict__,
                    "worktree_root": data_dir / "worktrees",
                }
            )
            db_path = data_dir / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "create table jobs ("
                    " job_id text primary key,"
                    " parent_job_id text,"
                    " role text,"
                    " state text,"
                    " updated_at real,"
                    " created_at real,"
                    " artifacts_dir text,"
                    " labels text,"
                    " trace text"
                    ")"
                )
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 90.0, 89.0, ?, ?, ?)",
                    (
                        "impl-job-validated",
                        "ticket-1",
                        "implementer_local",
                        str(repo_root / "artifacts" / "impl-job-validated"),
                        json.dumps({"key": "local_impl_guard_slice1"}),
                        json.dumps(
                            {
                                "result_summary": "Applied compatibility wrapper and validated it.",
                                "slice_id": "slice1",
                                "local_patch_info": {
                                    "changed_files": ["orchestrator/delegation.py"],
                                    "validation_ok": True,
                                },
                            }
                        ),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            class _FakeQueue:
                def jobs_by_parent(self, *, parent_job_id: str, limit: int = 2000) -> list[object]:
                    return []

                def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                    return {
                        "order_id": order_id,
                        "chat_id": 8355547734,
                        "title": "Proactive Sprint: codexbot Reliability + Delivery",
                        "body": f"Improve reliability. [repo:{repo_id}]",
                    }

                def list_orders_global(self, status: str = "active", limit: int = 400) -> list[dict[str, object]]:
                    return []

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
                    if repo_id != "codexbot-12345678":
                        return None
                    return {
                        "repo_id": repo_id,
                        "path": str(repo_root / "repo"),
                        "default_branch": "main",
                        "autonomy_enabled": True,
                        "priority": 2,
                        "runtime_mode": "ceo-bounded",
                        "daily_budget": 0.0,
                        "status": "active",
                        "metadata": {},
                    }

                def get_job(self, job_id: str) -> object | None:
                    return SimpleNamespace(
                        trace={},
                        labels={},
                        parent_job_id="",
                        job_id=job_id,
                        chat_id=8355547734,
                    )

            globals_map = bot._apply_autonomous_local_first_policy.__globals__
            original_file = globals_map.get("__file__")
            globals_map["__file__"] = str(repo_root / "bot.py")
            try:
                specs = bot._apply_autonomous_local_first_policy(
                    cfg=cfg,
                    specs=[
                        bot.TaskSpec(
                            key="auto_architect_local_slice1",
                            role="architect_local",
                            text="Plan another small slice.",
                            mode_hint="ro",
                            priority=2,
                            depends_on=[],
                            requires_approval=False,
                            acceptance_criteria=["Return one plan."],
                            definition_of_done=["Plan returned."],
                            eta_minutes=10,
                            sla_tier="normal",
                        )
                    ],
                    orch_q=_FakeQueue(),
                    root_ticket="ticket-1",
                    now=100.0,
                )
            finally:
                if original_file is not None:
                    globals_map["__file__"] = original_file
                else:
                    globals_map.pop("__file__", None)

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].role, "reviewer_local")
            self.assertIn("impl-job-validated", specs[0].text)
            self.assertIn("slice1", specs[0].text)
            self.assertIn("Applied compatibility wrapper", specs[0].text)
            self.assertIn("orchestrator/delegation.py", specs[0].text)

    def test_finalize_codex_implementer_change_applies_diff_when_workspace_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)

            target = repo / "bot.py"
            original = (
                "from pathlib import Path\n\n"
                "def _skill_segment_ok(name: str) -> bool:\n"
                "    s = (name or '').strip()\n"
                "    if not s or s in ('.', '..'):\n"
                "        return False\n"
                "    if s != Path(s).name:\n"
                "        return False\n"
                "    return bool(_SKILL_SEGMENT_RE.match(s))\n"
            )
            updated = original.replace("_SKILL_SEGMENT_RE.match(s)", "_SKILL_SEGMENT_RE.fullmatch(s)")
            target.write_text(original, encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            target.write_text(updated, encoding="utf-8")
            diff_text = subprocess.run(
                ["git", "diff", "--no-ext-diff", "--unified=3"],
                cwd=str(repo),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            target.write_text(original, encoding="utf-8")

            artifacts_dir = Path(td) / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            body = (
                "Applied the strict skill segment validation.\n\n"
                f"```diff\n{diff_text}```\n\n"
                "Validation: python3 -m py_compile bot.py\n"
            )

            artifacts, patch_info, patch_error = bot._finalize_codex_implementer_change(
                task=SimpleNamespace(),
                artifacts_dir=artifacts_dir,
                content=body,
                worktree_dir=repo,
            )

            self.assertIsNone(patch_error)
            self.assertIn("bot.py", patch_info.get("changed_files", []))
            self.assertTrue(bool(patch_info.get("validation_ok", False)))
            self.assertTrue(any(path.endswith("local_ollama_git_diff.patch") for path in artifacts))
            self.assertIn("_SKILL_SEGMENT_RE.fullmatch(s)", target.read_text(encoding="utf-8"))

    def test_finalize_codex_implementer_change_applies_begin_patch_update(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)

            target = repo / "orchestrator" / "delegation.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            original = (
                "def parse_orchestrator_subtasks(text):\n"
                "    ticket_line = f\" Ticket: {ticket_id}.\" if ticket_id else \"\"\n"
                "    text = (\n"
                "        \"Provide a single, bounded implementer-ready slice for proactive idle watchdog reseed.\"\n"
                "        \" Include exact file(s), one concrete change, validation command, and risks.\"\n"
                "        f\"{ticket_line}\"\n"
                "    )\n"
                "    acceptance_criteria = [\"Return one implementer-ready slice with file-scoped change and validation.\"]\n"
                "    return text, acceptance_criteria\n"
            )
            updated_snippet = "Match the ticket acceptance criteria"
            target.write_text(original, encoding="utf-8")
            subprocess.run(["git", "add", "orchestrator/delegation.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            artifacts_dir = Path(td) / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            body = (
                "```diff\n"
                "*** Begin Patch\n"
                "*** Update File: orchestrator/delegation.py\n"
                "@@\n"
                "     text = (\n"
                "         \"Provide a single, bounded implementer-ready slice for proactive idle watchdog reseed.\"\n"
                "-        \" Include exact file(s), one concrete change, validation command, and risks.\"\n"
                "+        \" Match the ticket acceptance criteria: include exact file(s), one concrete change,\"\n"
                "+        \" a diff or rewrite block, validation command, and risks.\"\n"
                "         f\"{ticket_line}\"\n"
                "     )\n"
                "@@\n"
                "-    acceptance_criteria = [\"Return one implementer-ready slice with file-scoped change and validation.\"]\n"
                "+    acceptance_criteria = [\n"
                "+        \"Return one implementer-ready slice with file-scoped change, diff or rewrite, and validation.\",\n"
                "+    ]\n"
                "*** End Patch\n"
                "```\n"
            )

            artifacts, patch_info, patch_error = bot._finalize_codex_implementer_change(
                task=SimpleNamespace(),
                artifacts_dir=artifacts_dir,
                content=body,
                worktree_dir=repo,
            )

            self.assertIsNone(patch_error)
            self.assertIn("orchestrator/delegation.py", patch_info.get("changed_files", []))
            self.assertEqual(str(patch_info.get("apply_mode") or ""), "apply_patch")
            self.assertTrue(bool(patch_info.get("validation_ok", False)))
            self.assertTrue(any(path.endswith("local_ollama_git_diff.patch") for path in artifacts))
            self.assertIn(updated_snippet, target.read_text(encoding="utf-8"))

    def test_finalize_codex_implementer_change_treats_already_fixed_blocker_as_no_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "bot.py").write_text("def ok():\n    return True\n", encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            artifacts_dir = Path(td) / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            content = (
                "BLOCKER: `bot.py` already uses `_SKILL_SEGMENT_RE.fullmatch(s)` in `_skill_segment_ok`, "
                "so there is no concrete code change to apply for the requested fix."
            )

            artifacts, patch_info, patch_error = bot._finalize_codex_implementer_change(
                task=SimpleNamespace(),
                artifacts_dir=artifacts_dir,
                content=content,
                worktree_dir=repo,
            )

            self.assertIsNone(patch_error)
            self.assertTrue(bool(patch_info.get("no_code_change_required", False)))
            self.assertTrue(any(path.endswith("codex_implementer_no_change.txt") for path in artifacts))

    def test_orchestrator_run_codex_applies_implementer_diff_even_when_mode_is_ro(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)

            target = repo / "bot.py"
            original = "VALUE = 1\n"
            updated = "VALUE = 2\n"
            target.write_text(original, encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            target.write_text(updated, encoding="utf-8")
            diff_text = subprocess.run(
                ["git", "diff", "--no-ext-diff", "--unified=3"],
                cwd=str(repo),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            target.write_text(original, encoding="utf-8")

            cfg = TestStateHandling()._cfg(Path(td) / "state.json")
            cfg = bot.dataclasses.replace(
                cfg,
                codex_workdir=repo,
                worktree_root=Path(td) / "worktrees",
                artifacts_root=Path(td) / "artifacts_root",
                orchestrator_sessions_enabled=False,
                orchestrator_live_update_seconds=0,
                codex_timeout_seconds=5,
            )
            cfg.artifacts_root.mkdir(parents=True, exist_ok=True)
            cfg.worktree_root.mkdir(parents=True, exist_ok=True)

            repo_id = "codexbot-12345678"

            class _FakeQueue:
                def __init__(self) -> None:
                    self.updated: list[tuple[str, dict[str, object]]] = []

                def lease_workspace(self, *, role: str, job_id: str, slots: int = 1) -> int:
                    return 1

                def release_workspace(self, *, job_id: str) -> None:
                    return None

                def update_trace(self, job_id: str, **kwargs: object) -> None:
                    self.updated.append((job_id, dict(kwargs)))

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
                    if repo_id != "codexbot-12345678":
                        return None
                    return {
                        "repo_id": repo_id,
                        "path": str(repo),
                        "default_branch": "main",
                        "autonomy_enabled": True,
                        "priority": 2,
                        "runtime_mode": "ceo-bounded",
                        "daily_budget": 0.0,
                        "status": "active",
                        "metadata": {},
                    }

            task = bot.Task.new(
                job_id="job-impl-ro",
                source="test",
                role="implementer_local",
                input_text="Implement one bounded improvement using the provided handoff.",
                request_type="task",
                priority=1,
                model="gpt-5.3-codex",
                effort="medium",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=0.0,
                chat_id=1,
                state="running",
                artifacts_dir=str((cfg.artifacts_root / "job-impl-ro").resolve()),
                trace={
                    "repo_id": repo_id,
                    "repo_path": str(repo),
                    "repo_default_branch": "main",
                },
            )

            class _FakeProc:
                def __init__(self) -> None:
                    self.pid = 12345
                    self.returncode = 0

                def poll(self) -> int:
                    return 0

                def wait(self, timeout: float | None = None) -> int:
                    return 0

            case = self

            def _fake_start(self, *, argv: list[str], mode_hint: str):  # type: ignore[no-untyped-def]
                case.assertEqual(mode_hint, "ro")
                stdout_path = Path(td) / "codex_stdout.jsonl"
                stderr_path = Path(td) / "codex_stderr.log"
                last_msg_path = Path(td) / "codex_last.txt"
                stdout_path.write_text("", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                last_msg_path.write_text(
                    "Applied the bounded change.\n\n"
                    f"```diff\n{diff_text}```\n\n"
                    "EXPECTED_VALIDATION: python -m py_compile bot.py\n",
                    encoding="utf-8",
                )
                return bot.CodexRunner.Running(
                    proc=_FakeProc(),
                    start_time=0.0,
                    cmd=["codex", "exec"],
                    last_msg_path=last_msg_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )

            fake_queue = _FakeQueue()
            profiles = {
                "implementer_local": {
                    "execution_backend": "codex",
                    "model": "gpt-5.3-codex",
                    "effort": "medium",
                    "max_parallel_jobs": 1,
                }
            }

            with patch.object(bot.CodexRunner, "start", _fake_start), patch.object(
                bot, "_orchestrator_min_evidence_gate", return_value=(True, "", {})
            ):
                result = bot._orchestrator_run_codex(
                    cfg,
                    task,
                    stop_event=threading.Event(),
                    orch_q=fake_queue,
                    profiles=profiles,
                )

            self.assertEqual(result["status"], "ok")
            self.assertIn("patch_info", result["structured_digest"])
            self.assertTrue(bool(result["structured_digest"]["patch_info"].get("validation_ok", False)))
            worktree_target = bot._task_scoped_repo_worktree_root(cfg, task=task, repo_base_dir=repo, repo_id=repo_id) / "implementer_local" / "slot1" / "bot.py"
            self.assertTrue(worktree_target.is_file())
            self.assertEqual(worktree_target.read_text(encoding="utf-8"), updated)

    def test_orchestrator_run_codex_extracts_json_payload_from_body_into_structured_digest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "bot.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            cfg = TestStateHandling()._cfg(Path(td) / "state.json")
            cfg = bot.dataclasses.replace(
                cfg,
                codex_workdir=repo,
                worktree_root=Path(td) / "worktrees",
                artifacts_root=Path(td) / "artifacts_root",
                orchestrator_sessions_enabled=False,
                orchestrator_live_update_seconds=0,
                codex_timeout_seconds=5,
            )
            cfg.artifacts_root.mkdir(parents=True, exist_ok=True)
            cfg.worktree_root.mkdir(parents=True, exist_ok=True)

            class _FakeQueue:
                def lease_workspace(self, *, role: str, job_id: str, slots: int = 1) -> int:
                    return 1

                def release_workspace(self, *, job_id: str) -> None:
                    return None

                def update_trace(self, job_id: str, **kwargs: object) -> None:
                    return None

            task = bot.Task.new(
                job_id="job-skynet-json",
                source="test",
                role="skynet",
                input_text="Drive one bounded proactive tick.",
                request_type="maintenance",
                priority=1,
                model="gpt-5.3-codex",
                effort="medium",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=0.0,
                chat_id=1,
                state="running",
                artifacts_dir=str((cfg.artifacts_root / "job-skynet-json").resolve()),
            )

            class _FakeProc:
                def __init__(self) -> None:
                    self.pid = 12345
                    self.returncode = 0

                def poll(self) -> int:
                    return 0

                def wait(self, timeout: float | None = None) -> int:
                    return 0

            def _fake_start(self, *, argv: list[str], mode_hint: str):  # type: ignore[no-untyped-def]
                stdout_path = Path(td) / "codex_stdout.jsonl"
                stderr_path = Path(td) / "codex_stderr.log"
                last_msg_path = Path(td) / "codex_last.txt"
                stdout_path.write_text("", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                last_msg_path.write_text(
                    "Skynet tick complete.\n\n"
                    "```json\n"
                    "{\n"
                    "  \"summary\": \"Blocked but recoverable.\",\n"
                    "  \"next_action\": {\n"
                    "    \"type\": \"LOCAL_REPLAN_REQUEST\",\n"
                    "    \"subtasks\": [\n"
                    "      {\n"
                    "        \"key\": \"arch_replan_1\",\n"
                    "        \"role\": \"architect_local\",\n"
                    "        \"text\": \"Define one bounded retry slice for bot.py\",\n"
                    "        \"mode_hint\": \"ro\",\n"
                    "        \"priority\": 1,\n"
                    "        \"acceptance_criteria\": [\"Return exact files and one validation command.\"],\n"
                    "        \"definition_of_done\": [\"Actionable implementer handoff is ready.\"]\n"
                    "      }\n"
                    "    ]\n"
                    "  }\n"
                    "}\n"
                    "```\n",
                    encoding="utf-8",
                )
                return bot.CodexRunner.Running(
                    proc=_FakeProc(),
                    start_time=0.0,
                    cmd=["codex", "exec"],
                    last_msg_path=last_msg_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )

            with patch.object(bot.CodexRunner, "start", _fake_start), patch.object(
                bot, "_orchestrator_min_evidence_gate", return_value=(True, "", {})
            ):
                result = bot._orchestrator_run_codex(
                    cfg,
                    task,
                    stop_event=threading.Event(),
                    orch_q=_FakeQueue(),
                    profiles={"skynet": {"execution_backend": "codex", "model": "gpt-5.3-codex", "effort": "medium"}},
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["next_action"], "LOCAL_REPLAN_REQUEST")
            structured = result["structured_digest"]
            self.assertIsInstance(structured.get("next_action"), dict)
            specs = bot.parse_orchestrator_subtasks(structured)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].role, "architect_local")

    def test_orchestrator_run_codex_blocks_controller_repo_writes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "bot.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            cfg = TestStateHandling()._cfg(Path(td) / "state.json")
            cfg = bot.dataclasses.replace(
                cfg,
                codex_workdir=repo,
                worktree_root=Path(td) / "worktrees",
                artifacts_root=Path(td) / "artifacts_root",
                orchestrator_sessions_enabled=False,
                orchestrator_live_update_seconds=0,
                codex_timeout_seconds=5,
            )
            cfg.artifacts_root.mkdir(parents=True, exist_ok=True)
            cfg.worktree_root.mkdir(parents=True, exist_ok=True)

            repo_id = "codexbot-12345678"

            class _FakeQueue:
                def lease_workspace(self, *, role: str, job_id: str, slots: int = 1) -> int:
                    return 1

                def release_workspace(self, *, job_id: str) -> None:
                    return None

                def update_trace(self, job_id: str, **kwargs: object) -> None:
                    return None

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
                    if repo_id != "codexbot-12345678":
                        return None
                    return {
                        "repo_id": repo_id,
                        "path": str(repo),
                        "default_branch": "main",
                        "autonomy_enabled": True,
                        "priority": 2,
                        "runtime_mode": "ceo-bounded",
                        "daily_budget": 0.0,
                        "status": "active",
                        "metadata": {},
                    }

            task = bot.Task.new(
                job_id="job-skynet-write",
                source="test",
                role="skynet",
                input_text="Drive one bounded proactive tick.",
                request_type="maintenance",
                priority=1,
                model="gpt-5.4",
                effort="high",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=0.0,
                chat_id=1,
                state="running",
                artifacts_dir=str((cfg.artifacts_root / "job-skynet-write").resolve()),
                trace={
                    "repo_id": repo_id,
                    "repo_path": str(repo),
                    "repo_default_branch": "main",
                    "proactive_lane": True,
                },
            )

            class _FakeProc:
                def __init__(self) -> None:
                    self.pid = 12345
                    self.returncode = 0

                def poll(self) -> int:
                    return 0

                def wait(self, timeout: float | None = None) -> int:
                    return 0

            def _fake_start(self, *, argv: list[str], mode_hint: str):  # type: ignore[no-untyped-def]
                case_worktree = Path(self._cfg.codex_workdir)
                (case_worktree / "bot.py").write_text("VALUE = 2\n", encoding="utf-8")
                stdout_path = Path(td) / "codex_stdout.jsonl"
                stderr_path = Path(td) / "codex_stderr.log"
                last_msg_path = Path(td) / "codex_last.txt"
                stdout_path.write_text("", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                last_msg_path.write_text("Applied a direct improvement in bot.py\n", encoding="utf-8")
                return bot.CodexRunner.Running(
                    proc=_FakeProc(),
                    start_time=0.0,
                    cmd=["codex", "exec"],
                    last_msg_path=last_msg_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )

            with patch.object(bot.CodexRunner, "start", _fake_start), patch.object(
                bot, "_orchestrator_min_evidence_gate", return_value=(True, "", {})
            ):
                result = bot._orchestrator_run_codex(
                    cfg,
                    task,
                    stop_event=threading.Event(),
                    orch_q=_FakeQueue(),
                    profiles={"skynet": {"execution_backend": "codex", "model": "gpt-5.4", "effort": "high"}},
                )

            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["next_action"], "delegate_local_subtask")
            violation = result["structured_digest"].get("write_policy_violation")
            self.assertIsInstance(violation, dict)
            self.assertIn("bot.py", violation.get("changed_paths", []))
            self.assertIn("Write policy violation", result["summary"])

class _FakeTelegramAPI:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, int | None]] = []

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        disable_web_page_preview: bool = True,
    ) -> int | None:
        self.sent.append((int(chat_id), str(text), int(reply_to_message_id) if reply_to_message_id is not None else None))
        return 123


class TestCeoQueryFastPath(unittest.TestCase):
    def _cfg(self, state_file: Path) -> bot.BotConfig:
        return TestStateHandling()._cfg(state_file)

    def test_who_am_i_is_answered_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            api = _FakeTelegramAPI()
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=10,
                user_id=20,
                message_id=30,
                username="alex",
                text="Hola, quien soy yo?",
            )

            handled = bot._maybe_handle_ceo_query(
                api=api,
                cfg=cfg,
                msg=msg,
                orchestrator_profiles={"jarvis": {"model": "gpt-5.2"}},
                orchestrator_queue=None,
            )
            self.assertTrue(handled)
            self.assertTrue(api.sent)
            _chat_id, text, _rt = api.sent[-1]
            self.assertIn("Jarvis: CEO = Alejandro Ponce", text)
            self.assertIn("user_id=20", text)
            self.assertIn("chat_id=10", text)

    def test_how_many_agents_uses_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            api = _FakeTelegramAPI()
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=10,
                user_id=20,
                message_id=30,
                username=None,
                text="How many agents do we have?",
            )
            profs = {"jarvis": {"model": "gpt-5.2"}, "backend": {"model": "gpt-5.2"}, "qa": {"model": "gpt-5.2"}}
            handled = bot._maybe_handle_ceo_query(
                api=api,
                cfg=cfg,
                msg=msg,
                orchestrator_profiles=profs,
                orchestrator_queue=None,
            )
            self.assertTrue(handled)
            _chat_id, text, _rt = api.sent[-1]
            self.assertIn("Jarvis: employees (agents) = 3", text)
            self.assertIn("jarvis", text)

    def test_natural_language_status_falls_through_to_jarvis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            api = _FakeTelegramAPI()
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=10,
                user_id=20,
                message_id=30,
                username=None,
                text="Que proyectos tienes ahorita en progreso?",
            )

            handled = bot._maybe_handle_ceo_query(
                api=api,
                cfg=cfg,
                msg=msg,
                orchestrator_profiles={"jarvis": {"model": "gpt-5.3-codex-spark"}},
                orchestrator_queue=None,
            )
            self.assertFalse(handled)
            self.assertFalse(api.sent)

    def test_pending_query_falls_through_to_jarvis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            api = _FakeTelegramAPI()
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=10,
                user_id=20,
                message_id=30,
                username=None,
                text="que tareas tenemos pendientes?",
            )
            handled = bot._maybe_handle_ceo_query(
                api=api,
                cfg=cfg,
                msg=msg,
                orchestrator_profiles={"jarvis": {"model": "gpt-5.3-codex-spark"}},
                orchestrator_queue=None,
            )
            self.assertFalse(handled)
            self.assertFalse(api.sent)

    def test_queries_do_not_trigger_when_slash_command(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            api = _FakeTelegramAPI()
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=10,
                user_id=20,
                message_id=30,
                username=None,
                text="/whoami",
            )
            handled = bot._maybe_handle_ceo_query(
                api=api,
                cfg=cfg,
                msg=msg,
                orchestrator_profiles=None,
                orchestrator_queue=None,
            )
            self.assertFalse(handled)
            self.assertFalse(api.sent)


class TestJarvisFirstRouting(unittest.TestCase):
    def _cfg(self, state_file: Path) -> bot.BotConfig:
        return TestStateHandling()._cfg(state_file)

    def test_orchestrator_task_defaults_to_jarvis_role(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "orchestrator_default_role": "jarvis"})
            job = bot.Job(
                chat_id=1,
                reply_to_message_id=0,
                user_text="Build a UI dashboard with screenshots.",
                argv=["exec", "Build a UI dashboard with screenshots."],
                mode_hint="ro",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            )
            profiles = {"jarvis": {"model": "gpt-5.2", "effort": "high", "mode_hint": "ro"}}
            task = bot._orchestrator_task_from_job(cfg, job, profiles=profiles, user_id=123)
            self.assertEqual(task.role, "jarvis")

    def test_explicit_role_marker_overrides_jarvis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "orchestrator_default_role": "jarvis"})
            job = bot.Job(
                chat_id=1,
                reply_to_message_id=0,
                user_text="@frontend Build a UI dashboard.",
                argv=["exec", "@frontend Build a UI dashboard."],
                mode_hint="ro",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            )
            profiles = {"jarvis": {"model": "gpt-5.2", "effort": "high", "mode_hint": "ro"}}
            task = bot._orchestrator_task_from_job(cfg, job, profiles=profiles, user_id=123)
            self.assertEqual(task.role, "frontend")

    def test_query_tasks_use_spark_high_and_suppress_ticket_card(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "orchestrator_default_role": "jarvis"})
            job = bot.Job(
                chat_id=1,
                reply_to_message_id=0,
                user_text="Que tenemos pendiente?",
                argv=["exec", "Que tenemos pendiente?"],
                mode_hint="rw",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            )

            task = bot._orchestrator_task_from_job(cfg, job, profiles=None, user_id=123)
            self.assertEqual(task.request_type, "query")
            self.assertEqual(task.mode_hint, "ro")
            self.assertEqual(task.effort, "high")
            self.assertEqual(task.model, "gpt-5.3-codex-spark")
            self.assertFalse(task.requires_approval)
            self.assertEqual(int(task.priority), 1)
            self.assertTrue(bool((task.trace or {}).get("suppress_ticket_card", False)))
            self.assertEqual(int((task.trace or {}).get("max_runtime_seconds") or 0), 120)

    def test_intent_query_forces_query_lane_even_if_request_type_looked_like_task(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "orchestrator_default_role": "jarvis"})
            job = bot.Job(
                chat_id=1,
                reply_to_message_id=0,
                user_text="ok, dame 15 proyectos interesantes que podemos hacer",
                argv=["exec", "ok, dame 15 proyectos interesantes que podemos hacer"],
                mode_hint="rw",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            )

            task = bot._orchestrator_task_from_job(
                cfg,
                job,
                profiles=None,
                user_id=123,
                intent_type="query",
            )
            self.assertEqual(task.request_type, "query")
            self.assertEqual(task.mode_hint, "ro")
            self.assertEqual(task.effort, "high")
            self.assertEqual(task.model, "gpt-5.3-codex-spark")
            self.assertTrue(bool((task.trace or {}).get("suppress_ticket_card", False)))


    def test_alias_p_maps_to_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/p")
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("Dos opciones:", resp)

    def test_alias_x_maps_to_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/x")
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertEqual(resp, "__cancel__")

    def test_alias_s_maps_to_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/s")
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("workdir:", resp)

    def test_alias_v_maps_to_voice(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/v")
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("voice transcription:", resp)

    def test_banned_flag_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/exec --dangerously-bypass-approvals-and-sandbox echo hi",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertIn("Not allowed:", resp)
            self.assertIsNone(job)

    def test_banned_flag_allowed_in_unsafe_direct_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "unsafe_direct_codex": True})
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/exec --dangerously-bypass-approvals-and-sandbox echo hi",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "")
            self.assertIsNotNone(job)

    def test_config_override_sandbox_permissions_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text='/exec -c \'sandbox_permissions=["disk-full-read-access"]\' echo hi',
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertIn("Not allowed: -c/--config sandbox_permissions", resp)
            self.assertIsNone(job)

    def test_codex_apply_rejected_even_with_global_flags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/codex --oss apply",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertIn("Not allowed: codex apply", resp)
            self.assertIsNone(job)

    def test_codex_apply_allowed_in_unsafe_direct_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "unsafe_direct_codex": True})
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/codex --oss apply",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "")
            self.assertIsNotNone(job)

    def test_add_dir_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/exec --add-dir /tmp hi")
            resp, job = bot._parse_job(cfg, msg)
            self.assertIn("Not allowed:", resp)
            self.assertIsNone(job)

    def test_review_model_override_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/review -c model=o3 .",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "")
            self.assertIsNotNone(job)

    def test_full_shortcut_parses(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/full echo hi")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "")
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(job.mode_hint, "full")
            self.assertEqual(job.argv, ["exec", "echo hi"])

    def test_permissions_shows_current_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/permissions")
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("Dos opciones:", resp)
            self.assertIn("Default (selected)", resp)
            self.assertIn("selected_access_mode: default", resp)

    def test_permissions_set_full_persists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            cfg = self._cfg(state_file)
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/permissions full")
            resp, job = bot._parse_job(cfg, msg)
            self.assertIn("access_mode_selected=full", resp)
            self.assertIn("dangerous_bypass=inactive", resp)
            self.assertIsNone(job)
            saved = json.loads(state_file.read_text())
            self.assertEqual(saved["access_mode_by_chat"]["1"], "full")

            msg2 = bot.IncomingMessage(update_id=2, chat_id=1, user_id=2, message_id=11, username="u", text="/permissions")
            resp2, job2 = bot._parse_job(cfg, msg2)
            self.assertIsNone(job2)
            self.assertIn("Breakglass", resp2)
            self.assertIn("status: inactive", resp2)
            self.assertIn("selected_access_mode: full", resp2)


class TestTelegramUploads(unittest.TestCase):
    def test_send_document_guesses_png_mimetype(self) -> None:
        api = bot.TelegramAPI(
            "dummy",
            http_timeout_seconds=1,
            http_max_retries=0,
            http_retry_initial_seconds=0.0,
            http_retry_max_seconds=0.0,
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
            with patch.object(api, "_request_multipart") as m:
                api.send_document(1, p)
                self.assertEqual(m.call_args.args[0], "sendDocument")
                self.assertEqual(m.call_args.kwargs["content_type"], "image/png")

    def test_send_document_adds_charset_for_text(self) -> None:
        api = bot.TelegramAPI(
            "dummy",
            http_timeout_seconds=1,
            http_max_retries=0,
            http_retry_initial_seconds=0.0,
            http_retry_max_seconds=0.0,
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.txt"
            p.write_text("hi\n", encoding="utf-8")
            with patch.object(api, "_request_multipart") as m:
                api.send_document(1, p)
                self.assertEqual(m.call_args.kwargs["content_type"], "text/plain; charset=utf-8")


class TestPngArtifacts(unittest.TestCase):
    def _cfg(self, workdir: Path) -> bot.BotConfig:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            state_file.write_text("{}\n", encoding="utf-8")
            # Reuse the shared config builder, but override workdir and state_file.
            base = TestStateHandling()._cfg(state_file)
            return bot.BotConfig(**{**base.__dict__, "codex_workdir": workdir.resolve()})

    def test_collects_referenced_png_within_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td).resolve()
            cfg = self._cfg(wd)
            p = wd / "out.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
            out = bot._collect_png_artifacts(cfg, start_time=999999.0, text="Saved to out.png\n")
            self.assertIn(p.resolve(), out)

    def test_ignores_png_outside_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td).resolve()
            cfg = self._cfg(wd)
            # Absolute path outside workdir should be ignored.
            out = bot._collect_png_artifacts(cfg, start_time=0.0, text="Saved to /etc/passwd.png\n")
            self.assertEqual(out, [])

    def test_collects_recent_pngs_by_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td).resolve()
            cfg = self._cfg(wd)
            old = wd / "old.png"
            new = wd / "new.png"
            old.write_bytes(b"\x89PNG\r\n\x1a\n")
            new.write_bytes(b"\x89PNG\r\n\x1a\n")
            os.utime(old, (1000, 1000))
            os.utime(new, (2000, 2000))
            out = bot._collect_png_artifacts(cfg, start_time=1500.0, text="")
            self.assertIn(new.resolve(), out)
            self.assertNotIn(old.resolve(), out)


class TestDrainPendingUpdates(unittest.TestCase):
    def test_drain_handles_api_failure(self) -> None:
        # Draining is best-effort; startup should not crash on transient network/DNS errors.
        with tempfile.TemporaryDirectory() as td:
            cfg = TestStateHandling()._cfg(Path(td) / "state.json")

            class _API:
                def get_updates(self, *, offset: int, timeout_seconds: int) -> list[dict]:
                    raise RuntimeError("boom")

            off = bot._drain_pending_updates(cfg, _API())  # type: ignore[arg-type]
            self.assertEqual(off, 0)

    def test_drain_returns_partial_offset_on_midway_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = TestStateHandling()._cfg(Path(td) / "state.json")

            class _API:
                def __init__(self) -> None:
                    self.calls = 0

                def get_updates(self, *, offset: int, timeout_seconds: int) -> list[dict]:
                    self.calls += 1
                    if self.calls == 1:
                        return [{"update_id": 41}, {"update_id": 42}]
                    raise RuntimeError("boom")

            off = bot._drain_pending_updates(cfg, _API())  # type: ignore[arg-type]
            self.assertEqual(off, 43)


class TestCodexConfigParsing(unittest.TestCase):
    def test_codex_defaults_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.toml"
            p.write_text('model = "gpt-5.3-codex"\nmodel_reasoning_effort = "xhigh"\n', encoding="utf-8")
            model, effort = bot._codex_defaults_from_config(p)
            self.assertEqual(model, "gpt-5.3-codex")
            self.assertEqual(effort, "xhigh")

    def test_job_model_label_prefers_argv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = TestStateHandling()._cfg(Path(td) / "state.json")
            model, _effort = bot._job_model_label(cfg, ["exec", "--model", "my-model"])
            self.assertEqual(model, "my-model")

    def test_format_model_for_display(self) -> None:
        self.assertIn("gpt-5.3-codex", bot._format_model_for_display("gpt-5.3-codex", "xhigh"))
        self.assertIn("codex 5.3", bot._format_model_for_display("gpt-5.3-codex", "xhigh"))
        self.assertIn("effort=xhigh", bot._format_model_for_display("gpt-5.3-codex", "xhigh"))
        self.assertEqual(bot._format_model_for_display("", ""), "n/a")


class TestDangerousBypassThreaded(unittest.TestCase):
    def _cfg(self) -> bot.BotConfig:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            state_file.write_text("{}\n", encoding="utf-8")
            base = TestStateHandling()._cfg(state_file)
            # Keep workdir stable for the runner; doesn't need to exist beyond command construction.
            return bot.BotConfig(
                **{
                    **base.__dict__,
                    "codex_workdir": Path(".").resolve(),
                    "codex_dangerous_bypass_sandbox": True,
                }
            )

    def test_threaded_new_injects_dangerous_bypass_and_omits_sandbox(self) -> None:
        cfg = self._cfg()
        runner = bot.CodexRunner(cfg)
        with patch.object(bot.subprocess, "Popen") as popen, patch("bot._breakglass_is_active", return_value=(True, {"reason": "test"})):
            popen.side_effect = FileNotFoundError("no codex in test")
            with self.assertRaises(FileNotFoundError):
                runner.start_threaded_new(prompt="hi", mode_hint="ro")
            cmd = popen.call_args.args[0]
            self.assertIn("--dangerously-bypass-approvals-and-sandbox", cmd)
            self.assertNotIn("--sandbox", cmd)

    def test_threaded_resume_injects_dangerous_bypass_and_omits_sandbox(self) -> None:
        cfg = self._cfg()
        runner = bot.CodexRunner(cfg)
        with patch.object(bot.subprocess, "Popen") as popen, patch("bot._breakglass_is_active", return_value=(True, {"reason": "test"})):
            popen.side_effect = FileNotFoundError("no codex in test")
            with self.assertRaises(FileNotFoundError):
                runner.start_threaded_resume(thread_id="t_123", prompt="hi", mode_hint="rw")
            cmd = popen.call_args.args[0]
            self.assertIn("--dangerously-bypass-approvals-and-sandbox", cmd)
            self.assertNotIn("--sandbox", cmd)

    def test_threaded_new_can_force_read_only_without_bypass(self) -> None:
        cfg = self._cfg()
        runner = bot.CodexRunner(cfg, allow_bypass=False, forced_mode="ro")
        with patch.object(bot.subprocess, "Popen") as popen, patch("bot._breakglass_is_active", return_value=(True, {"reason": "test"})):
            popen.side_effect = FileNotFoundError("no codex in test")
            with self.assertRaises(FileNotFoundError):
                runner.start_threaded_new(prompt="hi", mode_hint="full")
            cmd = popen.call_args.args[0]
            self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)
            self.assertIn("--sandbox", cmd)
            self.assertIn("read-only", cmd)

    def test_threaded_new_forced_read_only_ignores_global_bypass(self) -> None:
        cfg = self._cfg()
        runner = bot.CodexRunner(cfg, forced_mode="ro")
        with patch.object(bot.subprocess, "Popen") as popen, patch("bot._breakglass_is_active", return_value=(True, {"reason": "test"})):
            popen.side_effect = FileNotFoundError("no codex in test")
            with self.assertRaises(FileNotFoundError):
                runner.start_threaded_new(prompt="hi", mode_hint="full")
            cmd = popen.call_args.args[0]
            self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)
            self.assertIn("--sandbox", cmd)
            self.assertIn("read-only", cmd)


class TestThreadedImages(unittest.TestCase):
    def test_threaded_new_includes_image_flags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            state_file.write_text("{}\n", encoding="utf-8")
            cfg = TestStateHandling()._cfg(state_file)
            runner = bot.CodexRunner(cfg)
            img = Path(td) / "x.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n")
            with patch.object(bot.subprocess, "Popen") as popen:
                popen.side_effect = FileNotFoundError("no codex in test")
                with self.assertRaises(FileNotFoundError):
                    runner.start_threaded_new(prompt="hi", mode_hint="ro", image_paths=[img])
                cmd = popen.call_args.args[0]
                # The image flag should be present before the prompt at the end.
                self.assertIn("--image", cmd)
                self.assertIn(str(img), cmd)


class TestTelegramFormatting(unittest.TestCase):
    def test_markdownish_to_html_respects_inline_code(self) -> None:
        s = '**B** *i* _i2_ `**not-bold**`'
        out = bot._markdownish_to_telegram_html(s)
        self.assertIn("<b>B</b>", out)
        self.assertIn("<i>i</i>", out)
        self.assertIn("<i>i2</i>", out)
        self.assertIn("<code>**not-bold**</code>", out)

    def test_markdownish_to_html_links(self) -> None:
        s = "[OpenAI](https://openai.com)"
        out = bot._markdownish_to_telegram_html(s)
        self.assertIn('<a href="https://openai.com">OpenAI</a>', out)

    def test_markdownish_to_html_heading_and_list(self) -> None:
        s = "# Title\n- a\n* b\n"
        out = bot._markdownish_to_telegram_html(s)
        self.assertIn("<b>Title</b>\n", out)
        self.assertIn("• a\n", out)
        self.assertIn("• b\n", out)

    def test_markdownish_to_html_fenced_code_drops_lang(self) -> None:
        s = "```bash\nls\n```"
        out = bot._markdownish_to_telegram_html(s)
        self.assertIn("<pre><code>ls\n</code></pre>", out)


if __name__ == "__main__":
    unittest.main()


class TestAuthCommandVisibility(unittest.TestCase):
    def _cfg(self, state_file: Path, *, auth_enabled: bool) -> bot.BotConfig:
        cfg = TestStateHandling()._cfg(state_file)
        return bot.BotConfig(**{**cfg.__dict__, "auth_enabled": bool(auth_enabled)})

    def test_help_hides_login_logout_when_auth_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json", auth_enabled=False)
            txt = bot._help_text(cfg)
            self.assertNotIn("/login", txt)
            self.assertNotIn("/logout", txt)

    def test_help_shows_login_logout_when_auth_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json", auth_enabled=True)
            txt = bot._help_text(cfg)
            self.assertIn("/login", txt)
            self.assertIn("/logout", txt)

    def test_command_suggestions_omit_login_logout_when_auth_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json", auth_enabled=False)
            cmds = bot._telegram_commands_for_suggestions(cfg)
            names = {c[0] for c in cmds}
            self.assertNotIn("login", names)
            self.assertNotIn("logout", names)

    def test_command_suggestions_include_login_logout_when_auth_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json", auth_enabled=True)
            cmds = bot._telegram_commands_for_suggestions(cfg)
            names = {c[0] for c in cmds}
            self.assertIn("login", names)
            self.assertIn("logout", names)

    def test_parse_job_login_logout_behavior_depends_on_auth_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "state.json"

            cfg_off = self._cfg(state, auth_enabled=False)
            msg_login = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/login u p")
            resp, job = bot._parse_job(cfg_off, msg_login)
            self.assertIsNone(job)
            self.assertIn("Auth disabled", resp)
            self.assertFalse(resp.startswith("__login__:"))

            msg_logout = bot.IncomingMessage(update_id=2, chat_id=1, user_id=2, message_id=11, username="u", text="/logout")
            resp2, job2 = bot._parse_job(cfg_off, msg_logout)
            self.assertIsNone(job2)
            self.assertIn("Auth disabled", resp2)
            self.assertNotEqual(resp2, "__logout__")

            cfg_on = self._cfg(state, auth_enabled=True)
            resp3, job3 = bot._parse_job(cfg_on, msg_login)
            self.assertIsNone(job3)
            self.assertTrue(resp3.startswith("__login__:"))

            resp4, job4 = bot._parse_job(cfg_on, msg_logout)
            self.assertIsNone(job4)
            self.assertEqual(resp4, "__logout__")


class _FakeAPI:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_message(self, chat_id: int, text: str, reply_to_message_id: int | None = None):
        self.messages.append(str(text))
        return 1

    def send_photo(self, *args, **kwargs):
        return None

    def send_document(self, *args, **kwargs):
        return None

    def send_voice(self, *args, **kwargs):
        return None


class TestHardeningControls(unittest.TestCase):
    def _cfg(self, state_file: Path) -> bot.BotConfig:
        return TestStateHandling()._cfg(state_file)

    def test_breakglass_requires_admin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/breakglass on 5 emergency",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("No permitido", resp)

    def test_breakglass_enables_dangerous_bypass_only_while_active(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(
                **{
                    **cfg.__dict__,
                    "admin_chat_ids": frozenset({1}),
                    "codex_dangerous_bypass_sandbox": True,
                    "breakglass_ttl_seconds": 60,
                }
            )

            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/breakglass on 1 emergency fix",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("breakglass enabled", resp)

            bot._set_access_mode(cfg, "full", chat_id=1)
            self.assertTrue(bot._effective_bypass_sandbox(cfg, chat_id=1))

            active, raw = bot._breakglass_is_active(cfg)
            self.assertTrue(active)
            exp = float(raw.get("expires_at") or 0.0)
            with patch("bot.time.time", return_value=exp + 1.0):
                self.assertFalse(bot._effective_bypass_sandbox(cfg, chat_id=1))

    def test_effective_bypass_sandbox_auto_refreshes_startup_breakglass_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(
                **{
                    **cfg.__dict__,
                    "codex_dangerous_bypass_sandbox": True,
                    "breakglass_ttl_seconds": 60,
                    "breakglass_start_reason": "persistent_factory_runtime",
                }
            )

            bot._set_access_mode(cfg, "full", chat_id=1)
            self.assertTrue(bot._effective_bypass_sandbox(cfg, chat_id=1))
            active, raw = bot._breakglass_is_active(cfg)
            self.assertTrue(active)
            self.assertEqual(str(raw.get("reason") or ""), "persistent_factory_runtime")

    def test_permissions_full_ack_reports_selected_mode_and_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/permissions full",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("access_mode_selected=full", resp)
            self.assertIn("dangerous_bypass=inactive", resp)
            self.assertIn("breakglass=inactive", resp)

    def test_botpermissions_shows_selected_mode_separately_from_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            bot._set_access_mode(cfg, "full", chat_id=1)
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/botpermissions",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("access_mode_selected: full", resp)
            self.assertIn("dangerous_bypass: inactive", resp)
            self.assertIn("breakglass: inactive", resp)
            self.assertIn("note: full selected, but dangerous bypass is OFF until breakglass is active", resp)

    def test_breakglass_status_shows_selected_mode_and_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            bot._set_access_mode(cfg, "full", chat_id=1)
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/breakglass",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("access_mode_selected: full", resp)
            self.assertIn("dangerous_bypass: inactive", resp)

    def test_auth_touch_session_extends_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "auth_enabled": True, "auth_session_ttl_seconds": 30})
            bot._set_auth_sessions(
                cfg,
                {
                    "1": {
                        "username": "u",
                        "profile": "",
                        "logged_in_at": 100.0,
                        "last_active_at": 100.0,
                        "expires_at": 130.0,
                    }
                },
            )
            with patch("bot._auth_now", return_value=200.0):
                bot._auth_touch_session(cfg, chat_id=1)
            with patch("bot._auth_now", return_value=220.0):
                active, sess = bot._auth_is_session_active(cfg, chat_id=1)
            self.assertTrue(active)
            self.assertEqual(int(float(sess.get("expires_at") or 0)), 230)

    def test_runbook_ok_is_suppressed_in_minimal_notify_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "orchestrator_notify_mode": "minimal"})
            api = _FakeAPI()
            task = SimpleNamespace(
                labels={"runbook": "sre_health"},
                trace={"runbook_id": "sre_health"},
                role="sre",
                parent_job_id="",
                is_autonomous=True,
                job_id="12345678-1234-1234-1234-123456789012",
                chat_id=1,
                reply_to_message_id=None,
            )
            result = {
                "status": "ok",
                "summary": "Host healthy",
                "logs": "",
                "next_action": None,
                "artifacts": [],
            }
            bot._send_orchestrator_result(api, task, result, cfg=cfg)
            self.assertEqual(api.messages, [])


    def test_notify_policy_command_sets_scope(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=1,
                user_id=2,
                message_id=10,
                username="u",
                text="/notify policy critical",
            )
            resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("notify policy", resp.lower())
            self.assertEqual(bot._effective_notify_scope(cfg, chat_id=1), "critical")

    def test_notify_dedupe_suppresses_duplicate_worker_updates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            cfg = bot.BotConfig(
                **{
                    **cfg.__dict__,
                    "orchestrator_notify_mode": "minimal",
                    "notify_scope": "state_change",
                    "notify_dedupe_cooldown_seconds": 3600,
                    "notify_child_worker_completions": True,
                }
            )
            api = _FakeAPI()
            task = SimpleNamespace(
                labels={"ticket": "abcd1234"},
                trace={},
                role="backend",
                parent_job_id="abcd1234",
                is_autonomous=False,
                job_id="12345678-1234-1234-1234-123456789012",
                chat_id=1,
                reply_to_message_id=None,
            )
            result = {
                "status": "ok",
                "summary": "Service deployed",
                "logs": "",
                "next_action": None,
                "artifacts": [],
            }
            bot._send_orchestrator_result(api, task, result, cfg=cfg)
            bot._send_orchestrator_result(api, task, result, cfg=cfg)
            self.assertEqual(len(api.messages), 1)


class TestLocalSpecialistDelegation(unittest.TestCase):
    def test_extract_codex_token_usage_parses_turn_summary(self) -> None:
        usage = bot._extract_codex_token_usage(
            "[out] [turn] completed tokens in=530686 out=7145\n"
        )
        self.assertEqual(
            usage,
            {
                "input_tokens": 530686,
                "output_tokens": 7145,
                "total_tokens": 537831,
            },
        )

    def test_inject_local_specialists_skips_manual_ceo_work_by_default(self) -> None:
        specs = [
            bot.TaskSpec(
                key="backend_fix",
                role="backend",
                text="Fix API contract mismatch.",
                mode_hint="rw",
                priority=2,
            ),
            bot.TaskSpec(
                key="frontend_touchup",
                role="frontend",
                text="Adjust card spacing and typography.",
                mode_hint="rw",
                priority=2,
            ),
        ]
        out = bot._inject_local_specialist_specs(
            specs=specs,
            root_ticket="ticket-123",
            existing_keys=set(),
            request_type="task",
            is_top_level_manual=True,
            proactive_lane=False,
            allow_delegation=False,
        )
        roles = [str(x.role or "").strip().lower() for x in out]
        self.assertNotIn("architect_local", roles)
        self.assertNotIn("implementer_local", roles)
        self.assertNotIn("reviewer_local", roles)

    def test_inject_local_specialists_respects_disable_flag(self) -> None:
        specs = [
            bot.TaskSpec(
                key="backend_fix",
                role="backend",
                text="Fix API contract mismatch.",
                mode_hint="rw",
                priority=2,
            )
        ]
        with patch.dict(os.environ, {"BOT_LOCAL_SPECIALISTS_ENFORCE": "0"}, clear=False):
            out = bot._inject_local_specialist_specs(
                specs=specs,
                root_ticket="ticket-123",
                existing_keys=set(),
                request_type="task",
                is_top_level_manual=True,
                proactive_lane=False,
                allow_delegation=False,
            )
        roles = [str(x.role or "").strip().lower() for x in out]
        self.assertNotIn("architect_local", roles)
        self.assertNotIn("implementer_local", roles)
        self.assertNotIn("reviewer_local", roles)

    def test_inject_local_specialists_adds_all_three_roles_for_proactive_lane(self) -> None:
        specs = [
            bot.TaskSpec(
                key="backend_fix",
                role="backend",
                text="Fix API contract mismatch.",
                mode_hint="rw",
                priority=2,
            ),
            bot.TaskSpec(
                key="frontend_touchup",
                role="frontend",
                text="Adjust card spacing and typography.",
                mode_hint="rw",
                priority=2,
            ),
        ]
        out = bot._inject_local_specialist_specs(
            specs=specs,
            root_ticket="ticket-123",
            existing_keys=set(),
            request_type="task",
            is_top_level_manual=False,
            proactive_lane=True,
            allow_delegation=False,
        )
        roles = [str(x.role or "").strip().lower() for x in out]
        self.assertIn("architect_local", roles)
        self.assertIn("implementer_local", roles)
        self.assertIn("reviewer_local", roles)

    def test_inject_local_specialists_manual_override_enables_all_three_roles(self) -> None:
        specs = [
            bot.TaskSpec(
                key="backend_fix",
                role="backend",
                text="Fix API contract mismatch.",
                mode_hint="rw",
                priority=2,
            ),
            bot.TaskSpec(
                key="frontend_touchup",
                role="frontend",
                text="Adjust card spacing and typography.",
                mode_hint="rw",
                priority=2,
            ),
        ]
        with patch.dict(os.environ, {"BOT_CEO_INJECT_LOCAL_SPECIALISTS": "1"}, clear=False):
            out = bot._inject_local_specialist_specs(
                specs=specs,
                root_ticket="ticket-123",
                existing_keys=set(),
                request_type="task",
                is_top_level_manual=True,
                proactive_lane=False,
                allow_delegation=False,
            )
        roles = [str(x.role or "").strip().lower() for x in out]
        self.assertIn("architect_local", roles)
        self.assertIn("implementer_local", roles)
        self.assertIn("reviewer_local", roles)

    def test_normalize_contract_forces_local_roles_read_only(self) -> None:
        spec = bot.TaskSpec(
            key="local_impl",
            role="implementer_local",
            text="Draft implementation strategy.",
            mode_hint="rw",
            priority=2,
            requires_approval=True,
            acceptance_criteria=["Produce ordered steps"],
            definition_of_done=["Plan published"],
            eta_minutes=30,
            sla_tier="high",
        )
        normalized = bot._normalize_task_spec_contract(spec, root_ticket="ticket-123")
        self.assertEqual(normalized.mode_hint, "ro")
        self.assertFalse(normalized.requires_approval)

    def test_autonomous_local_first_hard_enforcement_keeps_only_local_roles_even_if_flag_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            cfg = TestStateHandling()._cfg(repo_root / "state.json")

            class _FakeQueue:
                def jobs_by_parent(self, *, parent_job_id: str, limit: int = 2000) -> list[object]:
                    return []

                def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                    return {
                        "order_id": order_id,
                        "chat_id": 8355547734,
                        "title": "Proactive Sprint: codexbot Reliability + Delivery",
                        "body": "Ship one bounded improvement. [repo:codexbot-12345678]",
                    }

                def list_orders_global(self, status: str = "active", limit: int = 400) -> list[dict[str, object]]:
                    return []

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
                    if repo_id != "codexbot-12345678":
                        return None
                    return {
                        "repo_id": repo_id,
                        "path": str(repo_root / "repo"),
                        "default_branch": "main",
                        "autonomy_enabled": True,
                        "priority": 2,
                        "runtime_mode": "ceo-bounded",
                        "daily_budget": 0.0,
                        "status": "active",
                        "metadata": {},
                    }

                def get_job(self, job_id: str) -> object | None:
                    return SimpleNamespace(
                        trace={},
                        labels={},
                        parent_job_id="",
                        job_id=job_id,
                        chat_id=8355547734,
                    )

            globals_map = bot._apply_autonomous_local_first_policy.__globals__
            original_file = globals_map.get("__file__")
            globals_map["__file__"] = str(repo_root / "bot.py")
            try:
                with patch.dict(os.environ, {"BOT_AUTONOMOUS_LOCAL_FIRST": "0"}, clear=False):
                    specs = bot._apply_autonomous_local_first_policy(
                        cfg=cfg,
                        specs=[
                            bot.TaskSpec(
                                key="delivery_fix",
                                role="backend",
                                text="Implement one bounded reliability fix with concrete evidence.",
                                mode_hint="rw",
                                priority=1,
                                depends_on=[],
                                requires_approval=False,
                                acceptance_criteria=["Ship one concrete fix."],
                                definition_of_done=["Improvement is implemented."],
                                eta_minutes=45,
                                sla_tier="high",
                            ),
                            bot.TaskSpec(
                                key="delivery_review",
                                role="qa",
                                text="Validate the change with PASS/FAIL evidence.",
                                mode_hint="ro",
                                priority=1,
                                depends_on=["delivery_fix"],
                                requires_approval=False,
                                acceptance_criteria=["Return a verdict."],
                                definition_of_done=["Validation is complete."],
                                eta_minutes=30,
                                sla_tier="high",
                            ),
                        ],
                        orch_q=_FakeQueue(),
                        root_ticket="ticket-1",
                        now=100.0,
                        enforce_hard_local_only=True,
                    )
            finally:
                if original_file is not None:
                    globals_map["__file__"] = original_file
                else:
                    globals_map.pop("__file__", None)

            roles = [str(spec.role or "").strip().lower() for spec in specs]
            self.assertGreaterEqual(len(roles), 1)
            self.assertTrue(set(roles).issubset({"architect_local", "implementer_local", "reviewer_local"}))
            self.assertNotIn("backend", roles)
            self.assertNotIn("qa", roles)

    def test_autonomous_local_first_reuses_fresh_architect_handoff_instead_of_replanning_architect(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            data_dir = repo_root / "data"
            repo_id = "codexbot-12345678"
            cfg = bot.BotConfig(
                **{
                    **TestStateHandling()._cfg(repo_root / "state.json").__dict__,
                    "worktree_root": data_dir / "worktrees",
                }
            )
            repo_worktree_root = bot._repo_worktree_root(cfg, repo_id=repo_id)
            worktree_dir = repo_worktree_root / "implementer_local" / "slot1"
            worktree_dir.mkdir(parents=True, exist_ok=True)
            (worktree_dir / "bot.py").write_text("VALUE = 1\n", encoding="utf-8")

            db_path = data_dir / "jobs.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "create table jobs ("
                    " job_id text primary key,"
                    " parent_job_id text,"
                    " role text,"
                    " state text,"
                    " updated_at real,"
                    " created_at real,"
                    " artifacts_dir text,"
                    " labels text,"
                    " trace text"
                    ")"
                )
                arch_summary = (
                    "FILES:\n"
                    "- bot.py\n"
                    "CHANGE:\n"
                    "- tighten the retry guard in `_enqueue_order_autopilot_task`\n"
                    "VALIDATION:\n"
                    "- python3 -m py_compile bot.py\n"
                    "RISK:\n"
                    "- low\n"
                )
                conn.execute(
                    "insert into jobs (job_id, parent_job_id, role, state, updated_at, created_at, artifacts_dir, labels, trace)"
                    " values (?, ?, ?, 'done', 60.0, 59.0, ?, ?, ?)",
                    (
                        "arch-job-1",
                        "ticket-1",
                        "architect_local",
                        str(repo_root / "artifacts" / "arch-job-1"),
                        json.dumps({"key": "local_arch_guard_slice1"}),
                        json.dumps({"result_summary": arch_summary}),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            class _FakeQueue:
                def jobs_by_parent(self, *, parent_job_id: str, limit: int = 2000) -> list[object]:
                    return []

                def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                    return {
                        "order_id": order_id,
                        "chat_id": 8355547734,
                        "title": "Proactive Sprint: codexbot Reliability + Delivery",
                        "body": f"Improve reliability. [repo:{repo_id}]",
                    }

                def list_orders_global(self, status: str = "active", limit: int = 400) -> list[dict[str, object]]:
                    return []

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
                    if repo_id != "codexbot-12345678":
                        return None
                    return {
                        "repo_id": repo_id,
                        "path": str(repo_root / "repo"),
                        "default_branch": "main",
                        "autonomy_enabled": True,
                        "priority": 2,
                        "runtime_mode": "ceo-bounded",
                        "daily_budget": 0.0,
                        "status": "active",
                        "metadata": {},
                    }

                def get_job(self, job_id: str) -> object | None:
                    return SimpleNamespace(
                        trace={},
                        labels={},
                        parent_job_id="",
                        job_id=job_id,
                        chat_id=8355547734,
                    )

            globals_map = bot._apply_autonomous_local_first_policy.__globals__
            original_file = globals_map.get("__file__")
            globals_map["__file__"] = str(repo_root / "bot.py")
            try:
                specs = bot._apply_autonomous_local_first_policy(
                    cfg=cfg,
                    specs=[
                        bot.TaskSpec(
                            key="auto_architect_local_slice1",
                            role="architect_local",
                            text="Plan the next minimal bounded slice.",
                            mode_hint="ro",
                            priority=2,
                            depends_on=[],
                            requires_approval=False,
                            acceptance_criteria=["Return one plan."],
                            definition_of_done=["Plan returned."],
                            eta_minutes=10,
                            sla_tier="normal",
                        )
                    ],
                    orch_q=_FakeQueue(),
                    root_ticket="ticket-1",
                    now=100.0,
                )
            finally:
                if original_file is not None:
                    globals_map["__file__"] = original_file
                else:
                    globals_map.pop("__file__", None)

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].role, "implementer_local")
            self.assertIn("PRIMARY_ARCHITECT_HANDOFF", specs[0].text)
            self.assertIn("bot.py", specs[0].text)


class TestLocalImplementerPatchApply(unittest.TestCase):
    def test_sanitize_git_patch_text_strips_mode_headers_and_focused_markers(self) -> None:
        raw = (
            "diff --git a/bot.py b/bot.py\n"
            "old mode 100755\n"
            "new mode 100644\n"
            "--- a/bot.py\n"
            "+++ b/bot.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def value():\n"
            "# [FOCUSED_EXACT_TARGET_CONTEXT bot.py:1-4 symbol=value]\n"
            "-    return 1\n"
            "+    return 2\n"
        )
        cleaned, changed = bot._sanitize_git_patch_text(raw)
        self.assertTrue(changed)
        self.assertNotIn("old mode 100755", cleaned)
        self.assertNotIn("new mode 100644", cleaned)
        self.assertNotIn("FOCUSED_EXACT_TARGET_CONTEXT", cleaned)
        self.assertIn("return 2", cleaned)

    def test_apply_local_implementer_patch_uses_3way_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            (repo / "bot.py").write_text("def value():\n    return 1\n", encoding="utf-8")
            patch_text = (
                "diff --git a/bot.py b/bot.py\n"
                "--- a/bot.py\n"
                "+++ b/bot.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value():\n"
                "-    return 1\n"
                "+    return 2\n"
            )
            artifacts_dir = Path(td) / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            task = SimpleNamespace(
                artifacts_dir=str(artifacts_dir),
                input_text="Implement ticket-1 in bot.py",
                trace={},
                parent_job_id="ticket-1",
            )
            fake_finalize = (["artifact"], {"apply_mode": "patch_3way", "changed_files": ["bot.py"]})
            cp = subprocess.CompletedProcess
            side_effects = [
                cp(args=["git", "rev-parse", "--show-toplevel"], returncode=0, stdout=str(repo) + "\n", stderr=""),
                cp(args=["git", "apply", "--check", str(artifacts_dir / "local_ollama_patch.diff")], returncode=1, stdout="", stderr="error: patch failed: bot.py:1\nerror: bot.py: patch does not apply\n"),
                cp(args=["git", "apply", "--check", "--recount", str(artifacts_dir / "local_ollama_patch.diff")], returncode=1, stdout="", stderr="error: patch failed: bot.py:1\nerror: bot.py: patch does not apply\n"),
                cp(args=["git", "apply", "--reverse", "--check", str(artifacts_dir / "local_ollama_patch.diff")], returncode=1, stdout="", stderr=""),
                cp(args=["git", "apply", "--check", "--3way", str(artifacts_dir / "local_ollama_patch.diff")], returncode=0, stdout="", stderr=""),
                cp(args=["git", "apply", "--3way", str(artifacts_dir / "local_ollama_patch.diff")], returncode=0, stdout="Applied patch to 'bot.py' cleanly.\n", stderr=""),
            ]
            with patch.object(bot.subprocess, "run", side_effect=side_effects) as run_mock, patch.object(
                bot,
                "_finalize_local_implementer_change",
                return_value=fake_finalize,
            ) as finalize_mock:
                artifacts, patch_info = bot._apply_local_implementer_patch(
                    task=task,
                    artifacts_dir=artifacts_dir,
                    content=f"```diff\n{patch_text}```\n",
                    worktree_dir=repo,
                )
            self.assertIn("bot.py", patch_info.get("changed_files") or [])
            self.assertEqual(patch_info.get("apply_mode"), "patch_3way")
            note_path = artifacts_dir / "local_ollama_patch_3way.txt"
            self.assertTrue(note_path.exists())
            self.assertIn("git apply --3way", note_path.read_text(encoding="utf-8"))
            finalize_kwargs = finalize_mock.call_args.kwargs
            self.assertEqual(finalize_kwargs.get("apply_mode"), "patch_3way")
            self.assertTrue(any("--3way" in " ".join(map(str, call.args[0])) for call in run_mock.call_args_list[4:6]))
            self.assertEqual(artifacts, fake_finalize[0])


class TestImplementerFailureSelection(unittest.TestCase):
    def test_implementer_failure_actionable_signal_detects_py_compile_syntax_error(self) -> None:
        summary = (
            "FAILED_VALIDATION_OUTPUT:\n"
            "$ python3 -m py_compile tools/visual_preview_audit.py\n"
            'File "tools/visual_preview_audit.py", line 281\n'
            "    {\n"
            "    ^^\n"
            "SyntaxError: did you forget parentheses around the comprehension target?\n"
        )
        self.assertTrue(bot._implementer_failure_summary_has_actionable_signal(summary))

    def test_implementer_failure_actionable_signal_rejects_generic_blocker_wrapped_as_validation(self) -> None:
        summary = (
            "FAILED_VALIDATION_OUTPUT:\n"
            "Local Ollama execution failed for role=implementer_local model=qwen3.5:27b: "
            "implementer_local blocker: The request mandates editing `tools/visual_preview_audit.py` "
            "but provides no specific failing test output or description of broken behavior.\n"
        )
        self.assertFalse(bot._implementer_failure_summary_has_actionable_signal(summary))

    def test_implementer_failure_actionable_signal_rejects_missing_failure_scenario_blocker(self) -> None:
        summary = (
            "BLOCKER: The request mandates editing `tools/visual_preview_audit.py` but provides no concrete "
            "failing test output, error message, or description of the missing/broken functionality. "
            "Without a concrete failure scenario, I cannot safely generate a targeted fix.\n\n"
            "FAILED_VALIDATION_OUTPUT:\n"
            "Local Ollama execution failed for role=implementer_local model=qwen3.5:27b: "
            "implementer_local blocker: same generic request for more context.\n\n"
            "To proceed, please provide:\n"
            "1. The specific test name or path that is failing.\n"
            "2. The exact error message or traceback from the failure.\n"
            "3. A description of the expected vs. actual behavior.\n"
        )
        self.assertFalse(bot._implementer_failure_summary_has_actionable_signal(summary))

    def test_select_preferred_implementer_failure_prefers_actionable_terminal(self) -> None:
        generic_blocker = (
            "BLOCKER: please provide failing test output for tools/visual_preview_audit.py",
            "job-newer",
            "slice-newer",
            2,
            "blocked",
            200.0,
        )
        actionable_terminal = (
            "FAILED_VALIDATION_OUTPUT:\n"
            "$ python3 -m py_compile tools/visual_preview_audit.py\n"
            'File "tools/visual_preview_audit.py", line 281\n'
            "SyntaxError: did you forget parentheses around the comprehension target?\n",
            "job-older",
            "slice-older",
            1,
            "terminal",
            100.0,
        )
        chosen = bot._select_preferred_implementer_failure([generic_blocker, actionable_terminal])
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen[1], "job-older")

    def test_ignore_cancelled_local_failure_when_summary_is_benign(self) -> None:
        self.assertTrue(
            bot._should_ignore_cancelled_local_failure(
                state_norm="cancelled",
                failure_class="retriable",
                attempt_n=7,
                summary="cancelled",
            )
        )
        self.assertTrue(
            bot._should_ignore_cancelled_local_failure(
                state_norm="cancelled",
                failure_class="retriable",
                attempt_n=3,
                summary="",
            )
        )
        self.assertFalse(
            bot._should_ignore_cancelled_local_failure(
                state_norm="cancelled",
                failure_class="terminal",
                attempt_n=3,
                summary="FAILED_VALIDATION_OUTPUT: SyntaxError in tools/visual_preview_audit.py",
            )
        )


class TestSkynetLocalOnlyProactivePolicy(unittest.TestCase):
    def test_skynet_factory_local_only_enabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BOT_SKYNET_FACTORY_LOCAL_ONLY", None)
            self.assertTrue(bot._skynet_factory_local_only_enabled())

    def test_skynet_factory_local_only_can_be_disabled_explicitly(self) -> None:
        with patch.dict(os.environ, {"BOT_SKYNET_FACTORY_LOCAL_ONLY": "0"}, clear=False):
            self.assertFalse(bot._skynet_factory_local_only_enabled())

    def test_local_role_worktree_dir_scopes_repo_workspace_by_change_token(self) -> None:
        cfg = SimpleNamespace(worktree_root=Path("/tmp/poncebot-worktrees"))
        path_a = bot._local_role_worktree_dir(
            "implementer_local",
            cfg=cfg,
            repo_id="codexbot-12345678",
            change_token="slice_a",
        )
        path_b = bot._local_role_worktree_dir(
            "implementer_local",
            cfg=cfg,
            repo_id="codexbot-12345678",
            change_token="slice_b",
        )
        self.assertNotEqual(path_a, path_b)
        self.assertIn("/changes/slice_a/", str(path_a))
        self.assertIn("/changes/slice_b/", str(path_b))

    def test_task_scoped_repo_worktree_root_uses_change_token(self) -> None:
        cfg = SimpleNamespace(worktree_root=Path("/tmp/poncebot-worktrees"))
        task_a = SimpleNamespace(trace={"slice_id": "slice_a"}, labels={}, job_id="job-a")
        task_b = SimpleNamespace(trace={"slice_id": "slice_b"}, labels={}, job_id="job-b")
        root_a = bot._task_scoped_repo_worktree_root(
            cfg,
            task=task_a,
            repo_base_dir=Path("/tmp/repo"),
            repo_id="codexbot-12345678",
        )
        root_b = bot._task_scoped_repo_worktree_root(
            cfg,
            task=task_b,
            repo_base_dir=Path("/tmp/repo"),
            repo_id="codexbot-12345678",
        )
        self.assertNotEqual(root_a, root_b)
        self.assertIn("/changes/slice_a", str(root_a))
        self.assertIn("/changes/slice_b", str(root_b))

    def test_local_workspace_change_token_uses_order_namespace_when_key_is_generic(self) -> None:
        task = SimpleNamespace(
            trace={},
            labels={"key": "project"},
            parent_job_id="",
            job_id="533df86f-8895-4685-8f33-287375e01121",
        )
        self.assertEqual(bot._local_workspace_change_token(task), "533df86f")

    def test_local_workspace_change_token_combines_order_namespace_with_concrete_slice(self) -> None:
        task = SimpleNamespace(
            trace={"slice_id": "82f10b4b_r2"},
            labels={"key": "project"},
            parent_job_id="533df86f-8895-4685-8f33-287375e01121",
            job_id="4b6f0ceb-36e0-43d2-8615-a785b4c2bd47",
        )
        self.assertEqual(bot._local_workspace_change_token(task), "533df86f_82f10b4b_r2")

    def test_proactive_cli_promotion_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BOT_PROACTIVE_ALLOW_CLI_PROMOTION", None)
            self.assertFalse(bot._proactive_cli_promotion_enabled())

    def test_order_meaningful_improvement_ignores_backend_qa_only(self) -> None:
        root = SimpleNamespace(job_id="root", trace={}, created_at=0.0, updated_at=0.0)
        children = [
            SimpleNamespace(
                role="backend",
                state="done",
                trace={"result_status": "done", "result_summary": "Implemented concrete improvement with evidence."},
                labels={},
                created_at=100.0,
                updated_at=100.0,
            ),
            SimpleNamespace(
                role="qa",
                state="done",
                trace={"result_status": "done", "result_summary": "PASS validation with evidence artifacts."},
                labels={},
                created_at=120.0,
                updated_at=120.0,
            ),
        ]

        class FakeQueue:
            def get_job(self, job_id: str):
                return root if job_id == "root" else None

            def jobs_by_parent(self, parent_job_id: str, limit: int = 600):
                return list(children) if parent_job_id == "root" else []

        self.assertFalse(bot._order_has_meaningful_improvement(orch_q=FakeQueue(), root_ticket="root"))

    def test_order_meaningful_improvement_true_when_already_merged(self) -> None:
        root = SimpleNamespace(job_id="root", trace={"merged_to_main": True}, created_at=0.0, updated_at=0.0)

        class FakeQueue:
            def get_job(self, job_id: str):
                return root if job_id == "root" else None

            def jobs_by_parent(self, parent_job_id: str, limit: int = 600):
                return []

        self.assertTrue(bot._order_has_meaningful_improvement(orch_q=FakeQueue(), root_ticket="root"))


class TestDeployCheckoutSync(unittest.TestCase):
    def test_git_ensure_branch_from_main_ignores_stale_remote_tracking_refs(self) -> None:
        repo = Path("/tmp/factory-repo")
        calls: list[list[str]] = []

        def fake_run_git(target_repo: Path, args: list[str], *, check: bool = False, env=None):
            calls.append(list(args))
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        def fake_ref_exists(_repo: Path, ref: str) -> bool:
            return ref in {
                "refs/remotes/origin/feature/order-test",
                "refs/heads/feature/order-test",
            }

        with patch("bot._run_git", side_effect=fake_run_git):
            with patch("bot._git_ref_exists", side_effect=fake_ref_exists):
                with patch("bot._git_remote_branch_exists", return_value=False):
                    ok, msg = bot._git_ensure_branch_from_main(
                        repo,
                        "feature/order-test",
                        default_branch="main",
                    )

        self.assertTrue(ok)
        self.assertEqual(msg, "published_local_branch")
        self.assertIn(
            ["push", "origin", "refs/heads/feature/order-test:refs/heads/feature/order-test"],
            calls,
        )

    def test_sync_worktree_to_order_branch_uses_local_branch_when_remote_fetch_has_only_stale_origin_ref(self) -> None:
        base_repo = Path("/tmp/factory-repo")
        worktree_dir = Path("/tmp/factory-repo/worker")
        calls: list[tuple[Path, list[str]]] = []

        def fake_run_git(target_repo: Path, args: list[str], *, check: bool = False, env=None):
            calls.append((target_repo, list(args)))
            if args[:2] == ["reset", "--hard"]:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        def fake_remote_ref(target_repo: Path, branch: str) -> str:
            if target_repo == worktree_dir:
                return "origin/feature/order-test"
            return ""

        def fake_local_ref(target_repo: Path, branch: str) -> str:
            if target_repo == base_repo:
                return "refs/heads/feature/order-test"
            return ""

        with patch("bot._git_ensure_branch_from_main", return_value=(True, "published_local_branch")):
            with patch("bot._git_fetch_remote_branch", return_value=(False, "fatal: couldn't find remote ref feature/order-test")):
                with patch("bot._git_remote_branch_ref", side_effect=fake_remote_ref):
                    with patch("bot._git_local_branch_ref", side_effect=fake_local_ref):
                        with patch("bot._run_git", side_effect=fake_run_git):
                            ok, msg = bot._sync_worktree_to_order_branch(
                                base_repo=base_repo,
                                worktree_dir=worktree_dir,
                                order_branch="feature/order-test",
                                default_branch="main",
                            )

        self.assertTrue(ok)
        self.assertEqual(msg, "synced")
        self.assertIn(
            (worktree_dir, ["reset", "--hard", "refs/heads/feature/order-test"]),
            calls,
        )

    def test_sync_repo_checkout_prefers_managed_deploy_worktree(self) -> None:
        repo = Path("/tmp/factory-repo")
        deploy_checkout = repo / "data" / "deploy_worktrees" / "main" / "deploy" / "slot1"
        calls: list[tuple[Path, list[str]]] = []

        def fake_run_git(target_repo: Path, args: list[str], *, check: bool = False):
            calls.append((target_repo, list(args)))
            if args[:2] == ["status", "--porcelain"]:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            if args[:2] == ["merge", "--ff-only"]:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            if args[:3] == ["rev-parse", "--short", "HEAD"]:
                return subprocess.CompletedProcess(args, 0, stdout="af259ae\n", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with patch("bot._git_fetch_remote_branch", return_value=(True, "ok")):
            with patch("bot._prepare_managed_deploy_checkout", return_value=(True, "deploy_checkout_ready", deploy_checkout)):
                with patch("bot._run_git", side_effect=fake_run_git):
                    ok, msg, commit, synced_dir = bot._sync_repo_checkout_to_default_branch(repo=repo, default_branch="main")

        self.assertTrue(ok)
        self.assertEqual(msg, "synced")
        self.assertEqual(commit, "af259ae")
        self.assertEqual(synced_dir, deploy_checkout)
        self.assertTrue(any(target == deploy_checkout and args[:2] == ["status", "--porcelain"] for target, args in calls))
        self.assertFalse(any(target == repo and args[:2] == ["status", "--porcelain"] for target, args in calls))

    def test_deploy_after_order_merge_rewrites_repo_relative_cwd_to_synced_checkout(self) -> None:
        cfg = SimpleNamespace()
        repo_dir = Path("/tmp/factory-repo").resolve()
        deploy_repo_dir = (repo_dir / "data" / "deploy_worktrees" / "main" / "deploy" / "slot1").resolve()
        observed_cwds: list[str] = []

        def fake_run(cmd, *, cwd, env, stdout, stderr, text, timeout, check):
            observed_cwds.append(str(cwd))
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

        with patch(
            "bot._repo_deploy_policy",
            return_value={
                "cwd": str(repo_dir / "deploy"),
                "command": ["./deploy.sh"],
                "verify_command": [],
                "background": False,
                "service": "",
                "timeout_seconds": 60,
            },
        ):
            with patch("bot._sync_repo_checkout_to_default_branch", return_value=(True, "synced", "af259ae", deploy_repo_dir)):
                with patch("subprocess.run", side_effect=fake_run):
                    result = bot._deploy_after_order_merge(
                        cfg=cfg,
                        repo_record={"repo_id": "codexbot-6fb8d5b9"},
                        repo_dir=repo_dir,
                        default_branch="main",
                        order_id="533df86f-8895-4685-8f33-287375e01121",
                        order_branch="feature/order-533df86f-proactive-sprint-codexbot-reliability-",
                        merge_commit="af259ae",
                    )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(observed_cwds, [str((deploy_repo_dir / "deploy").resolve())])

class TestVoiceNormalization(unittest.TestCase):
    def test_normalize_tts_strips_sender_prefix(self) -> None:
        out = bot._normalize_tts_speak_text("Jarvis: merge completed", backend="piper")
        self.assertFalse(out.lower().startswith("jarvis:"))
        self.assertIn("merch", out.lower())

    def test_normalize_tts_keeps_mixed_terms_readable(self) -> None:
        out = bot._normalize_tts_speak_text("rebase y merge en branch main", backend="piper")
        self.assertIn("rei beis", out.lower())
        self.assertIn("merch", out.lower())


class TestRunningWatchdogEnvParsing(unittest.TestCase):
    def test_watchdog_env_invalid_values_fall_back_to_default(self) -> None:
        default = 240.0

        for raw in ("nan", "inf", "-inf", "not-a-number"):
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {"BOT_LOCAL_RUNNING_WATCHDOG_SILENT_SECONDS": raw}, clear=True):
                    self.assertEqual(
                        bot._parse_finite_float_env("BOT_LOCAL_RUNNING_WATCHDOG_SILENT_SECONDS", default),
                        default,
                    )
