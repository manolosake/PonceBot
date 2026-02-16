from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

import bot
from orchestrator.agents import load_agent_profiles
from orchestrator.queue import OrchestratorQueue
from orchestrator.runner import run_task
from orchestrator.runbooks import load_runbooks
from orchestrator.screenshot import validate_screenshot_url
from orchestrator.schemas.result import TaskResult
from orchestrator.schemas.task import Task
from orchestrator.storage import SQLiteTaskStorage
import threading
import time as _time


def _cfg(state_file: Path, workdir: Path | None = None) -> bot.BotConfig:
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
        whispercpp_bin="whispercpp",
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
        codex_workdir=(workdir or Path(".").resolve()),
        codex_timeout_seconds=1,
        codex_use_oss=False,
        codex_local_provider="ollama",
        codex_oss_model="qwen2.5-coder:7b",
        codex_openai_model="gpt-5.2",
        codex_default_mode="ro",
        codex_force_full_access=False,
        codex_dangerous_bypass_sandbox=False,
    )


class TestOrchestratorCommands(unittest.TestCase):
    def test_parse_orchestrator_commands(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td) / "state.json")
            msg_agents = bot.IncomingMessage(1, 1, 2, 10, "u", "/agents")
            resp_agents, _ = bot._parse_job(cfg, msg_agents)
            self.assertEqual(resp_agents, "__orch_agents__")

            msg_pause = bot.IncomingMessage(2, 1, 2, 11, "u", "/pause backend")
            resp_pause, _ = bot._parse_job(cfg, msg_pause)
            self.assertEqual(resp_pause, "__orch_pause:backend")

            msg_resume = bot.IncomingMessage(3, 1, 2, 12, "u", "/resume backend")
            resp_resume, _ = bot._parse_job(cfg, msg_resume)
            self.assertEqual(resp_resume, "__orch_resume:backend")

            msg_cancel = bot.IncomingMessage(4, 1, 2, 13, "u", "/cancel 8f9c")
            resp_cancel, _ = bot._parse_job(cfg, msg_cancel)
            self.assertEqual(resp_cancel, "__orch_cancel_job:8f9c")

            msg_brief = bot.IncomingMessage(5, 1, 2, 14, "u", "/brief")
            resp_brief, _ = bot._parse_job(cfg, msg_brief)
            self.assertEqual(resp_brief, "__orch_brief__")

            msg_snapshot = bot.IncomingMessage(6, 1, 2, 15, "u", "/snapshot https://example.com")
            resp_snapshot, job_snapshot = bot._parse_job(cfg, msg_snapshot)
            self.assertEqual(resp_snapshot, "")
            self.assertIsNotNone(job_snapshot)
            assert job_snapshot is not None
            self.assertEqual(job_snapshot.mode_hint, "rw")
            self.assertIn("@frontend", job_snapshot.user_text)


class TestOrchestratorMarkerResponse(unittest.TestCase):
    def test_invalid_role_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td) / "state.json")
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"backend": {"role": "backend"}})  # type: ignore[arg-type]

            class API:
                def __init__(self) -> None:
                    self.messages: list[tuple[int, str]] = []

                def send_message(
                    self, chat_id: int, text: str, *, reply_to_message_id: int | None = None
                ) -> None:
                    self.messages.append((chat_id, text))

            api = API()
            handled = bot._send_orchestrator_marker_response(
                kind="pause",
                payload="invalid",
                cfg=cfg,
                api=api,  # type: ignore[arg-type]
                chat_id=1,
                reply_to_message_id=None,
                orch_q=q,
                profiles={"backend": {"role": "backend"}},
            )
            self.assertTrue(handled)
            self.assertEqual(len(api.messages), 1)
            self.assertIn("Rol invalido", api.messages[0][1])


class DummyExecutor:
    def __init__(self, value: object) -> None:
        self.value = value

    def run_task(self, _task: Task) -> object:
        return self.value


class TestOrchestratorRunner(unittest.TestCase):
    def test_blocked_task_requires_approval(self) -> None:
        task = Task.new(
            source="telegram",
            role="backend",
            input_text="deploy prod",
            request_type="task",
            priority=1,
            model="gpt-5.2",
            effort="medium",
            mode_hint="full",
            requires_approval=True,
            max_cost_window_usd=8.0,
            chat_id=1,
            state="queued",
        )
        result = run_task(task, executor=DummyExecutor({"summary": "should not run", "status": "ok"}))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.next_action, "approve")

    def test_run_task_coerces_dict_result(self) -> None:
        task = Task.new(
            source="telegram",
            role="backend",
            input_text="echo hi",
            request_type="task",
            priority=1,
            model="gpt-5.2",
            effort="low",
            mode_hint="ro",
            requires_approval=False,
            max_cost_window_usd=8.0,
            chat_id=1,
            state="queued",
        )
        result = run_task(task, executor=DummyExecutor({"status": "ok", "summary": "done", "artifacts": ["a.png"]}))
        self.assertIsInstance(result, TaskResult)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.artifacts, ["a.png"])


class TestOrchestratorStorage(unittest.TestCase):
    def test_approve_does_not_deadlock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            job_id = q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="deploy",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="full",
                    requires_approval=True,
                    max_cost_window_usd=8.0,
                    chat_id=1,
                    state="queued",
                )
            )

            done: list[bool] = []

            def _approve() -> None:
                ok = q.set_job_approved(job_id)
                done.append(ok)

            t = threading.Thread(target=_approve, daemon=True)
            t.start()
            t.join(timeout=1.0)
            self.assertFalse(t.is_alive(), "approve deadlocked")
            self.assertEqual(done, [True])

    def test_role_pause_resumes_and_blocks_queue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            storage = SQLiteTaskStorage(db_path)
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            task = Task.new(
                source="telegram",
                role="backend",
                input_text="build ui",
                request_type="task",
                priority=2,
                model="gpt-5.2",
                effort="medium",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=8.0,
                chat_id=123,
                state="queued",
            )
            qid = q.submit_task(task)
            task = q.get_job(qid)
            self.assertIsNotNone(task)
            assert task is not None

            q.pause_role("backend")
            self.assertIsNone(q.take_next())

            q.resume_role("backend")
            taken = q.take_next()
            self.assertIsNotNone(taken)
            assert taken is not None
            self.assertEqual(taken.job_id, task.job_id)

    def test_claim_skips_paused_role_and_takes_other(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            q.pause_role("backend")
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="backend work",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=8.0,
                    chat_id=1,
                    state="queued",
                )
            )
            frontend_id = q.submit_task(
                Task.new(
                    source="telegram",
                    role="frontend",
                    input_text="frontend work",
                    request_type="task",
                    priority=2,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=8.0,
                    chat_id=1,
                    state="queued",
                )
            )
            taken = q.take_next()
            self.assertIsNotNone(taken)
            assert taken is not None
            self.assertEqual(taken.job_id, frontend_id)

    def test_cancel_running_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            t1 = q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="one",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=8.0,
                    chat_id=1,
                    state="running",
                )
            )
            t2 = q.submit_task(
                Task.new(
                    source="telegram",
                    role="frontend",
                    input_text="two",
                    request_type="task",
                    priority=1,
                    model="gpt-5.2",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=8.0,
                    chat_id=2,
                    state="running",
                )
            )

            self.assertTrue(storage.update_state(t1, "running"))
            self.assertTrue(storage.update_state(t2, "running"))
            canceled = q.cancel_running_jobs()
            self.assertEqual(canceled, 2)

            for jid in (t1, t2):
                row = q.get_job(jid)
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(row.state, "cancelled")

    def test_recover_clears_workspace_leases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            slot = q.lease_workspace(role="backend", job_id="deadbeef", slots=1)
            self.assertEqual(slot, 1)
            self.assertEqual(q.get_workspace_lease(job_id="deadbeef"), ("backend", 1))

            recovered = q.recover_stale_running()
            self.assertEqual(recovered, 0)
            self.assertIsNone(q.get_workspace_lease(job_id="deadbeef"))


class TestYamlLikeParsing(unittest.TestCase):
    def test_agents_yaml_parses_lists(self) -> None:
        profiles = load_agent_profiles(Path(__file__).resolve().parent / "orchestrator" / "agents.yaml")
        self.assertIn("frontend", profiles)
        fe = profiles["frontend"]
        tools = fe.get("allowed_tools")
        self.assertIsInstance(tools, list)
        assert isinstance(tools, list)
        self.assertIn("screenshot", [str(x) for x in tools])

    def test_runbooks_yaml_parses_multiline_prompt(self) -> None:
        rbs = load_runbooks(Path(__file__).resolve().parent / "orchestrator" / "runbooks.yaml")
        self.assertGreaterEqual(len(rbs), 1)
        rb = rbs[0]
        self.assertNotEqual(rb.prompt.strip(), "|")
        self.assertIn("Revisa", rb.prompt)


class TestScreenshotUrlValidation(unittest.TestCase):
    def test_blocks_non_http_schemes(self) -> None:
        chk = validate_screenshot_url("file:///etc/passwd")
        self.assertFalse(chk.ok)
        self.assertFalse(chk.overrideable)
        self.assertIn("blocked scheme", chk.reason)

    def test_blocks_private_ip_by_default_and_allows_with_override(self) -> None:
        chk = validate_screenshot_url("http://127.0.0.1")
        self.assertFalse(chk.ok)
        self.assertTrue(chk.overrideable)
        chk2 = validate_screenshot_url("http://127.0.0.1", allow_private=True)
        self.assertTrue(chk2.ok)

    def test_blocks_hostname_resolving_private(self) -> None:
        def _resolver(_host: str, _port: int) -> list[object]:
            # Minimal getaddrinfo-like shape: tuple with [4][0] as ip string.
            return [(None, None, None, None, ("127.0.0.1", 80))]

        chk = validate_screenshot_url("https://example.test", resolver=_resolver)
        self.assertFalse(chk.ok)
        self.assertTrue(chk.overrideable)
        self.assertIn("blocked ip", chk.reason)

    def test_allowlist_requires_exact_host_unless_approved(self) -> None:
        chk = validate_screenshot_url("https://notexample.com", allowed_hosts={"example.com"})
        self.assertFalse(chk.ok)
        self.assertTrue(chk.overrideable)
        def _resolver(_host: str, _port: int) -> list[object]:
            return [(None, None, None, None, ("93.184.216.34", 443))]

        chk2 = validate_screenshot_url(
            "https://notexample.com",
            allowed_hosts={"example.com"},
            allow_private=True,
            resolver=_resolver,
        )
        self.assertTrue(chk2.ok)


class TestOrchestratorDependencyGating(unittest.TestCase):
    def test_wrapup_allows_terminal_dependency_states(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            dep_failed = q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="dep failed",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="failed",
                )
            )
            dep_cancel = q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="dep cancelled",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="cancelled",
                )
            )
            wrap = Task.new(
                source="telegram",
                role="orchestrator",
                input_text="wrap up",
                request_type="review",
                priority=2,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
                depends_on=[dep_failed, dep_cancel],
                trace={"wrapup_for": "ticket123"},
            )
            wrap_id = q.submit_task(wrap)
            taken = q.take_next()
            self.assertIsNotNone(taken)
            assert taken is not None
            self.assertEqual(taken.job_id, wrap_id)

    def test_non_wrapup_requires_done_dependencies_and_defers_due_at(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            dep = q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="dep queued",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="running",
                )
            )
            blocked = Task.new(
                source="telegram",
                role="backend",
                input_text="blocked by dep",
                request_type="task",
                priority=2,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
                depends_on=[dep],
            )
            jid = q.submit_task(blocked)
            now = _time.time()
            taken = q.take_next()
            self.assertIsNone(taken)
            refreshed = q.get_job(jid)
            self.assertIsNotNone(refreshed)
            assert refreshed is not None
            self.assertIsNotNone(refreshed.due_at)
            assert refreshed.due_at is not None
            self.assertGreater(refreshed.due_at, now)


class TestRetryScheduling(unittest.TestCase):
    def test_bump_retry_moves_job_to_queued_and_increments_counter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            t = Task.new(
                source="telegram",
                role="backend",
                input_text="unstable",
                request_type="task",
                priority=1,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="running",
                retry_count=0,
                max_retries=2,
            )
            jid = q.submit_task(t)
            due = _time.time() + 60.0
            ok = q.bump_retry(jid, due_at=due, error="boom")
            self.assertTrue(ok)
            refreshed = q.get_job(jid)
            self.assertIsNotNone(refreshed)
            assert refreshed is not None
            self.assertEqual(refreshed.state, "queued")
            self.assertEqual(refreshed.retry_count, 1)
            self.assertIsNotNone(refreshed.due_at)
