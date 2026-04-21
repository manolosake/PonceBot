from __future__ import annotations
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bot
from orchestrator.agents import load_agent_profiles
from orchestrator.delegation import parse_orchestrator_subtasks
from orchestrator.queue import OrchestratorQueue
from orchestrator.runner import run_task
from orchestrator.runbooks import load_runbooks
from orchestrator.screenshot import validate_screenshot_url
from orchestrator.schemas.result import TaskResult
from orchestrator.schemas.task import Task
from orchestrator.storage import SQLiteTaskStorage
from orchestrator.dispatcher import detect_ceo_intent, detect_request_type
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
        admin_user_ids=frozenset(),
        admin_chat_ids=frozenset(),
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

            msg_pause_autonomy = bot.IncomingMessage(21, 1, 2, 31, "u", "/pause_autonomy")
            resp_pause_autonomy, _ = bot._parse_job(cfg, msg_pause_autonomy)
            self.assertEqual(resp_pause_autonomy, "__orch_pause_autonomy__")

            msg_resume = bot.IncomingMessage(3, 1, 2, 12, "u", "/resume backend")
            resp_resume, _ = bot._parse_job(cfg, msg_resume)
            self.assertEqual(resp_resume, "__orch_resume:backend")

            msg_resume_autonomy = bot.IncomingMessage(22, 1, 2, 32, "u", "/resume_autonomy")
            resp_resume_autonomy, _ = bot._parse_job(cfg, msg_resume_autonomy)
            self.assertEqual(resp_resume_autonomy, "__orch_resume_autonomy__")

            msg_cancel = bot.IncomingMessage(4, 1, 2, 13, "u", "/cancel 8f9c")
            resp_cancel, _ = bot._parse_job(cfg, msg_cancel)
            self.assertEqual(resp_cancel, "__orch_cancel_job:8f9c")

            msg_purge = bot.IncomingMessage(5, 1, 2, 14, "u", "/purge")
            resp_purge, _ = bot._parse_job(cfg, msg_purge)
            self.assertEqual(resp_purge, "__orch_purge_queue:global")

            msg_purge_nl = bot.IncomingMessage(6, 1, 2, 15, "u", "Vamos a limpiar la cola, que no haya tareas.")
            resp_purge_nl, job_purge_nl = bot._parse_job(cfg, msg_purge_nl)
            self.assertEqual(resp_purge_nl, "")
            self.assertIsNotNone(job_purge_nl)
            assert job_purge_nl is not None
            self.assertEqual(job_purge_nl.argv, ["exec", "Vamos a limpiar la cola, que no haya tareas."])

            msg_job_list = bot.IncomingMessage(7, 1, 2, 16, "u", "/job")
            resp_job_list, _ = bot._parse_job(cfg, msg_job_list)
            self.assertEqual(resp_job_list, "__orch_job:list")

            msg_job_show = bot.IncomingMessage(8, 1, 2, 17, "u", "/job show 8f9c")
            resp_job_show, _ = bot._parse_job(cfg, msg_job_show)
            self.assertEqual(resp_job_show, "__orch_job:show 8f9c")

            msg_job_del = bot.IncomingMessage(9, 1, 2, 18, "u", "/job del 8f9c")
            resp_job_del, _ = bot._parse_job(cfg, msg_job_del)
            self.assertEqual(resp_job_del, "__orch_job:del 8f9c")

            msg_brief = bot.IncomingMessage(10, 1, 2, 19, "u", "/brief")
            resp_brief, _ = bot._parse_job(cfg, msg_brief)
            self.assertEqual(resp_brief, "__orch_brief__")

            msg_snapshot = bot.IncomingMessage(11, 1, 2, 20, "u", "/snapshot https://example.com")
            resp_snapshot, job_snapshot = bot._parse_job(cfg, msg_snapshot)
            self.assertEqual(resp_snapshot, "")
            self.assertIsNotNone(job_snapshot)
            assert job_snapshot is not None
            self.assertEqual(job_snapshot.mode_hint, "rw")
            self.assertIn("@frontend", job_snapshot.user_text)


class TestRequestTypeDetection(unittest.TestCase):
    def test_fix_status_string_is_a_task_not_status(self) -> None:
        self.assertEqual(detect_request_type("corrige lo que dice job role status"), "task")

    def test_plain_server_status_is_status(self) -> None:
        self.assertEqual(detect_request_type("quiero saber el estado del servidor"), "status")

    def test_multi_intent_with_action_wins_over_status(self) -> None:
        self.assertEqual(detect_request_type("quiero saber el estado del servidor y agrega mas detalles"), "task")

    def test_still_running_is_status(self) -> None:
        self.assertEqual(detect_request_type("estan trabajando?"), "status")

    def test_what_is_the_team_doing_is_status(self) -> None:
        self.assertEqual(detect_request_type("Que está haciendo el equipo?"), "status")

    def test_whats_pending_is_query(self) -> None:
        self.assertEqual(detect_request_type("que tienes pendiente jarvis?"), "query")


    def test_projects_in_progress_question_is_query(self) -> None:
        self.assertEqual(detect_request_type("Que proyectos tienes ahorita en progreso?"), "query")

    def test_spanish_update_question_is_task(self) -> None:
        self.assertEqual(detect_request_type("podrias actualizar el dashboard?"), "task")

    def test_conversational_creo_que_is_query(self) -> None:
        self.assertEqual(detect_request_type("creo que el servidor esta bien"), "query")


class TestCeoIntentDetection(unittest.TestCase):
    def test_ideation_prompt_is_query(self) -> None:
        self.assertEqual(
            detect_ceo_intent("ok, dame 15 proyectos interesantes que podemos hacer"),
            "query",
        )

    def test_explicit_execution_project_is_new_order(self) -> None:
        self.assertEqual(
            detect_ceo_intent("quiero que crees un proyecto backend nuevo y lo despliegues"),
            "order_project_new",
        )

    def test_reply_defaults_to_change(self) -> None:
        self.assertEqual(
            detect_ceo_intent("ajusta eso por favor", reply_context={"reply_to_message_id": 123}),
            "order_project_change",
        )

    def test_stop_all_prompt_maps_to_control_stop_all(self) -> None:
        self.assertEqual(detect_ceo_intent("paren todo ahora mismo"), "control_stop_all")


class TestDelegationParsing(unittest.TestCase):
    def test_orchestrator_subtasks_mode_hint_is_optional(self) -> None:
        specs = parse_orchestrator_subtasks({"subtasks": [{"key": "a", "role": "backend", "text": "do the thing"}]})
        self.assertEqual(len(specs), 1)
        # Missing mode_hint => empty string => caller applies role profile default.
        self.assertEqual(specs[0].mode_hint, "")

        specs2 = parse_orchestrator_subtasks(
            {"subtasks": [{"key": "a", "role": "backend", "text": "do the thing", "mode_hint": "ro"}]}
        )
        self.assertEqual(specs2[0].mode_hint, "ro")

        specs3 = parse_orchestrator_subtasks(
            {"subtasks": [{"key": "a", "role": "backend", "text": "do the thing", "mode_hint": "nope"}]}
        )
        # Invalid but explicit => fall back to a safe value.
        self.assertEqual(specs3[0].mode_hint, "ro")

    def test_orchestrator_subtasks_contract_fields_are_parsed(self) -> None:
        specs = parse_orchestrator_subtasks(
            {
                "subtasks": [
                    {
                        "key": "ship_api",
                        "role": "backend",
                        "text": "Implement endpoint",
                        "acceptance_criteria": ["returns 200", "includes schema validation"],
                        "definition_of_done": ["tests pass", "docs updated"],
                        "eta_minutes": 95,
                        "sla_tier": "high",
                    }
                ]
            }
        )
        self.assertEqual(len(specs), 1)
        s = specs[0]
        self.assertEqual(s.acceptance_criteria, ["returns 200", "includes schema validation"])
        self.assertEqual(s.definition_of_done, ["tests pass", "docs updated"])
        self.assertEqual(int(s.eta_minutes or 0), 95)
        self.assertEqual(s.sla_tier, "high")


class TestOrchestratorEvidenceGate(unittest.TestCase):
    def test_backend_requires_meaningful_evidence(self) -> None:
        t = Task.new(
            source="telegram",
            role="backend",
            input_text="Do backend work",
            request_type="task",
            priority=1,
            model="gpt-5.3-codex",
            effort="medium",
            mode_hint="rw",
            requires_approval=False,
            max_cost_window_usd=5.0,
            chat_id=1,
        )
        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=t,
            summary="ok",
            artifacts=[],
            logs="",
            structured={},
        )
        self.assertFalse(ok)
        self.assertIn("Evidence gate", str(reason))
        self.assertEqual(int(meta.get("artifacts_count") or 0), 0)

    def test_qa_requires_pass_fail_style_language(self) -> None:
        t = Task.new(
            source="telegram",
            role="qa",
            input_text="Run tests",
            request_type="task",
            priority=1,
            model="gpt-5.3-codex",
            effort="medium",
            mode_hint="rw",
            requires_approval=False,
            max_cost_window_usd=5.0,
            chat_id=1,
        )
        ok, reason, _meta = bot._orchestrator_min_evidence_gate(
            task=t,
            summary="Checked things quickly.",
            artifacts=[],
            logs="",
            structured={},
        )
        self.assertFalse(ok)
        self.assertIn("QA evidence gate", str(reason))


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
                user_id=2,
                reply_to_message_id=None,
                orch_q=q,
                profiles={"backend": {"role": "backend"}},
            )
            self.assertTrue(handled)
            self.assertEqual(len(api.messages), 1)
            self.assertIn("Rol invalido", api.messages[0][1])

    def test_pause_autonomy_cancels_autonomous_lane_and_sets_manual_pause(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td) / "state.json")
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"implementer_local": {"role": "implementer_local"}})  # type: ignore[arg-type]

            root_auto = Task.new(
                source="telegram",
                role="skynet",
                input_text="auto lane",
                request_type="task",
                priority=1,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="running",
                is_autonomous=True,
            ).with_updates(job_id="1" * 36)
            child_auto = Task.new(
                source="telegram",
                role="implementer_local",
                input_text="local child",
                request_type="task",
                priority=1,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
                parent_job_id="1" * 36,
            ).with_updates(job_id="2" * 36)
            manual_root = Task.new(
                source="telegram",
                role="jarvis",
                input_text="manual lane",
                request_type="task",
                priority=1,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
                is_autonomous=False,
            ).with_updates(job_id="3" * 36)

            storage.submit_task(root_auto)
            storage.submit_task(child_auto)
            storage.submit_task(manual_root)

            class API:
                def __init__(self) -> None:
                    self.messages: list[tuple[int, str]] = []

                def send_message(
                    self, chat_id: int, text: str, *, reply_to_message_id: int | None = None
                ) -> None:
                    self.messages.append((chat_id, text))

            api = API()
            handled = bot._send_orchestrator_marker_response(
                kind="pause_autonomy",
                payload="",
                cfg=cfg,
                api=api,  # type: ignore[arg-type]
                chat_id=1,
                user_id=2,
                reply_to_message_id=None,
                orch_q=q,
                profiles={"implementer_local": {"role": "implementer_local"}},
            )
            self.assertTrue(handled)
            self.assertEqual(storage.get_job("1" * 36).state, "cancelled")
            self.assertEqual(storage.get_job("2" * 36).state, "cancelled")
            self.assertEqual(storage.get_job("3" * 36).state, "queued")
            state = bot._proactive_lane_state(cfg)
            self.assertTrue(bool(state.get("paused", False)))
            self.assertTrue(bool(state.get("manual_pause", False)))
            self.assertIn("Autonomia pausada", api.messages[0][1])


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

    def test_delete_job_removes_row(self) -> None:
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
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=8.0,
                    chat_id=1,
                    state="queued",
                )
            )
            self.assertIsNotNone(q.get_job(job_id))
            ok = q.delete_job(job_id[:8])
            self.assertTrue(ok)
            self.assertIsNone(q.get_job(job_id))

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

    def test_job_id_prefix_resolves_for_get_and_approve(self) -> None:
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
            prefix = job_id[:8]
            t = q.get_job(prefix)
            self.assertIsNotNone(t)
            assert t is not None
            self.assertEqual(t.job_id, job_id)
            ok = q.set_job_approved(prefix)
            self.assertTrue(ok)

    def test_update_trace_does_not_emit_job_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            storage = SQLiteTaskStorage(db_path)
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            job_id = q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="do thing",
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

            with sqlite3.connect(db_path) as conn:
                before = int(conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0])

            ok = q.update_trace(job_id, live_phase="running", live_stdout_tail="hello")
            self.assertTrue(ok)
            task = q.get_job(job_id)
            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual(str((task.trace or {}).get("live_phase")), "running")

            with sqlite3.connect(db_path) as conn:
                after = int(conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0])

            self.assertEqual(before, after)

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

    def test_sync_order_phase_marks_ready_for_merge_when_wrapup_done(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-12345678"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Test order",
                body="body",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="planning",
                project_id="proj-1",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="subtask",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    labels={"ticket": order_id, "kind": "subtask", "key": "impl"},
                    state="done",
                    job_id="job-subtask",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="jarvis",
                    input_text="wrap",
                    request_type="review",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    labels={"ticket": order_id, "kind": "wrapup"},
                    state="done",
                    job_id="job-wrapup",
                )
            )
            bot._sync_order_phase_from_runtime(orch_q=q, root_ticket=order_id, chat_id=1)
            got = q.get_order(order_id, chat_id=1)
            assert got is not None
            self.assertEqual(str(got.get("phase")), "ready_for_merge")
            self.assertEqual(str(got.get("status")), "active")

    def test_sync_order_phase_marks_done_when_wrapup_done_and_merged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-merged-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Merged order",
                body="body",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="planning",
                project_id="proj-1",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="jarvis",
                    input_text="root",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={"merged_to_main": True},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="jarvis",
                    input_text="wrap",
                    request_type="review",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    labels={"ticket": order_id, "kind": "wrapup"},
                    state="done",
                    job_id="job-wrapup",
                )
            )
            bot._sync_order_phase_from_runtime(orch_q=q, root_ticket=order_id, chat_id=1)
            got = q.get_order(order_id, chat_id=1)
            assert got is not None
            self.assertEqual(str(got.get("phase")), "done")
            self.assertEqual(str(got.get("status")), "done")

    def test_sync_order_phase_closes_proactive_order_on_blocked_with_root_cause(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-proactive-blocked-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: Reliability",
                body="AUTONOMOUS PROACTIVE SPRINT\n[proactive:poncebot-core]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="review",
                project_id="proj-1",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={"proactive_lane": True},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="implementer_local",
                    input_text="impl",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="failed",
                    job_id="job-impl-failed",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="final sweep",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="done",
                    trace={
                        "result_summary": (
                            "NO-GO to remain active. "
                            "Close this order as BLOCKED_WITH_ROOT_CAUSE because repeated local retries produced no validated improvement."
                        )
                    },
                    labels={"ticket": order_id, "kind": "final_sweep"},
                    job_id="job-skynet-final",
                )
            )

            bot._sync_order_phase_from_runtime(orch_q=q, root_ticket=order_id, chat_id=1)

            got = q.get_order(order_id, chat_id=1)
            assert got is not None
            self.assertEqual(str(got.get("phase")), "done")
            self.assertEqual(str(got.get("status")), "done")

            root = q.get_job(order_id)
            assert root is not None
            self.assertTrue(bool((root.trace or {}).get("proactive_blocked_with_root_cause", False)))

    def test_sync_order_phase_does_not_close_proactive_order_on_go_or_negated_close_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-proactive-go-no-close-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: Reliability",
                body="AUTONOMOUS PROACTIVE SPRINT\n[proactive:poncebot-core]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="review",
                project_id="proj-1",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={"proactive_lane": True},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="final sweep",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="done",
                    trace={
                        "result_summary": (
                            "GO to remain active. Do not close this order yet; "
                            "continue delegated remediation and QA+reviewer gates."
                        )
                    },
                    labels={"ticket": order_id, "kind": "final_sweep"},
                    job_id="job-skynet-final-go",
                )
            )

            bot._sync_order_phase_from_runtime(orch_q=q, root_ticket=order_id, chat_id=1)

            got = q.get_order(order_id, chat_id=1)
            assert got is not None
            self.assertEqual(str(got.get("status")), "active")
            self.assertNotEqual(str(got.get("phase")), "done")

            root = q.get_job(order_id)
            assert root is not None
            self.assertFalse(bool((root.trace or {}).get("proactive_blocked_with_root_cause", False)))

    def test_set_order_phase_accepts_ready_for_merge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-rfm-001"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Order",
                body="body",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="planning",
                project_id="proj-1",
            )
            ok = q.set_order_phase(order_id, chat_id=1, phase="ready_for_merge")
            self.assertTrue(ok)
            got = q.get_order(order_id, chat_id=1)
            assert got is not None
            self.assertEqual(str(got.get("phase")), "ready_for_merge")


class TestOrderBranchPolicy(unittest.TestCase):
    def test_order_branch_name_uses_feature_prefix(self) -> None:
        b = bot._order_branch_name("abcd1234-ffff", "Nuevo proyecto server core")
        self.assertTrue(b.startswith("feature/order-abcd1234-"))

    def test_resolve_order_branch_from_parent_trace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            root_id = "ord-branch-001"
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="jarvis",
                    input_text="root",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={"order_branch": "feature/order-ord-bran-root"},
                    job_id=root_id,
                )
            )
            child_id = q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="child",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=root_id,
                    state="queued",
                )
            )
            child = q.get_job(child_id)
            self.assertIsNotNone(child)
            assert child is not None
            self.assertEqual(
                bot._resolve_order_branch_from_task(child, q),
                "feature/order-ord-bran-root",
            )


class TestFinalSweepGuard(unittest.TestCase):
    def test_final_sweep_enqueues_when_active_order_has_no_children_and_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-no-children-stale"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Manual order",
                body="body",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="review",
                project_id="proj-1",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="jarvis",
                    input_text="root",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    job_id=order_id,
                )
            )

            now = _time.time()
            stale_ts = now - 600.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
                    (float(stale_ts), float(stale_ts), order_id),
                )
                conn.commit()

            with patch.dict(
                os.environ,
                {
                    "BOT_JARVIS_IDLE_ORDER_STALE_SECONDS": "120",
                    "BOT_JARVIS_FINAL_SWEEP_COOLDOWN_SECONDS": "600",
                },
                clear=False,
            ):
                created = bot._jarvis_final_sweep_tick(cfg=cfg, orch_q=q, profiles=None, now=now)

            self.assertEqual(created, 1)
            children = q.jobs_by_parent(parent_job_id=order_id, limit=50)
            sweep_jobs = [j for j in children if str((j.labels or {}).get("kind") or "").strip().lower() == "final_sweep"]
            self.assertEqual(len(sweep_jobs), 1)
            sweep = sweep_jobs[0]
            self.assertEqual(str(sweep.state or ""), "queued")
            self.assertEqual(str((sweep.trace or {}).get("final_sweep_reason") or ""), "idle_no_open_jobs")
            self.assertIn("Do not leave this order active with zero live jobs.", str(sweep.input_text or ""))

    def test_final_sweep_does_not_enqueue_when_no_children_but_root_is_recent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-no-children-fresh"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Manual order",
                body="body",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="review",
                project_id="proj-1",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="jarvis",
                    input_text="root",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    job_id=order_id,
                )
            )

            now = _time.time()
            recent_ts = now - 30.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
                    (float(recent_ts), float(recent_ts), order_id),
                )
                conn.commit()

            with patch.dict(
                os.environ,
                {
                    "BOT_JARVIS_IDLE_ORDER_STALE_SECONDS": "300",
                    "BOT_JARVIS_FINAL_SWEEP_COOLDOWN_SECONDS": "600",
                },
                clear=False,
            ):
                created = bot._jarvis_final_sweep_tick(cfg=cfg, orch_q=q, profiles=None, now=now)

            self.assertEqual(created, 0)
            children = q.jobs_by_parent(parent_job_id=order_id, limit=50)
            self.assertEqual(children, [])

    def test_final_sweep_auto_closes_proactive_terminal_only_order_after_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-terminal-only-close"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: Reliability",
                body="AUTONOMOUS PROACTIVE SPRINT\n[proactive:poncebot-core]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="review",
                project_id="proj-1",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={"proactive_lane": True},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="implementer_local",
                    input_text="attempt 1",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="failed",
                    job_id="job-terminal-1",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="reviewer_local",
                    input_text="attempt 2",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="done",
                    job_id="job-terminal-2",
                )
            )

            now = _time.time()
            stale_ts = now - 1200.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
                    (float(stale_ts), float(stale_ts), order_id),
                )
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE parent_job_id = ?",
                    (float(stale_ts), float(stale_ts), order_id),
                )
                conn.commit()

            with patch.dict(
                os.environ,
                {
                    "BOT_JARVIS_IDLE_ORDER_STALE_SECONDS": "120",
                    "BOT_JARVIS_IDLE_TERMINAL_CLOSE_THRESHOLD": "2",
                    "BOT_JARVIS_FINAL_SWEEP_COOLDOWN_SECONDS": "1",
                },
                clear=False,
            ):
                created = bot._jarvis_final_sweep_tick(cfg=cfg, orch_q=q, profiles=None, now=now)

            self.assertEqual(created, 0)
            order = q.get_order(order_id, chat_id=1)
            self.assertIsNotNone(order)
            assert order is not None
            self.assertEqual(str(order.get("status") or ""), "done")
            self.assertEqual(str(order.get("phase") or ""), "done")
            root = q.get_job(order_id)
            self.assertIsNotNone(root)
            assert root is not None
            self.assertTrue(bool((root.trace or {}).get("final_sweep_terminal_only_closed", False)))


class TestAutopilotTick(unittest.TestCase):
    def test_autopilot_tick_skips_when_proactive_lane_is_paused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-autopilot-paused"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Paused proactive order",
                body="AUTONOMOUS PROACTIVE SPRINT",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="review",
                project_id="proj-1",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="jarvis",
                    input_text="root",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    job_id=order_id,
                )
            )
            bot._set_proactive_lane_pause(
                cfg,
                paused=True,
                reason="manual_pause_command",
                manual=True,
            )

            created = bot._autopilot_tick(
                cfg=cfg,
                orch_q=q,
                profiles=None,
                chat_id=1,
                now=_time.time(),
            )

            self.assertEqual(created, 0)
            children = q.jobs_by_parent(parent_job_id=order_id, limit=100)
            self.assertEqual(children, [])


class TestRunningWatchdog(unittest.TestCase):
    def test_requeues_silent_local_ollama_before_full_runtime_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            job_id = "job-local-silent"
            artifacts_dir = td_path / "artifacts" / job_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            live_md = artifacts_dir / "local_ollama_live.md"
            live_stream = artifacts_dir / "local_ollama_stream.jsonl"
            live_md.write_text("", encoding="utf-8")
            live_stream.write_text("", encoding="utf-8")

            q.submit_task(
                Task.new(
                    source="telegram",
                    role="implementer_local",
                    input_text="silent local run",
                    request_type="task",
                    priority=1,
                    model="qwen3.5:27b",
                    effort="high",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="running",
                    artifacts_dir=str(artifacts_dir),
                    trace={
                        "live_phase": "local_ollama",
                        "live_stdout_path": str(live_md),
                        "live_stderr_path": str(live_stream),
                        "max_runtime_seconds": 1800,
                    },
                    job_id=job_id,
                )
            )

            now = _time.time()
            stale_ts = now - 250.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
                    (float(stale_ts), float(stale_ts), job_id),
                )
                conn.commit()

            with patch.dict(os.environ, {"BOT_LOCAL_RUNNING_WATCHDOG_SILENT_SECONDS": "180"}, clear=False):
                recovered = bot._running_watchdog_tick(cfg=cfg, orch_q=q, profiles=None, now=now)

            self.assertEqual(recovered, 1)
            job = q.get_job(job_id)
            assert job is not None
            self.assertEqual(job.state, "queued")
            self.assertIn("no local Ollama output", str((job.trace or {}).get("result_summary") or ""))
            self.assertTrue(bool((job.trace or {}).get("running_watchdog_silent_local", False)))

    def test_does_not_requeue_local_ollama_when_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            job_id = "job-local-active"
            artifacts_dir = td_path / "artifacts" / job_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            live_md = artifacts_dir / "local_ollama_live.md"
            live_stream = artifacts_dir / "local_ollama_stream.jsonl"
            live_md.write_text("partial output", encoding="utf-8")
            live_stream.write_text("", encoding="utf-8")

            q.submit_task(
                Task.new(
                    source="telegram",
                    role="implementer_local",
                    input_text="active local run",
                    request_type="task",
                    priority=1,
                    model="qwen3.5:27b",
                    effort="high",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="running",
                    artifacts_dir=str(artifacts_dir),
                    trace={
                        "live_phase": "local_ollama",
                        "live_stdout_path": str(live_md),
                        "live_stderr_path": str(live_stream),
                        "max_runtime_seconds": 1800,
                    },
                    job_id=job_id,
                )
            )

            now = _time.time()
            stale_ts = now - 250.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
                    (float(stale_ts), float(stale_ts), job_id),
                )
                conn.commit()

            with patch.dict(os.environ, {"BOT_LOCAL_RUNNING_WATCHDOG_SILENT_SECONDS": "180"}, clear=False):
                recovered = bot._running_watchdog_tick(cfg=cfg, orch_q=q, profiles=None, now=now)

            self.assertEqual(recovered, 0)
            job = q.get_job(job_id)
            assert job is not None
            self.assertEqual(job.state, "running")


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
        self.assertTrue(rb.prompt.strip())


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

    def test_non_wrapup_requires_done_dependencies_and_moves_to_waiting_deps(self) -> None:
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
            self.assertEqual(refreshed.state, "waiting_deps")
            self.assertIsNone(refreshed.due_at)
            self.assertTrue(str(refreshed.blocked_reason or "").startswith("dependencies_pending"))
            self.assertGreater(float(refreshed.updated_at), now)


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


class TestSendOrchestratorResult(unittest.TestCase):
    def test_skips_empty_artifact_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            empty = Path(td) / "empty.txt"
            empty.write_text("", encoding="utf-8")

            class API:
                def __init__(self) -> None:
                    self.docs: list[Path] = []
                    self.photos: list[Path] = []
                    self.messages: list[str] = []

                def send_message(self, _chat_id: int, text: str, *, reply_to_message_id: int | None = None) -> None:
                    self.messages.append(text)

                def send_document(self, _chat_id: int, path: Path, *, filename: str, reply_to_message_id: int | None = None) -> None:
                    self.docs.append(path)

                def send_photo(self, _chat_id: int, path: Path, *, caption: str, reply_to_message_id: int | None = None) -> None:
                    self.photos.append(path)

            api = API()
            task = Task.new(
                source="telegram",
                role="backend",
                input_text="x",
                request_type="task",
                priority=2,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
            )
            cfg = _cfg(Path(td) / "state.json")
            bot._send_orchestrator_result(  # type: ignore[arg-type]
                api,  # type: ignore[arg-type]
                task,
                {"status": "ok", "summary": "done", "artifacts": [str(empty)]},
                cfg=cfg,
            )
            self.assertGreaterEqual(len(api.messages), 1)
            self.assertEqual(api.docs, [])
            self.assertEqual(api.photos, [])

    def test_summary_uses_humanized_role_header(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            class API:
                def __init__(self) -> None:
                    self.messages: list[str] = []

                def send_message(self, _chat_id: int, text: str, *, reply_to_message_id: int | None = None) -> None:
                    self.messages.append(text)

                def send_document(self, _chat_id: int, path: Path, *, filename: str, reply_to_message_id: int | None = None) -> None:
                    self.messages.append(f"doc:{filename}")

                def send_photo(self, _chat_id: int, path: Path, *, caption: str, reply_to_message_id: int | None = None) -> None:
                    self.messages.append(f"photo:{caption}")

            api = API()
            task = Task.new(
                source="telegram",
                role="jarvis",
                input_text="hola",
                request_type="task",
                priority=2,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="done",
            )
            cfg = _cfg(Path(td) / "state.json")
            bot._send_orchestrator_result(  # type: ignore[arg-type]
                api,  # type: ignore[arg-type]
                task,
                {"status": "ok", "summary": "Resumen de prueba", "artifacts": []},
                cfg=cfg,
            )
            self.assertTrue(api.messages)
            self.assertTrue(api.messages[0].startswith("Jarvis:"))
            self.assertIn("Resumen de prueba", api.messages[0])
            self.assertNotIn("job=", api.messages[0])
            self.assertNotIn("role=", api.messages[0])
            self.assertNotIn("status=", api.messages[0])


class TestCeoOrders(unittest.TestCase):
    def test_upsert_list_get_and_set_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            st = SQLiteTaskStorage(db)

            oid = "11111111-1111-1111-1111-111111111111"
            st.upsert_order(order_id=oid, chat_id=1, title="ExecutiveDashboard MVP", body="Ship the first dashboard", status="active", priority=2)

            rows = st.list_orders(chat_id=1, status="active", limit=10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(str(rows[0]["order_id"]), oid)

            got = st.get_order(oid[:8], chat_id=1)
            self.assertIsNotNone(got)
            assert got is not None
            self.assertEqual(str(got["status"]), "active")

            ok = st.set_order_status(oid[:8], chat_id=1, status="paused")
            self.assertTrue(ok)
            got2 = st.get_order(oid, chat_id=1)
            self.assertIsNotNone(got2)
            assert got2 is not None
            self.assertEqual(str(got2["status"]), "paused")

            bad = st.set_order_status(oid[:8], chat_id=1, status="nope")
            self.assertFalse(bad)


    def test_set_order_done_closes_linked_project_when_no_active_orders_remain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            st = SQLiteTaskStorage(db)

            pid = "proj-aaaaaaaa"
            oid = "aaaaaaaa-1111-1111-1111-111111111111"

            st.upsert_project(project_id=pid, name="Ideas project", path="/tmp/ideas", status="active", created_by="ceo")
            st.upsert_order(
                order_id=oid,
                chat_id=1,
                title="Ideation order",
                body="dame ideas",
                status="active",
                priority=2,
                intent_type="order_project_new",
                project_id=pid,
            )

            ok = st.set_order_status(oid[:8], chat_id=1, status="done")
            self.assertTrue(ok)

            done_ids = {str(p["project_id"]) for p in st.list_projects(status="done", limit=20)}
            self.assertIn(pid, done_ids)

    def test_project_stays_active_while_another_order_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            st = SQLiteTaskStorage(db)

            pid = "proj-bbbbbbbb"
            oid1 = "bbbbbbbb-1111-1111-1111-111111111111"
            oid2 = "bbbbbbbb-2222-2222-2222-222222222222"

            st.upsert_project(project_id=pid, name="Shared project", path="/tmp/shared", status="active", created_by="ceo")
            st.upsert_order(order_id=oid1, chat_id=1, title="order 1", body="work 1", status="active", priority=2, intent_type="order_project_new", project_id=pid)
            st.upsert_order(order_id=oid2, chat_id=1, title="order 2", body="work 2", status="active", priority=2, intent_type="order_project_change", project_id=pid)

            ok1 = st.set_order_status(oid1, chat_id=1, status="done")
            self.assertTrue(ok1)
            active_ids = {str(p["project_id"]) for p in st.list_projects(status="active", limit=20)}
            self.assertIn(pid, active_ids)

            ok2 = st.set_order_status(oid2, chat_id=1, status="done")
            self.assertTrue(ok2)
            done_ids = {str(p["project_id"]) for p in st.list_projects(status="done", limit=20)}
            self.assertIn(pid, done_ids)


class TestStopAllGlobal(unittest.TestCase):
    def test_stop_all_global_cancels_jobs_closes_orders_projects_and_clears_leases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            st = SQLiteTaskStorage(db)
            q = OrchestratorQueue(storage=st, role_profiles=None)

            base = Task.new(
                source="telegram",
                role="backend",
                input_text="work item",
                request_type="task",
                priority=2,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
            )
            tasks = [
                base.with_updates(job_id="a" * 36, state="queued", input_text="queued work"),
                base.with_updates(job_id="b" * 36, state="running", input_text="running work"),
                base.with_updates(job_id="c" * 36, state="blocked", input_text="blocked work"),
                base.with_updates(job_id="d" * 36, state="waiting_deps", input_text="waiting work"),
                base.with_updates(job_id="e" * 36, state="blocked_approval", input_text="approval work"),
                base.with_updates(job_id="f" * 36, state="done", input_text="done work"),
            ]
            for task in tasks:
                st.submit_task(task)

            st.upsert_project(project_id="proj-stop", name="Stop project", path="/tmp/stop", status="active", created_by="ceo")
            st.upsert_order(
                order_id="11111111-1111-1111-1111-111111111111",
                chat_id=1,
                title="Stop order",
                body="cancel everything",
                status="active",
                priority=1,
                project_id="proj-stop",
            )

            slot = q.lease_workspace(role="backend", job_id="b" * 36, slots=1)
            self.assertIsNotNone(slot)
            self.assertIsNotNone(q.get_workspace_lease(job_id="b" * 36))

            out = q.stop_all_global(
                reason="ceo_stop_all",
                actor="jarvis",
                chat_id=None,
                close_orders=True,
                close_projects=True,
                clear_workspace_leases=True,
            )

            self.assertEqual(out["jobs_cancelled"], 5)
            self.assertEqual(out["orders_done"], 1)
            self.assertEqual(out["projects_done"], 1)
            self.assertEqual(out["workspace_leases_cleared"], 1)

            for job_id in ("a" * 36, "b" * 36, "c" * 36, "d" * 36, "e" * 36):
                job = st.get_job(job_id)
                self.assertIsNotNone(job)
                assert job is not None
                self.assertEqual(job.state, "cancelled")
            done_job = st.get_job("f" * 36)
            self.assertIsNotNone(done_job)
            assert done_job is not None
            self.assertEqual(done_job.state, "done")

            order = st.get_order("11111111", chat_id=1)
            self.assertIsNotNone(order)
            assert order is not None
            self.assertEqual(str(order["status"]), "done")
            self.assertEqual(str(order["phase"]), "done")

            done_projects = {str(p["project_id"]) for p in st.list_projects(status="done", limit=20)}
            self.assertIn("proj-stop", done_projects)
            self.assertIsNone(q.get_workspace_lease(job_id="b" * 36))

            with st._conn() as conn:
                row = conn.execute(
                    "SELECT event_type, actor, details FROM audit_log ORDER BY id DESC LIMIT 1"
                ).fetchone()
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(str(row["event_type"]), "stop_all_global")
            self.assertEqual(str(row["actor"]), "jarvis")
            details = json.loads(str(row["details"] or "{}"))
            self.assertEqual(int(details.get("jobs_cancelled") or 0), 5)
            self.assertEqual(int(details.get("orders_done") or 0), 1)
            self.assertEqual(int(details.get("projects_done") or 0), 1)
            self.assertEqual(int(details.get("workspace_leases_cleared") or 0), 1)


class TestBulkCancel(unittest.TestCase):
    def test_cancel_by_states_keeps_running(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            st = SQLiteTaskStorage(db)

            t_queued = Task.new(
                source="telegram",
                role="backend",
                input_text="queued work",
                request_type="task",
                priority=2,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
            )
            t_blocked = t_queued.with_updates(job_id="b" * 36, state="blocked", input_text="blocked work")
            t_running = t_queued.with_updates(job_id="c" * 36, state="running", input_text="running work")
            t_done = t_queued.with_updates(job_id="d" * 36, state="done", input_text="done work")

            st.submit_task(t_queued)
            st.submit_task(t_blocked)
            st.submit_task(t_running)
            st.submit_task(t_done)

            n = st.cancel_by_states(states=("queued", "blocked"), reason="test_purge")
            self.assertEqual(n, 2)

            got_q = st.get_job(t_queued.job_id)
            got_b = st.get_job(t_blocked.job_id)
            got_r = st.get_job(t_running.job_id)
            got_d = st.get_job(t_done.job_id)
            assert got_q is not None and got_b is not None and got_r is not None and got_d is not None
            self.assertEqual(got_q.state, "cancelled")
            self.assertEqual(got_b.state, "cancelled")
            self.assertEqual(got_r.state, "running")
            self.assertEqual(got_d.state, "done")



class TestTraceEvents(unittest.TestCase):
    def test_append_and_list_trace_events(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage)
            eid = q.append_trace_event(
                order_id="ord-1",
                job_id="job-1",
                agent_role="backend",
                event_type="job.running",
                severity="info",
                message="job running",
                payload={"phase": "exec"},
            )
            rows = q.list_trace_events(order_id="ord-1", limit=20)
            self.assertTrue(rows)
            self.assertEqual(rows[0]["id"], eid)
            self.assertEqual(rows[0]["event_type"], "job.running")

    def test_trace_noise_summary_counts_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage)
            q.append_trace_event(order_id="ord-1", job_id="job-1", agent_role="backend", event_type="job.state", severity="info", message="same")
            q.append_trace_event(order_id="ord-1", job_id="job-1", agent_role="backend", event_type="job.state", severity="info", message="same")
            summary = q.trace_noise_summary(window_seconds=3600)
            self.assertGreaterEqual(int(summary.get("total_events") or 0), 2)
            self.assertGreaterEqual(int(summary.get("duplicate_events") or 0), 1)
