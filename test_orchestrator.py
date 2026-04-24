from __future__ import annotations
import json
import os
import sqlite3
import subprocess
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

            msg_factory = bot.IncomingMessage(23, 1, 2, 33, "u", "/factory")
            resp_factory, _ = bot._parse_job(cfg, msg_factory)
            self.assertEqual(resp_factory, "__orch_factory__")

            msg_factory_pause = bot.IncomingMessage(24, 1, 2, 34, "u", "/factory pause 2h")
            resp_factory_pause, _ = bot._parse_job(cfg, msg_factory_pause)
            self.assertEqual(resp_factory_pause, "__orch_factory:pause 2h")

            msg_factory_scope = bot.IncomingMessage(25, 1, 2, 35, "u", "/factory scope set poncebot android dashboard")
            resp_factory_scope, _ = bot._parse_job(cfg, msg_factory_scope)
            self.assertEqual(resp_factory_scope, "__orch_factory:scope set poncebot android dashboard")

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

    def test_orchestrator_subtasks_normalize_key_alignment(self) -> None:
        specs = parse_orchestrator_subtasks(
            {
                "subtasks": [
                    {
                        "key": " Ship API ",
                        "role": "backend",
                        "text": "Implement endpoint",
                        "depends_on": [" ARCH ", " Ship API "],
                    }
                ]
            }
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].key, "ship_api")
        self.assertEqual(specs[0].depends_on, ["arch"])


class TestOrchestratorRoleNormalization(unittest.TestCase):
    def test_known_roles_are_stable(self) -> None:
        for role in bot._ORCHESTRATOR_ROLES:
            self.assertEqual(bot._coerce_orchestrator_role(role), role)

    def test_ceo_and_unknown_roles_normalize(self) -> None:
        self.assertEqual(bot._coerce_orchestrator_role("ceo"), "jarvis")
        self.assertEqual(bot._coerce_orchestrator_role("orchestrator"), "jarvis")
        self.assertEqual(bot._coerce_orchestrator_role("unknown_role"), "backend")


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

    def test_autopilot_tick_skips_proactive_orders_while_autonomy_is_paused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td) / "state.json")
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles={"skynet": {"role": "skynet"}})  # type: ignore[arg-type]

            oid = "11111111-2222-3333-4444-555555555555"
            storage.upsert_order(
                order_id=oid,
                chat_id=1,
                title="Proactive Sprint: pause me",
                body="AUTONOMOUS PROACTIVE SPRINT\n[proactive:pause-test]\nLane: bot",
                status="active",
                priority=3,
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
                profiles={"skynet": {"role": "skynet"}},
                chat_id=1,
                now=12345.0,
            )

            self.assertEqual(created, 0)
            rows = storage.jobs_by_state(state="queued", limit=20)
            self.assertEqual(rows, [])

    def test_active_order_watchdog_requeues_idle_proactive_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            cfg = _cfg(Path(td) / "state.json")
            storage = SQLiteTaskStorage(db_path)
            q = OrchestratorQueue(storage=storage, role_profiles={"skynet": {"role": "skynet"}})  # type: ignore[arg-type]

            oid = "66666666-7777-8888-9999-aaaaaaaaaaaa"
            storage.upsert_order(
                order_id=oid,
                chat_id=1,
                title="Proactive Sprint: revive me",
                body="AUTONOMOUS PROACTIVE SPRINT\n[proactive:watchdog-test]\nLane: bot",
                status="active",
                priority=3,
                phase="review",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={"proactive_lane": True},
                    job_id=oid,
                )
            )

            now = _time.time() + 500.0
            with patch.dict(
                os.environ,
                {
                    "BOT_ACTIVE_ORDER_WATCHDOG_ENABLED": "1",
                    "BOT_ACTIVE_ORDER_WATCHDOG_IDLE_SECONDS": "90",
                    "BOT_ACTIVE_ORDER_WATCHDOG_COOLDOWN_SECONDS": "180",
                    "BOT_ACTIVE_ORDER_WATCHDOG_MAX_PER_TICK": "1",
                },
                clear=False,
            ):
                created = bot._active_order_watchdog_tick(
                    cfg=cfg,
                    orch_q=q,
                    profiles={"skynet": {"role": "skynet"}},
                    chat_id=1,
                    now=now,
                )

            self.assertEqual(created, 1)
            children = [row for row in q.jobs_by_parent(parent_job_id=oid, limit=20) if row.job_id != oid]
            self.assertEqual(len(children), 1)
            child = children[0]
            self.assertEqual(child.role, "skynet")
            self.assertEqual(child.state, "queued")
            self.assertEqual(child.parent_job_id, oid)
            self.assertEqual(str((child.labels or {}).get("kind") or ""), "autopilot")
            self.assertEqual(str((child.trace or {}).get("autopilot_reason") or ""), "proactive_idle_watchdog")

    def test_active_order_watchdog_ignores_recent_order_touch_when_runtime_is_idle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            cfg = _cfg(Path(td) / "state.json")
            storage = SQLiteTaskStorage(db_path)
            q = OrchestratorQueue(storage=storage, role_profiles={"skynet": {"role": "skynet"}})  # type: ignore[arg-type]

            oid = "bbbbbbbb-7777-8888-9999-aaaaaaaaaaaa"
            storage.upsert_order(
                order_id=oid,
                chat_id=1,
                title="Proactive Sprint: touched order",
                body="AUTONOMOUS PROACTIVE SPRINT\n[proactive:watchdog-order-touch]\nLane: bot",
                status="active",
                priority=3,
                phase="review",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={"proactive_lane": True},
                    job_id=oid,
                )
            )

            base_now = _time.time()
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE ceo_orders SET updated_at = ? WHERE order_id = ?",
                    (base_now + 95.0, oid),
                )
                conn.execute(
                    "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
                    (base_now + 95.0, oid),
                )
                conn.commit()

            with patch.dict(
                os.environ,
                {
                    "BOT_ACTIVE_ORDER_WATCHDOG_ENABLED": "1",
                    "BOT_ACTIVE_ORDER_WATCHDOG_IDLE_SECONDS": "90",
                    "BOT_ACTIVE_ORDER_WATCHDOG_COOLDOWN_SECONDS": "180",
                    "BOT_ACTIVE_ORDER_WATCHDOG_MAX_PER_TICK": "1",
                },
                clear=False,
            ):
                created = bot._active_order_watchdog_tick(
                    cfg=cfg,
                    orch_q=q,
                    profiles={"skynet": {"role": "skynet"}},
                    chat_id=1,
                    now=base_now + 100.0,
                )

            self.assertEqual(created, 1)
            children = [row for row in q.jobs_by_parent(parent_job_id=oid, limit=20) if row.job_id != oid]
            self.assertEqual(len(children), 1)
            child = children[0]
            self.assertEqual(child.role, "skynet")
            self.assertEqual(child.state, "queued")
            self.assertEqual(str((child.trace or {}).get("autopilot_reason") or ""), "proactive_idle_watchdog")


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

    def test_sync_order_phase_keeps_order_active_when_deploy_failed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-deploy-failed-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Merged order with failed deploy",
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
                    trace={"merged_to_main": True, "deploy_status": "failed"},
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
            self.assertEqual(str(got.get("phase")), "review")
            self.assertEqual(str(got.get("status")), "active")

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

    def test_sync_order_phase_closes_proactive_order_on_verified_improvement_even_with_old_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-proactive-verified-01"
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
                    input_text="stale failed child",
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
                    job_id="job-old-failed",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="implementer_local",
                    input_text="implemented slice",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="done",
                    labels={"ticket": order_id, "kind": "subtask", "key": "local_impl_guard_slice1"},
                    trace={
                        "slice_id": "slice1",
                        "slice_patch_applied": True,
                        "slice_validation_ok": True,
                        "local_patch_info": {"changed_files": ["bot.py"], "validation_ok": True},
                    },
                    job_id="job-impl-verified",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="reviewer_local",
                    input_text="reviewed slice",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="done",
                    labels={"ticket": order_id, "kind": "subtask", "key": "local_review_guard_slice1"},
                    trace={"slice_id": "slice1", "result_summary": "PASS READY"},
                    job_id="job-review-verified",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="verified pass",
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
                        "slice_id": "slice1",
                        "result_summary": (
                            "PASS: VERIFIED_IMPROVEMENT. "
                            "Validated existing behavior and closed one bounded reliability improvement."
                        )
                    },
                    labels={"ticket": order_id, "kind": "autopilot"},
                    job_id="job-skynet-verified",
                )
            )

            bot._sync_order_phase_from_runtime(orch_q=q, root_ticket=order_id, chat_id=1)

            got = q.get_order(order_id, chat_id=1)
            assert got is not None
            self.assertEqual(str(got.get("phase")), "done")
            self.assertEqual(str(got.get("status")), "done")

            root = q.get_job(order_id)
            assert root is not None
            self.assertTrue(bool((root.trace or {}).get("proactive_improvement_closed", False)))

    def test_sync_order_phase_marks_ready_for_merge_for_verified_proactive_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-proactive-verified-merge-01"
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
                    trace={"proactive_lane": True, "order_branch": "feature/order-test"},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="implementer_local",
                    input_text="implemented slice",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="done",
                    labels={"ticket": order_id, "kind": "subtask", "key": "local_impl_guard_slice1"},
                    trace={
                        "slice_id": "slice1",
                        "slice_patch_applied": True,
                        "slice_validation_ok": True,
                        "local_patch_info": {"changed_files": ["bot.py"], "validation_ok": True},
                    },
                    job_id="job-impl-verified-merge",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="reviewer_local",
                    input_text="reviewed slice",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="done",
                    labels={"ticket": order_id, "kind": "subtask", "key": "local_review_guard_slice1"},
                    trace={"slice_id": "slice1", "result_summary": "PASS READY"},
                    job_id="job-review-verified-merge",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="verified pass",
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
                        "slice_id": "slice1",
                        "result_summary": (
                            "PASS: VERIFIED_IMPROVEMENT. "
                            "Validated existing behavior and closed one bounded reliability improvement."
                        )
                    },
                    labels={"ticket": order_id, "kind": "autopilot"},
                    job_id="job-skynet-verified-merge",
                )
            )

            with patch.object(bot, "_order_trace_requires_merge", return_value=(True, "feature/order-test")):
                bot._sync_order_phase_from_runtime(orch_q=q, root_ticket=order_id, chat_id=1)

            got = q.get_order(order_id, chat_id=1)
            assert got is not None
            self.assertEqual(str(got.get("phase")), "ready_for_merge")
            self.assertEqual(str(got.get("status")), "active")

            root = q.get_job(order_id)
            assert root is not None
            self.assertTrue(bool((root.trace or {}).get("proactive_improvement_closed", False)))
            self.assertTrue(bool((root.trace or {}).get("merge_ready", False)))

    def test_sync_order_phase_marks_ready_for_merge_for_verified_proactive_backend_qa_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-proactive-backend-qa-merge-01"
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
                    trace={"proactive_lane": True, "order_branch": "feature/order-test"},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="implemented slice",
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
                    labels={"ticket": order_id, "kind": "subtask", "key": "proactive_cli_seed_r1"},
                    trace={
                        "slice_id": "proactive_cli_seed_r1",
                        "patch_info": {"changed_files": ["bot.py"], "validation_ok": True},
                        "result_summary": "Implemented one bounded reliability improvement with tests and evidence.",
                    },
                    job_id="job-backend-verified-merge",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="qa",
                    input_text="reviewed slice",
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
                    labels={"ticket": order_id, "kind": "subtask", "key": "proactive_cli_seed_r1_qa"},
                    trace={
                        "review_ready": True,
                        "result_summary": (
                            "PASS validation with evidence artifacts, targeted regression coverage, "
                            "and explicit residual risks for the proactive_cli_seed_r1 slice."
                        ),
                        "result_artifacts": ["qa/report.txt"],
                    },
                    job_id="job-qa-verified-merge",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="verified pass",
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
                        "slice_id": "proactive_cli_seed_r1",
                        "result_summary": (
                            "PASS: VERIFIED_IMPROVEMENT. "
                            "Validated one bounded backend plus QA proactive improvement."
                        ),
                    },
                    labels={"ticket": order_id, "kind": "autopilot"},
                    job_id="job-skynet-backend-qa-verified-merge",
                )
            )

            with patch.object(bot, "_order_trace_requires_merge", return_value=(True, "feature/order-test")):
                bot._sync_order_phase_from_runtime(orch_q=q, root_ticket=order_id, chat_id=1)

            got = q.get_order(order_id, chat_id=1)
            assert got is not None
            self.assertEqual(str(got.get("phase")), "ready_for_merge")
            self.assertEqual(str(got.get("status")), "active")

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

    def test_final_sweep_default_terminal_threshold_closes_at_ten(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-default-threshold"
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
            for i in range(11):
                q.submit_task(
                    Task.new(
                        source="telegram",
                        role="implementer_local",
                        input_text=f"attempt {i + 1}",
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
                        job_id=f"job-terminal-default-{i + 1}",
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

    def test_final_sweep_blocker_predicate_ignores_controller_review_overhead(self) -> None:
        overhead_children = [
            Task.new(
                source="telegram",
                role="qa",
                input_text="qa follow-up",
                request_type="review",
                priority=2,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="running",
                parent_job_id="ord-overhead",
            ),
            Task.new(
                source="telegram",
                role="reviewer_local",
                input_text="review pass",
                request_type="review",
                priority=2,
                model="",
                effort="",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="queued",
                parent_job_id="ord-overhead",
            ),
        ]
        blockers_overhead = bot._final_sweep_blocker_count(children=overhead_children)
        self.assertEqual(blockers_overhead, 0)

        real_blocker_children = overhead_children + [
            Task.new(
                source="telegram",
                role="backend",
                input_text="real delivery work",
                request_type="task",
                priority=1,
                model="",
                effort="",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="running",
                parent_job_id="ord-overhead",
            )
        ]
        blockers_real = bot._final_sweep_blocker_count(children=real_blocker_children)
        self.assertGreater(blockers_real, 0)

    def test_final_sweep_auto_closes_with_only_review_overhead_open(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-review-overhead-only"
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
                    state="done",
                    job_id="job-terminal-overhead-1",
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
                    job_id="job-terminal-overhead-2",
                )
            )
            # Controller/review overhead stays open but should not self-block closure.
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="qa",
                    input_text="post-check",
                    request_type="review",
                    priority=2,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="running",
                    job_id="job-overhead-qa-open",
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

    def test_final_sweep_default_stale_window_enqueues_sooner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-default-stale-window"
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

            now = _time.time()
            stale_ts = now - 180.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
                    (float(stale_ts), float(stale_ts), order_id),
                )
                conn.commit()

            with patch.dict(
                os.environ,
                {
                    # Keep defaults for stale threshold and close threshold.
                    "BOT_JARVIS_FINAL_SWEEP_COOLDOWN_SECONDS": "1",
                },
                clear=False,
            ):
                created = bot._jarvis_final_sweep_tick(cfg=cfg, orch_q=q, profiles=None, now=now)

            self.assertEqual(created, 1)
            children = q.jobs_by_parent(parent_job_id=order_id, limit=50)
            sweep_children = [c for c in children if str((c.labels or {}).get("kind") or "").strip().lower() == "final_sweep"]
            self.assertEqual(len(sweep_children), 1)

    def test_final_sweep_invalid_stale_env_uses_safe_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-invalid-stale-env"
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

            now = _time.time()
            stale_ts = now - 180.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
                    (float(stale_ts), float(stale_ts), order_id),
                )
                conn.commit()

            with patch.dict(
                os.environ,
                {
                    # Broken env value should fall back to safe default instead of disabling stale sweep.
                    "BOT_JARVIS_IDLE_ORDER_STALE_SECONDS": "not-a-number",
                    "BOT_JARVIS_FINAL_SWEEP_COOLDOWN_SECONDS": "1",
                },
                clear=False,
            ):
                created = bot._jarvis_final_sweep_tick(cfg=cfg, orch_q=q, profiles=None, now=now)

            self.assertEqual(created, 1)
            children = q.jobs_by_parent(parent_job_id=order_id, limit=50)
            sweep_children = [c for c in children if str((c.labels or {}).get("kind") or "").strip().lower() == "final_sweep"]
            self.assertEqual(len(sweep_children), 1)

    def test_final_sweep_enqueues_for_stale_active_order_without_root_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-orphaned-order"
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

            now = _time.time()
            stale_ts = now - 180.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE ceo_orders SET created_at = ?, updated_at = ? WHERE order_id = ?",
                    (float(stale_ts), float(stale_ts), order_id),
                )
                conn.commit()

            with patch.dict(
                os.environ,
                {
                    "BOT_JARVIS_FINAL_SWEEP_COOLDOWN_SECONDS": "1",
                },
                clear=False,
            ):
                created = bot._jarvis_final_sweep_tick(cfg=cfg, orch_q=q, profiles=None, now=now)

            self.assertEqual(created, 1)
            children = q.jobs_by_parent(parent_job_id=order_id, limit=50)
            sweep_children = [c for c in children if str((c.labels or {}).get("kind") or "").strip().lower() == "final_sweep"]
            self.assertEqual(len(sweep_children), 1)

    def test_final_sweep_uses_first_valid_timestamp_when_updated_at_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-invalid-updated-at"
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

            now = _time.time()
            stale_ts = now - 180.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE ceo_orders SET created_at = ?, updated_at = ? WHERE order_id = ?",
                    (float(stale_ts), "bad-timestamp", order_id),
                )
                conn.commit()

            with patch.dict(
                os.environ,
                {
                    "BOT_JARVIS_FINAL_SWEEP_COOLDOWN_SECONDS": "1",
                },
                clear=False,
            ):
                created = bot._jarvis_final_sweep_tick(cfg=cfg, orch_q=q, profiles=None, now=now)

            self.assertEqual(created, 1)
            children = q.jobs_by_parent(parent_job_id=order_id, limit=50)
            sweep_children = [c for c in children if str((c.labels or {}).get("kind") or "").strip().lower() == "final_sweep"]
            self.assertEqual(len(sweep_children), 1)

    def test_final_sweep_rejects_non_finite_timestamp_values(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-non-finite-ts"
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

            now = _time.time()
            stale_ts = now - 180.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
                    (float(stale_ts), "nan", order_id),
                )
                conn.execute(
                    "UPDATE ceo_orders SET updated_at = ? WHERE order_id = ?",
                    ("inf", order_id),
                )
                conn.commit()

            with patch.dict(
                os.environ,
                {
                    "BOT_JARVIS_FINAL_SWEEP_COOLDOWN_SECONDS": "1",
                },
                clear=False,
            ):
                created = bot._jarvis_final_sweep_tick(cfg=cfg, orch_q=q, profiles=None, now=now)

            self.assertEqual(created, 1)
            children = q.jobs_by_parent(parent_job_id=order_id, limit=50)
            sweep_children = [c for c in children if str((c.labels or {}).get("kind") or "").strip().lower() == "final_sweep"]
            self.assertEqual(len(sweep_children), 1)

    def test_final_sweep_does_not_reenqueue_when_done_sweep_requests_root_cause_close(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            order_id = "ord-final-sweep-root-cause-closed"
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
                    role="jarvis",
                    input_text="final sweep",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="done",
                    labels={"ticket": order_id, "kind": "final_sweep"},
                    trace={"result_summary": "NO-GO: close this order due blocked with root cause."},
                    job_id="job-final-sweep-done-close",
                )
            )

            now = _time.time()
            stale_ts = now - 1800.0
            with storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
                    (float(stale_ts), float(stale_ts), order_id),
                )
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
                    (float(stale_ts), float(stale_ts), "job-final-sweep-done-close"),
                )
                conn.commit()

            with patch.dict(
                os.environ,
                {
                    "BOT_JARVIS_IDLE_ORDER_STALE_SECONDS": "120",
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
            children = q.jobs_by_parent(parent_job_id=order_id, limit=100)
            queued_sweeps = [
                c
                for c in children
                if str((c.labels or {}).get("kind") or "").strip().lower() == "final_sweep"
                and str(c.state or "").strip().lower() == "queued"
            ]
            self.assertEqual(queued_sweeps, [])


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

    def test_skynet_profile_uses_gpt55_high_effort(self) -> None:
        profiles = load_agent_profiles(Path(__file__).resolve().parent / "orchestrator" / "agents.yaml")
        skynet = profiles["skynet"]
        self.assertEqual(skynet.get("model"), "gpt-5.5")
        self.assertEqual(skynet.get("effort"), "high")

    def test_runbooks_yaml_parses_multiline_prompt(self) -> None:
        rbs = load_runbooks(Path(__file__).resolve().parent / "orchestrator" / "runbooks.yaml")
        self.assertGreaterEqual(len(rbs), 1)
        rb = rbs[0]
        self.assertNotEqual(rb.prompt.strip(), "|")
        self.assertTrue(rb.prompt.strip())

    def test_release_manager_governance_command_pins_qa_role(self) -> None:
        profiles = load_agent_profiles(Path(__file__).resolve().parent / "orchestrator" / "agents.yaml")
        rm = profiles["release_mgr"]
        system_prompt = str(rm.get("system_prompt") or "")
        self.assertIn("tools/release_governance.py", system_prompt)
        self.assertIn("--role qa", system_prompt)


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

    def test_terminal_dependency_update_immediately_releases_waiting_deps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            dep = q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="dep running",
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
                role="qa",
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
            self.assertIsNone(q.take_next())  # transitions blocked task to waiting_deps
            waiting = q.get_job(jid)
            self.assertIsNotNone(waiting)
            assert waiting is not None
            self.assertEqual(waiting.state, "waiting_deps")

            self.assertTrue(q.update_state(dep, "done"))
            released = q.get_job(jid)
            self.assertIsNotNone(released)
            assert released is not None
            self.assertEqual(released.state, "queued")
            self.assertIsNone(released.blocked_reason)


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


class TestFactoryScope(unittest.TestCase):
    def test_scope_aliases_select_expected_repos(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            repo_rows = [
                ("codexbot-6fb8d5b9", "/home/aponce/codexbot"),
                ("codexbot_voiceout-068e9f37", "/home/aponce/codexbot_voiceout"),
                ("executivedashboard-9ab2eb91", "/home/aponce/ExecutiveDashboard"),
                ("executivedashboard.tmp-3e0fce26", "/home/aponce/workspaces/ExecutiveDashboard.tmp"),
                ("omnicrewapp.android-ba7b4a67", "/home/aponce/OmniCrewApp.android"),
                ("wormhole-4b21fa9e", "/home/aponce/wormhole"),
            ]
            for repo_id, repo_path in repo_rows:
                q.upsert_repo(
                    repo_id=repo_id,
                    path=repo_path,
                    default_branch="main",
                    autonomy_enabled=True,
                    priority=2,
                    runtime_mode="ceo-bounded",
                    daily_budget=0.0,
                    status="active",
                    metadata={},
                )

            selected, missing = bot._resolve_factory_repo_selectors(
                q,
                ["poncebot", "android", "dashboard"],
            )
            selected_ids = {str(repo.get("repo_id") or "") for repo in selected}

            self.assertEqual(missing, [])
            self.assertEqual(
                selected_ids,
                {
                    "codexbot-6fb8d5b9",
                    "codexbot_voiceout-068e9f37",
                    "executivedashboard-9ab2eb91",
                    "omnicrewapp.android-ba7b4a67",
                },
            )
            self.assertNotIn("executivedashboard.tmp-3e0fce26", selected_ids)

    def test_scope_set_enables_only_selected_repos(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            repo_rows = [
                ("codexbot-6fb8d5b9", "/home/aponce/codexbot"),
                ("codexbot_voiceout-068e9f37", "/home/aponce/codexbot_voiceout"),
                ("executivedashboard-9ab2eb91", "/home/aponce/ExecutiveDashboard"),
                ("omnicrewapp.android-ba7b4a67", "/home/aponce/OmniCrewApp.android"),
                ("wormhole-4b21fa9e", "/home/aponce/wormhole"),
            ]
            for repo_id, repo_path in repo_rows:
                q.upsert_repo(
                    repo_id=repo_id,
                    path=repo_path,
                    default_branch="main",
                    autonomy_enabled=True,
                    priority=2,
                    runtime_mode="ceo-bounded",
                    daily_budget=0.0,
                    status="active",
                    metadata={},
                )

            selected_repo_ids = {
                "codexbot-6fb8d5b9",
                "codexbot_voiceout-068e9f37",
                "executivedashboard-9ab2eb91",
                "omnicrewapp.android-ba7b4a67",
            }
            out = bot._factory_apply_scope_action(
                q,
                action="set",
                selected_repo_ids=selected_repo_ids,
            )

            repos = {str(repo["repo_id"]): repo for repo in q.list_repos(limit=20)}
            self.assertEqual(int(out["enabled"]), 0)
            self.assertEqual(int(out["disabled"]), 1)
            self.assertTrue(bool(repos["codexbot-6fb8d5b9"]["autonomy_enabled"]))
            self.assertTrue(bool(repos["codexbot_voiceout-068e9f37"]["autonomy_enabled"]))
            self.assertTrue(bool(repos["executivedashboard-9ab2eb91"]["autonomy_enabled"]))
            self.assertTrue(bool(repos["omnicrewapp.android-ba7b4a67"]["autonomy_enabled"]))
            self.assertFalse(bool(repos["wormhole-4b21fa9e"]["autonomy_enabled"]))
            self.assertEqual(str(repos["wormhole-4b21fa9e"]["status"]), "disabled")


class TestMergeAndDeployFlow(unittest.TestCase):
    def _git_env(self) -> dict[str, str]:
        return {
            **dict(os.environ),
            "GIT_AUTHOR_NAME": "Codex",
            "GIT_AUTHOR_EMAIL": "codex@example.com",
            "GIT_COMMITTER_NAME": "Codex",
            "GIT_COMMITTER_EMAIL": "codex@example.com",
        }

    def _git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=self._git_env(),
        )

    def test_autocommit_push_order_branch_forces_server_git_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            remote = td_path / "remote.git"
            repo = td_path / "repo"
            remote.mkdir(parents=True, exist_ok=True)
            repo.mkdir(parents=True, exist_ok=True)

            self._git(remote, "init", "--bare")
            self._git(repo, "init")
            self._git(repo, "checkout", "-b", "main")
            self._git(repo, "remote", "add", "origin", str(remote))
            self._git(repo, "config", "user.name", "manolosake")
            self._git(repo, "config", "user.email", "manolosake@gmail.com")
            (repo / "README.md").write_text("base\n", encoding="utf-8")
            self._git(repo, "add", "README.md")
            self._git(repo, "commit", "-m", "initial")
            self._git(repo, "push", "-u", "origin", "main")

            (repo / "change.txt").write_text("autonomous change\n", encoding="utf-8")
            task = Task.new(
                source="telegram",
                role="implementer_local",
                input_text="change",
                request_type="task",
                priority=1,
                model="",
                effort="",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                parent_job_id="order-identity-0001",
                labels={"key": "identity"},
                job_id="job-identity-0001",
            )

            with patch.dict(
                os.environ,
                {
                    "GIT_AUTHOR_NAME": "PonceBot",
                    "GIT_AUTHOR_EMAIL": "poncebot@local",
                    "GIT_COMMITTER_NAME": "PonceBot",
                    "GIT_COMMITTER_EMAIL": "poncebot@local",
                },
                clear=False,
            ):
                result = bot._autocommit_push_order_branch(
                    worktree_dir=repo,
                    order_branch="feature/order-identity",
                    task=task,
                )

            self.assertEqual(str(result.get("status")), "ok")
            log = subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(remote),
                    "log",
                    "-1",
                    "--format=%an <%ae>",
                    "refs/heads/feature/order-identity",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=self._git_env(),
            )
            self.assertEqual(log.stdout.strip(), "manolosake <manolosake@gmail.com>")

    def test_reconcile_order_branch_with_main_updates_remote_branch_before_merge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            remote = td_path / "remote.git"
            repo = td_path / "repo"
            remote.mkdir(parents=True, exist_ok=True)
            repo.mkdir(parents=True, exist_ok=True)

            self._git(remote, "init", "--bare")
            self._git(repo, "init")
            self._git(repo, "checkout", "-b", "main")
            self._git(repo, "remote", "add", "origin", str(remote))

            (repo / "shared.txt").write_text("base\n", encoding="utf-8")
            (repo / "feature_only.txt").write_text("feature-base\n", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "init")
            self._git(repo, "push", "-u", "origin", "main")

            self._git(repo, "checkout", "-b", "feature/order-merge")
            (repo / "feature_only.txt").write_text("feature-change\n", encoding="utf-8")
            self._git(repo, "add", "feature_only.txt")
            self._git(repo, "commit", "-m", "feature change")
            self._git(repo, "push", "-u", "origin", "feature/order-merge")

            self._git(repo, "checkout", "main")
            (repo / "shared.txt").write_text("main-change\n", encoding="utf-8")
            self._git(repo, "add", "shared.txt")
            self._git(repo, "commit", "-m", "main advance")
            self._git(repo, "push", "origin", "main")

            ok, msg, branch_head = bot._reconcile_order_branch_with_main(
                repo=repo,
                order_branch="feature/order-merge",
                order_id="ord-reconcile-01",
                default_branch="main",
            )
            self.assertTrue(ok, msg)
            self.assertEqual(msg, "branch_reconciled_with_main")
            self.assertTrue(bool(branch_head))

            self._git(repo, "fetch", "origin", "--prune")
            self.assertTrue(bot._git_is_ancestor(repo, "origin/main", "origin/feature/order-merge"))

            merged_ok, merged_msg, merge_commit = bot._merge_order_branch_to_main(
                repo=repo,
                order_branch="feature/order-merge",
                order_id="ord-reconcile-01",
                default_branch="main",
            )
            self.assertTrue(merged_ok, merged_msg)
            self.assertEqual(merged_msg, "merged_to_main")
            self.assertTrue(bool(merge_commit))

    def test_merge_order_branch_to_main_rejects_no_delta_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            remote = td_path / "remote.git"
            repo = td_path / "repo"
            remote.mkdir(parents=True, exist_ok=True)
            repo.mkdir(parents=True, exist_ok=True)

            self._git(remote, "init", "--bare")
            self._git(repo, "init")
            self._git(repo, "checkout", "-b", "main")
            self._git(repo, "remote", "add", "origin", str(remote))

            (repo / "app.txt").write_text("base\n", encoding="utf-8")
            self._git(repo, "add", "app.txt")
            self._git(repo, "commit", "-m", "init")
            self._git(repo, "push", "-u", "origin", "main")

            self._git(repo, "checkout", "-b", "feature/order-no-delta")
            self._git(repo, "push", "-u", "origin", "feature/order-no-delta")
            self._git(repo, "checkout", "main")

            merged_ok, merged_msg, merge_commit = bot._merge_order_branch_to_main(
                repo=repo,
                order_branch="feature/order-no-delta",
                order_id="ord-no-delta",
                default_branch="main",
            )

            self.assertFalse(merged_ok)
            self.assertEqual(merged_msg, "branch_has_no_delta_vs_main")
            self.assertIsNone(merge_commit)

    def test_reconcile_order_branch_with_main_reports_conflict_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            remote = td_path / "remote.git"
            repo = td_path / "repo"
            remote.mkdir(parents=True, exist_ok=True)
            repo.mkdir(parents=True, exist_ok=True)

            self._git(remote, "init", "--bare")
            self._git(repo, "init")
            self._git(repo, "checkout", "-b", "main")
            self._git(repo, "remote", "add", "origin", str(remote))

            (repo / "shared.txt").write_text("base\n", encoding="utf-8")
            self._git(repo, "add", "shared.txt")
            self._git(repo, "commit", "-m", "init")
            self._git(repo, "push", "-u", "origin", "main")

            self._git(repo, "checkout", "-b", "feature/order-conflict")
            (repo / "shared.txt").write_text("feature-change\n", encoding="utf-8")
            self._git(repo, "add", "shared.txt")
            self._git(repo, "commit", "-m", "feature change")
            self._git(repo, "push", "-u", "origin", "feature/order-conflict")

            self._git(repo, "checkout", "main")
            (repo / "shared.txt").write_text("main-change\n", encoding="utf-8")
            self._git(repo, "add", "shared.txt")
            self._git(repo, "commit", "-m", "main change")
            self._git(repo, "push", "origin", "main")

            ok, msg, branch_head = bot._reconcile_order_branch_with_main(
                repo=repo,
                order_branch="feature/order-conflict",
                order_id="ord-reconcile-conflict",
                default_branch="main",
            )
            self.assertFalse(ok)
            self.assertIsNone(branch_head)
            self.assertIn("Conflicts:", msg)
            self.assertIn("shared.txt", msg)

    def test_order_merge_uses_repo_context_and_records_deploy_trace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path / "base-repo")
            cfg.codex_workdir.mkdir(parents=True, exist_ok=True)
            repo_path = td_path / "repo-secondary"
            repo_path.mkdir(parents=True, exist_ok=True)

            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            q.upsert_repo(
                repo_id="repo-secondary",
                path=str(repo_path),
                default_branch="main",
                autonomy_enabled=True,
                priority=1,
                runtime_mode="ceo-bounded",
                daily_budget=0.0,
                status="active",
                metadata={},
            )
            order_id = "ord-merge-repo-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Repo scoped merge",
                body="[repo:repo-secondary]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="ready_for_merge",
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
                    trace={
                        "order_branch": "feature/repo-secondary",
                        "repo_id": "repo-secondary",
                        "repo_path": str(repo_path),
                    },
                    job_id=order_id,
                )
            )

            with patch.object(bot, "_merge_order_branch_to_main", return_value=(True, "merged_to_main", "abc1234")) as merge_mock, patch.object(
                bot,
                "_deploy_after_order_merge",
                return_value={
                    "status": "ok",
                    "reason": "deploy_ok",
                    "summary": "Deploy completed successfully.",
                    "command": "bash scripts/deploy.sh",
                    "deployed_commit": "abc1234",
                },
            ):
                msg = bot._order_command_text(
                    cfg,
                    q,
                    chat_id=1,
                    payload=f"merge {order_id}",
                    user_id=2,
                    actor="jarvis",
                )

            merge_mock.assert_called_once()
            kwargs = merge_mock.call_args.kwargs
            self.assertEqual(Path(str(kwargs["repo"])).resolve(), repo_path.resolve())
            self.assertEqual(str(kwargs["default_branch"]), "main")
            self.assertIn("Deploy OK", msg)

            root = q.get_job(order_id)
            order = q.get_order(order_id, chat_id=1)
            assert root is not None and order is not None
            self.assertEqual(str(order.get("status")), "done")
            self.assertEqual(str(order.get("phase")), "done")
            self.assertEqual(str((root.trace or {}).get("deploy_status") or ""), "ok")

    def test_merge_no_delta_blocks_order_without_retry_ready(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path / "base-repo")
            cfg.codex_workdir.mkdir(parents=True, exist_ok=True)
            repo_path = td_path / "repo-secondary"
            repo_path.mkdir(parents=True, exist_ok=True)

            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            q.upsert_repo(
                repo_id="repo-secondary",
                path=str(repo_path),
                default_branch="main",
                autonomy_enabled=True,
                priority=1,
                runtime_mode="ceo-bounded",
                daily_budget=0.0,
                status="active",
                metadata={},
            )
            order_id = "ord-merge-no-delta"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Repo scoped no delta",
                body="[repo:repo-secondary]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="ready_for_merge",
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
                    trace={
                        "order_branch": "feature/repo-secondary",
                        "repo_id": "repo-secondary",
                        "repo_path": str(repo_path),
                        "merge_ready": True,
                    },
                    job_id=order_id,
                )
            )

            with patch.object(
                bot,
                "_merge_order_branch_to_main",
                return_value=(False, "branch_has_no_delta_vs_main", None),
            ):
                msg = bot._order_command_text(
                    cfg,
                    q,
                    chat_id=1,
                    payload=f"merge {order_id}",
                    user_id=2,
                    actor="jarvis",
                )

            self.assertIn("branch_has_no_delta_vs_main", msg)
            order = q.get_order(order_id, chat_id=1)
            root = q.get_job(order_id)
            assert order is not None and root is not None
            trace = dict(root.trace or {})
            self.assertEqual(str(order.get("status") or ""), "done")
            self.assertEqual(str(order.get("phase") or ""), "done")
            self.assertEqual(str(root.state), "failed")
            self.assertFalse(bool(trace.get("merge_ready", False)))
            self.assertTrue(bool(trace.get("merge_no_delta_active", False)))
            self.assertEqual(str(trace.get("result_status") or ""), "merge_no_delta")

    def test_merge_conflict_enqueues_skynet_resolution_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = _cfg(td_path / "state.json", workdir=td_path / "base-repo")
            cfg.codex_workdir.mkdir(parents=True, exist_ok=True)
            repo_path = td_path / "repo-secondary"
            repo_path.mkdir(parents=True, exist_ok=True)

            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            q.upsert_repo(
                repo_id="repo-secondary",
                path=str(repo_path),
                default_branch="main",
                autonomy_enabled=True,
                priority=1,
                runtime_mode="ceo-bounded",
                daily_budget=0.0,
                status="active",
                metadata={},
            )
            order_id = "ord-merge-conflict-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Repo scoped merge conflict",
                body="[repo:repo-secondary]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                phase="ready_for_merge",
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
                    trace={
                        "order_branch": "feature/repo-secondary",
                        "repo_id": "repo-secondary",
                        "repo_path": str(repo_path),
                    },
                    job_id=order_id,
                )
            )

            with patch.object(
                bot,
                "_merge_order_branch_to_main",
                return_value=(False, "CONFLICT (content): Merge conflict in bot.py", None),
            ):
                msg = bot._order_command_text(
                    cfg,
                    q,
                    chat_id=1,
                    payload=f"merge {order_id}",
                    user_id=2,
                    actor="jarvis",
                )

            self.assertIn("Skynet queued merge-conflict resolution", msg)
            order = q.get_order(order_id, chat_id=1)
            root = q.get_job(order_id)
            assert order is not None and root is not None
            self.assertEqual(str(order.get("phase") or ""), "planning")
            self.assertFalse(bool((root.trace or {}).get("merge_ready", False)))
            self.assertTrue(bool((root.trace or {}).get("merge_conflict_active", False)))
            children = [row for row in q.jobs_by_parent(parent_job_id=order_id, limit=20) if row.job_id != order_id]
            fix_jobs = [row for row in children if str((row.labels or {}).get("kind") or "").strip() == "merge_conflict_replan"]
            self.assertEqual(len(fix_jobs), 1)
            fix = fix_jobs[0]
            self.assertEqual(str(fix.role), "skynet")
            self.assertTrue(bool((fix.trace or {}).get("merge_conflict_resolution", False)))
            self.assertEqual(str((fix.trace or {}).get("merge_resolution_base_ref") or ""), "origin/main")
            self.assertEqual(str((fix.trace or {}).get("merge_resolution_target_ref") or ""), "origin/feature/repo-secondary")

    def test_git_is_non_fast_forward_push_error(self) -> None:
        self.assertTrue(
            bot._git_is_non_fast_forward_push_error(
                "Updates were rejected because a pushed branch tip is behind its remote counterpart."
            )
        )
        self.assertTrue(bot._git_is_non_fast_forward_push_error("! [rejected] HEAD -> main (non-fast-forward)"))
        self.assertFalse(bot._git_is_non_fast_forward_push_error("permission denied"))

    def test_git_ref_exists_detects_valid_refs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            self._git(repo, "init")
            self._git(repo, "checkout", "-b", "main")
            (repo / "app.txt").write_text("v1\n", encoding="utf-8")
            self._git(repo, "add", "app.txt")
            self._git(repo, "commit", "-m", "init")

            self.assertTrue(bot._git_ref_exists(repo, "HEAD"))
            self.assertTrue(bot._git_ref_exists(repo, "refs/heads/main"))
            self.assertFalse(bot._git_ref_exists(repo, "refs/heads/does-not-exist"))

    def test_deploy_after_merge_fast_forwards_checkout_and_runs_repo_script(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            remote = td_path / "remote.git"
            repo = td_path / "repo"
            remote.mkdir(parents=True, exist_ok=True)
            repo.mkdir(parents=True, exist_ok=True)

            self._git(remote, "init", "--bare")
            self._git(repo, "init")
            self._git(repo, "checkout", "-b", "main")
            self._git(repo, "remote", "add", "origin", str(remote))

            (repo / "scripts").mkdir(parents=True, exist_ok=True)
            (repo / "scripts" / "deploy.sh").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s' \"${PONCEBOT_DEPLOY_COMMIT}\" > .deployed_commit\n",
                encoding="utf-8",
            )
            (repo / "app.txt").write_text("v1\n", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "init")
            self._git(repo, "push", "-u", "origin", "main")

            self._git(repo, "checkout", "-b", "feature/test-deploy")
            (repo / "app.txt").write_text("v2\n", encoding="utf-8")
            self._git(repo, "add", "app.txt")
            self._git(repo, "commit", "-m", "feature change")
            self._git(repo, "push", "-u", "origin", "feature/test-deploy")
            self._git(repo, "checkout", "main")

            merged_ok, merged_msg, merge_commit = bot._merge_order_branch_to_main(
                repo=repo,
                order_branch="feature/test-deploy",
                order_id="ord-deploy-script",
                default_branch="main",
            )
            self.assertTrue(merged_ok, merged_msg)
            assert merge_commit is not None

            cfg = _cfg(td_path / "state.json", workdir=repo)
            repo_record = {
                "repo_id": "repo-main",
                "path": str(repo),
                "default_branch": "main",
                "metadata": {
                    "deploy": {
                        "enabled": True,
                        "source": "repo_script",
                        "cwd": str(repo),
                        "command": ["bash", "scripts/deploy.sh"],
                        "background": False,
                        "timeout_seconds": 30,
                    }
                },
            }
            result = bot._deploy_after_order_merge(
                cfg=cfg,
                repo_record=repo_record,
                repo_dir=repo,
                default_branch="main",
                order_id="ord-deploy-script",
                order_branch="feature/test-deploy",
                merge_commit=merge_commit,
            )
            self.assertEqual(str(result.get("status") or ""), "ok")
            self.assertEqual((repo / ".deployed_commit").read_text(encoding="utf-8"), str(merge_commit))
            self.assertEqual((repo / "app.txt").read_text(encoding="utf-8"), "v2\n")

    def test_deploy_after_merge_uses_runtime_worktree_when_repo_not_on_main(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            remote = td_path / "remote.git"
            repo = td_path / "repo"
            remote.mkdir(parents=True, exist_ok=True)
            repo.mkdir(parents=True, exist_ok=True)

            self._git(remote, "init", "--bare")
            self._git(repo, "init")
            self._git(repo, "checkout", "-b", "main")
            self._git(repo, "remote", "add", "origin", str(remote))

            (repo / "scripts").mkdir(parents=True, exist_ok=True)
            (repo / "scripts" / "deploy.sh").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s' \"${PONCEBOT_DEPLOY_COMMIT}\" > .deployed_commit\n"
                "git rev-parse --abbrev-ref HEAD > .deployed_branch\n",
                encoding="utf-8",
            )
            (repo / "app.txt").write_text("v1\n", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "init")
            self._git(repo, "push", "-u", "origin", "main")

            self._git(repo, "checkout", "-b", "feature/test-runtime")
            (repo / "app.txt").write_text("v2\n", encoding="utf-8")
            self._git(repo, "add", "app.txt")
            self._git(repo, "commit", "-m", "feature change")
            self._git(repo, "push", "-u", "origin", "feature/test-runtime")
            self._git(repo, "checkout", "-b", "workspace/dev")

            merged_ok, merged_msg, merge_commit = bot._merge_order_branch_to_main(
                repo=repo,
                order_branch="feature/test-runtime",
                order_id="ord-deploy-runtime",
                default_branch="main",
            )
            self.assertTrue(merged_ok, merged_msg)
            assert merge_commit is not None

            cfg = _cfg(td_path / "state.json", workdir=repo)
            repo_record = {
                "repo_id": "repo-main",
                "path": str(repo),
                "default_branch": "main",
                "metadata": {
                    "deploy": {
                        "enabled": True,
                        "source": "repo_script",
                        "cwd": str(repo),
                        "command": ["bash", "scripts/deploy.sh"],
                        "background": False,
                        "timeout_seconds": 30,
                    }
                },
            }
            result = bot._deploy_after_order_merge(
                cfg=cfg,
                repo_record=repo_record,
                repo_dir=repo,
                default_branch="main",
                order_id="ord-deploy-runtime",
                order_branch="feature/test-runtime",
                merge_commit=merge_commit,
            )
            self.assertEqual(str(result.get("status") or ""), "ok")

            runtime_dir = (repo / "data" / "runtime_worktrees" / "main").resolve()
            self.assertTrue(runtime_dir.exists())
            self.assertEqual((runtime_dir / ".deployed_commit").read_text(encoding="utf-8"), str(merge_commit))
            self.assertEqual((runtime_dir / ".deployed_branch").read_text(encoding="utf-8").strip(), "poncebot/runtime/main")
            self.assertEqual(self._git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip(), "workspace/dev")

    def test_deploy_after_merge_uses_fresh_runtime_when_stable_runtime_is_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            remote = td_path / "remote.git"
            repo = td_path / "repo"
            remote.mkdir(parents=True, exist_ok=True)
            repo.mkdir(parents=True, exist_ok=True)

            self._git(remote, "init", "--bare")
            self._git(repo, "init")
            self._git(repo, "checkout", "-b", "main")
            self._git(repo, "remote", "add", "origin", str(remote))
            (repo / "scripts").mkdir(parents=True, exist_ok=True)
            (repo / "scripts" / "deploy.sh").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s' \"${PONCEBOT_DEPLOY_COMMIT}\" > .deployed_commit\n"
                "git rev-parse --abbrev-ref HEAD > .deployed_branch\n",
                encoding="utf-8",
            )
            (repo / "app.txt").write_text("v1\n", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "init")
            self._git(repo, "push", "-u", "origin", "main")

            runtime_dir = (repo / "data" / "runtime_worktrees" / "main").resolve()
            self._git(repo, "worktree", "add", "-B", "poncebot/runtime/main", str(runtime_dir), "origin/main")
            (runtime_dir / "app.txt").write_text("dirty runtime\n", encoding="utf-8")

            self._git(repo, "checkout", "-b", "feature/test-dirty-runtime")
            (repo / "app.txt").write_text("v2\n", encoding="utf-8")
            self._git(repo, "add", "app.txt")
            self._git(repo, "commit", "-m", "feature change")
            self._git(repo, "push", "-u", "origin", "feature/test-dirty-runtime")
            self._git(repo, "checkout", "-b", "workspace/dev")

            merged_ok, merged_msg, merge_commit = bot._merge_order_branch_to_main(
                repo=repo,
                order_branch="feature/test-dirty-runtime",
                order_id="ord-deploy-dirty-runtime",
                default_branch="main",
            )
            self.assertTrue(merged_ok, merged_msg)
            assert merge_commit is not None

            cfg = _cfg(td_path / "state.json", workdir=repo)
            repo_record = {
                "repo_id": "repo-main",
                "path": str(repo),
                "default_branch": "main",
                "metadata": {
                    "deploy": {
                        "enabled": True,
                        "source": "repo_script",
                        "cwd": str(repo),
                        "command": ["bash", "scripts/deploy.sh"],
                        "background": False,
                        "timeout_seconds": 30,
                    }
                },
            }

            result = bot._deploy_after_order_merge(
                cfg=cfg,
                repo_record=repo_record,
                repo_dir=repo,
                default_branch="main",
                order_id="ord-deploy-dirty-runtime",
                order_branch="feature/test-dirty-runtime",
                merge_commit=merge_commit,
            )

            self.assertEqual(str(result.get("status") or ""), "ok")
            self.assertEqual((runtime_dir / "app.txt").read_text(encoding="utf-8"), "dirty runtime\n")
            fresh_dirs = sorted((repo / "data" / "runtime_worktrees").glob("main_deploy_*"))
            deployed_dirs = [p for p in fresh_dirs if (p / ".deployed_commit").exists()]
            self.assertEqual(len(deployed_dirs), 1)
            deployed_dir = deployed_dirs[0]
            self.assertEqual((deployed_dir / ".deployed_commit").read_text(encoding="utf-8"), str(merge_commit))
            self.assertTrue((deployed_dir / ".deployed_branch").read_text(encoding="utf-8").strip().startswith("poncebot/runtime/main-deploy-"))

    def test_auto_merge_unsuspends_when_new_merge_ready_signal_arrives(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            repo_path = td_path / "repo"
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / ".git").mkdir(parents=True, exist_ok=True)

            cfg = _cfg(td_path / "state.json", workdir=repo_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            q.upsert_repo(
                repo_id="repo-secondary",
                path=str(repo_path),
                default_branch="main",
                autonomy_enabled=True,
                priority=1,
                runtime_mode="ceo-bounded",
                daily_budget=0.0,
                status="active",
                metadata={},
            )

            order_id = "ord-merge-retry-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Retry merge after new verified improvement",
                body="[repo:repo-secondary]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                project_id="proj-1",
            )
            q.set_order_phase(order_id, chat_id=1, phase="ready_for_merge")
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={
                        "order_branch": "feature/repo-secondary",
                        "repo_id": "repo-secondary",
                        "repo_path": str(repo_path),
                        "merge_ready": True,
                        "merge_ready_at": 1000.0,
                        "proactive_improvement_closed": True,
                        "proactive_improvement_closed_at": 1000.0,
                        "merge_auto_attempts": 6,
                        "merge_auto_suspended": True,
                        "merge_auto_suspended_at": 900.0,
                        "merge_auto_failed_at": 900.0,
                        "merge_auto_error": "max_attempts_reached",
                    },
                    job_id=order_id,
                )
            )

            with patch.object(bot, "_jarvis_auto_approve_merge_enabled", return_value=True), patch.object(
                bot,
                "_sync_order_phase_from_runtime",
                return_value=None,
            ), patch.object(
                bot,
                "_order_trace_requires_merge",
                return_value=(True, "feature/repo-secondary"),
            ), patch.object(
                bot,
                "_order_command_text",
                return_value="Order ord-merge-retry-01 auto-merged by Jarvis to main",
            ), patch.object(bot, "_send_chunked_text", return_value=None):
                merged = bot._auto_merge_ready_orders_tick(
                    cfg=cfg,
                    api=object(),
                    orch_q=q,
                    now=1001.0,
                )

            self.assertEqual(merged, 1)
            root = q.get_job(order_id)
            assert root is not None
            trace = dict(root.trace or {})
            self.assertEqual(int(trace.get("merge_auto_attempts") or 0), 0)
            self.assertFalse(bool(trace.get("merge_auto_suspended", False)))
            self.assertIn("new_merge_ready_signal", str(trace.get("merge_auto_resume_reason") or ""))
            self.assertIsNone(trace.get("merge_auto_failed_at"))

    def test_auto_merge_tick_heals_recent_done_merge_ready_order_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            repo_path = td_path / "repo"
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / ".git").mkdir(parents=True, exist_ok=True)

            cfg = _cfg(td_path / "state.json", workdir=repo_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            q.upsert_repo(
                repo_id="repo-secondary",
                path=str(repo_path),
                default_branch="main",
                autonomy_enabled=True,
                priority=1,
                runtime_mode="ceo-bounded",
                daily_budget=0.0,
                status="active",
                metadata={},
            )

            order_id = "ord-merge-heal-default-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Heal recent done merge-ready order",
                body="[repo:repo-secondary]",
                status="done",
                priority=1,
                intent_type="order_project_new",
                project_id="proj-1",
                phase="done",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={
                        "order_branch": "feature/repo-secondary",
                        "repo_id": "repo-secondary",
                        "repo_path": str(repo_path),
                        "merge_ready": True,
                        "merge_ready_at": 1000.0,
                        "proactive_no_change_validated": True,
                    },
                    job_id=order_id,
                )
            )

            with patch.object(bot, "_jarvis_auto_approve_merge_enabled", return_value=True), patch.object(
                bot,
                "_sync_order_phase_from_runtime",
                return_value=None,
            ), patch.object(
                bot,
                "_order_trace_requires_merge",
                return_value=(True, "feature/repo-secondary"),
            ), patch.object(
                bot,
                "_order_command_text",
                return_value="Order ord-merge-heal-default-01 auto-merged by Jarvis to main",
            ), patch.object(bot, "_send_chunked_text", return_value=None):
                merged = bot._auto_merge_ready_orders_tick(
                    cfg=cfg,
                    api=object(),
                    orch_q=q,
                    now=1001.0,
                )

            self.assertEqual(merged, 1)
            order = q.get_order(order_id, chat_id=1)
            root = q.get_job(order_id)
            assert order is not None and root is not None
            self.assertEqual(str(order.get("status") or ""), "active")
            self.assertEqual(str(order.get("phase") or ""), "ready_for_merge")
            self.assertEqual(float((root.trace or {}).get("merge_reopened_for_auto_merge_at") or 0.0), 1001.0)

    def test_auto_merge_tick_requeues_stale_conflict_to_skynet_and_clears_ready(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            repo_path = td_path / "repo"
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / ".git").mkdir(parents=True, exist_ok=True)

            cfg = _cfg(td_path / "state.json", workdir=repo_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            q.upsert_repo(
                repo_id="repo-secondary",
                path=str(repo_path),
                default_branch="main",
                autonomy_enabled=True,
                priority=1,
                runtime_mode="ceo-bounded",
                daily_budget=0.0,
                status="active",
                metadata={},
            )

            order_id = "ord-merge-conflict-heal-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Heal stale merge conflict",
                body="[repo:repo-secondary]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                project_id="proj-1",
                phase="planning",
            )
            q.set_order_phase(order_id, chat_id=1, phase="ready_for_merge")
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={
                        "order_branch": "feature/repo-secondary",
                        "repo_id": "repo-secondary",
                        "repo_path": str(repo_path),
                        "merge_ready": True,
                        "merge_ready_at": 1000.0,
                        "merge_auto_failed_at": 995.0,
                        "merge_auto_error": "CONFLICT (content): Merge conflict in bot.py",
                    },
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="jarvis",
                    input_text="legacy merge conflict replan",
                    request_type="review",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="done",
                    labels={"ticket": order_id, "kind": "merge_conflict_replan"},
                    trace={"merge_error": "old conflict"},
                    job_id="legacy-jarvis-replan",
                )
            )

            class _API:
                def __init__(self) -> None:
                    self.msgs: list[str] = []

                def send_message(self, _chat_id: int, text: str, reply_to_message_id=None) -> None:
                    self.msgs.append(text)

            api = _API()
            with patch.object(bot, "_jarvis_auto_approve_merge_enabled", return_value=True), patch.object(
                bot,
                "_sync_order_phase_from_runtime",
                return_value=None,
            ), patch.object(
                bot,
                "_order_trace_requires_merge",
                return_value=(True, "feature/repo-secondary"),
            ):
                merged = bot._auto_merge_ready_orders_tick(
                    cfg=cfg,
                    api=api,
                    orch_q=q,
                    now=1200.0,
                )

            self.assertEqual(merged, 0)
            order = q.get_order(order_id, chat_id=1)
            root = q.get_job(order_id)
            assert order is not None and root is not None
            self.assertEqual(str(order.get("phase") or ""), "planning")
            self.assertFalse(bool((root.trace or {}).get("merge_ready", False)))
            self.assertTrue(bool((root.trace or {}).get("merge_conflict_active", False)))
            children = [row for row in q.jobs_by_parent(parent_job_id=order_id, limit=20) if row.job_id != order_id]
            skynet_replans = [
                row for row in children
                if str((row.labels or {}).get("kind") or "").strip() == "merge_conflict_replan"
                and str(row.role or "") == "skynet"
            ]
            self.assertEqual(len(skynet_replans), 1)
            self.assertTrue(bool((skynet_replans[0].trace or {}).get("merge_conflict_resolution", False)))

    def test_sync_order_phase_clears_stale_merge_error_when_ready_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            repo_path = td_path / "repo"
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / ".git").mkdir(parents=True, exist_ok=True)

            cfg = _cfg(td_path / "state.json", workdir=repo_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            q.upsert_repo(
                repo_id="repo-secondary",
                path=str(repo_path),
                default_branch="main",
                autonomy_enabled=True,
                priority=1,
                runtime_mode="ceo-bounded",
                daily_budget=0.0,
                status="active",
                metadata={},
            )

            order_id = "ord-merge-ready-reset-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: codexbot Reliability + Delivery",
                body="[repo:repo-secondary]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                project_id="proj-1",
                phase="planning",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={
                        "order_branch": "feature/repo-secondary",
                        "repo_id": "repo-secondary",
                        "repo_path": str(repo_path),
                        "proactive_improvement_verified": True,
                        "merge_auto_error": "CONFLICT (content): Merge conflict in bot.py",
                        "merge_auto_failed_at": 900.0,
                        "merge_conflict_active": True,
                    },
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="implementer_local",
                    input_text="implemented slice",
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
                    labels={"ticket": order_id, "kind": "subtask", "key": "local_impl_guard_slice1"},
                    trace={
                        "slice_id": "slice1",
                        "slice_patch_applied": True,
                        "slice_validation_ok": True,
                        "local_patch_info": {"changed_files": ["bot.py"], "validation_ok": True},
                    },
                    job_id="impl-verified",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="reviewer_local",
                    input_text="reviewed slice",
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
                    labels={"ticket": order_id, "kind": "subtask", "key": "local_review_guard_slice1"},
                    trace={"slice_id": "slice1", "result_summary": "PASS READY"},
                    job_id="review-verified",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="verified improvement",
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
                    labels={"ticket": order_id, "kind": "autopilot"},
                    trace={"result_summary": "PASS: VERIFIED_IMPROVEMENT", "improvement_verified": True, "slice_id": "slice1"},
                    job_id="child-verified",
                )
            )

            with patch.object(bot, "_order_trace_requires_merge", return_value=(True, "feature/repo-secondary")):
                bot._sync_order_phase_from_runtime(
                    orch_q=q,
                    root_ticket=order_id,
                    chat_id=1,
                )

            order = q.get_order(order_id, chat_id=1)
            root = q.get_job(order_id)
            assert order is not None and root is not None
            trace = dict(root.trace or {})
            self.assertEqual(str(order.get("phase") or ""), "ready_for_merge")
            self.assertTrue(bool(trace.get("merge_ready", False)))
            self.assertIsNone(trace.get("merge_error"))
            self.assertIsNone(trace.get("merge_failed_at"))
            self.assertIsNone(trace.get("merge_auto_error"))
            self.assertIsNone(trace.get("merge_auto_failed_at"))
            self.assertFalse(bool(trace.get("merge_conflict_active", False)))
            self.assertIsNone(trace.get("merge_conflict_replan_exhausted_at"))

    def test_sync_order_phase_ignores_controller_verified_summary_without_local_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            repo_path = td_path / "repo"
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / ".git").mkdir(parents=True, exist_ok=True)

            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            q.upsert_repo(
                repo_id="repo-secondary",
                path=str(repo_path),
                default_branch="main",
                autonomy_enabled=True,
                priority=1,
                runtime_mode="ceo-bounded",
                daily_budget=0.0,
                status="active",
                metadata={},
            )

            order_id = "ord-summary-only-verified-01"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: codexbot Reliability + Delivery",
                body="[repo:repo-secondary]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                project_id="proj-1",
                phase="planning",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={
                        "order_branch": "feature/repo-secondary",
                        "repo_id": "repo-secondary",
                        "repo_path": str(repo_path),
                    },
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="verified improvement",
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
                    labels={"ticket": order_id, "kind": "autopilot"},
                    trace={"result_summary": "PASS: VERIFIED_IMPROVEMENT"},
                    job_id="child-summary-only",
                )
            )

            with patch.object(bot, "_order_trace_requires_merge", return_value=(True, "feature/repo-secondary")):
                bot._sync_order_phase_from_runtime(
                    orch_q=q,
                    root_ticket=order_id,
                    chat_id=1,
                )

            order = q.get_order(order_id, chat_id=1)
            root = q.get_job(order_id)
            assert order is not None and root is not None
            trace = dict(root.trace or {})
            self.assertNotEqual(str(order.get("phase") or ""), "ready_for_merge")
            self.assertFalse(bool(trace.get("merge_ready", False)))

    def test_auto_merge_tick_clears_stale_merge_error_after_branch_reconciled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            repo_path = td_path / "repo"
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / ".git").mkdir(parents=True, exist_ok=True)

            cfg = _cfg(td_path / "state.json", workdir=repo_path)
            storage = SQLiteTaskStorage(td_path / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            q.upsert_repo(
                repo_id="repo-secondary",
                path=str(repo_path),
                default_branch="main",
                autonomy_enabled=True,
                priority=1,
                runtime_mode="ceo-bounded",
                daily_budget=0.0,
                status="active",
                metadata={},
            )

            order_id = "ord-merge-error-heal-02"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Heal stale merge error after reconcile",
                body="[repo:repo-secondary]",
                status="active",
                priority=1,
                intent_type="order_project_new",
                project_id="proj-1",
                phase="ready_for_merge",
            )
            q.set_order_phase(order_id, chat_id=1, phase="ready_for_merge")
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    trace={
                        "order_branch": "feature/repo-secondary",
                        "repo_id": "repo-secondary",
                        "repo_path": str(repo_path),
                        "merge_ready": True,
                        "merge_ready_at": 1000.0,
                        "proactive_improvement_closed_at": 1000.0,
                        "merge_failed_at": 900.0,
                        "merge_error": "CONFLICT (content): Merge conflict in bot.py",
                        "merge_conflict_active": True,
                    },
                    job_id=order_id,
                )
            )

            class _API:
                def __init__(self) -> None:
                    self.msgs: list[str] = []

                def send_message(self, _chat_id: int, text: str, reply_to_message_id=None) -> None:
                    self.msgs.append(text)

            api = _API()
            with patch.object(bot, "_jarvis_auto_approve_merge_enabled", return_value=True), patch.object(
                bot,
                "_sync_order_phase_from_runtime",
                return_value=None,
            ), patch.object(
                bot,
                "_order_trace_requires_merge",
                return_value=(True, "feature/repo-secondary"),
            ), patch.object(
                bot,
                "_repo_context_for_order",
                return_value=({"repo_id": "repo-secondary", "path": str(repo_path)}, repo_path, "main"),
            ), patch.object(
                bot,
                "_git_pick_main_ref",
                return_value="origin/main",
            ), patch.object(
                bot,
                "_git_remote_branch_ref",
                return_value="origin/feature/repo-secondary",
            ), patch.object(
                bot,
                "_git_is_ancestor",
                return_value=True,
            ), patch.object(
                bot,
                "_order_command_text",
                return_value="Order ord-merg merged to main. commit=abc123",
            ) as merge_cmd:
                merged = bot._auto_merge_ready_orders_tick(
                    cfg=cfg,
                    api=api,
                    orch_q=q,
                    now=1200.0,
                )

            self.assertEqual(merged, 1)
            self.assertEqual(merge_cmd.call_count, 1)
            root = q.get_job(order_id)
            assert root is not None
            trace = dict(root.trace or {})
            self.assertIsNone(trace.get("merge_error"))
            self.assertIsNone(trace.get("merge_failed_at"))
            self.assertIsNone(trace.get("merge_auto_error"))
            self.assertFalse(bool(trace.get("merge_conflict_active", False)))

class TestFactoryRepoOrderGuardrail(unittest.TestCase):
    def test_duplicate_active_proactive_orders_for_same_repo_pause_newer_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            cfg = _cfg(Path(td) / "state.json", workdir=Path(td))

            primary_order = "ord-proactive-primary"
            secondary_order = "ord-proactive-secondary"
            repo_tag = "codexbot-abc12345"

            for oid in (primary_order, secondary_order):
                q.upsert_order(
                    order_id=oid,
                    chat_id=1,
                    title="Proactive Sprint: codexbot Reliability + Delivery",
                    body=f"AUTONOMOUS PROACTIVE SPRINT\n[proactive:repo_{repo_tag}]\n[repo:{repo_tag}]",
                    status="active",
                    priority=1,
                    intent_type="order_project_new",
                    phase="review",
                    project_id=repo_tag,
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
                        trace={
                            "proactive_lane": True,
                            "repo_id": repo_tag,
                            "order_branch": f"feature/{oid}",
                            "merge_ready": True,
                        },
                        job_id=oid,
                    )
                )

            q.submit_task(
                Task.new(
                    source="telegram",
                    role="jarvis",
                    input_text="resolve merge conflict",
                    request_type="review",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=primary_order,
                    state="running",
                    labels={"ticket": primary_order, "kind": "merge_conflict_replan"},
                    job_id="job-primary-running",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="jarvis",
                    input_text="secondary follow-up",
                    request_type="review",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=secondary_order,
                    state="queued",
                    labels={"ticket": secondary_order, "kind": "merge_conflict_replan"},
                    job_id="job-secondary-queued",
                )
            )

            paused = bot._collapse_duplicate_proactive_repo_orders_tick(
                cfg=cfg,
                orch_q=q,
                now=1234.0,
            )

            self.assertEqual(paused, 1)
            primary = q.get_order(primary_order, chat_id=1)
            secondary = q.get_order(secondary_order, chat_id=1)
            assert primary is not None and secondary is not None
            self.assertEqual(str(primary.get("status")), "active")
            self.assertEqual(str(secondary.get("status")), "paused")
            self.assertEqual(str(secondary.get("phase")), "paused")

            secondary_job = q.get_job("job-secondary-queued")
            assert secondary_job is not None
            self.assertEqual(str(secondary_job.state), "cancelled")

            root = q.get_job(secondary_order)
            assert root is not None
            self.assertEqual(str((root.trace or {}).get("repo_conflict_primary") or ""), primary_order)


class TestBulkCancel(unittest.TestCase):
    def test_skynet_local_only_scrub_cancels_non_local_children_for_proactive_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            storage = SQLiteTaskStorage(db)
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            cfg = _cfg(Path(td) / "state.json")

            order_id = "11111111-1111-1111-1111-111111111111"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: codexbot Reliability + Delivery",
                body="Ship one bounded improvement. [repo:codexbot-12345678]",
                status="active",
                priority=2,
                phase="delegated",
                project_id="codexbot-12345678",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="root",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="backend work",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="blocked_approval",
                    labels={"ticket": order_id, "kind": "subtask", "key": "backend_fix"},
                    job_id="22222222-2222-2222-2222-222222222222",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="qa",
                    input_text="qa work",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="waiting_deps",
                    labels={"ticket": order_id, "kind": "subtask", "key": "qa_gate"},
                    job_id="33333333-3333-3333-3333-333333333333",
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="architect_local",
                    input_text="local plan",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="queued",
                    labels={"ticket": order_id, "kind": "subtask", "key": "local_arch_guard_1"},
                    job_id="44444444-4444-4444-4444-444444444444",
                )
            )

            with patch.dict(os.environ, {"BOT_SKYNET_FACTORY_LOCAL_ONLY": "1"}, clear=False):
                cancelled = bot._cancel_non_local_children_for_skynet_factory_orders_tick(
                    orch_q=q,
                    now=1234.0,
                    chat_id=1,
                )

            self.assertEqual(cancelled, 2)
            backend_job = q.get_job("22222222-2222-2222-2222-222222222222")
            qa_job = q.get_job("33333333-3333-3333-3333-333333333333")
            local_job = q.get_job("44444444-4444-4444-4444-444444444444")
            root_job = q.get_job(order_id)
            assert backend_job is not None and qa_job is not None and local_job is not None and root_job is not None
            self.assertEqual(str(backend_job.state), "cancelled")
            self.assertEqual(str(qa_job.state), "cancelled")
            self.assertEqual(str(local_job.state), "queued")
            self.assertEqual(int((root_job.trace or {}).get("skynet_local_only_scrubbed_children") or 0), 2)

    def test_skynet_local_only_scrub_revives_idle_proactive_order_when_no_local_jobs_remain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            storage = SQLiteTaskStorage(db)
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            cfg = _cfg(Path(td) / "state.json")

            order_id = "aaaa1111-1111-1111-1111-111111111111"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: codexbot Reliability + Delivery",
                body="Ship one bounded improvement. [repo:codexbot-12345678]",
                status="active",
                priority=2,
                phase="review",
                project_id="codexbot-12345678",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="AUTONOMOUS PROACTIVE SPRINT",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    trace={"proactive_lane": True, "initiative_key": "repo_codexbot-12345678", "autopilot_last_enqueued_at": 999.0},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="backend",
                    input_text="legacy backend work",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="waiting_deps",
                    labels={"ticket": order_id, "kind": "subtask", "key": "legacy_backend"},
                    job_id="bbbb2222-2222-2222-2222-222222222222",
                )
            )

            with patch.dict(os.environ, {"BOT_SKYNET_FACTORY_LOCAL_ONLY": "1"}, clear=False):
                cancelled = bot._cancel_non_local_children_for_skynet_factory_orders_tick(
                    orch_q=q,
                    now=1234.0,
                    chat_id=1,
                )

            self.assertEqual(cancelled, 1)
            backend_job = q.get_job("bbbb2222-2222-2222-2222-222222222222")
            root_job = q.get_job(order_id)
            order = q.get_order(order_id, chat_id=1)
            assert backend_job is not None and root_job is not None and order is not None
            self.assertEqual(str(backend_job.state), "cancelled")
            self.assertEqual(str(order.get("phase") or ""), "planning")
            self.assertTrue(bool((root_job.trace or {}).get("local_only_revive_requested", False)))
            self.assertEqual(float((root_job.trace or {}).get("autopilot_last_enqueued_at") or 0.0), 0.0)

    def test_sla_tick_cancels_non_local_legacy_job_instead_of_enqueuing_non_local_replan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            storage = SQLiteTaskStorage(db)
            q = OrchestratorQueue(storage=storage, role_profiles=None)
            cfg = _cfg(Path(td) / "state.json")

            order_id = "cccc3333-3333-3333-3333-333333333333"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: codexbot Reliability + Delivery",
                body="Ship one bounded improvement. [repo:codexbot-12345678]",
                status="active",
                priority=2,
                phase="delegated",
                project_id="codexbot-12345678",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="skynet",
                    input_text="AUTONOMOUS PROACTIVE SPRINT",
                    request_type="maintenance",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    trace={"proactive_lane": True, "initiative_key": "repo_codexbot-12345678"},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="release_mgr",
                    input_text="legacy release gate",
                    request_type="task",
                    priority=1,
                    model="",
                    effort="",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    parent_job_id=order_id,
                    state="waiting_deps",
                    labels={"ticket": order_id, "kind": "subtask", "key": "legacy_release"},
                    ttl_seconds=60,
                    created_at=100.0,
                    job_id="dddd4444-4444-4444-4444-444444444444",
                )
            )

            created = bot._sla_overdue_tick(
                cfg=cfg,
                orch_q=q,
                profiles=None,
                now=600.0,
            )

            self.assertEqual(created, 0)
            legacy_job = q.get_job("dddd4444-4444-4444-4444-444444444444")
            order = q.get_order(order_id, chat_id=1)
            root_job = q.get_job(order_id)
            assert legacy_job is not None and order is not None and root_job is not None
            self.assertEqual(str(legacy_job.state), "cancelled")
            self.assertEqual(str(order.get("phase") or ""), "planning")
            self.assertEqual(float((root_job.trace or {}).get("autopilot_last_enqueued_at") or 0.0), 0.0)
            open_children = q.jobs_by_parent(parent_job_id=order_id, limit=20)
            self.assertFalse(
                any(
                    str((row.labels or {}).get("kind") or "").strip().lower() == "sla_replan"
                    and str(row.state or "").strip().lower() in {"queued", "waiting_deps", "blocked_approval", "running"}
                    for row in open_children
                )
            )

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


class _FakeProposalAPI:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, int | None, int]] = []

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        disable_web_page_preview: bool = True,
    ) -> int | None:
        message_id = 100 + len(self.sent)
        self.sent.append((int(chat_id), str(text), (None if reply_to_message_id is None else int(reply_to_message_id)), int(message_id)))
        return int(message_id)


class TestFactoryCeoStrategyApproval(unittest.TestCase):
    def _profiles(self) -> dict[str, dict[str, object]]:
        return {
            "skynet": {
                "name": "Skynet",
                "model": "gpt-5.4",
                "effort": "high",
                "mode_hint": "ro",
                "max_runtime_seconds": 1200,
            }
        }

    def _seed_pending_proposal(
        self,
        td: str,
    ) -> tuple[bot.BotConfig, OrchestratorQueue, _FakeProposalAPI, str, str, dict[str, object]]:
        root = Path(td)
        cfg = _cfg(root / "state.json", workdir=root)
        repo_dir = root / "repo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        storage = SQLiteTaskStorage(root / "jobs.sqlite")
        q = OrchestratorQueue(storage=storage, role_profiles=None)
        api = _FakeProposalAPI()

        order_id = "11111111-1111-1111-1111-111111111111"
        source_job_id = "22222222-2222-2222-2222-222222222222"
        root_task = Task.new(
            source="telegram",
            role="skynet",
            input_text="AUTONOMOUS PROACTIVE SPRINT",
            request_type="maintenance",
            priority=1,
            model="gpt-5.4",
            effort="high",
            mode_hint="ro",
            requires_approval=False,
            max_cost_window_usd=1.0,
            chat_id=1,
            state="done",
            is_autonomous=True,
            trace={
                "allow_delegation": True,
                "proactive_lane": True,
                "repo_id": "codexbot-6fb8d5b9",
                "repo_path": str(repo_dir),
                "factory_order": True,
            },
            job_id=order_id,
        )
        q.submit_task(root_task)
        q.upsert_order(
            order_id=order_id,
            chat_id=1,
            title="Proactive Sprint: codexbot Reliability + Delivery",
            body="AUTONOMOUS PROACTIVE SPRINT\n[repo:codexbot-6fb8d5b9]",
            status="active",
            priority=2,
            phase="planning",
        )
        source_task = Task.new(
            source="telegram",
            role="skynet",
            input_text="Need approval for a larger plan.",
            request_type="maintenance",
            priority=1,
            model="gpt-5.4",
            effort="high",
            mode_hint="ro",
            requires_approval=False,
            max_cost_window_usd=1.0,
            chat_id=1,
            state="blocked_approval",
            is_autonomous=True,
            parent_job_id=order_id,
            labels={"ticket": order_id, "kind": "autopilot"},
            trace={
                "allow_delegation": True,
                "proactive_lane": True,
                "repo_id": "codexbot-6fb8d5b9",
                "repo_path": str(repo_dir),
                "factory_order": True,
            },
            job_id=source_job_id,
        )
        q.submit_task(source_task)
        proposal = bot._register_factory_ceo_strategy_proposal(
            cfg=cfg,
            api=api,
            orch_q=q,
            task=source_task,
            summary="Skynet thinks a larger capability is worth building.",
            structured_digest={
                "next_action": {
                    "type": "ceo_approval_needed",
                    "proposal_title": "Add proactive strategic ideation lane",
                    "proposal_summary": "Expand the factory so it can propose and execute larger initiatives with CEO approval.",
                    "scope": ["new strategic proposal flow", "CEO timeout fallback"],
                    "risks": ["more scope risk", "approval latency"],
                    "questions": ["Do we want this enabled now?"],
                    "fallback_work": "Keep doing bounded reliability and delivery improvements.",
                }
            },
        )
        assert proposal is not None
        return cfg, q, api, order_id, source_job_id, proposal

    def test_register_strategy_proposal_sets_pending_order_phase(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg, q, api, order_id, _source_job_id, proposal = self._seed_pending_proposal(td)
            pending = bot._pending_factory_ceo_strategy_proposal_for_order(cfg, order_id=order_id, now=_time.time())
            self.assertIsNotNone(pending)
            order = q.get_order(order_id, chat_id=1)
            self.assertIsNotNone(order)
            assert order is not None
            self.assertEqual(str(order.get("phase") or ""), "ceo_approval_pending")
            root_job = q.get_job(order_id)
            assert root_job is not None
            self.assertTrue(bool((root_job.trace or {}).get("ceo_strategy_approval_pending", False)))
            self.assertTrue(api.sent)
            self.assertIn("Skynet requests CEO approval", api.sent[-1][1])
            self.assertEqual(int(proposal.get("message_id") or 0), api.sent[-1][3])

    def test_reply_approval_queues_followup_and_clears_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg, q, api, order_id, source_job_id, proposal = self._seed_pending_proposal(td)
            handled = bot._maybe_handle_factory_ceo_strategy_message(
                cfg=cfg,
                api=api,
                orch_q=q,
                profiles=self._profiles(),
                incoming=bot.IncomingMessage(
                    update_id=1,
                    chat_id=1,
                    user_id=2,
                    message_id=555,
                    username="ceo",
                    text="si",
                    reply_to_message_id=int(proposal.get("message_id") or 0),
                    reply_to_text="proposal",
                ),
            )
            self.assertTrue(handled)
            self.assertIsNone(
                bot._pending_factory_ceo_strategy_proposal_for_order(cfg, order_id=order_id, now=_time.time())
            )
            source_job = q.get_job(source_job_id)
            assert source_job is not None
            self.assertEqual(str(source_job.state), "done")
            followups = [
                row
                for row in q.jobs_by_parent(parent_job_id=order_id, limit=20)
                if str((row.labels or {}).get("kind") or "") == "ceo_strategy_followup"
            ]
            self.assertEqual(len(followups), 1)
            self.assertEqual(str(followups[0].state), "queued")
            self.assertEqual(str(followups[0].role), "skynet")

    def test_timeout_tick_queues_bounded_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg, q, api, order_id, source_job_id, _proposal = self._seed_pending_proposal(td)
            bot._update_factory_ceo_strategy_records(
                cfg,
                lambda records: records[order_id].update({"expires_at": _time.time() - 5.0}),
            )
            resolved = bot._factory_ceo_strategy_timeout_tick(
                cfg=cfg,
                api=api,
                orch_q=q,
                profiles=self._profiles(),
                now=_time.time(),
            )
            self.assertEqual(resolved, 1)
            self.assertIsNone(
                bot._pending_factory_ceo_strategy_proposal_for_order(cfg, order_id=order_id, now=_time.time())
            )
            source_job = q.get_job(source_job_id)
            assert source_job is not None
            self.assertEqual(str(source_job.state), "done")
            followups = [
                row
                for row in q.jobs_by_parent(parent_job_id=order_id, limit=20)
                if str((row.labels or {}).get("kind") or "") == "ceo_strategy_followup"
            ]
            self.assertEqual(len(followups), 1)
            self.assertEqual(str((followups[0].labels or {}).get("decision") or ""), "timeout")

    def test_autopilot_skip_when_strategy_proposal_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg, q, _api, order_id, _source_job_id, _proposal = self._seed_pending_proposal(td)
            order_row = q.get_order(order_id, chat_id=1)
            assert order_row is not None
            created = bot._enqueue_order_autopilot_task(
                cfg=cfg,
                orch_q=q,
                profiles=self._profiles(),
                order_row=order_row,
                chat_id=1,
                now=_time.time(),
                reason="proactive_autopilot",
                min_idle_seconds=0.0,
                cooldown_seconds=0.0,
            )
            self.assertFalse(created)


class TestSkynetLocalRecovery(unittest.TestCase):
    def _profiles(self) -> dict[str, dict[str, object]]:
        return {
            "skynet": {
                "name": "Skynet",
                "model": "gpt-5.3-codex",
                "effort": "medium",
                "mode_hint": "ro",
                "max_runtime_seconds": 1200,
            }
        }

    def _seed_factory_order(self, td: str) -> tuple[bot.BotConfig, OrchestratorQueue, str, Path]:
        root = Path(td)
        cfg = _cfg(root / "state.json", workdir=root)
        storage = SQLiteTaskStorage(root / "jobs.sqlite")
        q = OrchestratorQueue(storage=storage, role_profiles=None)
        repo_dir = root / "repo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        order_id = "99999999-1111-1111-1111-111111111111"
        q.upsert_order(
            order_id=order_id,
            chat_id=1,
            title="Proactive Sprint: codexbot Reliability + Delivery",
            body="Ship one bounded improvement. [repo:codexbot-6fb8d5b9]",
            status="active",
            priority=2,
            phase="review",
            project_id="codexbot-6fb8d5b9",
        )
        q.submit_task(
            Task.new(
                source="telegram",
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
                    "repo_id": "codexbot-6fb8d5b9",
                    "repo_path": str(repo_dir),
                    "factory_order": True,
                },
                job_id=order_id,
            )
        )
        return cfg, q, order_id, repo_dir

    def _blocked_final_sweep_task(self, *, order_id: str, repo_dir: Path, state: str = "blocked") -> Task:
        return Task.new(
            source="telegram",
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
            state=state,
            is_autonomous=True,
            parent_job_id=order_id,
            labels={"ticket": order_id, "kind": "final_sweep"},
            trace={
                "allow_delegation": True,
                "proactive_lane": True,
                "repo_id": "codexbot-6fb8d5b9",
                "repo_path": str(repo_dir),
                "result_next_action": "delegate_local_subtask",
                "result_summary": (
                    "Write policy violation: skynet modified repository files directly. "
                    "Skynet factory work must delegate code changes to local specialists instead of editing in the controller lane."
                ),
                "structured_digest": {
                    "write_policy_violation": {
                        "role": "skynet",
                        "reason": "controller_write_policy_violation",
                        "changed_paths": ["bot.py"],
                    },
                    "next_action": {
                        "type": "delegate_now",
                        "subtasks": [
                            {
                                "key": "local_arch_guard_1773946787",
                                "role": "architect_local",
                                "text": "Plan one bounded recovery slice for codexbot.",
                                "acceptance_criteria": ["Name exact files and one validation command."],
                                "definition_of_done": ["Architect handoff is actionable for implementer_local."],
                            }
                        ],
                    },
                },
            },
            job_id="aaaa1111-bbbb-cccc-dddd-eeeeffff0000",
        )

    def test_autopilot_defers_proactive_order_with_pending_branch_to_merge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_obj, q, order_id, _repo_dir = self._seed_factory_order(td)
            q.update_trace(order_id, order_branch="feature/order-pending-branch", merged_to_main=False)
            order_row = q.get_order(order_id, chat_id=1)
            assert order_row is not None

            created = bot._enqueue_order_autopilot_task(
                cfg=cfg_obj,
                orch_q=q,
                profiles=self._profiles(),
                order_row=order_row,
                chat_id=1,
                now=_time.time(),
                reason="proactive_idle_watchdog",
                min_idle_seconds=0.0,
                cooldown_seconds=0.0,
            )

            self.assertFalse(created)
            order = q.get_order(order_id, chat_id=1)
            self.assertIsNotNone(order)
            assert order is not None
            self.assertEqual(str(order.get("status") or ""), "active")
            self.assertEqual(str(order.get("phase") or ""), "ready_for_merge")
            root = q.get_job(order_id)
            self.assertIsNotNone(root)
            assert root is not None
            self.assertTrue(bool((root.trace or {}).get("merge_ready", False)))
            self.assertTrue(bool((root.trace or {}).get("autopilot_deferred_to_merge", False)))
            autopilots = [
                child
                for child in q.jobs_by_parent(parent_job_id=order_id, limit=20)
                if str((child.labels or {}).get("kind") or "").strip().lower() == "autopilot"
            ]
            self.assertEqual(autopilots, [])

    def test_sync_order_phase_preserves_explicit_merge_ready_without_live_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _cfg_obj, q, order_id, _repo_dir = self._seed_factory_order(td)
            q.update_trace(
                order_id,
                order_branch="feature/order-pending-branch",
                merge_ready=True,
                merge_ready_at=1000.0,
                merged_to_main=False,
            )
            q.set_order_phase(order_id, chat_id=1, phase="ready_for_merge")
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="implementer_local",
                    input_text="old terminal failure",
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
                    trace={"result_status": "failed", "result_summary": "old failed attempt"},
                    job_id="job-terminal-failure-merge-ready",
                )
            )

            bot._sync_order_phase_from_runtime(orch_q=q, root_ticket=order_id, chat_id=1)

            order = q.get_order(order_id, chat_id=1)
            self.assertIsNotNone(order)
            assert order is not None
            self.assertEqual(str(order.get("status") or ""), "active")
            self.assertEqual(str(order.get("phase") or ""), "ready_for_merge")

    def test_blocked_controller_write_policy_violation_requests_local_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _cfg_obj, _q, order_id, repo_dir = self._seed_factory_order(td)
            task = self._blocked_final_sweep_task(order_id=order_id, repo_dir=repo_dir)
            trace = dict((task.trace or {}))
            structured = trace.get("structured_digest")
            self.assertTrue(
                bot._task_requests_local_controller_recovery(
                    task,
                    orch_state="blocked",
                    next_action="delegate_local_subtask",
                    structured_digest=structured,
                )
            )

    def test_blocked_controller_write_policy_violation_without_subtasks_still_requests_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _cfg_obj, _q, order_id, repo_dir = self._seed_factory_order(td)
            task = self._blocked_final_sweep_task(order_id=order_id, repo_dir=repo_dir)
            trace = dict((task.trace or {}))
            trace["structured_digest"] = {
                "write_policy_violation": {
                    "role": "skynet",
                    "reason": "controller_write_policy_violation",
                    "changed_paths": ["test_status_http.py"],
                }
            }
            task = task.with_updates(trace=trace)
            self.assertTrue(
                bot._task_requests_local_controller_recovery(
                    task,
                    orch_state="blocked",
                    next_action="delegate_local_subtask",
                    structured_digest=trace["structured_digest"],
                )
            )

    def test_sync_order_phase_ignores_recoverable_blocked_controller_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _cfg_obj, q, order_id, repo_dir = self._seed_factory_order(td)
            q.submit_task(self._blocked_final_sweep_task(order_id=order_id, repo_dir=repo_dir))
            q.submit_task(
                Task.new(
                    source="telegram",
                    role="architect_local",
                    input_text="Plan a bounded local slice.",
                    request_type="task",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="queued",
                    is_autonomous=True,
                    parent_job_id=order_id,
                    labels={"ticket": order_id, "kind": "subtask", "key": "local_arch_guard_1773946787"},
                    trace={"slice_id": "1773946787", "slice_status": "planned"},
                    job_id="bbbb1111-bbbb-cccc-dddd-eeeeffff0000",
                )
            )

            bot._sync_order_phase_from_runtime(orch_q=q, root_ticket=order_id, chat_id=1)

            order = q.get_order(order_id, chat_id=1)
            self.assertIsNotNone(order)
            assert order is not None
            self.assertEqual(str(order.get("phase") or ""), "delegated")

    def test_final_sweep_ignores_recoverable_blocked_controller_job_when_no_other_work_is_live(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_obj, q, order_id, repo_dir = self._seed_factory_order(td)
            q.submit_task(self._blocked_final_sweep_task(order_id=order_id, repo_dir=repo_dir))

            created = bot._jarvis_final_sweep_tick(
                cfg=cfg_obj,
                orch_q=q,
                profiles=None,
                now=_time.time(),
            )

            self.assertEqual(created, 0)
            children = q.jobs_by_parent(parent_job_id=order_id, limit=20)
            final_sweeps = [
                child for child in children
                if str((child.labels or {}).get("kind") or "").strip().lower() == "final_sweep"
                and child.job_id != "aaaa1111-bbbb-cccc-dddd-eeeeffff0000"
            ]
            self.assertEqual(final_sweeps, [])

    def test_final_sweep_auto_closes_proactive_terminal_only_order_after_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_obj, q, order_id, _repo_dir = self._seed_factory_order(td)
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
            with q._storage._conn() as conn:
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
                created = bot._jarvis_final_sweep_tick(
                    cfg=cfg_obj,
                    orch_q=q,
                    profiles=self._profiles(),
                    now=now,
                )

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

    def test_final_sweep_moves_terminal_only_proactive_order_with_branch_to_merge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_obj, q, order_id, _repo_dir = self._seed_factory_order(td)
            q.update_trace(order_id, order_branch="feature/order-pending-merge", merged_to_main=False)
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
                    state="done",
                    job_id="job-terminal-branch-1",
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
                    job_id="job-terminal-branch-2",
                )
            )

            now = _time.time()
            stale_ts = now - 1200.0
            with q._storage._conn() as conn:
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
                created = bot._jarvis_final_sweep_tick(
                    cfg=cfg_obj,
                    orch_q=q,
                    profiles=self._profiles(),
                    now=now,
                )

            self.assertEqual(created, 0)
            order = q.get_order(order_id, chat_id=1)
            self.assertIsNotNone(order)
            assert order is not None
            self.assertEqual(str(order.get("status") or ""), "active")
            self.assertEqual(str(order.get("phase") or ""), "ready_for_merge")
            root = q.get_job(order_id)
            self.assertIsNotNone(root)
            assert root is not None
            self.assertTrue(bool((root.trace or {}).get("merge_ready", False)))
            self.assertTrue(bool((root.trace or {}).get("final_sweep_terminal_only_ready_for_merge", False)))
            self.assertFalse(bool((root.trace or {}).get("final_sweep_terminal_only_closed", False)))

    def test_final_sweep_default_terminal_threshold_closes_at_ten(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_obj, q, order_id, _repo_dir = self._seed_factory_order(td)
            for i in range(10):
                q.submit_task(
                    Task.new(
                        source="telegram",
                        role="implementer_local",
                        input_text=f"attempt {i + 1}",
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
                        job_id=f"job-terminal-default-10-{i + 1}",
                    )
                )

            now = _time.time()
            stale_ts = now - 1200.0
            with q._storage._conn() as conn:
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
                    "BOT_JARVIS_FINAL_SWEEP_COOLDOWN_SECONDS": "1",
                },
                clear=False,
            ):
                os.environ.pop("BOT_JARVIS_IDLE_TERMINAL_CLOSE_THRESHOLD", None)
                created = bot._jarvis_final_sweep_tick(
                    cfg=cfg_obj,
                    orch_q=q,
                    profiles=self._profiles(),
                    now=now,
                )

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

    def test_final_sweep_default_terminal_threshold_does_not_close_at_nine(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_obj, q, order_id, _repo_dir = self._seed_factory_order(td)
            for i in range(9):
                q.submit_task(
                    Task.new(
                        source="telegram",
                        role="implementer_local",
                        input_text=f"attempt {i + 1}",
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
                        job_id=f"job-terminal-default-9-{i + 1}",
                    )
                )

            now = _time.time()
            stale_ts = now - 1200.0
            with q._storage._conn() as conn:
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
                    "BOT_JARVIS_FINAL_SWEEP_COOLDOWN_SECONDS": "1",
                },
                clear=False,
            ):
                os.environ.pop("BOT_JARVIS_IDLE_TERMINAL_CLOSE_THRESHOLD", None)
                created = bot._jarvis_final_sweep_tick(
                    cfg=cfg_obj,
                    orch_q=q,
                    profiles=self._profiles(),
                    now=now,
                )

            self.assertEqual(created, 1)
            order = q.get_order(order_id, chat_id=1)
            self.assertIsNotNone(order)
            assert order is not None
            self.assertEqual(str(order.get("status") or ""), "active")
            self.assertEqual(str(order.get("phase") or ""), "planning")
            root = q.get_job(order_id)
            self.assertIsNotNone(root)
            assert root is not None
            self.assertFalse(bool((root.trace or {}).get("final_sweep_terminal_only_closed", False)))

    def test_stale_blocked_controller_job_is_cancelled_for_local_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _cfg_obj, q, order_id, repo_dir = self._seed_factory_order(td)
            blocked = self._blocked_final_sweep_task(order_id=order_id, repo_dir=repo_dir)
            q.submit_task(blocked)
            stale_ts = float(_time.time() - 7200.0)
            with q._storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ?, stalled_since = ? WHERE job_id = ?",
                    (stale_ts, stale_ts, stale_ts, blocked.job_id),
                )
                conn.commit()

            with patch.dict(os.environ, {"BOT_HYGIENE_STALE_BLOCKED_SECONDS": "60"}, clear=False):
                cleaned = bot._cleanup_stale_blocked_jobs(orch_q=q, now=_time.time())

            self.assertEqual(cleaned, 1)
            refreshed = q.get_job(blocked.job_id)
            self.assertIsNotNone(refreshed)
            assert refreshed is not None
            self.assertEqual(refreshed.state, "cancelled")
            self.assertTrue(bool((refreshed.trace or {}).get("stale_controller_local_recovery", False)))
            order = q.get_order(order_id, chat_id=1)
            self.assertIsNotNone(order)
            assert order is not None
            self.assertEqual(str(order.get("phase") or ""), "review")

    def test_stale_blocked_controller_job_without_stalled_since_is_cancelled_for_local_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _cfg_obj, q, order_id, repo_dir = self._seed_factory_order(td)
            blocked = self._blocked_final_sweep_task(order_id=order_id, repo_dir=repo_dir)
            q.submit_task(blocked)
            stale_ts = float(_time.time() - 7200.0)
            with q._storage._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ?, stalled_since = NULL WHERE job_id = ?",
                    (stale_ts, stale_ts, blocked.job_id),
                )
                conn.commit()

            with patch.dict(os.environ, {"BOT_HYGIENE_STALE_BLOCKED_SECONDS": "60"}, clear=False):
                cleaned = bot._cleanup_stale_blocked_jobs(orch_q=q, now=_time.time())

            self.assertEqual(cleaned, 1)
            refreshed = q.get_job(blocked.job_id)
            self.assertIsNotNone(refreshed)
            assert refreshed is not None
            self.assertEqual(refreshed.state, "cancelled")
            self.assertTrue(bool((refreshed.trace or {}).get("stale_controller_local_recovery", False)))

    def test_autopilot_skip_when_no_progress_backoff_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td) / "state.json", workdir=Path(td))
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            q = OrchestratorQueue(storage=storage, role_profiles=None)

            order_id = "11111111-1111-1111-1111-111111111111"
            now = _time.time()
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: codexbot Reliability + Delivery",
                body="Ship one bounded improvement. [repo:codexbot-6fb8d5b9]",
                status="active",
                priority=2,
                phase="planning",
                project_id="codexbot-6fb8d5b9",
            )
            q.submit_task(
                Task.new(
                    source="telegram",
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
                    is_autonomous=True,
                    trace={
                        "allow_delegation": True,
                        "proactive_lane": True,
                        "repo_id": "codexbot-6fb8d5b9",
                        "autopilot_no_progress_backoff_until": float(now + 600.0),
                    },
                    job_id=order_id,
                )
            )

            order_row = q.get_order(order_id, chat_id=1)
            assert order_row is not None
            created = bot._enqueue_order_autopilot_task(
                cfg=cfg,
                orch_q=q,
                profiles=self._profiles(),
                order_row=order_row,
                chat_id=1,
                now=now,
                reason="proactive_autopilot",
                min_idle_seconds=0.0,
                cooldown_seconds=0.0,
            )
            self.assertFalse(created)
