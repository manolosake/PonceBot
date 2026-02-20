import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
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

    def test_plain_greeting_is_direct_response(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="Hola")
            resp, job = bot._parse_job(cfg, msg)
            self.assertIn("Jarvis:", resp)
            self.assertIsNone(job)


    def test_ack_is_silent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="Ok")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "")
            self.assertIsNone(job)

    def test_dashboard_all_returns_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/dashboard all")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "__orch_dashboard:all")
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
            self.assertTrue(bool((task.trace or {}).get("suppress_ticket_card", False)))
            self.assertEqual(int((task.trace or {}).get("max_runtime_seconds") or 0), 120)


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
            self.assertIn("Default (current)", resp)

    def test_permissions_set_full_persists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            cfg = self._cfg(state_file)
            msg = bot.IncomingMessage(update_id=1, chat_id=1, user_id=2, message_id=10, username="u", text="/permissions full")
            resp, job = bot._parse_job(cfg, msg)
            self.assertEqual(resp, "OK. permissions=full")
            self.assertIsNone(job)
            saved = json.loads(state_file.read_text())
            self.assertEqual(saved["access_mode_by_chat"]["1"], "full")

            msg2 = bot.IncomingMessage(update_id=2, chat_id=1, user_id=2, message_id=11, username="u", text="/permissions")
            resp2, job2 = bot._parse_job(cfg, msg2)
            self.assertIsNone(job2)
            self.assertIn("Full access (current)", resp2)


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
        with patch.object(bot.subprocess, "Popen") as popen:
            popen.side_effect = FileNotFoundError("no codex in test")
            with self.assertRaises(FileNotFoundError):
                runner.start_threaded_new(prompt="hi", mode_hint="ro")
            cmd = popen.call_args.args[0]
            self.assertIn("--dangerously-bypass-approvals-and-sandbox", cmd)
            self.assertNotIn("--sandbox", cmd)

    def test_threaded_resume_injects_dangerous_bypass_and_omits_sandbox(self) -> None:
        cfg = self._cfg()
        runner = bot.CodexRunner(cfg)
        with patch.object(bot.subprocess, "Popen") as popen:
            popen.side_effect = FileNotFoundError("no codex in test")
            with self.assertRaises(FileNotFoundError):
                runner.start_threaded_resume(thread_id="t_123", prompt="hi", mode_hint="rw")
            cmd = popen.call_args.args[0]
            self.assertIn("--dangerously-bypass-approvals-and-sandbox", cmd)
            self.assertNotIn("--sandbox", cmd)


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
