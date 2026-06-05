from __future__ import annotations

import json
import os
import shutil
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


class TestFactoryRepoHygiene(unittest.TestCase):
    def _git(self, repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        repo.mkdir()
        self._git(repo, "init", "-b", "main")
        self._git(repo, "config", "user.name", "Tester")
        self._git(repo, "config", "user.email", "tester@example.com")
        (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
        self._git(repo, "add", "README.md")
        self._git(repo, "commit", "-m", "init")
        return repo

    def test_factory_autocleans_only_ephemeral_untracked_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = self._repo(Path(td))
            preview = repo / ".codexbot_preview"
            preview.mkdir()
            (preview / "screenshot.png").write_bytes(b"png")
            output = repo / "output" / "playwright"
            output.mkdir(parents=True)
            (output / "trace.json").write_text("{}", encoding="utf-8")

            blocker = bot._factory_repo_autonomy_blocker(repo, default_branch="main")

            self.assertEqual("", blocker)
            self.assertFalse((preview / "screenshot.png").exists())
            self.assertFalse(output.exists())

    def test_factory_keeps_untracked_preview_html_as_material_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = self._repo(Path(td))
            preview = repo / ".codexbot_preview"
            preview.mkdir()
            (preview / "preview.html").write_text("<h1>Ship me</h1>", encoding="utf-8")

            blocker = bot._factory_repo_autonomy_blocker(repo, default_branch="main")

            self.assertIn("repo checkout has uncommitted changes", blocker)
            self.assertTrue((preview / "preview.html").exists())

    def test_no_runtime_deploy_policy_is_validated_checkout_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = self._repo(Path(td))
            result = bot._deploy_after_order_merge(
                cfg=SimpleNamespace(codex_workdir=Path(td) / "codexbot"),
                repo_record={"repo_id": "demo", "metadata": {}},
                repo_dir=repo,
                default_branch="main",
                order_id="order-1",
                order_branch="feature/order-1-demo",
                merge_commit="abc123",
            )

            self.assertEqual("ok", result["status"])
            self.assertEqual("validated_checkout", result["reason"])
            self.assertNotIn("skipped", result["summary"].lower())

    def test_no_delta_controller_snapshot_never_counts_as_published_project(self) -> None:
        self.assertEqual(
            "rejected_low_value",
            bot._controller_snapshot_no_delta_outcome_status(
                {
                    "result_status": "published_project",
                    "result_summary": "Private GitHub repo exists, but no material patch was produced.",
                    "result_next_action": "published_project",
                }
            ),
        )


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

    def test_factory_focus_policy_disables_dated_incubator_containers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "20260525-studio-cycle-new-product-incubator-abcdef12"
            repo.mkdir(parents=True)
            row = {
                "repo_id": "incubator-abcdef12",
                "path": str(repo),
                "autonomy_enabled": True,
                "priority": 3,
                "status": "active",
                "metadata": {"repo_name": repo.name},
            }

            bot._factory_apply_repo_focus_policy([row], now=123.0)

        self.assertFalse(bool(row["autonomy_enabled"]))
        self.assertEqual(row["status"], "disabled")
        self.assertIn("dated Studio incubator container", row["metadata"]["portfolio_focus_reason"])

    def test_factory_focus_policy_disables_temporary_checkouts_even_if_primary_named(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "ExecutiveDashboard.tmp"
            repo.mkdir(parents=True)
            row = {
                "repo_id": "executivedashboard-tmp",
                "path": str(repo),
                "autonomy_enabled": True,
                "priority": 1,
                "status": "active",
                "metadata": {"repo_name": repo.name},
            }

            bot._factory_apply_repo_focus_policy([row], now=123.0)

        self.assertFalse(bool(row["autonomy_enabled"]))
        self.assertEqual(row["status"], "disabled")
        self.assertIn("temporary checkout", row["metadata"]["portfolio_focus_reason"])

    def test_factory_focus_policy_caps_active_portfolio_scope(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = []
            for name in ("alpha-desk", "beta-desk", "gamma-desk"):
                repo = root / name
                repo.mkdir(parents=True)
                (repo / "README.md").write_text(f"# {name}\n", encoding="utf-8")
                rows.append(
                    {
                        "repo_id": name,
                        "path": str(repo),
                        "autonomy_enabled": True,
                        "priority": 3,
                        "status": "active",
                        "metadata": {"repo_name": name},
                    }
                )

            with patch.dict(os.environ, {"BOT_FACTORY_ACTIVE_PORTFOLIO_FOCUS_CAP": "2"}), patch.object(
                bot, "_factory_repo_origin_url", side_effect=lambda path: f"git@github.com:manolosake/{Path(path).name}.git"
            ), patch.object(bot, "_factory_repo_last_commit_ts", return_value=1.0):
                bot._factory_apply_repo_focus_policy(rows, now=123.0)

        enabled = [row["repo_id"] for row in rows if bool(row["autonomy_enabled"])]
        disabled = [row["repo_id"] for row in rows if not bool(row["autonomy_enabled"])]
        self.assertEqual(enabled, ["alpha-desk", "beta-desk"])
        self.assertEqual(disabled, ["gamma-desk"])
        self.assertIn("outside active portfolio cap 2", rows[2]["metadata"]["portfolio_focus_reason"])

    def test_factory_repo_default_branch_sync_reconciles_temporary_branch(self) -> None:
        metadata = {
            "repo_name": "ExecutiveDashboard",
            "proactive_branch_note": "Use the live deploy branch; origin/main is divergent on this host.",
        }

        branch = bot._factory_repo_default_branch_for_sync(
            existing={"default_branch": "codex/r530-main-clean-20260305-045022"},
            discovered={"default_branch": "main"},
            metadata=metadata,
            now=123.0,
        )

        self.assertEqual(branch, "main")
        self.assertEqual(metadata["canonical_branch"], "main")
        self.assertEqual(metadata["previous_default_branch"], "codex/r530-main-clean-20260305-045022")
        self.assertEqual(metadata["default_branch_migrated_by"], "factory_sync")
        self.assertNotIn("proactive_branch_note", metadata)

    def test_factory_repo_default_branch_sync_preserves_pinned_branch(self) -> None:
        metadata = {"default_branch_pinned": True}

        branch = bot._factory_repo_default_branch_for_sync(
            existing={"default_branch": "codex/codexbot-workflow-v2"},
            discovered={"default_branch": "main"},
            metadata=metadata,
            now=123.0,
        )

        self.assertEqual(branch, "codex/codexbot-workflow-v2")
        self.assertNotIn("canonical_branch", metadata)

    def test_factory_repo_policy_priority_promotes_primary_repos(self) -> None:
        base_repo = Path("/home/aponce/codexbot")

        self.assertEqual(bot._factory_repo_policy_priority(Path("/home/aponce/codexbot"), base_repo=base_repo), 1)
        self.assertEqual(bot._factory_repo_policy_priority(Path("/home/aponce/ExecutiveDashboard"), base_repo=base_repo), 1)
        self.assertEqual(bot._factory_repo_policy_priority(Path("/home/aponce/OmniCrewApp.android"), base_repo=base_repo), 2)
        self.assertEqual(bot._factory_repo_policy_priority(Path("/home/aponce/Documents/ReceiptJury"), base_repo=base_repo), 3)

    def test_factory_sync_disables_focus_gated_repo_even_if_existing_was_active(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "gamma-desk"
            repo.mkdir()
            cfg = self._cfg(root / "state.json")
            repo_id = "gamma-desk"
            discovered = {
                "repo_id": repo_id,
                "path": str(repo.resolve()),
                "default_branch": "main",
                "autonomy_enabled": False,
                "priority": 3,
                "runtime_mode": "ceo-bounded",
                "daily_budget": 0.0,
                "status": "disabled",
                "metadata": {
                    "repo_name": "gamma-desk",
                    "portfolio_focus_reason": "portfolio focus gate: outside active portfolio cap 2",
                },
            }

            class _FakeQueue:
                def __init__(self) -> None:
                    self.rows = {repo_id: {**discovered, "autonomy_enabled": True, "status": "active"}}
                    self.events: list[dict[str, object]] = []

                def list_repos(self, limit: int = 5000):
                    return list(self.rows.values())[:limit]

                def upsert_repo(self, **kwargs: object) -> None:
                    self.rows[str(kwargs["repo_id"])] = dict(kwargs)

                def set_repo_status(self, *, repo_id: str, status: str) -> None:
                    self.rows[repo_id]["status"] = status

                def append_audit_event(self, **kwargs: object) -> None:
                    self.events.append(dict(kwargs))

            q = _FakeQueue()
            with patch.object(bot, "_discover_factory_repos", return_value=[discovered]):
                rows = bot._factory_sync_repo_registry(cfg=cfg, orch_q=q, now=123.0, force=True)  # type: ignore[arg-type]

        synced = {str(row.get("repo_id")): row for row in rows}[repo_id]
        self.assertFalse(bool(synced["autonomy_enabled"]))
        self.assertEqual(synced["status"], "disabled")
        self.assertIn("portfolio focus gate", synced["metadata"]["last_autonomy_blocker"])

    def test_ceo_plane_routes_plain_on_demand_text_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            job = bot.Job(
                chat_id=1,
                reply_to_message_id=2,
                user_text="Mejora el dashboard sin tocar la proactividad",
                argv=["exec", "Mejora el dashboard sin tocar la proactividad"],
                mode_hint="full",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            )

            with patch.dict(os.environ, {"BOT_CEO_PLANE_ROUTE_ENABLED": "1"}):
                self.assertTrue(bot._should_route_to_ceo_plane(cfg, job))

            with patch.dict(os.environ, {"BOT_CEO_PLANE_ROUTE_ENABLED": "0"}):
                self.assertFalse(bot._should_route_to_ceo_plane(cfg, job))

    def test_ceo_plane_does_not_route_local_commands_or_artifact_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td) / "state.json")
            command_job = bot.Job(
                chat_id=1,
                reply_to_message_id=2,
                user_text="/orders",
                argv=["orders"],
                mode_hint="ro",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[],
                force_new_thread=False,
            )
            upload_job = bot.Job(
                chat_id=1,
                reply_to_message_id=2,
                user_text="revisa este archivo",
                argv=["exec", "revisa este archivo"],
                mode_hint="ro",
                epoch=0,
                threaded=True,
                image_paths=[],
                upload_paths=[Path("/tmp/file.txt")],
                force_new_thread=False,
            )

            with patch.dict(os.environ, {"BOT_CEO_PLANE_ROUTE_ENABLED": "1"}):
                self.assertFalse(bot._should_route_to_ceo_plane(cfg, command_job))
                self.assertFalse(bot._should_route_to_ceo_plane(cfg, upload_job))

    def test_factory_sync_recovers_blocked_repo_when_preflight_clears(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "ExecutiveDashboard"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "manolosake"], cwd=str(repo), check=True)
            subprocess.run(["git", "config", "user.email", "manolosake@gmail.com"], cwd=str(repo), check=True)
            subprocess.run(["git", "checkout", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "README.md").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=str(repo), check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=str(repo), check=True, capture_output=True, text=True)

            cfg = self._cfg(root / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "codex_workdir": root / "codexbot"})
            repo_id = "executivedashboard-12345678"
            discovered = {
                "repo_id": repo_id,
                "path": str(repo.resolve()),
                "default_branch": "main",
                "autonomy_enabled": True,
                "priority": 2,
                "runtime_mode": "ceo-bounded",
                "daily_budget": 0.0,
                "status": "active",
                "metadata": {"repo_name": "ExecutiveDashboard"},
            }

            class _FakeQueue:
                def __init__(self) -> None:
                    self.rows = {
                        repo_id: {
                            **discovered,
                            "status": "blocked",
                            "metadata": {
                                "repo_name": "ExecutiveDashboard",
                                "last_autonomy_blocker": "repo checkout branch 'feature/x' differs from configured default_branch 'main'",
                            },
                        }
                    }
                    self.events: list[dict[str, object]] = []

                def list_repos(self, limit: int = 5000):
                    return list(self.rows.values())[:limit]

                def upsert_repo(self, **kwargs: object) -> None:
                    self.rows[str(kwargs["repo_id"])] = dict(kwargs)

                def set_repo_status(self, *, repo_id: str, status: str) -> None:
                    self.rows[repo_id]["status"] = status

                def append_audit_event(self, **kwargs: object) -> None:
                    self.events.append(dict(kwargs))

            q = _FakeQueue()
            with patch.object(bot, "_discover_factory_repos", return_value=[discovered]):
                rows = bot._factory_sync_repo_registry(cfg=cfg, orch_q=q, now=123.0, force=True)  # type: ignore[arg-type]

        synced = {str(row.get("repo_id")): row for row in rows}[repo_id]
        self.assertEqual(synced["status"], "active")
        self.assertEqual(synced["priority"], 1)
        self.assertNotIn("last_autonomy_blocker", synced["metadata"])
        self.assertEqual(synced["metadata"]["autonomy_recovered_from"], "blocked")
        self.assertTrue(any(event.get("event_type") == "repo.status_reconciled" for event in q.events))

    def test_factory_sync_blocks_repo_when_preflight_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "ExecutiveDashboard"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "manolosake"], cwd=str(repo), check=True)
            subprocess.run(["git", "config", "user.email", "manolosake@gmail.com"], cwd=str(repo), check=True)
            subprocess.run(["git", "checkout", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "README.md").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=str(repo), check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "README.md").write_text("dirty\n", encoding="utf-8")

            cfg = self._cfg(root / "state.json")
            cfg = bot.BotConfig(**{**cfg.__dict__, "codex_workdir": root / "codexbot"})
            repo_id = "executivedashboard-12345678"
            discovered = {
                "repo_id": repo_id,
                "path": str(repo.resolve()),
                "default_branch": "main",
                "autonomy_enabled": True,
                "priority": 2,
                "runtime_mode": "ceo-bounded",
                "daily_budget": 0.0,
                "status": "active",
                "metadata": {"repo_name": "ExecutiveDashboard"},
            }

            class _FakeQueue:
                def __init__(self) -> None:
                    self.rows = {repo_id: dict(discovered)}
                    self.events: list[dict[str, object]] = []

                def list_repos(self, limit: int = 5000):
                    return list(self.rows.values())[:limit]

                def upsert_repo(self, **kwargs: object) -> None:
                    self.rows[str(kwargs["repo_id"])] = dict(kwargs)

                def set_repo_status(self, *, repo_id: str, status: str) -> None:
                    self.rows[repo_id]["status"] = status

                def append_audit_event(self, **kwargs: object) -> None:
                    self.events.append(dict(kwargs))

            q = _FakeQueue()
            with patch.object(bot, "_discover_factory_repos", return_value=[discovered]):
                rows = bot._factory_sync_repo_registry(cfg=cfg, orch_q=q, now=123.0, force=True)  # type: ignore[arg-type]

        synced = {str(row.get("repo_id")): row for row in rows}[repo_id]
        self.assertEqual(synced["status"], "blocked")
        self.assertIn("uncommitted changes", synced["metadata"]["last_autonomy_blocker"])
        self.assertTrue(any(event.get("event_type") == "repo.status_reconciled" for event in q.events))

    def test_factory_repo_autonomy_blocker_rejects_branch_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "manolosake"], cwd=str(repo), check=True)
            subprocess.run(["git", "config", "user.email", "manolosake@gmail.com"], cwd=str(repo), check=True)
            subprocess.run(["git", "checkout", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "README.md").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=str(repo), check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "checkout", "-b", "feature/voice-out"], cwd=str(repo), check=True, capture_output=True, text=True)

            reason = bot._factory_repo_autonomy_blocker(repo, default_branch="main")

        self.assertIn("feature/voice-out", reason)
        self.assertIn("default_branch 'main'", reason)

    def test_spawn_proactive_order_blocks_mismatched_repo_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "manolosake"], cwd=str(repo), check=True)
            subprocess.run(["git", "config", "user.email", "manolosake@gmail.com"], cwd=str(repo), check=True)
            subprocess.run(["git", "checkout", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "README.md").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=str(repo), check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "checkout", "-b", "feature/voice-out"], cwd=str(repo), check=True, capture_output=True, text=True)
            cfg = self._cfg(root / "state.json")

            class _FakeQueue:
                def __init__(self) -> None:
                    self.statuses: list[tuple[str, str]] = []
                    self.events: list[dict[str, object]] = []

                def set_repo_status(self, *, repo_id: str, status: str) -> None:
                    self.statuses.append((repo_id, status))

                def append_audit_event(self, **kwargs: object) -> None:
                    self.events.append(dict(kwargs))

                def submit_task(self, *_args: object, **_kwargs: object) -> None:
                    raise AssertionError("blocked repo must not submit a proactive task")

            q = _FakeQueue()
            created = bot._spawn_proactive_order(
                cfg=cfg,
                orch_q=q,  # type: ignore[arg-type]
                profiles={},
                chat_id=1,
                now=123.0,
                initiative={"key": "repo_voice", "title": "Voice Sprint"},
                repo_record={
                    "repo_id": "voice-12345678",
                    "path": str(repo),
                    "default_branch": "main",
                    "autonomy_enabled": True,
                    "priority": 2,
                    "runtime_mode": "ceo-bounded",
                    "daily_budget": 0.0,
                    "status": "active",
                    "metadata": {},
                },
            )

        self.assertFalse(created)
        self.assertEqual(q.statuses, [("voice-12345678", "blocked")])
        self.assertEqual(q.events[0]["event_type"], "factory.repo_autonomy_blocked")
        self.assertIn("feature/voice-out", str(q.events[0]["details"]))

    def test_spawn_proactive_order_requires_feature_bias_and_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "manolosake"], cwd=str(repo), check=True)
            subprocess.run(["git", "config", "user.email", "manolosake@gmail.com"], cwd=str(repo), check=True)
            subprocess.run(["git", "checkout", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "README.md").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=str(repo), check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=str(repo), check=True, capture_output=True, text=True)
            origin = root / "origin.git"
            subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True, text=True)
            subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=str(repo), check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            cfg = self._cfg(root / "state.json")

            class _FakeQueue:
                def __init__(self) -> None:
                    self.submitted: list[bot.Task] = []
                    self.orders: list[dict[str, object]] = []
                    self.events: list[dict[str, object]] = []

                def submit_task(self, task: bot.Task) -> None:
                    self.submitted.append(task)

                def upsert_order(self, **kwargs: object) -> None:
                    self.orders.append(dict(kwargs))

                def update_trace(self, *_args: object, **_kwargs: object) -> None:
                    pass

                def append_audit_event(self, **kwargs: object) -> None:
                    self.events.append(dict(kwargs))

                def upsert_agent_runtime_state(self, **_kwargs: object) -> None:
                    pass

                def append_agent_mailbox(self, **_kwargs: object) -> None:
                    pass

            q = _FakeQueue()
            created = bot._spawn_proactive_order(
                cfg=cfg,
                orch_q=q,  # type: ignore[arg-type]
                profiles={},
                chat_id=1,
                now=123.0,
                initiative={
                    "key": "repo_dashboard",
                    "title": "Proactive Sprint: Dashboard Impact + Product Delivery",
                    "goal": "Ship visible operator value.",
                    "success": "Ship, merge, push, and deploy one visible improvement.",
                },
                repo_record={
                    "repo_id": "dashboard-12345678",
                    "path": str(repo),
                    "default_branch": "main",
                    "autonomy_enabled": True,
                    "priority": 2,
                    "runtime_mode": "ceo-bounded",
                    "daily_budget": 0.0,
                    "status": "active",
                    "metadata": {},
                },
            )

        self.assertTrue(created)
        self.assertEqual(len(q.submitted), 1)
        prompt = q.submitted[0].input_text
        self.assertIn("Required work classification", prompt)
        self.assertIn("If no P0/P1 bug is evidenced", prompt)
        self.assertIn("A valid feature must be notable", prompt)
        self.assertIn("merged to the repo default branch, pushed, and deployed", prompt)
        self.assertIn("Do not mark branch-only work as done", prompt)
        self.assertEqual(q.orders[0]["title"], "Proactive Sprint: Dashboard Impact + Product Delivery")

    def test_local_slice_expected_validation_cmd_prefers_unittest_for_test_module_files(self) -> None:
        cmd = bot._local_slice_expected_validation_cmd(["tests/test_scheduler.py", "bot.py"])
        self.assertEqual(cmd, "python3 -m unittest -q tests/test_scheduler.py")

    def test_local_slice_expected_validation_cmd_uses_pytest_bootstrap_when_unittest_not_applicable(self) -> None:
        cmd = bot._local_slice_expected_validation_cmd(["tests/conftest.py"])
        self.assertEqual(cmd, "./scripts/bootstrap_pytest_python3.sh -m pytest -q tests/conftest.py")

    def test_local_slice_expected_validation_cmd_keeps_py_compile_for_non_test_targets(self) -> None:
        cmd = bot._local_slice_expected_validation_cmd(["bot.py", "tools/health.py", "README.md"])
        self.assertEqual(cmd, "python3 -m py_compile bot.py tools/health.py")

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

    def test_github_token_can_fallback_to_git_credential_store(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            cred_path = home / ".config" / "omnicrew" / "git-credentials"
            cred_path.parent.mkdir(parents=True, exist_ok=True)
            cred_path.write_text("https://x-access-token:ghp_from_store_123@github.com\n", encoding="utf-8")

            with patch.dict(os.environ, {"HOME": str(home), "GITHUB_TOKEN": "", "GH_TOKEN": ""}, clear=False):
                token, source = bot._github_token_from_env_or_git_credentials()

        self.assertEqual(token, "ghp_from_store_123")
        self.assertTrue(source.endswith("git-credentials"))

    def test_project_incubator_github_repo_name_prefers_readme_title(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "20260511-studio-cycle-new-product-incubator-abcd1234"
            project.mkdir(parents=True, exist_ok=True)
            (project / "PROJECT_MANIFEST.json").write_text(
                json.dumps({"name": "Studio Cycle New Product Incubator"}),
                encoding="utf-8",
            )
            (project / "README.md").write_text("# BidPulse Local\n\nDemo project.\n", encoding="utf-8")

            self.assertEqual(bot._project_incubator_github_repo_name(project), "bidpulse-local")

    def test_publish_project_incubator_pushes_existing_github_remote_without_api(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "bidpulse-local"
            project.mkdir(parents=True, exist_ok=True)
            (project / ".git").mkdir()
            (project / "README.md").write_text("# BidPulse Local\n", encoding="utf-8")

            calls: list[list[str]] = []

            def fake_run_git(_path: Path, args: list[str], *, check: bool = False):
                calls.append(list(args))
                if args == ["status", "--short"]:
                    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
                if args == ["branch", "--show-current"]:
                    return subprocess.CompletedProcess(args, 0, stdout="main\n", stderr="")
                if args == ["rev-parse", "--short", "HEAD"]:
                    return subprocess.CompletedProcess(args, 0, stdout="abc1234\n", stderr="")
                if args == ["remote", "get-url", "origin"]:
                    return subprocess.CompletedProcess(args, 0, stdout="https://github.com/manolosake/bidpulse-local.git\n", stderr="")
                if args == ["push", "-u", "origin", "main"]:
                    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="unexpected")

            with patch.object(bot, "_run_git", side_effect=fake_run_git), patch.object(
                bot, "_github_token_from_env_or_git_credentials", return_value=("", "")
            ):
                result = bot._publish_project_incubator_private_github(project, description="BidPulse Local")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["reason"], "github_remote_verified_and_pushed")
        self.assertEqual(result["remote_name"], "origin")
        self.assertEqual(result["github_repo"], "manolosake/bidpulse-local")
        self.assertIn(["push", "-u", "origin", "main"], calls)

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

    def test_no_write_roles_are_forced_read_only(self) -> None:
        self.assertTrue(bot._role_requires_enforced_read_only("skynet"))
        self.assertTrue(bot._role_requires_enforced_read_only("jarvis"))
        self.assertTrue(bot._role_requires_enforced_read_only("architect_local"))
        self.assertFalse(bot._role_requires_enforced_read_only("implementer_local"))
        self.assertTrue(bot._role_requires_enforced_read_only("reviewer_local"))

    def test_no_write_forced_mode_can_use_full_when_host_sandbox_is_broken(self) -> None:
        cfg = self._cfg(Path(tempfile.gettempdir()) / "state.json")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(bot._orchestrator_forced_mode_for_role(cfg, role="skynet", chat_id=1), "ro")
        with patch.dict(os.environ, {"BOT_NO_WRITE_ROLE_FORCED_MODE": "full"}):
            self.assertEqual(bot._orchestrator_forced_mode_for_role(cfg, role="skynet", chat_id=1), "full")
        with patch.dict(os.environ, {"BOT_NO_WRITE_ROLE_FORCED_MODE": "danger-full-access"}):
            self.assertEqual(bot._orchestrator_forced_mode_for_role(cfg, role="skynet", chat_id=1), "full")
        with patch.dict(os.environ, {"BOT_NO_WRITE_ROLE_FORCED_MODE": "workspace-write"}):
            self.assertEqual(bot._orchestrator_forced_mode_for_role(cfg, role="reviewer_local", chat_id=1), "rw")
        with patch.dict(os.environ, {"BOT_NO_WRITE_ROLE_FORCED_MODE": "invalid"}):
            self.assertEqual(bot._orchestrator_forced_mode_for_role(cfg, role="jarvis", chat_id=1), "ro")
        self.assertIsNone(bot._orchestrator_forced_mode_for_role(cfg, role="backend", chat_id=1))

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
        self.assertTrue(bot._response_signals_no_code_change("No-op reapplication; patch is already present with no new delta."))

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

    def test_collect_order_local_autonomy_funnel_counts_controller_pass_as_closed(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            role="skynet",
            state="done",
            trace={},
            labels={},
            created_at=90.0,
            updated_at=100.0,
        )
        impl = SimpleNamespace(
            job_id="impl",
            role="backend",
            state="done",
            labels={"key": "local_impl_recover_cli_alias"},
            trace={
                "result_summary": "Applied bounded CLI improvement with validation evidence.",
                "slice_patch_applied": True,
                "slice_validation_ok": True,
                "patch_info": {"changed_files": ["bot.py", "test_bot.py"], "validation_ok": True},
            },
            created_at=110.0,
            updated_at=120.0,
        )
        qa = SimpleNamespace(
            job_id="qa",
            role="qa",
            state="done",
            labels={"key": "local_review_recover_cli_alias"},
            trace={"result_summary": "READY: focused and regression validation passed."},
            created_at=125.0,
            updated_at=130.0,
        )
        controller = SimpleNamespace(
            job_id="review",
            role="skynet",
            state="done",
            labels={},
            trace={"result_summary": "PASS for the backend slice."},
            created_at=135.0,
            updated_at=140.0,
        )

        class FakeQueue:
            def get_job(self, job_id: str):
                return root if job_id == "root" else None

            def jobs_by_parent(self, parent_job_id: str, limit: int = 800):
                return [impl, qa, controller] if parent_job_id == "root" else []

        funnel = bot._collect_order_local_autonomy_funnel(orch_q=FakeQueue(), root_ticket="root", now=150.0)

        self.assertEqual(funnel["quality_gate_status"], "closed")
        self.assertEqual(funnel["slices_closed"], 1)
        self.assertTrue(funnel["improvement_verified"])
        self.assertTrue(bot._order_has_meaningful_improvement(orch_q=FakeQueue(), root_ticket="root"))

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


    def test_controller_local_recovery_specs_synthesizes_local_patch_flow_from_write_policy(self) -> None:
        specs = bot._controller_local_recovery_specs(
            {
                "summary": "Prepared a minimal reliability improvement in test_status_http.py.",
                "order_branch": "feature/test-order",
                "controller_recovery_artifacts": [
                    "/tmp/artifacts/root/changes.patch",
                    "/tmp/artifacts/root/write_policy_controller_snapshot_stash.txt",
                ],
                "write_policy_violation": {
                    "changed_paths": ["test_status_http.py"],
                },
            }
        )
        self.assertEqual([spec.role for spec in specs], ["implementer_local", "reviewer_local"])
        self.assertIn("test_status_http.py", specs[0].text)
        self.assertIn("/tmp/artifacts/root/changes.patch", specs[0].text)
        self.assertIn("Do not use `stash@{0}`", specs[0].text)
        self.assertIn("ticket-scoped no-op evidence", specs[0].text)
        self.assertIn("python3 -m unittest -q test_status_http", specs[0].text)
        self.assertIn("policy-approved no-op", specs[1].text)
        self.assertEqual(specs[1].depends_on, [specs[0].key])

    def test_controller_local_recovery_specs_normalizes_scoped_write_policy_paths(self) -> None:
        specs = bot._controller_local_recovery_specs(
            {
                "summary": "Prepared a bounded controller recovery patch.",
                "write_policy_violation": {
                    "changed_paths": [
                        "controller_snapshot:HEAD:aaa..bbb",
                        "worktree:",
                        "worktree:bot.py",
                        "base_repo:BRANCH:main..feature/bad",
                        "base_repo:tools/health.py",
                    ],
                },
            }
        )
        self.assertEqual([spec.role for spec in specs], ["implementer_local", "reviewer_local"])
        self.assertIn("- `bot.py`", specs[0].text)
        self.assertIn("- `tools/health.py`", specs[0].text)
        self.assertNotIn("worktree:", specs[0].text)
        self.assertNotIn("base_repo:", specs[0].text)
        self.assertNotIn("controller_snapshot:", specs[0].text)
        self.assertNotIn("HEAD:", specs[0].text)
        self.assertNotIn("BRANCH:", specs[0].text)
        self.assertIn("python3 -m py_compile bot.py tools/health.py", specs[0].text)
        self.assertIn("python3 -m py_compile bot.py tools/health.py", specs[1].text)

    def test_controller_local_recovery_routes_unexplained_write_policy_to_architect(self) -> None:
        specs = bot._controller_local_recovery_specs(
            {
                "order_branch": "feature/test-order",
                "controller_recovery_artifacts": ["/tmp/artifacts/root/changes.patch"],
                "write_policy_violation": {
                    "changed_paths": ["controller_snapshot:app/src/main/java/com/example/MainActivity.kt"],
                },
            }
        )

        self.assertEqual([spec.role for spec in specs], ["architect_local"])
        self.assertIn("Do not replay the controller patch", specs[0].text)
        self.assertIn("NO_CODE_CHANGE", specs[0].text)
        self.assertIn("app/src/main/java/com/example/MainActivity.kt", specs[0].text)
        self.assertIn("/tmp/artifacts/root/changes.patch", specs[0].text)

    def test_controller_local_recovery_routes_cli_promotion_to_backend_triage(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BOT_PROACTIVE_ALLOW_CLI_PROMOTION": "1",
                "BOT_SKYNET_FACTORY_LOCAL_ONLY": "0",
            },
            clear=False,
        ):
            specs = bot._controller_local_recovery_specs(
                {
                    "order_branch": "feature/test-order",
                    "controller_recovery_artifacts": ["/tmp/artifacts/root/changes.patch"],
                    "write_policy_violation": {
                        "changed_paths": ["controller_snapshot:server/app.py"],
                    },
                }
            )

        self.assertEqual([spec.role for spec in specs], ["backend", "qa"])
        self.assertEqual(specs[0].mode_hint, "rw")
        self.assertTrue(specs[1].key.startswith("local_review_recover_"))
        self.assertEqual(specs[1].depends_on, [specs[0].key])
        self.assertIn("Do not replay the controller patch", specs[0].text)
        self.assertIn("implement the smallest wired change", specs[0].text)
        self.assertIn("NO_CODE_CHANGE", specs[0].text)
        self.assertIn("Do not fail only because the workspace is not on a `local_*` slice", specs[1].text)

    def test_idle_no_open_jobs_runbook_writes_deterministic_pass_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jobs.sqlite"
            with sqlite3.connect(db) as con:
                con.execute(
                    """
                    CREATE TABLE jobs (
                        job_id TEXT PRIMARY KEY,
                        parent_job_id TEXT,
                        role TEXT,
                        state TEXT,
                        owner TEXT,
                        input_text TEXT,
                        created_at REAL,
                        updated_at REAL,
                        due_at REAL,
                        stalled_since REAL,
                        retry_count INTEGER,
                        max_retries INTEGER
                    )
                    """
                )
                con.execute(
                    "INSERT INTO jobs(job_id, role, state, input_text, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("done-1", "sre", "done", "closed", 10.0, 20.0),
                )

            events: list[dict[str, object]] = []

            class FakeQueue:
                def append_audit_event(self, *, event_type: str, actor: str = "system", details: dict[str, object] | None = None) -> None:
                    events.append({"event_type": event_type, "actor": actor, "details": details or {}})

            cfg = SimpleNamespace(orchestrator_db_path=db, artifacts_root=root / "artifacts")
            fake_proc = SimpleNamespace(stdout="python /home/aponce/codexbot/bot.py\n", returncode=0)

            with patch.object(bot.subprocess, "run", return_value=fake_proc):
                ok, artifacts_dir = bot._run_idle_no_open_jobs_runbook(
                    cfg=cfg,
                    orch_q=FakeQueue(),
                    now=1234.0,
                )

            gate_path = artifacts_dir / "idle_no_open_jobs" / "close_gate.txt"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertTrue(ok)
            self.assertEqual(gate["gate"], "PASS")
            self.assertEqual(gate["status_open_jobs"], 0)
            self.assertEqual(gate["jobs_open_jobs"], 0)
            self.assertTrue((artifacts_dir / "idle_no_open_jobs" / "status_snapshot.json").exists())
            self.assertTrue((artifacts_dir / "idle_no_open_jobs" / "job_liveness.txt").exists())
            self.assertTrue((artifacts_dir / "idle_no_open_jobs" / "process_liveness.txt").exists())
            self.assertEqual(events[0]["event_type"], "runbook.idle_no_open_jobs.deterministic")
            self.assertEqual(events[0]["details"]["gate"], "PASS")

    def test_controller_snapshot_safe_prompt_scrubs_source_paths(self) -> None:
        prompt = (
            "Registered repo root: /srv/codexbot\n"
            "Use /tmp/worktrees/skynet/slot1 for inspection.\n"
        )
        safe = bot._controller_snapshot_safe_prompt(
            prompt,
            snapshot_dir=Path("/tmp/artifacts/job/controller_snapshot"),
            source_paths=[Path("/srv/codexbot"), "/tmp/worktrees/skynet/slot1"],
        )
        self.assertIn("CONTROLLER_SNAPSHOT_MODE", safe)
        self.assertIn("/tmp/artifacts/job/controller_snapshot", safe)
        self.assertNotIn("/srv/codexbot", safe)
        self.assertNotIn("/tmp/worktrees/skynet/slot1", safe)


class TestStudioOutcomeMemory(unittest.TestCase):
    class _FakeQueue:
        def list_orders_global(self, status=None, limit: int = 240):
            return []

    def _insert_cycle_outcome(
        self,
        db: Path,
        *,
        now: float,
        key: str,
        repo_id: str,
        selected_type: str,
        outcome_status: str,
        outcome_summary: str,
    ) -> None:
        bot._studio_ensure_schema(db)
        with sqlite3.connect(db) as conn:
            conn.execute(
                """
                INSERT INTO studio_cycles(
                    cycle_id, version, ts, status, selected_key, selected_type, selected_repo_id,
                    selected_repo_path, selected_lane, thesis, rationale, debate_summary,
                    operator_visible_outcome, evidence_target, risk_summary, prompt_packet,
                    opportunities_json, outcome_status, outcome_summary, order_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"cycle-{key}-{outcome_status}",
                    bot._STUDIO_CYCLE_VERSION,
                    now - 600,
                    "failed",
                    key,
                    selected_type,
                    repo_id,
                    "",
                    "studio",
                    "Improve selection",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "[]",
                    outcome_status,
                    outcome_summary,
                    None,
                    now - 600,
                    now - 600,
                ),
            )
            conn.commit()

    def test_recent_negative_studio_outcome_penalizes_matching_core_opportunity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jobs.sqlite"
            now = 100_000.0
            cfg = SimpleNamespace(orchestrator_db_path=db)
            repos = [
                {
                    "repo_id": "codexbot-core",
                    "path": "",
                    "status": "active",
                    "autonomy_enabled": True,
                    "priority": 1,
                    "metadata": {},
                },
                {
                    "repo_id": "executivedashboard",
                    "path": "",
                    "status": "active",
                    "autonomy_enabled": True,
                    "priority": 1,
                    "metadata": {},
                },
            ]

            without_memory = bot._studio_build_opportunities(
                cfg=cfg,  # type: ignore[arg-type]
                orch_q=self._FakeQueue(),  # type: ignore[arg-type]
                repos=repos,
                occupied_keys=set(),
                occupied_repo_ids=set(),
                now=now,
            )
            self.assertEqual(without_memory[0]["repo_id"], "codexbot-core")

            self._insert_cycle_outcome(
                db,
                now=now,
                key="repo-codexbot-core",
                repo_id="codexbot-core",
                selected_type="DEEP_IMPROVEMENT",
                outcome_status="failed_root_caused",
                outcome_summary="Repeated the same thesis and did not produce a mergeable delta.",
            )

            with_memory = bot._studio_build_opportunities(
                cfg=cfg,  # type: ignore[arg-type]
                orch_q=self._FakeQueue(),  # type: ignore[arg-type]
                repos=repos,
                occupied_keys=set(),
                occupied_repo_ids=set(),
                now=now,
            )

        by_repo = {str(item["repo_id"]): item for item in with_memory}
        self.assertEqual(with_memory[0]["repo_id"], "executivedashboard")
        self.assertLess(by_repo["codexbot-core"]["score"], by_repo["executivedashboard"]["score"])
        self.assertTrue(by_repo["codexbot-core"]["recent_studio_outcome_risks"])
        self.assertIn("recent Studio caution", by_repo["codexbot-core"]["risk_summary"])
        self.assertIn("lower confidence", by_repo["codexbot-core"]["why_better_than_alternatives"])

    def test_prompt_packet_includes_recent_studio_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jobs.sqlite"
            now = 200_000.0
            self._insert_cycle_outcome(
                db,
                now=now,
                key="repo-codexbot-core",
                repo_id="codexbot-core",
                selected_type="DEEP_IMPROVEMENT",
                outcome_status="blocked_need_operator",
                outcome_summary="Needs operator decision before retrying the same repo.",
            )
            memory = bot._studio_selection_memory(
                cfg=SimpleNamespace(orchestrator_db_path=db),  # type: ignore[arg-type]
                orch_q=self._FakeQueue(),  # type: ignore[arg-type]
                now=now,
            )
            selected = {
                "type": "DEEP_IMPROVEMENT",
                "repo_name": "codexbot",
                "score": 58,
                "thesis": "Improve Studio selectivity.",
                "operator_visible_outcome": "A clearer Studio selection loop.",
            }

            packet = bot._studio_cycle_prompt_packet(selected=selected, opportunities=[selected], memory=memory)

        self.assertIn("Recent Studio outcomes", packet)
        self.assertIn("blocked_need_operator repo=codexbot-core type=DEEP_IMPROVEMENT", packet)
        self.assertIn("Needs operator decision", packet)

    def test_prompt_packet_includes_selection_kill_sheet(self) -> None:
        selected = {
            "key": "repo-codexbot",
            "type": "DEEP_IMPROVEMENT",
            "repo_name": "codexbot",
            "score": 120,
            "thesis": "Repair Studio selection.",
            "operator_visible_outcome": "A clearer selection loop.",
        }
        rejected = {
            "key": "new-project-incubator",
            "type": "NEW_PROJECT",
            "repo_name": "New product incubator",
            "score": 26,
            "thesis": "Create another product.",
            "operator_visible_outcome": "A project folder.",
            "risk_summary": (
                "avoid public launch; "
                "Studio Governor incubator_quality_gate: 2 failed/rejected new-project outcomes in 24h.; "
                "new-project quality gate is active"
            ),
            "recent_studio_outcome_risks": [],
            "recent_studio_saturation": [],
        }
        memory = {
            "studio_governor": {
                "mode": "incubator_quality_gate",
                "avoid_keys": ["new-project-incubator"],
                "directives": ["New-project work is cooling down."],
            }
        }

        packet = bot._studio_cycle_prompt_packet(selected=selected, opportunities=[selected, rejected], memory=memory)

        self.assertIn("Selection kill-sheet:", packet)
        self.assertIn("Killed NEW_PROJECT · New product incubator · score 26", packet)
        self.assertIn("incubator_quality_gate", packet)
        self.assertIn("new-project quality gate is active", packet)
        self.assertIn("weaker than selected score 120", packet)

    def test_record_cycle_persists_selection_kill_sheet_in_debate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            selected = {
                "key": "repo-codexbot",
                "type": "DEEP_IMPROVEMENT",
                "repo_name": "codexbot",
                "score": 120,
                "thesis": "Repair Studio selection.",
                "operator_visible_outcome": "A clearer selection loop.",
            }
            rejected = {
                "key": "new-project-incubator",
                "type": "NEW_PROJECT",
                "repo_name": "New product incubator",
                "score": 26,
                "thesis": "Create another product.",
                "operator_visible_outcome": "A project folder.",
                "risk_summary": (
                    "Studio Governor incubator_quality_gate: 2 failed/rejected new-project outcomes in 24h.; "
                    "new-project quality gate is active"
                ),
            }
            memory = {
                "studio_governor": {
                    "mode": "incubator_quality_gate",
                    "avoid_keys": ["new-project-incubator"],
                    "directives": ["New-project work is cooling down."],
                }
            }

            cycle_id = bot._studio_record_cycle(
                cfg=SimpleNamespace(orchestrator_db_path=db),  # type: ignore[arg-type]
                selected=selected,
                opportunities=[selected, rejected],
                memory=memory,
                now=201_000.0,
            )
            with sqlite3.connect(db) as conn:
                debate_summary = conn.execute(
                    "SELECT debate_summary FROM studio_cycles WHERE cycle_id = ?",
                    (cycle_id,),
                ).fetchone()[0]

        self.assertIn("Selection kill-sheet", debate_summary)
        self.assertIn("Killed NEW_PROJECT · New product incubator · score 26", debate_summary)
        self.assertIn("incubator_quality_gate", debate_summary)
        self.assertIn("weaker than selected score 120", debate_summary)

    def test_studio_governor_activates_economic_discipline_on_high_churn(self) -> None:
        now = 500_000.0
        negatives = [
            {
                "key": f"repo-churn-{idx}",
                "repo_id": f"churn-{idx}",
                "type": "FEATURE",
                "lane": "dashboard",
                "status": "rejected_low_value",
                "summary": "Weak bet rejected as low value churn.",
                "updated_at": now - (idx * 600.0),
            }
            for idx in range(7)
        ]
        positives = [
            {
                "key": f"repo-ship-{idx}",
                "repo_id": f"ship-{idx}",
                "type": "FEATURE",
                "lane": "dashboard",
                "status": "shipped_to_main",
                "summary": "Shipped a visible feature to main.",
                "updated_at": now - (idx * 900.0),
            }
            for idx in range(2)
        ]

        governor = bot._studio_governor_assessment(
            {
                "recent_studio_negative_outcomes": negatives,
                "recent_studio_positive_outcomes": positives,
            },
            now=now,
        )

        self.assertEqual(governor["mode"], "economic_discipline_gate")
        self.assertTrue(governor["economic_gate_active"])
        self.assertEqual(governor["economic_recent_negative_count_72h"], 7)
        self.assertEqual(governor["economic_recent_positive_count_72h"], 2)
        self.assertGreater(governor["economic_recent_negative_ratio_72h"], 0.60)
        self.assertIn("Spend Codex only", governor["force_next_action"])

    def test_economic_discipline_filter_removes_weak_candidates_before_codex(self) -> None:
        governor = {
            "mode": "economic_discipline_gate",
            "economic_gate_active": True,
        }
        strong = {
            "key": "repo-codexbot",
            "type": "DEEP_IMPROVEMENT",
            "repo_name": "codexbot",
            "score": 96,
            "factory_value": {"score": 100, "shipability_score": 100},
        }
        low_score = {
            "key": "repo-shallow",
            "type": "PRODUCT_WORKFLOW",
            "repo_name": "Shallow",
            "score": 74,
            "factory_value": {"score": 100, "shipability_score": 100},
        }
        incomplete_value = {
            "key": "repo-no-ship",
            "type": "FEATURE",
            "repo_name": "No Ship",
            "score": 90,
            "factory_value": {"score": 75, "shipability_score": 50},
        }

        filtered = bot._studio_apply_economic_discipline_filter(
            governor,
            [strong, low_score, incomplete_value],
        )

        self.assertEqual(filtered, [strong])
        self.assertIn("economic_discipline_gate rejected candidate", low_score["economic_gate_rejection_reason"])
        self.assertIn("score 74<", low_score["economic_gate_rejection_reason"])
        self.assertIn("factory_value 75<", incomplete_value["economic_gate_rejection_reason"])
        self.assertIn("shipability 50<", incomplete_value["economic_gate_rejection_reason"])

    def test_published_project_outcome_records_portfolio_asset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            now = 225_000.0
            order_id = "order-portfolio-1"
            bot._studio_ensure_schema(db)
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE jobs(job_id TEXT PRIMARY KEY, trace TEXT NOT NULL DEFAULT '{}')")
                conn.execute(
                    "INSERT INTO jobs(job_id, trace) VALUES (?, ?)",
                    (
                        order_id,
                        json.dumps(
                            {
                                "github_publication": {
                                    "ok": True,
                                    "github_repo": "manolosake/lead-offer-brief",
                                    "remote_url": "https://github.com/manolosake/lead-offer-brief.git",
                                    "branch": "main",
                                    "head": "2efec0a",
                                    "private": True,
                                },
                                "project_incubator_delivery": {
                                    "project_path": "/home/aponce/lead-offer-brief",
                                    "project_head": "2efec0a",
                                },
                            }
                        ),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO studio_cycles(
                        cycle_id, version, ts, status, selected_key, selected_type, selected_repo_id,
                        selected_repo_path, selected_lane, thesis, rationale, debate_summary,
                        operator_visible_outcome, evidence_target, risk_summary, prompt_packet,
                        opportunities_json, outcome_status, outcome_summary, order_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "cycle-portfolio",
                        bot._STUDIO_CYCLE_VERSION,
                        now,
                        "active",
                        "new-project-incubator",
                        "NEW_PROJECT",
                        "",
                        "",
                        "incubator",
                        "Create a sellable project",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "[]",
                        "",
                        "",
                        order_id,
                        now,
                        now,
                    ),
                )
                conn.commit()

            bot._studio_complete_cycle_for_order_db(
                db_path=db,
                order_id=order_id,
                outcome_status="published_project",
                outcome_summary="Published Lead Offer Brief as private GitHub repo manolosake/lead-offer-brief; 3 tests passed.",
                now=now + 60,
            )
            memory = bot._studio_recent_cycle_outcome_memory(db, now=now + 120)
            packet = bot._studio_cycle_prompt_packet(
                selected={
                    "type": "NEW_PROJECT",
                    "repo_name": "New product incubator",
                    "score": 96,
                    "thesis": "Create or advance a monetizable product.",
                    "operator_visible_outcome": "Published product evidence.",
                },
                opportunities=[],
                memory=memory,
            )

        self.assertEqual(memory["studio_portfolio_total"], 1)
        self.assertEqual(memory["studio_portfolio_recent_count_6h"], 1)
        self.assertIn("Lead Offer Brief", memory["studio_portfolio_recent_projects"][0])
        self.assertIn("manolosake/lead-offer-brief", memory["studio_portfolio_recent_projects"][0])
        self.assertIn("Portfolio assets", packet)
        self.assertIn("Portfolio compounding", packet)

    def test_portfolio_parser_ignores_non_repo_slash_phrases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            now = 226_000.0
            order_id = "order-portfolio-summary"
            bot._studio_ensure_schema(db)
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE jobs(job_id TEXT PRIMARY KEY, trace TEXT NOT NULL DEFAULT '{}')")
                conn.execute("INSERT INTO jobs(job_id, trace) VALUES (?, ?)", (order_id, "{}"))
                conn.execute(
                    """
                    INSERT INTO studio_cycles(
                        cycle_id, version, ts, status, selected_key, selected_type, selected_repo_id,
                        selected_repo_path, selected_lane, thesis, rationale, debate_summary,
                        operator_visible_outcome, evidence_target, risk_summary, prompt_packet,
                        opportunities_json, outcome_status, outcome_summary, order_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "cycle-summary-parser",
                        bot._STUDIO_CYCLE_VERSION,
                        now,
                        "active",
                        "new-project-incubator",
                        "NEW_PROJECT",
                        "",
                        "",
                        "incubator",
                        "Create a sellable project",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "[]",
                        "",
                        "",
                        order_id,
                        now,
                        now,
                    ),
                )
                conn.commit()

            bot._studio_complete_cycle_for_order_db(
                db_path=db,
                order_id=order_id,
                outcome_status="published_project",
                outcome_summary=(
                    "PASS. Avoided repeat/no-delta work. Built `/home/aponce/lead-offer-brief`, "
                    "repo `manolosake/lead-offer-brief` private on GitHub."
                ),
                now=now + 60,
            )
            with sqlite3.connect(db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT github_repo FROM studio_portfolio_projects").fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row["github_repo"], "manolosake/lead-offer-brief")

    def test_portfolio_parser_reads_git_remote_and_head_from_project_path(self) -> None:
        def run(cmd: list[str], cwd: Path) -> str:
            return subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            project = root / "winback-workbench"
            project.mkdir()
            run(["git", "init", "-b", "main"], cwd=project)
            run(["git", "config", "user.email", "test@example.com"], cwd=project)
            run(["git", "config", "user.name", "test"], cwd=project)
            (project / "README.md").write_text("# WinbackWorkbench\n", encoding="utf-8")
            run(["git", "add", "README.md"], cwd=project)
            run(["git", "commit", "-m", "initial"], cwd=project)
            run(["git", "remote", "add", "origin", "https://github.com/manolosake/winback-workbench.git"], cwd=project)
            head = run(["git", "rev-parse", "--short", "HEAD"], cwd=project)
            db = root / "jobs.sqlite"
            now = 227_000.0
            order_id = "order-portfolio-git"
            bot._studio_ensure_schema(db)
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE jobs(job_id TEXT PRIMARY KEY, trace TEXT NOT NULL DEFAULT '{}')")
                conn.execute(
                    "INSERT INTO jobs(job_id, trace) VALUES (?, ?)",
                    (
                        order_id,
                        json.dumps({"project_incubator_delivery": {"project_path": str(project)}}),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO studio_cycles(
                        cycle_id, version, ts, status, selected_key, selected_type, selected_repo_id,
                        selected_repo_path, selected_lane, thesis, rationale, debate_summary,
                        operator_visible_outcome, evidence_target, risk_summary, prompt_packet,
                        opportunities_json, outcome_status, outcome_summary, order_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "cycle-git-parser",
                        bot._STUDIO_CYCLE_VERSION,
                        now,
                        "active",
                        "new-project-incubator",
                        "NEW_PROJECT",
                        "",
                        "",
                        "incubator",
                        "Create a sellable project",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "[]",
                        "",
                        "",
                        order_id,
                        now,
                        now,
                    ),
                )
                conn.commit()

            bot._studio_complete_cycle_for_order_db(
                db_path=db,
                order_id=order_id,
                outcome_status="published_project",
                outcome_summary="PASS. Built WinbackWorkbench and published it privately.",
                now=now + 60,
            )
            with sqlite3.connect(db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT github_repo, latest_head FROM studio_portfolio_projects").fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row["github_repo"], "manolosake/winback-workbench")
        self.assertEqual(row["latest_head"], head)

    def test_portfolio_parser_confirms_private_visibility_from_github_api(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.sqlite"
            now = 228_000.0
            order_id = "order-portfolio-private"
            bot._studio_ensure_schema(db)
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE jobs(job_id TEXT PRIMARY KEY, trace TEXT NOT NULL DEFAULT '{}')")
                conn.execute("INSERT INTO jobs(job_id, trace) VALUES (?, ?)", (order_id, "{}"))
                conn.execute(
                    """
                    INSERT INTO studio_cycles(
                        cycle_id, version, ts, status, selected_key, selected_type, selected_repo_id,
                        selected_repo_path, selected_lane, thesis, rationale, debate_summary,
                        operator_visible_outcome, evidence_target, risk_summary, prompt_packet,
                        opportunities_json, outcome_status, outcome_summary, order_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "cycle-private-parser",
                        bot._STUDIO_CYCLE_VERSION,
                        now,
                        "active",
                        "new-project-incubator",
                        "NEW_PROJECT",
                        "",
                        "",
                        "incubator",
                        "Create a sellable project",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "[]",
                        "",
                        "",
                        order_id,
                        now,
                        now,
                    ),
                )
                conn.commit()

            with patch.object(bot, "_github_token_from_env_or_git_credentials", return_value=("token", "test")), patch.object(
                bot, "_github_api_json", return_value=(True, {"private": True})
            ):
                bot._studio_complete_cycle_for_order_db(
                    db_path=db,
                    order_id=order_id,
                    outcome_status="published_project",
                    outcome_summary="Published repo manolosake/winback-workbench with tests passed.",
                    now=now + 60,
                )
            with sqlite3.connect(db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT status, private FROM studio_portfolio_projects").fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "published_private")
        self.assertEqual(row["private"], 1)

    def test_incubator_opportunity_requires_monetization_evidence(self) -> None:
        item = bot._studio_incubator_opportunity(
            cfg=SimpleNamespace(),  # type: ignore[arg-type]
            now=250_000.0,
            memory={},
        )

        self.assertEqual(item["type"], "NEW_PROJECT")
        self.assertGreaterEqual(item["score"], 96)
        self.assertIn("sellable", item["thesis"])
        self.assertIn("rentable", item["thesis"])
        self.assertIn("pricing", item["operator_visible_outcome"].lower())
        self.assertIn("monetization", item["evidence_target"].lower())
        self.assertIn("Direct revenue", item["business_model"])
        self.assertIn("target customer", item["monetization_path"])
        self.assertIn("target buyer", item["commercial_evidence_target"])
        self.assertIn("factory_value", item)
        self.assertEqual(item["factory_value"]["missing_dimensions"], [])
        self.assertGreaterEqual(item["factory_value"]["score"], 100)
        self.assertGreaterEqual(item["factory_value"]["shipability_score"], 100)

    def test_factory_value_penalizes_missing_validation_and_ship_path(self) -> None:
        item = {
            "key": "weak-bet",
            "type": "FEATURE",
            "repo_name": "Weak Bet",
            "score": 80,
            "thesis": "Improve a vague internal surface.",
            "operator_visible_outcome": "A nicer implementation.",
            "business_model": "Operator savings through less waste.",
            "monetization_path": "Save money by reducing manual review time.",
            "commercial_evidence_target": "operator savings",
            "evidence_target": "brief notes only",
            "risk_summary": "small bounded change",
        }

        assessed = bot._studio_apply_factory_value_assessment(item)

        self.assertLess(assessed["score"], 80)
        self.assertIn("validation/toolchain readiness", assessed["factory_value"]["missing_dimensions"])
        self.assertIn("ship/recovery path", assessed["factory_value"]["missing_dimensions"])
        self.assertLess(assessed["factory_value"]["shipability_score"], 100)

    def test_repo_opportunity_factory_value_tracks_toolchain_gap(self) -> None:
        now = 250_000.0
        memory = {
            "studio_readiness": {
                "repo_checks": [
                    {
                        "repo_id": "sampleapp",
                        "repo_path": "/home/aponce/sampleapp",
                        "status": "red",
                        "stacks": ["python"],
                        "missing_tools": ["pytest"],
                    }
                ]
            }
        }
        repo = {
            "repo_id": "sampleapp",
            "path": "/home/aponce/sampleapp",
            "status": "active",
            "autonomy_enabled": True,
            "priority": 1,
            "metadata": {},
        }

        item = bot._studio_opportunity_for_repo(repo, now=now, memory=memory)

        self.assertIn("factory_value", item)
        self.assertIn("validation/toolchain readiness", item["factory_value"]["missing_dimensions"])
        self.assertIn("missing tools: pytest", item["factory_value"]["dimensions"]["validation_toolchain_readiness"]["evidence"])

    def test_incubator_opportunity_compounds_when_portfolio_is_fresh(self) -> None:
        item = bot._studio_incubator_opportunity(
            cfg=SimpleNamespace(),  # type: ignore[arg-type]
            now=250_000.0,
            memory={
                "studio_portfolio_total": 5,
                "studio_portfolio_recent_count_6h": 3,
                "studio_portfolio_recent_projects": [
                    "WinbackWorkbench · manolosake/winback-workbench · head 973e958",
                    "Lead Offer Brief · manolosake/lead-offer-brief · head 2efec0a",
                ],
            },
        )

        self.assertLess(item["score"], 96)
        self.assertIn("Advance the strongest published incubator product", item["thesis"])
        self.assertIn("compounding one asset", item["why_better_than_alternatives"])
        self.assertIn("avoid duplicating recent products", item["risk_summary"])

    def test_prompt_packet_includes_business_objective_and_self_improvement_guard(self) -> None:
        selected = {
            "type": "NEW_PROJECT",
            "repo_name": "New product incubator",
            "score": 99,
            "thesis": "Create a rentable automation product.",
            "operator_visible_outcome": "A private MVP with buyer and pricing hypothesis.",
            "business_model": "Direct revenue option.",
            "monetization_path": "Sellable tool for a named buyer.",
            "commercial_evidence_target": "buyer, price, demo, validation.",
        }

        packet = bot._studio_cycle_prompt_packet(selected=selected, opportunities=[selected], memory={})

        self.assertIn("Factory objective", packet)
        self.assertIn("make money", packet)
        self.assertIn("sellable products", packet)
        self.assertIn("rentable tools", packet)
        self.assertIn("never as work-for-work's-sake", packet)
        self.assertIn("Selected monetization path", packet)
        self.assertIn("Can this make money", packet)

    def test_prompt_packet_includes_selected_factory_value(self) -> None:
        selected = {
            "type": "NEW_PROJECT",
            "repo_name": "New product incubator",
            "score": 100,
            "thesis": "Create a rentable automation product.",
            "operator_visible_outcome": "A private MVP with validation logs and GitHub publication.",
            "business_model": "Direct revenue option.",
            "monetization_path": "Sellable tool for a named buyer.",
            "commercial_evidence_target": "buyer, price, demo, validation.",
            "evidence_target": "validation command/log and private GitHub remote.",
        }
        selected = bot._studio_apply_factory_value_assessment(selected)

        packet = bot._studio_cycle_prompt_packet(selected=selected, opportunities=[selected], memory={})

        self.assertIn("Selected factory value:", packet)
        self.assertIn("shipability=100", packet)
        self.assertIn("missing=none", packet)

    def test_kill_sheet_names_missing_factory_value_dimensions(self) -> None:
        selected = {
            "key": "repo-codexbot",
            "type": "DEEP_IMPROVEMENT",
            "repo_name": "codexbot",
            "score": 100,
            "thesis": "Repair Studio selection.",
            "operator_visible_outcome": "A clearer selection loop.",
        }
        rejected = {
            "key": "weak-bet",
            "type": "FEATURE",
            "repo_name": "Weak Bet",
            "score": 80,
            "thesis": "Improve a vague internal surface.",
            "operator_visible_outcome": "A nicer implementation.",
            "business_model": "Operator savings.",
            "monetization_path": "Save review time.",
            "commercial_evidence_target": "operator savings",
            "evidence_target": "brief notes only",
            "risk_summary": "small bounded change",
            "recent_studio_outcome_risks": [],
            "recent_studio_saturation": [],
        }
        rejected = bot._studio_apply_factory_value_assessment(rejected)

        lines = bot._studio_selection_kill_sheet(selected, [selected, rejected], {})

        self.assertTrue(lines)
        self.assertIn("missing validation/toolchain readiness", lines[0])
        self.assertIn("missing ship/recovery path", lines[0])
        self.assertIn("weaker than selected score 100", lines[0])

    def test_prompt_packet_includes_r530_resource_policy(self) -> None:
        selected = {
            "type": "DEEP_IMPROVEMENT",
            "repo_name": "codexbot",
            "score": 94,
            "thesis": "Use deeper local validation to improve shipping quality.",
            "operator_visible_outcome": "A safer autonomous delivery loop.",
            "business_model": "Factory leverage.",
            "monetization_path": "Better shipping for revenue projects.",
            "commercial_evidence_target": "fewer failed shipments.",
        }

        packet = bot._studio_cycle_prompt_packet(selected=selected, opportunities=[selected], memory={})

        self.assertIn("r530 resources", packet)
        self.assertIn("local CPU, memory, disk", packet)
        self.assertIn("local models", packet)
        self.assertIn("keeping core services responsive", packet)
        self.assertIn("Would using more local r530 compute", packet)

    def test_recent_same_surface_shipment_penalizes_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jobs.sqlite"
            now = 400_000.0
            self._insert_cycle_outcome(
                db,
                now=now + 590,
                key="repo-executivedashboard",
                repo_id="executivedashboard",
                selected_type="FEATURE",
                outcome_status="shipped_to_main",
                outcome_summary="Merged to main and deployed the dashboard feature.",
            )
            memory = bot._studio_selection_memory(
                cfg=SimpleNamespace(orchestrator_db_path=db),  # type: ignore[arg-type]
                orch_q=self._FakeQueue(),  # type: ignore[arg-type]
                now=now,
            )
            repo = {
                "repo_id": "executivedashboard",
                "path": "/home/aponce/ExecutiveDashboard",
                "status": "active",
                "autonomy_enabled": True,
                "priority": 1,
                "metadata": {},
            }

            item = bot._studio_opportunity_for_repo(repo, now=now, memory=memory)

        self.assertLess(item["score"], 86)
        self.assertTrue(item["recent_studio_saturation"])
        self.assertIn("fresh angle", item["why_better_than_alternatives"])

    def test_artifact_only_delivery_claim_detects_no_delta_backend(self) -> None:
        task = SimpleNamespace(
            job_id="job-ghost",
            role="backend",
            state="done",
            updated_at=123.0,
            created_at=100.0,
            trace={
                "result_summary": "Implemented and validated a dashboard improvement with artifacts.",
                "result_artifacts": ["/tmp/job/unified_diff.patch"],
                "structured_digest": {
                    "branch_sync": {"status": "skipped", "reason": "no_changes"},
                },
            },
        )

        self.assertTrue(bot._task_is_artifact_only_delivery_claim(task, trace=task.trace))
        self.assertIs(bot._latest_artifact_only_delivery_claim([task]), task)

    def test_artifact_only_delivery_claim_ignores_validated_patch(self) -> None:
        task = SimpleNamespace(
            job_id="job-real",
            role="implementer_local",
            state="done",
            updated_at=123.0,
            created_at=100.0,
            trace={
                "result_summary": "Implemented and validated a real patch.",
                "local_patch_info": {
                    "changed_files": ["server/app.py"],
                    "validation_ok": True,
                },
                "slice_patch_applied": True,
                "slice_validation_ok": True,
            },
        )

        self.assertFalse(bot._task_is_artifact_only_delivery_claim(task, trace=task.trace))

    def test_stale_selected_cycle_without_order_is_closed_with_root_cause(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jobs.sqlite"
            now = 300_000.0
            bot._studio_ensure_schema(db)
            with sqlite3.connect(db) as conn:
                for cycle_id, updated_at in (("old-selected", now - 7200), ("fresh-selected", now - 120)):
                    conn.execute(
                        """
                        INSERT INTO studio_cycles(
                            cycle_id, version, ts, status, selected_key, selected_type, selected_repo_id,
                            selected_repo_path, selected_lane, thesis, rationale, debate_summary,
                            operator_visible_outcome, evidence_target, risk_summary, prompt_packet,
                            opportunities_json, outcome_status, outcome_summary, order_id, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cycle_id,
                            bot._STUDIO_CYCLE_VERSION,
                            updated_at,
                            "selected",
                            f"repo-{cycle_id}",
                            "FEATURE",
                            "executivedashboard",
                            "/tmp/repo",
                            "studio",
                            "Improve visible outcomes",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "[]",
                            "",
                            "",
                            None,
                            updated_at,
                            updated_at,
                        ),
                    )
                conn.commit()

            closed = bot._studio_finalize_stale_selected_cycles(db, now=now, max_age_seconds=1800)

            with sqlite3.connect(db) as conn:
                conn.row_factory = sqlite3.Row
                old = dict(conn.execute("SELECT status, outcome_status, outcome_summary FROM studio_cycles WHERE cycle_id = ?", ("old-selected",)).fetchone())
                fresh = dict(conn.execute("SELECT status, outcome_status FROM studio_cycles WHERE cycle_id = ?", ("fresh-selected",)).fetchone())

        self.assertEqual(closed, 1)
        self.assertEqual(old["status"], "failed")
        self.assertEqual(old["outcome_status"], "failed_root_caused")
        self.assertIn("no order was attached", old["outcome_summary"])
        self.assertEqual(fresh["status"], "selected")
        self.assertEqual(fresh["outcome_status"], "")


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


class TestOrchestratorMinEvidenceGate(unittest.TestCase):
    def _studio_task(self, *, role: str = "implementer_local") -> bot.Task:
        return bot.Task.new(
            job_id="studio-task",
            source="test",
            role=role,
            input_text="Ship the selected Studio deep improvement slice.",
            request_type="task",
            priority=1,
            model="gpt-5.3-codex",
            effort="medium",
            mode_hint="rw",
            requires_approval=False,
            max_cost_window_usd=0.0,
            chat_id=1,
            state="running",
            trace={
                "studio_cycle_id": "cycle-123",
                "studio_selected_type": "DEEP_IMPROVEMENT",
            },
        )

    def _valid_studio_evidence(self) -> dict[str, object]:
        return {
            "candidate_bets": [
                {"id": "runtime_gate", "summary": "Runtime evidence gate"},
                {"id": "backend_flag", "summary": "Backend guard default flag"},
                {"id": "prompt_only", "summary": "Prompt-only instruction"},
            ],
            "killed_bets": [
                {
                    "id": "backend_flag",
                    "reason": "Backend-only validation would miss runtime completion payloads before acceptance.",
                },
                {
                    "id": "prompt_only",
                    "reason": "Prompt-only text can be ignored by the runtime and leaves no auditable blocker.",
                }
            ],
            "selected_bet": {
                "id": "runtime_gate",
                "summary": "Runtime Studio DEEP_IMPROVEMENT decision-evidence gate",
                "reason": "It blocks completion exactly where missing decision evidence becomes harmful.",
            },
            "critic_answers": {
                "blast_radius": "Scoped to successful non-controller Studio DEEP_IMPROVEMENT tasks only.",
                "diagnosis": "Gate metadata records counts and missing evidence blocks for review.",
                "tests": "Focused unit tests cover missing, partial, and valid evidence paths.",
            },
            "debate_summary": "Runtime gating was selected over broader rewrites because it preserves auditable decision evidence at completion.",
        }

    def _valid_factory_delta(self) -> dict[str, object]:
        return {
            "capability_changed": "Studio DEEP_IMPROVEMENT completion now requires a named factory capability delta.",
            "measurable_delta": "Missing capability delta evidence changes completion from accepted to blocked.",
            "evidence": "Focused unit tests validate missing, partial, and valid factory_delta payloads.",
        }

    def test_extract_structured_result_payload_preserves_studio_decision_evidence(self) -> None:
        payload = bot._extract_structured_result_payload(
            "done\n```json\n"
            "{"
            "\"summary\":\"Ready with decision evidence.\","
            "\"studio_decision_evidence\":{"
            "\"candidate_bets\":[{\"id\":\"a\"},{\"id\":\"b\"},{\"id\":\"c\"}],"
            "\"killed_bets\":[{\"id\":\"b\",\"reason\":\"Too broad for this sprint slice.\"}],"
            "\"selected_bet\":{\"id\":\"a\",\"summary\":\"Runtime gate\",\"reason\":\"It blocks missing evidence at completion time.\"},"
            "\"critic_answers\":{\"risk\":\"Runtime-only and scoped.\",\"test\":\"Unit tests pin behavior.\",\"ops\":\"Metadata aids diagnosis.\"},"
            "\"debate_summary\":\"The runtime gate was selected because it is auditable and narrowly scoped.\""
            "}"
            "}\n```"
        )

        self.assertIsInstance(payload, dict)
        self.assertIn("studio_decision_evidence", payload)

    def test_extract_structured_result_payload_preserves_factory_delta(self) -> None:
        payload = bot._extract_structured_result_payload(
            "done\n```json\n"
            "{"
            "\"summary\":\"Ready with factory delta.\","
            "\"factory_delta\":{"
            "\"capability_changed\":\"Factory completion now names changed capability.\","
            "\"measurable_delta\":\"Missing deltas are blocked instead of accepted.\","
            "\"evidence\":\"Focused tests cover the validation contract.\""
            "}"
            "}\n```"
        )

        self.assertIsInstance(payload, dict)
        self.assertIn("factory_delta", payload)

    def test_blocks_missing_studio_deep_improvement_evidence(self) -> None:
        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(),
            summary="Implemented the chosen Studio deep improvement slice with tests and validation output.",
            artifacts=[],
            logs="validation output " * 12,
            structured={},
        )

        self.assertFalse(ok)
        self.assertIn("Studio DEEP_IMPROVEMENT decision evidence", str(reason))
        self.assertTrue(meta["studio_decision_evidence_required"])
        self.assertFalse(meta["studio_decision_evidence_present"])
        issues = meta["studio_decision_evidence"]["issues"]
        self.assertIn("candidate_bets must list at least 3 options", issues)

    def test_architect_local_blocks_missing_studio_deep_improvement_evidence(self) -> None:
        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(role="architect_local"),
            summary="Completed the Studio selection review and chose the deep improvement continuation path.",
            artifacts=[],
            logs="architect selection-review validation output " * 4,
            structured={"factory_delta": self._valid_factory_delta()},
        )

        self.assertFalse(ok)
        self.assertIn("Studio DEEP_IMPROVEMENT decision evidence", str(reason))
        self.assertTrue(meta["studio_decision_evidence_required"])
        self.assertFalse(meta["studio_decision_evidence_present"])
        self.assertEqual(meta["studio_selected_type"], "DEEP_IMPROVEMENT")

    def test_inherited_studio_context_blocks_delegated_child_without_decision_evidence(self) -> None:
        parent_trace = {
            "studio_cycle_id": "cycle-456",
            "studio_selected_type": "DEEP_IMPROVEMENT",
            "studio_thesis": "Runtime enforcement prevents prompt-only completion claims.",
            "studio_operator_visible_outcome": "Operators get auditable Studio improvement decisions.",
            "studio_evidence_target": "Completion is blocked without structured decision evidence.",
            "studio_risk_summary": "Scoped to delegated non-controller Studio work.",
        }
        child_trace = {
            "source": "telegram",
            "delegated_by": "root-task",
            "delegated_key": "implement_gate",
            "requested_role": "implementer_local",
            "profile_role": "implementer_local",
            "slice_id": "root_slice",
        }
        bot._inherit_studio_trace_context(parent_trace, child_trace)

        self.assertEqual(child_trace["studio_cycle_id"], "cycle-456")
        self.assertEqual(child_trace["studio_selected_type"], "DEEP_IMPROVEMENT")
        self.assertEqual(
            child_trace["studio_operator_visible_outcome"],
            "Operators get auditable Studio improvement decisions.",
        )

        child = bot.Task.new(
            job_id="delegated-child",
            source="test",
            role="implementer_local",
            input_text="Implement the delegated Studio deep improvement slice.",
            request_type="task",
            priority=1,
            model="gpt-5.3-codex",
            effort="medium",
            mode_hint="rw",
            requires_approval=False,
            max_cost_window_usd=0.0,
            chat_id=1,
            state="running",
            parent_job_id="root-task",
            labels={"ticket": "root-task", "kind": "subtask", "key": "implement_gate"},
            trace=child_trace,
        )

        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=child,
            summary="Implemented the delegated Studio deep improvement slice with targeted validation evidence.",
            artifacts=[],
            logs="validation output " * 12,
            structured={},
        )

        self.assertFalse(ok)
        self.assertIn("Studio DEEP_IMPROVEMENT decision evidence", str(reason))
        self.assertEqual(meta["studio_cycle_id"], "cycle-456")
        self.assertEqual(meta["studio_selected_type"], "DEEP_IMPROVEMENT")

    def test_reviewer_local_inherited_studio_context_passes_without_decision_evidence(self) -> None:
        task = self._studio_task(role="reviewer_local")

        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=task,
            summary=(
                "Reviewed the implementation against the requested behavior and validation "
                "contract; no blocking issues found."
            ),
            artifacts=[],
            logs="review note with concrete validation detail " * 4,
            structured={"factory_delta": self._valid_factory_delta()},
        )

        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertNotIn("studio_decision_evidence_required", meta)
        self.assertTrue(meta["factory_delta_required"])
        self.assertTrue(meta["summary_substantial"])

    def test_qa_inherited_studio_context_passes_without_decision_evidence(self) -> None:
        task = self._studio_task(role="qa")

        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=task,
            summary=(
                "PASS: validated the requested Studio improvement with focused tests and "
                "checked the relevant regression behavior."
            ),
            artifacts=[],
            logs="",
            structured={"factory_delta": self._valid_factory_delta()},
        )

        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertNotIn("studio_decision_evidence_required", meta)
        self.assertTrue(meta["factory_delta_required"])
        self.assertTrue(meta["summary_substantial"])

    def test_blocks_missing_studio_deep_improvement_factory_delta_for_qa(self) -> None:
        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(role="qa"),
            summary="PASS: validated the requested Studio improvement with focused tests.",
            artifacts=[],
            logs="",
            structured={},
        )

        self.assertFalse(ok)
        self.assertIn("Studio DEEP_IMPROVEMENT factory delta", str(reason))
        self.assertTrue(meta["factory_delta_required"])
        self.assertFalse(meta["factory_delta_present"])
        self.assertIn("factory_delta.capability_changed", "; ".join(meta["factory_delta"]["issues"]))

    def test_blocks_partial_studio_deep_improvement_factory_delta_for_reviewer(self) -> None:
        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(role="reviewer_local"),
            summary="Reviewed the implementation against the requested behavior and validation contract.",
            artifacts=[],
            logs="review note with concrete validation detail " * 4,
            structured={"factory_delta": {"capability_changed": "Factory capability names changed."}},
        )

        self.assertFalse(ok)
        self.assertIn("Studio DEEP_IMPROVEMENT factory delta", str(reason))
        diag = meta["factory_delta"]
        self.assertTrue(diag["capability_changed_present"])
        self.assertFalse(diag["measurable_delta_present"])
        self.assertFalse(diag["evidence_present"])

    def test_blocks_partial_studio_deep_improvement_evidence(self) -> None:
        evidence = self._valid_studio_evidence()
        evidence["candidate_bets"] = [{"id": "only_one"}]
        evidence["killed_bets"] = [{"id": "weak", "reason": "thin"}]
        evidence["critic_answers"] = {"one": "This answer is substantive enough."}

        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(),
            summary="Implemented the chosen Studio deep improvement slice with tests and validation output.",
            artifacts=[],
            logs="validation output " * 12,
            structured={"studio_decision_evidence": evidence},
        )

        self.assertFalse(ok)
        self.assertIn("Studio DEEP_IMPROVEMENT decision evidence", str(reason))
        diag = meta["studio_decision_evidence"]
        self.assertEqual(diag["candidate_bets_count"], 1)
        self.assertEqual(diag["killed_bets_with_reason_count"], 0)
        self.assertEqual(diag["critic_answers_count"], 1)

    def test_blocks_unresolved_non_selected_studio_candidate_bet(self) -> None:
        evidence = self._valid_studio_evidence()
        evidence["killed_bets"] = [
            {
                "id": "prompt_only",
                "reason": "Prompt-only text can be ignored by the runtime and leaves no auditable blocker.",
            }
        ]

        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(),
            summary="Implemented the chosen Studio deep improvement slice with tests and validation output.",
            artifacts=[],
            logs="validation output " * 12,
            structured={"studio_decision_evidence": evidence},
        )

        self.assertFalse(ok)
        self.assertIn("non-selected candidate", str(reason))
        diag = meta["studio_decision_evidence"]
        self.assertEqual(diag["unresolved_candidate_bet_ids"], ["backend_flag"])
        self.assertIn(
            "every non-selected candidate_bets item must be killed or selected",
            diag["issues"],
        )

    def test_blocks_unknown_killed_studio_bet(self) -> None:
        evidence = self._valid_studio_evidence()
        evidence["killed_bets"] = [
            {
                "id": "backend_flag",
                "reason": "Backend-only validation would miss runtime completion payloads before acceptance.",
            },
            {
                "id": "prompt_only",
                "reason": "Prompt-only text can be ignored by the runtime and leaves no auditable blocker.",
            },
            {
                "id": "ghost_bet",
                "reason": "This killed bet is not one of the candidates and should be rejected.",
            },
        ]

        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(),
            summary="Implemented the chosen Studio deep improvement slice with tests and validation output.",
            artifacts=[],
            logs="validation output " * 12,
            structured={"studio_decision_evidence": evidence},
        )

        self.assertFalse(ok)
        self.assertIn("killed_bets ids must exist in candidate_bets", str(reason))
        diag = meta["studio_decision_evidence"]
        self.assertEqual(diag["unknown_killed_bet_ids"], ["ghost_bet"])

    def test_blocks_unknown_selected_studio_bet(self) -> None:
        evidence = self._valid_studio_evidence()
        evidence["selected_bet"] = {
            "id": "ghost_selected",
            "summary": "Runtime Studio DEEP_IMPROVEMENT decision-evidence gate",
            "reason": "This selected bet is not one of the candidates and should be rejected.",
        }

        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(),
            summary="Implemented the chosen Studio deep improvement slice with tests and validation output.",
            artifacts=[],
            logs="validation output " * 12,
            structured={"studio_decision_evidence": evidence},
        )

        self.assertFalse(ok)
        self.assertIn("selected_bet id must exist in candidate_bets", str(reason))
        diag = meta["studio_decision_evidence"]
        self.assertEqual(diag["selected_bet_id"], "ghost_selected")
        self.assertFalse(diag["selected_bet_in_candidates"])

    def test_blocks_selected_studio_bet_also_killed(self) -> None:
        evidence = self._valid_studio_evidence()
        evidence["killed_bets"] = [
            {
                "id": "runtime_gate",
                "reason": "The selected bet cannot also be killed by the same decision record.",
            },
            {
                "id": "backend_flag",
                "reason": "Backend-only validation would miss runtime completion payloads before acceptance.",
            },
            {
                "id": "prompt_only",
                "reason": "Prompt-only text can be ignored by the runtime and leaves no auditable blocker.",
            },
        ]

        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(),
            summary="Implemented the chosen Studio deep improvement slice with tests and validation output.",
            artifacts=[],
            logs="validation output " * 12,
            structured={"studio_decision_evidence": evidence},
        )

        self.assertFalse(ok)
        self.assertIn("selected_bet must not also be listed in killed_bets", str(reason))
        diag = meta["studio_decision_evidence"]
        self.assertEqual(diag["selected_bet_id"], "runtime_gate")
        self.assertTrue(diag["selected_bet_killed"])

    def test_valid_studio_deep_improvement_evidence_passes(self) -> None:
        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(),
            summary="Implemented the chosen Studio deep improvement slice with tests and validation output.",
            artifacts=[],
            logs="validation output " * 12,
            structured={
                "studio_decision_evidence": self._valid_studio_evidence(),
                "factory_delta": self._valid_factory_delta(),
            },
        )

        self.assertTrue(ok)
        self.assertIsNone(reason)
        diag = meta["studio_decision_evidence"]
        self.assertEqual(diag["candidate_bets_count"], 3)
        self.assertEqual(diag["killed_bets_with_reason_count"], 2)
        self.assertEqual(diag["unresolved_candidate_bet_ids"], [])
        self.assertTrue(diag["selected_bet_in_candidates"])
        self.assertFalse(diag["selected_bet_killed"])
        self.assertEqual(diag["critic_answers_count"], 3)
        self.assertEqual(diag["issues"], [])
        self.assertEqual(meta["factory_delta"]["issues"], [])

    def test_valid_architect_local_studio_deep_improvement_evidence_passes(self) -> None:
        ok, reason, meta = bot._orchestrator_min_evidence_gate(
            task=self._studio_task(role="architect_local"),
            summary="Completed the Studio selection review and chose the deep improvement continuation path.",
            artifacts=[],
            logs="architect selection-review validation output " * 4,
            structured={
                "studio_decision_evidence": self._valid_studio_evidence(),
                "factory_delta": self._valid_factory_delta(),
            },
        )

        self.assertTrue(ok)
        self.assertIsNone(reason)
        diag = meta["studio_decision_evidence"]
        self.assertEqual(diag["candidate_bets_count"], 3)
        self.assertEqual(diag["killed_bets_with_reason_count"], 2)
        self.assertEqual(diag["issues"], [])


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

    def test_local_ollama_error_preserves_attempt_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td) / "artifacts"
            cfg = bot.dataclasses.replace(
                TestStateHandling()._cfg(Path(td) / "state.json"),
                artifacts_root=Path(td) / "artifacts_root",
                codex_timeout_seconds=1,
            )
            task = bot.Task.new(
                job_id="job-local-empty",
                source="test",
                role="architect_local",
                input_text="Return a bounded plan.",
                request_type="task",
                priority=1,
                model="primary:latest",
                effort="medium",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=0.0,
                chat_id=1,
                state="running",
                artifacts_dir=str(artifacts_dir),
                trace={},
            )
            seen_models: list[str] = []

            def _empty_ollama_chat(**kwargs: object) -> dict[str, object]:
                seen_models.append(str(kwargs.get("model") or ""))
                return {"content": "", "events": 0, "response": {"done": True}}

            profiles = {
                "architect_local": {
                    "model": "primary:latest",
                    "local_fallback_model": "fallback:latest",
                    "system_prompt": "You are a focused local planner.",
                }
            }
            with patch.object(bot, "_ollama_installed_model_names", return_value=[]), patch.object(
                bot, "_ollama_chat", side_effect=_empty_ollama_chat
            ):
                result = bot._orchestrator_run_local_ollama(
                    cfg,
                    task,
                    stop_event=threading.Event(),
                    orch_q=None,
                    profiles=profiles,
                )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["next_action"], "retry")
            structured = result["structured_digest"]
            self.assertEqual(structured["requested_model"], "primary:latest")
            self.assertEqual(structured["model"], "qwen3.5:latest")
            self.assertEqual(structured["fallback_candidate_count"], 3)
            self.assertIn("primary:latest: empty response", structured["attempt_errors"])
            self.assertIn("fallback:latest: empty response", structured["attempt_errors"])
            self.assertIn("qwen3.5:latest: empty response", structured["attempt_errors"])
            attempts_path = Path(str(structured["attempts_artifact"]))
            self.assertTrue(attempts_path.is_file())
            self.assertIn(str(attempts_path), result["artifacts"])
            attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
            self.assertEqual(attempts["candidate_models"], seen_models)
            self.assertEqual(attempts["attempt_errors"], structured["attempt_errors"])

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
                with patch.dict(os.environ, {"BOT_AUTONOMOUS_LOCAL_FIRST": "1"}, clear=False):
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
                with patch.dict(os.environ, {"BOT_AUTONOMOUS_LOCAL_FIRST": "1"}, clear=False):
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
                with patch.dict(os.environ, {"BOT_AUTONOMOUS_LOCAL_FIRST": "1"}, clear=False):
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

    def test_finalize_codex_implementer_change_strips_stale_file_mode_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)

            target = repo / "tools" / "proactive_blocker_replay.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("#!/usr/bin/env python3\n\nVALUE = \"old\"\n", encoding="utf-8")
            target.chmod(0o644)
            subprocess.run(["git", "add", "tools/proactive_blocker_replay.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            diff_text = (
                "diff --git a/tools/proactive_blocker_replay.py b/tools/proactive_blocker_replay.py\n"
                "old mode 100755\n"
                "new mode 100644\n"
                "index 1111111..2222222 100755\n"
                "--- a/tools/proactive_blocker_replay.py\n"
                "+++ b/tools/proactive_blocker_replay.py\n"
                "@@ -1,3 +1,3 @@\n"
                " #!/usr/bin/env python3\n"
                " \n"
                "-VALUE = \"old\"\n"
                "+VALUE = \"new\"\n"
            )
            artifacts_dir = Path(td) / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            body = f"```diff\n{diff_text}```\n"

            artifacts, patch_info, patch_error = bot._finalize_codex_implementer_change(
                task=SimpleNamespace(),
                artifacts_dir=artifacts_dir,
                content=body,
                worktree_dir=repo,
            )

            self.assertIsNone(patch_error)
            self.assertIn("tools/proactive_blocker_replay.py", patch_info.get("changed_files", []))
            self.assertTrue(bool(patch_info.get("validation_ok", False)))
            self.assertTrue(bool(patch_info.get("patch_repaired", False)))
            patch_artifact = Path(str(patch_info.get("patch_artifact") or ""))
            applied_patch_text = patch_artifact.read_text(encoding="utf-8")
            self.assertNotIn("old mode 100755", applied_patch_text)
            self.assertNotIn("new mode 100644", applied_patch_text)
            self.assertIn("index 1111111..2222222\n", applied_patch_text)
            self.assertIn("VALUE = \"new\"", target.read_text(encoding="utf-8"))

    def test_finalize_codex_implementer_change_preserves_valid_content_and_mode_diff(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)

            target = repo / "script.sh"
            target.write_text("#!/bin/sh\nprintf 'old\\n'\n", encoding="utf-8")
            target.chmod(0o644)
            subprocess.run(["git", "add", "script.sh"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            target.write_text("#!/bin/sh\nprintf 'new\\n'\n", encoding="utf-8")
            target.chmod(0o755)
            diff_text = subprocess.run(
                ["git", "diff", "--no-ext-diff", "--unified=3"],
                cwd=str(repo),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            target.write_text("#!/bin/sh\nprintf 'old\\n'\n", encoding="utf-8")
            target.chmod(0o644)

            artifacts_dir = Path(td) / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            body = f"```diff\n{diff_text}```\n"

            artifacts, patch_info, patch_error = bot._finalize_codex_implementer_change(
                task=SimpleNamespace(),
                artifacts_dir=artifacts_dir,
                content=body,
                worktree_dir=repo,
            )

            self.assertIsNone(patch_error)
            self.assertIn("script.sh", patch_info.get("changed_files", []))
            self.assertTrue(bool(patch_info.get("validation_ok", False)))
            self.assertFalse(bool(patch_info.get("patch_repaired", False)))
            self.assertFalse(bool(patch_info.get("mode_metadata_stripped", False)))
            self.assertTrue(os.access(target, os.X_OK))
            self.assertIn("printf 'new\\n'", target.read_text(encoding="utf-8"))
            patch_artifact = Path(str(patch_info.get("patch_artifact") or ""))
            applied_patch_text = patch_artifact.read_text(encoding="utf-8")
            self.assertIn("old mode 100644", applied_patch_text)
            self.assertIn("new mode 100755", applied_patch_text)
            self.assertTrue(any(path.endswith("local_ollama_git_diff.patch") for path in artifacts))

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
            worktree_target = bot._repo_worktree_root(cfg, repo_id=repo_id) / "implementer_local" / "slot1" / "bot.py"
            self.assertTrue(worktree_target.is_file())
            self.assertEqual(worktree_target.read_text(encoding="utf-8"), updated)

    def test_orchestrator_run_codex_refuses_base_repo_fallback_when_worktree_sync_fails(self) -> None:
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
                def __init__(self) -> None:
                    self.updated: list[tuple[str, dict[str, object]]] = []
                    self.released = False

                def lease_workspace(self, *, role: str, job_id: str, slots: int = 1) -> int:
                    return 1

                def release_workspace(self, *, job_id: str) -> None:
                    self.released = True

                def update_trace(self, job_id: str, **kwargs: object) -> None:
                    self.updated.append((job_id, dict(kwargs)))

                def get_repo(self, repo_id: str) -> dict[str, object] | None:
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
                job_id="job-sync-fail",
                source="test",
                role="implementer_local",
                input_text="Implement one bounded improvement.",
                request_type="task",
                priority=1,
                model="gpt-5.3-codex",
                effort="medium",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=0.0,
                chat_id=1,
                state="running",
                artifacts_dir=str((cfg.artifacts_root / "job-sync-fail").resolve()),
                trace={
                    "repo_id": repo_id,
                    "repo_path": str(repo),
                    "repo_default_branch": "main",
                    "order_branch": "feature/missing-order-branch",
                },
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

            with patch.object(bot, "_sync_worktree_to_order_branch", return_value=(False, "fetch_failed:missing")):
                with patch.object(bot.CodexRunner, "start", side_effect=AssertionError("must not run in base repo")):
                    result = bot._orchestrator_run_codex(
                        cfg,
                        task,
                        stop_event=threading.Event(),
                        orch_q=fake_queue,
                        profiles=profiles,
                    )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["next_action"], "fix_workspace_sync")
            self.assertIn("refused base repo fallback", result["summary"])
            self.assertTrue(fake_queue.released)
            self.assertTrue(any(update[1].get("live_phase") == "workspace_setup_failed" for update in fake_queue.updated))
            self.assertTrue((Path(task.artifacts_dir) / "worktree_error.txt").is_file())
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(repo),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertEqual(status.strip(), "")

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

            self.assertEqual(result["status"], "ok")
            self.assertNotIn("write_policy_violation", result["structured_digest"])
            worktree = bot._repo_worktree_root(cfg, repo_id=repo_id) / "skynet" / "slot1"
            self.assertEqual(bot._git_status_porcelain(worktree), "")
            self.assertIn("Applied a direct improvement", result["summary"])

    def test_write_policy_stash_captures_untracked_recovery_patch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            artifacts_dir = Path(td) / "artifacts"
            repo.mkdir(parents=True, exist_ok=True)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "README.md").write_text("# repo\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)

            new_file = repo / "tools" / "order_release_readiness_report.py"
            new_file.parent.mkdir(parents=True, exist_ok=True)
            new_file.write_text("def main():\n    return 'ready'\n", encoding="utf-8")

            cleanup = bot._stash_read_only_write_violation(
                repo,
                role="skynet",
                job_id="8125f62d-bb33-4f29-818f-26447593afff",
                label="controller_snapshot",
                artifacts_dir=artifacts_dir,
            )

            patch_path = Path(str(cleanup.get("recovery_patch_artifact") or ""))
            self.assertTrue(patch_path.exists())
            patch_text = patch_path.read_text(encoding="utf-8")
            self.assertIn("diff --git a/tools/order_release_readiness_report.py b/tools/order_release_readiness_report.py", patch_text)
            self.assertIn("new file mode", patch_text)
            self.assertIn("return 'ready'", patch_text)
            self.assertEqual(bot._git_status_porcelain(repo), "")
            self.assertEqual(
                subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=str(repo), text=True).returncode,
                0,
            )

    def test_orchestrator_run_codex_blocks_controller_commits_with_clean_tree(self) -> None:
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
                job_id="job-skynet-clean-commit",
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
                artifacts_dir=str((cfg.artifacts_root / "job-skynet-clean-commit").resolve()),
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
                subprocess.run(["git", "add", "bot.py"], cwd=str(case_worktree), check=True, capture_output=True, text=True)
                subprocess.run(
                    ["git", "commit", "-m", "controller direct commit"],
                    cwd=str(case_worktree),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                stdout_path = Path(td) / "codex_stdout.jsonl"
                stderr_path = Path(td) / "codex_stderr.log"
                last_msg_path = Path(td) / "codex_last.txt"
                stdout_path.write_text("", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                last_msg_path.write_text("Committed a direct improvement with a clean tree\n", encoding="utf-8")
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

            self.assertEqual(result["status"], "ok")
            self.assertNotIn("write_policy_violation", result["structured_digest"])
            worktree = bot._repo_worktree_root(cfg, repo_id=repo_id) / "skynet" / "slot1"
            self.assertEqual(bot._git_status_porcelain(worktree), "")
            self.assertIn("Committed a direct improvement", result["summary"])

    def test_orchestrator_run_codex_recovers_base_repo_branch_violation(self) -> None:
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
                job_id="job-skynet-base-branch",
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
                artifacts_dir=str((cfg.artifacts_root / "job-skynet-base-branch").resolve()),
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
                subprocess.run(
                    ["git", "checkout", "-b", "feature/skynet-base-write"],
                    cwd=str(repo),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                (repo / "bot.py").write_text("VALUE = 2\n", encoding="utf-8")
                stdout_path = Path(td) / "codex_stdout.jsonl"
                stderr_path = Path(td) / "codex_stderr.log"
                last_msg_path = Path(td) / "codex_last.txt"
                stdout_path.write_text("", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                last_msg_path.write_text("Controller touched the base repo directly.\n", encoding="utf-8")
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
            violation = result["structured_digest"].get("write_policy_violation")
            self.assertIsInstance(violation, dict)
            self.assertTrue(any(str(path).startswith("base_repo:BRANCH:") for path in violation.get("changed_paths", [])))
            self.assertTrue(violation.get("branch_changes"))
            self.assertEqual(bot._git_current_branch(repo), "main")
            self.assertEqual(bot._git_status_porcelain(repo), "")
            self.assertEqual((repo / "bot.py").read_text(encoding="utf-8"), "VALUE = 1\n")
            stash = subprocess.run(["git", "stash", "list"], cwd=str(repo), check=True, capture_output=True, text=True)
            self.assertIn("codexbot-read-only-violation-skynet-job-skyn-base_repo", stash.stdout)

    def test_orchestrator_run_codex_ignores_preexisting_dirty_base_repo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "bot.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=str(repo), check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True, text=True)
            (repo / "bot.py").write_text("VALUE = 99\n", encoding="utf-8")

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
                job_id="job-skynet-preexisting-dirty",
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
                artifacts_dir=str((cfg.artifacts_root / "job-skynet-preexisting-dirty").resolve()),
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
                stdout_path = Path(td) / "codex_stdout.jsonl"
                stderr_path = Path(td) / "codex_stderr.log"
                last_msg_path = Path(td) / "codex_last.txt"
                stdout_path.write_text("", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                last_msg_path.write_text("Skynet delegated no direct repo changes.\n", encoding="utf-8")
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

            self.assertEqual(result["status"], "ok")
            self.assertNotIn("write_policy_violation", result["structured_digest"])
            self.assertIn("VALUE = 99", (repo / "bot.py").read_text(encoding="utf-8"))
            self.assertIn("bot.py", bot._git_status_porcelain(repo))
            stash = subprocess.run(["git", "stash", "list"], cwd=str(repo), check=True, capture_output=True, text=True)
            self.assertEqual(stash.stdout.strip(), "")

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
            self.assertEqual(task.model, "gpt-5.5")
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
            self.assertEqual(task.model, "gpt-5.5")
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

            with self.assertLogs("codexbot", level="WARNING") as cm:
                off = bot._drain_pending_updates(cfg, _API())  # type: ignore[arg-type]
            self.assertEqual(off, 0)
            joined = "\n".join(cm.output)
            self.assertIn("WARNING:codexbot:Failed to drain pending Telegram updates; continuing without drain: boom", joined)
            self.assertNotIn("Traceback", joined)

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

            with self.assertLogs("codexbot", level="INFO") as cm:
                off = bot._drain_pending_updates(cfg, _API())  # type: ignore[arg-type]
            self.assertEqual(off, 43)
            joined = "\n".join(cm.output)
            self.assertIn("WARNING:codexbot:Failed to drain pending Telegram updates; continuing without drain: boom", joined)
            self.assertIn("INFO:codexbot:Drained 2 pending Telegram updates; next offset=43", joined)


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


class TestBypassAwareEnforcement(unittest.TestCase):
    def test_forced_mode_still_applies_when_bypass_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            state_file.write_text("{}\n", encoding="utf-8")
            cfg = TestStateHandling()._cfg(state_file)
            with patch.dict(os.environ, {}, clear=True), patch("bot._effective_bypass_sandbox", return_value=True):
                forced = bot._orchestrator_forced_mode_for_role(cfg, role="skynet", chat_id=123)  # type: ignore[attr-defined]
            self.assertEqual(forced, "ro")

    def test_controller_git_write_guard_blocks_mutating_git_commands(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            guard = Path(td) / "git"
            guard.write_text(bot._controller_git_write_guard_script(), encoding="utf-8")
            guard.chmod(0o755)
            env = dict(os.environ)
            env["PONCEBOT_REAL_GIT"] = "/bin/echo"
            env["PONCEBOT_GIT_WRITE_GUARD_ROOTS"] = "/tmp/repo"

            ok = subprocess.run([str(guard), "status", "--short"], env=env, capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0)
            self.assertIn("status --short", ok.stdout)

            blocked = subprocess.run([str(guard), "-C", "/tmp/repo", "push", "origin", "main"], env=env, capture_output=True, text=True)
            self.assertEqual(blocked.returncode, 126)
            self.assertIn("blocked: git push", blocked.stderr)

    def test_controller_git_write_guard_allows_temp_git_repos_outside_protected_roots(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            guard = root / "git"
            guard.write_text(bot._controller_git_write_guard_script(), encoding="utf-8")
            guard.chmod(0o755)

            protected = root / "protected"
            protected.mkdir()
            subprocess.run(["git", "-C", str(protected), "init"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(protected), "config", "user.name", "tester"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(protected), "config", "user.email", "tester@example.com"], check=True, capture_output=True, text=True)
            (protected / "README.md").write_text("protected\n", encoding="utf-8")

            env = dict(os.environ)
            env["PONCEBOT_REAL_GIT"] = os.environ.get("PONCEBOT_REAL_GIT") or shutil.which("git") or "/usr/bin/git"
            env["PONCEBOT_GIT_WRITE_GUARD_ROOTS"] = str(protected.resolve())

            blocked = subprocess.run([str(guard), "-C", str(protected), "add", "README.md"], env=env, capture_output=True, text=True)
            self.assertEqual(blocked.returncode, 126)
            self.assertIn("blocked: git add", blocked.stderr)

            repo = root / "temp_repo"
            repo.mkdir()
            subprocess.run([str(guard), "-C", str(repo), "init"], env=env, check=True, capture_output=True, text=True)
            subprocess.run([str(guard), "-C", str(repo), "config", "user.name", "tester"], env=env, check=True, capture_output=True, text=True)
            subprocess.run([str(guard), "-C", str(repo), "config", "user.email", "tester@example.com"], env=env, check=True, capture_output=True, text=True)
            (repo / "app.txt").write_text("temp\n", encoding="utf-8")
            subprocess.run([str(guard), "-C", str(repo), "add", "app.txt"], env=env, check=True, capture_output=True, text=True)
            subprocess.run([str(guard), "-C", str(repo), "commit", "-m", "init"], env=env, check=True, capture_output=True, text=True)

            clone_dir = root / "clone_repo"
            subprocess.run([str(guard), "clone", str(repo), str(clone_dir)], env=env, cwd=str(root), check=True, capture_output=True, text=True)
            subprocess.run([str(guard), "-C", str(clone_dir), "checkout", "-b", "feature/test"], env=env, check=True, capture_output=True, text=True)

            branch = subprocess.run(
                ["git", "-C", str(clone_dir), "branch", "--show-current"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(branch, "feature/test")

    def test_controller_git_write_guard_allows_pytest_temp_repos_under_protected_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            guard = root / "git"
            guard.write_text(bot._controller_git_write_guard_script(), encoding="utf-8")
            guard.chmod(0o755)

            protected = root / "protected"
            protected.mkdir()
            subprocess.run(["git", "-C", str(protected), "init"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(protected), "config", "user.name", "tester"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(protected), "config", "user.email", "tester@example.com"], check=True, capture_output=True, text=True)
            (protected / "README.md").write_text("protected\n", encoding="utf-8")

            pytest_repo = protected / ".codexbot_tmp" / "pytest-of-tester" / "pytest-0" / "test_guard0" / "repo"
            pytest_repo.mkdir(parents=True)

            env = dict(os.environ)
            env["PONCEBOT_REAL_GIT"] = os.environ.get("PONCEBOT_REAL_GIT") or shutil.which("git") or "/usr/bin/git"
            env["PONCEBOT_GIT_WRITE_GUARD_ROOTS"] = str(protected.resolve())

            pytest_plain_dir = protected / ".codexbot_tmp" / "pytest-of-tester" / "pytest-0" / "test_plain0"
            pytest_plain_dir.mkdir(parents=True)
            env["PYTEST_CURRENT_TEST"] = "test_bot.py::test_guard (call)"
            protected_readme_from_plain = os.path.relpath(protected / "README.md", pytest_plain_dir)
            blocked_parent_worktree = subprocess.run(
                [str(guard), "-C", str(pytest_plain_dir), "add", protected_readme_from_plain],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(blocked_parent_worktree.returncode, 126)
            self.assertIn("blocked: git add", blocked_parent_worktree.stderr)

            subprocess.run([str(guard), "-C", str(pytest_repo), "init"], env=env, check=True, capture_output=True, text=True)
            subprocess.run([str(guard), "-C", str(pytest_repo), "config", "user.name", "tester"], env=env, check=True, capture_output=True, text=True)
            subprocess.run([str(guard), "-C", str(pytest_repo), "config", "user.email", "tester@example.com"], env=env, check=True, capture_output=True, text=True)
            (pytest_repo / "app.txt").write_text("temp\n", encoding="utf-8")

            env.pop("PYTEST_CURRENT_TEST", None)
            blocked_temp = subprocess.run([str(guard), "-C", str(pytest_repo), "add", "app.txt"], env=env, capture_output=True, text=True)
            self.assertEqual(blocked_temp.returncode, 126)
            self.assertIn("blocked: git add", blocked_temp.stderr)

            env["PYTEST_CURRENT_TEST"] = "test_bot.py::test_guard (call)"
            subprocess.run([str(guard), "-C", str(pytest_repo), "add", "app.txt"], env=env, check=True, capture_output=True, text=True)
            subprocess.run([str(guard), "-C", str(pytest_repo), "commit", "-m", "init"], env=env, check=True, capture_output=True, text=True)

            blocked_source = subprocess.run([str(guard), "-C", str(protected), "add", "README.md"], env=env, capture_output=True, text=True)
            self.assertEqual(blocked_source.returncode, 126)
            self.assertIn("blocked: git add", blocked_source.stderr)

    def test_controller_git_write_guard_allows_pytest_tmp_repos_under_protected_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            guard = root / "git"
            guard.write_text(bot._controller_git_write_guard_script(), encoding="utf-8")
            guard.chmod(0o755)

            protected = root / "protected"
            protected.mkdir()
            subprocess.run(["git", "-C", str(protected), "init"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(protected), "config", "user.name", "tester"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(protected), "config", "user.email", "tester@example.com"], check=True, capture_output=True, text=True)
            (protected / "README.md").write_text("protected\n", encoding="utf-8")

            pytest_repo = protected / ".codexbot_tmp" / "tmpabc123" / "repo"
            pytest_repo.mkdir(parents=True)

            env = dict(os.environ)
            env["PONCEBOT_REAL_GIT"] = os.environ.get("PONCEBOT_REAL_GIT") or shutil.which("git") or "/usr/bin/git"
            env["PONCEBOT_GIT_WRITE_GUARD_ROOTS"] = str(protected.resolve())

            subprocess.run([str(guard), "-C", str(pytest_repo), "init"], env=env, check=True, capture_output=True, text=True)
            subprocess.run([str(guard), "-C", str(pytest_repo), "config", "user.name", "tester"], env=env, check=True, capture_output=True, text=True)
            subprocess.run([str(guard), "-C", str(pytest_repo), "config", "user.email", "tester@example.com"], env=env, check=True, capture_output=True, text=True)
            (pytest_repo / "app.txt").write_text("temp\n", encoding="utf-8")

            env.pop("PYTEST_CURRENT_TEST", None)
            blocked_temp = subprocess.run([str(guard), "-C", str(pytest_repo), "add", "app.txt"], env=env, capture_output=True, text=True)
            self.assertEqual(blocked_temp.returncode, 126)
            self.assertIn("blocked: git add", blocked_temp.stderr)

            env["PYTEST_CURRENT_TEST"] = "test_bot.py::test_guard (call)"
            subprocess.run([str(guard), "-C", str(pytest_repo), "add", "app.txt"], env=env, check=True, capture_output=True, text=True)
            subprocess.run([str(guard), "-C", str(pytest_repo), "commit", "-m", "init"], env=env, check=True, capture_output=True, text=True)

            blocked_source = subprocess.run([str(guard), "-C", str(protected), "add", "README.md"], env=env, capture_output=True, text=True)
            self.assertEqual(blocked_source.returncode, 126)
            self.assertIn("blocked: git add", blocked_source.stderr)

    def test_codex_runner_exports_guard_roots_for_controller_git_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td) / "snapshot"
            source = Path(td) / "source"
            workdir.mkdir()
            source.mkdir()
            state_file = Path(td) / "state.json"
            state_file.write_text("{}\n", encoding="utf-8")
            cfg = TestStateHandling()._cfg(state_file)
            cfg = bot.dataclasses.replace(cfg, codex_workdir=workdir)
            runner = bot.CodexRunner(
                cfg,
                guard_git_writes=True,
                guard_git_write_roots=[workdir, source],
            )

            with patch.object(bot.subprocess, "Popen") as popen:
                popen.side_effect = FileNotFoundError("no codex in test")
                with self.assertRaises(FileNotFoundError):
                    runner.start_threaded_new(prompt="hi", mode_hint="ro")
                env = popen.call_args.kwargs["env"]

            exported = str(env.get("PONCEBOT_GIT_WRITE_GUARD_ROOTS") or "")
            self.assertIn(str(workdir.resolve()), exported)
            self.assertIn(str(source.resolve()), exported)

    def test_read_only_violation_stash_cleans_worktree(self) -> None:

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            artifacts = Path(td) / "artifacts"
            repo.mkdir()
            artifacts.mkdir()
            subprocess.run(["git", "-C", str(repo), "init"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "manolosake"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "manolosake@gmail.com"], check=True)
            tracked = repo / "bot.py"
            tracked.write_text("print('old')\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "bot.py"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            tracked.write_text("print('new')\n", encoding="utf-8")
            (repo / "scratch.txt").write_text("temp\n", encoding="utf-8")

            result = bot._stash_read_only_write_violation(
                repo,
                role="skynet",
                job_id="12345678-aaaa",
                label="worktree",
                artifacts_dir=artifacts,
            )

            self.assertTrue(result["stash_created"])
            self.assertEqual(bot._git_status_porcelain(repo), "")
            self.assertIn("bot.py", result["changed_paths"])
            self.assertTrue(Path(result["artifact"]).exists())

    def test_git_changed_paths_ignores_codexbot_temp_artifacts(self) -> None:
        status = "\n".join(
            [
                " M .codexbot_tmp/adb.1000.log",
                " D .codexbot_tmp/android-aponce/RECOVERY_NOTE.txt",
                "?? .codexbot_tmp/new.log",
                " M bot.py",
            ]
        )

        self.assertEqual(bot._git_changed_paths_from_porcelain(status), ["bot.py"])

    def test_autocommit_removes_internal_worktree_marker_before_push(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = root / "origin.git"
            repo = root / "repo"
            subprocess.run(["git", "init", "--bare", str(origin)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "clone", str(origin), str(repo)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "manolosake"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "manolosake@gmail.com"], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", "-b", "main"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (repo / "README.md").write_text("base\n", encoding="utf-8")
            (repo / ".poncebot_managed_worktree").write_text("accidentally tracked\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "README.md", ".poncebot_managed_worktree"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "-C", str(repo), "push", "origin", "main"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "-C", str(repo), "checkout", "-b", "feature/order-marker"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            (repo / "app.txt").write_text("real change\n", encoding="utf-8")
            task = bot.Task.new(
                job_id="job-marker-cleanup",
                source="test",
                role="backend",
                input_text="Implement change.",
                request_type="task",
                priority=1,
                model="gpt-5.5",
                effort="high",
                mode_hint="rw",
                requires_approval=False,
                max_cost_window_usd=0.0,
                chat_id=1,
                labels={"key": "marker_cleanup"},
            )

            result = bot._autocommit_push_order_branch(
                worktree_dir=repo,
                order_branch="feature/order-marker",
                task=task,
            )

            self.assertEqual(result["status"], "ok")
            tree = subprocess.run(
                ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "origin/feature/order-marker"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            self.assertIn("app.txt", tree)
            self.assertNotIn(".poncebot_managed_worktree", tree)


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
            with patch.object(bot.LOG, "warning") as warn_mock:
                resp, job = bot._parse_job(cfg, msg)
            self.assertIsNone(job)
            self.assertIn("breakglass enabled", resp)
            warn_mock.assert_called_once()
            self.assertIn("BREAKGLASS ENABLED", str(warn_mock.call_args.args[0]))

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
            with patch.object(bot.LOG, "warning") as warning_mock:
                self.assertTrue(bot._effective_bypass_sandbox(cfg, chat_id=1))
            warning_mock.assert_not_called()
            active, raw = bot._breakglass_is_active(cfg)
            self.assertTrue(active)
            self.assertEqual(str(raw.get("reason") or ""), "persistent_factory_runtime")
            self.assertEqual(str(raw.get("source") or ""), "auto_env_refresh")
            audit = bot._get_state(cfg).get("security_audit")
            rows = list(audit) if isinstance(audit, list) else []
            self.assertTrue(rows)
            self.assertEqual(str(rows[-1].get("event") or ""), "breakglass.enabled")
            details = rows[-1].get("details")
            self.assertTrue(isinstance(details, dict))
            details = details if isinstance(details, dict) else {}
            self.assertEqual(str(details.get("source") or ""), "auto_env_refresh")
            self.assertEqual(str(details.get("reason") or ""), "persistent_factory_runtime")

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

    def test_inject_local_specialists_skips_proactive_lane_by_default_codex_first(self) -> None:
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
        self.assertNotIn("architect_local", roles)
        self.assertNotIn("implementer_local", roles)
        self.assertNotIn("reviewer_local", roles)

    def test_inject_local_specialists_adds_all_three_roles_for_proactive_lane_when_local_only(self) -> None:
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
        with patch.dict(os.environ, {"BOT_SKYNET_FACTORY_LOCAL_ONLY": "1"}, clear=False):
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

    def test_normalize_contract_promotes_write_enabled_roles_to_rw(self) -> None:
        spec = bot.TaskSpec(
            key="backend_impl",
            role="backend",
            text="Implement a bounded feature.",
            mode_hint="ro",
            priority=1,
            requires_approval=False,
            acceptance_criteria=["Patch is produced"],
            definition_of_done=["Tests run"],
            eta_minutes=45,
            sla_tier="high",
        )
        normalized = bot._normalize_task_spec_contract(spec, root_ticket="ticket-123")
        self.assertEqual(normalized.mode_hint, "rw")

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
                with patch.dict(os.environ, {"BOT_AUTONOMOUS_LOCAL_FIRST": "1"}, clear=False):
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


class TestOrderBranchGitHelpers(unittest.TestCase):
    def test_ensure_branch_fetches_default_branch_outside_narrow_refspec(self) -> None:
        def run(cmd: list[str], cwd: Path | None = None) -> str:
            return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, capture_output=True, text=True).stdout.strip()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = root / "origin.git"
            seed = root / "seed"
            repo = root / "repo"

            run(["git", "init", "--bare", str(origin)])
            run(["git", "clone", str(origin), str(seed)])
            run(["git", "config", "user.email", "test@example.com"], cwd=seed)
            run(["git", "config", "user.name", "test"], cwd=seed)
            (seed / "marker.txt").write_text("main\n", encoding="utf-8")
            run(["git", "add", "marker.txt"], cwd=seed)
            run(["git", "commit", "-m", "main"], cwd=seed)
            run(["git", "push", "origin", "HEAD:main"], cwd=seed)

            run(["git", "checkout", "-b", "codex/codexbot-workflow-v2"], cwd=seed)
            (seed / "marker.txt").write_text("old default\n", encoding="utf-8")
            run(["git", "commit", "-am", "old default"], cwd=seed)
            run(["git", "push", "origin", "HEAD:codex/codexbot-workflow-v2"], cwd=seed)

            run(["git", "clone", "--branch", "main", str(origin), str(repo)])
            run(["git", "config", "--replace-all", "remote.origin.fetch", "+refs/heads/main:refs/remotes/origin/main"], cwd=repo)
            run(
                [
                    "git",
                    "fetch",
                    "origin",
                    "refs/heads/codex/codexbot-workflow-v2:refs/remotes/origin/codex/codexbot-workflow-v2",
                ],
                cwd=repo,
            )
            stale_default = run(["git", "rev-parse", "refs/remotes/origin/codex/codexbot-workflow-v2"], cwd=repo)

            (seed / "marker.txt").write_text("new default\n", encoding="utf-8")
            run(["git", "commit", "-am", "new default"], cwd=seed)
            run(["git", "push", "origin", "HEAD:codex/codexbot-workflow-v2"], cwd=seed)
            latest_default = run(["git", "rev-parse", "HEAD"], cwd=seed)
            self.assertNotEqual(stale_default, latest_default)

            ok, msg = bot._git_ensure_branch_from_main(
                repo,
                "feature/order-test",
                default_branch="codex/codexbot-workflow-v2",
            )

            self.assertTrue(ok, msg)
            order_head = run(["git", "--git-dir", str(origin), "rev-parse", "refs/heads/feature/order-test"])
            self.assertEqual(order_head, latest_default)

    def test_merge_order_branch_fast_forwards_base_checkout_after_push(self) -> None:
        def run(cmd: list[str], cwd: Path | None = None) -> str:
            return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, capture_output=True, text=True).stdout.strip()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = root / "origin.git"
            seed = root / "seed"
            repo = root / "repo"

            run(["git", "init", "--bare", str(origin)])
            run(["git", "clone", str(origin), str(seed)])
            run(["git", "config", "user.email", "test@example.com"], cwd=seed)
            run(["git", "config", "user.name", "test"], cwd=seed)
            (seed / "README.md").write_text("main\n", encoding="utf-8")
            run(["git", "add", "README.md"], cwd=seed)
            run(["git", "commit", "-m", "main"], cwd=seed)
            run(["git", "push", "origin", "HEAD:main"], cwd=seed)

            run(["git", "checkout", "-b", "feature/order-sync"], cwd=seed)
            (seed / "feature.txt").write_text("feature\n", encoding="utf-8")
            run(["git", "add", "feature.txt"], cwd=seed)
            run(["git", "commit", "-m", "feature"], cwd=seed)
            run(["git", "push", "origin", "HEAD:feature/order-sync"], cwd=seed)

            run(["git", "clone", "--branch", "main", str(origin), str(repo)])
            run(["git", "config", "user.email", "test@example.com"], cwd=repo)
            run(["git", "config", "user.name", "test"], cwd=repo)
            before = run(["git", "rev-parse", "HEAD"], cwd=repo)

            ok, msg, merge_commit = bot._merge_order_branch_to_main(
                repo=repo,
                order_branch="feature/order-sync",
                order_id="order-sync",
                default_branch="main",
            )

            self.assertTrue(ok, msg)
            self.assertEqual(msg, "merged_to_main")
            self.assertNotEqual(before, merge_commit)
            self.assertEqual(run(["git", "rev-parse", "--short", "HEAD"], cwd=repo), merge_commit)
            self.assertEqual(run(["git", "rev-parse", "--short", "origin/main"], cwd=repo), merge_commit)

    def test_merge_order_branch_rejects_whitespace_only_delta(self) -> None:
        def run(cmd: list[str], cwd: Path | None = None) -> str:
            return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, capture_output=True, text=True).stdout.strip()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = root / "origin.git"
            seed = root / "seed"
            repo = root / "repo"

            run(["git", "init", "--bare", str(origin)])
            run(["git", "clone", str(origin), str(seed)])
            run(["git", "config", "user.email", "test@example.com"], cwd=seed)
            run(["git", "config", "user.name", "test"], cwd=seed)
            (seed / "README.md").write_text("alpha\n\nbeta\n", encoding="utf-8")
            run(["git", "add", "README.md"], cwd=seed)
            run(["git", "commit", "-m", "main"], cwd=seed)
            run(["git", "push", "origin", "HEAD:main"], cwd=seed)

            run(["git", "checkout", "-b", "feature/order-whitespace"], cwd=seed)
            (seed / "README.md").write_text("alpha\nbeta\n", encoding="utf-8")
            run(["git", "commit", "-am", "remove blank line"], cwd=seed)
            run(["git", "push", "origin", "HEAD:feature/order-whitespace"], cwd=seed)

            run(["git", "clone", "--branch", "main", str(origin), str(repo)])
            run(["git", "config", "user.email", "test@example.com"], cwd=repo)
            run(["git", "config", "user.name", "test"], cwd=repo)
            before = run(["git", "rev-parse", "origin/main"], cwd=repo)

            ok, msg, merge_commit = bot._merge_order_branch_to_main(
                repo=repo,
                order_branch="feature/order-whitespace",
                order_id="order-whitespace",
                default_branch="main",
            )

            self.assertFalse(ok)
            self.assertEqual(msg, "branch_has_no_material_delta_vs_main")
            self.assertIsNone(merge_commit)
            self.assertEqual(run(["git", "rev-parse", "origin/main"], cwd=repo), before)


class TestSkynetLocalOnlyProactivePolicy(unittest.TestCase):
    def test_skynet_factory_local_only_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BOT_SKYNET_FACTORY_LOCAL_ONLY", None)
            self.assertFalse(bot._skynet_factory_local_only_enabled())

    def test_proactive_catalog_includes_project_incubator(self) -> None:
        cfg = SimpleNamespace(mobile_app_project_dir=Path("/tmp/mobile"))

        with patch.dict(os.environ, {"BOT_PROJECT_INCUBATOR_ROOT": "/home/aponce"}, clear=False):
            catalog = bot._proactive_initiatives_catalog(cfg)

        incubator = next((item for item in catalog if item.get("key") == "project-incubator"), None)
        self.assertIsNotNone(incubator)
        if incubator is None:
            self.fail("project-incubator initiative missing")
        self.assertEqual(incubator["lane"], "incubator")
        self.assertEqual(incubator["project_hint"], "/home/aponce")
        self.assertIn("new project", f"{incubator['title']} {incubator['goal']}".lower())

    def test_project_incubator_workspace_uses_requested_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "home"

            workspace = bot._ensure_project_workspace(
                project_id="abc12345",
                title="New Useful Thing",
                created_by="skynet",
                root_dir=root,
            )

            path = Path(workspace["path"])
            self.assertEqual(path.parent, root.resolve())
            self.assertTrue((path / "PROJECT_MANIFEST.json").exists())

    def test_project_incubator_workspace_reuses_existing_project_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "home"

            first = bot._ensure_project_workspace(
                project_id="abc12345",
                title="First Title",
                created_by="skynet",
                root_dir=root,
            )
            second = bot._ensure_project_workspace(
                project_id="abc12345",
                title="Second Title",
                created_by="jarvis",
                root_dir=root,
            )

            first_path = Path(first["path"])
            second_path = Path(second["path"])
            self.assertEqual(first_path, second_path)
            manifest = json.loads((second_path / "PROJECT_MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["project_id"], "abc12345")
            self.assertEqual(manifest["name"], "First Title")
            self.assertEqual(manifest["created_by"], "skynet")
            self.assertEqual(len([p for p in root.iterdir() if p.is_dir()]), 1)

    def test_project_incubator_workspace_repairs_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "home"
            workspace_dir = root / "20260511-sample-abcd1234"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / "PROJECT_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "project_id": "abc12345",
                        "name": "Useful Project",
                        "path": "/tmp/stale-path",
                        "runtime_mode": "venv",
                        "ports": [],
                        "created_by": "skynet",
                        "created_at": "2026-05-11T10:52:07Z",
                        "notes": "original",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            workspace = bot._ensure_project_workspace(
                project_id="abc12345",
                title="Useful Project",
                created_by="jarvis",
                root_dir=root,
            )

            path = Path(workspace["path"])
            self.assertEqual(path, workspace_dir.resolve())
            manifest = json.loads((path / "PROJECT_MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["path"], str(workspace_dir.resolve()))
            self.assertEqual(manifest["created_at"], "2026-05-11T10:52:07Z")
            self.assertEqual(manifest["notes"], "original")

    def test_project_incubator_workspace_recovers_missing_manifest_from_existing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "home"
            workspace_dir = root / "20260511-useful-project-abc12345"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / "runproof-board").mkdir()

            workspace = bot._ensure_project_workspace(
                project_id="abc12345-0000-0000-0000-000000000000",
                title="Useful Project",
                created_by="jarvis",
                root_dir=root,
            )

            path = Path(workspace["path"])
            self.assertEqual(path, workspace_dir.resolve())
            manifest = json.loads((path / "PROJECT_MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["project_id"], "abc12345-0000-0000-0000-000000000000")
            self.assertEqual(manifest["name"], "Useful Project")
            self.assertEqual(manifest["created_by"], "jarvis")
            self.assertEqual(len([p for p in root.iterdir() if p.is_dir()]), 1)

    def test_project_incubator_workspace_recovers_corrupt_manifest_from_existing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "home"
            workspace_dir = root / "20260511-useful-project-abc12345"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / "PROJECT_MANIFEST.json").write_text("{broken json\n", encoding="utf-8")

            workspace = bot._ensure_project_workspace(
                project_id="abc12345-0000-0000-0000-000000000000",
                title="Useful Project",
                created_by="jarvis",
                root_dir=root,
            )

            path = Path(workspace["path"])
            self.assertEqual(path, workspace_dir.resolve())
            manifest = json.loads((path / "PROJECT_MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["project_id"], "abc12345-0000-0000-0000-000000000000")
            self.assertEqual(manifest["path"], str(workspace_dir.resolve()))
            self.assertEqual(len([p for p in root.iterdir() if p.is_dir()]), 1)

    def test_project_incubator_due_after_repo_sprints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = SimpleNamespace(state_file=Path(td) / "state.json")
            with patch.dict(
                os.environ,
                {
                    "BOT_PROACTIVE_PROJECT_INCUBATOR_ENABLED": "1",
                    "BOT_PROACTIVE_PROJECT_INCUBATOR_EVERY_N_REPO_SPRINTS": "3",
                },
                clear=False,
            ):
                bot._update_state(
                    cfg,
                    lambda st: st.update(
                        {
                            "proactive_repo_index": 3,
                            "proactive_project_incubator_index": 0,
                        }
                    ),
                )
                self.assertTrue(bot._project_incubator_due(cfg, occupied_keys=set()))
                self.assertFalse(bot._project_incubator_due(cfg, occupied_keys={"project-incubator"}))

                bot._mark_project_incubator_spawned(cfg)

                self.assertFalse(bot._project_incubator_due(cfg, occupied_keys=set()))

    def test_project_incubator_external_delivery_satisfies_merge_gate(self) -> None:
        def run(cmd: list[str], cwd: Path) -> str:
            return subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "home"
            project = root / "local-ops-briefboard"
            project.mkdir(parents=True)
            run(["git", "init", "-b", "main"], cwd=project)
            run(["git", "config", "user.email", "test@example.com"], cwd=project)
            run(["git", "config", "user.name", "test"], cwd=project)
            (project / "README.md").write_text("# Local Ops Briefboard\n", encoding="utf-8")
            run(["git", "add", "README.md"], cwd=project)
            run(["git", "commit", "-m", "initial"], cwd=project)

            trace = {
                "initiative_key": "project-incubator",
                "initiative_lane": "incubator",
                "order_branch": "feature/order-incubator",
                "result_summary": (
                    f"VERIFIED_IMPROVEMENT: `NEW_PROJECT_PHASE`. Created and validated `{project}`, "
                    "a new git-backed local operator project with README and validation evidence."
                ),
            }

            with patch.dict(os.environ, {"BOT_PROJECT_INCUBATOR_ROOT": str(root)}, clear=False):
                evidence = bot._trace_project_incubator_delivery_evidence(trace)
                merge_required, branch = bot._order_trace_requires_merge(trace)

            self.assertTrue(evidence["ok"], evidence)
            self.assertEqual(evidence["project_path"], str(project.resolve()))
            self.assertFalse(merge_required)
            self.assertEqual(branch, "feature/order-incubator")

    def test_project_incubator_nested_git_project_is_valid_delivery(self) -> None:
        def run(cmd: list[str], cwd: Path) -> str:
            return subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "home"
            project = root / "20260511-studio-cycle-new-product-incubator-b0b901de" / "scopeguard-mvp"
            project.mkdir(parents=True)
            run(["git", "init", "-b", "main"], cwd=project)
            run(["git", "config", "user.email", "test@example.com"], cwd=project)
            run(["git", "config", "user.name", "test"], cwd=project)
            (project / "README.md").write_text("# ScopeGuard MVP\n\nRunnable demo and validation evidence.\n", encoding="utf-8")
            run(["git", "add", "README.md"], cwd=project)
            run(["git", "commit", "-m", "Build ScopeGuard MVP"], cwd=project)
            trace = {
                "initiative_key": "project-incubator",
                "initiative_lane": "incubator",
                "result_summary": f"Implemented scopeguard-mvp, validated the demo offline, and left the git-backed MVP clean at {project}.",
                "structured_digest": {
                    "branch_sync": {"status": "skipped", "reason": "no_changes"},
                },
            }
            task = SimpleNamespace(role="backend", state="done", trace=trace, created_at=1.0, updated_at=2.0)

            with patch.dict(os.environ, {"BOT_PROJECT_INCUBATOR_ROOT": str(root)}, clear=False):
                evidence = bot._trace_project_incubator_delivery_evidence(trace)
                artifact_only = bot._task_is_artifact_only_delivery_claim(task, trace=trace)

        self.assertTrue(evidence["ok"], evidence)
        self.assertEqual(evidence["project_path"], str(project.resolve()))
        self.assertFalse(artifact_only)

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

    def test_recovery_funnel_closes_after_root_verified_signal_and_qa_ready(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            role="skynet",
            state="blocked",
            trace={
                "result_status": "blocked",
                "structured_digest": {
                    "summary": "VERIFIED_IMPROVEMENT: hardened CSV ingestion and added regression coverage.",
                },
            },
            labels={},
            created_at=90.0,
            updated_at=100.0,
        )
        children = [
            SimpleNamespace(
                job_id="impl",
                role="backend",
                state="done",
                labels={"key": "local_impl_recover_csv_overflow"},
                trace={
                    "result_summary": "Applied bounded fix with validation evidence.",
                    "slice_patch_applied": True,
                    "slice_validation_ok": True,
                    "patch_info": {"changed_files": ["server/csv_data.py", "test_csv_data.py"], "validation_ok": True},
                },
                created_at=110.0,
                updated_at=120.0,
            ),
            SimpleNamespace(
                job_id="qa",
                role="qa",
                state="done",
                labels={"key": "local_review_recover_csv_overflow"},
                trace={
                    "result_summary": (
                        "READY for the newest implementer recovery slice.\n"
                        "Expected command failed due a workspace path prefix mismatch.\n"
                        "Equivalent local compile passed and the unittest suite passed."
                    ),
                },
                created_at=125.0,
                updated_at=130.0,
            ),
        ]

        class FakeQueue:
            def get_job(self, job_id: str):
                return root if job_id == "root" else None

            def jobs_by_parent(self, parent_job_id: str, limit: int = 600):
                return list(children) if parent_job_id == "root" else []

        funnel = bot._collect_order_local_autonomy_funnel(orch_q=FakeQueue(), root_ticket="root", now=140.0)

        self.assertEqual(funnel["quality_gate_status"], "closed")
        self.assertEqual(funnel["slices_closed"], 1)
        self.assertTrue(funnel["improvement_verified"])
        self.assertTrue(bot._order_has_meaningful_improvement(orch_q=FakeQueue(), root_ticket="root"))

    def test_recovery_funnel_closes_legacy_cli_qa_suffix_after_write_policy_block(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            role="skynet",
            state="blocked",
            trace={
                "result_summary": (
                    "Write policy violation: skynet modified repository files directly. "
                    "Skynet factory work must delegate code changes to local specialists."
                ),
                "structured_digest": {
                    "write_policy_violation": {
                        "changed_paths": ["controller_snapshot:server/app.py"],
                    },
                },
            },
            labels={},
            created_at=90.0,
            updated_at=100.0,
        )
        children = [
            SimpleNamespace(
                job_id="impl",
                role="backend",
                state="done",
                labels={"key": "local_impl_recover_e78acab431"},
                trace={
                    "result_summary": "Implemented a bounded backend recovery with validation evidence.",
                    "slice_patch_applied": True,
                    "slice_validation_ok": True,
                    "patch_info": {"changed_files": ["server/app.py"], "validation_ok": True},
                },
                created_at=110.0,
                updated_at=120.0,
            ),
            SimpleNamespace(
                job_id="qa",
                role="qa",
                state="done",
                labels={"key": "local_impl_recover_e78acab431_qa"},
                trace={
                    "result_summary": "PASS: READY because compile and focused regression tests passed.",
                },
                created_at=125.0,
                updated_at=130.0,
            ),
        ]

        class FakeQueue:
            def get_job(self, job_id: str):
                return root if job_id == "root" else None

            def jobs_by_parent(self, parent_job_id: str, limit: int = 600):
                return list(children) if parent_job_id == "root" else []

        self.assertEqual(bot._slice_id_from_local_key("local_impl_recover_e78acab431_qa"), "e78acab431")
        funnel = bot._collect_order_local_autonomy_funnel(orch_q=FakeQueue(), root_ticket="root", now=140.0)

        self.assertEqual(funnel["quality_gate_status"], "closed")
        self.assertEqual(funnel["slices_closed"], 1)
        self.assertTrue(funnel["improvement_verified"])
        self.assertTrue(bot._order_has_meaningful_improvement(orch_q=FakeQueue(), root_ticket="root"))

    def test_sync_order_phase_keeps_proactive_order_in_review_when_qa_needs_rework(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            trace={"order_branch": "feature/order-root", "merge_ready": True},
            labels={},
            parent_job_id="",
            chat_id=1,
            created_at=90.0,
            updated_at=100.0,
        )
        children = [
            SimpleNamespace(
                role="backend",
                state="done",
                labels={"key": "local_impl_guard_slice1"},
                trace={
                    "result_summary": "Implemented bounded fix with tests and evidence.",
                    "slice_patch_applied": True,
                    "slice_validation_ok": True,
                    "patch_info": {"changed_files": ["bot.py"], "validation_ok": True},
                },
                created_at=100.0,
                updated_at=110.0,
            ),
            SimpleNamespace(
                role="qa",
                state="done",
                labels={"key": "local_impl_guard_slice1_qa"},
                trace={"result_summary": "Verdict: NEEDS_REWORK. Blocking issue remains."},
                created_at=120.0,
                updated_at=130.0,
            ),
        ]

        class FakeQueue:
            def __init__(self) -> None:
                self.phases: list[str] = []
                self.statuses: list[str] = []
                self.trace_updates: list[dict[str, object]] = []

            def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                return {
                    "order_id": order_id,
                    "chat_id": chat_id,
                    "status": "active",
                    "phase": "review",
                    "title": "Proactive Sprint: codexbot Reliability + Delivery",
                    "body": "AUTONOMOUS PROACTIVE SPRINT",
                }

            def get_job(self, job_id: str):
                return root

            def jobs_by_parent(self, *, parent_job_id: str, limit: int = 600):
                return children

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def set_order_status(self, order_id: str, chat_id: int, status: str) -> None:
                self.statuses.append(status)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                self.trace_updates.append(dict(kwargs))

        q = FakeQueue()
        with patch.object(bot, "_repo_context_for_order", return_value=({}, Path("."), "main")), patch.object(
            bot, "_order_trace_requires_merge", return_value=(True, "feature/order-root")
        ), patch.object(bot, "_order_has_meaningful_improvement", return_value=True), patch.object(
            bot,
            "_collect_order_local_autonomy_funnel",
            return_value={
                "slices_started": 1,
                "slices_applied": 1,
                "slices_validated": 1,
                "slices_closed": 0,
                "implementer_fail_rate": 0.0,
                "mean_time_to_validated_improvement": None,
                "loop_breaker_count": 0,
                "quality_gate_status": "review",
                "improvement_verified": False,
            },
        ), patch.object(bot, "_order_has_verified_no_change_resolution", return_value=False):
            bot._sync_order_phase_from_runtime(orch_q=q, root_ticket="root", chat_id=1)

        self.assertEqual(q.statuses[-1], "active")
        self.assertEqual(q.phases[-1], "review")
        self.assertFalse(any(update.get("merge_ready") is True for update in q.trace_updates))

    def test_sync_order_phase_blocks_ready_for_merge_when_root_still_blocked(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            state="blocked",
            trace={"order_branch": "feature/order-root", "merge_ready": True},
            labels={},
            parent_job_id="",
            chat_id=1,
            created_at=90.0,
            updated_at=100.0,
        )
        children = [
            SimpleNamespace(
                job_id="impl",
                role="backend",
                state="done",
                labels={"key": "local_impl_guard_slice1"},
                trace={
                    "result_summary": "Implemented bounded fix with tests and evidence.",
                    "slice_patch_applied": True,
                    "slice_validation_ok": True,
                    "patch_info": {"changed_files": ["bot.py"], "validation_ok": True},
                },
                created_at=100.0,
                updated_at=110.0,
            ),
            SimpleNamespace(
                job_id="qa",
                role="qa",
                state="done",
                labels={"key": "local_impl_guard_slice1_qa"},
                trace={"result_summary": "READY. Validation passed."},
                created_at=120.0,
                updated_at=130.0,
            ),
        ]

        class FakeQueue:
            def __init__(self) -> None:
                self.phases: list[str] = []
                self.statuses: list[str] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                return {
                    "order_id": order_id,
                    "chat_id": chat_id,
                    "status": "active",
                    "phase": "ready_for_merge",
                    "title": "Proactive Sprint: codexbot Reliability + Delivery",
                    "body": "AUTONOMOUS PROACTIVE SPRINT",
                }

            def get_job(self, job_id: str):
                return root

            def jobs_by_parent(self, *, parent_job_id: str, limit: int = 600):
                return children

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def set_order_status(self, order_id: str, chat_id: int, status: str) -> None:
                self.statuses.append(status)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        q = FakeQueue()
        with patch.object(bot, "_repo_context_for_order", return_value=({}, Path("."), "main")), patch.object(
            bot, "_order_trace_requires_merge", return_value=(True, "feature/order-root")
        ), patch.object(bot, "_order_has_meaningful_improvement", return_value=True), patch.object(
            bot,
            "_collect_order_local_autonomy_funnel",
            return_value={
                "slices_started": 1,
                "slices_applied": 1,
                "slices_validated": 1,
                "slices_closed": 0,
                "implementer_fail_rate": 0.0,
                "mean_time_to_validated_improvement": None,
                "loop_breaker_count": 0,
                "quality_gate_status": "review",
                "improvement_verified": False,
            },
        ), patch.object(bot, "_order_has_verified_no_change_resolution", return_value=False):
            bot._sync_order_phase_from_runtime(orch_q=q, root_ticket="root", chat_id=1)

        self.assertEqual(q.statuses[-1], "active")
        self.assertEqual(q.phases[-1], "review")
        self.assertTrue(any(update.get("merge_ready") is False for update in q.trace_updates))
        self.assertTrue(any(update.get("operational_gate_reason") == "root_job_still_blocked" for update in q.trace_updates))
        self.assertEqual(q.audit_events[-1]["event_type"], "order.operational_gate_blocked")

    def test_sync_order_phase_promotes_cancelled_root_with_verified_local_delivery(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            role="skynet",
            state="cancelled",
            trace={"order_branch": "feature/order-root"},
            labels={},
            parent_job_id="",
            chat_id=1,
            created_at=90.0,
            updated_at=100.0,
        )
        children = [
            SimpleNamespace(
                job_id="impl",
                role="backend",
                state="done",
                labels={"key": "proactive_cli_reseed_r1"},
                trace={"result_summary": "Implemented concrete change. Tests passed."},
                parent_job_id="root",
                created_at=110.0,
                updated_at=120.0,
            )
        ]

        class FakeQueue:
            def __init__(self) -> None:
                self.phases: list[str] = []
                self.statuses: list[str] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                return {
                    "order_id": order_id,
                    "chat_id": chat_id,
                    "status": "active",
                    "phase": "review",
                    "title": "Proactive Sprint: codexbot Reliability + Delivery",
                    "body": "AUTONOMOUS PROACTIVE SPRINT",
                }

            def get_job(self, job_id: str):
                return root

            def jobs_by_parent(self, parent_job_id: str, limit: int = 600):
                return children

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def set_order_status(self, order_id: str, chat_id: int, status: str) -> None:
                self.statuses.append(status)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                root.trace.update(kwargs)
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        q = FakeQueue()
        with patch.object(bot, "_repo_context_for_order", return_value=({}, Path("."), "main")), patch.object(
            bot, "_order_trace_requires_merge", return_value=(True, "feature/order-root")
        ), patch.object(bot, "_order_has_meaningful_improvement", return_value=True), patch.object(
            bot,
            "_collect_order_local_autonomy_funnel",
            return_value={
                "slices_started": 1,
                "slices_applied": 1,
                "slices_validated": 1,
                "slices_closed": 1,
                "implementer_fail_rate": 0.0,
                "mean_time_to_validated_improvement": 120.0,
                "loop_breaker_count": 0,
                "quality_gate_status": "closed",
                "improvement_verified": True,
            },
        ), patch.object(bot, "_order_has_verified_no_change_resolution", return_value=False):
            bot._sync_order_phase_from_runtime(orch_q=q, root_ticket="root", chat_id=1)

        self.assertEqual(q.statuses[-1], "active")
        self.assertEqual(q.phases[-1], "ready_for_merge")
        self.assertTrue(root.trace["proactive_improvement_closed"])
        self.assertTrue(root.trace["merge_ready"])
        self.assertFalse(any(event["event_type"] == "order.operational_gate_blocked" for event in q.audit_events))

    def _operational_gate_queue(
        self,
        *,
        root_state: str = "done",
        root_trace: dict[str, object] | None = None,
        children: list[SimpleNamespace] | None = None,
    ):
        root = SimpleNamespace(
            job_id="root",
            role="skynet",
            state=root_state,
            trace=dict(root_trace or {}),
            labels={},
            parent_job_id="",
            chat_id=1,
            created_at=90.0,
            updated_at=100.0,
        )

        class FakeQueue:
            def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                return {
                    "order_id": order_id,
                    "chat_id": chat_id,
                    "status": "active",
                    "phase": "ready_for_merge",
                    "title": "Proactive Sprint: codexbot Reliability + Delivery",
                    "body": "AUTONOMOUS PROACTIVE SPRINT",
                }

            def get_job(self, job_id: str):
                return root if job_id == "root" else None

            def jobs_by_parent(self, parent_job_id: str, limit: int = 600):
                return list(children or []) if parent_job_id == "root" else []

        return FakeQueue()

    def _init_git_repo(self, repo: Path, *, clean_commit: bool) -> None:
        subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if clean_commit:
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            (repo / "README.md").write_text("test\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(
                ["git", "commit", "-m", "Initial test commit"],
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _operational_gate_recovery_cfg(self, td: str) -> bot.BotConfig:
        base = TestStateHandling()._cfg(Path(td) / "state.json")
        return bot.BotConfig(**{**base.__dict__, "artifacts_root": Path(td) / "artifacts"})

    def test_operational_gate_reviewer_recovery_skips_without_implementer_delivery_slice(self) -> None:
        class FakeQueue:
            def __init__(self) -> None:
                self.submitted: list[bot.Task] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def submit_task(self, task: bot.Task) -> None:
                self.submitted.append(task)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        children = [
            SimpleNamespace(
                job_id="qa",
                role="reviewer_local",
                state="done",
                labels={"key": "local_review_operational_gate_123"},
                trace={"result_summary": "READY. Validation passed."},
                created_at=120.0,
                updated_at=130.0,
            ),
            SimpleNamespace(
                job_id="arch",
                role="architect_local",
                state="done",
                labels={"key": "local_arch_guard_plan"},
                trace={"result_summary": "Planned recovery."},
                created_at=100.0,
                updated_at=110.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as td:
            q = FakeQueue()
            enqueued = bot._enqueue_operational_gate_reviewer_recovery(
                cfg=self._operational_gate_recovery_cfg(td),
                orch_q=q,
                profiles=None,
                order_row={"order_id": "root"},
                children=children,
                root_trace={},
                chat_id=1,
                now=123.0,
                gate_reason="root_job_still_running",
            )

        self.assertFalse(enqueued)
        self.assertEqual(q.submitted, [])
        self.assertTrue(
            any(update.get("operational_gate_review_skip_reason") == "missing_implementer_delivery_slice" for update in q.trace_updates)
        )
        self.assertEqual(q.audit_events[-1]["event_type"], "order.operational_gate_review_skipped")

    def test_operational_gate_reviewer_recovery_dedupes_missing_implementer_skip(self) -> None:
        class FakeQueue:
            def __init__(self) -> None:
                self.submitted: list[bot.Task] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def submit_task(self, task: bot.Task) -> None:
                self.submitted.append(task)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        with tempfile.TemporaryDirectory() as td:
            q = FakeQueue()
            enqueued = bot._enqueue_operational_gate_reviewer_recovery(
                cfg=self._operational_gate_recovery_cfg(td),
                orch_q=q,
                profiles=None,
                order_row={"order_id": "root"},
                children=[],
                root_trace={
                    "operational_gate_review_skipped_at": 100.0,
                    "operational_gate_review_skip_reason": "missing_implementer_delivery_slice",
                    "operational_gate_review_reason": "root_job_terminal_failed_without_delivery",
                },
                chat_id=1,
                now=200.0,
                gate_reason="root_job_terminal_failed_without_delivery",
            )

        self.assertFalse(enqueued)
        self.assertEqual(q.submitted, [])
        self.assertEqual(q.trace_updates, [])
        self.assertEqual(q.audit_events, [])

    def test_operational_gate_reviewer_recovery_targets_done_implementer_slice(self) -> None:
        class FakeQueue:
            def __init__(self) -> None:
                self.submitted: list[bot.Task] = []
                self.statuses: list[str] = []
                self.phases: list[str] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def submit_task(self, task: bot.Task) -> None:
                self.submitted.append(task)

            def set_order_status(self, order_id: str, chat_id: int, status: str) -> None:
                self.statuses.append(status)

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        children = [
            SimpleNamespace(
                job_id="impl",
                role="implementer_local",
                state="done",
                labels={"key": "local_impl_guard_slice1"},
                trace={
                    "slice_id": "impl_delivery_slice",
                    "result_summary": "Implemented reliability fix and ran focused unittest.",
                },
                created_at=100.0,
                updated_at=120.0,
            )
        ]

        with tempfile.TemporaryDirectory() as td:
            q = FakeQueue()
            enqueued = bot._enqueue_operational_gate_reviewer_recovery(
                cfg=self._operational_gate_recovery_cfg(td),
                orch_q=q,
                profiles=None,
                order_row={"order_id": "root"},
                children=children,
                root_trace={},
                chat_id=1,
                now=123.0,
                gate_reason="root_job_still_running",
            )

        self.assertTrue(enqueued)
        self.assertEqual(len(q.submitted), 1)
        self.assertEqual(q.submitted[0].role, "reviewer_local")
        self.assertEqual(q.submitted[0].trace["slice_id"], "impl_delivery_slice")
        self.assertEqual(q.trace_updates[-1]["operational_gate_review_target_slice_id"], "impl_delivery_slice")

    def test_qa_needs_rework_seeds_backend_rework_and_qa_recheck(self) -> None:
        class FakeQueue:
            def __init__(self, children: list[SimpleNamespace]) -> None:
                self.children = children
                self.submitted: list[bot.Task] = []
                self.phases: list[str] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def jobs_by_parent(self, parent_job_id: str, limit: int = 400):
                return list(self.children)

            def get_job(self, job_id: str):
                return SimpleNamespace(job_id=job_id, trace={})

            def submit_task(self, task: bot.Task) -> None:
                self.submitted.append(task)

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        children = [
            SimpleNamespace(
                job_id="impl",
                role="backend",
                state="done",
                labels={"key": "local_impl_recover_casefix"},
                trace={
                    "result_summary": (
                        "Applied the authoritative controller patch and captured validation evidence. "
                        "Validation passed with git diff --stat and the resulting patch artifact."
                    ),
                    "result_artifacts": ["/tmp/impl-artifacts/changes.patch"],
                    "structured_digest": {
                        "branch_sync": {
                            "status": "ok",
                            "branch": "feature/order-root",
                            "commit": "abc1234",
                        }
                    },
                },
                artifacts_dir="/tmp/impl-artifacts",
                created_at=100.0,
                updated_at=110.0,
            ),
            SimpleNamespace(
                job_id="qa",
                role="qa",
                state="done",
                labels={"key": "local_review_recover_casefix"},
                trace={
                    "result_summary": "NEEDS_REWORK: restore .poncebot_managed_worktree and rerun tests.",
                    "result_status": "done",
                },
                artifacts_dir="/tmp/qa-artifacts",
                created_at=111.0,
                updated_at=120.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as td:
            q = FakeQueue(children)
            enqueued = bot._enqueue_reviewer_local_rework_if_due(
                cfg=self._operational_gate_recovery_cfg(td),
                orch_q=q,
                profiles=None,
                order_row={"order_id": "root", "title": "Proactive Sprint: ExecutiveDashboard Reliability"},
                chat_id=1,
                now=5000.0,
            )

        self.assertTrue(enqueued)
        self.assertEqual([task.role for task in q.submitted], ["backend", "qa"])
        self.assertEqual(q.submitted[0].labels["key"], "local_impl_recover_casefix_r2")
        self.assertEqual(q.submitted[1].labels["key"], "local_review_recover_casefix_r2")
        self.assertEqual(q.submitted[1].depends_on, [q.submitted[0].job_id])
        self.assertEqual(q.phases[-1], "executing")
        self.assertEqual(q.trace_updates[-1]["local_rework_review_job_id"], q.submitted[1].job_id)

    def test_qa_rework_loop_stops_after_newer_no_change_recovery(self) -> None:
        class FakeQueue:
            def __init__(self, children: list[SimpleNamespace]) -> None:
                self.children = children
                self.submitted: list[bot.Task] = []
                self.phases: list[str] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def jobs_by_parent(self, parent_job_id: str, limit: int = 400):
                return list(self.children)

            def get_job(self, job_id: str):
                return SimpleNamespace(job_id=job_id, trace={})

            def submit_task(self, task: bot.Task) -> None:
                self.submitted.append(task)

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        children = [
            SimpleNamespace(
                job_id="impl",
                role="backend",
                state="done",
                labels={"key": "local_impl_recover_casefix"},
                trace={
                    "result_summary": "Applied the patch and validation passed.",
                    "result_artifacts": ["/tmp/impl-artifacts/changes.patch"],
                },
                artifacts_dir="/tmp/impl-artifacts",
                created_at=100.0,
                updated_at=110.0,
            ),
            SimpleNamespace(
                job_id="qa",
                role="qa",
                state="done",
                labels={"key": "local_review_recover_casefix"},
                trace={
                    "result_summary": "NEEDS_REWORK: publish the rework slice evidence before another review.",
                    "result_status": "done",
                },
                artifacts_dir="/tmp/qa-artifacts",
                created_at=111.0,
                updated_at=120.0,
            ),
            SimpleNamespace(
                job_id="impl-no-change",
                role="backend",
                state="done",
                labels={"key": "local_impl_recover_casefix_r2"},
                trace={
                    "result_summary": (
                        "No repository code change was required. The exact blocker was missing "
                        "slice-local evidence, so I added fresh validation evidence."
                    ),
                    "result_status": "done",
                },
                artifacts_dir="/tmp/impl-r2-artifacts",
                created_at=130.0,
                updated_at=140.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as td:
            q = FakeQueue(children)
            enqueued = bot._enqueue_reviewer_local_rework_if_due(
                cfg=self._operational_gate_recovery_cfg(td),
                orch_q=q,
                profiles=None,
                order_row={"order_id": "root", "title": "Proactive Sprint: ExecutiveDashboard Reliability"},
                chat_id=1,
                now=5000.0,
            )

        self.assertFalse(enqueued)
        self.assertEqual(q.submitted, [])
        self.assertEqual(q.phases, [])

    def test_autoship_deploy_failure_recovers_from_main_deploy_state(self) -> None:
        class FakeQueue:
            def __init__(self, db_path: Path) -> None:
                self._storage = SimpleNamespace(path=db_path)
                self.order_status: list[tuple[str, str]] = []
                self.order_phase: list[tuple[str, str]] = []
                self.state_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def set_order_status(self, order_id: str, chat_id: int, status: str) -> None:
                self.order_status.append((order_id, status))

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.order_phase.append((order_id, phase))

            def update_state(self, job_id: str, state: str, **kwargs: object) -> bool:
                self.state_updates.append({"job_id": job_id, "state": state, **kwargs})
                return True

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite"
            (Path(td) / "main_deploy_state.json").write_text(
                json.dumps(
                    {
                        "repos": {
                            "repo-1": {
                                "status": "ok",
                                "deployed_head": "abc1234567890fed",
                                "url": "",
                                "last_deploy_at": 123.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            q = FakeQueue(db_path)

            recovered = bot._recover_controller_snapshot_autoship_deploy_failure(
                orch_q=q,
                order_id="order-1",
                chat_id=1,
                repo_record={"repo_id": "repo-1"},
                root_trace={"repo_id": "repo-1"},
                autoship={"reason": "snapshot_autoship_deploy_failed", "commit": "abc1234567890fed"},
                now=200.0,
            )

        self.assertTrue(recovered)
        self.assertEqual(q.order_status[-1], ("order-1", "done"))
        self.assertEqual(q.order_phase[-1], ("order-1", "done"))
        self.assertEqual(q.state_updates[-1]["state"], "done")
        self.assertEqual(q.state_updates[-1]["deploy_status"], "ok")
        self.assertEqual(q.state_updates[-1]["controller_snapshot_autoship_done"], True)
        self.assertEqual(q.state_updates[-1]["controller_snapshot_autoship_commit"], "abc1234567890fed")
        self.assertEqual(q.audit_events[-1]["event_type"], "order.late_deploy_success_recovered")

    def test_recovery_rework_includes_original_controller_patch_artifacts(self) -> None:
        class FakeQueue:
            def __init__(self, children: list[SimpleNamespace]) -> None:
                self.children = children
                self.submitted: list[bot.Task] = []
                self.phases: list[str] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def jobs_by_parent(self, parent_job_id: str, limit: int = 400):
                return list(self.children)

            def get_job(self, job_id: str):
                return SimpleNamespace(
                    job_id=job_id,
                    trace={
                        "order_branch": "feature/order-root",
                        "result_artifacts": ["/tmp/root/changes.patch"],
                        "structured_digest": {
                            "write_policy_violation": {
                                "status_artifact": "/tmp/root/controller_write_policy_violation.txt",
                                "cleanup": [{"artifact": "/tmp/root/write_policy_controller_snapshot_stash.txt"}],
                            }
                        },
                    },
                )

            def submit_task(self, task: bot.Task) -> None:
                self.submitted.append(task)

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        children = [
            SimpleNamespace(
                job_id="impl",
                role="backend",
                state="done",
                labels={"key": "local_impl_recover_status_api"},
                trace={
                    "result_summary": "No-op reapplication; patch is already present with no new delta.",
                    "result_status": "done",
                },
                artifacts_dir="/tmp/impl-artifacts",
                created_at=100.0,
                updated_at=110.0,
            ),
            SimpleNamespace(
                job_id="qa",
                role="qa",
                state="done",
                labels={"key": "local_review_recover_status_api"},
                trace={
                    "result_summary": "NEEDS_REWORK: recovery slice did not use ticket-scoped controller patch evidence.",
                    "result_status": "done",
                },
                artifacts_dir="/tmp/qa-artifacts",
                created_at=111.0,
                updated_at=120.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as td:
            q = FakeQueue(children)
            enqueued = bot._enqueue_reviewer_local_rework_if_due(
                cfg=self._operational_gate_recovery_cfg(td),
                orch_q=q,
                profiles=None,
                order_row={"order_id": "root", "title": "Proactive Sprint: PonceBot"},
                chat_id=1,
                now=130.0,
            )

        self.assertTrue(enqueued)
        self.assertEqual([task.role for task in q.submitted], ["backend", "qa"])
        prompt = q.submitted[0].input_text
        self.assertIn("ORIGINAL_CONTROLLER_RECOVERY_ARTIFACTS", prompt)
        self.assertIn("/tmp/root/changes.patch", prompt)
        self.assertIn("controller_write_policy_violation.txt", prompt)
        self.assertIn("Do not use stash@{0}", prompt)
        self.assertIn("never replay stash@{0}", prompt)

    def test_worktree_pool_reclaims_same_namespace_legacy_branch(self) -> None:
        from orchestrator import workspaces

        def run(cmd: list[str], cwd: Path | None = None) -> str:
            return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, capture_output=True, text=True).stdout.strip()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            pool = root / "pool"
            repo.mkdir()
            run(["git", "init", "-b", "main"], cwd=repo)
            run(["git", "config", "user.email", "test@example.com"], cwd=repo)
            run(["git", "config", "user.name", "test"], cwd=repo)
            (repo / "README.md").write_text("test\n", encoding="utf-8")
            run(["git", "add", "README.md"], cwd=repo)
            run(["git", "commit", "-m", "Initial"], cwd=repo)

            skynet_slot = pool / "skynet" / "slot1"
            skynet_slot.parent.mkdir(parents=True)
            sibling_branch = workspaces._managed_worktree_branch(base_repo=repo, root=pool, role="backend", slot=1)
            expected_branch = workspaces._managed_worktree_branch(base_repo=repo, root=pool, role="skynet", slot=1)
            run(["git", "worktree", "add", "-B", sibling_branch, str(skynet_slot), "main"], cwd=repo)
            sentinel = skynet_slot / ".poncebot_managed_worktree"
            self.assertFalse(sentinel.exists())

            workspaces.ensure_worktree_pool(base_repo=repo, root=pool, role="skynet", slots=1)
            self.assertTrue(workspaces._managed_metadata_path(skynet_slot).exists())
            self.assertFalse(sentinel.exists())

            workspaces.prepare_clean_workspace(skynet_slot)
            self.assertEqual(run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=skynet_slot), expected_branch)

    def test_operational_gate_blocks_running_root_even_when_merge_ready(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            q = self._operational_gate_queue(
                root_state="running",
                root_trace={"merge_ready": True, "proactive_improvement_closed": True},
            )

            ok, reason, payload = bot._order_operational_maturity_gate(
                orch_q=q,
                order_id="root",
                chat_id=1,
                repo_dir=Path(td),
                default_branch="main",
                merge_required=True,
                now=123.0,
            )

        self.assertFalse(ok)
        self.assertEqual(reason, "root_job_still_running")
        self.assertEqual(payload["operational_gate_root_state"], "running")

    def test_operational_gate_blocks_failed_root_with_pass_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            q = self._operational_gate_queue(
                root_state="failed",
                root_trace={
                    "merge_ready": True,
                    "result_summary": "PASS: verified improvement.",
                },
            )

            ok, reason, _payload = bot._order_operational_maturity_gate(
                orch_q=q,
                order_id="root",
                chat_id=1,
                repo_dir=Path(td),
                default_branch="main",
                merge_required=True,
                now=123.0,
            )

        self.assertFalse(ok)
        self.assertEqual(reason, "root_job_terminal_failed_without_delivery")

    def test_operational_gate_allows_blocked_root_after_recovered_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            self._init_git_repo(repo, clean_commit=True)
            q = self._operational_gate_queue(
                root_state="blocked",
                root_trace={"merge_ready": True, "proactive_improvement_closed": True},
            )

            ok, reason, payload = bot._order_operational_maturity_gate(
                orch_q=q,
                order_id="root",
                chat_id=1,
                repo_dir=repo,
                default_branch="main",
                merge_required=True,
                now=123.0,
            )

        self.assertTrue(ok)
        self.assertEqual(reason, "passed")
        self.assertEqual(payload["operational_gate_root_state"], "blocked")

    def test_operational_gate_blocks_dirty_repo_before_merge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            self._init_git_repo(repo, clean_commit=False)
            (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
            q = self._operational_gate_queue(
                root_state="done",
                root_trace={"merge_ready": True, "proactive_improvement_closed": True},
            )

            ok, reason, payload = bot._order_operational_maturity_gate(
                orch_q=q,
                order_id="root",
                chat_id=1,
                repo_dir=repo,
                default_branch="main",
                merge_required=True,
                now=123.0,
            )

        self.assertFalse(ok)
        self.assertEqual(reason, "repo_dirty_before_merge")
        self.assertTrue(any("dirty.txt" in line for line in payload["operational_gate_dirty_paths"]))

    def test_operational_gate_passes_closed_proactive_clean_repo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            self._init_git_repo(repo, clean_commit=True)
            q = self._operational_gate_queue(
                root_state="done",
                root_trace={"merge_ready": True, "proactive_improvement_closed": True},
            )

            ok, reason, payload = bot._order_operational_maturity_gate(
                orch_q=q,
                order_id="root",
                chat_id=1,
                repo_dir=repo,
                default_branch="main",
                merge_required=True,
                now=123.0,
            )

        self.assertTrue(ok)
        self.assertEqual(reason, "passed")
        self.assertEqual(payload["operational_gate_root_state"], "done")

    def test_operational_maturity_sweep_corrects_late_ready_writer(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            role="skynet",
            state="blocked",
            trace={"merge_ready": True, "order_branch": "feature/order-root"},
            labels={},
            parent_job_id="",
            chat_id=1,

            created_at=90.0,
            updated_at=100.0,
        )

        class FakeQueue:
            def __init__(self) -> None:
                self.phases: list[str] = []
                self.statuses: list[str] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def list_orders_global(self, status: str, limit: int = 240):
                return [
                    {
                        "order_id": "root",
                        "chat_id": 1,
                        "status": "active",
                        "phase": "ready_for_merge",
                        "title": "Proactive Sprint: codexbot Reliability + Delivery",
                        "body": "AUTONOMOUS PROACTIVE SPRINT",
                    }
                ]

            def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                return self.list_orders_global(status="active")[0]

            def get_job(self, job_id: str):
                return root if job_id == "root" else None

            def jobs_by_parent(self, parent_job_id: str, limit: int = 600):
                return []

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def set_order_status(self, order_id: str, chat_id: int, status: str) -> None:
                self.statuses.append(status)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                root.trace.update(kwargs)
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        q = FakeQueue()
        with patch.object(bot, "_repo_context_for_order", return_value=({}, Path("."), "main")), patch.object(
            bot, "_order_trace_requires_merge", return_value=(True, "feature/order-root")
        ):
            corrected = bot._operational_maturity_sweep_tick(cfg=None, orch_q=q, now=123.0)

        self.assertEqual(corrected, 1)
        self.assertEqual(q.statuses[-1], "active")
        self.assertEqual(q.phases[-1], "review")
        self.assertFalse(root.trace["merge_ready"])
        self.assertEqual(root.trace["operational_gate_reason"], "root_job_still_blocked")
        self.assertEqual(q.audit_events[-1]["details"]["source"], "maturity_sweep")

    def test_terminal_root_failure_closes_proactive_order_as_failed(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            role="skynet",
            state="failed",
            trace={"workspace_error": "checkout failed"},
            labels={},
            parent_job_id="",
            chat_id=1,
            created_at=90.0,
            updated_at=100.0,
        )

        class FakeQueue:
            def __init__(self) -> None:
                self.statuses: list[str] = []
                self.phases: list[str] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def set_order_status(self, order_id: str, chat_id: int, status: str) -> None:
                self.statuses.append(status)

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                root.trace.update(kwargs)
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        q = FakeQueue()
        closed = bot._close_proactive_terminal_root_failure(
            orch_q=q,
            order_row={
                "order_id": "root",
                "status": "active",
                "phase": "review",
                "title": "Proactive Sprint: codexbot Reliability + Delivery",
                "body": "AUTONOMOUS PROACTIVE SPRINT",
            },
            root_job=root,
            children=[],
            chat_id=1,
            now=123.0,
        )

        self.assertTrue(closed)
        self.assertEqual(q.statuses[-1], "failed")
        self.assertEqual(q.phases[-1], "failed")
        self.assertTrue(root.trace["proactive_terminal_failure_closed"])
        self.assertEqual(root.trace["operational_gate_status"], "failed")
        self.assertEqual(q.audit_events[-1]["event_type"], "order.proactive_terminal_failure_closed")

    def test_autopilot_waits_silently_while_root_job_runs(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            role="skynet",
            state="running",
            trace={},
            labels={},
            parent_job_id="",
            chat_id=1,
            created_at=90.0,
            updated_at=100.0,
        )

        class FakeQueue:
            def __init__(self) -> None:
                self.statuses: list[str] = []
                self.phases: list[str] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def jobs_by_parent(self, parent_job_id: str, limit: int = 200):
                return []

            def get_job(self, job_id: str):
                return root

            def set_order_status(self, order_id: str, chat_id: int, status: str) -> None:
                self.statuses.append(status)

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                root.trace.update(kwargs)
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        q = FakeQueue()
        with patch.object(bot, "_order_has_pending_factory_ceo_strategy_proposal", return_value=False):
            enqueued = bot._enqueue_order_autopilot_task(
                cfg=None,
                orch_q=q,
                profiles=None,
                order_row={
                    "order_id": "root",
                    "status": "active",
                    "phase": "review",
                    "title": "Proactive Sprint: codexbot Reliability + Delivery",
                    "body": "AUTONOMOUS PROACTIVE SPRINT",
                },
                chat_id=1,
                now=123.0,
                reason="tick",
            )

        self.assertFalse(enqueued)
        self.assertEqual(q.statuses[-1], "active")
        self.assertEqual(q.phases[-1], "executing")
        self.assertEqual(root.trace["operational_gate_status"], "waiting")
        self.assertEqual(q.audit_events, [])

    def test_autopilot_seeds_qa_rework_before_blocked_root_wait(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            role="skynet",
            state="blocked",
            trace={},
            labels={},
            parent_job_id="",
            chat_id=1,
            created_at=90.0,
            updated_at=100.0,
        )
        children = [
            SimpleNamespace(
                job_id="impl",
                role="backend",
                state="done",
                labels={"key": "local_impl_recover_casefix"},
                trace={
                    "result_summary": (
                        "Applied the authoritative controller patch and captured validation evidence. "
                        "Validation passed with git diff --stat and the resulting patch artifact."
                    ),
                    "result_artifacts": ["/tmp/impl-artifacts/changes.patch"],
                },
                artifacts_dir="/tmp/impl-artifacts",
                created_at=100.0,
                updated_at=110.0,
            ),
            SimpleNamespace(
                job_id="qa",
                role="qa",
                state="done",
                labels={"key": "local_review_recover_casefix"},
                trace={"result_summary": "FAIL: NEEDS_REWORK. Remove unrelated guard file and wire the UI."},
                artifacts_dir="/tmp/qa-artifacts",
                created_at=111.0,
                updated_at=120.0,
            ),
        ]

        class FakeQueue:
            def __init__(self) -> None:
                self.submitted: list[bot.Task] = []
                self.phases: list[str] = []
                self.trace_updates: list[dict[str, object]] = []
                self.audit_events: list[dict[str, object]] = []

            def jobs_by_parent(self, parent_job_id: str, limit: int = 200):
                return list(children)

            def get_job(self, job_id: str):
                return root

            def submit_task(self, task: bot.Task) -> None:
                self.submitted.append(task)

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.phases.append(phase)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                root.trace.update(kwargs)
                self.trace_updates.append(dict(kwargs))

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        with tempfile.TemporaryDirectory() as td:
            q = FakeQueue()
            with patch.object(bot, "_order_has_pending_factory_ceo_strategy_proposal", return_value=False):
                enqueued = bot._enqueue_order_autopilot_task(
                    cfg=self._operational_gate_recovery_cfg(td),
                    orch_q=q,
                    profiles=None,
                    order_row={
                        "order_id": "root",
                        "status": "active",
                        "phase": "review",
                        "title": "Proactive Sprint: ExecutiveDashboard Reliability",
                        "body": "AUTONOMOUS PROACTIVE SPRINT",
                    },
                    chat_id=1,
                    now=5000.0,
                    reason="tick",
                )

        self.assertTrue(enqueued)
        self.assertEqual([task.role for task in q.submitted], ["backend", "qa"])
        self.assertEqual(q.phases[-1], "executing")
        self.assertIn("local_rework_seeded_job_id", q.trace_updates[-1])
        self.assertNotEqual(root.trace.get("operational_gate_status"), "waiting")

    def test_deploy_verify_retries_transient_startup_failure(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: object):
            calls.append(command)
            if len(calls) < 3:
                return SimpleNamespace(returncode=1, stdout="", stderr="service booting")
            return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

        with patch.object(bot.subprocess, "run", side_effect=fake_run), patch.object(bot.time, "sleep") as sleep:
            result = bot._run_deploy_verify_with_retries(
                verify_command=["curl", "-fsS", "http://127.0.0.1:8000/healthz"],
                cwd=".",
                env={},
                timeout_seconds=60,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleep.call_count, 2)

    def test_auto_merge_does_not_count_deploy_failure_as_success(self) -> None:
        root = SimpleNamespace(
            job_id="root",
            state="done",
            trace={"merge_ready": True, "proactive_improvement_closed": True, "order_branch": "feature/root"},
            labels={},
            created_at=90.0,
            updated_at=100.0,
        )

        class FakeQueue:
            def __init__(self) -> None:
                self.order = {
                    "order_id": "root",
                    "chat_id": 1,
                    "status": "active",
                    "phase": "ready_for_merge",
                    "title": "Proactive Sprint: ExecutiveDashboard Reliability",
                    "body": "AUTONOMOUS PROACTIVE SPRINT",
                }
                self.statuses: list[str] = []
                self.phases: list[str] = []
                self.audit_events: list[dict[str, object]] = []

            def is_paused_globally(self) -> bool:
                return False

            def list_orders_global(self, status: str, limit: int = 240):
                return [dict(self.order)] if status == "active" else []

            def get_order(self, order_id: str, chat_id: int = 0) -> dict[str, object]:
                return dict(self.order)

            def get_job(self, job_id: str):
                return root if job_id == "root" else None

            def jobs_by_parent(self, parent_job_id: str, limit: int = 600):
                return []

            def set_order_status(self, order_id: str, chat_id: int, status: str) -> None:
                self.order["status"] = status
                self.statuses.append(status)

            def set_order_phase(self, order_id: str, chat_id: int, phase: str) -> None:
                self.order["phase"] = phase
                self.phases.append(phase)

            def update_trace(self, job_id: str, **kwargs: object) -> None:
                root.trace.update(kwargs)

            def append_audit_event(self, *, event_type: str, actor: str, details: dict[str, object]) -> None:
                self.audit_events.append({"event_type": event_type, "actor": actor, "details": details})

        q = FakeQueue()

        def fake_merge(*_args: object, **_kwargs: object) -> str:
            root.trace.update({"merged_to_main": True, "deploy_status": "failed"})
            q.set_order_status("root", chat_id=1, status="active")
            q.set_order_phase("root", chat_id=1, phase="review")
            return "Order root auto-merged by Jarvis to codex/r530-main-clean-20260305-045022. Deploy failed: health check failed."

        with patch.object(bot, "_jarvis_auto_approve_merge_enabled", return_value=True), patch.object(
            bot, "_sync_order_phase_from_runtime", return_value=None
        ), patch.object(bot, "_repo_context_for_order", return_value=({}, Path("."), "codex/r530-main-clean-20260305-045022")), patch.object(
            bot, "_order_trace_requires_merge", return_value=(True, "feature/root")
        ), patch.object(bot, "_order_operational_maturity_gate", return_value=(True, "passed", {})), patch.object(
            bot, "_order_command_text", side_effect=fake_merge
        ), patch.object(bot, "_send_chunked_text", return_value=None):
            merged = bot._auto_merge_ready_orders_tick(cfg=None, api=None, orch_q=q, now=123.0)

        self.assertEqual(merged, 0)
        self.assertEqual(q.statuses[-1], "active")
        self.assertEqual(q.phases[-1], "review")
        self.assertIn("Deploy failed", root.trace["merge_auto_error"])
        self.assertEqual(q.audit_events[-1]["event_type"], "order.auto_merge_failed")


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


class TestFocusReceiptHelpers(unittest.TestCase):
    def test_parse_job_routes_focus_receipt_command_to_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = TestStateHandling()._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=7,
                user_id=2,
                message_id=10,
                username="u",
                text="/focus ack all 3 Need owner acknowledgement",
            )

            resp, job = bot._parse_job(cfg, msg)

        self.assertIsNone(job)
        self.assertEqual(
            resp,
            bot._focus_payload_marker(
                {
                    "mode": "receipt",
                    "scope": "all",
                    "rank": 3,
                    "state": "acknowledged",
                    "summary": "need owner acknowledgement",
                }
            ),
        )

    def test_parse_focus_command_tail_supports_receipt_commands(self) -> None:
        marker, job = bot._parse_focus_command_tail("ack all 3 Need owner acknowledgement")

        self.assertIsNone(job)
        self.assertEqual(
            marker,
            bot._focus_payload_marker(
                {
                    "mode": "receipt",
                    "scope": "all",
                    "rank": 3,
                    "state": "acknowledged",
                    "summary": "Need owner acknowledgement",
                }
            ),
        )

    def test_parse_focus_command_tail_supports_briefing_aliases(self) -> None:
        marker, job = bot._parse_focus_command_tail("brief chat 2")

        self.assertIsNone(job)
        self.assertEqual(
            marker,
            bot._focus_payload_marker({"mode": "briefing", "scope": "chat", "rank": 2}),
        )

    def test_operator_focus_receipt_text_formats_selected_item(self) -> None:
        receipt_payload = {
            "selection": {"rank": 2},
            "item_identity": {
                "urgency": "high",
                "category": "release",
                "label": "Confirm release owner",
                "action_id": "focus:release-owner",
            },
            "receipt": {
                "state": "completed",
                "persisted": True,
                "persistence_reason": "decision_log_written",
                "summary": "Owner acknowledged the handoff.",
                "next_action": "Proceed with the release checklist.",
            },
        }

        rendered = bot._operator_focus_receipt_text(receipt_payload, scope_label="all", rank=1)

        self.assertIn("Jarvis: focus receipt (all rank=2)", rendered)
        self.assertIn("state=completed persisted=yes reason=decision_log_written", rendered)
        self.assertIn("item: high/release - Confirm release owner (focus:release-owner)", rendered)
        self.assertIn("summary: Owner acknowledged the handoff.", rendered)
        self.assertIn("next: Proceed with the release checklist.", rendered)

    def test_send_orchestrator_marker_response_focus_receipt_uses_global_scope(self) -> None:
        api = _FakeAPI()
        cfg = TestStateHandling()._cfg(Path(tempfile.mkdtemp()) / "state.json")
        orch_q = object()
        receipt_payload = {
            "selection": {"rank": 3},
            "item_identity": {
                "urgency": "high",
                "category": "release",
                "label": "Confirm release owner",
                "action_id": "focus:release-owner",
            },
            "receipt": {
                "state": "acknowledged",
                "persisted": True,
                "persistence_reason": "decision_log_appended",
                "summary": "Owner acknowledged the handoff.",
                "next_action": "Proceed with the release checklist.",
            },
        }

        with patch.object(bot, "StatusService") as status_service_cls:
            svc = status_service_cls.return_value
            svc.operator_focus_receipt.return_value = receipt_payload

            handled = bot._send_orchestrator_marker_response(
                "focus",
                json.dumps(
                    {
                        "mode": "receipt",
                        "scope": "all",
                        "rank": 3,
                        "state": "acknowledged",
                        "summary": "Owner acknowledged the handoff.",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                cfg,
                api,
                chat_id=7,
                user_id=2,
                reply_to_message_id=10,
                orch_q=orch_q,  # type: ignore[arg-type]
                profiles=None,
            )

        self.assertTrue(handled)
        svc.operator_focus_receipt.assert_called_once_with(
            chat_id=None,
            rank=3,
            state="acknowledged",
            summary="Owner acknowledged the handoff.",
            actor="jarvis",
        )
        self.assertTrue(api.messages)
        self.assertIn("Jarvis: focus receipt (global rank=3)", api.messages[-1])


class TestPlanCommandHelpers(unittest.TestCase):
    def test_parse_job_routes_action_plan_alias_to_plan_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = TestStateHandling()._cfg(Path(td) / "state.json")
            msg = bot.IncomingMessage(
                update_id=1,
                chat_id=7,
                user_id=2,
                message_id=10,
                username="u",
                text="/action_plan all 7",
            )

            resp, job = bot._parse_job(cfg, msg)

        self.assertIsNone(job)
        self.assertEqual(resp, bot._plan_payload_marker({"scope": "all", "limit": 7}))

    def test_send_orchestrator_marker_response_handles_plan_marker(self) -> None:
        api = _FakeAPI()
        cfg = TestStateHandling()._cfg(Path(tempfile.mkdtemp()) / "state.json")
        orch_q = object()

        with patch.object(bot, "_send_chunked_text") as send_mock, patch.object(
            bot, "StatusService"
        ) as status_service_cls:
            svc = status_service_cls.return_value
            svc.proactive_action_plan.return_value = {
                "generated_at": "2026-05-11T00:00:00Z",
                "window": {"hours": 12},
                "lanes": [],
                "top_actions": [],
            }

            handled = bot._send_orchestrator_marker_response(
                "plan",
                json.dumps({"scope": "chat", "limit": 5}, sort_keys=True, separators=(",", ":")),
                cfg,
                api,
                chat_id=7,
                user_id=2,
                reply_to_message_id=10,
                orch_q=orch_q,  # type: ignore[arg-type]
                profiles=None,
            )

        self.assertTrue(handled)
        svc.proactive_action_plan.assert_called_once_with(chat_id=7, limit=5)
        send_mock.assert_called_once()
