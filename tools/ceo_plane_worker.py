#!/usr/bin/env python3
"""Isolated CEO on-demand worker for PonceBot.

This process does not poll Telegram. The primary gateway receives channel
messages and writes CEO jobs into this process' SQLite queue. This worker only
claims that isolated queue and replies through Telegram send APIs.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bot


def _ceo_plane_default_env(
    *,
    home: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = environ if environ is not None else os.environ
    home_path = Path(home if home is not None else env.get("HOME", "/home/aponce")).expanduser()
    base = home_path / "codexbot"
    data = base / "data"

    ceo_db = str(env.get("BOT_CEO_PLANE_DB_PATH") or (data / "ceo_jobs.sqlite"))
    ceo_state = str(env.get("BOT_CEO_PLANE_STATE_FILE") or (data / "ceo_state.json"))
    ceo_worktrees = str(env.get("BOT_CEO_PLANE_WORKTREE_ROOT") or (data / "ceo_worktrees"))
    ceo_artifacts = str(env.get("BOT_CEO_PLANE_ARTIFACTS_ROOT") or (data / "ceo_artifacts"))

    return {
        "BOT_CEO_PLANE_ONLY": "1",
        "BOT_PROACTIVE_LANE_ENABLED": "0",
        "BOT_PROACTIVE_ITERATION_ENABLED": "0",
        "BOT_CEO_ORDER_AUTOPILOT_ENABLED": "0",
        "BOT_ORCHESTRATOR_DAILY_DIGEST_SECONDS": "0",
        "BOT_STATUS_HTTP_ENABLED": "0",
        "BOT_NOTIFY_CHILD_WORKERS": "0",
        "BOT_CEO_PLANE_DB_PATH": ceo_db,
        "BOT_CEO_PLANE_STATE_FILE": ceo_state,
        "BOT_CEO_PLANE_WORKTREE_ROOT": ceo_worktrees,
        "BOT_CEO_PLANE_ARTIFACTS_ROOT": ceo_artifacts,
        "BOT_CEO_PLANE_SERVICE_NAME": str(env.get("BOT_CEO_PLANE_SERVICE_NAME") or "poncebot-ceo.service"),
        "BOT_ORCHESTRATOR_DB_PATH": ceo_db,
        "BOT_STATE_FILE": ceo_state,
        "BOT_WORKTREE_ROOT": ceo_worktrees,
        "BOT_ARTIFACTS_ROOT": ceo_artifacts,
    }


def _set_ceo_plane_defaults() -> None:
    os.environ.update(_ceo_plane_default_env())


def main() -> int:
    _set_ceo_plane_defaults()
    logging.basicConfig(
        level=os.environ.get("BOT_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("poncebot-ceo")

    cfg = bot._load_config()
    profiles = bot.load_agent_profiles(cfg.orchestrator_agent_profiles)
    try:
        profiles = bot._render_placeholders_obj(profiles, ceo_name=cfg.ceo_name)
    except Exception:
        pass

    storage = bot.SQLiteTaskStorage(cfg.orchestrator_db_path)
    orch_q = bot.OrchestratorQueue(storage=storage, role_profiles=profiles)
    recovered = orch_q.recover_stale_running()
    if recovered:
        log.info("Recovered %d stale CEO-plane jobs", recovered)

    api = bot.TelegramAPI(
        cfg.telegram_token,
        http_timeout_seconds=cfg.http_timeout_seconds,
        http_max_retries=cfg.http_max_retries,
        http_retry_initial_seconds=cfg.http_retry_initial_seconds,
        http_retry_max_seconds=cfg.http_retry_max_seconds,
        parse_mode=cfg.telegram_parse_mode,
    )

    stop_event = threading.Event()

    def _stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    worker_count = max(1, int(cfg.orchestrator_worker_count or 1))
    log.info(
        "CEO Command Plane started db=%s worktrees=%s artifacts=%s workers=%d",
        cfg.orchestrator_db_path,
        cfg.worktree_root,
        cfg.artifacts_root,
        worker_count,
    )
    for idx in range(worker_count):
        thread = threading.Thread(
            target=bot.orchestrator_worker_loop,
            kwargs={
                "cfg": cfg,
                "api": api,
                "orch_q": orch_q,
                "stop_event": stop_event,
                "profiles": profiles,
            },
            daemon=True,
            name=f"ceo-orch-worker-{idx + 1}",
        )
        thread.start()

    while not stop_event.is_set():
        time.sleep(1.0)

    log.info("CEO Command Plane stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
