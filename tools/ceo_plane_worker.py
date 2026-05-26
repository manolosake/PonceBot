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
import threading
import time

import bot


def _set_ceo_plane_defaults() -> None:
    os.environ.setdefault("BOT_CEO_PLANE_ONLY", "1")
    os.environ.setdefault("BOT_PROACTIVE_LANE_ENABLED", "0")
    os.environ.setdefault("BOT_PROACTIVE_ITERATION_ENABLED", "0")
    os.environ.setdefault("BOT_CEO_ORDER_AUTOPILOT_ENABLED", "0")
    os.environ.setdefault("BOT_ORCHESTRATOR_DAILY_DIGEST_SECONDS", "0")
    os.environ.setdefault("BOT_STATUS_HTTP_ENABLED", "0")
    os.environ.setdefault("BOT_NOTIFY_CHILD_WORKERS", "0")


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
