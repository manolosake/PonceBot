from __future__ import annotations

import logging
import threading
import time

from .queue import OrchestratorQueue

LOG = logging.getLogger("codexbot")


class OrchestratorScheduler:
    """Very small periodic scheduler for role-specific internal jobs."""

    def __init__(self, *, interval_seconds: int = 3600, enabled: bool = False) -> None:
        self._enabled = enabled
        self._interval = max(60, int(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._callbacks: list[callable[[], None]] = []
        self._final_sweep_queue: OrchestratorQueue | None = None
        self._final_sweep_max_age_s = 24 * 60 * 60

    def add_tick(self, fn: callable[[], None]) -> None:
        self._callbacks.append(fn)

    def enable_final_sweep(self, queue: OrchestratorQueue, *, max_age_s: float = 24 * 60 * 60) -> None:
        self._final_sweep_queue = queue
        self._final_sweep_max_age_s = float(max_age_s)

    def start(self, *, name: str = "orchestrator-scheduler") -> None:
        if not self._enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            now_ts = time.time()
            if self._final_sweep_queue is not None:
                try:
                    canceled = self._final_sweep_queue.cancel_stale_blocked_only(
                        now_ts=now_ts,
                        max_age_s=self._final_sweep_max_age_s,
                    )
                    if canceled:
                        LOG.info("scheduler final sweep canceled=%s", canceled)
                except Exception:
                    LOG.exception("scheduler final sweep failed")
            for cb in list(self._callbacks):
                try:
                    cb()
                except Exception:
                    pass
            self._stop.wait(self._interval)
