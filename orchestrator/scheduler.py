from __future__ import annotations

import threading
import time


class OrchestratorScheduler:
    """Very small periodic scheduler for role-specific internal jobs."""

    def __init__(self, *, interval_seconds: int = 3600, enabled: bool = False) -> None:
        self._enabled = enabled
        self._interval = max(60, int(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._callbacks: list[callable[[], None]] = []

    def add_tick(self, fn: callable[[], None]) -> None:
        self._callbacks.append(fn)

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
            for cb in list(self._callbacks):
                try:
                    cb()
                except Exception:
                    pass
            self._stop.wait(self._interval)
