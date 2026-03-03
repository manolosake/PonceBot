from __future__ import annotations

from contextlib import contextmanager
import json
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover - fcntl is not available on Windows.
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


Mutator = Callable[[dict[str, Any]], None]


class StateStore:
    """
    Process-safe JSON key/value document store.

    Guarantees:
    - Atomic replace on write.
    - Thread-safe updates in-process.
    - Best-effort cross-process lock on POSIX (fcntl).
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._lock_path = Path(str(self._path) + ".lock")
        self._mu = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def _file_lock(self) -> Any:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = self._lock_path.open("a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            raw = self._path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return {}
        except Exception:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_f = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=self._path.name + ".",
            suffix=".tmp",
            dir=str(self._path.parent),
            delete=False,
        )
        try:
            tmp_f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
            tmp_f.flush()
        finally:
            tmp_f.close()
        Path(tmp_f.name).replace(self._path)

    def read(self) -> dict[str, Any]:
        with self._mu:
            with self._file_lock():
                return self._read_unlocked()

    def replace(self, data: dict[str, Any]) -> dict[str, Any]:
        safe = data if isinstance(data, dict) else {}
        with self._mu:
            with self._file_lock():
                self._write_unlocked(safe)
        return dict(safe)

    def update(self, mutator: Mutator) -> dict[str, Any]:
        with self._mu:
            with self._file_lock():
                cur = self._read_unlocked()
                if not isinstance(cur, dict):
                    cur = {}
                mutator(cur)
                self._write_unlocked(cur)
                return dict(cur)

    def get(self, key: str, default: Any = None) -> Any:
        st = self.read()
        return st.get(key, default)

    def set(self, key: str, value: Any) -> dict[str, Any]:
        def _m(st: dict[str, Any]) -> None:
            st[str(key)] = value

        return self.update(_m)

    def delete(self, key: str) -> dict[str, Any]:
        def _m(st: dict[str, Any]) -> None:
            st.pop(str(key), None)

        return self.update(_m)

# binding-r5 unstaged
