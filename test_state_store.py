import os
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

from state_store import StateStore


class TestStateStoreConcurrency(unittest.TestCase):
    def test_concurrent_updates_preserve_all_writes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = StateStore(Path(td) / "state.json")
            workers = 8
            iterations = 60

            def worker(idx: int) -> None:
                key = f"w{idx}"
                for _ in range(iterations):
                    def _m(st):
                        st[key] = int(st.get(key, 0) or 0) + 1
                        st["total"] = int(st.get("total", 0) or 0) + 1

                    store.update(_m)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=5)
            alive_threads = [th.name for th in threads if th.is_alive()]
            self.assertEqual(
                alive_threads,
                [],
                f"Worker threads did not finish within join timeout: {alive_threads}",
            )

            state = store.read()
            self.assertEqual(int(state.get("total") or 0), workers * iterations)
            for i in range(workers):
                self.assertEqual(int(state.get(f"w{i}") or 0), iterations)


class TestStateStoreBasics(unittest.TestCase):
    def test_set_get_delete_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            self.assertEqual(store.get("missing", "d"), "d")
            store.set("alpha", 1)
            self.assertEqual(store.get("alpha"), 1)
            store.delete("alpha")
            self.assertIsNone(store.get("alpha"))

    def test_replace_with_non_dict_input_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            store.replace({"a": 1})
            store.replace([])  # type: ignore[arg-type]
            self.assertEqual(store.read(), {})

    def test_read_handles_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text("{not-json", encoding="utf-8")
            store = StateStore(p)
            self.assertEqual(store.read(), {})

    def test_set_calls_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            temp_fd = {}
            real_named_tempfile = tempfile.NamedTemporaryFile

            def _capture_temp_fd(*args, **kwargs):
                f = real_named_tempfile(*args, **kwargs)
                temp_fd["fd"] = f.fileno()
                return f

            with mock.patch("state_store.tempfile.NamedTemporaryFile", side_effect=_capture_temp_fd):
                with mock.patch("state_store.os.fsync") as fsync_mock:
                    store.set("alpha", 1)
                    self.assertIn(mock.call(temp_fd["fd"]), fsync_mock.call_args_list)

    def test_update_recovers_from_non_dict_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text("[]", encoding="utf-8")
            store = StateStore(p)

            def _m(st):
                st["k"] = "v"

            out = store.update(_m)
            self.assertEqual(out.get("k"), "v")
            self.assertEqual(store.read().get("k"), "v")

    def test_fsync_dir_calls_open_fsync_close(self) -> None:
        if not hasattr(os, "O_DIRECTORY"):
            self.skipTest("O_DIRECTORY not supported on this platform")
        with tempfile.TemporaryDirectory() as td:
            store = StateStore(Path(td) / "state.json")
            with mock.patch("state_store.os.open", return_value=123) as open_mock:
                with mock.patch("state_store.os.fsync", side_effect=OSError("boom")) as fsync_mock:
                    with mock.patch("state_store.os.close") as close_mock:
                        store._fsync_dir(Path(td))
            open_mock.assert_called_once_with(
                td,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            fsync_mock.assert_called_once_with(123)
            close_mock.assert_called_once_with(123)

    def test_write_cleanup_on_dump_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)

            def _boom(*_args, **_kwargs):
                raise ValueError("boom")

            with mock.patch("state_store.json.dumps", side_effect=_boom):
                with self.assertRaises(ValueError):
                    store.set("alpha", 1)

            tmp_files = list(Path(td).glob("state.json.*.tmp"))
            self.assertEqual(tmp_files, [])

    def test_write_cleanup_on_replace_failure_preserves_existing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text("{\"alpha\": 1}\n", encoding="utf-8")
            store = StateStore(p)

            def _boom(*_args, **_kwargs):
                raise OSError("replace failed")

            with mock.patch("state_store.Path.replace", side_effect=_boom):
                with self.assertRaises(OSError):
                    store.set("beta", 2)

            self.assertEqual(p.read_text(encoding="utf-8"), "{\"alpha\": 1}\n")
            tmp_files = list(Path(td).glob("state.json.*.tmp"))
            self.assertEqual(tmp_files, [])


if __name__ == "__main__":
    unittest.main()
