import tempfile
import threading
import unittest
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

    def test_update_handles_mixed_key_types_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            store.replace({"alpha": 1})

            def _m(st):
                st[2] = "two"

            out = store.update(_m)
            self.assertEqual(out.get(2), "two")
            self.assertEqual(store.read().get("2"), "two")

    def test_replace_normalizes_nested_mixed_key_types(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            store.replace({"outer": {"a": 1, 2: "two"}})
            data = store.read()
            self.assertEqual(data.get("outer", {}).get("a"), 1)
            self.assertEqual(data.get("outer", {}).get("2"), "two")


if __name__ == "__main__":
    unittest.main()
