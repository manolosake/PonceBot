import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from state_store import StateStore
from tools import coverage_gate as cg


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
    def test_path_property_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            self.assertEqual(store.path, p.resolve())

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

    def test_read_falls_back_to_empty_on_os_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            with mock.patch.object(Path, "read_text", side_effect=OSError("io failure")):
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

    def test_replace_normalizes_list_nested_mixed_key_types(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            store.replace({"items": [{1: "one", "inner": [{2: "two"}]}]})
            data = store.read()
            self.assertEqual(data.get("items", [])[0].get("1"), "one")
            self.assertEqual(data.get("items", [])[0].get("inner", [])[0].get("2"), "two")

    def test_replace_failure_cleans_orphan_tmp_and_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            err = OSError("replace failed")

            with mock.patch.object(Path, "replace", side_effect=err):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    store.replace({"k": "v"})

            leftovers = list(Path(td).glob("state.json.*.tmp"))
            self.assertEqual(leftovers, [])

    def test_replace_failure_keeps_original_error_when_unlink_cleanup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            err = OSError("replace failed")

            with mock.patch.object(Path, "replace", side_effect=err):
                with mock.patch.object(Path, "unlink", side_effect=OSError("unlink failed")):
                    with self.assertRaisesRegex(OSError, "replace failed"):
                        store.replace({"k": "v"})

    def test_update_mutator_error_does_not_corrupt_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            store = StateStore(p)
            store.replace({"stable": 1})

            def _m(_st):
                raise RuntimeError("mutator boom")

            with self.assertRaisesRegex(RuntimeError, "mutator boom"):
                store.update(_m)
            self.assertEqual(store.read(), {"stable": 1})


class TestCoverageGateDiscovery(unittest.TestCase):
    def test_discover_suite_uses_each_pattern(self) -> None:
        patterns = ["test_state_store.py", "test_release_governance_*.py"]
        with mock.patch.object(cg.unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()) as discover:
            suite = cg._discover_suite(patterns=patterns)
        self.assertIsInstance(suite, unittest.TestSuite)
        self.assertEqual(discover.call_count, 2)
        self.assertEqual(discover.call_args_list[0].kwargs["pattern"], "test_state_store.py")
        self.assertEqual(discover.call_args_list[1].kwargs["pattern"], "test_release_governance_*.py")


if __name__ == "__main__":
    unittest.main()
