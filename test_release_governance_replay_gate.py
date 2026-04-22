import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools import release_governance as rg


class TestReleaseGovernanceReplayGate(unittest.TestCase):
    def test_run_replay_gate_success_writes_exit_codes(self) -> None:
        calls: list[tuple[list[str], dict[str, str] | None]] = []

        def fake_try_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str, str]:
            calls.append((cmd, env))
            if cmd[-3:] == ["-m", "pytest", "--version"]:
                return 0, "pytest 9.0.3", ""
            if "test_state_store.py" in cmd:
                return 0, "8 tests collected", ""
            if cmd[-4:] == ["-m", "pytest", "--collect-only", "-q"]:
                return 0, "100 tests collected", ""
            raise AssertionError(f"unexpected command: {cmd}")

        with TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            scripts = root / "scripts"
            scripts.mkdir(parents=True, exist_ok=True)
            (scripts / "bootstrap_pytest_python3.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            with patch.object(rg, "_try_run", side_effect=fake_try_run):
                checks = rg._run_replay_gate(root, artifacts_dir=artifacts, python_bin="python3")

            self.assertEqual([c.key for c in checks], ["replay_c02_pytest_version", "replay_c03_collect_targeted", "replay_c04_collect_all"])
            self.assertTrue(all(c.ok for c in checks))
            self.assertEqual((artifacts / "c02.exit_code.txt").read_text(encoding="utf-8").strip(), "0")
            self.assertEqual((artifacts / "c03.exit_code.txt").read_text(encoding="utf-8").strip(), "0")
            self.assertEqual((artifacts / "c04.exit_code.txt").read_text(encoding="utf-8").strip(), "0")
            transcript = (artifacts / "command_transcript.txt").read_text(encoding="utf-8")
            self.assertIn("$ c02", transcript)
            self.assertIn("$ c03", transcript)
            self.assertIn("$ c04", transcript)
            self.assertIn("bootstrap_pytest_python3.sh", transcript)

            # c02-c04 should use clean-shell env.
            clean_env = rg._clean_shell_env()
            c02_env = calls[0][1] or {}
            c03_env = calls[1][1] or {}
            c04_env = calls[2][1] or {}
            for e in (c02_env, c03_env, c04_env):
                self.assertEqual(e.get("PATH"), clean_env.get("PATH"))
                self.assertEqual(e.get("HOME"), clean_env.get("HOME"))
                self.assertEqual(e.get("BOOTSTRAP_PYTEST_PYTHON_BIN"), "python3")
                self.assertIn("/.qa_replay_venv", str(e.get("BOOTSTRAP_PYTEST_VENV_DIR", "")))

    def test_run_replay_gate_records_failure_without_short_circuit(self) -> None:
        def fake_try_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str, str]:
            if cmd[-3:] == ["-m", "pytest", "--version"]:
                return 1, "", "pytest missing"
            if "test_state_store.py" in cmd:
                return 0, "8 tests collected", ""
            if cmd[-4:] == ["-m", "pytest", "--collect-only", "-q"]:
                return 0, "100 tests collected", ""
            raise AssertionError(f"unexpected command: {cmd}")

        with TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            scripts = root / "scripts"
            scripts.mkdir(parents=True, exist_ok=True)
            (scripts / "bootstrap_pytest_python3.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            with patch.object(rg, "_try_run", side_effect=fake_try_run):
                checks = rg._run_replay_gate(root, artifacts_dir=artifacts, python_bin="python3")
            self.assertEqual(len(checks), 3)
            self.assertEqual(checks[0].key, "replay_c02_pytest_version")
            self.assertFalse(checks[0].ok)
            self.assertIn("pytest missing", checks[0].details)
            transcript = (artifacts / "command_transcript.txt").read_text(encoding="utf-8")
            self.assertIn("$ c02", transcript)
            self.assertIn("$ c03", transcript)
            self.assertIn("$ c04", transcript)


if __name__ == "__main__":
    unittest.main()
