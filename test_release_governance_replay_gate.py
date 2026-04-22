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
            if cmd[:3] == ["python3", "-m", "venv"]:
                return 0, "", ""
            if cmd[1:5] == ["-m", "pip", "install", "--upgrade"]:
                return 0, "installed", ""
            if cmd[1:] == ["-m", "pytest", "--version"]:
                return 0, "pytest 9.0.3", ""
            if "test_state_store.py" in cmd:
                return 0, "8 tests collected", ""
            if cmd[1:] == ["-m", "pytest", "--collect-only", "-q"]:
                return 0, "100 tests collected", ""
            raise AssertionError(f"unexpected command: {cmd}")

        with TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
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

            # c02-c04 should use clean-shell env.
            clean_env = rg._clean_shell_env()
            c02_env = calls[2][1]
            c03_env = calls[3][1]
            c04_env = calls[4][1]
            self.assertEqual(c02_env, clean_env)
            self.assertEqual(c03_env, clean_env)
            self.assertEqual(c04_env, clean_env)

    def test_run_replay_gate_short_circuits_on_bootstrap_failure(self) -> None:
        def fake_try_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str, str]:
            if cmd[:3] == ["python3", "-m", "venv"]:
                return 1, "", "venv failed"
            raise AssertionError(f"unexpected command after failure: {cmd}")

        with TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            with patch.object(rg, "_try_run", side_effect=fake_try_run):
                checks = rg._run_replay_gate(root, artifacts_dir=artifacts, python_bin="python3")
            self.assertEqual(len(checks), 1)
            self.assertEqual(checks[0].key, "replay_bootstrap")
            self.assertFalse(checks[0].ok)
            self.assertEqual(checks[0].details, "c01_failed")
            transcript = (artifacts / "command_transcript.txt").read_text(encoding="utf-8")
            self.assertIn("$ c01", transcript)
            self.assertNotIn("$ c02", transcript)


if __name__ == "__main__":
    unittest.main()
