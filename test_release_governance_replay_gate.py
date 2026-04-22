import unittest
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools import release_governance as rg


class TestReleaseGovernanceReplayGate(unittest.TestCase):
    def test_close_gate_blocker_count_contract_semantics(self) -> None:
        overhead_only = [
            {"role": "qa", "state": "running"},
            {"role": "qa_local", "state": "queued"},
            {"role": "quality_assurance", "state": "waiting_deps"},
            {"role": "Quality-Assurance", "state": "blocked"},
            {"role": "reviewer_local", "state": "queued"},
            {"role": "skynet", "state": "blocked"},
        ]
        self.assertEqual(rg._close_gate_blocker_count(overhead_only), 0)

        with_real_blocker = overhead_only + [{"role": "backend", "state": "running"}]
        self.assertGreater(rg._close_gate_blocker_count(with_real_blocker), 0)

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

    def test_run_replay_gate_appends_close_gate_check_when_ticket_present(self) -> None:
        calls: list[tuple[list[str], dict[str, str] | None]] = []

        def fake_try_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str, str]:
            calls.append((cmd, env))
            return 0, "ok", ""

        with TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            scripts = root / "scripts"
            scripts.mkdir(parents=True, exist_ok=True)
            (scripts / "bootstrap_pytest_python3.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            db_path = root / "jobs.sqlite"
            with sqlite3.connect(db_path) as con:
                con.execute("CREATE TABLE jobs(job_id TEXT, parent_job_id TEXT, role TEXT, state TEXT)")
                con.execute(
                    "INSERT INTO jobs(job_id,parent_job_id,role,state) VALUES(?,?,?,?)",
                    ("j1", "ord-1", "qa", "running"),
                )
                con.commit()
            with patch.object(rg, "_try_run", side_effect=fake_try_run):
                with patch.dict("os.environ", {"ORCH_JOBS_DB": str(db_path)}, clear=False):
                    checks = rg._run_replay_gate(root, artifacts_dir=artifacts, python_bin="python3", ticket_id="ord-1")

            keys = [c.key for c in checks]
            self.assertIn("replay_c05_close_gate_contract", keys)
            c05 = [c for c in checks if c.key == "replay_c05_close_gate_contract"][0]
            self.assertTrue(c05.ok)  # qa-only overhead should not block
            self.assertIn("blocking_active_children=0", c05.details)
            self.assertEqual((artifacts / "c05.exit_code.txt").read_text(encoding="utf-8").strip(), "0")
            transcript = (artifacts / "command_transcript.txt").read_text(encoding="utf-8")
            self.assertIn("$ c05", transcript)

    def test_run_replay_gate_excludes_current_job_from_close_gate_blockers(self) -> None:
        def fake_try_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str, str]:
            return 0, "ok", ""

        with TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            scripts = root / "scripts"
            scripts.mkdir(parents=True, exist_ok=True)
            (scripts / "bootstrap_pytest_python3.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            db_path = root / "jobs.sqlite"
            with sqlite3.connect(db_path) as con:
                con.execute("CREATE TABLE jobs(job_id TEXT, parent_job_id TEXT, role TEXT, state TEXT)")
                con.execute(
                    "INSERT INTO jobs(job_id,parent_job_id,role,state) VALUES(?,?,?,?)",
                    ("cur-job", "ord-1", "backend", "running"),
                )
                con.execute(
                    "INSERT INTO jobs(job_id,parent_job_id,role,state) VALUES(?,?,?,?)",
                    ("qa-wait", "ord-1", "qa", "waiting_deps"),
                )
                con.commit()
            with patch.object(rg, "_try_run", side_effect=fake_try_run):
                with patch.dict("os.environ", {"ORCH_JOBS_DB": str(db_path)}, clear=False):
                    checks = rg._run_replay_gate(
                        root,
                        artifacts_dir=artifacts,
                        python_bin="python3",
                        ticket_id="ord-1",
                        job_id="cur-job",
                    )

            c05 = [c for c in checks if c.key == "replay_c05_close_gate_contract"][0]
            self.assertTrue(c05.ok)
            self.assertIn("blocking_active_children=0", c05.details)

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
