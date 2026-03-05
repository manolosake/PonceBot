from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.check_call(cmd, cwd=str(cwd))


class TestBackendDoneEvidenceGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.guard_script = (Path(__file__).resolve().parent / "tools" / "backend_done_evidence_guard.py").resolve()

    def _seed_repo_with_remote(self, base: Path, branch: str) -> tuple[Path, Path]:
        remote = base / "remote.git"
        local = base / "local"
        _run(["git", "init", "--bare", str(remote)], cwd=base)
        _run(["git", "init", str(local)], cwd=base)
        _run(["git", "config", "user.email", "test@example.com"], cwd=local)
        _run(["git", "config", "user.name", "Test User"], cwd=local)
        (local / "README.md").write_text("seed\n", encoding="utf-8")
        _run(["git", "add", "README.md"], cwd=local)
        _run(["git", "commit", "-m", "seed"], cwd=local)
        _run(["git", "branch", "-M", branch], cwd=local)
        _run(["git", "remote", "add", "origin", str(remote)], cwd=local)
        _run(["git", "push", "-u", "origin", branch], cwd=local)
        return local, remote

    def _run_guard(self, repo: Path, artifacts: Path, branch: str) -> tuple[int, dict]:
        proc = subprocess.run(
            [
                "python3",
                str(self.guard_script),
                "--repo-root",
                str(repo),
                "--artifacts-dir",
                str(artifacts),
                "--expected-branch",
                branch,
            ],
            cwd=Path(__file__).resolve().parent,
            text=True,
            capture_output=True,
            check=False,
        )
        report = json.loads((artifacts / "backend_done_evidence_guard_report.json").read_text(encoding="utf-8"))
        return int(proc.returncode), report

    def test_pass_when_remote_sha_present_push_proof_and_clean_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            branch = "feature/order-test"
            repo, _ = self._seed_repo_with_remote(base, branch)
            artifacts = base / "artifacts"

            rc, report = self._run_guard(repo, artifacts, branch)
            self.assertEqual(rc, 0)
            self.assertEqual(report["status"], "PASS")
            checks = {c["key"]: bool(c["ok"]) for c in report.get("checks", [])}
            self.assertTrue(checks.get("remote_sha_present"))
            self.assertTrue(checks.get("push_proof_matches_remote_sha"))
            self.assertTrue(checks.get("git_status_clean"))

    def test_fail_when_remote_sha_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            branch = "feature/order-test"
            repo, _ = self._seed_repo_with_remote(base, branch)
            artifacts = base / "artifacts"

            rc, report = self._run_guard(repo, artifacts, "feature/missing")
            self.assertEqual(rc, 2)
            self.assertEqual(report["status"], "FAIL")
            checks = {c["key"]: bool(c["ok"]) for c in report.get("checks", [])}
            self.assertFalse(checks.get("remote_sha_present"))

    def test_fail_when_git_status_not_clean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            branch = "feature/order-test"
            repo, _ = self._seed_repo_with_remote(base, branch)
            artifacts = base / "artifacts"

            (repo / "README.md").write_text("dirty\n", encoding="utf-8")
            rc, report = self._run_guard(repo, artifacts, branch)
            self.assertEqual(rc, 2)
            self.assertEqual(report["status"], "FAIL")
            checks = {c["key"]: bool(c["ok"]) for c in report.get("checks", [])}
            self.assertFalse(checks.get("git_status_clean"))


if __name__ == "__main__":
    unittest.main()
