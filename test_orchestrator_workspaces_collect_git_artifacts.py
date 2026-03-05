import subprocess
import tempfile
import unittest
from pathlib import Path

from orchestrator.workspaces import collect_git_artifacts


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)


class TestCollectGitArtifacts(unittest.TestCase):
    def test_collect_includes_staged_unstaged_and_untracked(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            _run(["git", "init"], repo)
            _run(["git", "config", "user.name", "Test User"], repo)
            _run(["git", "config", "user.email", "test@example.com"], repo)

            tracked = repo / "tracked.txt"
            tracked.write_text("v1\n", encoding="utf-8")
            _run(["git", "add", "tracked.txt"], repo)
            _run(["git", "commit", "-m", "init"], repo)

            # staged change on tracked file
            tracked.write_text("v2\n", encoding="utf-8")
            _run(["git", "add", "tracked.txt"], repo)
            # unstaged change on the same tracked file
            tracked.write_text("v2\nunstaged\n", encoding="utf-8")
            # untracked file
            (repo / "new_untracked.py").write_text("print('x')\n", encoding="utf-8")

            artifacts = repo / "artifacts"
            collect_git_artifacts(repo_dir=repo, artifacts_dir=artifacts)

            status = (artifacts / "git_status.txt").read_text(encoding="utf-8", errors="replace")
            patch = (artifacts / "changes.patch").read_text(encoding="utf-8", errors="replace")

            self.assertIn("tracked.txt", status)
            self.assertIn("?? new_untracked.py", status)
            self.assertIn("diff --git a/tracked.txt b/tracked.txt", patch)
            self.assertIn("new_untracked.py", patch)
            self.assertGreater((artifacts / "git_status.txt").stat().st_size, 0)
            self.assertGreater((artifacts / "changes.patch").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
