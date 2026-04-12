from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from orchestrator.workspaces import ensure_worktree_pool, prepare_clean_workspace


_SENTINEL = ".poncebot_managed_worktree"


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)


class TestWorkspaces(unittest.TestCase):
    def test_prepare_clean_workspace_preserves_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base"
            pool = Path(td) / "pool"
            base.mkdir(parents=True, exist_ok=True)

            _run(["git", "init", "-b", "main"], cwd=base)
            _run(["git", "config", "user.email", "test@example.com"], cwd=base)
            _run(["git", "config", "user.name", "test"], cwd=base)

            (base / ".gitignore").write_text(_SENTINEL + "\n", encoding="utf-8")
            (base / "README.md").write_text("hi\n", encoding="utf-8")
            _run(["git", "add", ".gitignore", "README.md"], cwd=base)
            _run(["git", "commit", "-m", "init"], cwd=base)

            ensure_worktree_pool(base_repo=base, root=pool, role="backend", slots=1)
            wt = pool / "backend" / "slot1"
            sentinel = wt / _SENTINEL
            self.assertTrue(sentinel.exists())

            prepare_clean_workspace(wt)
            self.assertTrue(sentinel.exists())

            (wt / "tmpfile.txt").write_text("x\n", encoding="utf-8")
            prepare_clean_workspace(wt)
            self.assertTrue(sentinel.exists())
