from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import orchestrator.workspaces as workspaces
from orchestrator.workspaces import ensure_worktree_pool, prepare_clean_workspace


_SENTINEL = ".poncebot_managed_worktree"


def _metadata_path(worktree: Path) -> Path:
    return workspaces._managed_metadata_path(worktree)


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)


class TestWorkspaces(unittest.TestCase):
    def test_ensure_worktree_pool_uses_unique_branches_per_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base"
            pool_a = Path(td) / "pool-a"
            pool_b = Path(td) / "pool-b"
            base.mkdir(parents=True, exist_ok=True)

            _run(["git", "init", "-b", "main"], cwd=base)
            _run(["git", "config", "user.email", "test@example.com"], cwd=base)
            _run(["git", "config", "user.name", "test"], cwd=base)

            (base / ".gitignore").write_text(_SENTINEL + "\n", encoding="utf-8")
            (base / "README.md").write_text("hi\n", encoding="utf-8")
            _run(["git", "add", ".gitignore", "README.md"], cwd=base)
            _run(["git", "commit", "-m", "init"], cwd=base)

            ensure_worktree_pool(base_repo=base, root=pool_a, role="skynet", slots=1)
            ensure_worktree_pool(base_repo=base, root=pool_b, role="skynet", slots=1)

            wt_a = pool_a / "skynet" / "slot1"
            wt_b = pool_b / "skynet" / "slot1"
            self.assertTrue(_metadata_path(wt_a).exists())
            self.assertTrue(_metadata_path(wt_b).exists())
            self.assertFalse((wt_a / _SENTINEL).exists())
            self.assertFalse((wt_b / _SENTINEL).exists())

            br_a = subprocess.run(
                ["git", "-C", str(wt_a), "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            br_b = subprocess.run(
                ["git", "-C", str(wt_b), "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertNotEqual(br_a, br_b)

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
            sentinel = _metadata_path(wt)
            self.assertTrue(sentinel.exists())

            prepare_clean_workspace(wt)
            self.assertTrue(sentinel.exists())

            (wt / "tmpfile.txt").write_text("x\n", encoding="utf-8")
            prepare_clean_workspace(wt)
            self.assertTrue(sentinel.exists())

    def test_prepare_clean_workspace_restores_managed_branch(self) -> None:
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
            sentinel = _metadata_path(wt)
            sentinel_data = json.loads(sentinel.read_text(encoding="utf-8"))
            managed_branch = str(sentinel_data["branch"])

            subprocess.run(
                ["git", "-C", str(wt), "checkout", "-B", "feature/tmp-work"],
                check=True,
                capture_output=True,
                text=True,
            )

            prepare_clean_workspace(wt)

            branch = subprocess.run(
                ["git", "-C", str(wt), "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(branch, managed_branch)
            self.assertTrue(sentinel.exists())

    def test_prepare_clean_workspace_clears_dirty_tracked_changes_before_branch_restore(self) -> None:
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

            ensure_worktree_pool(base_repo=base, root=pool, role="skynet", slots=1)
            wt = pool / "skynet" / "slot1"
            sentinel = _metadata_path(wt)
            sentinel_data = json.loads(sentinel.read_text(encoding="utf-8"))
            managed_branch = str(sentinel_data["branch"])

            (wt / "README.md").write_text("dirty\n", encoding="utf-8")

            prepare_clean_workspace(wt)

            branch = subprocess.run(
                ["git", "-C", str(wt), "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(wt), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            self.assertEqual(branch, managed_branch)
            self.assertEqual((wt / "README.md").read_text(encoding="utf-8"), "hi\n")
            self.assertEqual(status, "")
            self.assertTrue(sentinel.exists())

    def test_ensure_worktree_pool_serializes_concurrent_creates(self) -> None:
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

            real_run = workspaces._run
            errors: list[Exception] = []

            def wrapped_run(cmd: list[str], *, check: bool):
                if cmd[:4] == ["git", "-C", str(base), "worktree"] and len(cmd) > 4 and cmd[4] == "add":
                    time.sleep(0.15)
                return real_run(cmd, check=check)

            def worker() -> None:
                try:
                    ensure_worktree_pool(base_repo=base, root=pool, role="backend", slots=1)
                except Exception as exc:  # pragma: no cover - assertion below checks this path
                    errors.append(exc)

            with mock.patch.object(workspaces, "_run", side_effect=wrapped_run):
                t1 = threading.Thread(target=worker)
                t2 = threading.Thread(target=worker)
                t1.start()
                t2.start()
                t1.join()
                t2.join()

            self.assertEqual(errors, [])
            self.assertTrue(_metadata_path(pool / "backend" / "slot1").exists())

    def test_prepare_clean_workspace_survives_tracked_legacy_marker_on_base(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base"
            pool = Path(td) / "pool"
            base.mkdir(parents=True, exist_ok=True)

            _run(["git", "init", "-b", "main"], cwd=base)
            _run(["git", "config", "user.email", "test@example.com"], cwd=base)
            _run(["git", "config", "user.name", "test"], cwd=base)

            (base / "README.md").write_text("hi\n", encoding="utf-8")
            _run(["git", "add", "README.md"], cwd=base)
            _run(["git", "commit", "-m", "init"], cwd=base)

            ensure_worktree_pool(base_repo=base, root=pool, role="skynet", slots=1)
            wt = pool / "skynet" / "slot1"
            (wt / _SENTINEL).write_text("legacy local marker\n", encoding="utf-8")

            (base / _SENTINEL).write_text("accidentally tracked marker\n", encoding="utf-8")
            _run(["git", "add", "-f", _SENTINEL], cwd=base)
            _run(["git", "commit", "-m", "accidentally track marker"], cwd=base)

            prepare_clean_workspace(wt)

            status = subprocess.run(
                ["git", "-C", str(wt), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            head = subprocess.run(
                ["git", "-C", str(wt), "rev-parse", "--short", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            base_head = subprocess.run(
                ["git", "-C", str(base), "rev-parse", "--short", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            self.assertEqual(status, "")
            self.assertEqual(head, base_head)
            self.assertTrue(_metadata_path(wt).exists())
            self.assertTrue((wt / _SENTINEL).exists())
