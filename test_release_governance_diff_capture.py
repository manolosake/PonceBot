import unittest
from pathlib import Path
from unittest.mock import patch

from tools import release_governance as rg


class TestReleaseGovernanceDiffCapture(unittest.TestCase):
    def test_branch_diff_basis_when_branch_diverged(self) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], *, cwd: Path) -> str:
            calls.append(cmd)
            if cmd[:3] == ["git", "diff", "--stat"]:
                return " foo.py | 1 +"
            if cmd[:3] == ["git", "diff", "--name-only"]:
                return "foo.py"
            raise AssertionError(f"unexpected cmd: {cmd}")

        with patch.object(rg, "_run", side_effect=fake_run):
            basis, diffstat, changed = rg._collect_diff_capture(
                Path("."),
                base_ref="origin/main",
                head_ref="feature/x",
                ahead=1,
                behind=0,
                working_tree_dirty=True,
            )

        self.assertEqual(basis, "branch")
        self.assertEqual(diffstat, " foo.py | 1 +")
        self.assertEqual(changed, "foo.py")
        self.assertEqual(
            calls,
            [
                ["git", "diff", "--stat", "origin/main..feature/x"],
                ["git", "diff", "--name-only", "origin/main..feature/x"],
            ],
        )

    def test_working_tree_basis_when_no_branch_divergence(self) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], *, cwd: Path) -> str:
            calls.append(cmd)
            if cmd[:3] == ["git", "diff", "--stat"]:
                return " state_store.py | 2 +-"
            if cmd[:3] == ["git", "diff", "--name-only"]:
                return "state_store.py"
            raise AssertionError(f"unexpected cmd: {cmd}")

        with patch.object(rg, "_run", side_effect=fake_run):
            basis, diffstat, changed = rg._collect_diff_capture(
                Path("."),
                base_ref="origin/main",
                head_ref="feature/x",
                ahead=0,
                behind=0,
                working_tree_dirty=True,
            )

        self.assertEqual(basis, "working_tree")
        self.assertEqual(diffstat, " state_store.py | 2 +-")
        self.assertEqual(changed, "state_store.py")
        self.assertEqual(
            calls,
            [
                ["git", "diff", "--stat", "HEAD"],
                ["git", "diff", "--name-only", "HEAD"],
            ],
        )

    def test_none_basis_when_no_branch_or_working_tree_changes(self) -> None:
        with patch.object(rg, "_run") as mocked:
            basis, diffstat, changed = rg._collect_diff_capture(
                Path("."),
                base_ref="origin/main",
                head_ref="feature/x",
                ahead=0,
                behind=0,
                working_tree_dirty=False,
            )
        self.assertEqual(basis, "none")
        self.assertEqual(diffstat, "")
        self.assertEqual(changed, "")
        mocked.assert_not_called()

    def test_falls_back_when_branch_counts_diverged_but_branch_diff_empty(self) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], *, cwd: Path) -> str:
            calls.append(cmd)
            if cmd == ["git", "diff", "--stat", "origin/main..feature/x"]:
                return ""
            if cmd == ["git", "diff", "--name-only", "origin/main..feature/x"]:
                return ""
            if cmd == ["git", "diff", "--stat", "HEAD"]:
                return " file.txt | 1 +"
            if cmd == ["git", "diff", "--name-only", "HEAD"]:
                return "file.txt"
            raise AssertionError(f"unexpected cmd: {cmd}")

        with patch.object(rg, "_run", side_effect=fake_run):
            basis, diffstat, changed = rg._collect_diff_capture(
                Path("."),
                base_ref="origin/main",
                head_ref="feature/x",
                ahead=0,
                behind=2,
                working_tree_dirty=True,
            )

        self.assertEqual(basis, "working_tree")
        self.assertEqual(diffstat, " file.txt | 1 +")
        self.assertEqual(changed, "file.txt")
        self.assertEqual(
            calls,
            [
                ["git", "diff", "--stat", "origin/main..feature/x"],
                ["git", "diff", "--name-only", "origin/main..feature/x"],
                ["git", "diff", "--stat", "HEAD"],
                ["git", "diff", "--name-only", "HEAD"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
