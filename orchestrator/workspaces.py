from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import subprocess
import time


_SENTINEL = ".poncebot_managed_worktree"


@dataclass(frozen=True)
class WorktreeLease:
    role: str
    slot: int
    path: Path


def ensure_worktree_pool(*, base_repo: Path, root: Path, role: str, slots: int) -> None:
    base_repo = base_repo.resolve()
    root = root.resolve()
    role = (role or "").strip().lower()
    slots = max(1, int(slots))

    if not (base_repo / ".git").exists():
        raise RuntimeError(f"base_repo is not a git repo: {base_repo}")

    root.mkdir(parents=True, exist_ok=True)

    for slot in range(1, slots + 1):
        wt_dir = (root / role / f"slot{slot}").resolve()
        wt_dir.parent.mkdir(parents=True, exist_ok=True)
        sentinel = wt_dir / _SENTINEL
        if wt_dir.exists():
            if not sentinel.exists():
                # Back-compat: older deployments created worktrees without a sentinel. If the directory
                # appears to be one of our worktrees, mark it managed so future cleanups are safe.
                if _maybe_mark_managed_worktree(base_repo=base_repo, wt_dir=wt_dir, role=role, slot=slot, sentinel=sentinel):
                    continue
                raise RuntimeError(f"worktree exists but is not managed: {wt_dir}")
            continue

        branch = f"poncebot/{role}/slot{slot}"
        base_ref = _pick_base_ref(base_repo)
        _run(["git", "-C", str(base_repo), "fetch", "origin", "--prune"], check=False)
        _run(["git", "-C", str(base_repo), "worktree", "add", "-B", branch, str(wt_dir), base_ref], check=True)

        sentinel.write_text(
            json.dumps(
                {
                    "base_repo": str(base_repo),
                    "role": role,
                    "slot": slot,
                    "created_at": time.time(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            errors="replace",
        )


def _maybe_mark_managed_worktree(*, base_repo: Path, wt_dir: Path, role: str, slot: int, sentinel: Path) -> bool:
    """
    Best-effort: if a worktree directory already exists (but lacks our sentinel), only mark it
    managed when it looks like the expected PonceBot-managed worktree.

    This avoids wiping arbitrary directories while still recovering from older versions.
    """
    try:
        inside = _run(["git", "-C", str(wt_dir), "rev-parse", "--is-inside-work-tree"], check=False)
        if inside.returncode != 0:
            return False
    except Exception:
        return False

    expected_branch = f"poncebot/{role}/slot{slot}"
    try:
        br = _run(["git", "-C", str(wt_dir), "rev-parse", "--abbrev-ref", "HEAD"], check=False).stdout.strip()
    except Exception:
        br = ""
    if br and br not in ("HEAD", expected_branch):
        return False

    try:
        base_remote = _run(["git", "-C", str(base_repo), "config", "--get", "remote.origin.url"], check=False).stdout.strip()
        wt_remote = _run(["git", "-C", str(wt_dir), "config", "--get", "remote.origin.url"], check=False).stdout.strip()
        if base_remote and wt_remote and base_remote != wt_remote:
            return False
    except Exception:
        # If we can't validate remote safely, do not assume it's ours.
        return False

    try:
        sentinel.write_text(
            json.dumps(
                {
                    "base_repo": str(base_repo.resolve()),
                    "role": role,
                    "slot": int(slot),
                    "migrated_at": time.time(),
                    "note": "auto-marked as managed (legacy worktree without sentinel)",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            errors="replace",
        )
        return True
    except Exception:
        return False

def prepare_clean_workspace(path: Path) -> None:
    path = path.resolve()
    sentinel = path / _SENTINEL
    if not sentinel.exists():
        raise RuntimeError(f"Refusing to clean unmanaged worktree: {path}")

    # Best-effort: keep it clean and on latest base.
    _run(["git", "-C", str(path), "fetch", "origin", "--prune"], check=False)
    base_ref = _pick_base_ref(path)
    _run(["git", "-C", str(path), "reset", "--hard", base_ref], check=True)
    _run(["git", "-C", str(path), "clean", "-fdx", "-e", _SENTINEL], check=True)


def collect_git_artifacts(*, repo_dir: Path, artifacts_dir: Path) -> list[Path]:
    repo_dir = repo_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out: list[Path] = []

    status_p = artifacts_dir / "git_status.txt"
    diff_p = artifacts_dir / "changes.patch"

    st = _run(["git", "-C", str(repo_dir), "status", "--porcelain"], check=False)
    status_p.write_text(st.stdout or "", encoding="utf-8", errors="replace")
    out.append(status_p)

    df = _run(["git", "-C", str(repo_dir), "diff"], check=False)
    diff_p.write_text(df.stdout or "", encoding="utf-8", errors="replace")
    out.append(diff_p)

    return out


def _pick_base_ref(repo: Path) -> str:
    # Prefer origin/main if present, then main, else HEAD.
    if _run(["git", "-C", str(repo), "rev-parse", "--verify", "origin/main"], check=False).returncode == 0:
        return "origin/main"
    if _run(["git", "-C", str(repo), "rev-parse", "--verify", "main"], check=False).returncode == 0:
        return "main"
    return "HEAD"


def _run(cmd: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)
