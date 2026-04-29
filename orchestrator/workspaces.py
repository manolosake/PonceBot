from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import re
import subprocess
import threading
import time


_SENTINEL = ".poncebot_managed_worktree"
_GITDIR_SENTINEL = "poncebot/managed_worktree.json"
_POOL_LOCKS: dict[str, threading.Lock] = {}
_POOL_LOCKS_GUARD = threading.Lock()


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

    with _worktree_pool_lock(base_repo=base_repo, root=root, role=role):
        for slot in range(1, slots + 1):
            wt_dir = (root / role / f"slot{slot}").resolve()
            wt_dir.parent.mkdir(parents=True, exist_ok=True)
            if wt_dir.exists():
                if _read_managed_metadata(wt_dir):
                    _ensure_legacy_marker_ignored(wt_dir)
                    _migrate_legacy_marker_out_of_tree(wt_dir)
                    continue
                # Back-compat: older deployments created worktrees without git-dir metadata. If the
                # directory appears to be one of our worktrees, mark it managed so future cleanups are safe.
                if _maybe_mark_managed_worktree(base_repo=base_repo, root=root, wt_dir=wt_dir, role=role, slot=slot):
                    continue
                raise RuntimeError(f"worktree exists but is not managed: {wt_dir}")

            branch = _managed_worktree_branch(base_repo=base_repo, root=root, role=role, slot=slot)
            base_ref = _pick_base_ref(base_repo)
            _run(["git", "-C", str(base_repo), "fetch", "origin", "--prune"], check=False)
            _run(["git", "-C", str(base_repo), "worktree", "add", "-B", branch, str(wt_dir), base_ref], check=True)

            _write_managed_metadata(
                wt_dir,
                {
                    "base_repo": str(base_repo),
                    "branch": branch,
                    "role": role,
                    "slot": slot,
                    "created_at": time.time(),
                    "metadata_version": 2,
                },
            )
            _ensure_legacy_marker_ignored(wt_dir)
            _migrate_legacy_marker_out_of_tree(wt_dir)


def _maybe_mark_managed_worktree(*, base_repo: Path, root: Path, wt_dir: Path, role: str, slot: int) -> bool:
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

    expected_branch = _managed_worktree_branch(base_repo=base_repo, root=root, role=role, slot=slot)
    legacy_branch = _legacy_managed_worktree_branch(role=role, slot=slot)
    try:
        br = _run(["git", "-C", str(wt_dir), "rev-parse", "--abbrev-ref", "HEAD"], check=False).stdout.strip()
    except Exception:
        br = ""
    if br and br not in ("HEAD", expected_branch, legacy_branch):
        # Older pool bugs could leave a valid PonceBot worktree in the right slot
        # but checked out to a sibling managed branch. Reclaim only same-namespace
        # branches. Some older jobs also left exact pool slots on order branches;
        # reclaim those only when Git proves the slot belongs to this same repo.
        expected_prefix = f"poncebot/{_managed_worktree_namespace(base_repo=base_repo, root=root)}/"
        if not br.startswith(expected_prefix) and not _same_git_common_dir(base_repo, wt_dir):
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
        _write_managed_metadata(
            wt_dir,
            {
                "base_repo": str(base_repo.resolve()),
                "branch": expected_branch,
                "role": role,
                "slot": int(slot),
                "migrated_at": time.time(),
                "metadata_version": 2,
                "note": "auto-marked as managed (legacy worktree without git-dir metadata)",
            },
        )
        _ensure_legacy_marker_ignored(wt_dir)
        _migrate_legacy_marker_out_of_tree(wt_dir)
        return True
    except Exception:
        return False


def _same_git_common_dir(left: Path, right: Path) -> bool:
    try:
        left_common = _git_common_dir(left)
        right_common = _git_common_dir(right)
        return bool(left_common and right_common and left_common == right_common)
    except Exception:
        return False


def _git_common_dir(path: Path) -> Path | None:
    proc = _run(["git", "-C", str(path), "rev-parse", "--git-common-dir"], check=False)
    raw = str(proc.stdout or "").strip()
    if proc.returncode != 0 or not raw:
        return None
    common = Path(raw)
    if not common.is_absolute():
        common = (path.resolve() / common).resolve()
    return common.resolve()


def _managed_worktree_branch(*, base_repo: Path, root: Path, role: str, slot: int) -> str:
    return f"poncebot/{_managed_worktree_namespace(base_repo=base_repo, root=root)}/{role}/slot{int(slot)}"


def _legacy_managed_worktree_branch(*, role: str, slot: int) -> str:
    return f"poncebot/{role}/slot{int(slot)}"


def _managed_worktree_namespace(*, base_repo: Path, root: Path) -> str:
    base_repo = base_repo.resolve()
    root = root.resolve()
    try:
        label_source = str(root.relative_to(base_repo))
    except Exception:
        label_source = str(root.name or "worktree")
    label = _git_ref_component(label_source.replace("\\", "/").replace("/", "-"))
    digest = hashlib.sha1(str(root).encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{label}-{digest}" if label else f"worktree-{digest}"


def _git_ref_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip(".-/")
    return text[:48].strip(".-/")


def _worktree_pool_lock(*, base_repo: Path, root: Path, role: str) -> threading.Lock:
    key = f"{base_repo.resolve()}::{root.resolve()}::{(role or '').strip().lower()}"
    with _POOL_LOCKS_GUARD:
        lock = _POOL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _POOL_LOCKS[key] = lock
        return lock


def _managed_metadata_path(path: Path) -> Path:
    path = path.resolve()
    try:
        proc = _run(["git", "-C", str(path), "rev-parse", "--git-path", _GITDIR_SENTINEL], check=False)
        raw = str(proc.stdout or "").strip()
        if proc.returncode == 0 and raw:
            candidate = Path(raw)
            return candidate if candidate.is_absolute() else (path / candidate).resolve()
    except Exception:
        pass
    return (path / ".git" / _GITDIR_SENTINEL).resolve()


def _legacy_sentinel_path(path: Path) -> Path:
    return path.resolve() / _SENTINEL


def _git_path_is_tracked(repo_dir: Path, rel_path: str) -> bool:
    proc = _run(["git", "-C", str(repo_dir), "ls-files", "--error-unmatch", "--", rel_path], check=False)
    return proc.returncode == 0


def _read_managed_metadata(path: Path) -> dict[str, Any]:
    metadata_path = _managed_metadata_path(path)
    for candidate in (metadata_path, _legacy_sentinel_path(path)):
        if not candidate.exists():
            continue
        if candidate.name == _SENTINEL and _git_path_is_tracked(path, _SENTINEL):
            # A tracked marker is repository content, not trustworthy worktree metadata.
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8", errors="replace") or "{}")
        except Exception:
            continue
        if isinstance(data, dict) and str(data.get("branch") or "").strip():
            return data
    return {}


def _write_managed_metadata(path: Path, data: dict[str, Any]) -> Path:
    metadata_path = _managed_metadata_path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(dict(data or {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        errors="replace",
    )
    return metadata_path


def _ensure_legacy_marker_ignored(path: Path) -> None:
    try:
        proc = _run(["git", "-C", str(path), "rev-parse", "--git-path", "info/exclude"], check=False)
        raw = str(proc.stdout or "").strip()
        if proc.returncode != 0 or not raw:
            return
        exclude_path = Path(raw)
        if not exclude_path.is_absolute():
            exclude_path = (path.resolve() / exclude_path).resolve()
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        current = exclude_path.read_text(encoding="utf-8", errors="replace") if exclude_path.exists() else ""
        lines = {line.strip() for line in current.splitlines()}
        if _SENTINEL not in lines:
            suffix = "" if current.endswith("\n") or not current else "\n"
            exclude_path.write_text(current + suffix + _SENTINEL + "\n", encoding="utf-8", errors="replace")
    except Exception:
        return


def _migrate_legacy_marker_out_of_tree(path: Path) -> None:
    legacy = _legacy_sentinel_path(path)
    if not legacy.exists() or _git_path_is_tracked(path, _SENTINEL):
        return
    metadata = _read_managed_metadata(path)
    if metadata and not _managed_metadata_path(path).exists():
        _write_managed_metadata(path, metadata)
    try:
        if legacy.is_file() or legacy.is_symlink():
            legacy.unlink()
    except Exception:
        pass


def _managed_branch_from_metadata(path: Path) -> str:
    data = _read_managed_metadata(path)
    if not data:
        return ""
    return str((data or {}).get("branch") or "").strip()


def prepare_clean_workspace(path: Path) -> None:
    path = path.resolve()
    metadata = _read_managed_metadata(path)
    if not metadata:
        raise RuntimeError(f"Refusing to clean unmanaged worktree: {path}")
    if not _managed_metadata_path(path).exists():
        _write_managed_metadata(path, {**metadata, "migrated_at": time.time(), "metadata_version": 2})
    _ensure_legacy_marker_ignored(path)

    # Best-effort: keep it clean and on latest base.
    # First neutralize any dirty tracked/untracked state on the current HEAD so
    # branch switches cannot fail due to stale local modifications from a
    # previous run in the managed worktree.
    _run(["git", "-C", str(path), "fetch", "origin", "--prune"], check=False)
    _run(["git", "-C", str(path), "reset", "--hard"], check=True)
    _run(["git", "-C", str(path), "clean", "-fdx"], check=True)
    base_ref_repo = Path(str(metadata.get("base_repo") or "")).expanduser() if isinstance(metadata, dict) else Path()
    if not str(base_ref_repo) or not base_ref_repo.exists():
        base_ref_repo = path
    base_ref = _pick_base_ref(base_ref_repo)
    managed_branch = _managed_branch_from_metadata(path)
    if managed_branch:
        checkout = _run(["git", "-C", str(path), "checkout", "-B", managed_branch, base_ref], check=False)
        if checkout.returncode != 0:
            _run(["git", "-C", str(path), "clean", "-fdx"], check=True)
            _run(["git", "-C", str(path), "checkout", "--detach", base_ref], check=True)
    _run(["git", "-C", str(path), "reset", "--hard", base_ref], check=True)
    _run(["git", "-C", str(path), "clean", "-fdx"], check=True)
    _ensure_legacy_marker_ignored(path)


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
    # Prefer the deployed checkout branch. Managed role worktrees should track
    # the runtime branch (for example codex/codexbot-workflow-v2), not always
    # origin/main.
    current = _run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], check=False)
    current_branch = str(current.stdout or "").strip() if current.returncode == 0 else ""
    if current_branch and current_branch != "HEAD" and not current_branch.startswith("poncebot/"):
        if _run(["git", "-C", str(repo), "rev-parse", "--verify", f"origin/{current_branch}"], check=False).returncode == 0:
            return f"origin/{current_branch}"
        if _run(["git", "-C", str(repo), "rev-parse", "--verify", current_branch], check=False).returncode == 0:
            return current_branch

    # Fall back to conventional main/master names, then current HEAD.
    if _run(["git", "-C", str(repo), "rev-parse", "--verify", "origin/main"], check=False).returncode == 0:
        return "origin/main"
    if _run(["git", "-C", str(repo), "rev-parse", "--verify", "main"], check=False).returncode == 0:
        return "main"
    if _run(["git", "-C", str(repo), "rev-parse", "--verify", "origin/master"], check=False).returncode == 0:
        return "origin/master"
    if _run(["git", "-C", str(repo), "rev-parse", "--verify", "master"], check=False).returncode == 0:
        return "master"
    return "HEAD"


def _run(cmd: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)
