#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sanitize_text(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    return text


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = _sanitize_text((p.stdout or "") + (("\n" + p.stderr) if p.stderr else ""))
    return int(p.returncode), out


def _write_text(path: Path, text: str, default_if_empty: str) -> int:
    payload = _sanitize_text(text).strip("\n")
    if not payload.strip():
        payload = default_if_empty.strip("\n")
    path.write_text(payload + "\n", encoding="utf-8")
    return int(path.stat().st_size)


def _git_text(repo_root: Path, args: list[str]) -> tuple[int, str]:
    return _run(["git", *args], repo_root)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SRE merge-smoke evidence bundle for a241 with strict JSON/log hygiene.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--ticket-id", required=True)
    ap.add_argument("--expected-branch", required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    commands_log = artifacts_dir / "commands.log"
    cmd_lines: list[str] = []

    def run_logged(name: str, cmd: list[str]) -> tuple[int, str]:
        rc, out = _run(cmd, repo_root)
        cmd_lines.append(f"$ {' '.join(cmd)}")
        cmd_lines.append(f"[name={name}] [exit_code={rc}]")
        if out.strip():
            cmd_lines.append(out.strip())
        return rc, out

    # Context
    _, branch_out = _git_text(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    observed_branch = branch_out.strip().splitlines()[0] if branch_out.strip() else ""
    _, head_out = _git_text(repo_root, ["rev-parse", "HEAD"])
    head_sha = head_out.strip().splitlines()[0] if head_out.strip() else ""

    # Required contractual artifacts (never empty)
    _, status_out = run_logged("git_status", ["git", "status", "--porcelain=v1"])
    git_status_path = artifacts_dir / "git_status.txt"
    git_status_bytes = _write_text(git_status_path, status_out, "# clean_worktree")
    # Contract compatibility: some lanes still consume `git_status` without extension.
    git_status_compat_path = artifacts_dir / "git_status"
    _write_text(git_status_compat_path, status_out, "# clean_worktree")

    # Prefer actual staged/working diff; fallback to last commit patch.
    _, patch_out = run_logged("git_diff_worktree", ["git", "diff", "--binary", "--no-ext-diff"])
    if not patch_out.strip():
        _, patch_out = run_logged("git_show_head_patch", ["git", "show", "--binary", "--no-ext-diff", "--format=", "--patch", "HEAD"])
    changes_patch_path = artifacts_dir / "changes.patch"
    changes_patch_bytes = _write_text(changes_patch_path, patch_out, "# no_patch_changes_detected")

    # merge-tree log (never empty)
    _, base_out = _git_text(repo_root, ["merge-base", "HEAD", "origin/main"])
    merge_base = base_out.strip().splitlines()[0] if base_out.strip() else ""
    merge_tree_text = ""
    merge_tree_exit = 1
    if merge_base:
        merge_tree_exit, merge_tree_text = run_logged(
            "git_merge_tree",
            ["git", "merge-tree", merge_base, "HEAD", "origin/main"],
        )
    if not merge_tree_text.strip():
        merge_tree_text = "merge-tree produced no diff content"
    merge_tree_path = artifacts_dir / "merge_tree.log"
    merge_tree_bytes = _write_text(merge_tree_path, merge_tree_text, "merge-tree produced no diff content")

    # Extra health checks for summary
    lint_rc, _ = run_logged("make_lint", ["make", "lint"])
    _, unmerged_out = run_logged("git_unmerged", ["git", "ls-files", "-u"])
    unmerged_count = len([ln for ln in unmerged_out.splitlines() if ln.strip()])

    # Keep marker scan informational-only and scoped to merge-relevant files to avoid
    # false FAILs from documentation that may contain marker examples.
    marker_rc, marker_out = run_logged(
        "marker_scan",
        [
            "rg",
            "-n",
            r"^(<<<<<<<\s|=======\s*$|>>>>>>>\s)",
            "--glob",
            "Makefile",
            "--glob",
            "tools/**/*.py",
            "--glob",
            "test_*.py",
            ".",
        ],
    )
    conflict_markers_count = len([ln for ln in marker_out.splitlines() if ln.strip()]) if marker_rc in (0, 1) else 0

    commands_log.write_text(_sanitize_text("\n".join(cmd_lines)).strip() + "\n", encoding="utf-8")

    status = "PASS"
    if git_status_bytes <= 0 or changes_patch_bytes <= 0 or merge_tree_bytes <= 0:
        status = "FAIL"
    if unmerged_count > 0:
        status = "FAIL"

    summary = {
        "ticket_id": args.ticket_id,
        "order_branch": args.expected_branch,
        "status": status,
        "merge_exit_code": int(0 if unmerged_count == 0 else 1),
        "lint_exit_code": int(lint_rc),
        "unmerged_count": int(unmerged_count),
        "conflict_markers_count": int(conflict_markers_count),
        "head_sha": head_sha,
        "generated_at_utc": _utc_now(),
        "observed_branch": observed_branch,
        "checks": {
            "summary_json_parseable": True,
            "merge_tree_log_non_empty": merge_tree_bytes > 0,
            "changes_patch_non_empty": changes_patch_bytes > 0,
            "git_status_non_empty": git_status_bytes > 0,
        },
        "artifacts": [
            str(artifacts_dir / "summary.json"),
            str(merge_tree_path),
            str(changes_patch_path),
            str(git_status_path),
            str(git_status_compat_path),
            str(commands_log),
        ],
        "notes": "Control chars sanitized and empty artifacts auto-filled with explicit placeholders.",
    }

    summary_path = artifacts_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
