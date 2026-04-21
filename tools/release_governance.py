#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return (p.stdout or "").strip()


def _try_run(cmd: list[str], *, cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def _utc_iso(ts: float | None = None) -> str:
    t = float(time.time() if ts is None else ts)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


_SSH_GH_RE = re.compile(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$")
_HTTPS_GH_RE = re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$")


def _parse_github_slug(remote_url: str) -> str | None:
    u = (remote_url or "").strip()
    m = _SSH_GH_RE.match(u)
    if m:
        return f"{m.group('owner')}/{m.group('repo')}"
    m = _HTTPS_GH_RE.match(u)
    if m:
        return f"{m.group('owner')}/{m.group('repo')}"
    return None


def _pr_compare_url(*, slug: str, base: str, head: str) -> str:
    # Prefer compare URL because it works even if the branch name includes slashes.
    return f"https://github.com/{slug}/compare/{base}...{head}?expand=1"


def _new_pr_url(*, slug: str, head: str) -> str:
    return f"https://github.com/{slug}/pull/new/{head}"


def _token_exact_in_text(text: str, token: str) -> bool:
    t = (token or "").strip()
    if not t:
        return False
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", text or ""))


def _order_token_from_branch(branch: str) -> str:
    b = (branch or "").strip()
    m = re.search(r"order-([0-9a-fA-F]{8})", b)
    return f"order:{m.group(1).lower()}" if m else ""


def _key_token_from_depends_on(depends_on: str) -> str:
    d = (depends_on or "").strip()
    if not d:
        return ""
    return d if d.startswith("key:") else f"key:{d}"


def _traceability_count_from_log(log_text: str, *, order_token: str, key_token: str) -> int:
    count = 0
    for line in (log_text or "").splitlines():
        if _token_exact_in_text(line, order_token) and _token_exact_in_text(line, key_token):
            count += 1
    return count


def _load_depends_on_key(qa_result_path: Path | None) -> str:
    if qa_result_path is None or (not qa_result_path.exists()):
        return ""
    try:
        payload = json.loads(qa_result_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    dep = payload.get("depends_on")
    if isinstance(dep, list) and dep:
        return str(dep[0] or "").strip()
    if isinstance(dep, str):
        return dep.strip()
    return ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_head_branch(*, repo: Path, remote: str, explicit_head: str, order_branch: str) -> str:
    requested = (explicit_head or order_branch or "").strip()
    if requested:
        # Accept local branch, local ref, or remote branch.
        candidates = [
            requested,
            f"refs/heads/{requested}",
            f"refs/remotes/{remote}/{requested}",
        ]
        for cand in candidates:
            rc, _out, _err = _try_run(["git", "rev-parse", "--verify", "--quiet", cand], cwd=repo)
            if rc == 0:
                return requested
        # Try fetching the requested branch from remote explicitly, then verify again.
        _try_run(["git", "fetch", "--prune", remote, requested], cwd=repo)
        rc, _out, _err = _try_run(["git", "rev-parse", "--verify", "--quiet", f"refs/remotes/{remote}/{requested}"], cwd=repo)
        if rc == 0:
            return requested
        raise RuntimeError(f"unable to resolve head branch ref: {requested}")
    return _run(["git", "branch", "--show-current"], cwd=repo).strip()


@dataclass(frozen=True)
class Check:
    key: str
    ok: bool
    details: str


def _git_root(repo: Path) -> Path:
    return Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=repo))


def _git_status_clean(repo: Path) -> Check:
    out = _run(["git", "status", "--porcelain"], cwd=repo)
    ok = not bool(out.strip())
    return Check("repo_clean", ok, "clean" if ok else out[:2000])


def _git_has_tracked_data(repo: Path) -> Check:
    rc, out, err = _try_run(["git", "ls-files", "data"], cwd=repo)
    if rc != 0:
        return Check("repo_hygiene_no_tracked_data", False, err or "git ls-files failed")
    ok = not bool(out.strip())
    return Check("repo_hygiene_no_tracked_data", ok, "ok" if ok else f"tracked files under data/: {out.splitlines()[:10]}")


def _git_ahead_behind(repo: Path, *, base_ref: str, head_ref: str) -> tuple[int, int]:
    out = _run(["git", "rev-list", "--left-right", "--count", f"{base_ref}...{head_ref}"], cwd=repo)
    left, right = out.split()
    # left = commits only in base, right = commits only in head
    return int(right), int(left)


def _git_diffstat(repo: Path, *, base_ref: str, head_ref: str) -> str:
    return _run(["git", "diff", "--stat", f"{base_ref}..{head_ref}"], cwd=repo)


def _run_unit_tests(repo: Path) -> Check:
    rc, out, err = _try_run([sys.executable, "-m", "unittest", "-q"], cwd=repo)
    ok = (rc == 0)
    detail = out.strip() or err.strip() or ("ok" if ok else "tests failed")
    return Check("tests", ok, detail[:4000])


def main() -> int:
    ap = argparse.ArgumentParser(description="Release governance checklist + PR URL generator (no gh required).")
    ap.add_argument("--repo", default=".", help="path to repo (default: .)")
    ap.add_argument("--remote", default="origin", help="remote name (default: origin)")
    ap.add_argument("--base", default="main", help="base branch name (default: main)")
    ap.add_argument("--head", default="", help="head branch name (default: current branch)")
    ap.add_argument("--order-branch", default="", help="explicit order branch to prefer as release head")
    ap.add_argument("--artifacts-dir", default="", help="optional artifacts dir to write checklist files")
    ap.add_argument("--qa-result", default="", help="optional path to qa_result.json")
    ap.add_argument("--run-tests", action="store_true", help="run python -m unittest -q as part of checklist")
    ap.add_argument("--require-pr", action="store_true", help="fail checklist if head==base")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    root = _git_root(repo)

    # Determine branches.
    head_branch = _resolve_head_branch(
        repo=root,
        remote=args.remote,
        explicit_head=(args.head or "").strip(),
        order_branch=(args.order_branch or "").strip(),
    )
    base_branch = (args.base or "main").strip()

    remote_url = _run(["git", "remote", "get-url", args.remote], cwd=root)
    slug = _parse_github_slug(remote_url)

    checks: list[Check] = []
    checks.append(_git_status_clean(root))
    checks.append(_git_has_tracked_data(root))
    if args.run_tests:
        checks.append(_run_unit_tests(root))

    base_ref = f"{args.remote}/{base_branch}"
    head_ref = head_branch
    # Ensure base ref exists.
    _run(["git", "fetch", args.remote, base_branch], cwd=root)
    ahead, behind = _git_ahead_behind(root, base_ref=base_ref, head_ref=head_ref)
    diffstat = _git_diffstat(root, base_ref=base_ref, head_ref=head_ref) if (ahead or behind) else ""

    if args.require_pr:
        checks.append(Check("requires_pr_branch", head_branch != base_branch, f"head={head_branch} base={base_branch}"))

    qa_result_path: Path | None = None
    if args.qa_result:
        qa_result_path = Path(args.qa_result).expanduser().resolve()
    elif args.artifacts_dir:
        cand = Path(args.artifacts_dir).expanduser().resolve() / "qa_result.json"
        if cand.exists():
            qa_result_path = cand
    dep_key = _load_depends_on_key(qa_result_path)
    if dep_key:
        order_token = _order_token_from_branch(head_branch)
        key_token = _key_token_from_depends_on(dep_key)
        head_body = _run(["git", "log", "-1", "--pretty=%B", "HEAD"], cwd=root)
        head_tokens_ok = bool(
            order_token
            and key_token
            and _token_exact_in_text(head_body, order_token)
            and _token_exact_in_text(head_body, key_token)
        )
        checks.append(
            Check(
                "traceability_head_tokens_match",
                head_tokens_ok,
                f"order_token={order_token or '<missing>'} key_token={key_token or '<missing>'}",
            )
        )
        tl = _run(
            [
                "git",
                "log",
                "--all-match",
                "--oneline",
                "--grep",
                order_token,
                "--grep",
                key_token,
                "-n",
                "200",
            ],
            cwd=root,
        )
        tcount = _traceability_count_from_log(tl, order_token=order_token, key_token=key_token)
        checks.append(Check("traceability_count_positive", tcount > 0, f"count={tcount}"))

    ok_all = all(c.ok for c in checks) and (not args.require_pr or head_branch != base_branch)

    pr_compare = _pr_compare_url(slug=slug, base=base_branch, head=head_branch) if slug else None
    pr_new = _new_pr_url(slug=slug, head=head_branch) if slug else None

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _utc_iso(),
        "repo_root": str(root),
        "remote": args.remote,
        "remote_url": remote_url,
        "github_slug": slug,
        "base_branch": base_branch,
        "head_branch": head_branch,
        "base_ref": base_ref,
        "ahead_commits": int(ahead),
        "behind_commits": int(behind),
        "diffstat": diffstat,
        "checks": [{"key": c.key, "ok": bool(c.ok), "details": c.details} for c in checks],
        "ok": bool(ok_all),
        "pr_compare_url": pr_compare,
        "pr_new_url": pr_new,
        "order_branch_requested": (args.order_branch or "").strip() or None,
    }

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir else None
    if artifacts_dir:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checklist_path = artifacts_dir / "RELEASE_CHECKLIST.json"
        checklist_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if pr_compare:
            (artifacts_dir / "PR_URL.txt").write_text(pr_compare + "\n", encoding="utf-8")
        if diffstat:
            (artifacts_dir / "DIFFSTAT.txt").write_text(diffstat + "\n", encoding="utf-8")
        changed = _run(["git", "diff", "--name-only", f"{base_ref}..{head_ref}"], cwd=root)
        (artifacts_dir / "CHANGED_FILES.txt").write_text((changed + "\n") if changed else "(none)\n", encoding="utf-8")

        required = [checklist_path, artifacts_dir / "PR_URL.txt", artifacts_dir / "CHANGED_FILES.txt"]
        present_non_empty = []
        for p in required:
            present_non_empty.append(bool(p.exists() and p.is_file() and p.stat().st_size > 0))
        pr_targets_head = bool(pr_compare and (f"...{head_branch}?" in pr_compare))
        final_checks = {
            "checks_ok": bool(all(c.ok for c in checks) and (not args.require_pr or head_branch != base_branch)),
            "pr_url_targets_head": bool(pr_targets_head),
            "required_artifacts_non_empty": bool(all(present_non_empty)),
        }
        final_ok = bool(all(final_checks.values()))
        payload["final_validation"] = {
            "ok": final_ok,
            "checks": final_checks,
            "validated_at": _utc_iso(),
        }
        payload["ok"] = bool(final_ok)
        checklist_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (artifacts_dir / "FINAL_VALIDATION.json").write_text(
            json.dumps(payload["final_validation"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "generated_at": _utc_iso(),
            "files": [],
        }
        for p in sorted(artifacts_dir.iterdir()):
            if p.is_file():
                manifest["files"].append(
                    {
                        "name": p.name,
                        "size": int(p.stat().st_size),
                        "sha256": _sha256_file(p),
                    }
                )
        (artifacts_dir / "FINAL_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0 if bool(payload.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
