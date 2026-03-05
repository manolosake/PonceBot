#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return int(p.returncode), (p.stdout or "").strip(), (p.stderr or "").strip()


def _remote_head_sha(repo_root: Path, remote: str, branch: str) -> tuple[str, str]:
    rc, out, err = _run(["git", "ls-remote", "--heads", remote, branch], repo_root)
    if rc != 0:
        return "", err
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].endswith(f"/{branch}"):
            return parts[0].strip(), ""
    return "", ""


def _bool_check(key: str, ok: bool, details: str) -> dict[str, Any]:
    return {"key": key, "ok": bool(ok), "details": details}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Backend done evidence guard: blocks close when remote push proof/SHA/clean status are missing."
    )
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--expected-branch", required=True)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out).resolve() if args.out else (artifacts_dir / "backend_done_evidence_guard_report.json")

    _, observed_branch, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    _, local_sha, _ = _run(["git", "rev-parse", "HEAD"], repo_root)
    _, status_porcelain, _ = _run(["git", "status", "--porcelain"], repo_root)
    remote_sha, remote_sha_error = _remote_head_sha(repo_root, args.remote, args.expected_branch)

    checks: list[dict[str, Any]] = []
    checks.append(
        _bool_check(
            "branch_matches_expected",
            observed_branch == args.expected_branch,
            f"observed={observed_branch} expected={args.expected_branch}",
        )
    )
    checks.append(
        _bool_check(
            "git_status_clean",
            not bool(status_porcelain.strip()),
            "clean" if not status_porcelain.strip() else status_porcelain,
        )
    )
    checks.append(
        _bool_check(
            "remote_sha_present",
            bool(remote_sha),
            remote_sha if remote_sha else (remote_sha_error or "remote branch SHA not found"),
        )
    )
    checks.append(
        _bool_check(
            "push_proof_matches_remote_sha",
            bool(remote_sha) and bool(local_sha) and (remote_sha == local_sha),
            f"local_sha={local_sha} remote_sha={remote_sha}",
        )
    )

    status = "PASS" if all(c.get("ok") for c in checks) else "FAIL"
    result = {
        "status": status,
        "exit_code": 0 if status == "PASS" else 2,
        "generated_at_utc": _utc_now(),
        "repo_root": str(repo_root),
        "artifacts_dir": str(artifacts_dir),
        "expected_branch": args.expected_branch,
        "observed_branch": observed_branch,
        "remote": args.remote,
        "local_sha": local_sha,
        "remote_sha": remote_sha,
        "post_push_verification": {
            "proof_type": "git ls-remote --heads",
            "command": f"git ls-remote --heads {args.remote} {args.expected_branch}",
            "remote_sha_observed": remote_sha,
            "matches_local_head": bool(remote_sha) and bool(local_sha) and (remote_sha == local_sha),
        },
        "checks": checks,
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
