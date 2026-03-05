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


def _git(cmd: list[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return ""
    return (p.stdout or "").strip()


def build_trace(
    *,
    repo_root: Path,
    artifacts_dir: Path,
    ticket_id: str,
    expected_branch: str,
    reported_branch_mode: str,
    execution_id: str,
) -> dict[str, Any]:
    observed_branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    head_sha = _git(["git", "rev-parse", "HEAD"], repo_root)

    if reported_branch_mode == "expected":
        reported_branch = expected_branch
    else:
        reported_branch = observed_branch

    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "ticket_id": ticket_id,
        "artifacts_dir": str(artifacts_dir),
        "expected_branch": expected_branch,
        "reported_branch": reported_branch,
        "branch_matches_expected": bool(expected_branch and reported_branch == expected_branch),
        "observed_git_branch": observed_branch,
        "head_sha": head_sha,
        "execution": {
            "execution_id": execution_id,
            "telemetry_correlation_id": f"{ticket_id}:{execution_id}",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export backend traceability artifact with branch provenance fields.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--ticket-id", required=True)
    ap.add_argument("--expected-branch", required=True)
    ap.add_argument(
        "--reported-branch-mode",
        choices=("expected", "observed"),
        default="expected",
        help="expected: reported_branch follows ORDER_BRANCH; observed: reported_branch follows current git branch",
    )
    ap.add_argument("--execution-id", default="")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    execution_id = args.execution_id or f"exec-{int(time.time())}"

    trace = build_trace(
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
        ticket_id=args.ticket_id,
        expected_branch=args.expected_branch,
        reported_branch_mode=args.reported_branch_mode,
        execution_id=execution_id,
    )

    trace_path = artifacts_dir / "wormhole_scene_trace.json"
    summary_path = artifacts_dir / "backend_traceability_summary.json"
    log_path = artifacts_dir / "wormhole_trace_export.log"

    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "status": "PASS" if trace["branch_matches_expected"] else "FAIL",
        "ticket_id": args.ticket_id,
        "expected_branch": args.expected_branch,
        "reported_branch": trace["reported_branch"],
        "branch_matches_expected": trace["branch_matches_expected"],
        "observed_git_branch": trace["observed_git_branch"],
        "head_sha": trace["head_sha"],
        "trace_path": str(trace_path),
        "generated_at_utc": _utc_now(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log_path.write_text(
        (
            f"TIMESTAMP={_utc_now()}\n"
            f"TICKET_ID={args.ticket_id}\n"
            f"EXPECTED_BRANCH={args.expected_branch}\n"
            f"REPORTED_BRANCH={trace['reported_branch']}\n"
            f"OBSERVED_GIT_BRANCH={trace['observed_git_branch']}\n"
            f"BRANCH_MATCHES_EXPECTED={trace['branch_matches_expected']}\n"
            f"STATUS={summary['status']}\n"
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

