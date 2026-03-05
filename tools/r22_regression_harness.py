#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import struct
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6pJ8kAAAAASUVORK5CYII="
)


@dataclass
class RunResult:
    run_id: str
    scenario: str
    ok: bool
    export_rc: int
    guard_rc: int | None
    summary_status: str
    guard_status: str | None
    details: dict


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def png_with_dimensions(width: int, height: int, min_bytes: int = 5000) -> bytes:
    raw = bytearray(PNG_1X1)
    raw[16:24] = struct.pack(">II", width, height)
    if len(raw) < min_bytes:
        raw.extend(b"\x00" * (min_bytes - len(raw)))
    return bytes(raw)


def write_seed_bundle(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    smoke = run_dir / "smoke_stable"
    smoke.mkdir(parents=True, exist_ok=True)

    # Minimal coherent contractual files.
    status = "M\tMakefile\n"
    patch = "\n".join(
        [
            "diff --git a/Makefile b/Makefile",
            "--- a/Makefile",
            "+++ b/Makefile",
            "@@ -1 +1 @@",
            "-x",
            "+y",
        ]
    )
    (run_dir / "git_status.txt").write_text(status, encoding="utf-8")
    (run_dir / "changes.patch").write_text(patch + "\n", encoding="utf-8")
    (smoke / "git_status.txt").write_text(status, encoding="utf-8")
    (smoke / "changes.patch").write_text(patch + "\n", encoding="utf-8")
    (run_dir / "patch_apply_check.json").write_text(
        json.dumps({"status": "PASS", "files_in_patch": ["Makefile"]}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    png = png_with_dimensions(128, 128)
    (run_dir / "desktop_capture.png").write_bytes(png)
    (run_dir / "tablet_capture.png").write_bytes(png)
    (run_dir / "mobile_capture.png").write_bytes(png)


def run_export(repo_root: Path, run_dir: Path, ticket_id: str, expected_branch: str, run_id: str) -> int:
    cmd = [
        "python3",
        "tools/backend_traceability_runtime_export.py",
        "--repo-root",
        str(repo_root),
        "--artifacts-dir",
        str(run_dir),
        "--ticket-id",
        ticket_id,
        "--expected-branch",
        expected_branch,
        "--frontend-job-id",
        f"harness_{run_id}",
        "--target-artifact-dir",
        str(run_dir),
        "--execution-id",
        f"harness-{run_id}",
    ]
    return subprocess.call(cmd, cwd=repo_root)


def run_guard(repo_root: Path, run_dir: Path) -> int:
    cmd = [
        "python3",
        "tools/postseal_invariant_guard.py",
        "--artifacts-dir",
        str(run_dir),
    ]
    return subprocess.call(cmd, cwd=repo_root)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_nominal(repo_root: Path, suite_dir: Path, i: int, ticket_id: str, expected_branch: str) -> RunResult:
    run_id = f"nominal_{i}"
    run_dir = suite_dir / run_id
    write_seed_bundle(run_dir)
    export_rc = run_export(repo_root, run_dir, ticket_id, expected_branch, run_id)
    summary = load_json(run_dir / "backend_traceability_runtime_summary.json")
    freeze_status = summary.get("post_summary_freeze_probe", {}).get("status")
    freeze_ok = True if freeze_status is None else (freeze_status == "PASS")
    ok = export_rc == 0 and summary.get("status") == "PASS" and freeze_ok and summary.get("branch_provenance_match") is True
    return RunResult(
        run_id=run_id,
        scenario="nominal",
        ok=ok,
        export_rc=export_rc,
        guard_rc=None,
        summary_status=str(summary.get("status")),
        guard_status=None,
        details={
            "summary_path": str(run_dir / "backend_traceability_runtime_summary.json"),
            "freeze_probe_path": str(run_dir / "post_summary_freeze_probe.json"),
        },
    )


def run_corrupt(repo_root: Path, suite_dir: Path, i: int, ticket_id: str, expected_branch: str) -> RunResult:
    run_id = f"corrupt_{i}"
    run_dir = suite_dir / run_id
    write_seed_bundle(run_dir)
    export_rc = run_export(repo_root, run_dir, ticket_id, expected_branch, run_id)
    summary = load_json(run_dir / "backend_traceability_runtime_summary.json")

    # Simulate late mutation after summary.
    (run_dir / "changes.patch").write_text("diff --git a/Makefile b/Makefile\n# late mutation\n", encoding="utf-8")
    (run_dir / "git_status.txt").write_text("M\tLateMutation\n", encoding="utf-8")

    guard_rc = run_guard(repo_root, run_dir)
    guard = load_json(run_dir / "postseal_invariant_guard_report.json")
    ok = export_rc == 0 and summary.get("status") == "PASS" and guard_rc != 0 and guard.get("status") == "FAIL"
    return RunResult(
        run_id=run_id,
        scenario="corrupt_post_summary_mutation",
        ok=ok,
        export_rc=export_rc,
        guard_rc=guard_rc,
        summary_status=str(summary.get("status")),
        guard_status=str(guard.get("status")),
        details={
            "summary_path": str(run_dir / "backend_traceability_runtime_summary.json"),
            "guard_report_path": str(run_dir / "postseal_invariant_guard_report.json"),
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="R22 regression harness: nominal vs post-summary mutation detection.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--ticket-id", required=True)
    ap.add_argument("--expected-branch", required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    suite_dir = artifacts_dir / "r22_regression_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for i in range(1, args.runs + 1):
        results.append(run_nominal(repo_root, suite_dir, i, args.ticket_id, args.expected_branch))
    for i in range(1, args.runs + 1):
        results.append(run_corrupt(repo_root, suite_dir, i, args.ticket_id, args.expected_branch))

    nominal = [r for r in results if r.scenario == "nominal"]
    corrupt = [r for r in results if r.scenario != "nominal"]
    nominal_pass = sum(1 for r in nominal if r.ok)
    corrupt_detected = sum(1 for r in corrupt if r.ok)

    summary = {
        "schema": "r22_regression_harness_v1",
        "generated_at_utc": utc_now(),
        "ticket_id": args.ticket_id,
        "expected_branch": args.expected_branch,
        "runs_per_scenario": args.runs,
        "nominal_pass_count": nominal_pass,
        "corrupt_detected_count": corrupt_detected,
        "nominal_required": args.runs,
        "corrupt_required": args.runs,
        "status": "PASS" if (nominal_pass >= args.runs and corrupt_detected >= args.runs) else "FAIL",
        "results": [
            {
                "run_id": r.run_id,
                "scenario": r.scenario,
                "ok": r.ok,
                "export_rc": r.export_rc,
                "guard_rc": r.guard_rc,
                "summary_status": r.summary_status,
                "guard_status": r.guard_status,
                "details": r.details,
            }
            for r in results
        ],
    }

    report_path = artifacts_dir / "r22_regression_harness_report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
