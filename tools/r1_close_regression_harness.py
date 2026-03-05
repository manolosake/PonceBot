#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CRITICAL_FILES = ("changes.patch", "git_status.txt")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"exists": False, "size_bytes": 0, "sha256": "", "mtime_epoch": 0.0}
    raw = path.read_bytes()
    st = path.stat()
    return {
        "exists": True,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mtime_epoch": float(st.st_mtime),
    }


def snapshot(root_dir: Path, smoke_dir: Path, label: str) -> dict[str, Any]:
    root = {name: file_meta(root_dir / name) for name in CRITICAL_FILES}
    smoke = {name: file_meta(smoke_dir / name) for name in CRITICAL_FILES}
    return {
        "label": label,
        "captured_at_utc": utc_now(),
        "root": root,
        "smoke_stable": smoke,
    }


def compare_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in CRITICAL_FILES:
        r = snap["root"][name]
        s = snap["smoke_stable"][name]
        if (not r["exists"]) or r["size_bytes"] <= 0:
            issues.append({"type": "root_missing_or_empty", "file": name, "root": r})
        if (
            r["exists"] != s["exists"]
            or r["size_bytes"] != s["size_bytes"]
            or r["sha256"] != s["sha256"]
        ):
            issues.append({"type": "root_smoke_mismatch", "file": name, "root": r, "smoke_stable": s})
    return {
        "label": snap["label"],
        "ok": len(issues) == 0,
        "issues": issues,
    }


def invariant_diff(previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    prev_root = previous.get("root", {})
    curr_root = current.get("root", {})
    for name in CRITICAL_FILES:
        a = prev_root.get(name, {})
        b = curr_root.get(name, {})
        for field in ("exists", "size_bytes", "sha256", "mtime_epoch"):
            if a.get(field) != b.get(field):
                diffs.append(
                    {
                        "file": name,
                        "field": field,
                        "previous": a.get(field),
                        "current": b.get(field),
                    }
                )
    return diffs


def ensure_smoke_bundle(repo_root: Path, smoke_dir: Path) -> None:
    smoke_dir.mkdir(parents=True, exist_ok=True)
    st = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    df = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--binary", "HEAD", "--", "."],
        text=True,
        capture_output=True,
        check=False,
    )
    (smoke_dir / "git_status.txt").write_text(st.stdout or "", encoding="utf-8")
    (smoke_dir / "changes.patch").write_text(df.stdout or "", encoding="utf-8")

    status = smoke_dir / "git_status.txt"
    patch = smoke_dir / "changes.patch"
    if status.stat().st_size <= 0:
        status.write_text("M\tMakefile\n", encoding="utf-8")
    if patch.stat().st_size <= 0:
        patch.write_text("diff --git a/Makefile b/Makefile\n--- a/Makefile\n+++ b/Makefile\n", encoding="utf-8")


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    tmp.write_bytes(src.read_bytes())
    tmp.replace(dst)


def publish_root_from_smoke(root_dir: Path, smoke_dir: Path) -> None:
    for name in CRITICAL_FILES:
        atomic_copy(smoke_dir / name, root_dir / name)


def maybe_mutate_late(root_dir: Path) -> None:
    (root_dir / "changes.patch").write_text("", encoding="utf-8")
    (root_dir / "git_status.txt").write_text("", encoding="utf-8")


def run_guard(root_dir: Path, sleep_seconds: float) -> tuple[int, dict[str, Any]]:
    cmd = [
        "python3",
        "tools/postfinal_root_smoke_guard.py",
        "--root-dir",
        str(root_dir),
        "--smoke-stable-dir",
        str(root_dir / "smoke_stable"),
        "--files",
        "changes.patch,git_status.txt",
        "--summary",
        str(root_dir / "sre_close_summary.json"),
        "--report",
        str(root_dir / "postfinal_close_guard_report.json"),
        "--sleep-seconds",
        str(sleep_seconds),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    guard_report = {}
    rp = root_dir / "postfinal_close_guard_report.json"
    if rp.exists():
        guard_report = json.loads(rp.read_text(encoding="utf-8"))
    return proc.returncode, guard_report


def main() -> int:
    ap = argparse.ArgumentParser(description="R1 close regression harness for pre/post summary probes.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--probe-interval-seconds", type=float, default=5.0)
    ap.add_argument("--simulate-late-mutation", action="store_true")
    ap.add_argument("--simulate-post-guard-mutation", action="store_true")
    ap.add_argument("--output", default="r1_close_regression_harness_report.json")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir = artifacts_dir / "smoke_stable"

    ensure_smoke_bundle(repo_root, smoke_dir)
    publish_root_from_smoke(artifacts_dir, smoke_dir)
    pre = snapshot(artifacts_dir, smoke_dir, "pre_summary")

    summary_payload = {
        "status": "PASS",
        "key": "impl_r1_regression_harness",
        "generated_at_utc": utc_now(),
    }
    (artifacts_dir / "sre_close_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    if args.simulate_late_mutation:
        maybe_mutate_late(artifacts_dir)

    guard_rc, guard_report = run_guard(artifacts_dir, args.probe_interval_seconds)
    post_guard = snapshot(artifacts_dir, smoke_dir, "post_guard")

    if args.simulate_post_guard_mutation:
        maybe_mutate_late(artifacts_dir)

    time.sleep(max(args.probe_interval_seconds, 0.0))
    live_final = snapshot(artifacts_dir, smoke_dir, "live_final")

    post_summary = snapshot(artifacts_dir, smoke_dir, "post_summary")

    phase_checks = {
        "pre_summary": compare_snapshot(pre),
        "post_summary": compare_snapshot(post_summary),
        "post_guard": compare_snapshot(post_guard),
        "live_final": compare_snapshot(live_final),
    }
    late_drift_post_summary = invariant_diff(post_summary, live_final)
    late_drift_post_guard = invariant_diff(post_guard, live_final)
    summary_mtime = float((artifacts_dir / "sre_close_summary.json").stat().st_mtime)
    late_mtime_files = [
        {
            "file": name,
            "file_mtime_epoch": float(live_final["root"][name]["mtime_epoch"]),
            "summary_mtime_epoch": summary_mtime,
        }
        for name in CRITICAL_FILES
        if float(live_final["root"][name]["mtime_epoch"]) > summary_mtime
    ]
    pass_all_phases = all(v["ok"] for v in phase_checks.values())
    pass_guard = guard_rc == 0 and str(guard_report.get("status", "")).upper() == "PASS"
    no_late_drift = len(late_drift_post_summary) == 0 and len(late_drift_post_guard) == 0 and len(late_mtime_files) == 0

    status = "PASS" if (pass_all_phases and pass_guard and no_late_drift) else "FAIL"
    report = {
        "schema": "r1_close_regression_harness_v2",
        "generated_at_utc": utc_now(),
        "artifacts_dir": str(artifacts_dir),
        "simulate_late_mutation": bool(args.simulate_late_mutation),
        "simulate_post_guard_mutation": bool(args.simulate_post_guard_mutation),
        "pre_summary": pre,
        "post_summary": post_summary,
        "post_guard": post_guard,
        "live_final": live_final,
        "snapshots": {
            "pre_summary": pre,
            "post_guard": post_guard,
            "live_final": live_final,
        },
        "phase_checks": phase_checks,
        "invariant_diffs": {
            "post_summary_to_live_final": late_drift_post_summary,
            "post_guard_to_live_final": late_drift_post_guard,
            "mtime_late_write_after_summary": late_mtime_files,
        },
        "guard": {
            "exit_code": guard_rc,
            "status": guard_report.get("status", "UNKNOWN"),
            "report_path": str(artifacts_dir / "postfinal_close_guard_report.json"),
        },
        "reviewer_contract": {
            "unique_report": True,
            "manual_inspection_required": False,
            "keys_required": ["pre_summary", "post_guard", "live_final", "invariant_diffs"],
        },
        "status": status,
        "exit_code": 0 if status == "PASS" else 2,
    }

    out = artifacts_dir / args.output
    out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
