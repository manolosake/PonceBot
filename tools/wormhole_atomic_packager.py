#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _run_or_fail(cmd: list[str], *, cwd: Path, step: str) -> None:
    p = _run(cmd, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(
            f"{step} failed: {' '.join(cmd)}\nstdout:\n{(p.stdout or '').strip()}\nstderr:\n{(p.stderr or '').strip()}"
        )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_entry(path: Path) -> dict[str, object]:
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "sha256": _sha256_file(path),
        "mtime_epoch": int(st.st_mtime),
        "captured_at_utc": _utc_now(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Atomic packager for wormhole backend bundle")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--contract", default="docs/contracts/wormhole_scene_contract.v1.json")
    ap.add_argument("--ticket-id", required=True)
    ap.add_argument("--expected-branch", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    art = Path(args.artifacts_dir).resolve()
    art.mkdir(parents=True, exist_ok=True)

    # 1) Export contract + trace first.
    _run_or_fail(
        [
            sys.executable,
            "tools/wormhole_scene_contract.py",
            "export",
            "--repo-root",
            ".",
            "--contract",
            args.contract,
            "--artifacts-dir",
            str(art),
            "--ticket-id",
            args.ticket_id,
            "--expected-branch",
            args.expected_branch,
        ],
        cwd=repo,
        step="contract_export",
    )

    # 2) Capture status + patch atomically from same git view (HEAD diff).
    status_proc = _run(["git", "diff", "--name-status", "HEAD"], cwd=repo)
    if status_proc.returncode != 0:
        raise RuntimeError("git diff --name-status HEAD failed")
    _atomic_write(art / "git_status.txt", status_proc.stdout or "")

    patch_proc = _run(["git", "diff", "--binary", "HEAD"], cwd=repo)
    if patch_proc.returncode != 0:
        raise RuntimeError("git diff --binary HEAD failed")
    _atomic_write(art / "changes.patch", patch_proc.stdout or "")

    # 3) patch apply check
    _run_or_fail(
        [
            sys.executable,
            "tools/wormhole_patch_apply_check.py",
            "--artifacts-dir",
            str(art),
            "--repo-root",
            ".",
            "--out",
            str(art / "patch_apply_check.json"),
        ],
        cwd=repo,
        step="patch_apply_check",
    )

    # 4) atomic guard
    _run_or_fail(
        [
            sys.executable,
            "tools/publish_atomic_guard.py",
            "--artifacts-dir",
            str(art),
            "--out",
            str(art / "publish_atomic_guard_report.json"),
            "--log",
            str(art / "publish_atomic_guard.log"),
        ],
        cwd=repo,
        step="publish_atomic_guard",
    )

    # 5) integrity gate with branch/ticket lock
    _run_or_fail(
        [
            sys.executable,
            "tools/wormhole_scene_integrity_gate.py",
            "--artifacts-dir",
            str(art),
            "--contract-source",
            args.contract,
            "--expected-branch",
            args.expected_branch,
            "--expected-ticket-id",
            args.ticket_id,
            "--report-out",
            str(art / "wormhole_scene_integrity_report.json"),
        ],
        cwd=repo,
        step="integrity_gate",
    )

    # 6) final immutable manifest for critical files.
    critical = [
        art / "changes.patch",
        art / "git_status.txt",
        art / "patch_apply_check.json",
        art / "publish_atomic_guard_report.json",
        art / "wormhole_scene_contract_export_report.json",
        art / "wormhole_scene_integrity_report.json",
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "ticket_id": args.ticket_id,
        "expected_order_branch": args.expected_branch,
        "artifacts_dir": str(art),
        "files": [_manifest_entry(p) for p in critical],
    }
    _atomic_write(art / "bundle_immutability_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps({"status": "PASS", "artifacts_dir": str(art)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
