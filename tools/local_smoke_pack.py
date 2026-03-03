#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot(path: Path) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    size = int(path.stat().st_size) if exists else 0
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size,
        "sha256": _sha256(path) if size > 0 else "",
        "captured_at_utc": _utc_now(),
    }


def _run(cmd: list[str], cwd: Path, log_path: Path) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    payload = (
        f"TIMESTAMP={_utc_now()}\n"
        f"COMMAND={' '.join(cmd)}\n"
        f"EXIT_CODE={proc.returncode}\n"
        "----- STDOUT BEGIN -----\n"
        f"{(proc.stdout or '').rstrip()}\n"
        "----- STDOUT END -----\n"
        "----- STDERR BEGIN -----\n"
        f"{(proc.stderr or '').rstrip()}\n"
        "----- STDERR END -----\n"
    )
    log_path.write_text(payload, encoding="utf-8")
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local smoke pack (publish preflight + preview integrity gate).")
    parser.add_argument("--workspace", required=True, help="Workspace root to validate")
    parser.add_argument("--artifacts-dir", required=True, help="Artifacts output directory")
    parser.add_argument("--order-branch", required=True, help="Expected order branch")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    workspace = Path(args.workspace).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    preview_root = workspace / ".codexbot_preview"
    preview_html = preview_root / "preview.html"
    index_html = preview_root / "index.html"

    preview_snap = _snapshot(preview_html)
    index_snap = _snapshot(index_html)
    expected_preview_sha = preview_snap["sha256"]

    preflight_out = artifacts_dir / "preflight_report.json"
    preflight_log = artifacts_dir / "smoke_preflight.log"
    integrity_out = artifacts_dir / "preview_integrity_gate_report.json"
    integrity_log = artifacts_dir / "smoke_integrity_gate.log"

    preflight_code = _run(
        [
            "python3",
            str(repo_root / "tools" / "preflight_preview_gate.py"),
            "--workspace",
            str(workspace),
            "--out",
            str(preflight_out),
        ],
        cwd=repo_root,
        log_path=preflight_log,
    )
    integrity_code = _run(
        [
            "python3",
            str(repo_root / "tools" / "preview_integrity_gate.py"),
            "--workspace",
            str(workspace),
            "--expected-branch",
            args.order_branch,
            "--expected-preview-sha",
            str(expected_preview_sha),
            "--out",
            str(integrity_out),
        ],
        cwd=repo_root,
        log_path=integrity_log,
    )

    preflight_payload = json.loads(preflight_out.read_text(encoding="utf-8")) if preflight_out.exists() else {}
    integrity_payload = json.loads(integrity_out.read_text(encoding="utf-8")) if integrity_out.exists() else {}

    consistency = {
        "preflight_report_exists": preflight_out.exists(),
        "integrity_report_exists": integrity_out.exists(),
        "preview_path_matches": str(integrity_payload.get("paths", {}).get("preview_html", "")) == str(preview_html),
        "index_path_matches": str(integrity_payload.get("paths", {}).get("index_html", "")) == str(index_html),
        "preview_sha_matches_snapshot": str(integrity_payload.get("observed", {}).get("preview_sha256", "")) == str(
            preview_snap["sha256"]
        ),
        "index_sha_matches_snapshot": str(integrity_payload.get("observed", {}).get("index_sha256", "")) == str(
            index_snap["sha256"]
        ),
    }

    summary = {
        "status": (
            "PASS"
            if preflight_code == 0 and integrity_code == 0 and all(bool(v) for v in consistency.values())
            else "FAIL"
        ),
        "workspace": str(workspace),
        "preview_root": str(preview_root),
        "order_branch_expected": args.order_branch,
        "exit_codes": {
            "preflight_preview_gate": preflight_code,
            "preview_integrity_gate": integrity_code,
        },
        "paths": {
            "preflight_report": str(preflight_out),
            "preflight_log": str(preflight_log),
            "integrity_report": str(integrity_out),
            "integrity_log": str(integrity_log),
        },
        "snapshots": {
            "index.html": index_snap,
            "preview.html": preview_snap,
        },
        "consistency_checks": consistency,
        "captured_at_utc": _utc_now(),
    }
    summary_path = artifacts_dir / "smoke_pack_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (artifacts_dir / "smoke_pack.log").write_text(
        (
            f"TIMESTAMP={_utc_now()}\n"
            f"WORKSPACE={workspace}\n"
            f"ORDER_BRANCH={args.order_branch}\n"
            f"PREVIEW_SHA256={preview_snap['sha256']}\n"
            f"PREFLIGHT_EXIT_CODE={preflight_code}\n"
            f"INTEGRITY_EXIT_CODE={integrity_code}\n"
            f"OVERALL_STATUS={summary['status']}\n"
            f"EXIT_CODE={0 if summary['status'] == 'PASS' else 1}\n"
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
