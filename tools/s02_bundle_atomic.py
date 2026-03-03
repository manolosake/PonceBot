#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from s02_trace_checker import validate_s02_trace


QA_PREVIEW_DIR_DEFAULT = Path("/home/aponce/codexbot/data/worktrees/qa/slot1/.codexbot_preview")


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


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return int(proc.returncode), out


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_status(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return int(resp.status), ""
    except urllib.error.HTTPError as e:
        return int(e.code), str(e)
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


def _write_trace_files(artifacts_dir: Path, repo_root: Path) -> dict[str, Any]:
    ts = _utc_now()
    code, branch = _run(["git", "branch", "--show-current"], cwd=repo_root)
    branch = branch.strip() if code == 0 else "unknown"
    code, sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    sha = sha.strip() if code == 0 else "unknown"
    _, status = _run(["git", "status", "--short", "--branch"], cwd=repo_root)
    _, diff = _run(["git", "diff", "--patch", "--minimal"], cwd=repo_root)

    git_status_text = (
        f"captured_at_utc={ts}\n"
        f"branch={branch}\n"
        f"head_sha={sha}\n"
        f"---\n"
        f"{status}"
    )
    _atomic_write_text(artifacts_dir / "git_status.txt", git_status_text.rstrip() + "\n")

    if diff.strip():
        patch_text = diff.rstrip() + "\n"
    else:
        patch_text = (
            "NO_DIFF=true\n"
            f"captured_at_utc={ts}\n"
            f"head_sha={sha}\n"
            f"branch={branch}\n"
            "justification=no tracked changes to include in patch for this run\n"
        )
    _atomic_write_text(artifacts_dir / "changes.patch", patch_text)
    return {"captured_at_utc": ts, "branch": branch, "head_sha": sha}


def _run_preflight(artifacts_dir: Path, qa_preview_dir: Path) -> dict[str, Any]:
    index_path = qa_preview_dir / "index.html"
    preview_path = qa_preview_dir / "preview.html"
    index_snap = _snapshot(index_path)
    preview_snap = _snapshot(preview_path)

    port = _pick_port()
    server_cmd = [
        sys.executable,
        "-m",
        "http.server",
        str(port),
        "--bind",
        "127.0.0.1",
        "--directory",
        str(qa_preview_dir),
    ]
    proc = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)
    try:
        root_status, root_err = _http_status(f"http://127.0.0.1:{port}/")
        preview_status, preview_err = _http_status(f"http://127.0.0.1:{port}/preview.html")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    checks = {
        "index_exists_non_empty": bool(index_snap["exists"] and index_snap["size_bytes"] > 0),
        "preview_exists_non_empty": bool(preview_snap["exists"] and preview_snap["size_bytes"] > 0),
        "http_root_200": root_status == 200,
        "http_preview_200": preview_status == 200,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "checks": checks,
        "http": {
            "root": {"url": f"http://127.0.0.1:{port}/", "status_code": root_status, "error": root_err},
            "preview": {
                "url": f"http://127.0.0.1:{port}/preview.html",
                "status_code": preview_status,
                "error": preview_err,
            },
        },
        "files": {"index.html": index_snap, "preview.html": preview_snap},
        "qa_preview_dir": str(qa_preview_dir),
        "captured_at_utc": _utc_now(),
    }
    _atomic_write_json(artifacts_dir / "preflight_report.json", report)
    _atomic_write_text(
        artifacts_dir / "preflight.log",
        (
            f"TIMESTAMP={_utc_now()}\n"
            f"PREVIEW_DIR={qa_preview_dir}\n"
            f"HTTP_ROOT_STATUS={root_status}\n"
            f"HTTP_PREVIEW_STATUS={preview_status}\n"
            f"EXIT_CODE={0 if status == 'PASS' else 1}\n"
        ),
    )
    return report


def _run_verify(artifacts_dir: Path, repo_root: Path, verify_cmd: str) -> dict[str, Any]:
    code, output = _run(["bash", "-lc", verify_cmd], cwd=repo_root)
    log = (
        f"TIMESTAMP={_utc_now()}\n"
        f"CWD={repo_root}\n"
        f"COMMAND={verify_cmd}\n"
        f"EXIT_CODE={code}\n"
        "----- OUTPUT BEGIN -----\n"
        f"{output.rstrip()}\n"
        "----- OUTPUT END -----\n"
    )
    _atomic_write_text(artifacts_dir / "verify.log", log)
    return {"status": "PASS" if code == 0 else "FAIL", "exit_code": code}


def _post_publish_check(artifacts_dir: Path, validation: dict[str, Any]) -> dict[str, Any]:
    files = validation.get("files", {})
    expected_gs = files.get("git_status.txt", {})
    expected_patch = files.get("changes.patch", {})
    gs_path = artifacts_dir / "git_status.txt"
    patch_path = artifacts_dir / "changes.patch"
    actual_gs = _snapshot(gs_path)
    actual_patch = _snapshot(patch_path)

    checks = {
        "git_status_size_gt_zero": actual_gs["size_bytes"] > 0,
        "changes_patch_size_gt_zero": actual_patch["size_bytes"] > 0,
        "git_status_hash_matches_validation": actual_gs["sha256"] == expected_gs.get("sha256", ""),
        "changes_patch_hash_matches_validation": actual_patch["sha256"] == expected_patch.get("sha256", ""),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "checks": checks,
        "expected_from_validation": {
            "git_status.txt": expected_gs,
            "changes.patch": expected_patch,
        },
        "actual_after_publish": {
            "git_status.txt": actual_gs,
            "changes.patch": actual_patch,
        },
        "captured_at_utc": _utc_now(),
    }
    _atomic_write_json(artifacts_dir / "post_publish_check.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate S-02 bundle with atomic publish guards.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--qa-preview-dir", default=str(QA_PREVIEW_DIR_DEFAULT))
    parser.add_argument("--verify-cmd", default="make verify")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    qa_preview_dir = Path(args.qa_preview_dir).expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    meta = _write_trace_files(artifacts_dir, repo_root)
    preflight = _run_preflight(artifacts_dir, qa_preview_dir)
    verify = _run_verify(artifacts_dir, repo_root, args.verify_cmd)

    validation = validate_s02_trace(artifacts_dir)
    _atomic_write_json(artifacts_dir / "s02_trace_validation.json", validation)
    _atomic_write_text(
        artifacts_dir / "evidence_validation.log",
        (
            f"TIMESTAMP={_utc_now()}\n"
            f"ARTIFACTS_DIR={artifacts_dir}\n"
            f"VALIDATION_STATUS={validation.get('status')}\n"
            f"ERRORS={json.dumps(validation.get('errors', []), ensure_ascii=False)}\n"
            f"EXIT_CODE={0 if validation.get('status') == 'PASS' else 1}\n"
        ),
    )

    post_publish = _post_publish_check(artifacts_dir, validation)

    summary = {
        "status": "PASS"
        if (
            preflight["status"] == "PASS"
            and verify["status"] == "PASS"
            and validation.get("status") == "PASS"
            and post_publish["status"] == "PASS"
        )
        else "FAIL",
        "trace": meta,
        "preflight_report": str(artifacts_dir / "preflight_report.json"),
        "verify_log": str(artifacts_dir / "verify.log"),
        "trace_validation_report": str(artifacts_dir / "s02_trace_validation.json"),
        "post_publish_report": str(artifacts_dir / "post_publish_check.json"),
        "captured_at_utc": _utc_now(),
    }
    _atomic_write_json(artifacts_dir / "bundle_s02_summary.json", summary)

    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
