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
    mtime = float(path.stat().st_mtime) if exists else 0.0
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size,
        "sha256": _sha256(path) if size > 0 else "",
        "mtime_epoch": mtime,
        "captured_at_utc": _utc_now(),
    }


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _pre_execution_guard(
    artifacts_dir: Path,
    repo_root: Path,
    qa_preview_dir: Path,
    verify_cmd: str,
    expected_branch: str,
) -> dict[str, Any]:
    required_files = [
        repo_root / "tools" / "s02_bundle_atomic.py",
        repo_root / "tools" / "s02_trace_checker.py",
        repo_root / "tools" / "s02_independent_check.py",
        repo_root / "tools" / "security_check.py",
        repo_root / "Makefile",
    ]
    file_checks = {}
    errors: list[str] = []
    for p in required_files:
        ok = p.exists() and p.is_file()
        file_checks[str(p)] = ok
        if not ok:
            errors.append(f"missing required file: {p}")

    checks = {
        "artifacts_dir_absolute": artifacts_dir.is_absolute(),
        "artifacts_dir_writable": os.access(str(artifacts_dir), os.W_OK),
        "qa_preview_dir_absolute": qa_preview_dir.is_absolute(),
        "verify_cmd_non_empty": bool(verify_cmd.strip()),
        "expected_branch_non_empty": bool(expected_branch.strip()),
        "python_executable_exists": Path(sys.executable).exists(),
        "required_files_present": all(file_checks.values()),
    }
    if not checks["artifacts_dir_absolute"]:
        errors.append("artifacts_dir must be absolute")
    if not checks["artifacts_dir_writable"]:
        errors.append("artifacts_dir is not writable")
    if not checks["qa_preview_dir_absolute"]:
        errors.append("qa_preview_dir must be absolute")
    if not checks["verify_cmd_non_empty"]:
        errors.append("verify_cmd is empty")
    if not checks["expected_branch_non_empty"]:
        errors.append("expected_branch is empty (pass --expected-branch or ORDER_BRANCH)")
    if not checks["python_executable_exists"]:
        errors.append("python executable not found")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "required_files": file_checks,
        "errors": errors,
        "artifacts_dir": str(artifacts_dir),
        "qa_preview_dir": str(qa_preview_dir),
        "verify_cmd": verify_cmd,
        "expected_branch": expected_branch,
        "captured_at_utc": _utc_now(),
    }
    _atomic_write_json(artifacts_dir / "pre_execution_guard.json", report)
    _atomic_write_text(
        artifacts_dir / "pre_execution_guard.log",
        (
            f"TIMESTAMP={_utc_now()}\n"
            f"STATUS={report['status']}\n"
            f"ARTIFACTS_DIR={artifacts_dir}\n"
            f"QA_PREVIEW_DIR={qa_preview_dir}\n"
            f"VERIFY_CMD={verify_cmd}\n"
            f"EXPECTED_BRANCH={expected_branch}\n"
            f"ERRORS={json.dumps(errors, ensure_ascii=False)}\n"
            f"EXIT_CODE={0 if report['status'] == 'PASS' else 1}\n"
        ),
    )
    return report


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

    no_changes = not bool(diff.strip())
    if no_changes:
        patch_text = (
            "NO_DIFF=true\n"
            f"captured_at_utc={ts}\n"
            f"head_sha={sha}\n"
            f"branch={branch}\n"
            "justification=no tracked changes to include in patch for this run\n"
        )
    else:
        patch_text = diff.rstrip() + "\n"
    _atomic_write_text(artifacts_dir / "changes.patch", patch_text)

    return {"captured_at_utc": ts, "branch": branch, "head_sha": sha, "no_changes": no_changes}


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


def _post_publish_check(
    artifacts_dir: Path,
    validation: dict[str, Any],
    trace_meta: dict[str, Any],
    expected_branch: str,
) -> dict[str, Any]:
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
        "git_status_path_matches_validation": actual_gs["path"] == expected_gs.get("path", ""),
        "changes_patch_path_matches_validation": actual_patch["path"] == expected_patch.get("path", ""),
        "git_status_size_matches_validation": actual_gs["size_bytes"] == _to_int(expected_gs.get("size_bytes", -1)),
        "changes_patch_size_matches_validation": actual_patch["size_bytes"] == _to_int(expected_patch.get("size_bytes", -1)),
        "git_status_hash_matches_validation": actual_gs["sha256"] == expected_gs.get("sha256", ""),
        "changes_patch_hash_matches_validation": actual_patch["sha256"] == expected_patch.get("sha256", ""),
        "git_status_mtime_matches_validation": actual_gs["mtime_epoch"]
        == _to_float(expected_gs.get("mtime_epoch", -1.0)),
        "changes_patch_mtime_matches_validation": actual_patch["mtime_epoch"]
        == _to_float(expected_patch.get("mtime_epoch", -1.0)),
        "validation_artifact_dir_matches_bundle_dir": validation.get("artifacts_dir", "") == str(artifacts_dir),
        "branch_matches_expected": trace_meta.get("branch", "") == expected_branch,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "checks": checks,
        "branch_provenance": {
            "expected_branch": expected_branch,
            "trace_branch": trace_meta.get("branch", ""),
            "head_sha": trace_meta.get("head_sha", ""),
        },
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


def _run_independent_check(artifacts_dir: Path, repo_root: Path, expected_branch: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "s02_independent_check.py"),
        "--artifacts-dir",
        str(artifacts_dir),
        "--expected-branch",
        expected_branch,
    ]
    code, output = _run(cmd, cwd=repo_root)
    _atomic_write_text(
        artifacts_dir / "s02_independent_check.log",
        (
            f"TIMESTAMP={_utc_now()}\n"
            f"COMMAND={' '.join(cmd)}\n"
            f"EXIT_CODE={code}\n"
            "----- OUTPUT BEGIN -----\n"
            f"{output.rstrip()}\n"
            "----- OUTPUT END -----\n"
        ),
    )
    result_path = artifacts_dir / "s02_independent_check.json"
    result: dict[str, Any] = {"status": "FAIL", "exit_code": 1}
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {"status": "FAIL", "exit_code": 1, "error": "invalid_json"}
    return {
        "status": "PASS" if code == 0 and result.get("status") == "PASS" else "FAIL",
        "exit_code": code,
        "report_path": str(result_path),
        "log_path": str(artifacts_dir / "s02_independent_check.log"),
        "result": result,
    }


def _build_final_manifest(
    artifacts_dir: Path,
    *,
    ticket_id: str,
    expected_branch: str,
    reported_branch: str,
    files: list[str],
) -> dict[str, Any]:
    file_entries: dict[str, Any] = {}
    for name in files:
        file_entries[name] = _snapshot(artifacts_dir / name)
    payload: dict[str, Any] = {
        "artifacts_dir": str(artifacts_dir),
        "ticket_id": ticket_id,
        "order_branch_expected": expected_branch,
        "order_branch_reported": reported_branch,
        "generated_at_utc": _utc_now(),
        "files": file_entries,
        "status": "SEALED",
    }
    signing_blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(signing_blob).hexdigest()
    payload["signature"] = payload["manifest_sha256"]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate S-02 bundle with atomic publish guards.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--qa-preview-dir", default=str(QA_PREVIEW_DIR_DEFAULT))
    parser.add_argument("--verify-cmd", default="make verify")
    parser.add_argument("--ticket-id", default=(os.environ.get("TICKET_ID", "") or "").strip())
    parser.add_argument(
        "--expected-branch",
        default=(os.environ.get("ORDER_BRANCH", "") or "").strip(),
        help="Expected branch provenance for this bundle (defaults to ORDER_BRANCH env var).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    qa_preview_dir = Path(args.qa_preview_dir).expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    expected_branch = args.expected_branch.strip()
    pre_exec_guard = _pre_execution_guard(artifacts_dir, repo_root, qa_preview_dir, args.verify_cmd, expected_branch)
    if pre_exec_guard["status"] != "PASS":
        _atomic_write_json(
            artifacts_dir / "bundle_s02_summary.json",
            {
                "status": "FAIL",
                "reason": "pre_execution_guard_failed",
                "pre_execution_guard_report": str(artifacts_dir / "pre_execution_guard.json"),
                "captured_at_utc": _utc_now(),
            },
        )
        return 1

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

    post_publish = _post_publish_check(artifacts_dir, validation, meta, expected_branch)
    independent = _run_independent_check(artifacts_dir, repo_root, expected_branch)

    final_manifest = _build_final_manifest(
        artifacts_dir,
        ticket_id=args.ticket_id.strip() or "unknown",
        expected_branch=expected_branch,
        reported_branch=meta.get("branch", ""),
        files=[
            "git_status.txt",
            "changes.patch",
            "s02_trace_validation.json",
            "post_publish_check.json",
            "evidence_validation.log",
            "verify.log",
            "preflight_report.json",
            "s02_independent_check.json",
            "s02_independent_check.log",
        ],
    )
    _atomic_write_json(artifacts_dir / "s02_final_manifest_signed.json", final_manifest)

    base_pass = (
        pre_exec_guard["status"] == "PASS"
        and preflight["status"] == "PASS"
        and verify["status"] == "PASS"
        and validation.get("status") == "PASS"
        and post_publish["status"] == "PASS"
        and independent["status"] == "PASS"
    )
    # Contract rule: when there are no tracked changes, S-02 cannot be marked PASS.
    if meta["no_changes"]:
        status = "NO_CHANGES"
        exit_code = 2
    else:
        status = "PASS" if base_pass else "FAIL"
        exit_code = 0 if status == "PASS" else 1

    summary = {
        "status": status,
        "trace": meta,
        "expected_branch": expected_branch,
        "ticket_id": args.ticket_id.strip() or "unknown",
        "preflight_report": str(artifacts_dir / "preflight_report.json"),
        "pre_execution_guard_report": str(artifacts_dir / "pre_execution_guard.json"),
        "pre_execution_guard_log": str(artifacts_dir / "pre_execution_guard.log"),
        "verify_log": str(artifacts_dir / "verify.log"),
        "trace_validation_report": str(artifacts_dir / "s02_trace_validation.json"),
        "post_publish_report": str(artifacts_dir / "post_publish_check.json"),
        "independent_check_report": independent["report_path"],
        "independent_check_log": independent["log_path"],
        "final_manifest_signed": str(artifacts_dir / "s02_final_manifest_signed.json"),
        "captured_at_utc": _utc_now(),
    }
    _atomic_write_json(artifacts_dir / "bundle_s02_summary.json", summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
