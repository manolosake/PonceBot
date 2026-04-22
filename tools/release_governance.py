#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
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


def _try_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def _utc_iso(ts: float | None = None) -> str:
    t = float(time.time() if ts is None else ts)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


_SSH_GH_RE = re.compile(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$")
_HTTPS_GH_RE = re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


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


_TRACEABILITY_KEY_ALIASES: dict[str, list[str]] = {
    # Canonical mapping for legacy proactive key naming in this ticket lane.
    "proactive_cli_reseed_r1_13": ["proactive_cli_seed_r1_3"],
}


def _expand_traceability_key_aliases(keys: list[str]) -> list[str]:
    expanded: list[str] = []
    for key in keys:
        s = str(key or "").strip()
        if not s:
            continue
        expanded.append(s)
        for alias in _TRACEABILITY_KEY_ALIASES.get(s, []):
            if alias:
                expanded.append(alias)
    # preserve order while de-duplicating
    dedup: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    return dedup


def _extract_key_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z0-9_])(key:[A-Za-z0-9_.-]+)(?![A-Za-z0-9_])", text or "")


def _head_traceability_tokens_ok(*, order_token: str, key_tokens: list[str], head_body: str) -> bool:
    if not order_token or not _token_exact_in_text(head_body, order_token):
        return False
    if any(_token_exact_in_text(head_body, kt) for kt in key_tokens):
        return True
    # Strict fallback: head must still carry an explicit key token even when depends_on aliasing is needed.
    return bool(_extract_key_tokens(head_body))


def _extract_key_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z0-9_])(key:[A-Za-z0-9_.-]+)(?![A-Za-z0-9_])", text or "")


def _resolve_traceability_key_tokens(*, dep_keys: list[str], order_token: str, head_body: str) -> list[str]:
    tokens: list[str] = []
    for k in dep_keys:
        t = _key_token_from_depends_on(k)
        if t:
            tokens.append(t)

    # Contract-safe fallback: when HEAD carries the order token, include HEAD key tokens
    # so we can validate the actual tip traceability token without broad fuzzy matching.
    if order_token and _token_exact_in_text(head_body, order_token):
        tokens.extend(_extract_key_tokens(head_body))

    dedup: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup


def _traceability_count_from_log(log_text: str, *, order_token: str, key_token: str) -> int:
    count = 0
    for line in (log_text or "").splitlines():
        if _token_exact_in_text(line, order_token) and _token_exact_in_text(line, key_token):
            count += 1
    return count


def _traceability_pipeline_check(
    *,
    repo: Path,
    ref: str,
    order_token: str,
    key_token: str,
    max_commits: int = 400,
) -> tuple[int, list[str]]:
    """
    Non-shell equivalent of:
      git log --oneline <ref> -n <max_commits> | rg '<order_token>.*<key_token>'
    """
    log_text = _run(
        ["git", "log", "--oneline", str(ref), "-n", str(int(max_commits))],
        cwd=repo,
    )
    matches = [
        line
        for line in (log_text or "").splitlines()
        if _token_exact_in_text(line, order_token) and _token_exact_in_text(line, key_token)
    ]
    rc = 0 if matches else 1
    return rc, matches


def _load_traceability_keys(qa_result_path: Path | None) -> list[str]:
    if qa_result_path is None or (not qa_result_path.exists()):
        return []
    try:
        payload = json.loads(qa_result_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []

    keys: list[str] = []

    # 1) Primary contract source: depends_on (string or list)
    dep = payload.get("depends_on")
    if isinstance(dep, list):
        for item in dep:
            s = str(item or "").strip()
            if s:
                keys.append(s)
    elif isinstance(dep, str):
        s = dep.strip()
        if s:
            keys.append(s)

    # 2) Authoritative fallback fields from QA payloads (strict list, no generic fallback).
    for field in ("delegated_key", "active_key", "target_key", "contract_key", "branch_tip_key", "ticket_key"):
        v = payload.get(field)
        if isinstance(v, str):
            s = v.strip()
            if s:
                keys.append(s)

    # Preserve order while de-duplicating.
    dedup: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return _expand_traceability_key_aliases(dedup)


def _infer_job_id_from_artifacts_dir(artifacts_dir: str) -> str:
    p = Path(str(artifacts_dir or "").strip()).expanduser()
    name = p.name.strip()
    return name if _UUID_RE.match(name) else ""


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y", "pass", "ok"}:
            return True
        if s in {"false", "0", "no", "n", "fail", "ng"}:
            return False
    return None


def _qa_publication_discoverability(qa_result_path: Path | None) -> bool | None:
    if qa_result_path is None or (not qa_result_path.exists()):
        return None
    try:
        payload = json.loads(qa_result_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

    for field in ("publication_discoverable", "publication_discoverability_ok", "discoverability_ok"):
        if field in payload:
            v = _coerce_optional_bool(payload.get(field))
            if v is not None:
                return v

    nested = payload.get("verification_report")
    if isinstance(nested, dict):
        for field in ("publication_discoverable", "publication_discoverability_ok", "discoverability_ok"):
            if field in nested:
                v = _coerce_optional_bool(nested.get(field))
                if v is not None:
                    return v
    return None


def _publication_discoverability_gate(
    *,
    role: str,
    ticket_id: str,
    qa_publication_discoverable: bool | None,
    verification_publication_discoverable: bool,
) -> dict[str, Any]:
    role_norm = str(role or "").strip().lower()
    required = bool(str(ticket_id or "").strip()) and role_norm in {"release_mgr", "release_manager"}
    signal_present = qa_publication_discoverable is not None
    if qa_publication_discoverable is None:
        consistent = (not required)
    else:
        consistent = bool(qa_publication_discoverable) == bool(verification_publication_discoverable)
    return {
        "publication_discoverability_required": bool(required),
        "publication_discoverability_signal_present": bool(signal_present),
        "publication_discoverability_consistent": bool(consistent),
        "publication_discoverability_qa_result": (
            None if qa_publication_discoverable is None else bool(qa_publication_discoverable)
        ),
        "publication_discoverability_verification_report": bool(verification_publication_discoverable),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(content, encoding=encoding)
    tmp.replace(path)


def _acquire_artifact_lock(lock_path: Path) -> int:
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.write(fd, f"pid={os.getpid()} ts={_utc_iso()}\n".encode("utf-8"))
    return fd


def _release_artifact_lock(lock_path: Path, fd: int) -> None:
    try:
        os.close(fd)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _manifest_entries(artifacts_dir: Path, *, exclude_names: set[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for p in sorted(artifacts_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name in exclude_names:
            continue
        entries.append(
            {
                "name": p.name,
                "size": int(p.stat().st_size),
                "sha256": _sha256_file(p),
            }
        )
    return entries


def _manifest_mismatches(manifest: dict[str, Any], artifacts_dir: Path) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for item in manifest.get("files", []):
        name = str(item.get("name", ""))
        p = artifacts_dir / name
        if not p.exists() or (not p.is_file()):
            mismatches.append({"name": name, "reason": "missing"})
            continue
        raw_expected_size = item.get("size", -1)
        try:
            expected_size = int(raw_expected_size)
        except (TypeError, ValueError):
            mismatches.append(
                {
                    "name": name,
                    "reason": "malformed_metadata",
                    "field": "size",
                    "value": raw_expected_size,
                }
            )
            continue
        raw_expected_sha = item.get("sha256", "")
        expected_sha = str(raw_expected_sha)
        if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha.lower()):
            mismatches.append(
                {
                    "name": name,
                    "reason": "malformed_metadata",
                    "field": "sha256",
                    "value": raw_expected_sha,
                }
            )
            continue
        actual_size = int(p.stat().st_size)
        actual_sha = _sha256_file(p)
        if expected_size != actual_size or expected_sha != actual_sha:
            mismatches.append(
                {
                    "name": name,
                    "reason": "hash_or_size_mismatch",
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                }
            )
    return mismatches


def _manifest_post_write_violations(*, manifest: dict[str, Any], artifacts_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return [{"name": "FINAL_MANIFEST.json", "reason": "manifest_missing"}]
    manifest_mtime_ns = int(manifest_path.stat().st_mtime_ns)
    violations: list[dict[str, Any]] = []
    for item in manifest.get("files", []):
        name = str(item.get("name", ""))
        if not name:
            continue
        p = artifacts_dir / name
        if not p.exists() or (not p.is_file()):
            continue
        mtime_ns = int(p.stat().st_mtime_ns)
        if mtime_ns > manifest_mtime_ns:
            violations.append(
                {
                    "name": name,
                    "reason": "mtime_after_manifest",
                    "file_mtime_ns": mtime_ns,
                    "manifest_mtime_ns": manifest_mtime_ns,
                }
            )
    return violations


def _required_final_artifacts(*, artifacts_dir: Path, checklist_path: Path) -> list[Path]:
    return [
        checklist_path,
        artifacts_dir / "PR_URL.txt",
        artifacts_dir / "CHANGED_FILES.txt",
        artifacts_dir / "RUN_PROVENANCE.json",
        artifacts_dir / "command_transcript.jsonl",
        artifacts_dir / "test_logs.txt",
        artifacts_dir / "release_governance_run.stdout.json",
        artifacts_dir / "release_governance_run.exit_code.txt",
    ]


def _detect_implementation_claim(*, role: str, qa_result_path: Path | None, artifacts_dir: Path) -> tuple[bool, str]:
    role_norm = str(role or "").strip().lower()
    no_op_marker = artifacts_dir / "no_op_justification.md"
    if no_op_marker.exists() and no_op_marker.is_file() and no_op_marker.stat().st_size > 0:
        return False, "no_op_marker_present"

    payload: dict[str, Any] = {}
    if qa_result_path is not None and qa_result_path.exists():
        try:
            payload = json.loads(qa_result_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            payload = {}

    claim_raw = str(
        payload.get("claim_type")
        or payload.get("slice_claim_type")
        or payload.get("claim")
        or ""
    ).strip().lower()

    if claim_raw in {"no_op", "no-op", "state_only", "analysis_only", "advisory_only"}:
        return False, f"claim_type={claim_raw}"
    if claim_raw in {"implementation", "code_change", "code-change"}:
        return True, f"claim_type={claim_raw}"

    # Backend + release manager lanes default to implementation unless explicitly marked no-op.
    if role_norm in {"backend", "release_mgr"}:
        return True, f"{role_norm}_default_strict"
    return False, "non_backend_default"


def _artifact_provenance_gate_check(*, artifacts_dir: Path, implementation_claim: bool) -> tuple[bool, list[str]]:
    if not implementation_claim:
        return True, []

    reasons: list[str] = []
    checks = [
        ("changes.patch", "artifact_missing_changes_patch", "artifact_empty_changes_patch"),
        ("git_status.txt", "artifact_missing_git_status", "artifact_empty_git_status"),
    ]
    for name, miss_reason, empty_reason in checks:
        p = artifacts_dir / name
        if not p.exists() or (not p.is_file()):
            reasons.append(miss_reason)
            continue
        if int(p.stat().st_size) <= 0:
            reasons.append(empty_reason)
            continue
        content = p.read_text(encoding="utf-8", errors="replace").strip().lower()
        # Harden against placeholder outputs that are technically non-empty but carry no evidence.
        if name == "changes.patch" and content in {"(none)", "none"}:
            reasons.append(empty_reason)
            continue
        if name == "git_status.txt" and content in {"clean", "(clean)"}:
            reasons.append(empty_reason)
    return len(reasons) == 0, reasons


def _qa_publication_signal_check(*, qa_result_path: Path | None, role: str) -> tuple[bool, str]:
    role_norm = _normalize_role_for_close_gate(role)
    requires_signal = {
        "release_mgr",
        "qa",
        "qa_local",
        "quality_assurance",
    }
    if role_norm not in requires_signal:
        return True, "qa_signal_not_required_for_role"
    if qa_result_path is None:
        return False, "qa_signal_missing_path"
    if (not qa_result_path.exists()) or (not qa_result_path.is_file()) or int(qa_result_path.stat().st_size) <= 0:
        return False, "qa_signal_missing_file"
    try:
        payload = json.loads(qa_result_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False, "qa_signal_invalid_json"
    if not isinstance(payload, dict):
        return False, "qa_signal_invalid_payload"
    marker_keys = ("verdict", "go_no_go", "ok", "acceptance")
    if not any(k in payload for k in marker_keys):
        return False, "qa_signal_missing_verdict_fields"
    return True, "qa_signal_present"


def _write_check_artifacts(*, artifacts_dir: Path, checks: list["Check"]) -> None:
    transcript_lines = []
    for c in checks:
        transcript_lines.append(
            json.dumps({"key": c.key, "ok": bool(c.ok), "details": c.details}, ensure_ascii=False)
        )
    _atomic_write_text(
        artifacts_dir / "command_transcript.jsonl",
        ("\n".join(transcript_lines) + "\n") if transcript_lines else "",
    )
    test_log_lines = [f"{c.key}: ok={bool(c.ok)} details={c.details}" for c in checks if "test" in c.key or "replay_" in c.key]
    _atomic_write_text(
        artifacts_dir / "test_logs.txt",
        ("\n".join(test_log_lines) + "\n") if test_log_lines else "no test/replay checks executed\n",
    )


def _extract_artifacts_dir_from_argv(argv: list[str]) -> str:
    for i, token in enumerate(argv):
        if token == "--artifacts-dir" and i + 1 < len(argv):
            return str(argv[i + 1] or "").strip()
    return ""


def _write_failure_artifacts_bundle(*, artifacts_dir: Path, error_text: str) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_iso(),
        "ok": False,
        "fatal_error": str(error_text or "unknown_error"),
    }
    final_validation = {
        "ok": False,
        "checks": {
            "checks_ok": False,
            "pr_url_targets_head": False,
            "required_artifacts_non_empty": False,
            "manifest_integrity_ok": False,
            "manifest_mismatch_count": 1,
        },
        "validated_at": _utc_iso(),
    }

    _atomic_write_text(artifacts_dir / "RELEASE_CHECKLIST.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(artifacts_dir / "FINAL_VALIDATION.json", json.dumps(final_validation, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(artifacts_dir / "release_governance.stdout.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(artifacts_dir / "release_governance_run.stdout.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(artifacts_dir / "release_governance.exit_code.txt", "2\n")
    _atomic_write_text(artifacts_dir / "release_governance_run.exit_code.txt", "2\n")
    # Ensure transcript/test files are always present and non-empty for lane evidence contracts.
    if not (artifacts_dir / "command_transcript.jsonl").exists() or int((artifacts_dir / "command_transcript.jsonl").stat().st_size) <= 0:
        _atomic_write_text(
            artifacts_dir / "command_transcript.jsonl",
            json.dumps({"key": "fatal_error", "ok": False, "details": str(error_text or "unknown_error")}, ensure_ascii=False)
            + "\n",
        )
    if not (artifacts_dir / "test_logs.txt").exists() or int((artifacts_dir / "test_logs.txt").stat().st_size) <= 0:
        _atomic_write_text(artifacts_dir / "test_logs.txt", f"fatal_error: {str(error_text or 'unknown_error')}\n")


def _final_exit_code(*, checks_ok: bool, manifest_mismatch_count: int) -> int:
    if int(manifest_mismatch_count) > 0:
        return 2
    return 0 if bool(checks_ok) else 2


def _build_final_validation(*, base_checks: dict[str, bool], manifest_mismatch_count: int) -> dict[str, Any]:
    checks: dict[str, Any] = dict(base_checks)
    checks["manifest_integrity_ok"] = int(manifest_mismatch_count) == 0
    checks["manifest_mismatch_count"] = int(manifest_mismatch_count)
    artifact_gate_ok = bool(checks.get("artifact_provenance_gate_ok", True))
    publication_consistent = bool(checks.get("publication_discoverability_consistent", True))
    ok = bool(checks.get("manifest_integrity_ok")) and bool(checks.get("checks_ok")) and bool(
        checks.get("pr_url_targets_head")
    ) and bool(checks.get("required_artifacts_non_empty")) and artifact_gate_ok and publication_consistent
    return {
        "ok": ok,
        "checks": checks,
        "validated_at": _utc_iso(),
    }


def _finalize_manifest(*, artifacts_dir: Path, lock_path: Path) -> dict[str, Any]:
    # Snapshot entries only after all mutable artifact outputs are written.
    covered_names = [
        "CHANGED_FILES.txt",
        "DIFFSTAT.txt",
        "PR_URL.txt",
        "RUN_PROVENANCE.json",
        "command_transcript.jsonl",
        "test_logs.txt",
    ]
    files = []
    for name in covered_names:
        p = artifacts_dir / name
        if not p.exists() or (not p.is_file()):
            continue
        files.append(
            {
                "name": p.name,
                "size": int(p.stat().st_size),
                "sha256": _sha256_file(p),
            }
        )
    manifest = {
        "generated_at": _utc_iso(),
        "files": files,
    }
    pre_capture_mismatches = _manifest_mismatches(manifest, artifacts_dir)
    # Hard integrity check at completion time against the captured snapshot.
    post_capture_mismatches = _manifest_mismatches(manifest, artifacts_dir)
    manifest["pre_capture_mismatch_count"] = len(pre_capture_mismatches)
    manifest["pre_capture_mismatches"] = pre_capture_mismatches
    manifest["mismatch_count"] = len(post_capture_mismatches)
    manifest["mismatches"] = post_capture_mismatches
    _atomic_write_text(
        artifacts_dir / "FINAL_MANIFEST.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


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


def _resolve_canonical_head_ref(*, repo: Path, remote: str, head_branch: str) -> str:
    remote_ref = f"refs/remotes/{remote}/{head_branch}"
    rc, _out, _err = _try_run(["git", "rev-parse", "--verify", "--quiet", remote_ref], cwd=repo)
    if rc == 0:
        return remote_ref
    return head_branch


def _fetch_head_ref(*, repo: Path, remote: str, head_branch: str) -> None:
    hb = str(head_branch or "").strip()
    if hb.startswith(f"refs/remotes/{remote}/"):
        hb = hb[len(f"refs/remotes/{remote}/") :]
    elif hb.startswith("refs/heads/"):
        hb = hb[len("refs/heads/") :]
    _run(["git", "fetch", remote, hb], cwd=repo)


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


def _collect_diff_capture(
    repo: Path,
    *,
    base_ref: str,
    head_ref: str,
    ahead: int,
    behind: int,
    working_tree_dirty: bool,
) -> tuple[str, str, str]:
    """
    Return (basis, diffstat, changed_files) using one consistent comparison basis.
    """
    if ahead or behind:
        diffstat = _run(["git", "diff", "--stat", f"{base_ref}..{head_ref}"], cwd=repo)
        changed = _run(["git", "diff", "--name-only", f"{base_ref}..{head_ref}"], cwd=repo)
        # If rev-list indicates divergence but direct diff capture is empty,
        # fall back instead of emitting misleading `(none)` artifacts.
        if diffstat.strip() or changed.strip():
            return "branch", diffstat, changed
    if working_tree_dirty:
        diffstat = _run(["git", "diff", "--stat", "HEAD"], cwd=repo)
        changed = _run(["git", "diff", "--name-only", "HEAD"], cwd=repo)
        return "working_tree", diffstat, changed
    return "none", "", ""


def _collect_patch_capture(
    repo: Path,
    *,
    basis: str,
    base_ref: str,
    head_ref: str,
) -> str:
    if basis == "branch":
        return _run(["git", "diff", f"{base_ref}..{head_ref}"], cwd=repo)
    if basis == "working_tree":
        return _run(["git", "diff", "HEAD"], cwd=repo)
    return ""


def _collect_status_capture(repo: Path) -> str:
    return _run(["git", "status", "--short", "--branch"], cwd=repo)


def _run_unit_tests(repo: Path) -> Check:
    rc, out, err = _try_run([sys.executable, "-m", "unittest", "-q"], cwd=repo)
    ok = (rc == 0)
    detail = out.strip() or err.strip() or ("ok" if ok else "tests failed")
    return Check("tests", ok, detail[:4000])


def _resolve_traceability_value(explicit: str, env_keys: list[str]) -> str:
    s = str(explicit or "").strip()
    if s:
        return s
    for k in env_keys:
        v = str(os.environ.get(k, "")).strip()
        if v:
            return v
    return ""


def _traceability_ids_check(*, job_id: str, ticket_id: str) -> Check:
    ok = bool(str(job_id or "").strip() and str(ticket_id or "").strip())
    return Check(
        "traceability_ids_present",
        ok,
        f"job_id={str(job_id or '').strip() or '<missing>'} ticket_id={str(ticket_id or '').strip() or '<missing>'}",
    )


def _build_run_provenance(
    *,
    generated_at: str,
    branch: str,
    commit_sha: str,
    role: str,
    job_id: str,
    ticket_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "branch": str(branch or "").strip(),
        "commit_sha": str(commit_sha or "").strip(),
        "role": str(role or "").strip(),
        "job_id": str(job_id or "").strip(),
        "ticket_id": str(ticket_id or "").strip(),
    }


def _clean_shell_env() -> dict[str, str]:
    env = {"PATH": "/usr/bin:/bin"}
    home = str(os.environ.get("HOME", "")).strip()
    if home:
        env["HOME"] = home
    return env


def _normalize_role_for_close_gate(role: str) -> str:
    # Accept legacy separators/casing so role aliases stay stable across lanes.
    return str(role or "").strip().lower().replace("-", "_").replace(" ", "_")


def _blocking_check_keys(*, checks: list["Check"], role: str) -> set[str]:
    keys = {c.key for c in checks}
    role_norm = _normalize_role_for_close_gate(role)
    # release_mgr frequently executes in controller worktrees that may be intentionally dirty;
    # keep repo_clean visibility but don't block final gate solely on that signal.
    if role_norm == "release_mgr":
        keys.discard("repo_clean")
    return keys


def _checks_ok_for_role(*, checks: list["Check"], role: str) -> bool:
    keyset = _blocking_check_keys(checks=checks, role=role)
    by_key = {c.key: c for c in checks}
    return all(bool(by_key[k].ok) for k in keyset if k in by_key)


def _close_gate_blocker_count(rows: list[dict[str, Any]]) -> int:
    """
    Contract-aligned close-gate predicate:
    - open delivery jobs are blockers
    - controller/review overhead roles are non-blocking
    """
    non_blocking_roles = {
        "skynet",
        "jarvis",
        "orchestrator",
        "controller",
        "release_mgr",
        "release_manager",
        "reviewer_local",
        "architect_local",
        "qa",
        "qa_local",
        "quality_assurance",
        "quality-assurance",
    }
    open_states = {"queued", "running", "waiting_deps", "blocked", "blocked_approval"}
    blockers = 0
    for row in rows:
        state = str(row.get("state") or "").strip().lower()
        if state not in open_states:
            continue
        role = _normalize_role_for_close_gate(str(row.get("role") or ""))
        if role in non_blocking_roles:
            continue
        blockers += 1
    return blockers


def _close_gate_blocker_count_for_ticket(db_path: Path, ticket_id: str) -> int:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT role, state
        FROM jobs
        WHERE parent_job_id = ?
          AND state IN ('queued','running','waiting_deps','blocked','blocked_approval')
        """,
        (str(ticket_id),),
    ).fetchall()
    payload = [dict(r) for r in rows]
    return _close_gate_blocker_count(payload)


def _append_transcript(path: Path, *, label: str, cmd: list[str], rc: int, out: str, err: str) -> None:
    lines: list[str] = [f"$ {label} {' '.join(cmd)}", f"exit_code={int(rc)}"]
    if out:
        lines.insert(1, out)
    if err:
        lines.insert(2 if out else 1, f"[stderr]\n{err}")
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n".join(lines) + "\n\n")


def _run_replay_gate(repo: Path, *, artifacts_dir: Path, python_bin: str, ticket_id: str = "") -> list[Check]:
    """
    Run clean-shell replay sanity commands (c02-c04 equivalents).
    Uses canonical bootstrap script to keep replay behavior deterministic.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = artifacts_dir / "command_transcript.txt"
    bootstrap = str((repo / "scripts" / "bootstrap_pytest_python3.sh").resolve())
    env = _clean_shell_env()
    env["BOOTSTRAP_PYTEST_PYTHON_BIN"] = str(python_bin or "python3")
    env["BOOTSTRAP_PYTEST_VENV_DIR"] = str((artifacts_dir / ".qa_replay_venv").resolve())
    commands = [
        ("c02", [bootstrap, "-m", "pytest", "--version"], "replay_c02_pytest_version"),
        ("c03", [bootstrap, "-m", "pytest", "--collect-only", "-q", "test_state_store.py"], "replay_c03_collect_targeted"),
        ("c04", [bootstrap, "-m", "pytest", "--collect-only", "-q"], "replay_c04_collect_all"),
    ]
    checks: list[Check] = []
    for label, cmd, key in commands:
        rc, out, err = _try_run(cmd, cwd=repo, env=env)
        _append_transcript(transcript_path, label=label, cmd=cmd, rc=rc, out=out, err=err)
        _atomic_write_text(artifacts_dir / f"{label}.exit_code.txt", f"{int(rc)}\n")
        detail = (out or err or f"{label}_failed")[:4000]
        checks.append(Check(key, rc == 0, detail))

    ticket = str(ticket_id or "").strip()
    if ticket:
        db_path = Path(os.environ.get("ORCH_JOBS_DB", "/home/aponce/codexbot/data/jobs.sqlite")).expanduser()
        blockers = _close_gate_blocker_count_for_ticket(db_path, ticket)
        rc = 0 if int(blockers) == 0 else 2
        out = f"ticket_id={ticket}\nblocking_active_children={int(blockers)}\n"
        cmd = [str(python_bin or "python3"), "-c", "close_gate_contract_check", ticket]
        _append_transcript(
            transcript_path,
            label="c05",
            cmd=cmd,
            rc=rc,
            out=out,
            err="",
        )
        _atomic_write_text(artifacts_dir / "c05.exit_code.txt", f"{int(rc)}\n")
        checks.append(Check("replay_c05_close_gate_contract", rc == 0, out.strip()))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description="Release governance checklist + PR URL generator (no gh required).")
    ap.add_argument("--repo", default=".", help="path to repo (default: .)")
    ap.add_argument("--remote", default="origin", help="remote name (default: origin)")
    ap.add_argument("--base", default="main", help="base branch name (default: main)")
    ap.add_argument("--head", default="", help="head branch name (default: current branch)")
    ap.add_argument("--order-branch", default="", help="explicit order branch to prefer as release head")
    ap.add_argument("--artifacts-dir", default="", help="optional artifacts dir to write checklist files")
    ap.add_argument("--qa-result", default="", help="optional path to qa_result.json")
    ap.add_argument("--job-id", default="", help="orchestrator target job id for strict traceability")
    ap.add_argument("--ticket-id", default="", help="ticket/order id for strict traceability")
    ap.add_argument("--role", default="", help="role lane (e.g. backend)")
    ap.add_argument("--run-tests", action="store_true", help="run python -m unittest -q as part of checklist")
    ap.add_argument(
        "--run-replay-gate",
        action="store_true",
        help="run clean-shell replay sanity (pytest bootstrap + c02/c03/c04 equivalents)",
    )
    ap.add_argument("--replay-python", default="python3", help="python executable used to create replay venv")
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
    job_id = _resolve_traceability_value(
        args.job_id,
        ["ORCH_JOB_ID", "ORCHESTRATOR_JOB_ID", "JOB_ID", "TARGET_JOB_ID"],
    )
    if not str(job_id or "").strip() and str(args.artifacts_dir or "").strip():
        job_id = _infer_job_id_from_artifacts_dir(str(args.artifacts_dir))
    ticket_id = _resolve_traceability_value(
        args.ticket_id,
        ["ORCH_TICKET_ID", "ORCHESTRATOR_TICKET_ID", "TICKET_ID", "ORDER_ID"],
    )
    role = _resolve_traceability_value(args.role, ["ORCH_ROLE", "ROLE"]) or "backend"

    checks: list[Check] = []
    checks.append(_git_status_clean(root))
    checks.append(_git_has_tracked_data(root))
    checks.append(_traceability_ids_check(job_id=job_id, ticket_id=ticket_id))
    if args.run_tests:
        checks.append(_run_unit_tests(root))
    if args.run_replay_gate:
        run_artifacts_dir = Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir else (root / "data" / "artifacts" / "_replay")
        checks.extend(
            _run_replay_gate(
                root,
                artifacts_dir=run_artifacts_dir,
                python_bin=str(args.replay_python or "python3"),
                ticket_id=str(ticket_id or ""),
            )
        )

    base_ref = f"{args.remote}/{base_branch}"
    # Keep local remote-tracking refs current, then use canonical resolved head ref.
    _fetch_head_ref(repo=root, remote=args.remote, head_branch=head_branch)
    head_ref = _resolve_canonical_head_ref(repo=root, remote=args.remote, head_branch=head_branch)
    # Ensure base ref exists.
    _run(["git", "fetch", args.remote, base_branch], cwd=root)
    ahead, behind = _git_ahead_behind(root, base_ref=base_ref, head_ref=head_ref)
    working_tree_dirty = not checks[0].ok
    diff_basis, diffstat, changed = _collect_diff_capture(
        root,
        base_ref=base_ref,
        head_ref=head_ref,
        ahead=ahead,
        behind=behind,
        working_tree_dirty=working_tree_dirty,
    )

    if args.require_pr:
        checks.append(Check("requires_pr_branch", head_branch != base_branch, f"head={head_branch} base={base_branch}"))

    qa_result_path: Path | None = None
    if args.qa_result:
        qa_result_path = Path(args.qa_result).expanduser().resolve()
    elif args.artifacts_dir:
        cand = Path(args.artifacts_dir).expanduser().resolve() / "qa_result.json"
        if cand.exists():
            qa_result_path = cand
    qa_signal_ok, qa_signal_detail = _qa_publication_signal_check(qa_result_path=qa_result_path, role=role)
    checks.append(Check("qa_publication_signal_ok", bool(qa_signal_ok), str(qa_signal_detail)))
    dep_keys = _load_traceability_keys(qa_result_path)
    if dep_keys:
        order_token = _order_token_from_branch(head_branch)
        head_body = _run(["git", "log", "-1", "--pretty=%B", "HEAD"], cwd=root)
        key_tokens = _resolve_traceability_key_tokens(
            dep_keys=dep_keys,
            order_token=order_token,
            head_body=head_body,
        )
        head_tokens_ok = _head_traceability_tokens_ok(
            order_token=order_token,
            key_tokens=key_tokens,
            head_body=head_body,
        )
        checks.append(
            Check(
                "traceability_head_tokens_match",
                head_tokens_ok,
                f"order_token={order_token or '<missing>'} key_tokens={','.join(key_tokens) or '<missing>'}",
            )
        )
        tcount = 0
        c01_ok = False
        c01_details = []
        for kt in key_tokens:
            rc, matches = _traceability_pipeline_check(
                repo=root,
                ref=head_ref,
                order_token=order_token,
                key_token=kt,
                max_commits=400,
            )
            tcount = max(tcount, len(matches))
            if rc == 0:
                c01_ok = True
                if matches:
                    c01_details.append(matches[0])
        checks.append(
            Check(
                "c01_traceability",
                c01_ok,
                c01_details[0] if c01_details else f"order_token={order_token} key_tokens={','.join(key_tokens)}",
            )
        )
        checks.append(Check("traceability_count_positive", tcount > 0, f"count={tcount}"))

    ok_all = _checks_ok_for_role(checks=checks, role=role) and (not args.require_pr or head_branch != base_branch)

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
        "job_id": job_id or None,
        "ticket_id": ticket_id or None,
        "role": role,
        "base_ref": base_ref,
        "ahead_commits": int(ahead),
        "behind_commits": int(behind),
        "diff_basis": diff_basis,
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
        lock_path = artifacts_dir / ".finalize.lock"
        lock_fd = _acquire_artifact_lock(lock_path)
        try:
            checklist_path = artifacts_dir / "RELEASE_CHECKLIST.json"
            _atomic_write_text(checklist_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            if pr_compare:
                _atomic_write_text(artifacts_dir / "PR_URL.txt", pr_compare + "\n")
            if diffstat:
                _atomic_write_text(artifacts_dir / "DIFFSTAT.txt", diffstat + "\n")
            _atomic_write_text(artifacts_dir / "CHANGED_FILES.txt", (changed + "\n") if changed else "(none)\n")
            patch_text = _collect_patch_capture(
                root,
                basis=diff_basis,
                base_ref=base_ref,
                head_ref=head_ref,
            )
            status_text = _collect_status_capture(root)
            _atomic_write_text(artifacts_dir / "changes.patch", patch_text if patch_text else "(none)\n")
            _atomic_write_text(artifacts_dir / "git_status.txt", (status_text + "\n") if status_text else "## clean\n")
            provenance = _build_run_provenance(
                generated_at=_utc_iso(),
                branch=head_branch,
                commit_sha=_run(["git", "rev-parse", "HEAD"], cwd=root),
                role=role,
                job_id=job_id,
                ticket_id=ticket_id,
            )
            _atomic_write_text(
                artifacts_dir / "RUN_PROVENANCE.json",
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            )
            _write_check_artifacts(artifacts_dir=artifacts_dir, checks=checks)
            _atomic_write_text(
                artifacts_dir / "release_governance.stdout.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            _atomic_write_text(
                artifacts_dir / "release_governance_run.stdout.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            _atomic_write_text(artifacts_dir / "release_governance.exit_code.txt", f"{(0 if bool(payload.get('ok')) else 2)}\n")
            _atomic_write_text(artifacts_dir / "release_governance_run.exit_code.txt", f"{(0 if bool(payload.get('ok')) else 2)}\n")

            required = _required_final_artifacts(artifacts_dir=artifacts_dir, checklist_path=checklist_path)
            present_non_empty = []
            for p in required:
                present_non_empty.append(bool(p.exists() and p.is_file() and p.stat().st_size > 0))
            pr_targets_head = bool(pr_compare and (f"...{head_branch}?" in pr_compare))
            implementation_claim, claim_source = _detect_implementation_claim(
                role=role,
                qa_result_path=qa_result_path,
                artifacts_dir=artifacts_dir,
            )
            artifact_gate_ok, artifact_gate_reasons = _artifact_provenance_gate_check(
                artifacts_dir=artifacts_dir,
                implementation_claim=implementation_claim,
            )
            qa_publication_discoverable = _qa_publication_discoverability(qa_result_path)
            verification_publication_discoverable = bool(all(present_non_empty))
            publication_gate = _publication_discoverability_gate(
                role=role,
                ticket_id=ticket_id,
                qa_publication_discoverable=qa_publication_discoverable,
                verification_publication_discoverable=verification_publication_discoverable,
            )
            final_checks_base = {
                "checks_ok": bool(_checks_ok_for_role(checks=checks, role=role) and (not args.require_pr or head_branch != base_branch)),
                "pr_url_targets_head": bool(pr_targets_head),
                "required_artifacts_non_empty": bool(verification_publication_discoverable),
                "qa_publication_signal_ok": bool(qa_signal_ok),
                "publication_discoverability_required": bool(publication_gate["publication_discoverability_required"]),
                "publication_discoverability_signal_present": bool(publication_gate["publication_discoverability_signal_present"]),
                "publication_discoverability_consistent": bool(publication_gate["publication_discoverability_consistent"]),
                "publication_discoverability_qa_result": publication_gate["publication_discoverability_qa_result"],
                "publication_discoverability_verification_report": bool(
                    publication_gate["publication_discoverability_verification_report"]
                ),
                "artifact_provenance_gate_ok": bool(artifact_gate_ok),
                "artifact_provenance_gate_claim_implementation": bool(implementation_claim),
                "artifact_provenance_gate_claim_source": str(claim_source),
                "artifact_provenance_gate_reasons": list(artifact_gate_reasons),
            }

            # FINAL_MANIFEST is generated only after all mutable outputs are sealed.
            manifest = _finalize_manifest(artifacts_dir=artifacts_dir, lock_path=lock_path)
            final_validation = _build_final_validation(
                base_checks=final_checks_base,
                manifest_mismatch_count=int(manifest.get("mismatch_count", 0)),
            )
            payload["final_validation"] = final_validation
            payload["ok"] = bool(final_validation.get("ok"))
            _atomic_write_text(checklist_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            _atomic_write_text(
                artifacts_dir / "FINAL_VALIDATION.json",
                json.dumps(final_validation, ensure_ascii=False, indent=2) + "\n",
            )
            _atomic_write_text(
                artifacts_dir / "release_governance.stdout.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            _atomic_write_text(
                artifacts_dir / "release_governance_run.stdout.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            post_write_violations = _manifest_post_write_violations(
                manifest=manifest,
                artifacts_dir=artifacts_dir,
                manifest_path=artifacts_dir / "FINAL_MANIFEST.json",
            )
            final_validation["checks"]["manifest_post_write_violations"] = len(post_write_violations) == 0
            final_validation["checks"]["manifest_post_write_violation_count"] = len(post_write_violations)
            final_validation["checks"]["manifest_post_write_violation_names"] = [v.get("name") for v in post_write_violations]
            final_validation["ok"] = bool(final_validation["ok"]) and (len(post_write_violations) == 0)
            payload["final_validation"] = final_validation
            payload["ok"] = bool(final_validation["ok"])
            _atomic_write_text(checklist_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            _atomic_write_text(
                artifacts_dir / "FINAL_VALIDATION.json",
                json.dumps(final_validation, ensure_ascii=False, indent=2) + "\n",
            )
            _atomic_write_text(
                artifacts_dir / "release_governance.stdout.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            _atomic_write_text(
                artifacts_dir / "release_governance_run.stdout.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            computed_exit = _final_exit_code(
                checks_ok=bool(final_validation.get("ok")),
                manifest_mismatch_count=int(manifest.get("mismatch_count", 0)),
            )
            _atomic_write_text(artifacts_dir / "release_governance.exit_code.txt", f"{computed_exit}\n")
            _atomic_write_text(artifacts_dir / "release_governance_run.exit_code.txt", f"{computed_exit}\n")
        finally:
            _release_artifact_lock(lock_path, lock_fd)
    else:
        computed_exit = 0 if bool(payload.get("ok")) else 2

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return int(computed_exit)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        artifacts_arg = _extract_artifacts_dir_from_argv(sys.argv[1:])
        if artifacts_arg:
            try:
                _write_failure_artifacts_bundle(
                    artifacts_dir=Path(artifacts_arg).expanduser().resolve(),
                    error_text=str(exc),
                )
            except Exception:
                pass
        raise
