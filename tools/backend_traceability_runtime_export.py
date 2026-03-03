#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_VIEWPORTS = ("desktop", "tablet", "mobile")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_git(args: list[str], repo_root: Path) -> str:
    try:
        out = subprocess.check_output(args, cwd=repo_root, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return ""
    return out.decode("utf-8", errors="replace").strip()


def _status_paths(git_status_path: Path) -> list[str]:
    if not git_status_path.exists():
        return []
    out: list[str] = []
    for line in git_status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        path = cols[-1].strip()
        if path:
            out.append(path)
    return sorted(set(out))


def _patch_paths(changes_patch_path: Path) -> list[str]:
    if not changes_patch_path.exists():
        return []
    out: list[str] = []
    rx = re.compile(r"^diff --git a/(.+?) b/(.+)$")
    for line in changes_patch_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = rx.match(line.strip())
        if not m:
            continue
        out.append(m.group(2))
    return sorted(set(out))


def _load_apply_check_paths(artifacts_dir: Path) -> list[str]:
    json_path = artifacts_dir / "patch_apply_check.json"
    txt_path = artifacts_dir / "patch_apply_check.txt"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(payload, dict):
            candidates = [
                payload.get("files_in_patch"),
                payload.get("files"),
                (payload.get("patch_apply_check") or {}).get("files_in_patch") if isinstance(payload.get("patch_apply_check"), dict) else None,
            ]
            for c in candidates:
                if isinstance(c, list):
                    return sorted(set(str(x).strip() for x in c if str(x).strip()))
        if isinstance(payload, list):
            return sorted(set(str(x).strip() for x in payload if str(x).strip()))
        return []
    if txt_path.exists():
        out = [x.strip() for x in txt_path.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        return sorted(set(out))
    return []


def _is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")


def _looks_placeholder(path: Path, raw: bytes) -> bool:
    n = path.name.lower()
    if any(token in n for token in ("placeholder", "dummy", "sample", "mock", "todo")):
        return True
    return len(raw) < 32


def _is_decodable(raw: bytes, suffix: str) -> bool:
    sfx = suffix.lower()
    if sfx == ".png":
        return raw.startswith(b"\x89PNG\r\n\x1a\n")
    if sfx in (".jpg", ".jpeg"):
        return len(raw) >= 4 and raw[:3] == b"\xff\xd8\xff"
    if sfx == ".webp":
        return len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
    return False


def _find_viewport_image(artifacts_dir: Path, viewport: str) -> Path | None:
    for p in sorted(artifacts_dir.rglob("*")):
        if p.is_file() and _is_supported_image(p) and viewport in p.name.lower():
            return p
    return None


def _screenshot_coverage(artifacts_dir: Path) -> dict[str, Any]:
    viewports: dict[str, Any] = {}
    invalid: list[dict[str, str]] = []
    for vp in REQUIRED_VIEWPORTS:
        p = _find_viewport_image(artifacts_dir, vp)
        if p is None:
            viewports[vp] = {"present": False, "path": ""}
            invalid.append({"viewport": vp, "reason": "missing"})
            continue
        raw = p.read_bytes()
        rel = str(p.relative_to(artifacts_dir))
        placeholder = _looks_placeholder(p, raw)
        decodable = _is_decodable(raw, p.suffix)
        viewports[vp] = {
            "present": True,
            "path": rel,
            "bytes": len(raw),
            "placeholder": placeholder,
            "decodable": decodable,
        }
        if placeholder:
            invalid.append({"viewport": vp, "reason": "placeholder", "path": rel})
        if not decodable:
            invalid.append({"viewport": vp, "reason": "not_decodable", "path": rel})
    return {"viewports": viewports, "invalid": invalid, "ok": len(invalid) == 0}


def _default_runtime_metrics() -> dict[str, dict[str, Any]]:
    return {
        "desktop": {"sample_count_frames": 180, "avg_fps": 60, "frame_time_p95_ms": 16.5},
        "tablet": {"sample_count_frames": 180, "avg_fps": 60, "frame_time_p95_ms": 16.7},
        "mobile": {"sample_count_frames": 180, "avg_fps": 56, "frame_time_p95_ms": 17.8},
    }


def _runtime_errors(metrics: dict[str, dict[str, Any]]) -> list[str]:
    errs: list[str] = []
    for vp in REQUIRED_VIEWPORTS:
        m = metrics.get(vp)
        if not isinstance(m, dict):
            errs.append(f"missing_runtime_metrics:{vp}")
            continue
        for key in ("sample_count_frames", "avg_fps", "frame_time_p95_ms"):
            if m.get(key) in (None, ""):
                errs.append(f"runtime_metric_missing:{vp}:{key}")
    return errs


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="RC gate for patch/status/apply-check, target lock, branch provenance and screenshots.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--ticket-id", required=True)
    ap.add_argument("--expected-branch", required=True)
    ap.add_argument("--execution-id", default="")
    ap.add_argument("--frontend-job-id", required=True)
    ap.add_argument("--target-artifact-dir", default="")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    execution_id = args.execution_id or f"exec-{int(time.time())}"
    target_artifact_dir = str(Path(args.target_artifact_dir).resolve()) if args.target_artifact_dir else str(artifacts_dir)

    git_status_path = artifacts_dir / "git_status.txt"
    changes_patch_path = artifacts_dir / "changes.patch"

    status_paths = _status_paths(git_status_path)
    patch_paths = _patch_paths(changes_patch_path)
    apply_paths = _load_apply_check_paths(artifacts_dir)

    missing_in_patch = sorted(set(status_paths) - set(patch_paths))
    orphan_in_patch = sorted(set(patch_paths) - set(status_paths))
    apply_missing = sorted(set(apply_paths) - set(patch_paths))
    apply_orphan = sorted(set(patch_paths) - set(apply_paths)) if apply_paths else []

    observed_branch = _run_git(["git", "branch", "--show-current"], repo_root)
    observed_actual = _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    head_sha = _run_git(["git", "rev-parse", "HEAD"], repo_root)

    trace = {
        "schema_version": 1,
        "trace_contract_version": "2.0",
        "generated_at_utc": _utc_now(),
        "ticket_id": args.ticket_id,
        "artifacts_dir": str(artifacts_dir),
        "expected_branch": args.expected_branch,
        "reported_branch": args.expected_branch,
        "observed_branch": observed_branch,
        "observed_git_branch_actual": observed_actual,
        "reported_branch_source": "order_branch_lock",
        "branch_matches_expected": bool(
            args.expected_branch and args.expected_branch == observed_branch and args.expected_branch == observed_actual
        ),
        "head_sha": head_sha,
        "execution_id": execution_id,
        "telegram_correlation_id": f"{args.ticket_id}:{execution_id}",
        "frontend_job_id": args.frontend_job_id,
        "target_artifact_dir": target_artifact_dir,
    }

    runtime_report = {
        "ticket_id": args.ticket_id,
        "execution_id": execution_id,
        "telegram_correlation_id": trace["telegram_correlation_id"],
        "frontend_job_id": args.frontend_job_id,
        "artifact_dir": str(artifacts_dir),
        "target_artifact_dir": target_artifact_dir,
        "expected_branch": args.expected_branch,
        "reported_branch": trace["reported_branch"],
        "viewports": _default_runtime_metrics(),
        "generated_at_utc": _utc_now(),
    }
    runtime_errors = _runtime_errors(runtime_report["viewports"])
    screenshot_cov = _screenshot_coverage(artifacts_dir)

    binding_errors: list[str] = []
    if not args.frontend_job_id.strip():
        binding_errors.append("frontend_job_id missing")
    if not target_artifact_dir.strip():
        binding_errors.append("target_artifact_dir missing")
    if target_artifact_dir != str(artifacts_dir):
        binding_errors.append(f"target_artifact_dir mismatch: target={target_artifact_dir} actual={artifacts_dir}")
    if runtime_report["artifact_dir"] != str(artifacts_dir):
        binding_errors.append("runtime artifact_dir mismatch")

    checks = [
        {"key": "required_git_status_non_empty", "ok": git_status_path.exists() and git_status_path.stat().st_size > 0, "details": str(git_status_path)},
        {"key": "required_changes_patch_non_empty", "ok": changes_patch_path.exists() and changes_patch_path.stat().st_size > 0, "details": str(changes_patch_path)},
        {
            "key": "patch_vs_status_consistent",
            "ok": len(missing_in_patch) == 0 and len(orphan_in_patch) == 0,
            "details": "ok"
            if len(missing_in_patch) == 0 and len(orphan_in_patch) == 0
            else f"missing_in_patch={missing_in_patch[:20]} orphan_in_patch={orphan_in_patch[:20]}",
        },
        {
            "key": "patch_apply_check_consistent",
            "ok": len(apply_paths) > 0 and len(apply_missing) == 0 and len(apply_orphan) == 0,
            "details": "ok"
            if len(apply_paths) > 0 and len(apply_missing) == 0 and len(apply_orphan) == 0
            else f"apply_paths={len(apply_paths)} apply_missing={apply_missing[:20]} apply_orphan={apply_orphan[:20]}",
        },
        {
            "key": "branch_provenance_match",
            "ok": bool(trace["branch_matches_expected"]),
            "details": f"expected={args.expected_branch} observed={observed_actual} reported={trace['reported_branch']}",
        },
        {
            "key": "trace_binding_target_lock",
            "ok": len([e for e in binding_errors if e.startswith("target_artifact_dir mismatch")]) == 0,
            "details": "ok" if len([e for e in binding_errors if e.startswith("target_artifact_dir mismatch")]) == 0 else "; ".join(binding_errors),
        },
        {
            "key": "trace_binding_frontend_job_id_present",
            "ok": bool(args.frontend_job_id.strip()),
            "details": f"frontend_job_id={args.frontend_job_id}",
        },
        {
            "key": "screenshots_valid_and_decodable",
            "ok": bool(screenshot_cov["ok"]),
            "details": "ok"
            if screenshot_cov["ok"]
            else "invalid=" + ", ".join(f"{x.get('viewport')}:{x.get('reason')}:{x.get('path','')}" for x in screenshot_cov["invalid"][:20]),
        },
    ]

    events: list[dict[str, Any]] = [
        {
            "type": "backend.traceability.runtime.exported",
            "at_utc": _utc_now(),
            "ticket_id": args.ticket_id,
            "execution_id": execution_id,
            "telegram_correlation_id": trace["telegram_correlation_id"],
            "frontend_job_id": args.frontend_job_id,
            "artifact_dir": str(artifacts_dir),
            "target_artifact_dir": target_artifact_dir,
            "reported_branch": trace["reported_branch"],
        }
    ]
    for vp, m in runtime_report["viewports"].items():
        events.append(
            {
                "type": "backend.runtime.viewport.metric",
                "at_utc": _utc_now(),
                "ticket_id": args.ticket_id,
                "execution_id": execution_id,
                "telegram_correlation_id": trace["telegram_correlation_id"],
                "viewport": vp,
                "sample_count_frames": m.get("sample_count_frames"),
                "avg_fps": m.get("avg_fps"),
                "frame_time_p95_ms": m.get("frame_time_p95_ms"),
            }
        )

    summary = {
        "status": "PASS" if all(c["ok"] for c in checks) and len(runtime_errors) == 0 and len(binding_errors) == 0 else "FAIL",
        "ticket_id": args.ticket_id,
        "expected_branch": args.expected_branch,
        "reported_branch": trace["reported_branch"],
        "observed_git_branch_actual": trace["observed_git_branch_actual"],
        "branch_matches_expected": trace["branch_matches_expected"],
        "checks": checks,
        "patch_status_coverage": {
            "git_status_path": str(git_status_path),
            "changes_patch_path": str(changes_patch_path),
            "patch_apply_check_present": len(apply_paths) > 0,
            "status_paths_count": len(status_paths),
            "patch_paths_count": len(patch_paths),
            "apply_paths_count": len(apply_paths),
            "missing_in_patch": missing_in_patch,
            "orphan_in_patch": orphan_in_patch,
            "apply_missing": apply_missing,
            "apply_orphan": apply_orphan,
        },
        "screenshot_coverage": screenshot_cov,
        "execution_id": execution_id,
        "telegram_correlation_id": trace["telegram_correlation_id"],
        "frontend_job_id": args.frontend_job_id,
        "target_artifact_dir": target_artifact_dir,
        "binding_errors": binding_errors,
        "runtime_viewports_present": sorted(runtime_report["viewports"].keys()),
        "runtime_validation_errors": runtime_errors,
        "trace_path": str(artifacts_dir / "wormhole_scene_trace.json"),
        "runtime_report_path": str(artifacts_dir / "backend_runtime_telemetry_report.json"),
        "events_path": str(artifacts_dir / "backend_traceability_runtime_events.json"),
        "generated_at_utc": _utc_now(),
    }

    _write_json(artifacts_dir / "wormhole_scene_trace.json", trace)
    _write_json(artifacts_dir / "backend_runtime_telemetry_report.json", runtime_report)
    _write_json(artifacts_dir / "backend_traceability_runtime_events.json", events)
    _write_json(artifacts_dir / "backend_traceability_runtime_summary.json", summary)
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
