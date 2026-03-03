#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any


REQUIRED_VIEWPORTS = ("desktop", "tablet", "mobile")
REQUIRED_TELEMETRY_KEYS = ("sample_count_frames", "avg_fps", "p95_frame_ms", "degrade_level", "preset", "quality_tier")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return payload


def _parse_git_status_paths(path: Path) -> list[str]:
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                out.add(parts[-1].strip())
                continue
        if line.startswith("?? "):
            out.add(line[3:].strip())
            continue
        if len(line) > 3 and line[2] == " ":
            rest = line[3:].strip()
        elif len(line) > 2 and line[1] == " ":
            rest = line[2:].strip()
        else:
            rest = line.strip()
        if "->" in rest:
            rest = rest.split("->", 1)[1].strip()
        if rest:
            out.add(rest)
    return sorted(p for p in out if p)


def _parse_patch_paths(path: Path) -> list[str]:
    diff_re = re.compile(r"^diff --git a/(.+?) b/(.+)$")
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = diff_re.match(raw.strip())
        if m:
            rhs = m.group(2).strip()
            if rhs:
                out.add(rhs)
    return sorted(out)


def _normalize_telemetry(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if isinstance(payload.get("viewports"), dict):
        viewports = payload.get("viewports", {})
        return {str(k): v for k, v in viewports.items() if isinstance(v, dict)}
    if isinstance(payload.get("telemetry"), list):
        out: dict[str, dict[str, Any]] = {}
        for item in payload.get("telemetry", []):
            if isinstance(item, dict) and isinstance(item.get("viewport"), str):
                out[item["viewport"]] = item
        return out
    out: dict[str, dict[str, Any]] = {}
    for vp in REQUIRED_VIEWPORTS:
        if isinstance(payload.get(vp), dict):
            out[vp] = payload[vp]
    return out


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def run_validation(
    *,
    artifacts_dir: Path,
    expected_branch: str,
    backend_trace_path: Path,
    git_status_path: Path,
    changes_patch_path: Path,
    runtime_telemetry_path: Path,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    mismatches: list[dict[str, Any]] = []

    # Branch provenance.
    trace = _read_json(backend_trace_path)
    reported_branch = str(trace.get("reported_branch", "")).strip()
    trace_expected = str(trace.get("expected_order_branch", "")).strip() or str(trace.get("expected_branch", "")).strip()
    trace_matches = bool(trace.get("branch_matches_expected", False))
    checks["backend_trace_expected_branch_present"] = bool(trace_expected)
    checks["backend_trace_reported_branch_present"] = bool(reported_branch)
    checks["backend_trace_expected_matches_cli_expected"] = trace_expected == expected_branch
    checks["backend_trace_reported_matches_expected"] = reported_branch == expected_branch
    checks["backend_trace_branch_matches_expected_true"] = trace_matches is True
    if not checks["backend_trace_expected_matches_cli_expected"]:
        mismatches.append(
            {
                "code": "BRANCH_MISMATCH",
                "field": "expected_order_branch",
                "expected": expected_branch,
                "observed": trace_expected,
            }
        )
    if not checks["backend_trace_reported_matches_expected"]:
        mismatches.append(
            {
                "code": "BRANCH_MISMATCH",
                "field": "reported_branch",
                "expected": expected_branch,
                "observed": reported_branch,
            }
        )
    if not checks["backend_trace_branch_matches_expected_true"]:
        mismatches.append(
            {
                "code": "BRANCH_MISMATCH",
                "field": "branch_matches_expected",
                "expected": True,
                "observed": trace_matches,
            }
        )

    # Frontend bundle consistency (patch vs status).
    status_paths = _parse_git_status_paths(git_status_path)
    patch_paths = _parse_patch_paths(changes_patch_path)
    missing_in_patch = sorted(set(status_paths) - set(patch_paths))
    orphan_in_patch = sorted(set(patch_paths) - set(status_paths))
    checks["frontend_patch_covers_status_paths"] = len(missing_in_patch) == 0
    checks["frontend_patch_has_no_orphans"] = len(orphan_in_patch) == 0
    if missing_in_patch:
        mismatches.append(
            {
                "code": "FRONTEND_BUNDLE_INCOHERENT",
                "field": "missing_in_patch",
                "expected": [],
                "observed": missing_in_patch,
            }
        )
    if orphan_in_patch:
        mismatches.append(
            {
                "code": "FRONTEND_BUNDLE_INCOHERENT",
                "field": "orphan_in_patch",
                "expected": [],
                "observed": orphan_in_patch,
            }
        )

    # Runtime telemetry presence per viewport.
    runtime_payload = _read_json(runtime_telemetry_path)
    telemetry = _normalize_telemetry(runtime_payload)
    checks["runtime_telemetry_has_required_viewports"] = all(v in telemetry for v in REQUIRED_VIEWPORTS)
    for vp in REQUIRED_VIEWPORTS:
        vp_payload = telemetry.get(vp)
        if not isinstance(vp_payload, dict):
            mismatches.append(
                {
                    "code": "MISSING_VIEWPORT_TELEMETRY",
                    "field": vp,
                    "expected": "telemetry object",
                    "observed": None,
                }
            )
            continue
        for key in REQUIRED_TELEMETRY_KEYS:
            ok = key in vp_payload
            checks[f"runtime_telemetry_{vp}_{key}_present"] = ok
            if not ok:
                mismatches.append(
                    {
                        "code": "MISSING_VIEWPORT_TELEMETRY",
                        "field": f"{vp}.{key}",
                        "expected": "present",
                        "observed": "missing",
                    }
                )
        if "avg_fps" in vp_payload and not _is_finite_number(vp_payload.get("avg_fps")):
            mismatches.append(
                {
                    "code": "TELEMETRY_SCHEMA_INVALID",
                    "field": f"{vp}.avg_fps",
                    "expected": "finite number",
                    "observed": vp_payload.get("avg_fps"),
                }
            )
        if "p95_frame_ms" in vp_payload and not _is_finite_number(vp_payload.get("p95_frame_ms")):
            mismatches.append(
                {
                    "code": "TELEMETRY_SCHEMA_INVALID",
                    "field": f"{vp}.p95_frame_ms",
                    "expected": "finite number",
                    "observed": vp_payload.get("p95_frame_ms"),
                }
            )

    status = "PASS" if not mismatches else "FAIL"
    return {
        "status": status,
        "exit_code": 0 if status == "PASS" else 2,
        "captured_at_utc": _utc_now(),
        "artifacts_dir": str(artifacts_dir),
        "expected_order_branch": expected_branch,
        "reported_order_branch": reported_branch,
        "checks": checks,
        "mismatches": mismatches,
        "details": {
            "backend_trace_path": str(backend_trace_path),
            "git_status_path": str(git_status_path),
            "changes_patch_path": str(changes_patch_path),
            "runtime_telemetry_path": str(runtime_telemetry_path),
            "path_sets": {
                "git_status_paths": status_paths,
                "changes_patch_paths": patch_paths,
                "missing_in_patch": missing_in_patch,
                "orphan_in_patch": orphan_in_patch,
            },
            "runtime_viewports_present": sorted(telemetry.keys()),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-lane consistency validator for branch, patch/status and runtime telemetry.")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--expected-branch", required=True)
    ap.add_argument("--backend-trace", default="")
    ap.add_argument("--git-status", default="")
    ap.add_argument("--changes-patch", default="")
    ap.add_argument("--runtime-telemetry", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    art = Path(args.artifacts_dir).expanduser().resolve()
    backend_trace_path = Path(args.backend_trace).expanduser().resolve() if args.backend_trace else art / "wormhole_scene_trace.json"
    git_status_path = Path(args.git_status).expanduser().resolve() if args.git_status else art / "git_status.txt"
    changes_patch_path = Path(args.changes_patch).expanduser().resolve() if args.changes_patch else art / "changes.patch"
    runtime_telemetry_path = (
        Path(args.runtime_telemetry).expanduser().resolve() if args.runtime_telemetry else art / "runtime_telemetry_report.json"
    )
    out_path = Path(args.out).expanduser().resolve() if args.out else art / "crosslane_validator_report.json"

    report = run_validation(
        artifacts_dir=art,
        expected_branch=args.expected_branch,
        backend_trace_path=backend_trace_path,
        git_status_path=git_status_path,
        changes_patch_path=changes_patch_path,
        runtime_telemetry_path=runtime_telemetry_path,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
