#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stat_meta(path: Path) -> dict[str, Any]:
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


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    tmp.write_bytes(src.read_bytes())
    tmp.replace(dst)


def _contract_view(meta: dict[str, Any]) -> tuple[bool, int, str]:
    return (
        bool(meta.get("exists", False)),
        int(meta.get("size_bytes", 0)),
        str(meta.get("sha256", "")),
    )


def compare(root_dir: Path, smoke_dir: Path, files: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    root_snap: dict[str, Any] = {}
    smoke_snap: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    for rel in files:
        r = stat_meta(root_dir / rel)
        s = stat_meta(smoke_dir / rel)
        root_snap[rel] = r
        smoke_snap[rel] = s
        if _contract_view(r) != _contract_view(s):
            mismatches.append({"file": rel, "root": r, "smoke_stable": s})
    return mismatches, root_snap, smoke_snap


def detect_late_mutation(snapshot_t0: dict[str, Any], snapshot_t1: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in sorted(set(snapshot_t0) | set(snapshot_t1)):
        a = snapshot_t0.get(rel, {})
        b = snapshot_t1.get(rel, {})
        for field in ("exists", "size_bytes", "sha256", "mtime_epoch"):
            if a.get(field) != b.get(field):
                out.append({"file": rel, "field": field, "t0": a.get(field), "t1": b.get(field)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-final root vs smoke_stable contract guard.")
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--smoke-stable-dir", required=True)
    ap.add_argument("--files", default="changes.patch,git_status.txt")
    ap.add_argument("--report", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--sleep-seconds", type=float, default=5.0)
    ap.add_argument(
        "--sync-root-from-smoke",
        action="store_true",
        help="Atomically copy contractual files from smoke_stable into root before checks.",
    )
    ap.add_argument(
        "--allow-missing-summary",
        action="store_true",
        help="Allow pre-summary execution. When disabled, missing summary is a hard fail.",
    )
    args = ap.parse_args()

    root_dir = Path(args.root_dir).resolve()
    smoke_dir = Path(args.smoke_stable_dir).resolve()
    files = [x.strip() for x in args.files.split(",") if x.strip()]
    summary_path = Path(args.summary).resolve()
    report_path = Path(args.report).resolve()

    check_started = utc_now()
    smoke_pre = {rel: stat_meta(smoke_dir / rel) for rel in files}
    root_pre = {rel: stat_meta(root_dir / rel) for rel in files}

    if args.sync_root_from_smoke:
        for rel in files:
            src = smoke_dir / rel
            dst = root_dir / rel
            if src.exists() and src.is_file():
                atomic_copy(src, dst)

    mismatch_post_sync, root_post_sync, smoke_post_sync = compare(root_dir, smoke_dir, files)
    summary_stat = stat_meta(summary_path)
    summary_exists = bool(summary_stat.get("exists"))

    time.sleep(max(args.sleep_seconds, 0.0))
    root_terminal = {rel: stat_meta(root_dir / rel) for rel in files}
    summary_after = stat_meta(summary_path)

    late_mut = detect_late_mutation(root_post_sync, root_terminal)
    summary_mut = detect_late_mutation({"summary": summary_stat}, {"summary": summary_after})
    summary_mtime = float(summary_after.get("mtime_epoch", 0.0))

    mtime_after_summary: list[dict[str, Any]] = []
    non_empty_violations: list[dict[str, Any]] = []
    for rel, meta in root_terminal.items():
        if (not bool(meta.get("exists"))) or int(meta.get("size_bytes", 0)) <= 0:
            non_empty_violations.append({"file": rel, "current": meta})
        if summary_exists and float(meta.get("mtime_epoch", 0.0)) > summary_mtime:
            mtime_after_summary.append(
                {
                    "file": rel,
                    "file_mtime_epoch": float(meta.get("mtime_epoch", 0.0)),
                    "summary_mtime_epoch": summary_mtime,
                }
            )

    mismatch_terminal: list[dict[str, Any]] = []
    for rel in files:
        rt = root_terminal.get(rel, {})
        sp = smoke_post_sync.get(rel, {})
        if _contract_view(rt) != _contract_view(sp):
            mismatch_terminal.append({"file": rel, "root_terminal": rt, "smoke": sp})

    errors: list[dict[str, Any]] = []
    if mismatch_post_sync:
        errors.append({"type": "root_smoke_mismatch_post_summary", "count": len(mismatch_post_sync), "details": mismatch_post_sync})
    if mismatch_terminal:
        errors.append({"type": "root_smoke_mismatch_terminal", "count": len(mismatch_terminal), "details": mismatch_terminal})
    if non_empty_violations:
        errors.append({"type": "root_contract_file_empty_or_missing_terminal", "count": len(non_empty_violations), "details": non_empty_violations})
    if late_mut:
        errors.append({"type": "late_root_mutation_after_summary", "count": len(late_mut), "details": late_mut})
    if (not summary_exists) and (not args.allow_missing_summary):
        errors.append(
            {
                "type": "missing_summary_file",
                "path": str(summary_path),
                "reason": "final_contract_requires_terminal_snapshot_relative_to_summary",
            }
        )
    if summary_exists and summary_mut:
        errors.append({"type": "summary_mutation_during_guard_window", "count": len(summary_mut), "details": summary_mut})
    if mtime_after_summary:
        errors.append(
            {
                "type": "contract_file_mtime_newer_than_final_summary",
                "count": len(mtime_after_summary),
                "details": mtime_after_summary,
            }
        )

    report = {
        "check": "postfinal_root_smoke_guard",
        "checked_at_utc": utc_now(),
        "check_started_at_utc": check_started,
        "root_dir": str(root_dir),
        "smoke_stable_dir": str(smoke_dir),
        "files": files,
        "sync_root_from_smoke": bool(args.sync_root_from_smoke),
        "summary_required": not args.allow_missing_summary,
        "summary_exists": summary_exists,
        "root_snapshot_pre_summary": root_pre,
        "root_snapshot_post_summary": root_post_sync,
        "root_snapshot_terminal": root_terminal,
        "smoke_snapshot_pre_summary": smoke_pre,
        "smoke_snapshot_post_summary": smoke_post_sync,
        "root_snapshot_t0": root_post_sync,
        "root_snapshot_t1": root_terminal,
        "smoke_snapshot": smoke_post_sync,
        "summary_path": str(summary_path),
        "summary_snapshot_t0": summary_stat,
        "summary_snapshot_t1": summary_after,
        "summary_mtime_epoch": summary_mtime,
        "mismatch_count_post_summary": len(mismatch_post_sync),
        "mismatch_count_terminal": len(mismatch_terminal),
        "mismatch_count": len(mismatch_terminal),
        "late_mutation_count": len(late_mut),
        "summary_mutation_count": len(summary_mut),
        "mtime_after_summary_count": len(mtime_after_summary),
        "terminal_non_empty_violations": len(non_empty_violations),
        "errors": errors,
        "root_smoke_consistency_post_summary": len(mismatch_post_sync) == 0,
        "root_smoke_consistency_terminal": len(mismatch_terminal) == 0,
        "root_smoke_consistency": len(mismatch_terminal) == 0,
        "root_contract_files_non_empty_terminal": len(non_empty_violations) == 0,
        "root_mtime_not_newer_than_summary": len(mtime_after_summary) == 0,
        "status": "PASS" if not errors else "FAIL",
        "exit_code": 0 if not errors else 2,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
