#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def meta(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"exists": False, "size_bytes": 0, "sha256": "", "mtime_epoch": 0.0}
    b = path.read_bytes()
    st = path.stat()
    return {"exists": True, "size_bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "mtime_epoch": float(st.st_mtime)}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--files", default="changes.patch,git_status.txt")
    ap.add_argument("--guard-report", required=True)
    ap.add_argument("--probe2-report", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    art = Path(args.artifacts_dir).resolve()
    smoke = art / "smoke_stable"
    files = [x.strip() for x in args.files.split(",") if x.strip()]
    guard = load(Path(args.guard_report).resolve())
    probe2 = load(Path(args.probe2_report).resolve())
    live_now = {f: meta(art / f) for f in files}
    smoke_now = {f: meta(smoke / f) for f in files}
    p2_last = (probe2.get("snapshots") or [{}])[-1].get("files", {})

    errors = []
    if guard.get("status") != "PASS":
        errors.append({"type": "guard_not_pass", "status": guard.get("status"), "exit_code": guard.get("exit_code")})
    if probe2.get("status") != "PASS":
        errors.append({"type": "probe2_not_pass", "status": probe2.get("status"), "exit_code": probe2.get("exit_code")})
    for f in files:
        ln = live_now.get(f, {})
        p2 = p2_last.get(f, {})
        sm = smoke_now.get(f, {})
        if (ln.get("exists"), ln.get("size_bytes"), ln.get("sha256"), ln.get("mtime_epoch")) != (
            p2.get("exists"),
            p2.get("size_bytes"),
            p2.get("sha256"),
            p2.get("mtime_epoch"),
        ):
            errors.append({"type": "live_now_differs_from_probe2", "file": f, "live_now": ln, "probe2_last": p2})
        if (ln.get("exists"), ln.get("size_bytes"), ln.get("sha256")) != (
            sm.get("exists"),
            sm.get("size_bytes"),
            sm.get("sha256"),
        ):
            errors.append({"type": "live_now_differs_from_smoke_contract", "file": f, "live_now": ln, "smoke": sm})

    report = {
        "check": "terminal_done_decider",
        "checked_at_utc": utc_now(),
        "decision_timestamps": {
            "guard_checked_at_utc": guard.get("checked_at_utc"),
            "probe2_captured_at_utc": probe2.get("captured_at_utc"),
            "done_decision_checked_at_utc": utc_now(),
        },
        "guard_status": guard.get("status"),
        "probe2_status": probe2.get("status"),
        "live_now": live_now,
        "probe2_last": p2_last,
        "smoke_now": smoke_now,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "exit_code": 0 if not errors else 2,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
