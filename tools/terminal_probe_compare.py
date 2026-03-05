#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-a", required=True)
    ap.add_argument("--probe-b", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    a = load(Path(args.probe_a).resolve())
    b = load(Path(args.probe_b).resolve())
    errors = []

    if a.get("status") != "PASS":
        errors.append({"type": "probe_a_not_pass", "status": a.get("status"), "exit_code": a.get("exit_code")})
    if b.get("status") != "PASS":
        errors.append({"type": "probe_b_not_pass", "status": b.get("status"), "exit_code": b.get("exit_code")})

    sa = a.get("snapshots", [])
    sb = b.get("snapshots", [])
    if sa and sb:
        a_last = sa[-1].get("files", {})
        b_first = sb[0].get("files", {})
        files = sorted(set(a_last) | set(b_first))
        for f in files:
            for field in ("exists", "size_bytes", "sha256", "mtime_epoch"):
                if a_last.get(f, {}).get(field) != b_first.get(f, {}).get(field):
                    errors.append(
                        {
                            "type": "post_close_reprobe_drift",
                            "file": f,
                            "field": field,
                            "probe_a_last": a_last.get(f, {}).get(field),
                            "probe_b_first": b_first.get(f, {}).get(field),
                        }
                    )
    else:
        files = []
        errors.append({"type": "missing_probe_samples", "probe_a_samples": len(sa), "probe_b_samples": len(sb)})

    report = {
        "check": "terminal_probe_compare",
        "checked_at_utc": utc_now(),
        "files_compared": files,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "exit_code": 0 if not errors else 2,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
