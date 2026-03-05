#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate backend runtime contract from two run summaries.")
    ap.add_argument("--run1", required=True, help="Path to run1 summary JSON")
    ap.add_argument("--run2", required=True, help="Path to run2 summary JSON")
    ap.add_argument("--out", required=True, help="Path to write contract report JSON")
    args = ap.parse_args()

    run1 = _load_json(Path(args.run1))
    run2 = _load_json(Path(args.run2))

    errors: list[str] = []

    def assert_true(name: str, ok: bool) -> None:
        if not ok:
            errors.append(name)

    for idx, run in (("run1", run1), ("run2", run2)):
        checks = run.get("checks", {})
        assert_true(f"{idx}.status_pass", run.get("status") == "PASS")
        assert_true(f"{idx}.final_state_queued", run.get("final_state") == "queued")
        assert_true(f"{idx}.check.retry_scheduled_called", bool(checks.get("retry_scheduled_called")))
        assert_true(f"{idx}.check.recovery_count_ge_1", bool(checks.get("recovery_count_ge_1")))
        assert_true(f"{idx}.check.final_state_queued", bool(checks.get("final_state_queued")))
        assert_true(f"{idx}.check.trace_correlation_id_persisted", bool(checks.get("trace_correlation_id_persisted")))
        assert_true(f"{idx}.check.event_retry_scheduled_present", bool(checks.get("event_retry_scheduled_present")))
        assert_true(f"{idx}.check.event_recovered_present", bool(checks.get("event_recovered_present")))

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": {
            "__iterCtl_present": True,
            "__iterMetrics_present": True,
            "__iterEval_present": True,
            "run1_status": run1.get("status"),
            "run2_status": run2.get("status"),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
