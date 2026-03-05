#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.storage import SQLiteTaskStorage


def _collect_job_events(db_path: Path, job_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, ts, event_type, details FROM job_events WHERE job_id = ? ORDER BY id ASC",
            (job_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            details = json.loads(str(row["details"] or "{}"))
        except Exception:
            details = {}
        out.append(
            {
                "event_id": int(row["id"]),
                "ts": float(row["ts"]),
                "event_type": str(row["event_type"]),
                "details": details,
            }
        )
    return out


def run_smoke(db_path: Path) -> tuple[bool, dict[str, Any]]:
    storage = SQLiteTaskStorage(db_path)
    q = OrchestratorQueue(storage=storage, role_profiles=None)
    correlation_id = f"corr-{uuid.uuid4().hex[:12]}"

    task_id = q.submit_task(
        Task.new(
            source="smoke",
            role="backend",
            input_text="orchestration reliability smoke",
            request_type="task",
            priority=1,
            model="gpt-5.3-codex",
            effort="medium",
            mode_hint="rw",
            requires_approval=False,
            max_cost_window_usd=1.0,
            chat_id=42,
            max_retries=2,
            trace={"ticket_id": "89020d31-6653-411c-aae4-9468ad944308", "correlation_id": correlation_id},
        )
    )

    q.update_trace(task_id, smoke_case="failure_recovery")
    q.update_state(task_id, "running", live_phase="worker_exec", live_slot=2)
    retry_ok = q.bump_retry(task_id, due_at=time.time() + 2.0, error="simulated_worker_failure")
    q.update_state(task_id, "running", live_phase="crash_recovery_window")
    recovered = q.recover_stale_running()

    task = q.get_job(task_id)
    events = _collect_job_events(db_path, task_id)
    event_types = [str(ev.get("event_type")) for ev in events]
    checks = {
        "retry_scheduled_called": bool(retry_ok),
        "recovery_count_ge_1": int(recovered) >= 1,
        "final_state_queued": bool(task is not None and task.state == "queued"),
        "trace_correlation_id_persisted": bool(task is not None and str((task.trace or {}).get("correlation_id")) == correlation_id),
        "event_retry_scheduled_present": "retry_scheduled" in event_types,
        "event_recovered_present": "recovered" in event_types,
    }
    ok = all(bool(v) for v in checks.values())

    report = {
        "status": "PASS" if ok else "FAIL",
        "job_id": task_id,
        "job_id_short": task_id[:8],
        "correlation_id": correlation_id,
        "checks": checks,
        "final_state": None if task is None else task.state,
        "retry_count": None if task is None else int(task.retry_count),
        "events": events,
        "notes": {
            "failure_behavior": "On worker failure, task is retried with due_at and retry_count incremented.",
            "recovery_behavior": "On restart with stale running jobs, recover_stale_running returns them to queued and emits recovered event.",
        },
    }
    return ok, report


def main() -> int:
    ap = argparse.ArgumentParser(description="Run reliability smoke scenario (failure + recovery).")
    ap.add_argument("--db-path", default="", help="Optional sqlite path. Uses temp DB when omitted.")
    ap.add_argument("--json-out", default="", help="Path to write JSON report")
    args = ap.parse_args()

    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if str(args.db_path).strip():
            db_path = Path(str(args.db_path)).expanduser().resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            tmp_dir = tempfile.TemporaryDirectory(prefix="orch-smoke-")
            db_path = Path(tmp_dir.name) / "jobs.sqlite"

        started_at = time.time()
        ok, report = run_smoke(db_path)
        report["db_path"] = str(db_path)
        report["started_at"] = started_at
        report["finished_at"] = time.time()

        payload = json.dumps(report, indent=2, ensure_ascii=False)
        if str(args.json_out).strip():
            out_path = Path(str(args.json_out)).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0 if ok else 1
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
