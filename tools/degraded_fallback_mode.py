#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def _default_db() -> Path:
    env = os.environ.get("BOT_ORCHESTRATOR_DB_PATH", "").strip()
    if env:
        return Path(env).expanduser()
    return Path("/home/aponce/codexbot/data/jobs.sqlite")


def _is_internal_dep(dep_row: sqlite3.Row | None, *, role: str) -> bool:
    if dep_row is None:
        return False
    return str(dep_row["role"] or "") == role


def _append_job_event(cur: sqlite3.Cursor, *, job_id: str, event_type: str, actor: str, details: dict[str, Any]) -> None:
    cur.execute(
        "INSERT INTO job_events(job_id, ts, event_type, actor, details) VALUES (?, ?, ?, ?, ?)",
        (job_id, time.time(), event_type, actor, json.dumps(details, ensure_ascii=False)),
    )


def _append_audit(cur: sqlite3.Cursor, *, event_type: str, actor: str, details: dict[str, Any]) -> None:
    cur.execute(
        "INSERT INTO audit_log(ts, event_type, actor, details) VALUES (?, ?, ?, ?)",
        (time.time(), event_type, actor, json.dumps(details, ensure_ascii=False)),
    )


def cmd_enable(db_path: Path, *, role: str, reason: str, ticket: str, limit: int) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    now = time.time()
    session_id = f"degraded-{int(now)}-{uuid.uuid4().hex[:6]}"

    rows = cur.execute(
        "SELECT * FROM jobs WHERE role = ? AND state = 'waiting_deps' ORDER BY created_at ASC LIMIT ?",
        (role, int(limit)),
    ).fetchall()

    rewired: list[dict[str, Any]] = []
    skipped: list[str] = []

    for row in rows:
        job_id = str(row["job_id"])
        deps = json.loads(str(row["depends_on"] or "[]"))
        if not isinstance(deps, list) or not deps:
            skipped.append(job_id)
            continue

        trace = json.loads(str(row["trace"] or "{}"))
        if isinstance(trace, dict) and trace.get("degraded_mode_active"):
            skipped.append(job_id)
            continue

        external_deps: list[str] = []
        internal_deps: list[str] = []
        for dep in deps:
            dep_row = cur.execute("SELECT job_id, role, state FROM jobs WHERE job_id = ?", (str(dep),)).fetchone()
            if _is_internal_dep(dep_row, role=role):
                internal_deps.append(str(dep))
            else:
                external_deps.append(str(dep))

        if not external_deps:
            skipped.append(job_id)
            continue

        blocked_id = str(uuid.uuid4())
        labels = json.loads(str(row["labels"] or "{}"))
        if not isinstance(labels, dict):
            labels = {}
        blocked_labels = dict(labels)
        blocked_labels.update({"kind": "blocked_external_subgraph", "key": "degraded_fallback"})

        blocked_trace = trace if isinstance(trace, dict) else {}
        blocked_trace = dict(blocked_trace)
        blocked_trace.update(
            {
                "degraded_mode_active": True,
                "degraded_mode_session": session_id,
                "degraded_derived_from": job_id,
                "degraded_external_deps": external_deps,
                "degraded_internal_deps_snapshot": internal_deps,
                "ticket": ticket,
                "reason": reason,
            }
        )

        cur.execute(
            """
            INSERT INTO jobs(
              job_id,source,role,input_text,request_type,priority,model,effort,mode_hint,requires_approval,max_cost_window_usd,
              created_at,updated_at,due_at,state,chat_id,user_id,reply_to_message_id,trace,is_autonomous,parent_job_id,owner,depends_on,
              ttl_seconds,retry_count,max_retries,labels,requires_review,artifacts_dir,blocked_reason,plan_revision,stalled_since
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                blocked_id,
                row["source"],
                row["role"],
                f"[blocked-external-subgraph] degraded fallback split from {job_id[:8]} waiting on: {', '.join(external_deps)}",
                row["request_type"],
                row["priority"],
                row["model"],
                row["effort"],
                row["mode_hint"],
                row["requires_approval"],
                row["max_cost_window_usd"],
                now,
                now,
                row["due_at"],
                "waiting_deps",
                row["chat_id"],
                row["user_id"],
                row["reply_to_message_id"],
                json.dumps(blocked_trace, ensure_ascii=False),
                row["is_autonomous"],
                row["parent_job_id"],
                row["owner"],
                json.dumps(external_deps, ensure_ascii=False),
                row["ttl_seconds"],
                0,
                row["max_retries"],
                json.dumps(blocked_labels, ensure_ascii=False),
                row["requires_review"],
                row["artifacts_dir"],
                "blocked_by_external_dep",
                row["plan_revision"],
                now,
            ),
        )
        _append_job_event(
            cur,
            job_id=blocked_id,
            event_type="degraded_fallback_blocked_created",
            actor="degraded_fallback_mode",
            details={"derived_from": job_id, "external_deps": external_deps, "session_id": session_id},
        )

        new_state = "queued" if not internal_deps else "waiting_deps"
        new_trace = trace if isinstance(trace, dict) else {}
        new_trace = dict(new_trace)
        new_trace.update(
            {
                "degraded_mode_active": True,
                "degraded_mode_session": session_id,
                "degraded_external_moved_to": blocked_id,
                "degraded_external_deps": external_deps,
                "degraded_internal_deps": internal_deps,
                "ticket": ticket,
                "reason": reason,
            }
        )
        cur.execute(
            "UPDATE jobs SET state=?, depends_on=?, blocked_reason=NULL, stalled_since=NULL, updated_at=?, trace=? WHERE job_id=?",
            (new_state, json.dumps(internal_deps, ensure_ascii=False), now, json.dumps(new_trace, ensure_ascii=False), job_id),
        )
        _append_job_event(
            cur,
            job_id=job_id,
            event_type="degraded_fallback_enabled",
            actor="degraded_fallback_mode",
            details={
                "old_depends_on": deps,
                "new_depends_on": internal_deps,
                "blocked_external_node": blocked_id,
                "session_id": session_id,
                "new_state": new_state,
            },
        )

        rewired.append(
            {
                "job_id": job_id,
                "old_depends_on": deps,
                "new_depends_on": internal_deps,
                "new_state": new_state,
                "blocked_external_node": blocked_id,
                "external_deps": external_deps,
            }
        )

    _append_audit(
        cur,
        event_type="degraded_fallback_mode_enabled",
        actor="degraded_fallback_mode",
        details={
            "session_id": session_id,
            "role": role,
            "ticket": ticket,
            "reason": reason,
            "rewired_count": len(rewired),
            "skipped_count": len(skipped),
        },
    )
    conn.commit()
    conn.close()

    return {
        "status": "PASS",
        "mode": "enabled",
        "session_id": session_id,
        "rewired": rewired,
        "skipped_count": len(skipped),
    }


def cmd_disable(db_path: Path, *, session_id: str | None) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where = "trace LIKE '%\"degraded_mode_active\": true%'"
    params: tuple[Any, ...] = ()
    if session_id:
        where += " AND trace LIKE ?"
        params = (f"%{session_id}%",)

    rows = cur.execute(f"SELECT job_id, trace, state FROM jobs WHERE {where}", params).fetchall()
    updated = 0
    now = time.time()
    for row in rows:
        tr = json.loads(str(row["trace"] or "{}"))
        if not isinstance(tr, dict):
            tr = {}
        tr["degraded_mode_active"] = False
        tr["degraded_mode_disabled_at"] = now
        cur.execute("UPDATE jobs SET trace=?, updated_at=? WHERE job_id=?", (json.dumps(tr, ensure_ascii=False), now, row["job_id"]))
        _append_job_event(
            cur,
            job_id=str(row["job_id"]),
            event_type="degraded_fallback_disabled",
            actor="degraded_fallback_mode",
            details={"session_filter": session_id},
        )
        updated += 1

    _append_audit(
        cur,
        event_type="degraded_fallback_mode_disabled",
        actor="degraded_fallback_mode",
        details={"session_filter": session_id, "updated_jobs": updated},
    )
    conn.commit()
    conn.close()
    return {"status": "PASS", "mode": "disabled", "updated_jobs": updated, "session_filter": session_id}


def cmd_status(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    active = cur.execute(
        "SELECT COUNT(1) as c FROM jobs WHERE trace LIKE '%\"degraded_mode_active\": true%'"
    ).fetchone()["c"]
    latest = cur.execute(
        "SELECT ts, event_type, details FROM audit_log WHERE event_type IN ('degraded_fallback_mode_enabled','degraded_fallback_mode_disabled') ORDER BY ts DESC LIMIT 5"
    ).fetchall()
    conn.close()
    return {
        "status": "PASS",
        "active_jobs": int(active),
        "latest_events": [
            {"ts": float(r["ts"]), "event_type": str(r["event_type"]), "details": json.loads(str(r["details"] or "{}"))}
            for r in latest
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Temporary degraded fallback mode for backend delivery while external deps are blocked.")
    ap.add_argument("--db", default=str(_default_db()), help="Path to orchestrator jobs sqlite DB")
    sub = ap.add_subparsers(dest="cmd", required=True)

    en = sub.add_parser("enable", help="Enable degraded fallback mode and rewire waiting_deps backend jobs")
    en.add_argument("--role", default="backend")
    en.add_argument("--reason", default="external_dependency_blocked")
    en.add_argument("--ticket", default="89020d31-6653-411c-aae4-9468ad944308")
    en.add_argument("--limit", type=int, default=50)

    dis = sub.add_parser("disable", help="Disable degraded fallback markers for jobs")
    dis.add_argument("--session-id", default=None)

    sub.add_parser("status", help="Show degraded fallback status")

    args = ap.parse_args()
    db = Path(args.db).expanduser().resolve()

    if args.cmd == "enable":
        result = cmd_enable(db, role=str(args.role), reason=str(args.reason), ticket=str(args.ticket), limit=int(args.limit))
    elif args.cmd == "disable":
        result = cmd_disable(db, session_id=args.session_id)
    else:
        result = cmd_status(db)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
