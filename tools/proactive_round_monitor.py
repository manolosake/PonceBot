#!/usr/bin/env python3
import json
import os
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, UTC
from pathlib import Path

from orchestrator.storage import SQLiteTaskStorage
from orchestrator.queue import OrchestratorQueue

DB = "/home/aponce/codexbot/data/jobs.sqlite"
SERVICE = "codexbot.service"
RUNBOOK_ID = "jarvis_proactive_lane"

LOCAL_ROLES = {"architect_local", "implementer_local", "reviewer_local", "frontend_local", "ops_local"}
FORBIDDEN_CLI_ROLES = {"backend", "frontend", "qa", "sre", "security", "research", "release_mgr", "product_ops"}
OPEN_STATES = ("queued", "running", "waiting_deps", "blocked", "blocked_approval")

ROUND_SECONDS = 15 * 60
ACTIVATION_WAIT_SECONDS = 6 * 60
TICK_SECONDS = 20
ROUNDS_MIN = 3
ROUNDS_MAX = 5

stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
out_dir = Path("/home/aponce/codexbot/data/artifacts/proactive_monitor")
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / f"rounds_{stamp}.jsonl"
summary_path = out_dir / f"rounds_summary_{stamp}.json"

storage = SQLiteTaskStorage(Path(DB))
orch = OrchestratorQueue(storage)


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout.strip()


def db_conn():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def is_proactive_order(title: str, body: str) -> bool:
    blob = f"{title or ''} {body or ''}".lower()
    return ("proactive" in blob) or ("[proactive:" in blob)


def append_log(kind: str, payload: dict):
    evt = {
        "ts": time.time(),
        "kind": kind,
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt, ensure_ascii=False) + "\n")


def append_audit(cur, event_type: str, details: dict):
    cur.execute(
        "INSERT INTO audit_log (ts, event_type, actor, details) VALUES (?, ?, ?, ?)",
        (time.time(), event_type, "proactive_round_monitor", json.dumps(details, ensure_ascii=False)),
    )


def cancel_job(cur, job_id: str, from_state: str, reason: str) -> int:
    row = cur.execute("SELECT trace FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    tr_raw = row["trace"] if row else "{}"
    try:
        tr = json.loads(tr_raw or "{}")
    except Exception:
        tr = {}
    tr["cancel_reason"] = reason
    tr["cancel_ts"] = time.time()
    cur.execute(
        "UPDATE jobs SET state='cancelled', updated_at=?, blocked_reason=NULL, stalled_since=NULL, trace=? WHERE job_id=? AND state=?",
        (time.time(), json.dumps(tr, ensure_ascii=False), job_id, from_state),
    )
    return int(cur.rowcount or 0)


def proactive_orders(cur):
    rows = cur.execute(
        "SELECT order_id,status,phase,created_at,updated_at,COALESCE(title,'') title,COALESCE(body,'') body FROM ceo_orders ORDER BY created_at ASC"
    ).fetchall()
    return [r for r in rows if is_proactive_order(r["title"], r["body"])]


def classify_open(cur):
    open_rows = cur.execute(
        """
        SELECT job_id,state,role,parent_job_id,created_at,updated_at,labels,input_text
        FROM jobs
        WHERE state IN ('queued','running','waiting_deps','blocked','blocked_approval')
        ORDER BY created_at ASC
        """
    ).fetchall()
    return open_rows


def soft_reset(reason: str):
    out = orch.stop_all_global(
        reason=reason,
        actor="proactive_round_monitor",
        chat_id=None,
        close_orders=True,
        close_projects=True,
        clear_workspace_leases=True,
    )
    orch.set_runbook_last_run(runbook_id=RUNBOOK_ID, ts=0.0)
    return out


def tick_controls(round_id: int, round_start: float, now: float, activation_phase: bool, stats: dict):
    con = db_conn()
    cur = con.cursor()

    p_orders = proactive_orders(cur)
    p_ids = {str(r["order_id"]) for r in p_orders}
    active_p = [r for r in p_orders if str(r["status"]).lower() == "active"]

    open_rows = classify_open(cur)

    local_open = 0
    jarvis_open = 0
    forbidden_open = 0
    open_by_role = {}
    active_roots = set()
    needs_runbook_reset = False

    for r in open_rows:
        role = str(r["role"] or "").strip().lower()
        state = str(r["state"] or "").strip().lower()
        parent = str(r["parent_job_id"] or "").strip()
        root = parent or str(r["job_id"])
        updated_at = float(r["updated_at"] or r["created_at"] or now)
        age = max(0.0, now - updated_at)
        if role == "jarvis":
            jarvis_open += 1
        if role in LOCAL_ROLES:
            local_open += 1
        open_by_role[role] = open_by_role.get(role, 0) + 1

        is_pro = root in p_ids
        if is_pro:
            active_roots.add(root)

        if is_pro and role in FORBIDDEN_CLI_ROLES:
            forbidden_open += 1
            stats["forbidden_hits"] += 1
            cancelled = cancel_job(cur, str(r["job_id"]), state, "proactive_forbidden_cli_role")
            stats["cancelled_forbidden"] += cancelled
            append_audit(cur, "monitor.round.forbidden_cancel", {
                "round": round_id,
                "job_id": str(r["job_id"]),
                "root_order_id": root,
                "role": role,
                "state": state,
            })

        if is_pro and role in LOCAL_ROLES and state == "blocked" and age >= 600:
            stats["blocked_hits"] += 1
            cancelled = cancel_job(cur, str(r["job_id"]), state, "proactive_local_blocked_timeout")
            stats["cancelled_blocked"] += cancelled
            needs_runbook_reset = True
            stats["runbook_resets"] += 1
            append_audit(cur, "monitor.round.blocked_local_cancel", {
                "round": round_id,
                "job_id": str(r["job_id"]),
                "root_order_id": root,
                "role": role,
                "state": state,
                "age_seconds": int(age),
            })

    if jarvis_open > 0:
        stats["jarvis_seen"] = True
    if local_open > 0:
        stats["local_seen"] = True
    if active_p:
        stats["proactive_active_seen"] = True

    # Activation self-heal
    if activation_phase and (not active_p) and (now - round_start) > 120:
        needs_runbook_reset = True
        stats["runbook_resets"] += 1

    con.commit()

    if needs_runbook_reset:
        try:
            orch.set_runbook_last_run(runbook_id=RUNBOOK_ID, ts=0.0)
        except Exception:
            pass

    tick = {
        "round": round_id,
        "t_round_s": int(now - round_start),
        "activation_phase": bool(activation_phase),
        "open_jobs": len(open_rows),
        "active_proactive_orders": len(active_p),
        "jarvis_open": jarvis_open,
        "local_open": local_open,
        "forbidden_open": forbidden_open,
        "open_by_role": open_by_role,
        "active_proactive_roots": sorted(list(active_roots))[:6],
        "stats": {
            "forbidden_hits": int(stats["forbidden_hits"]),
            "blocked_hits": int(stats["blocked_hits"]),
            "cancelled_forbidden": int(stats["cancelled_forbidden"]),
            "cancelled_blocked": int(stats["cancelled_blocked"]),
            "runbook_resets": int(stats["runbook_resets"]),
            "jarvis_seen": bool(stats["jarvis_seen"]),
            "local_seen": bool(stats["local_seen"]),
            "proactive_active_seen": bool(stats["proactive_active_seen"]),
        },
    }
    append_log("tick", tick)
    print(
        f"ROUND {round_id} t={tick['t_round_s']}s open={tick['open_jobs']} active_pro={tick['active_proactive_orders']} "
        f"jarvis={tick['jarvis_open']} local={tick['local_open']} forbidden={tick['forbidden_open']} "
        f"resets={tick['stats']['runbook_resets']}"
    )

    return tick


all_rounds = []
pass_streak = 0

append_log("start", {"stamp": stamp})
print(f"ROUND_MONITOR_START {stamp} log={log_path}")

for round_id in range(1, ROUNDS_MAX + 1):
    # Pre-round hard clean + restart.
    code, out = run(["systemctl", "--user", "restart", SERVICE])
    append_log("restart", {"round": round_id, "code": code, "out": out[-300:]})

    reset_out = soft_reset(reason=f"proactive_round_{round_id}_preclean")
    append_log("preclean", {"round": round_id, "reset": reset_out})

    round_start = time.time()
    activation_deadline = round_start + ACTIVATION_WAIT_SECONDS
    round_deadline = round_start + ROUND_SECONDS

    stats = {
        "forbidden_hits": 0,
        "blocked_hits": 0,
        "cancelled_forbidden": 0,
        "cancelled_blocked": 0,
        "runbook_resets": 0,
        "jarvis_seen": False,
        "local_seen": False,
        "proactive_active_seen": False,
    }

    activated = False

    while time.time() < activation_deadline:
        now = time.time()
        tick = tick_controls(round_id, round_start, now, True, stats)
        if tick["active_proactive_orders"] > 0 and tick["local_open"] > 0:
            activated = True
            append_log("activated", {"round": round_id, "t_round_s": tick["t_round_s"]})
            break
        time.sleep(TICK_SECONDS)

    # If not activated, one extra forced nudge.
    if not activated:
        orch.set_runbook_last_run(runbook_id=RUNBOOK_ID, ts=0.0)
        stats["runbook_resets"] += 1
        append_log("activation_nudge", {"round": round_id})
        extra_deadline = time.time() + 180
        while time.time() < extra_deadline:
            now = time.time()
            tick = tick_controls(round_id, round_start, now, True, stats)
            if tick["active_proactive_orders"] > 0 and tick["local_open"] > 0:
                activated = True
                append_log("activated_after_nudge", {"round": round_id, "t_round_s": tick["t_round_s"]})
                break
            time.sleep(TICK_SECONDS)

    # Main round monitoring window.
    while time.time() < round_deadline:
        now = time.time()
        tick_controls(round_id, round_start, now, False, stats)
        time.sleep(TICK_SECONDS)

    round_result = {
        "round": round_id,
        "activated": bool(activated),
        "jarvis_seen": bool(stats["jarvis_seen"]),
        "local_seen": bool(stats["local_seen"]),
        "proactive_active_seen": bool(stats["proactive_active_seen"]),
        "forbidden_hits": int(stats["forbidden_hits"]),
        "blocked_hits": int(stats["blocked_hits"]),
        "cancelled_forbidden": int(stats["cancelled_forbidden"]),
        "cancelled_blocked": int(stats["cancelled_blocked"]),
        "runbook_resets": int(stats["runbook_resets"]),
        "duration_seconds": int(time.time() - round_start),
    }

    round_pass = (
        round_result["activated"]
        and round_result["jarvis_seen"]
        and round_result["local_seen"]
        and round_result["proactive_active_seen"]
        and round_result["forbidden_hits"] == 0
    )
    round_result["pass"] = bool(round_pass)

    all_rounds.append(round_result)
    append_log("round_result", round_result)
    print("ROUND_RESULT", json.dumps(round_result, ensure_ascii=False))

    if round_pass:
        pass_streak += 1
    else:
        pass_streak = 0

    # We require at least 3 good rounds; stop early only when achieved and >= min rounds.
    if round_id >= ROUNDS_MIN and pass_streak >= 3:
        break

summary = {
    "status": "PASS" if (len(all_rounds) >= ROUNDS_MIN and all(r.get("pass") for r in all_rounds[-3:])) else "FAIL",
    "rounds": all_rounds,
    "rounds_executed": len(all_rounds),
    "required_min_rounds": ROUNDS_MIN,
    "required_pass_streak": 3,
    "log_path": str(log_path),
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("ROUND_MONITOR_END", json.dumps({"status": summary["status"], "summary": str(summary_path)}, ensure_ascii=False))
append_log("end", {"status": summary["status"], "summary": str(summary_path)})
