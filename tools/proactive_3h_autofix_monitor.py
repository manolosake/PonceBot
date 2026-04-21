#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path("/home/aponce/codexbot")
DB_PATH = ROOT / "data" / "jobs.sqlite"
DEFAULT_LOG = ROOT / "data" / "artifacts" / "proactive_health" / "autofix_monitor.log"
OPEN_STATES = ("queued", "running", "waiting_deps", "blocked", "blocked_approval", "pending_review")
STALE_BLOCKED_ONLY_AGE_S = 900.0
STALE_BLOCKED_UNKNOWN_AGE_S = 300.0


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _log(msg: str, *, path: Path, **fields: Any) -> None:
    payload = {"ts": _ts(), "msg": msg, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    print(line.rstrip("\n"), flush=True)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in __import__("os").environ:
            __import__("os").environ[key] = value


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _is_proactive(order_row: sqlite3.Row) -> bool:
    title = str(order_row["title"] or "").lower()
    body = str(order_row["body"] or "").lower()
    blob = f"{title}\n{body}"
    return ("proactive sprint" in blob) or ("[proactive:" in blob)


def _active_proactive_orders(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT order_id, chat_id, title, body, status, phase, updated_at FROM ceo_orders WHERE status='active' ORDER BY updated_at DESC"
    ).fetchall()
    return [row for row in rows if _is_proactive(row)]


def _open_children(conn: sqlite3.Connection, *, order_id: str) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in OPEN_STATES)
    return conn.execute(
        f"""
        SELECT job_id, role, state, model, created_at, updated_at, trace
        FROM jobs
        WHERE parent_job_id=? AND state IN ({placeholders})
        ORDER BY created_at ASC
        """,
        (order_id, *OPEN_STATES),
    ).fetchall()


def _recent_role_jobs(
    conn: sqlite3.Connection,
    *,
    order_id: str,
    role: str,
    seconds: float,
) -> list[sqlite3.Row]:
    since = float(time.time()) - float(seconds)
    return conn.execute(
        """
        SELECT job_id, role, state, model, created_at, updated_at, trace
        FROM jobs
        WHERE parent_job_id=? AND role=? AND updated_at>=?
        ORDER BY updated_at DESC
        """,
        (order_id, role, since),
    ).fetchall()


def _terminal_children_count(conn: sqlite3.Connection, *, order_id: str) -> int:
    placeholders = ",".join("?" for _ in OPEN_STATES)
    row = conn.execute(
        f"SELECT COUNT(*) AS c FROM jobs WHERE parent_job_id=? AND state NOT IN ({placeholders})",
        (order_id, *OPEN_STATES),
    ).fetchone()
    return int((row["c"] if row is not None else 0) or 0)


def _recent_cancelled_blocked_cleanup_count(conn: sqlite3.Connection, *, order_id: str, seconds: float = 1800.0) -> int:
    since = float(time.time()) - float(seconds)
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM jobs
            WHERE parent_job_id=?
              AND state='cancelled'
              AND blocked_reason IN ('autofix_stale_blocked_only', 'autofix_superseded_blocker', 'autofix_superseded_by_newer_terminal')
              AND updated_at>=?
            """,
            (order_id, since),
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int((row["c"] if row is not None else 0) or 0)


def _job_blocked_reason(conn: sqlite3.Connection, *, job_id: str) -> str:
    try:
        row = conn.execute("SELECT blocked_reason FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    except sqlite3.OperationalError:
        return ""
    return str((row["blocked_reason"] if row is not None else "") or "").strip()


def _extract_waiting_job_id(blocked_reason: str) -> str:
    txt = str(blocked_reason or "").strip()
    for prefix in ("waiting_for_job:", "depends_on_job:", "blocked_on_job:"):
        if txt.startswith(prefix):
            return txt[len(prefix) :].strip()
    return ""


def _blocked_cleanup_age_threshold(blocked_reason: str) -> float:
    # Structured blocker reasons are granted more time; unknown blockers clear faster.
    if _extract_waiting_job_id(blocked_reason):
        return STALE_BLOCKED_ONLY_AGE_S
    txt = str(blocked_reason or "").strip().lower()
    if txt in ("", "unknown_blocker", "legacy_blocker", "approval_timeout"):
        return STALE_BLOCKED_UNKNOWN_AGE_S
    return STALE_BLOCKED_ONLY_AGE_S


def _job_is_terminal(conn: sqlite3.Connection, *, job_id: str) -> bool:
    row = conn.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        return False
    state = str(row["state"] or "").strip().lower()
    return state not in OPEN_STATES


def _is_superseded_by_newer_terminal_same_role(
    conn: sqlite3.Connection,
    *,
    order_id: str,
    role: str,
    created_at: float,
) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM jobs
        WHERE parent_job_id=?
          AND role=?
          AND created_at>?
          AND state NOT IN ({})
        """.format(",".join("?" for _ in OPEN_STATES)),
        (order_id, role, float(created_at), *OPEN_STATES),
    ).fetchone()
    return int((row["c"] if row is not None else 0) or 0) > 0


def _close_order_done(conn: sqlite3.Connection, *, order_id: str) -> bool:
    now = float(time.time())
    cur = conn.execute(
        "UPDATE ceo_orders SET status='done', phase='done', updated_at=? WHERE order_id=? AND status='active'",
        (now, order_id),
    )
    return int(cur.rowcount or 0) > 0


def _set_order_phase_planning(conn: sqlite3.Connection, *, order_id: str) -> None:
    now = float(time.time())
    conn.execute(
        "UPDATE ceo_orders SET phase='planning', updated_at=? WHERE order_id=? AND status='active'",
        (now, order_id),
    )


def _enqueue_fallback_followup(conn: sqlite3.Connection, *, order_id: str, chat_id: int) -> str:
    now = float(time.time())
    job_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, source, role, input_text, request_type, priority, model, effort, mode_hint,
            requires_approval, max_cost_window_usd, created_at, updated_at, due_at, state,
            chat_id, user_id, reply_to_message_id, is_autonomous, parent_job_id, owner,
            depends_on, ttl_seconds, retry_count, max_retries, labels, requires_review,
            artifacts_dir, blocked_reason, plan_revision, stalled_since, trace
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            "autopilot",
            "architect_local",
            "Fallback proactive reseed: synthesize one concrete reliability improvement with tests and evidence.",
            "task",
            2,
            "gpt-5",
            "medium",
            "rw",
            0,
            2.0,
            now,
            now,
            None,
            "queued",
            int(chat_id),
            None,
            None,
            1,
            order_id,
            "autofix_monitor",
            "[]",
            None,
            0,
            0,
            json.dumps({"kind": "proactive_fallback", "order_id": order_id}),
            0,
            None,
            None,
            0,
            None,
            json.dumps({"result_next_action": "execute_fallback_followup"}),
        ),
    )
    return job_id


def _update_job_state(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    state: str,
    blocked_reason: str,
    result_summary: str,
    result_next_action: str,
    failure_class: str = "retriable",
) -> None:
    row = conn.execute("SELECT trace FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    trace: dict[str, Any] = {}
    if row and row["trace"]:
        try:
            parsed = json.loads(str(row["trace"]))
            if isinstance(parsed, dict):
                trace = parsed
        except Exception:
            trace = {}
    now = float(time.time())
    trace.update(
        {
            "failure_class": failure_class,
            "result_summary": result_summary[:3000],
            "result_next_action": result_next_action,
            "autofix_monitor_updated_at": now,
        }
    )
    if state in ("failed", "cancelled"):
        trace.setdefault("slice_status", "failed_retriable")
        trace.setdefault("quality_gate_status", "failed_retriable")
    conn.execute(
        """
        UPDATE jobs
        SET state=?, blocked_reason=?, updated_at=?, due_at=NULL, trace=?
        WHERE job_id=?
        """,
        (state, blocked_reason, now, json.dumps(trace), job_id),
    )


def _trigger_autopilot(chat_id: int, *, log_path: Path) -> int:
    try:
        from bot import (
            SQLiteTaskStorage,
            OrchestratorQueue,
            _autopilot_tick,
            _load_config,
            _render_placeholders_obj,
            load_agent_profiles,
        )
    except Exception as exc:
        _log("autopilot_import_failed", path=log_path, error=str(exc))
        return 0
    try:
        cfg = _load_config()
        profiles = load_agent_profiles(cfg.orchestrator_agent_profiles)
        try:
            profiles = _render_placeholders_obj(profiles, ceo_name=cfg.ceo_name)
        except Exception:
            pass
        q = OrchestratorQueue(storage=SQLiteTaskStorage(cfg.orchestrator_db_path), role_profiles=profiles)
        created = int(_autopilot_tick(cfg=cfg, orch_q=q, profiles=profiles, chat_id=int(chat_id), now=time.time()))
        return created
    except Exception as exc:
        _log("autopilot_tick_failed", path=log_path, chat_id=int(chat_id), error=str(exc))
        return 0


def run_monitor(*, duration_s: int, interval_s: int, log_path: Path) -> int:
    start = float(time.time())
    end = start + float(duration_s)
    actions = 0
    loops = 0
    _log("monitor_started", path=log_path, duration_s=duration_s, interval_s=interval_s)

    while float(time.time()) < end:
        loops += 1
        now = float(time.time())
        try:
            conn = _connect()
            proactive_orders = _active_proactive_orders(conn)
            if not proactive_orders:
                _log("no_active_proactive_orders", path=log_path, loop=loops)
                conn.close()
                time.sleep(float(interval_s))
                continue

            for order in proactive_orders:
                order_id = str(order["order_id"] or "").strip()
                chat_id = int(order["chat_id"] or 0)
                open_jobs = _open_children(conn, order_id=order_id)
                by_role: dict[str, list[sqlite3.Row]] = {}
                for job in open_jobs:
                    by_role.setdefault(str(job["role"] or ""), []).append(job)

                running_impl = [j for j in by_role.get("implementer_local", []) if str(j["state"] or "") == "running"]
                running_arch = [j for j in by_role.get("architect_local", []) if str(j["state"] or "") == "running"]
                blocked_only = [j for j in open_jobs if str(j["state"] or "") in ("blocked", "blocked_approval")]
                waiting_dep = [j for j in open_jobs if str(j["state"] or "") == "waiting_deps"]

                # Strict WIP preference: keep implementer active and avoid parallel architect churn.
                if running_impl and running_arch:
                    for arch in running_arch:
                        _update_job_state(
                            conn,
                            job_id=str(arch["job_id"]),
                            state="cancelled",
                            blocked_reason="autofix_wip_focus",
                            result_summary="Autofix monitor cancelled architect while implementer is running to keep WIP focused.",
                            result_next_action="wait_for_implementer_result",
                        )
                        actions += 1
                        _log("cancel_arch_for_wip_focus", path=log_path, order_id=order_id, job_id=str(arch["job_id"]))

                # If architect keeps looping without implementer attempts, cancel stale architect to force reseed.
                recent_arch = _recent_role_jobs(conn, order_id=order_id, role="architect_local", seconds=3600)
                recent_impl = _recent_role_jobs(conn, order_id=order_id, role="implementer_local", seconds=900)
                if (not running_impl) and running_arch and len(recent_arch) >= 2 and len(recent_impl) == 0:
                    for arch in running_arch:
                        age_s = max(0.0, now - float(arch["updated_at"] or arch["created_at"] or now))
                        if age_s < 900.0:
                            continue
                        _update_job_state(
                            conn,
                            job_id=str(arch["job_id"]),
                            state="cancelled",
                            blocked_reason="autofix_arch_loop_breaker",
                            result_summary=f"Autofix monitor cancelled architect after {int(age_s)}s without implementer progress (loop-break).",
                            result_next_action="force_direct_implementer",
                        )
                        actions += 1
                        _log(
                            "cancel_arch_loop",
                            path=log_path,
                            order_id=order_id,
                            job_id=str(arch["job_id"]),
                            age_s=int(age_s),
                            recent_arch=len(recent_arch),
                            recent_impl=len(recent_impl),
                        )

                # Recover stale running implementer entries with no progress for too long.
                for impl in running_impl:
                    age_s = max(0.0, now - float(impl["updated_at"] or impl["created_at"] or now))
                    if age_s < 2400.0:
                        continue
                    _update_job_state(
                        conn,
                        job_id=str(impl["job_id"]),
                        state="queued",
                        blocked_reason="autofix_requeue_stale_impl",
                        result_summary=f"Autofix monitor requeued stale implementer after {int(age_s)}s without state update.",
                        result_next_action="retry_execution",
                    )
                    actions += 1
                    _log("requeue_stale_implementer", path=log_path, order_id=order_id, job_id=str(impl["job_id"]), age_s=int(age_s))

                # FINAL SWEEP: when only blocked jobs remain (and none are waiting_deps), clear stale blockers.
                if blocked_only and (len(blocked_only) == len(open_jobs)) and (not waiting_dep):
                    for job in blocked_only:
                        job_id = str(job["job_id"] or "")
                        role = str(job["role"] or "")
                        created_at = float(job["created_at"] or now)
                        blocked_reason = _job_blocked_reason(conn, job_id=job_id)
                        waiting_job_id = _extract_waiting_job_id(blocked_reason)
                        if waiting_job_id and _job_is_terminal(conn, job_id=waiting_job_id):
                            _update_job_state(
                                conn,
                                job_id=job_id,
                                state="cancelled",
                                blocked_reason="autofix_superseded_blocker",
                                result_summary=f"Autofix monitor cancelled blocked job because dependency {waiting_job_id[:8]} is already terminal.",
                                result_next_action="reseed_or_enqueue_followup",
                            )
                            actions += 1
                            _log(
                                "cancel_superseded_blocked_only",
                                path=log_path,
                                order_id=order_id,
                                job_id=job_id,
                                dependency_job_id=waiting_job_id,
                            )
                            continue

                        if _is_superseded_by_newer_terminal_same_role(
                            conn,
                            order_id=order_id,
                            role=role,
                            created_at=created_at,
                        ):
                            _update_job_state(
                                conn,
                                job_id=job_id,
                                state="cancelled",
                                blocked_reason="autofix_superseded_by_newer_terminal",
                                result_summary=f"Autofix monitor cancelled blocked job because a newer terminal job exists for role={role or 'unknown'}.",
                                result_next_action="reseed_or_enqueue_followup",
                            )
                            actions += 1
                            _log(
                                "cancel_superseded_by_newer_terminal",
                                path=log_path,
                                order_id=order_id,
                                job_id=job_id,
                                role=role or "unknown",
                            )
                            continue

                        age_s = max(0.0, now - float(job["updated_at"] or job["created_at"] or now))
                        threshold_s = _blocked_cleanup_age_threshold(blocked_reason)
                        if age_s < threshold_s:
                            continue
                        _update_job_state(
                            conn,
                            job_id=job_id,
                            state="cancelled",
                            blocked_reason="autofix_stale_blocked_only",
                            result_summary=f"Autofix monitor cancelled stale blocked-only job after {int(age_s)}s (threshold {int(threshold_s)}s) to unblock proactive lane.",
                            result_next_action="reseed_or_enqueue_followup",
                        )
                        actions += 1
                        _log(
                            "cancel_stale_blocked_only",
                            path=log_path,
                            order_id=order_id,
                            job_id=str(job["job_id"]),
                            age_s=int(age_s),
                        )

                conn.commit()

                # Keep proactive lane moving if no open work exists.
                refreshed_open = _open_children(conn, order_id=order_id)
                if not refreshed_open:
                    created = _trigger_autopilot(chat_id, log_path=log_path) if chat_id > 0 else 0
                    if created:
                        actions += int(created)
                        _log("autopilot_reseed", path=log_path, order_id=order_id, created=int(created))
                    else:
                        terminal_count = _terminal_children_count(conn, order_id=order_id)
                        recent_blocked_cleanup = _recent_cancelled_blocked_cleanup_count(conn, order_id=order_id)
                        if recent_blocked_cleanup > 0:
                            followup_job_id = _enqueue_fallback_followup(conn, order_id=order_id, chat_id=chat_id)
                            _set_order_phase_planning(conn, order_id=order_id)
                            conn.commit()
                            actions += 1
                            refreshed_open = _open_children(conn, order_id=order_id)
                            _log(
                                "enqueued_post_blocked_cleanup_followup",
                                path=log_path,
                                order_id=order_id,
                                job_id=followup_job_id,
                                cleaned_blocked=recent_blocked_cleanup,
                            )
                        elif terminal_count > 0 and _close_order_done(conn, order_id=order_id):
                            conn.commit()
                            actions += 1
                            _log(
                                "closed_idle_order_no_open_jobs",
                                path=log_path,
                                order_id=order_id,
                                terminal_children=terminal_count,
                            )
                        else:
                            followup_job_id = _enqueue_fallback_followup(conn, order_id=order_id, chat_id=chat_id)
                            _set_order_phase_planning(conn, order_id=order_id)
                            conn.commit()
                            actions += 1
                            refreshed_open = _open_children(conn, order_id=order_id)
                            _log(
                                "enqueued_fallback_followup",
                                path=log_path,
                                order_id=order_id,
                                job_id=followup_job_id,
                            )
                _log(
                    "loop_status",
                    path=log_path,
                    order_id=order_id,
                    open_jobs=len(refreshed_open),
                    running_implementer=len([j for j in refreshed_open if str(j["role"] or "") == "implementer_local" and str(j["state"] or "") == "running"]),
                    running_architect=len([j for j in refreshed_open if str(j["role"] or "") == "architect_local" and str(j["state"] or "") == "running"]),
                )

            conn.close()
        except Exception as exc:
            _log("monitor_loop_error", path=log_path, error=str(exc), loop=loops)

        time.sleep(float(interval_s))

    _log("monitor_finished", path=log_path, loops=loops, actions=actions, duration_s=duration_s)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="3h proactive monitor with auto-heal actions for local lane.")
    parser.add_argument("--duration", type=int, default=3 * 3600, help="Monitoring duration in seconds (default 10800).")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval in seconds.")
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG), help="JSONL log output path.")
    args = parser.parse_args()

    _load_env_file(ROOT / "codexbot.env")
    log_path = Path(args.log).resolve()
    return run_monitor(duration_s=max(120, int(args.duration)), interval_s=max(20, int(args.interval)), log_path=log_path)


if __name__ == "__main__":
    raise SystemExit(main())
