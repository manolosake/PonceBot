#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _to_dt(epoch: Any) -> str:
    try:
        n = float(epoch)
    except (TypeError, ValueError):
        return "-"
    if n <= 0:
        return "-"
    return datetime.fromtimestamp(n, tz=timezone.utc).isoformat()


def _loads_json(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(str(raw))
    except Exception:
        return {}
    return val if isinstance(val, dict) else {}


def _pick_order_ids(
    conn: sqlite3.Connection,
    *,
    order_id: str | None,
    job_id: str | None,
    source_message_id: int | None,
    chat_id: int | None,
    latest: int,
) -> list[str]:
    if order_id:
        return [order_id]

    if job_id:
        row = conn.execute(
            "SELECT job_id, parent_job_id FROM jobs WHERE job_id = ? LIMIT 1",
            (job_id,),
        ).fetchone()
        if row:
            parent = str(row["parent_job_id"] or "").strip()
            return [parent or str(row["job_id"])]
        return []

    if source_message_id is not None:
        if chat_id is None:
            rows = conn.execute(
                "SELECT order_id FROM ceo_orders WHERE source_message_id = ? ORDER BY updated_at DESC",
                (int(source_message_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT order_id FROM ceo_orders WHERE source_message_id = ? AND chat_id = ? ORDER BY updated_at DESC",
                (int(source_message_id), int(chat_id)),
            ).fetchall()
        return [str(r["order_id"]) for r in rows]

    if chat_id is None:
        rows = conn.execute(
            "SELECT order_id FROM ceo_orders ORDER BY updated_at DESC LIMIT ?",
            (max(1, int(latest)),),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT order_id FROM ceo_orders WHERE chat_id = ? ORDER BY updated_at DESC LIMIT ?",
            (int(chat_id), max(1, int(latest))),
        ).fetchall()
    return [str(r["order_id"]) for r in rows]


def _print_order_trace(conn: sqlite3.Connection, order_id: str) -> None:
    print("=" * 100)
    print(f"ORDER {order_id}")

    order = conn.execute(
        """
        SELECT order_id, chat_id, title, body, status, phase, priority, intent_type,
               source_message_id, reply_to_message_id, project_id, created_at, updated_at
        FROM ceo_orders
        WHERE order_id = ?
        LIMIT 1
        """,
        (order_id,),
    ).fetchone()

    if order is None:
        print("[warn] order not found in ceo_orders")
    else:
        print("\n[order]")
        print(
            f"chat_id={order['chat_id']} status={order['status']} phase={order['phase']} "
            f"priority={order['priority']} intent={order['intent_type']}"
        )
        print(
            f"source_message_id={order['source_message_id']} "
            f"reply_to_message_id={order['reply_to_message_id']} project_id={order['project_id']}"
        )
        print(f"created_at={_to_dt(order['created_at'])} updated_at={_to_dt(order['updated_at'])}")
        title = str(order["title"] or "").strip()
        body = str(order["body"] or "").strip().replace("\n", " ")
        if len(body) > 280:
            body = body[:280] + "..."
        print(f"title={title}")
        print(f"body={body}")

    jobs = conn.execute(
        """
        SELECT job_id, parent_job_id, role, request_type, state, source, chat_id,
               reply_to_message_id, depends_on, trace, created_at, updated_at
        FROM jobs
        WHERE job_id = ? OR parent_job_id = ?
        ORDER BY created_at ASC
        """,
        (order_id, order_id),
    ).fetchall()

    print("\n[jobs timeline]")
    if not jobs:
        print("(none)")
    else:
        for j in jobs:
            trace = _loads_json(j["trace"])
            deps = []
            try:
                raw_deps = json.loads(j["depends_on"] or "[]")
                if isinstance(raw_deps, list):
                    deps = [str(x)[:8] for x in raw_deps]
            except Exception:
                deps = []

            msg_src = trace.get("source_message_id")
            msg_reply = trace.get("reply_to_message_id")
            delegated_key = trace.get("delegated_key")
            result_status = trace.get("result_status")
            summary = str(trace.get("result_summary") or "").replace("\n", " ").strip()
            if len(summary) > 140:
                summary = summary[:140] + "..."

            print(
                f"- {_to_dt(j['created_at'])} {str(j['job_id'])[:8]} role={j['role']} req={j['request_type']} "
                f"state={j['state']} parent={str(j['parent_job_id'] or '-')[:8]} deps={deps or '-'}"
            )
            print(
                f"  source={j['source']} chat_id={j['chat_id']} reply_to={j['reply_to_message_id']} "
                f"trace.source_message_id={msg_src} trace.reply_to_message_id={msg_reply} delegated_key={delegated_key}"
            )
            if result_status or summary:
                print(f"  result_status={result_status or '-'} summary={summary or '-'}")

    decisions = conn.execute(
        """
        SELECT ts, order_id, job_id, kind, state, summary, next_action
        FROM decision_log
        WHERE order_id = ?
        ORDER BY ts ASC
        """,
        (order_id,),
    ).fetchall()

    print("\n[decision log]")
    if not decisions:
        print("(none)")
    else:
        for d in decisions:
            summary = str(d["summary"] or "").replace("\n", " ").strip()
            if len(summary) > 180:
                summary = summary[:180] + "..."
            print(
                f"- {_to_dt(d['ts'])} job={str(d['job_id'])[:8]} kind={d['kind']} state={d['state']} "
                f"next_action={d['next_action'] or '-'}"
            )
            print(f"  summary={summary}")

    edges = conn.execute(
        """
        SELECT ts, from_job_id, to_job_id, edge_type, to_role, to_key
        FROM delegation_log
        WHERE root_ticket_id = ?
        ORDER BY ts ASC
        """,
        (order_id,),
    ).fetchall()

    print("\n[delegation graph edges]")
    if not edges:
        print("(none)")
    else:
        for e in edges:
            print(
                f"- {_to_dt(e['ts'])} {str(e['from_job_id'])[:8]} -> {str(e['to_job_id'])[:8]} "
                f"type={e['edge_type']} role={e['to_role'] or '-'} key={e['to_key'] or '-'}"
            )

    audits = conn.execute(
        """
        SELECT ts, event_type, actor, details
        FROM audit_log
        ORDER BY ts DESC
        LIMIT 1200
        """
    ).fetchall()

    print("\n[audit events referencing order_id]")
    found = 0
    for a in audits:
        details = _loads_json(a["details"])
        if str(details.get("order_id") or "").strip() != order_id:
            continue
        found += 1
        print(f"- {_to_dt(a['ts'])} event={a['event_type']} actor={a['actor']} details={details}")
    if found == 0:
        print("(none)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Trace Telegram -> Jarvis -> agents flow from jobs.sqlite")
    ap.add_argument("--db", default="/home/aponce/codexbot/data/jobs.sqlite", help="Path to jobs.sqlite")
    ap.add_argument("--order-id", default="", help="Exact order_id to trace")
    ap.add_argument("--job-id", default="", help="Any job_id; resolver picks root order/job")
    ap.add_argument("--source-message-id", type=int, default=None, help="Telegram inbound message_id")
    ap.add_argument("--chat-id", type=int, default=None, help="Telegram chat_id filter")
    ap.add_argument("--latest", type=int, default=1, help="How many latest orders to inspect when no filter")
    args = ap.parse_args()

    db = Path(args.db).expanduser().resolve()
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    order_ids = _pick_order_ids(
        conn,
        order_id=(args.order_id.strip() or None),
        job_id=(args.job_id.strip() or None),
        source_message_id=args.source_message_id,
        chat_id=args.chat_id,
        latest=max(1, int(args.latest)),
    )

    if not order_ids:
        print("No matching orders/jobs found.")
        return 1

    for oid in order_ids:
        _print_order_trace(conn, oid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
