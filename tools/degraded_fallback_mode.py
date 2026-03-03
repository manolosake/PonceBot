#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path


FLAG_KEY = "backend.degraded_fallback_mode"


def _ensure_status_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS status_cache (
            cache_key TEXT PRIMARY KEY,
            updated_at REAL NOT NULL,
            ttl_seconds INTEGER NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )


def _set_mode(db_path: Path, enabled: bool, reason: str) -> dict[str, object]:
    now = time.time()
    payload = {
        "enabled": enabled,
        "reason": reason.strip() or "manual_toggle",
        "updated_at": now,
        "rollback_command": (
            f"python3 tools/degraded_fallback_mode.py disable --db {db_path}"
            if enabled
            else f"python3 tools/degraded_fallback_mode.py enable --db {db_path} --reason rollback"
        ),
    }
    with sqlite3.connect(str(db_path)) as conn:
        _ensure_status_cache_table(conn)
        conn.execute(
            """
            INSERT INTO status_cache(cache_key, updated_at, ttl_seconds, payload)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                updated_at=excluded.updated_at,
                ttl_seconds=excluded.ttl_seconds,
                payload=excluded.payload
            """,
            (FLAG_KEY, now, 7 * 24 * 3600, json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()
    return payload


def _status(db_path: Path) -> dict[str, object]:
    with sqlite3.connect(str(db_path)) as conn:
        _ensure_status_cache_table(conn)
        row = conn.execute(
            "SELECT payload FROM status_cache WHERE cache_key = ?",
            (FLAG_KEY,),
        ).fetchone()
    if not row:
        return {
            "enabled": False,
            "reason": "not_set",
            "updated_at": None,
            "rollback_command": f"python3 tools/degraded_fallback_mode.py enable --db {db_path} --reason emergency",
        }
    payload = json.loads(row[0])
    if "rollback_command" not in payload:
        payload["rollback_command"] = f"python3 tools/degraded_fallback_mode.py disable --db {db_path}"
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Toggle/query temporary degraded fallback mode.")
    ap.add_argument("action", choices=["enable", "disable", "status"])
    ap.add_argument(
        "--db",
        default="/home/aponce/codexbot/data/jobs.sqlite",
        help="SQLite path used by the orchestrator",
    )
    ap.add_argument("--reason", default="manual_toggle", help="Reason for enable/disable")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        print(json.dumps({"status": "ERROR", "error": f"db_not_found: {db_path}"}))
        return 2

    if args.action == "enable":
        payload = _set_mode(db_path, True, args.reason)
    elif args.action == "disable":
        payload = _set_mode(db_path, False, args.reason)
    else:
        payload = _status(db_path)

    print(json.dumps({"status": "OK", "mode": payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
