#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.storage import SQLiteTaskStorage


def _default_db_path() -> Path:
    env = os.environ.get("BOT_ORCHESTRATOR_DB_PATH", "").strip()
    if env:
        return Path(env).expanduser()
    return (Path(__file__).resolve().parents[1] / "data" / "jobs.sqlite")


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot operational hard reset for PonceBot orchestrator")
    parser.add_argument("--db", default=str(_default_db_path()), help="Path to jobs sqlite database")
    parser.add_argument("--reason", default="hard_reset_bootstrap", help="Audit reason")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    storage = SQLiteTaskStorage(db_path)
    result = storage.hard_reset_bootstrap(reason=str(args.reason or "hard_reset_bootstrap"))

    payload = {
        "ok": True,
        "db": str(db_path),
        "reason": str(args.reason or "hard_reset_bootstrap"),
        "jobs_cancelled": int(result.get("jobs_cancelled", 0)),
        "orders_done": int(result.get("orders_done", 0)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
