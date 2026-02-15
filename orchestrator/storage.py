from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any
import json
import sqlite3
import time
import uuid

from .schemas.task import Task


class SQLiteStorageError(RuntimeError):
    pass


class SQLiteTaskStorage:
    """Persistent store for orchestrator tasks and audit events.

    Storage is intentionally small and conservative: one DB file with explicit schema,
    FIFO-ish priority ordering, and role/approval metadata persisted for auditability.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    role TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    request_type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    effort TEXT NOT NULL,
                    mode_hint TEXT NOT NULL,
                    requires_approval INTEGER NOT NULL DEFAULT 0,
                    max_cost_window_usd REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    due_at REAL,
                    state TEXT NOT NULL,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER,
                    reply_to_message_id INTEGER,
                    trace TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    ts REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL DEFAULT 'system',
                    details TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(job_id, ts, event_type),
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS role_controls (
                    role TEXT PRIMARY KEY,
                    is_paused INTEGER NOT NULL DEFAULT 0
                )
                """
            )

            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state_due ON jobs(state, due_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_role_state_priority ON jobs(role, state, priority, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_chat_state ON jobs(chat_id, state)")
            conn.commit()

    def submit_task(self, task: Task) -> str:
        with self._lock:
            if not task.job_id:
                job_id = str(uuid.uuid4())
            else:
                job_id = task.job_id

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO jobs (
                        job_id, source, role, input_text, request_type, priority, model, effort,
                        mode_hint, requires_approval, max_cost_window_usd,
                        created_at, updated_at, due_at, state, chat_id, user_id,
                        reply_to_message_id, trace
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        str(task.source),
                        str(task.role),
                        str(task.input_text),
                        str(task.request_type),
                        int(task.priority),
                        str(task.model),
                        str(task.effort),
                        str(task.mode_hint),
                        1 if task.requires_approval else 0,
                        float(task.max_cost_window_usd),
                        float(task.created_at),
                        float(task.updated_at),
                        task.due_at,
                        str(task.state),
                        int(task.chat_id),
                        None if task.user_id is None else int(task.user_id),
                        None if task.reply_to_message_id is None else int(task.reply_to_message_id),
                        task.trace_json(),
                    ),
                )
                conn.commit()
                self._append_event(conn, job_id, "submitted", {"state": task.state, "role": task.role})
            return job_id

    def get_job(self, job_id: str) -> Task | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)).fetchone()
            return self._row_to_task(row) if row is not None else None

    def claim_next(
        self,
        *,
        role: str | None = None,
        limit_roles: set[str] | None = None,
        max_parallel_by_role: dict[str, int] | None = None,
    ) -> Task | None:
        if max_parallel_by_role is None:
            max_parallel_by_role = {}

        now = time.time()
        with self._lock:
            with self._connect() as conn:
                roles_csv = None
                if role:
                    roles_csv = str(role)
                elif limit_roles:
                    if not limit_roles:
                        return None
                    roles_csv = ",".join(sorted(limit_roles))

                where = ["state = ?", "(due_at IS NULL OR due_at <= ?)" ]
                params: list[Any] = ["queued", now]

                if roles_csv:
                    placeholders = ",".join("?" for _ in roles_csv.split(",") )
                    where.append(f"role IN ({placeholders})")
                    params.extend(roles_csv.split(","))

                for r in limit_roles or set():
                    if self._is_role_paused(conn, r):
                        where.append("role != ?")
                        params.append(r)

                base_sql = f"SELECT * FROM jobs WHERE {' AND '.join(where)} ORDER BY priority ASC, created_at ASC LIMIT 1"
                row = conn.execute(base_sql, params).fetchone()
                if row is None:
                    return None

                role_name = str(row["role"])
                running = self._count_jobs(conn, role=role_name, state="running")
                max_parallel = max(1, int(max_parallel_by_role.get(role_name, 1)))
                if running >= max_parallel:
                    return None

                if self._is_role_paused(conn, role_name):
                    return None

                conn.execute(
                    "UPDATE jobs SET state = ?, updated_at = ? WHERE job_id = ? AND state = ?",
                    ("running", now, row["job_id"], "queued"),
                )
                if conn.total_changes:
                    self._append_event(conn, row["job_id"], "dequeued", {"role": role_name})
                    conn.commit()
                    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
                    return self._row_to_task(row)
                conn.commit()
                return None

    def update_state(self, job_id: str, state: str, **metadata: Any) -> bool:
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT state, trace FROM jobs WHERE job_id = ?", (str(job_id),)).fetchone()
                if row is None:
                    return False

                trace = Task.from_trace_json(row["trace"])
                trace.update(metadata)
                conn.execute(
                    "UPDATE jobs SET state = ?, updated_at = ?, trace = ? WHERE job_id = ?",
                    (str(state), now, json.dumps(trace, ensure_ascii=False), str(job_id)),
                )
                if conn.total_changes:
                    self._append_event(conn, str(job_id), f"state:{state}", metadata)
                    conn.commit()
                    return True
                return False

    def cancel(self, job_id: str) -> bool:
        return self.update_state(str(job_id), "cancelled", reason="user_requested")

    def recover_stale_running(self) -> int:
        """
        On startup, return interrupted "running" jobs back to "queued" so workers can retry them.

        Returns:
            Number of jobs moved from `running` to `queued`.
        """
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT job_id FROM jobs WHERE state='running'",
                ).fetchall()
                if not rows:
                    return 0
                total = 0
                for row in rows:
                    jid = row["job_id"]
                    updated = conn.execute(
                        "UPDATE jobs SET state=?, updated_at=? WHERE job_id=?",
                        ("queued", time.time(), str(jid)),
                    )
                    if updated.rowcount:
                        self._append_event(conn, str(jid), "recovered", {"from": "running", "to": "queued"})
                        total += 1
                conn.commit()
                return total

    def pause_role(self, role: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO role_controls(role, is_paused) VALUES (?, 1) ON CONFLICT(role) DO UPDATE SET is_paused = 1",
                (str(role),),
            )
            conn.commit()

    def resume_role(self, role: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO role_controls(role, is_paused) VALUES (?, 0) ON CONFLICT(role) DO UPDATE SET is_paused = 0",
                (str(role),),
            )
            conn.commit()

    def pause_all_roles(self) -> None:
        self._set_global_pause(paused=True)

    def resume_all_roles(self) -> None:
        self._set_global_pause(paused=False)

    def is_orchestrator_paused(self) -> bool:
        return self._is_global_pause()

    def is_paused_globally(self) -> bool:
        return self.is_orchestrator_paused()

    def cancel_running_jobs(self, *, reason: str = "emergency_stop") -> int:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("SELECT job_id, trace FROM jobs WHERE state = 'running'").fetchall()
                if not rows:
                    return 0

                total = 0
                now = time.time()
                for row in rows:
                    trace = Task.from_trace_json(row["trace"])
                    if not isinstance(trace, dict):
                        trace = {}
                    trace["cancel_reason"] = reason
                    updated = conn.execute(
                        "UPDATE jobs SET state = ?, updated_at = ?, trace = ? WHERE job_id = ? AND state = 'running'",
                        ("cancelled", now, json.dumps(trace, ensure_ascii=False), row["job_id"]),
                    )
                    if updated.rowcount:
                        self._append_event(conn, str(row["job_id"]), "emergency_cancel", {"reason": reason})
                        total += 1
                conn.commit()
                return total

    def is_role_paused(self, role: str) -> bool:
        with self._connect() as conn:
            return self._is_role_paused(conn, str(role))

    def _is_role_paused(self, conn: sqlite3.Connection, role: str) -> bool:
        row = conn.execute("SELECT is_paused FROM role_controls WHERE role = ?", (str(role),)).fetchone()
        if row is None:
            return False
        return bool(int(row["is_paused"]))

    def _set_global_pause(self, *, paused: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO role_controls(role, is_paused) VALUES (?, ?) ON CONFLICT(role) DO UPDATE SET is_paused = ?",
                ("__orchestrator__", 1 if paused else 0, 1 if paused else 0),
            )
            conn.commit()

    def _is_global_pause(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT is_paused FROM role_controls WHERE role = ?",
                ("__orchestrator__",),
            ).fetchone()
        if row is None:
            return False
        return bool(int(row["is_paused"]))

    def get_role_health(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        with self._connect() as conn:
            states = ["queued", "running", "blocked", "done", "failed", "cancelled"]
            rows = conn.execute(
                "SELECT role, state, COUNT(1) as c FROM jobs GROUP BY role, state"
            ).fetchall()
            for row in rows:
                role = str(row["role"])
                state = str(row["state"])
                out.setdefault(role, {})[state] = int(row["c"])
            for role in self._known_roles(conn):
                out.setdefault(role, {})
                for state in states:
                    out[role].setdefault(state, 0)
            # ensure all known roles appear even with no jobs
            for role in out:
                out[role].setdefault("paused", 1 if self._is_role_paused(conn, role) else 0)
        return out

    def _known_roles(self, conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute("SELECT DISTINCT role FROM jobs").fetchall()
        roles = [str(r["role"]) for r in rows if r and r["role"]]
        if not roles:
            return []
        return roles

    def jobs_by_state(self, *, state: str, limit: int = 50) -> list[Task]:
        with self._connect() as conn:
            q = conn.execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY created_at DESC LIMIT ?",
                (str(state), int(limit)),
            ).fetchall()
            return [self._row_to_task(r) for r in q]

    def queued_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(1) as c FROM jobs WHERE state = 'queued'").fetchone()
            return int(row["c"]) if row else 0

    def running_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(1) as c FROM jobs WHERE state = 'running'").fetchone()
            return int(row["c"]) if row else 0

    def _count_jobs(self, conn: sqlite3.Connection, *, role: str, state: str) -> int:
        row = conn.execute(
            "SELECT COUNT(1) as c FROM jobs WHERE role = ? AND state = ?",
            (str(role), str(state)),
        ).fetchone()
        return int(row["c"]) if row else 0

    def _append_event(self, conn: sqlite3.Connection, job_id: str, event_type: str, details: dict[str, Any] | str) -> None:
        if isinstance(details, str):
            detail_payload = details
        else:
            detail_payload = json.dumps(details or {}, ensure_ascii=False)
        conn.execute(
            "INSERT INTO job_events (job_id, ts, event_type, details) VALUES (?, ?, ?, ?)",
            (str(job_id), time.time(), str(event_type), detail_payload),
        )

    def set_job_approved(self, job_id: str, approved: bool = True) -> bool:
        """
        Mark a job as approved. Returns True if job exists.
        """
        trace_payload: dict[str, Any] = {"approved": bool(approved), "approved_at": time.time()}
        return self.update_state(job_id, "queued", **trace_payload)

    def clear_job_approval(self, job_id: str) -> bool:
        """
        Remove explicit approval marker from a job while preserving other trace fields.
        """
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT trace FROM jobs WHERE job_id=?", (str(job_id),)).fetchone()
                if row is None:
                    return False
                trace = Task.from_trace_json(row["trace"])
                if not isinstance(trace, dict):
                    trace = {}
                trace.pop("approved", None)
                trace.pop("approved_at", None)
                conn.execute(
                    "UPDATE jobs SET trace = ? WHERE job_id = ?",
                    (json.dumps(trace, ensure_ascii=False), str(job_id)),
                )
                conn.commit()
                self._append_event(conn, str(job_id), "approval_cleared", {})
                return True

    def _row_to_task(self, row: sqlite3.Row | None) -> Task | None:
        if row is None:
            return None
        trace = Task.from_trace_json(row["trace"])
        return Task(
            job_id=row["job_id"],
            source=row["source"],
            role=row["role"],
            input_text=row["input_text"],
            request_type=row["request_type"],
            priority=int(row["priority"]),
            model=row["model"],
            effort=row["effort"],
            mode_hint=row["mode_hint"],
            requires_approval=bool(int(row["requires_approval"])),
            max_cost_window_usd=float(row["max_cost_window_usd"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            due_at=row["due_at"],
            state=row["state"],
            chat_id=int(row["chat_id"]),
            user_id=None if row["user_id"] is None else int(row["user_id"]),
            reply_to_message_id=None if row["reply_to_message_id"] is None else int(row["reply_to_message_id"]),
            trace=trace,
        )
