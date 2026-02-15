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


_ALLOWED_TASK_STATES = (
    "queued",
    "running",
    "blocked",
    "done",
    "failed",
    "cancelled",
)


def _coalesce_value(value: Any, default: Any) -> Any:
    return default if value is None else value


def _coerce_bool(value: Any, default: bool = False) -> bool:
    try:
        return bool(int(value))
    except Exception:
        return default


def _coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _coerce_json_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


class SQLiteTaskStorage:
    """Persistent store for orchestrator tasks and audit events.

    The schema is intentionally simple and resilient: a single `jobs` table for task state,
    per-job audit events, role pause controls, cost caps, and approval decisions.
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
                    is_autonomous INTEGER NOT NULL DEFAULT 0,
                    parent_job_id TEXT,
                    owner TEXT,
                    depends_on TEXT NOT NULL DEFAULT '[]',
                    ttl_seconds INTEGER,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 0,
                    labels TEXT NOT NULL DEFAULT '{}',
                    requires_review INTEGER NOT NULL DEFAULT 0,
                    artifacts_dir TEXT,
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approver_log (
                    job_id TEXT PRIMARY KEY,
                    approved_by INTEGER,
                    approved_at REAL NOT NULL,
                    reason TEXT,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cost_caps (
                    role TEXT PRIMARY KEY,
                    max_cost_window_usd REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

            # Backward-compatible migration for older DBs that may exist in the field.
            self._migrate_jobs_table(conn)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state_due ON jobs(state, due_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_role_state_priority ON jobs(role, state, priority, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_chat_state ON jobs(chat_id, state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_autonomous_due ON jobs(state, is_autonomous, due_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner, state)")
            conn.commit()

    def _migrate_jobs_table(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        required_cols = {
            "is_autonomous": "INTEGER NOT NULL DEFAULT 0",
            "parent_job_id": "TEXT",
            "owner": "TEXT",
            "depends_on": "TEXT NOT NULL DEFAULT '[]'",
            "ttl_seconds": "INTEGER",
            "retry_count": "INTEGER NOT NULL DEFAULT 0",
            "max_retries": "INTEGER NOT NULL DEFAULT 0",
            "labels": "TEXT NOT NULL DEFAULT '{}'",
            "requires_review": "INTEGER NOT NULL DEFAULT 0",
            "artifacts_dir": "TEXT",
        }
        for col, ddl in required_cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ddl}")

    def submit_task(self, task: Task) -> str:
        with self._lock:
            job_id = task.job_id or str(uuid.uuid4())
            with self._connect() as conn:
                columns = (
                    "job_id",
                    "source",
                    "role",
                    "input_text",
                    "request_type",
                    "priority",
                    "model",
                    "effort",
                    "mode_hint",
                    "requires_approval",
                    "max_cost_window_usd",
                    "created_at",
                    "updated_at",
                    "due_at",
                    "state",
                    "chat_id",
                    "user_id",
                    "reply_to_message_id",
                    "is_autonomous",
                    "parent_job_id",
                    "owner",
                    "depends_on",
                    "ttl_seconds",
                    "retry_count",
                    "max_retries",
                    "labels",
                    "requires_review",
                    "artifacts_dir",
                    "trace",
                )
                values = (
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
                    1 if task.is_autonomous else 0,
                    task.parent_job_id,
                    task.owner,
                    json.dumps(list(task.depends_on or []), ensure_ascii=False),
                    None if task.ttl_seconds is None else int(task.ttl_seconds),
                    int(task.retry_count),
                    int(task.max_retries),
                    json.dumps(dict(task.labels or {}), ensure_ascii=False),
                    1 if task.requires_review else 0,
                    task.artifacts_dir,
                    task.trace_json(),
                )
                query = (
                    "INSERT OR REPLACE INTO jobs ("
                    + ", ".join(columns)
                    + ") VALUES ("
                    + ", ".join("?" for _ in columns)
                    + ")"
                )
                conn.execute(
                    query,
                    values,
                )
                conn.commit()
                self._append_event(conn, job_id, "submitted", {"state": task.state, "role": task.role})
            return job_id

    def submit_batch(self, tasks: list[Task]) -> list[str]:
        return [self.submit_task(task) for task in tasks]

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
        return self._claim(
            role=role,
            limit_roles=limit_roles,
            max_parallel_by_role=max_parallel_by_role,
            require_autonomous=None,
            budget=None,
            max_rows=1,
        )

    def dequeue_with_budget(
        self,
        *,
        role: str | None = None,
        limit_roles: set[str] | None = None,
        budget: float | None = None,
        max_parallel_by_role: dict[str, int] | None = None,
    ) -> Task | None:
        return self._claim(
            role=role,
            limit_roles=limit_roles,
            max_parallel_by_role=max_parallel_by_role,
            require_autonomous=None,
            budget=budget,
            max_rows=1,
        )

    def claim_autonomous_due_jobs(self, *, limit: int = 5) -> list[Task]:
        now = time.time()
        tasks = self._claim(
            role=None,
            limit_roles=None,
            max_parallel_by_role=None,
            require_autonomous=True,
            budget=None,
            max_rows=max(1, int(limit)),
        )
        if tasks is None:
            return []
        return tasks if isinstance(tasks, list) else [tasks]

    def _claim(
        self,
        *,
        role: str | None,
        limit_roles: set[str] | None,
        max_parallel_by_role: dict[str, int] | None,
        require_autonomous: bool | None,
        budget: float | None,
        max_rows: int,
    ) -> Task | list[Task] | None:
        if max_parallel_by_role is None:
            max_parallel_by_role = {}

        now = time.time()
        with self._lock:
            with self._connect() as conn:
                if self._is_global_pause(conn):
                    return None

                roles_csv = None
                if role:
                    roles_csv = str(role)
                elif limit_roles:
                    if not limit_roles:
                        return None
                    roles_csv = ",".join(sorted(limit_roles))

                where = ["state = ?", "(due_at IS NULL OR due_at <= ?)"]
                params: list[Any] = ["queued", now]

                if roles_csv:
                    placeholders = ",".join("?" for _ in roles_csv.split(","))
                    where.append(f"role IN ({placeholders})")
                    params.extend(roles_csv.split(","))

                if require_autonomous is not None:
                    where.append("is_autonomous = ?")
                    params.append(1 if require_autonomous else 0)

                if budget is not None:
                    where.append("(max_cost_window_usd <= ?)")
                    params.append(float(budget))

                for r in limit_roles or set():
                    if self._is_role_paused(conn, r):
                        where.append("role != ?")
                        params.append(r)

                base_sql = (
                    f"SELECT * FROM jobs WHERE {' AND '.join(where)} "
                    f"ORDER BY priority ASC, created_at ASC LIMIT {int(max_rows)}"
                )
                rows = conn.execute(base_sql, params).fetchall()
                if not rows:
                    return [] if max_rows > 1 else None

                out: list[Task] = []
                for row in rows:
                    role_name = str(row["role"])
                    if self._is_role_paused(conn, role_name):
                        continue

                    running = self._count_jobs(conn, role=role_name, state="running")
                    max_parallel = max(1, int(max_parallel_by_role.get(role_name, 1)))
                    if running >= max_parallel:
                        continue

                    updated = conn.execute(
                        "UPDATE jobs SET state = ?, updated_at = ? WHERE job_id = ? AND state = ?",
                        ("running", now, row["job_id"], "queued"),
                    )
                    if updated.rowcount:
                        self._append_event(conn, row["job_id"], "dequeued", {"role": role_name})
                        conn.commit()
                        next_row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
                        task = self._row_to_task(next_row)
                        if task is not None:
                            out.append(task)
                    if len(out) >= max_rows:
                        break

                if not out:
                    return [] if max_rows > 1 else None
                if max_rows == 1:
                    return out[0]
                return out

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

    def requeue(self, job_id: str, *, reason: str | None = None) -> bool:
        metadata: dict[str, Any] = {}
        if reason:
            metadata["requeue_reason"] = reason
        return self.update_state(str(job_id), "queued", **metadata)

    def recover_stale_running(self) -> int:
        """
        On startup, return interrupted "running" jobs back to "queued" so workers can retry them.

        Returns:
            Number of jobs moved from `running` to `queued`.
        """
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("SELECT job_id FROM jobs WHERE state='running'").fetchall()
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

    def _is_global_pause(self, conn: sqlite3.Connection | None = None) -> bool:
        if conn is None:
            with self._connect() as c:
                return self._is_global_pause(c)
        row = conn.execute("SELECT is_paused FROM role_controls WHERE role = ?", ("__orchestrator__",)).fetchone()
        if row is None:
            return False
        return bool(int(row["is_paused"]))

    def get_role_health(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        with self._connect() as conn:
            rows = conn.execute("SELECT role, state, COUNT(1) as c FROM jobs GROUP BY role, state").fetchall()
            for row in rows:
                role = str(row["role"])
                state = str(row["state"])
                out.setdefault(role, {})[state] = int(row["c"])

            for role in self._known_roles(conn):
                out.setdefault(role, {})
                for state in _ALLOWED_TASK_STATES:
                    out[role].setdefault(state, 0)

            for role in out:
                out[role].setdefault("paused", 1 if self._is_role_paused(conn, role) else 0)
            return out

    def get_role_backlog(self, *, state: str | None = None) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        with self._connect() as conn:
            base_sql = "SELECT role, state, COUNT(1) as c FROM jobs"
            params: list[Any] = []
            if state:
                base_sql += " WHERE state = ?"
                params.append(str(state))
            base_sql += " GROUP BY role, state"
            rows = conn.execute(base_sql, params).fetchall()
            for row in rows:
                role = str(row["role"])
                st = str(row["state"])
                out.setdefault(role, {})[st] = int(row["c"])
            for role in self._known_roles(conn):
                out.setdefault(role, {})
                for st in _ALLOWED_TASK_STATES:
                    out[role].setdefault(st, 0)
            return out

    def set_cost_cap(self, role: str, cost_usd: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cost_caps(role, max_cost_window_usd, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(role) DO UPDATE SET max_cost_window_usd = ?, updated_at = ?",
                (str(role), float(cost_usd), time.time(), float(cost_usd), time.time()),
            )
            conn.commit()

    def get_cost_cap(self, role: str, *, default: float | None = None) -> float | None:
        with self._connect() as conn:
            row = conn.execute("SELECT max_cost_window_usd FROM cost_caps WHERE role = ?", (str(role),)).fetchone()
            if row is None:
                return default
            return _coerce_float(row["max_cost_window_usd"], default)

    def get_default_cost_cap(self) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(max_cost_window_usd) FROM cost_caps").fetchone()
            if row is None or row[0] is None:
                return 0.0
            try:
                return float(row[0])
            except Exception:
                return 0.0

    def peek(
        self,
        *,
        role: str | None = None,
        state: str | None = None,
        limit: int = 20,
    ) -> list[Task]:
        where: list[str] = []
        params: list[Any] = []
        if role:
            where.append("role = ?")
            params.append(role)
        if state:
            where.append("state = ?")
            params.append(state)
        clause = ""
        if where:
            clause = " WHERE " + " AND ".join(where)
        with self._connect() as conn:
            q = conn.execute(
                f"SELECT * FROM jobs{clause} ORDER BY created_at DESC LIMIT ?",
                [*params, int(limit)],
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

    def jobs_by_state(self, *, state: str, limit: int = 50) -> list[Task]:
        with self._connect() as conn:
            q = conn.execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY created_at DESC LIMIT ?",
                (str(state), int(limit)),
            ).fetchall()
            return [self._row_to_task(r) for r in q]

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

    def set_job_approved(self, job_id: str, *, approved: bool = True, approved_by: int | None = None, reason: str | None = None) -> bool:
        trace_payload: dict[str, Any] = {"approved": bool(approved), "approved_at": time.time()}
        if approved_by is not None:
            trace_payload["approved_by"] = int(approved_by)
        if reason:
            trace_payload["approved_reason"] = str(reason)

        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT job_id FROM jobs WHERE job_id = ?", (str(job_id),)).fetchone()
                if row is None:
                    return False
                conn.execute(
                    "INSERT OR REPLACE INTO approver_log(job_id, approved_by, approved_at, reason) VALUES (?, ?, ?, ?)",
                    (str(job_id), None if approved_by is None else int(approved_by), time.time(), reason),
                )
                updated = self.update_state(str(job_id), "queued", **trace_payload)
                if updated:
                    self._append_event(conn, str(job_id), "approved", {"approved": approved, "approved_by": approved_by, "reason": reason})
                    conn.commit()
                return updated

    def clear_job_approval(self, job_id: str) -> bool:
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
                trace.pop("approved_by", None)
                conn.execute("UPDATE jobs SET trace = ? WHERE job_id = ?", (json.dumps(trace, ensure_ascii=False), str(job_id)))
                conn.execute("DELETE FROM approver_log WHERE job_id = ?", (str(job_id),))
                conn.commit()
                self._append_event(conn, str(job_id), "approval_cleared", {})
                return True

    def _known_roles(self, conn: sqlite3.Connection) -> list[str]:
        role_rows = conn.execute("SELECT DISTINCT role FROM jobs").fetchall()
        control_rows = conn.execute("SELECT role FROM role_controls").fetchall()
        roles: set[str] = set()
        for r in role_rows:
            role = r["role"]
            if role and role != "__orchestrator__":
                roles.add(str(role))
        for r in control_rows:
            role = r["role"]
            if role and role != "__orchestrator__":
                roles.add(str(role))
        return sorted(roles)

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
            priority=_coerce_int(row["priority"], 2),
            model=row["model"],
            effort=row["effort"],
            mode_hint=row["mode_hint"],
            requires_approval=_coerce_bool(row["requires_approval"], False),
            max_cost_window_usd=_coerce_float(row["max_cost_window_usd"], 0.0),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            due_at=_coerce_float(row["due_at"], None),
            state=row["state"],
            chat_id=int(row["chat_id"]),
            user_id=None if row["user_id"] is None else int(row["user_id"]),
            reply_to_message_id=None if row["reply_to_message_id"] is None else int(row["reply_to_message_id"]),
            is_autonomous=_coerce_bool(row["is_autonomous"], False),
            parent_job_id=None if row["parent_job_id"] is None else str(row["parent_job_id"]),
            owner=None if row["owner"] is None else str(row["owner"]),
            depends_on=_coerce_json_list(row["depends_on"]),
            ttl_seconds=_coerce_int(row["ttl_seconds"], None),
            retry_count=_coerce_int(row["retry_count"], 0),
            max_retries=_coerce_int(row["max_retries"], 0),
            labels=_coerce_json_dict(row["labels"]),
            requires_review=_coerce_bool(row["requires_review"], False),
            artifacts_dir=None if row["artifacts_dir"] is None else str(row["artifacts_dir"]),
            trace=trace,
        )
