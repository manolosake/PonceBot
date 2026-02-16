from __future__ import annotations

from contextlib import contextmanager
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
        # NOTE: Callers should prefer `_conn()` which closes connections reliably.
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=3000;")
        return conn

    @contextmanager
    def _conn(self) -> Any:
        conn = self._connect()
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _init_db(self) -> None:
        with self._conn() as conn:
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
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(chat_id, role)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_leases (
                    role TEXT NOT NULL,
                    slot INTEGER NOT NULL,
                    job_id TEXT NOT NULL,
                    leased_at REAL NOT NULL,
                    PRIMARY KEY(role, slot),
                    UNIQUE(job_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runbook_state (
                    runbook_id TEXT PRIMARY KEY,
                    last_run_at REAL NOT NULL DEFAULT 0
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ceo_orders (
                    order_id TEXT PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

            # Backward-compatible migration for older DBs that may exist in the field.
            self._migrate_jobs_table(conn)
            self._migrate_roles(conn)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state_due ON jobs(state, due_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_role_state_priority ON jobs(role, state, priority, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_chat_state ON jobs(chat_id, state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_autonomous_due ON jobs(state, is_autonomous, due_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner, state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_sessions_chat_role ON agent_sessions(chat_id, role)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_leases_role ON workspace_leases(role, slot)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ceo_orders_chat_status ON ceo_orders(chat_id, status, priority, updated_at)")
            conn.commit()

    def _migrate_roles(self, conn: sqlite3.Connection) -> None:
        """
        Idempotent migration for role renames.

        Current supported renames:
        - ceo -> jarvis
        - orchestrator -> jarvis
        """
        try:
            conn.execute("UPDATE jobs SET role = 'jarvis' WHERE role IN ('ceo', 'orchestrator')")
        except Exception:
            pass

        # agent_sessions: avoid PK(chat_id, role) conflicts.
        try:
            conn.execute(
                "DELETE FROM agent_sessions "
                "WHERE role IN ('ceo', 'orchestrator') AND EXISTS(SELECT 1 FROM agent_sessions a2 WHERE a2.chat_id = agent_sessions.chat_id AND a2.role = 'jarvis')"
            )
            conn.execute("UPDATE agent_sessions SET role = 'jarvis' WHERE role IN ('ceo', 'orchestrator')")
        except Exception:
            pass

        # role_controls: avoid PK(role) conflicts.
        try:
            conn.execute(
                "DELETE FROM role_controls "
                "WHERE role IN ('ceo', 'orchestrator') AND EXISTS(SELECT 1 FROM role_controls r2 WHERE r2.role = 'jarvis')"
            )
            conn.execute("UPDATE role_controls SET role = 'jarvis' WHERE role IN ('ceo', 'orchestrator')")
        except Exception:
            pass

        # workspace_leases: avoid PK(role, slot) conflicts.
        try:
            conn.execute(
                "DELETE FROM workspace_leases "
                "WHERE role IN ('ceo', 'orchestrator') AND EXISTS(SELECT 1 FROM workspace_leases w2 WHERE w2.role = 'jarvis' AND w2.slot = workspace_leases.slot)"
            )
            conn.execute("UPDATE workspace_leases SET role = 'jarvis' WHERE role IN ('ceo', 'orchestrator')")
        except Exception:
            pass

        # cost caps: avoid PK(role) conflicts.
        try:
            conn.execute(
                "DELETE FROM cost_caps "
                "WHERE role IN ('ceo', 'orchestrator') AND EXISTS(SELECT 1 FROM cost_caps c2 WHERE c2.role = 'jarvis')"
            )
            conn.execute("UPDATE cost_caps SET role = 'jarvis' WHERE role IN ('ceo', 'orchestrator')")
        except Exception:
            pass

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
            with self._conn() as conn:
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
                artifacts_dir = task.artifacts_dir
                if artifacts_dir is None:
                    # Default artifacts directory lives next to the SQLite DB (typically ./data/artifacts/<job_id>/).
                    artifacts_dir = str((self._path.parent / "artifacts" / job_id).resolve())
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
                    artifacts_dir,
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
                self._append_event(conn, job_id, "submitted", {"state": task.state, "role": task.role})
                conn.commit()
            return job_id

    def submit_batch(self, tasks: list[Task]) -> list[str]:
        return [self.submit_task(task) for task in tasks]

    def _resolve_job_id_in_conn(self, conn: sqlite3.Connection, job_id: str) -> str | None:
        """
        Resolve a job id that may be a full UUID or a short prefix.

        Grounded goal: allow humans to use the 8-char prefix shown in chat for /job, /ticket, /approve, /cancel.
        """
        jid = (job_id or "").strip()
        if not jid:
            return None

        row = conn.execute("SELECT job_id FROM jobs WHERE job_id = ?", (jid,)).fetchone()
        if row is not None:
            return str(row["job_id"])

        # Prefix resolution: only attempt for reasonably-long prefixes to avoid accidental matches.
        if 4 <= len(jid) < 36:
            rows = conn.execute("SELECT job_id FROM jobs WHERE job_id LIKE ? LIMIT 2", (jid + "%",)).fetchall()
            if len(rows) == 1:
                return str(rows[0]["job_id"])

        return None

    def get_job(self, job_id: str) -> Task | None:
        with self._conn() as conn:
            resolved = self._resolve_job_id_in_conn(conn, str(job_id))
            if not resolved:
                return None
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (resolved,)).fetchone()
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
            with self._conn() as conn:
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

                # Avoid starvation: scan more than `max_rows` so we can skip paused roles,
                # saturated roles, or dependency-blocked tasks.
                scan_limit = max(50, int(max_rows) * 25)
                base_sql = (
                    f"SELECT * FROM jobs WHERE {' AND '.join(where)} "
                    f"ORDER BY priority ASC, created_at ASC LIMIT {int(scan_limit)}"
                )
                rows = conn.execute(base_sql, params).fetchall()
                if not rows:
                    return [] if max_rows > 1 else None

                out: list[Task] = []
                touched_due = False
                for row in rows:
                    role_name = str(row["role"])
                    if self._is_role_paused(conn, role_name):
                        continue

                    deps = _coerce_json_list(row["depends_on"])
                    allow_terminal = False
                    try:
                        trace = Task.from_trace_json(row["trace"])
                        allow_terminal = bool(trace.get("wrapup_for"))
                    except Exception:
                        allow_terminal = False
                    if deps and not self._deps_satisfied(conn, deps, allow_terminal=allow_terminal):
                        # Avoid "spin": push due_at forward a bit so other ready tasks can be scanned/claimed.
                        try:
                            conn.execute(
                                "UPDATE jobs SET due_at = ?, updated_at = ? WHERE job_id = ? AND state = ?",
                                (now + 10.0, time.time(), str(row["job_id"]), "queued"),
                            )
                            touched_due = True
                        except Exception:
                            pass
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
                        next_row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
                        task = self._row_to_task(next_row)
                        if task is not None:
                            out.append(task)
                        conn.commit()
                    if len(out) >= max_rows:
                        break

                if not out:
                    if touched_due:
                        try:
                            conn.commit()
                        except Exception:
                            pass
                    return [] if max_rows > 1 else None
                if max_rows == 1:
                    return out[0]
                return out

    def update_state(self, job_id: str, state: str, **metadata: Any) -> bool:
        with self._lock:
            with self._conn() as conn:
                resolved = self._resolve_job_id_in_conn(conn, str(job_id))
                if not resolved:
                    return False
                ok = self._update_state_in_conn(conn, job_id=resolved, state=str(state), metadata=metadata)
                if ok:
                    conn.commit()
                return ok

    def update_trace(self, job_id: str, **metadata: Any) -> bool:
        """
        Update trace metadata without changing state and without appending an audit event.

        Intended for high-frequency "live" updates (stdout tail, phase, etc) where emitting
        job_events rows would be noisy and expensive.
        """
        jid = str(job_id).strip()
        if not jid:
            return False
        if not metadata:
            return False

        # Keep trace bounded; callers may pass tails/logs.
        cleaned: dict[str, Any] = {}
        for k, v in metadata.items():
            key = str(k).strip()
            if not key:
                continue
            if v is None:
                cleaned[key] = None
                continue
            if isinstance(v, str):
                s = v.strip("\n")
                if len(s) > 4000:
                    s = s[:4000] + "..."
                cleaned[key] = s
                continue
            cleaned[key] = v

        if not cleaned:
            return False

        with self._lock:
            with self._conn() as conn:
                resolved = self._resolve_job_id_in_conn(conn, jid)
                if not resolved:
                    return False
                row = conn.execute("SELECT trace FROM jobs WHERE job_id = ?", (resolved,)).fetchone()
                if row is None:
                    return False
                trace = Task.from_trace_json(row["trace"])
                if not isinstance(trace, dict):
                    trace = {}
                for k, v in cleaned.items():
                    if v is None:
                        trace.pop(k, None)
                    else:
                        trace[k] = v
                cur = conn.execute(
                    "UPDATE jobs SET updated_at = ?, trace = ? WHERE job_id = ?",
                    (time.time(), json.dumps(trace, ensure_ascii=False), resolved),
                )
                if cur.rowcount:
                    conn.commit()
                    return True
                return False

    def _update_state_in_conn(self, conn: sqlite3.Connection, *, job_id: str, state: str, metadata: dict[str, Any]) -> bool:
        now = time.time()
        row = conn.execute("SELECT state, trace FROM jobs WHERE job_id = ?", (str(job_id),)).fetchone()
        if row is None:
            return False

        trace = Task.from_trace_json(row["trace"])
        trace.update(metadata)
        cur = conn.execute(
            "UPDATE jobs SET state = ?, updated_at = ?, trace = ? WHERE job_id = ?",
            (str(state), now, json.dumps(trace, ensure_ascii=False), str(job_id)),
        )
        if cur.rowcount:
            self._append_event(conn, str(job_id), f"state:{state}", metadata)
            return True
        return False

    def cancel(self, job_id: str) -> bool:
        return self.update_state(str(job_id), "cancelled", reason="user_requested")

    def requeue(self, job_id: str, *, reason: str | None = None) -> bool:
        metadata: dict[str, Any] = {}
        if reason:
            metadata["requeue_reason"] = reason
        return self.update_state(str(job_id), "queued", **metadata)

    def bump_retry(self, job_id: str, *, due_at: float, error: str | None = None) -> bool:
        """
        Schedule a retry by moving the job back to `queued` with a future `due_at` and incrementing retry_count.

        This is used by worker code to implement retry/backoff without losing audit history.
        """
        jid = str(job_id).strip()
        if not jid:
            return False
        try:
            due = float(due_at)
        except Exception:
            due = time.time() + 30.0

        err = (error or "").strip()
        if len(err) > 4000:
            err = err[:4000] + "..."

        with self._lock:
            with self._conn() as conn:
                resolved = self._resolve_job_id_in_conn(conn, jid)
                if not resolved:
                    return False
                row = conn.execute(
                    "SELECT retry_count, max_retries, trace FROM jobs WHERE job_id = ?",
                    (resolved,),
                ).fetchone()
                if row is None:
                    return False
                try:
                    retries = int(row["retry_count"] or 0)
                except Exception:
                    retries = 0
                try:
                    max_retries = int(row["max_retries"] or 0)
                except Exception:
                    max_retries = 0
                if max_retries <= 0:
                    return False
                retries = max(0, retries) + 1
                if retries > max_retries:
                    return False

                trace = Task.from_trace_json(row["trace"])
                trace["retry_scheduled_at"] = time.time()
                trace["retry_due_at"] = due
                trace["retry_count"] = retries
                if err:
                    trace["last_error"] = err

                cur = conn.execute(
                    "UPDATE jobs SET state = ?, due_at = ?, retry_count = ?, updated_at = ?, trace = ? WHERE job_id = ?",
                    ("queued", float(due), int(retries), time.time(), json.dumps(trace, ensure_ascii=False), resolved),
                )
                if cur.rowcount:
                    self._append_event(conn, resolved, "retry_scheduled", {"retry_count": retries, "due_at": due})
                    conn.commit()
                    return True
                return False

    def recover_stale_running(self) -> int:
        """
        On startup, return interrupted "running" jobs back to "queued" so workers can retry them.

        Returns:
            Number of jobs moved from `running` to `queued`.
        """
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute("SELECT job_id FROM jobs WHERE state='running'").fetchall()
                # Workspace leases are process-local; any surviving rows are stale after a restart/crash.
                try:
                    conn.execute("DELETE FROM workspace_leases")
                except Exception:
                    pass
                if not rows:
                    conn.commit()
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
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO role_controls(role, is_paused) VALUES (?, 1) ON CONFLICT(role) DO UPDATE SET is_paused = 1",
                (str(role),),
            )
            conn.commit()

    def resume_role(self, role: str) -> None:
        with self._conn() as conn:
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
            with self._conn() as conn:
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
        with self._conn() as conn:
            return self._is_role_paused(conn, str(role))

    def _is_role_paused(self, conn: sqlite3.Connection, role: str) -> bool:
        row = conn.execute("SELECT is_paused FROM role_controls WHERE role = ?", (str(role),)).fetchone()
        if row is None:
            return False
        return bool(int(row["is_paused"]))

    def _set_global_pause(self, *, paused: bool) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO role_controls(role, is_paused) VALUES (?, ?) ON CONFLICT(role) DO UPDATE SET is_paused = ?",
                ("__orchestrator__", 1 if paused else 0, 1 if paused else 0),
            )
            conn.commit()

    def _is_global_pause(self, conn: sqlite3.Connection | None = None) -> bool:
        if conn is None:
            with self._conn() as c:
                return self._is_global_pause(c)
        row = conn.execute("SELECT is_paused FROM role_controls WHERE role = ?", ("__orchestrator__",)).fetchone()
        if row is None:
            return False
        return bool(int(row["is_paused"]))

    def get_role_health(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        with self._conn() as conn:
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
        with self._conn() as conn:
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
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cost_caps(role, max_cost_window_usd, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(role) DO UPDATE SET max_cost_window_usd = ?, updated_at = ?",
                (str(role), float(cost_usd), time.time(), float(cost_usd), time.time()),
            )
            conn.commit()

    def get_cost_cap(self, role: str, *, default: float | None = None) -> float | None:
        with self._conn() as conn:
            row = conn.execute("SELECT max_cost_window_usd FROM cost_caps WHERE role = ?", (str(role),)).fetchone()
            if row is None:
                return default
            return _coerce_float(row["max_cost_window_usd"], default)

    def get_default_cost_cap(self) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(max_cost_window_usd) FROM cost_caps").fetchone()
            if row is None or row[0] is None:
                return 0.0
            try:
                return float(row[0])
            except Exception:
                return 0.0

    def _resolve_order_id_in_conn(self, conn: sqlite3.Connection, order_id: str, *, chat_id: int) -> str | None:
        oid = (order_id or "").strip()
        if not oid:
            return None

        row = conn.execute(
            "SELECT order_id FROM ceo_orders WHERE order_id = ? AND chat_id = ?",
            (oid, int(chat_id)),
        ).fetchone()
        if row is not None:
            return str(row["order_id"])

        if 4 <= len(oid) < 36:
            rows = conn.execute(
                "SELECT order_id FROM ceo_orders WHERE chat_id = ? AND order_id LIKE ? LIMIT 2",
                (int(chat_id), oid + "%"),
            ).fetchall()
            if len(rows) == 1:
                return str(rows[0]["order_id"])

        return None

    def upsert_order(
        self,
        *,
        order_id: str,
        chat_id: int,
        title: str,
        body: str,
        status: str = "active",
        priority: int = 2,
    ) -> None:
        """
        Persist a CEO "order" (an ongoing objective) for autopilot.

        Ground truth: this is separate from jobs; orders can stay active after the initial ticket is done.
        """
        st = (status or "active").strip().lower()
        if st not in ("active", "paused", "done"):
            st = "active"
        try:
            pr = int(priority)
        except Exception:
            pr = 2
        pr = max(1, min(3, pr))

        oid = str(order_id).strip()
        if not oid:
            raise ValueError("order_id required")

        with self._lock:
            with self._conn() as conn:
                now = time.time()
                existing = conn.execute(
                    "SELECT created_at FROM ceo_orders WHERE order_id = ? AND chat_id = ?",
                    (oid, int(chat_id)),
                ).fetchone()
                created_at = float(existing["created_at"]) if existing is not None else float(now)
                conn.execute(
                    "INSERT OR REPLACE INTO ceo_orders(order_id, chat_id, title, body, status, priority, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        oid,
                        int(chat_id),
                        str(title or "").strip(),
                        str(body or "").strip(),
                        st,
                        int(pr),
                        float(created_at),
                        float(now),
                    ),
                )
                conn.commit()

    def list_orders(self, *, chat_id: int, status: str | None, limit: int = 50) -> list[dict[str, Any]]:
        st = (status or "").strip().lower() or None
        if st is not None and st not in ("active", "paused", "done"):
            st = None
        lim = max(1, min(200, int(limit)))
        with self._conn() as conn:
            if st is None:
                rows = conn.execute(
                    "SELECT * FROM ceo_orders WHERE chat_id = ? ORDER BY status ASC, priority ASC, updated_at DESC LIMIT ?",
                    (int(chat_id), int(lim)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ceo_orders WHERE chat_id = ? AND status = ? ORDER BY priority ASC, updated_at DESC LIMIT ?",
                    (int(chat_id), st, int(lim)),
                ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                out.append({str(k): r[k] for k in r.keys()})
            return out

    def get_order(self, order_id: str, *, chat_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            resolved = self._resolve_order_id_in_conn(conn, str(order_id), chat_id=int(chat_id))
            if not resolved:
                return None
            row = conn.execute(
                "SELECT * FROM ceo_orders WHERE order_id = ? AND chat_id = ?",
                (resolved, int(chat_id)),
            ).fetchone()
            if row is None:
                return None
            return {str(k): row[k] for k in row.keys()}

    def set_order_status(self, order_id: str, *, chat_id: int, status: str) -> bool:
        st = (status or "").strip().lower()
        if st not in ("active", "paused", "done"):
            return False
        with self._lock:
            with self._conn() as conn:
                resolved = self._resolve_order_id_in_conn(conn, str(order_id), chat_id=int(chat_id))
                if not resolved:
                    return False
                conn.execute(
                    "UPDATE ceo_orders SET status = ?, updated_at = ? WHERE order_id = ? AND chat_id = ?",
                    (st, time.time(), resolved, int(chat_id)),
                )
                conn.commit()
                return True

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
        with self._conn() as conn:
            q = conn.execute(
                f"SELECT * FROM jobs{clause} ORDER BY created_at DESC LIMIT ?",
                [*params, int(limit)],
            ).fetchall()
            return [self._row_to_task(r) for r in q]

    def queued_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(1) as c FROM jobs WHERE state = 'queued'").fetchone()
            return int(row["c"]) if row else 0

    def running_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(1) as c FROM jobs WHERE state = 'running'").fetchone()
            return int(row["c"]) if row else 0

    def jobs_by_state(self, *, state: str, limit: int = 50) -> list[Task]:
        with self._conn() as conn:
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
            with self._conn() as conn:
                resolved = self._resolve_job_id_in_conn(conn, str(job_id))
                if not resolved:
                    return False
                conn.execute(
                    "INSERT OR REPLACE INTO approver_log(job_id, approved_by, approved_at, reason) VALUES (?, ?, ?, ?)",
                    (resolved, None if approved_by is None else int(approved_by), time.time(), reason),
                )
                updated = self._update_state_in_conn(conn, job_id=resolved, state="queued", metadata=trace_payload)
                if updated:
                    self._append_event(conn, resolved, "approved", {"approved": approved, "approved_by": approved_by, "reason": reason})
                    conn.commit()
                return updated

    def clear_job_approval(self, job_id: str) -> bool:
        with self._lock:
            with self._conn() as conn:
                resolved = self._resolve_job_id_in_conn(conn, str(job_id))
                if not resolved:
                    return False
                row = conn.execute("SELECT trace FROM jobs WHERE job_id=?", (resolved,)).fetchone()
                if row is None:
                    return False
                trace = Task.from_trace_json(row["trace"])
                if not isinstance(trace, dict):
                    trace = {}
                trace.pop("approved", None)
                trace.pop("approved_at", None)
                trace.pop("approved_by", None)
                conn.execute("UPDATE jobs SET trace = ? WHERE job_id = ?", (json.dumps(trace, ensure_ascii=False), resolved))
                conn.execute("DELETE FROM approver_log WHERE job_id = ?", (resolved,))
                self._append_event(conn, resolved, "approval_cleared", {})
                conn.commit()
                return True

    def get_agent_thread(self, *, chat_id: int, role: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT thread_id FROM agent_sessions WHERE chat_id = ? AND role = ?",
                (int(chat_id), str(role)),
            ).fetchone()
            if row is None:
                return None
            tid = row["thread_id"]
            return str(tid).strip() if tid else None

    def set_agent_thread(self, *, chat_id: int, role: str, thread_id: str) -> None:
        tid = (thread_id or "").strip()
        if not tid:
            return
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO agent_sessions(chat_id, role, thread_id, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(chat_id, role) DO UPDATE SET thread_id = ?, updated_at = ?",
                (int(chat_id), str(role), tid, time.time(), tid, time.time()),
            )
            conn.commit()

    def clear_agent_thread(self, *, chat_id: int, role: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM agent_sessions WHERE chat_id = ? AND role = ?",
                (int(chat_id), str(role)),
            )
            conn.commit()
            return bool(cur.rowcount)

    def clear_agent_threads(self, *, chat_id: int) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM agent_sessions WHERE chat_id = ?", (int(chat_id),))
            conn.commit()
            return int(cur.rowcount or 0)

    def lease_workspace(self, *, role: str, job_id: str, slots: int) -> int | None:
        """
        Atomically lease a worktree slot for a given role. Returns slot number or None if none free.
        """
        role = (role or "").strip().lower()
        job_id = (job_id or "").strip()
        if not role or not job_id:
            return None
        slots = max(1, int(slots))
        with self._lock:
            with self._conn() as conn:
                # Already leased by this job?
                row = conn.execute("SELECT slot FROM workspace_leases WHERE job_id = ?", (job_id,)).fetchone()
                if row is not None:
                    try:
                        return int(row["slot"])
                    except Exception:
                        return None

                for slot in range(1, slots + 1):
                    existing = conn.execute(
                        "SELECT job_id FROM workspace_leases WHERE role = ? AND slot = ?",
                        (role, int(slot)),
                    ).fetchone()
                    if existing is not None:
                        continue
                    try:
                        conn.execute(
                            "INSERT INTO workspace_leases(role, slot, job_id, leased_at) VALUES (?, ?, ?, ?)",
                            (role, int(slot), job_id, time.time()),
                        )
                        conn.commit()
                        return int(slot)
                    except sqlite3.IntegrityError:
                        continue
        return None

    def release_workspace(self, *, job_id: str) -> bool:
        job_id = (job_id or "").strip()
        if not job_id:
            return False
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute("DELETE FROM workspace_leases WHERE job_id = ?", (job_id,))
                conn.commit()
                return bool(cur.rowcount)

    def get_workspace_lease(self, *, job_id: str) -> tuple[str, int] | None:
        job_id = (job_id or "").strip()
        if not job_id:
            return None
        with self._conn() as conn:
            row = conn.execute("SELECT role, slot FROM workspace_leases WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            return (str(row["role"]), int(row["slot"]))

    def get_runbook_last_run(self, *, runbook_id: str) -> float:
        rid = (runbook_id or "").strip()
        if not rid:
            return 0.0
        with self._conn() as conn:
            row = conn.execute("SELECT last_run_at FROM runbook_state WHERE runbook_id = ?", (rid,)).fetchone()
            if row is None:
                return 0.0
            try:
                return float(row["last_run_at"])
            except Exception:
                return 0.0

    def set_runbook_last_run(self, *, runbook_id: str, ts: float) -> None:
        rid = (runbook_id or "").strip()
        if not rid:
            return
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runbook_state(runbook_id, last_run_at) VALUES (?, ?) "
                "ON CONFLICT(runbook_id) DO UPDATE SET last_run_at = ?",
                (rid, float(ts), float(ts)),
            )
            conn.commit()

    def jobs_by_parent(self, *, parent_job_id: str, limit: int = 200) -> list[Task]:
        pid = (parent_job_id or "").strip()
        if not pid:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE parent_job_id = ? ORDER BY created_at ASC LIMIT ?",
                (pid, int(limit)),
            ).fetchall()
            return [self._row_to_task(r) for r in rows]

    def inbox(self, *, role: str | None = None, limit: int = 25) -> list[Task]:
        where = ["state IN ('queued', 'blocked', 'failed')"]
        params: list[Any] = []
        if role:
            where.append("role = ?")
            params.append(str(role))
        clause = " WHERE " + " AND ".join(where)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs{clause} ORDER BY updated_at DESC LIMIT ?",
                [*params, int(limit)],
            ).fetchall()
            return [self._row_to_task(r) for r in rows]

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

    def _deps_satisfied(self, conn: sqlite3.Connection, deps: list[str], *, allow_terminal: bool = False) -> bool:
        deps = [str(d).strip() for d in (deps or []) if str(d).strip()]
        if not deps:
            return True
        placeholders = ",".join("?" for _ in deps)
        states = ("done", "failed", "cancelled") if allow_terminal else ("done",)
        state_placeholders = ",".join("?" for _ in states)
        row = conn.execute(
            f"SELECT COUNT(1) as c FROM jobs WHERE job_id IN ({placeholders}) AND state IN ({state_placeholders})",
            [*deps, *states],
        ).fetchone()
        try:
            return int(row["c"]) == len(deps) if row is not None else False
        except Exception:
            return False

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
