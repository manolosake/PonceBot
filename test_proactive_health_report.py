from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tools.proactive_health_report import order_autonomy_funnel
from tools.proactive_health_report import classify_open_job_mode, is_blocked_without_open_jobs
from tools.proactive_health_report import AUTONOMY_MINIMUM_SLO, IMPLEMENTER_FAIL_RATE_TREND_MIN_ATTEMPTS


class TestProactiveHealthReport(unittest.TestCase):
    def _run_report(self, root: Path, db: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        env = {**dict(os.environ)}
        env["CODEXBOT_ORCH_DB"] = str(db)
        env["CODEXBOT_STATE_FILE"] = str(root / "state.json")
        env["CODEXBOT_PROACTIVE_HEALTH_OUT_DIR"] = str(out_dir)
        return subprocess.run(
            ["python3", "tools/proactive_health_report.py"],
            cwd=str(Path(__file__).resolve().parent),
            env=env,
            capture_output=True,
            text=True,
        )

    def _create_report_db(self, db: Path, *, implementer_states: list[str]) -> None:
        now = time.time()
        with sqlite3.connect(db) as con:
            con.execute(
                """
                CREATE TABLE ceo_orders (
                    order_id TEXT,
                    status TEXT,
                    phase TEXT,
                    title TEXT,
                    body TEXT,
                    updated_at REAL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE jobs (
                    job_id TEXT,
                    parent_job_id TEXT,
                    state TEXT,
                    role TEXT,
                    labels TEXT,
                    trace TEXT,
                    updated_at REAL,
                    created_at REAL
                )
                """
            )
            con.execute("CREATE TABLE audit_log (ts REAL, event_type TEXT)")
            con.execute(
                """
                INSERT INTO ceo_orders(order_id, status, phase, title, body, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("order-fail-rate", "active", "planning", "Proactive Sprint: reliability", "", now),
            )
            con.executemany(
                """
                INSERT INTO jobs(job_id, parent_job_id, state, role, labels, trace, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f"impl-{idx}",
                        "order-fail-rate",
                        state,
                        "implementer_local",
                        "{}",
                        json.dumps({"slice_id": f"slice_{idx}", "result_summary": "attempt finished"}),
                        now - idx,
                        now - idx - 1,
                    )
                    for idx, state in enumerate(implementer_states)
                ],
            )

    def _create_factory_report_db(
        self,
        db: Path,
        *,
        repos: list[tuple[str, str, int]],
        heartbeats: list[tuple[str, float]],
        include_active_order: bool = False,
    ) -> None:
        now = time.time()
        with sqlite3.connect(db) as con:
            con.execute(
                """
                CREATE TABLE ceo_orders (
                    order_id TEXT,
                    status TEXT,
                    phase TEXT,
                    title TEXT,
                    body TEXT,
                    updated_at REAL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE jobs (
                    job_id TEXT,
                    parent_job_id TEXT,
                    state TEXT,
                    role TEXT,
                    labels TEXT,
                    trace TEXT,
                    updated_at REAL,
                    created_at REAL
                )
                """
            )
            con.execute("CREATE TABLE audit_log (ts REAL, event_type TEXT)")
            con.execute(
                """
                CREATE TABLE repo_registry (
                    repo_id TEXT,
                    path TEXT,
                    priority INTEGER,
                    updated_at REAL,
                    status TEXT,
                    autonomy_enabled INTEGER
                )
                """
            )
            con.execute(
                """
                CREATE TABLE agent_runtime_state (
                    agent_key TEXT,
                    repo_id TEXT,
                    role TEXT,
                    heartbeat_at REAL,
                    updated_at REAL
                )
                """
            )
            if include_active_order:
                con.execute(
                    """
                    INSERT INTO ceo_orders(order_id, status, phase, title, body, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("order-factory-active", "active", "planning", "Proactive Sprint: factory", "", now),
                )
            con.executemany(
                """
                INSERT INTO repo_registry(repo_id, path, priority, updated_at, status, autonomy_enabled)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(repo_id, f"/tmp/{repo_id}", idx, now - idx, status, autonomy_enabled) for idx, (repo_id, status, autonomy_enabled) in enumerate(repos)],
            )
            con.executemany(
                """
                INSERT INTO agent_runtime_state(agent_key, repo_id, role, heartbeat_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(f"{repo_id}-{idx}", repo_id, "skynet", heartbeat_at, now - idx) for idx, (repo_id, heartbeat_at) in enumerate(heartbeats)],
            )

    def test_missing_db_writes_error_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            env = {**dict(os.environ)}
            env["CODEXBOT_ORCH_DB"] = str(root / "missing.sqlite")
            env["CODEXBOT_STATE_FILE"] = str(root / "state.json")
            env["CODEXBOT_PROACTIVE_HEALTH_OUT_DIR"] = str(out_dir)
            proc = subprocess.run(
                ["python3", "tools/proactive_health_report.py"],
                cwd=str(Path(__file__).resolve().parent),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            latest = out_dir / "latest.json"
            self.assertTrue(latest.exists())
            payload = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("operational_status"), "CRITICAL")
            self.assertEqual(payload.get("error"), "db_missing")

    def test_high_implementer_fail_rate_sets_trend_warn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
            self._create_report_db(
                db,
                implementer_states=["failed", "cancelled", "done", "done", "done"],
            )

            proc = self._run_report(root, db, out_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload.get("trend_status"), "WARN")
            self.assertEqual(payload.get("status"), "WARN")
            flags = payload.get("trend_flags") or []
            flag = next((item for item in flags if item.get("type") == "high_implementer_fail_rate"), None)
            self.assertIsNotNone(flag)
            self.assertEqual(flag.get("attempts"), 5)
            self.assertEqual(flag.get("failures"), 2)
            self.assertEqual(flag.get("fail_rate"), 0.4)
            self.assertEqual(flag.get("threshold"), AUTONOMY_MINIMUM_SLO["implementer_fail_rate_lte"])
            self.assertEqual(flag.get("minimum_attempts"), IMPLEMENTER_FAIL_RATE_TREND_MIN_ATTEMPTS)
            self.assertEqual(
                payload["autonomy_funnel"]["implementer_fail_rate_trend_min_attempts"],
                IMPLEMENTER_FAIL_RATE_TREND_MIN_ATTEMPTS,
            )

    def test_implementer_fail_rate_trend_ignores_low_sample_size(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
            self._create_report_db(db, implementer_states=["failed", "done"])

            proc = self._run_report(root, db, out_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload.get("trend_status"), "OK")
            self.assertEqual(payload.get("status"), "OK")
            flags = payload.get("trend_flags") or []
            self.assertFalse(any(item.get("type") == "high_implementer_fail_rate" for item in flags))
            self.assertEqual(payload["metrics"]["implementer_attempts"], 2)
            self.assertEqual(payload["metrics"]["implementer_failures"], 1)
            self.assertEqual(payload["metrics"]["implementer_fail_rate"], 0.5)

    def test_factory_stale_heartbeats_excludes_disabled_and_blocked_repos(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
            stale = time.time() - 3600
            self._create_factory_report_db(
                db,
                repos=[
                    ("disabled-repo", "active", 0),
                    ("blocked-repo", "blocked", 1),
                    ("inactive-repo", "disabled", 1),
                ],
                heartbeats=[
                    ("disabled-repo", stale),
                    ("blocked-repo", stale),
                    ("inactive-repo", stale),
                ],
            )

            proc = self._run_report(root, db, out_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["factory_enabled_repos"], 0)
            self.assertEqual(payload["metrics"]["factory_registered_repos"], 3)
            self.assertEqual(payload["metrics"]["factory_stale_heartbeats"], 0)
            self.assertEqual(payload["factory"]["stale_heartbeats"], 0)
            anomalies = payload.get("anomalies") or []
            self.assertFalse(any(item.get("type") == "factory_stale_heartbeats" for item in anomalies))
            self.assertEqual(payload.get("operational_status"), "OK")

    def test_factory_stale_heartbeats_includes_active_autonomy_enabled_repos(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
            stale = time.time() - 3600
            self._create_factory_report_db(
                db,
                repos=[
                    ("enabled-repo", "active", 1),
                    ("disabled-repo", "active", 0),
                    ("blocked-repo", "blocked", 1),
                ],
                heartbeats=[
                    ("enabled-repo", stale),
                    ("disabled-repo", stale),
                    ("blocked-repo", stale),
                ],
                include_active_order=True,
            )

            proc = self._run_report(root, db, out_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["factory_enabled_repos"], 1)
            self.assertEqual(payload["metrics"]["factory_registered_repos"], 3)
            self.assertEqual(payload["metrics"]["factory_stale_heartbeats"], 1)
            self.assertEqual(payload["factory"]["stale_heartbeats"], 1)
            anomalies = payload.get("anomalies") or []
            stale_anomaly = next((item for item in anomalies if item.get("type") == "factory_stale_heartbeats"), None)
            self.assertIsNotNone(stale_anomaly)
            self.assertEqual(stale_anomaly.get("count"), 1)
            self.assertEqual(payload.get("operational_status"), "WARN")

    def test_paused_idle_backlog_is_separate_from_active_idle_report_noise(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
            now = time.time()
            stale = now - 1200
            with sqlite3.connect(db) as con:
                con.execute(
                    """
                    CREATE TABLE ceo_orders (
                        order_id TEXT,
                        status TEXT,
                        phase TEXT,
                        title TEXT,
                        body TEXT,
                        updated_at REAL
                    )
                    """
                )
                con.execute(
                    """
                    CREATE TABLE jobs (
                        job_id TEXT,
                        parent_job_id TEXT,
                        state TEXT,
                        role TEXT,
                        labels TEXT,
                        trace TEXT,
                        updated_at REAL,
                        created_at REAL
                    )
                    """
                )
                con.execute("CREATE TABLE audit_log (ts REAL, event_type TEXT)")
                con.executemany(
                    """
                    INSERT INTO ceo_orders(order_id, status, phase, title, body, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ("active-idle-order", "active", "planning", "Proactive Sprint: active", "", stale),
                        ("paused-idle-order", "paused", "planning", "Proactive Sprint: paused", "", stale),
                    ],
                )

            env = {**dict(os.environ)}
            env["CODEXBOT_ORCH_DB"] = str(db)
            env["CODEXBOT_STATE_FILE"] = str(root / "state.json")
            env["CODEXBOT_PROACTIVE_HEALTH_OUT_DIR"] = str(out_dir)
            proc = subprocess.run(
                ["python3", "tools/proactive_health_report.py"],
                cwd=str(Path(__file__).resolve().parent),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            anomalies = payload.get("anomalies") or []
            by_order = {item.get("order_id"): item for item in anomalies}
            self.assertEqual(by_order["active-idle-order"].get("type"), "idle_without_improvement")
            paused = by_order["paused-idle-order"]
            self.assertEqual(paused.get("type"), "paused_idle_backlog")
            self.assertEqual(paused.get("phase"), "planning")
            self.assertEqual(paused.get("open_jobs"), 0)
            self.assertGreaterEqual(int(paused.get("last_activity_age_s") or 0), 900)
            self.assertEqual(paused.get("recommended_action"), "resume_or_close_stale_order")

            markdown = (out_dir / "latest.md").read_text(encoding="utf-8")
            self.assertIn("paused_idle_backlog", markdown)
            self.assertIn("resume_or_close_stale_order", markdown)

    def test_order_autonomy_funnel_no_code_change_slice_can_close(self) -> None:
        order_id = "order-nochange-1234"
        now = 2000.0
        since = 0.0
        children = [
            {
                "role": "implementer_local",
                "state": "done",
                "created_at": 1100.0,
                "updated_at": 1200.0,
                "trace": json.dumps(
                    {
                        "slice_id": "slice_a",
                        "slice_no_code_change": True,
                        "result_summary": "No code changes required; existing behavior already satisfied.",
                    }
                ),
                "labels": "{}",
            },
            {
                "role": "reviewer_local",
                "state": "done",
                "created_at": 1210.0,
                "updated_at": 1300.0,
                "trace": json.dumps(
                    {
                        "slice_id": "slice_a",
                        "result_summary": "READY: verified expected behavior and evidence.",
                    }
                ),
                "labels": "{}",
            },
            {
                "role": "skynet",
                "state": "done",
                "created_at": 1310.0,
                "updated_at": 1400.0,
                "trace": json.dumps(
                    {
                        "slice_id": "slice_a",
                        "improvement_verified": True,
                        "result_summary": "improvement_verified",
                    }
                ),
                "labels": "{}",
            },
        ]

        funnel = order_autonomy_funnel(children, order_id=order_id, now=now, since=since)
        self.assertEqual(funnel["slices_validated"], 1)
        self.assertEqual(funnel["slices_closed"], 1)
        self.assertEqual(funnel["quality_gate_status"], "closed")
        self.assertTrue(funnel["improvement_verified"])
    def test_classify_open_job_mode_respects_open_jobs_semantics(self) -> None:
        self.assertEqual(classify_open_job_mode({}, 0), "idle")
        self.assertEqual(classify_open_job_mode({"skynet": 1}, 1), "controller")
        self.assertEqual(classify_open_job_mode({"architect_local": 1}, 1), "local")
        self.assertEqual(classify_open_job_mode({"backend": 1}, 1), "cli")
        self.assertEqual(classify_open_job_mode({"backend": 1, "reviewer_local": 1}, 2), "mixed")
        self.assertEqual(classify_open_job_mode({"backend": 0, "architect_local": 0, "skynet": 1}, 1), "controller")
        self.assertEqual(classify_open_job_mode({"backend": "0", "reviewer_local": "nope", "skynet": 1}, 1), "controller")
        self.assertEqual(classify_open_job_mode({"backend": "2", "reviewer_local": None}, 2), "cli")

    def test_is_blocked_without_open_jobs_detects_stall_condition(self) -> None:
        self.assertTrue(is_blocked_without_open_jobs("blocked", 0))
        self.assertTrue(is_blocked_without_open_jobs("blocked_waiting_only", 0))
        self.assertTrue(is_blocked_without_open_jobs("blocked_approval", 0))
        self.assertFalse(is_blocked_without_open_jobs("blocked", 1))
        self.assertFalse(is_blocked_without_open_jobs("review", 0))


if __name__ == "__main__":
    unittest.main()
