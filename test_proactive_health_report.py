from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tools.proactive_health_report import order_autonomy_funnel, summarize_outcome_quality
from tools.proactive_health_report import classify_open_job_mode, is_blocked_without_open_jobs
from tools.proactive_health_report import AUTONOMY_MINIMUM_SLO, IMPLEMENTER_FAIL_RATE_TREND_MIN_ATTEMPTS
from tools.proactive_health_report import STALE_HEARTBEAT_DETAIL_LIMIT


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
        heartbeats: list[tuple[str, float | None]],
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
            actions = payload.get("recommended_actions") or []
            action = next((item for item in actions if item.get("evidence_type") == "high_implementer_fail_rate"), None)
            self.assertIsNotNone(action)
            self.assertEqual(action.get("priority"), "P1")
            self.assertEqual(action.get("count"), 2)

            markdown = (out_dir / "latest.md").read_text(encoding="utf-8")
            self.assertIn("## Recommended Actions", markdown)
            self.assertIn("high_implementer_fail_rate", markdown)

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

    def test_churn_heavy_low_outcome_quality_sets_trend_warn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
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
                    ("order-churn", "active", "planning", "Proactive Sprint: churn", "", now),
                )
                con.executemany(
                    """
                    INSERT INTO jobs(job_id, parent_job_id, state, role, labels, trace, updated_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            f"impl-churn-{idx}",
                            "order-churn",
                            "done",
                            "implementer_local",
                            "{}",
                            json.dumps(
                                {
                                    "slice_id": f"slice_{idx}",
                                    "slice_patch_applied": True,
                                    "slice_validation_ok": True,
                                    "result_summary": "Applied and locally validated a patch but no closed outcome yet.",
                                }
                            ),
                            now - idx,
                            now - idx - 1,
                        )
                        for idx in range(4)
                    ],
                )

            proc = self._run_report(root, db, out_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload.get("trend_status"), "WARN")
            self.assertEqual(payload.get("status"), "WARN")
            self.assertEqual(payload["metrics"]["outcome_quality_status"], "WARN")
            self.assertEqual(payload["metrics"]["validated_outcome_rate"], 1.0)
            self.assertEqual(payload["metrics"]["verified_outcome_rate"], 0.0)
            self.assertEqual(payload["metrics"]["closed_outcome_rate"], 0.0)
            self.assertEqual(payload["metrics"]["verified_outcomes"], 0)
            self.assertEqual(payload["metrics"]["churn_gap"], 4)
            self.assertEqual(payload["autonomy_funnel"]["outcome_quality_status"], "WARN")
            self.assertEqual(payload["autonomy_funnel"]["outcome_quality"]["churn_gap"], 4)
            flags = payload.get("trend_flags") or []
            flag = next((item for item in flags if item.get("type") == "low_outcome_quality"), None)
            self.assertIsNotNone(flag)
            self.assertEqual(flag.get("outcome_quality_status"), "WARN")
            self.assertEqual(flag.get("validated_outcome_rate"), 1.0)
            self.assertEqual(flag.get("verified_outcome_rate"), 0.0)
            self.assertEqual(flag.get("slices_started"), 4)
            self.assertEqual(flag.get("slices_applied"), 4)
            self.assertEqual(flag.get("slices_validated"), 4)
            self.assertEqual(flag.get("slices_closed"), 0)
            self.assertEqual(flag.get("churn_gap"), 4)
            actions = payload.get("recommended_actions") or []
            action = next((item for item in actions if item.get("evidence_type") == "low_outcome_quality"), None)
            self.assertIsNotNone(action)
            self.assertEqual(action.get("priority"), "P1")
            self.assertEqual(action.get("count"), 4)
            self.assertIn("replan", action.get("action") or "")

            markdown = (out_dir / "latest.md").read_text(encoding="utf-8")
            self.assertIn("Outcome quality (24h): status=WARN", markdown)
            self.assertIn("low_outcome_quality", markdown)

    def test_summarize_outcome_quality_treats_validated_without_closed_as_churn(self) -> None:
        summary = summarize_outcome_quality(
            {
                "slices_started": 4,
                "slices_applied": 4,
                "slices_validated": 4,
                "slices_closed": 0,
            }
        )

        self.assertEqual(summary["outcome_quality_status"], "WARN")
        self.assertEqual(summary["validated_outcome_rate"], 1.0)
        self.assertEqual(summary["verified_outcome_rate"], 0.0)
        self.assertEqual(summary["closed_outcome_rate"], 0.0)
        self.assertEqual(summary["verified_outcomes"], 0)
        self.assertGreater(summary["churn_gap"], 0)

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
            self.assertEqual(payload["factory"]["stale_heartbeat_details"], [])
            anomalies = payload.get("anomalies") or []
            self.assertFalse(any(item.get("type") == "factory_stale_heartbeats" for item in anomalies))
            self.assertEqual(payload.get("operational_status"), "OK")

    def test_factory_stale_heartbeats_are_advisory_for_uncovered_repos(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
            stale = time.time() - 3600
            self._create_factory_report_db(
                db,
                repos=[
                    ("enabled-repo", "active", 1),
                    ("enabled-missing", "active", 1),
                    ("enabled-no-row", "active", 1),
                    ("disabled-repo", "active", 0),
                    ("blocked-repo", "blocked", 1),
                ],
                heartbeats=[
                    ("enabled-repo", stale),
                    ("enabled-missing", None),
                    ("disabled-repo", stale),
                    ("blocked-repo", stale),
                ],
                include_active_order=True,
            )

            proc = self._run_report(root, db, out_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["factory_enabled_repos"], 3)
            self.assertEqual(payload["metrics"]["factory_registered_repos"], 5)
            self.assertEqual(payload["metrics"]["factory_stale_heartbeats"], 0)
            self.assertEqual(payload["metrics"]["factory_stale_heartbeats_total"], 3)
            self.assertEqual(payload["factory"]["stale_heartbeats"], 0)
            self.assertEqual(payload["factory"]["stale_heartbeats_total"], 3)
            self.assertEqual(payload["factory"]["stale_heartbeat_detail_limit"], STALE_HEARTBEAT_DETAIL_LIMIT)
            self.assertFalse(payload["factory"]["stale_heartbeat_details_truncated"])
            self.assertEqual(payload["factory"]["stale_heartbeat_details"], [])
            details = payload["factory"]["stale_heartbeat_total_details"]
            self.assertEqual([item["repo_id"] for item in details], ["enabled-missing", "enabled-no-row", "enabled-repo"])
            self.assertEqual(details[0]["agent_key"], "enabled-missing-1")
            self.assertEqual(details[0]["role"], "skynet")
            self.assertIsNone(details[0]["heartbeat_at"])
            self.assertIsNone(details[0]["heartbeat_age_s"])
            self.assertEqual(details[0]["reason"], "missing_heartbeat")
            self.assertEqual(details[1]["agent_key"], "")
            self.assertEqual(details[1]["role"], "")
            self.assertIsNone(details[1]["heartbeat_at"])
            self.assertIsNone(details[1]["heartbeat_age_s"])
            self.assertEqual(details[1]["reason"], "missing_runtime_state_row")
            self.assertEqual(details[2]["agent_key"], "enabled-repo-0")
            self.assertEqual(details[2]["role"], "skynet")
            self.assertGreaterEqual(details[2]["heartbeat_age_s"], 3600)
            self.assertEqual(details[2]["reason"], "stale_heartbeat")
            anomalies = payload.get("anomalies") or []
            self.assertFalse(any(item.get("type") == "factory_stale_heartbeats" for item in anomalies))
            self.assertEqual(payload.get("operational_status"), "OK")
            actions = payload.get("recommended_actions") or []
            action = next((item for item in actions if item.get("evidence_type") == "factory_stale_heartbeats"), None)
            self.assertIsNone(action)

            markdown = (out_dir / "latest.md").read_text(encoding="utf-8")
            self.assertIn("Factory stale active heartbeats: 0 total_unhealthy=3", markdown)
            self.assertIn("## Recommended Actions", markdown)

    def test_factory_stale_heartbeats_warn_for_covered_active_repo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
            stale = time.time() - 3600
            self._create_factory_report_db(
                db,
                repos=[
                    ("enabled-repo", "active", 1),
                    ("uncovered-repo", "active", 1),
                ],
                heartbeats=[
                    ("enabled-repo", stale),
                    ("uncovered-repo", stale),
                ],
            )
            with sqlite3.connect(db) as con:
                con.execute(
                    """
                    INSERT INTO ceo_orders(order_id, status, phase, title, body, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "order-covered",
                        "active",
                        "planning",
                        "Proactive Sprint: enabled-repo",
                        "[repo:enabled-repo]",
                        time.time(),
                    ),
                )

            proc = self._run_report(root, db, out_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["factory_stale_heartbeats"], 1)
            self.assertEqual(payload["metrics"]["factory_stale_heartbeats_total"], 2)
            details = payload["factory"]["stale_heartbeat_details"]
            self.assertEqual([item["repo_id"] for item in details], ["enabled-repo"])
            self.assertEqual(details[0]["coverage_state"], "covered_active")
            stale_anomaly = next((item for item in payload.get("anomalies") or [] if item.get("type") == "factory_stale_heartbeats"), None)
            self.assertIsNotNone(stale_anomaly)
            self.assertEqual(stale_anomaly.get("count"), 1)
            self.assertEqual(payload.get("operational_status"), "WARN")

    def test_factory_next_targets_include_active_order_quality_gate_not_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
            now = time.time()
            self._create_factory_report_db(
                db,
                repos=[
                    ("quality-repo", "active", 1),
                ],
                heartbeats=[
                    ("quality-repo", now - 60),
                ],
            )
            with sqlite3.connect(db) as con:
                con.execute(
                    """
                    INSERT INTO ceo_orders(order_id, status, phase, title, body, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "order-quality-open",
                        "active",
                        "planning",
                        "Proactive Sprint: quality-repo",
                        "[repo:quality-repo]",
                        now,
                    ),
                )
                con.execute(
                    """
                    INSERT INTO jobs(job_id, parent_job_id, state, role, labels, trace, updated_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "impl-quality-1",
                        "order-quality-open",
                        "done",
                        "implementer_local",
                        "{}",
                        json.dumps(
                            {
                                "slice_id": "slice_quality",
                                "slice_patch_applied": True,
                                "slice_validation_ok": True,
                                "result_summary": "Applied and locally validated the slice; awaiting controller quality closure.",
                            }
                        ),
                        now,
                        now - 5,
                    ),
                )

            proc = self._run_report(root, db, out_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload.get("operational_status"), "OK")
            self.assertEqual(payload["metrics"]["factory_stale_heartbeats"], 0)
            self.assertEqual(payload["metrics"]["factory_uncovered_enabled_repos"], 0)
            coverage = payload["factory"]["coverage_details"]
            self.assertEqual(len(coverage), 1)
            self.assertEqual(coverage[0]["coverage_state"], "covered_active")
            self.assertEqual(coverage[0]["order_quality_gate_status"], "validated")
            self.assertEqual(coverage[0]["order_slices_validated"], 1)
            self.assertEqual(coverage[0]["order_slices_closed"], 0)
            self.assertEqual(coverage[0]["order_quality_attention_reason"], "quality_gate_not_closed")
            self.assertEqual(coverage[0]["order_quality_recommended_action"], "run_quality_gate")

            targets = payload["factory"]["next_targets"]
            self.assertEqual(len(targets), 1)
            target = targets[0]
            self.assertEqual(target["repo_id"], "quality-repo")
            self.assertEqual(target["coverage_state"], "covered_active")
            self.assertEqual(target["attention_reason"], "quality_gate_not_closed")
            self.assertEqual(target["recommended_action"], "run_quality_gate")
            self.assertEqual(target["order_quality_gate_status"], "validated")
            self.assertEqual(target["order_slices_validated"], 1)
            self.assertEqual(target["order_slices_closed"], 0)

            markdown = (out_dir / "latest.md").read_text(encoding="utf-8")
            self.assertIn("## Factory Next Targets", markdown)
            self.assertIn("repo=quality-repo", markdown)
            self.assertIn("reason=quality_gate_not_closed", markdown)
            self.assertIn("action=run_quality_gate", markdown)
            self.assertIn("order_gate=validated", markdown)

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
            active = by_order["active-idle-order"]
            self.assertEqual(active.get("type"), "idle_without_improvement")
            self.assertEqual(active.get("recommended_action"), "close_or_reseed_idle_order")
            paused = by_order["paused-idle-order"]
            self.assertEqual(paused.get("type"), "paused_idle_backlog")
            self.assertEqual(paused.get("phase"), "planning")
            self.assertEqual(paused.get("open_jobs"), 0)
            self.assertGreaterEqual(int(paused.get("last_activity_age_s") or 0), 900)
            self.assertEqual(paused.get("recommended_action"), "resume_or_close_stale_order")
            actions = payload.get("recommended_actions") or []
            paused_action = next((item for item in actions if item.get("order_id") == "paused-idle-order"), None)
            self.assertIsNotNone(paused_action)
            self.assertEqual(paused_action.get("priority"), "P1")
            self.assertEqual(paused_action.get("evidence_type"), "paused_idle_backlog")
            self.assertIn("Resume", paused_action.get("action") or "")

            markdown = (out_dir / "latest.md").read_text(encoding="utf-8")
            self.assertIn("## Recommended Actions", markdown)
            self.assertIn("paused_idle_backlog", markdown)
            self.assertIn("resume_or_close_stale_order", markdown)

    def test_ancient_paused_idle_backlog_is_historical_not_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
            now = time.time()
            ancient = now - 10 * 86400
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
                    ("ancient-paused", "paused", "planning", "Proactive Sprint: old paused", "", ancient),
                )

            proc = self._run_report(root, db, out_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["historical_paused_backlog"], 1)
            self.assertFalse(any(item.get("type") == "paused_idle_backlog" for item in payload.get("anomalies") or []))
            self.assertEqual(payload.get("recommended_actions"), [])
            historical = payload["factory"]["historical_paused_backlog"]
            self.assertEqual(historical[0]["order_id"], "ancient-paused")
            self.assertEqual(historical[0]["recommended_action"], "archived_history_no_current_action")

    def test_healthy_report_has_no_recommended_actions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            db = root / "jobs.sqlite"
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

            proc = self._run_report(root, db, out_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload.get("status"), "OK")
            self.assertEqual(payload.get("recommended_actions"), [])
            self.assertEqual(payload["factory"]["next_targets"], [])
            self.assertEqual(payload["metrics"]["outcome_quality_status"], "OK")
            self.assertFalse(any(item.get("type") == "low_outcome_quality" for item in payload.get("trend_flags") or []))

            markdown = (out_dir / "latest.md").read_text(encoding="utf-8")
            self.assertIn("Outcome quality (24h): status=OK", markdown)
            self.assertIn("## Recommended Actions", markdown)
            self.assertIn("- None", markdown)

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
