from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.status_service import StatusService
from orchestrator.storage import SQLiteTaskStorage


class TestStatusWorkflowSummary(unittest.TestCase):
    def _make_ready_proactive_order(
        self,
        q: OrchestratorQueue,
        *,
        order_id: str,
        trace_extra: dict[str, object] | None = None,
    ) -> None:
        trace = {
            "proactive_lane": True,
            "proactive_slices_applied": 1,
            "proactive_slices_validated": 1,
            "proactive_slices_closed": 1,
            "proactive_quality_gate_status": "validated",
            "proactive_improvement_closed": True,
            "merge_ready": True,
        }
        trace.update(trace_extra or {})
        q.upsert_order(
            order_id=order_id,
            chat_id=1,
            title="Proactive Sprint: release target gate",
            body="Ship one bounded improvement.",
            status="active",
            priority=1,
            phase="review",
            project_id="codexbot",
        )
        q.submit_task(
            Task.new(
                source="scheduler",
                role="skynet",
                input_text="AUTONOMOUS PROACTIVE SPRINT",
                request_type="maintenance",
                priority=1,
                model="gpt-5.3-codex",
                effort="medium",
                mode_hint="ro",
                requires_approval=False,
                max_cost_window_usd=1.0,
                chat_id=1,
                state="done",
                is_autonomous=True,
                trace=trace,
                job_id=order_id,
            )
        )

    def test_order_evidence_packet_includes_workflow_logs_and_artifact_refs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = SQLiteTaskStorage(root / "jobs.sqlite")
            profiles = {
                "skynet": {"role": "skynet", "max_parallel_jobs": 1},
                "implementer_local": {"role": "implementer_local", "max_parallel_jobs": 1},
            }
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)

            order_id = "11111111-2222-3333-4444-666666666666"
            child_id = "aaaaaaaa-bbbb-cccc-dddd-ffffffffffff"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Evidence packet",
                body="Collect order evidence.",
                status="active",
                priority=1,
                phase="executing",
                project_id="codexbot",
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="skynet",
                    input_text="Root order",
                    request_type="maintenance",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    is_autonomous=True,
                    artifacts_dir=str(root / "artifacts" / order_id),
                    trace={"proactive_slices_applied": 1, "result_artifacts": ["root-summary.md"]},
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="implementer_local",
                    input_text="Implement evidence endpoint",
                    request_type="task",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    is_autonomous=True,
                    parent_job_id=order_id,
                    artifacts_dir=str(root / "artifacts" / child_id),
                    labels={"ticket": order_id, "kind": "subtask"},
                    trace={"result_artifacts": ["tests/status.log", "diff.patch"]},
                    job_id=child_id,
                )
            )
            q.append_trace_event(
                order_id=order_id,
                job_id=child_id,
                agent_role="implementer_local",
                event_type="job.done",
                severity="info",
                message="child completed",
                artifact_id="artifact-1",
                payload={"result_artifacts": ["trace/final.json"]},
            )
            q.append_decision_log(
                order_id=order_id,
                job_id=child_id,
                kind="implementation",
                state="done",
                summary="Endpoint implemented",
                next_action=None,
                details={"ok": True},
            )
            q.append_delegation_edge(
                root_ticket_id=order_id,
                from_job_id=order_id,
                to_job_id=child_id,
                edge_type="delegated",
                to_role="implementer_local",
                to_key="evidence_packet",
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)

            self.assertEqual(packet["api_version"], "v1")
            self.assertEqual(packet["schema_version"], 1)
            self.assertEqual(packet["order_id"], order_id)
            self.assertEqual(packet["workflow"]["current_stage"], "validation")
            self.assertEqual(len(packet["children"]), 1)
            self.assertEqual(packet["children"][0]["job_id"], child_id)
            self.assertEqual(len(packet["traces"]), 1)
            self.assertEqual(packet["traces"][0]["artifact_id"], "artifact-1")
            self.assertEqual(len(packet["decision_log"]), 1)
            self.assertEqual(len(packet["delegation_log"]), 1)

            artifacts = packet["artifacts"]
            artifact_paths = {str(a.get("path")) for a in artifacts if a.get("path")}
            artifact_ids = {str(a.get("artifact_id")) for a in artifacts if a.get("artifact_id")}
            self.assertIn(str(root / "artifacts" / order_id), artifact_paths)
            self.assertIn(str(root / "artifacts" / child_id), artifact_paths)
            self.assertIn("root-summary.md", artifact_paths)
            self.assertIn("tests/status.log", artifact_paths)
            self.assertIn("diff.patch", artifact_paths)
            self.assertIn("trace/final.json", artifact_paths)
            self.assertIn("artifact-1", artifact_ids)
            self.assertEqual(packet["counts"]["children"], 1)
            self.assertEqual(packet["counts"]["artifacts"], len(artifacts))

    def test_snapshot_includes_end_to_end_order_workflow_and_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = SQLiteTaskStorage(root / "jobs.sqlite")
            profiles = {
                "skynet": {"role": "skynet", "max_parallel_jobs": 1},
                "backend": {"role": "backend", "max_parallel_jobs": 1},
                "qa": {"role": "qa", "max_parallel_jobs": 1},
            }
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)

            order_id = "11111111-2222-3333-4444-555555555555"
            q.upsert_order(
                order_id=order_id,
                chat_id=1,
                title="Proactive Sprint: codexbot Reliability + Delivery",
                body="Ship one bounded improvement.",
                status="active",
                priority=2,
                phase="review",
                project_id="codexbot-6fb8d5b9",
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="skynet",
                    input_text="AUTONOMOUS PROACTIVE SPRINT",
                    request_type="maintenance",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    is_autonomous=True,
                    trace={
                        "allow_delegation": True,
                        "proactive_lane": True,
                        "proactive_slices_started": 1,
                        "proactive_slices_applied": 1,
                        "proactive_slices_validated": 0,
                        "proactive_slices_closed": 0,
                        "proactive_quality_gate_status": "applied",
                    },
                    job_id=order_id,
                )
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="backend",
                    input_text="Implement one bounded reliability fix.",
                    request_type="task",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="rw",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="done",
                    is_autonomous=True,
                    parent_job_id=order_id,
                    labels={"ticket": order_id, "kind": "subtask", "key": "backend_fix"},
                    job_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                )
            )
            q.submit_task(
                Task.new(
                    source="scheduler",
                    role="skynet",
                    input_text="FINAL SWEEP",
                    request_type="maintenance",
                    priority=1,
                    model="gpt-5.3-codex",
                    effort="medium",
                    mode_hint="ro",
                    requires_approval=False,
                    max_cost_window_usd=1.0,
                    chat_id=1,
                    state="failed",
                    is_autonomous=True,
                    parent_job_id=order_id,
                    labels={"ticket": order_id, "kind": "final_sweep"},
                    trace={
                        "result_summary": "Codex returned code 1.",
                        "live_stdout_tail": (
                            "{\"detail\":\"The 'gpt-5.2-codex' model is not supported when using Codex with a ChatGPT account.\"}"
                        ),
                    },
                    job_id="ffffffff-1111-2222-3333-444444444444",
                )
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            snap = svc.snapshot(chat_id=1)

            self.assertEqual(len(snap["orders_active"]), 1)
            workflow = snap["orders_active"][0]["workflow"]
            self.assertEqual(workflow["current_stage"], "skynet_review")

            stages = {stage["stage"]: stage for stage in workflow["stages"]}
            self.assertEqual(stages["skynet_plan"]["status"], "done")
            self.assertEqual(stages["delivery"]["status"], "done")
            self.assertEqual(stages["validation"]["status"], "pending")
            self.assertEqual(stages["skynet_review"]["status"], "failed")
            self.assertEqual(stages["deploy"]["status"], "pending")

            blockers = workflow["blockers"]
            self.assertEqual(len(blockers), 1)
            self.assertEqual(blockers[0]["stage"], "skynet_review")
            self.assertIn("unsupported", str(blockers[0]["summary"]).lower())
            self.assertEqual(snap["order_workflows"][0]["order_id"], order_id)

    def test_release_readiness_requires_concrete_release_target_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "22222222-3333-4444-5555-666666666666"
            self._make_ready_proactive_order(q, order_id=order_id)

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertIn("Missing concrete release target evidence", checks["release_target_evidence"]["summary"])
            self.assertIn("release_target_evidence", readiness["summary"])
            self.assertIn("Missing concrete release target evidence", readiness["next_action"])

            plan = svc.proactive_action_plan(chat_id=1, limit=10)
            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["release"]["count"], 0)
            self.assertEqual(lanes["advance"]["count"], 1)
            self.assertEqual(lanes["advance"]["orders"][0]["order_id"], order_id)
            self.assertIsNone(lanes["release"]["execution_packet"])

            advance_packet = lanes["advance"]["execution_packet"]
            self.assertIsInstance(advance_packet, dict)
            self.assertEqual(advance_packet["order_id"], order_id)
            self.assertEqual(advance_packet["lane"], "advance")
            self.assertIn(advance_packet["owner_role"], {"implementer_local", "reviewer_local", "release_mgr", "architect_local"})
            self.assertIn(order_id, advance_packet["inspect_endpoint"])
            self.assertIn(order_id, advance_packet["handoff_endpoint"])
            self.assertTrue(advance_packet["acceptance_criteria"])
            self.assertTrue(advance_packet["definition_of_done"])
            self.assertTrue(advance_packet["evidence_required"])
            self.assertTrue(advance_packet["suggested_validation"])
            self.assertIn("ROLE:", advance_packet["assignment_prompt"])
            self.assertEqual(plan["top_execution_packet"], advance_packet)
            self.assertIsInstance(advance_packet.get("outcome_contract"), dict)
            self.assertEqual(
                plan["summary"]["next_delegate"],
                {
                    "owner_role": advance_packet["owner_role"],
                    "order_id": order_id,
                    "lane": "advance",
                    "action": advance_packet["action"],
                    "inspect_endpoint": advance_packet["inspect_endpoint"],
                    "handoff_endpoint": advance_packet["handoff_endpoint"],
                    "acceptance_criteria": advance_packet["acceptance_criteria"],
                    "evidence_required": advance_packet["evidence_required"],
                    "suggested_validation": advance_packet["suggested_validation"],
                    "definition_of_done": advance_packet["definition_of_done"],
                    "assignment_prompt": advance_packet["assignment_prompt"],
                    "outcome_contract": advance_packet["outcome_contract"],
                },
            )

    def test_release_readiness_allows_merge_ready_order_with_order_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "33333333-4444-5555-6666-777777777777"
            self._make_ready_proactive_order(
                q,
                order_id=order_id,
                trace_extra={"order_branch": "feature/release-target-gate"},
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "ready")
            self.assertEqual(readiness["verdict"], "go")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pass")
            self.assertEqual(checks["release_target_evidence"]["evidence"][0]["key"], "order_branch")

            plan = svc.proactive_action_plan(chat_id=1, limit=10)
            lanes = {lane["lane"]: lane for lane in plan["lanes"]}
            self.assertEqual(lanes["release"]["count"], 1)
            self.assertEqual(lanes["release"]["orders"][0]["order_id"], order_id)
            release_packet = lanes["release"]["execution_packet"]
            self.assertIsInstance(release_packet, dict)
            self.assertEqual(release_packet["order_id"], order_id)
            self.assertEqual(release_packet["lane"], "release")
            self.assertEqual(release_packet["owner_role"], "release_mgr")
            self.assertIn(order_id, release_packet["inspect_endpoint"])
            self.assertIn(order_id, release_packet["handoff_endpoint"])
            self.assertIn("Release or merge", " ".join(release_packet["definition_of_done"]))
            self.assertIn("merge or release evidence", " ".join(release_packet["evidence_required"]))
            self.assertIsInstance(release_packet.get("outcome_contract"), dict)

    def test_release_readiness_allows_recovered_github_publication_trace_without_order_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "44444444-5555-6666-7777-888888888888"
            self._make_ready_proactive_order(
                q,
                order_id=order_id,
                trace_extra={
                    "github_publication": {
                        "ok": True,
                        "github_repo": "manolosake/signaldeck",
                        "github_url": "https://github.com/manolosake/signaldeck.git",
                        "remote_url": "https://github.com/manolosake/signaldeck.git",
                        "branch": "main",
                        "default_branch": "main",
                        "head": "2efec0a",
                        "latest_head": "2efec0a",
                        "project_path": "/home/aponce/signaldeck",
                        "private": True,
                    }
                },
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "ready")
            self.assertEqual(readiness["verdict"], "go")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pass")
            self.assertEqual(checks["release_target_evidence"]["evidence"][0]["key"], "github_publication")
            self.assertEqual(
                checks["release_target_evidence"]["evidence"][0]["value"],
                "manolosake/signaldeck",
            )

    def test_release_readiness_allows_recovered_github_publication_trace_with_ssh_remote_url(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "45454545-5656-6767-7878-898989898989"
            self._make_ready_proactive_order(
                q,
                order_id=order_id,
                trace_extra={
                    "github_publication": {
                        "ok": True,
                        "remote_url": "ssh://git@github.com/manolosake/signaldeck.git",
                        "branch": "main",
                        "default_branch": "main",
                        "head": "2efec0a",
                        "latest_head": "2efec0a",
                        "project_path": "/home/aponce/signaldeck",
                        "private": True,
                    }
                },
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "ready")
            self.assertEqual(readiness["verdict"], "go")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pass")
            self.assertEqual(checks["release_target_evidence"]["evidence"][0]["key"], "github_publication")
            self.assertEqual(
                checks["release_target_evidence"]["evidence"][0]["value"],
                "ssh://git@github.com/manolosake/signaldeck.git",
            )

    def test_release_readiness_allows_persisted_private_publication_recovery_without_root_trace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "45959595-6060-7171-8282-939393939393"
            self._make_ready_proactive_order(q, order_id=order_id)

            with sqlite3.connect(storage.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE studio_publication_recovery (
                        recovery_id TEXT PRIMARY KEY,
                        project_key TEXT NOT NULL,
                        project_name TEXT,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        latest_head TEXT,
                        private INTEGER,
                        missing_json TEXT NOT NULL DEFAULT '[]',
                        required_action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        source_order_id TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO studio_publication_recovery(
                        recovery_id, project_key, project_name, project_path, github_repo, github_url,
                        latest_head, private, missing_json, required_action, status, reason, source_order_id, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "recovery-1",
                        "signaldeck",
                        "SignalDeck",
                        "/home/aponce/signaldeck",
                        "manolosake/signaldeck",
                        "https://github.com/manolosake/signaldeck.git",
                        "2efec0a",
                        1,
                        "[]",
                        "resolve_publication_contract",
                        "published_private",
                        "Recovered GitHub target evidence from SQLite.",
                        order_id,
                        11.0,
                        21.0,
                    ),
                )
                conn.commit()

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "ready")
            self.assertEqual(readiness["verdict"], "go")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pass")
            self.assertEqual(checks["release_target_evidence"]["evidence"][0]["key"], "persisted_github_publication")
            self.assertEqual(
                checks["release_target_evidence"]["evidence"][0]["value"],
                "manolosake/signaldeck",
            )

    def test_release_readiness_rejects_recovery_only_publication_without_private_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "45969696-7070-8181-9292-030303030303"
            self._make_ready_proactive_order(q, order_id=order_id)

            with sqlite3.connect(storage.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE studio_publication_recovery (
                        recovery_id TEXT PRIMARY KEY,
                        project_key TEXT NOT NULL,
                        project_name TEXT,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        latest_head TEXT,
                        private INTEGER,
                        missing_json TEXT NOT NULL DEFAULT '[]',
                        required_action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        source_order_id TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO studio_publication_recovery(
                        recovery_id, project_key, project_name, project_path, github_repo, github_url,
                        latest_head, private, missing_json, required_action, status, reason, source_order_id, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "recovery-missing-private",
                        "signaldeck",
                        "SignalDeck",
                        "/home/aponce/signaldeck",
                        "manolosake/signaldeck",
                        "https://github.com/manolosake/signaldeck.git",
                        "2efec0a",
                        None,
                        "[\"private\"]",
                        "resolve_publication_contract",
                        "published_private",
                        "Missing explicit private visibility evidence should keep readiness pending.",
                        order_id,
                        11.0,
                        21.0,
                    ),
                )
                conn.commit()

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertEqual(checks["release_target_evidence"]["evidence"], [])

    def test_release_readiness_merges_portfolio_latest_order_match_with_older_recovery_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            source_order_id = "47070707-8181-9292-0303-141414141414"
            latest_order_id = "47181818-9292-0303-1414-252525252525"
            self._make_ready_proactive_order(q, order_id=latest_order_id)

            with sqlite3.connect(storage.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE studio_portfolio_projects (
                        project_key TEXT PRIMARY KEY,
                        project_name TEXT NOT NULL,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        default_branch TEXT,
                        latest_head TEXT,
                        private INTEGER,
                        status TEXT NOT NULL,
                        source_order_id TEXT,
                        latest_order_id TEXT,
                        latest_outcome_status TEXT,
                        latest_summary TEXT,
                        validation_summary TEXT,
                        monetization_summary TEXT,
                        next_milestone TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE studio_publication_recovery (
                        recovery_id TEXT PRIMARY KEY,
                        project_key TEXT NOT NULL,
                        project_name TEXT,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        latest_head TEXT,
                        missing_json TEXT NOT NULL DEFAULT '[]',
                        required_action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        source_order_id TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO studio_portfolio_projects(
                        project_key, project_name, project_path, github_repo, github_url, default_branch,
                        latest_head, private, status, source_order_id, latest_order_id, latest_outcome_status,
                        latest_summary, validation_summary, monetization_summary, next_milestone, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "signaldeck",
                        "SignalDeck",
                        "/home/aponce/signaldeck",
                        "manolosake/signaldeck",
                        "",
                        "main",
                        "",
                        1,
                        "published_private",
                        source_order_id,
                        latest_order_id,
                        "published_project",
                        "Portfolio row moved forward to the latest order.",
                        None,
                        None,
                        None,
                        10.0,
                        20.0,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO studio_publication_recovery(
                        recovery_id, project_key, project_name, project_path, github_repo, github_url,
                        latest_head, missing_json, required_action, status, reason, source_order_id, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "recovery-older-order",
                        "signaldeck",
                        "SignalDeck",
                        "/home/aponce/signaldeck",
                        "",
                        "https://github.com/manolosake/signaldeck.git",
                        "2efec0a",
                        "[]",
                        "resolve_publication_contract",
                        "resolved",
                        "Recovered GitHub target evidence from the original source order.",
                        source_order_id,
                        11.0,
                        21.0,
                    ),
                )
                conn.commit()

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(latest_order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "ready")
            self.assertEqual(readiness["verdict"], "go")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pass")
            self.assertEqual(checks["release_target_evidence"]["evidence"][0]["key"], "persisted_github_publication")
            self.assertEqual(
                checks["release_target_evidence"]["evidence"][0]["value"],
                "manolosake/signaldeck",
            )

    def test_release_readiness_does_not_inherit_recovery_only_evidence_from_older_order_via_project_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            older_order_id = "47191919-0303-1414-2525-363636363636"
            current_order_id = "47202020-1414-2525-3636-474747474747"
            self._make_ready_proactive_order(q, order_id=current_order_id)

            with sqlite3.connect(storage.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE studio_portfolio_projects (
                        project_key TEXT PRIMARY KEY,
                        project_name TEXT NOT NULL,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        default_branch TEXT,
                        latest_head TEXT,
                        private INTEGER,
                        status TEXT NOT NULL,
                        source_order_id TEXT,
                        latest_order_id TEXT,
                        latest_outcome_status TEXT,
                        latest_summary TEXT,
                        validation_summary TEXT,
                        monetization_summary TEXT,
                        next_milestone TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE studio_publication_recovery (
                        recovery_id TEXT PRIMARY KEY,
                        project_key TEXT NOT NULL,
                        project_name TEXT,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        latest_head TEXT,
                        private INTEGER,
                        missing_json TEXT NOT NULL DEFAULT '[]',
                        required_action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        source_order_id TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO studio_portfolio_projects(
                        project_key, project_name, project_path, github_repo, github_url, default_branch,
                        latest_head, private, status, source_order_id, latest_order_id, latest_outcome_status,
                        latest_summary, validation_summary, monetization_summary, next_milestone, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "signaldeck",
                        "SignalDeck",
                        "/home/aponce/signaldeck",
                        "manolosake/signaldeck",
                        "",
                        "main",
                        "",
                        0,
                        "draft",
                        current_order_id,
                        current_order_id,
                        "work_in_progress",
                        "Current order portfolio row is not yet published/private.",
                        None,
                        None,
                        None,
                        10.0,
                        30.0,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO studio_publication_recovery(
                        recovery_id, project_key, project_name, project_path, github_repo, github_url,
                        latest_head, private, missing_json, required_action, status, reason, source_order_id, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "recovery-older-order",
                        "signaldeck",
                        "SignalDeck",
                        "/home/aponce/signaldeck",
                        "manolosake/signaldeck",
                        "https://github.com/manolosake/signaldeck.git",
                        "2efec0a",
                        1,
                        "[]",
                        "resolve_publication_contract",
                        "published_private",
                        "Older order has complete recovery evidence, but the current order should not inherit it.",
                        older_order_id,
                        11.0,
                        21.0,
                    ),
                )
                conn.commit()

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(current_order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertEqual(checks["release_target_evidence"]["evidence"], [])

    def test_release_readiness_prefers_resolved_recovery_row_over_newer_incomplete_row_for_same_project(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            source_order_id = "47292929-0303-1414-2525-363636363636"
            latest_order_id = "47303030-1414-2525-3636-474747474747"
            self._make_ready_proactive_order(q, order_id=latest_order_id)

            with sqlite3.connect(storage.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE studio_portfolio_projects (
                        project_key TEXT PRIMARY KEY,
                        project_name TEXT NOT NULL,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        default_branch TEXT,
                        latest_head TEXT,
                        private INTEGER,
                        status TEXT NOT NULL,
                        source_order_id TEXT,
                        latest_order_id TEXT,
                        latest_outcome_status TEXT,
                        latest_summary TEXT,
                        validation_summary TEXT,
                        monetization_summary TEXT,
                        next_milestone TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE studio_publication_recovery (
                        recovery_id TEXT PRIMARY KEY,
                        project_key TEXT NOT NULL,
                        project_name TEXT,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        latest_head TEXT,
                        missing_json TEXT NOT NULL DEFAULT '[]',
                        required_action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        source_order_id TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO studio_portfolio_projects(
                        project_key, project_name, project_path, github_repo, github_url, default_branch,
                        latest_head, private, status, source_order_id, latest_order_id, latest_outcome_status,
                        latest_summary, validation_summary, monetization_summary, next_milestone, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "signaldeck",
                        "SignalDeck",
                        "/home/aponce/signaldeck",
                        "manolosake/signaldeck",
                        "",
                        "main",
                        "",
                        1,
                        "published_private",
                        source_order_id,
                        latest_order_id,
                        "published_project",
                        "Portfolio row relies on recovery evidence.",
                        None,
                        None,
                        None,
                        10.0,
                        30.0,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO studio_publication_recovery(
                        recovery_id, project_key, project_name, project_path, github_repo, github_url,
                        latest_head, missing_json, required_action, status, reason, source_order_id, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "recovery-resolved",
                        "signaldeck",
                        "SignalDeck",
                        "/home/aponce/signaldeck",
                        "",
                        "https://github.com/manolosake/signaldeck.git",
                        "2efec0a",
                        "[]",
                        "resolve_publication_contract",
                        "resolved",
                        "Older resolved row has the complete GitHub publication evidence.",
                        source_order_id,
                        11.0,
                        21.0,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO studio_publication_recovery(
                        recovery_id, project_key, project_name, project_path, github_repo, github_url,
                        latest_head, missing_json, required_action, status, reason, source_order_id, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "recovery-open",
                        "signaldeck",
                        "SignalDeck",
                        "/home/aponce/signaldeck",
                        "",
                        "",
                        "",
                        "[\"github_url\", \"latest_head\"]",
                        "resolve_publication_contract",
                        "open",
                        "Newer row is incomplete and should not shadow the resolved record.",
                        source_order_id,
                        12.0,
                        31.0,
                    ),
                )
                conn.commit()

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(latest_order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "ready")
            self.assertEqual(readiness["verdict"], "go")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pass")
            self.assertEqual(checks["release_target_evidence"]["evidence"][0]["key"], "persisted_github_publication")
            self.assertEqual(
                checks["release_target_evidence"]["evidence"][0]["value"],
                "manolosake/signaldeck",
            )

    def test_release_readiness_rejects_persisted_publication_recovery_without_valid_github_url(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "46060606-7171-8282-9393-040404040404"
            self._make_ready_proactive_order(q, order_id=order_id)

            with sqlite3.connect(storage.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE studio_portfolio_projects (
                        project_key TEXT PRIMARY KEY,
                        project_name TEXT NOT NULL,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        default_branch TEXT,
                        latest_head TEXT,
                        private INTEGER,
                        status TEXT NOT NULL,
                        source_order_id TEXT,
                        latest_order_id TEXT,
                        latest_outcome_status TEXT,
                        latest_summary TEXT,
                        validation_summary TEXT,
                        monetization_summary TEXT,
                        next_milestone TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE studio_publication_recovery (
                        recovery_id TEXT PRIMARY KEY,
                        project_key TEXT NOT NULL,
                        project_name TEXT,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        latest_head TEXT,
                        missing_json TEXT NOT NULL DEFAULT '[]',
                        required_action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        source_order_id TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO studio_portfolio_projects(
                        project_key, project_name, project_path, github_repo, github_url, default_branch,
                        latest_head, private, status, source_order_id, latest_order_id, latest_outcome_status,
                        latest_summary, validation_summary, monetization_summary, next_milestone, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "broken-repo",
                        "Broken Repo",
                        "/home/aponce/broken-repo",
                        "manolosake/broken-repo",
                        "",
                        "main",
                        "2efec0a",
                        1,
                        "published_private",
                        order_id,
                        order_id,
                        "published_project",
                        "Persisted publication exists but target is malformed.",
                        None,
                        None,
                        None,
                        10.0,
                        20.0,
                    ),
                )
                conn.commit()

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertEqual(checks["release_target_evidence"]["evidence"], [])

    def test_release_readiness_rejects_persisted_publication_recovery_on_repo_url_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "46171717-8282-9393-0404-151515151515"
            self._make_ready_proactive_order(q, order_id=order_id)

            with sqlite3.connect(storage.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE studio_portfolio_projects (
                        project_key TEXT PRIMARY KEY,
                        project_name TEXT NOT NULL,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        default_branch TEXT,
                        latest_head TEXT,
                        private INTEGER,
                        status TEXT NOT NULL,
                        source_order_id TEXT,
                        latest_order_id TEXT,
                        latest_outcome_status TEXT,
                        latest_summary TEXT,
                        validation_summary TEXT,
                        monetization_summary TEXT,
                        next_milestone TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO studio_portfolio_projects(
                        project_key, project_name, project_path, github_repo, github_url, default_branch,
                        latest_head, private, status, source_order_id, latest_order_id, latest_outcome_status,
                        latest_summary, validation_summary, monetization_summary, next_milestone, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "mismatch-repo",
                        "Mismatch Repo",
                        "/home/aponce/mismatch-repo",
                        "manolosake/right-repo",
                        "https://github.com/manolosake/wrong-repo.git",
                        "main",
                        "2efec0a",
                        1,
                        "published_private",
                        order_id,
                        order_id,
                        "published_project",
                        "Persisted publication has mismatched repo identity.",
                        None,
                        None,
                        None,
                        10.0,
                        20.0,
                    ),
                )
                conn.commit()

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertEqual(checks["release_target_evidence"]["evidence"], [])

    def test_release_readiness_rejects_persisted_publication_recovery_with_non_github_url(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "46282828-9393-0404-1515-262626262626"
            self._make_ready_proactive_order(q, order_id=order_id)

            with sqlite3.connect(storage.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE studio_portfolio_projects (
                        project_key TEXT PRIMARY KEY,
                        project_name TEXT NOT NULL,
                        project_path TEXT,
                        github_repo TEXT,
                        github_url TEXT,
                        default_branch TEXT,
                        latest_head TEXT,
                        private INTEGER,
                        status TEXT NOT NULL,
                        source_order_id TEXT,
                        latest_order_id TEXT,
                        latest_outcome_status TEXT,
                        latest_summary TEXT,
                        validation_summary TEXT,
                        monetization_summary TEXT,
                        next_milestone TEXT,
                        first_seen_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO studio_portfolio_projects(
                        project_key, project_name, project_path, github_repo, github_url, default_branch,
                        latest_head, private, status, source_order_id, latest_order_id, latest_outcome_status,
                        latest_summary, validation_summary, monetization_summary, next_milestone, first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "nongithub-repo",
                        "NonGitHub Repo",
                        "/home/aponce/nongithub-repo",
                        "manolosake/nongithub-repo",
                        "https://gitlab.com/manolosake/nongithub-repo.git",
                        "main",
                        "2efec0a",
                        1,
                        "published_private",
                        order_id,
                        order_id,
                        "published_project",
                        "Persisted publication points to a non-GitHub remote.",
                        None,
                        None,
                        None,
                        10.0,
                        20.0,
                    ),
                )
                conn.commit()

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertEqual(checks["release_target_evidence"]["evidence"], [])

    def test_release_readiness_rejects_recovered_github_publication_trace_with_malformed_ssh_remote_url(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "46464646-5757-6868-7979-909090909090"
            self._make_ready_proactive_order(
                q,
                order_id=order_id,
                trace_extra={
                    "github_publication": {
                        "ok": True,
                        "remote_url": "ssh://git@github.com/",
                        "branch": "main",
                        "default_branch": "main",
                        "head": "2efec0a",
                        "latest_head": "2efec0a",
                        "project_path": "/home/aponce/signaldeck",
                        "private": True,
                    }
                },
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertEqual(checks["release_target_evidence"]["evidence"], [])

    def test_release_readiness_rejects_local_only_recovered_github_publication(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "55555555-6666-7777-8888-999999999999"
            self._make_ready_proactive_order(
                q,
                order_id=order_id,
                trace_extra={
                    "github_publication": {
                        "ok": True,
                        "project_path": "/home/aponce/local-only-project",
                        "latest_head": "2efec0a",
                    }
                },
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertEqual(checks["release_target_evidence"]["evidence"], [])

    def test_release_readiness_rejects_incomplete_recovered_github_publication(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "66666666-7777-8888-9999-000000000000"
            self._make_ready_proactive_order(
                q,
                order_id=order_id,
                trace_extra={
                    "github_publication": {
                        "ok": True,
                        "github_repo": "manolosake/signaldeck",
                        "github_url": "https://github.com/manolosake/signaldeck.git",
                        "project_path": "/home/aponce/signaldeck",
                    }
                },
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertEqual(checks["release_target_evidence"]["evidence"], [])

    def test_release_readiness_rejects_recovered_github_publication_when_private_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "77777777-8888-9999-0000-111111111111"
            self._make_ready_proactive_order(
                q,
                order_id=order_id,
                trace_extra={
                    "github_publication": {
                        "ok": True,
                        "github_repo": "manolosake/signaldeck",
                        "github_url": "https://github.com/manolosake/signaldeck.git",
                        "remote_url": "https://github.com/manolosake/signaldeck.git",
                        "branch": "main",
                        "default_branch": "main",
                        "head": "2efec0a",
                        "latest_head": "2efec0a",
                        "project_path": "/home/aponce/signaldeck",
                        "private": False,
                    }
                },
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertEqual(checks["release_target_evidence"]["evidence"], [])

    def test_release_readiness_rejects_recovered_github_publication_when_private_is_false_string(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = SQLiteTaskStorage(Path(td) / "jobs.sqlite")
            profiles = {"skynet": {"role": "skynet", "max_parallel_jobs": 1}}
            q = OrchestratorQueue(storage=storage, role_profiles=profiles)
            order_id = "88888888-9999-0000-1111-222222222222"
            self._make_ready_proactive_order(
                q,
                order_id=order_id,
                trace_extra={
                    "github_publication": {
                        "ok": True,
                        "github_repo": "manolosake/signaldeck",
                        "github_url": "https://github.com/manolosake/signaldeck.git",
                        "remote_url": "https://github.com/manolosake/signaldeck.git",
                        "branch": "main",
                        "default_branch": "main",
                        "head": "2efec0a",
                        "latest_head": "2efec0a",
                        "project_path": "/home/aponce/signaldeck",
                        "private": "false",
                    }
                },
            )

            svc = StatusService(orch_q=q, role_profiles=profiles, cache_ttl_seconds=0)
            packet = svc.order_evidence_packet(order_id)
            readiness = packet["release_readiness"]

            self.assertEqual(readiness["state"], "not_ready")
            self.assertEqual(readiness["verdict"], "wait")
            checks = {check["key"]: check for check in readiness["checks"]}
            self.assertEqual(checks["release_target_evidence"]["status"], "pending")
            self.assertEqual(checks["release_target_evidence"]["evidence"], [])


if __name__ == "__main__":
    unittest.main()
