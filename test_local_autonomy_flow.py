from __future__ import annotations

import tempfile
import time
import subprocess
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bot
from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.storage import SQLiteTaskStorage


def _role_profiles() -> dict[str, dict[str, str]]:
    return {
        "architect_local": {"role": "architect_local"},
        "implementer_local": {"role": "implementer_local"},
        "reviewer_local": {"role": "reviewer_local"},
        "skynet": {"role": "skynet"},
    }


def _new_task(
    *,
    role: str,
    parent_job_id: str,
    state: str = "queued",
    key: str = "k",
    trace: dict | None = None,
) -> Task:
    return Task.new(
        source="test",
        role=role,
        input_text="test",
        request_type="task",
        priority=2,
        model="test-model",
        effort="medium",
        mode_hint="ro",
        requires_approval=False,
        max_cost_window_usd=1.0,
        chat_id=1,
        parent_job_id=parent_job_id,
        state=state,
        labels={"ticket": parent_job_id, "key": key},
        trace=(trace or {}),
        artifacts_dir="/tmp",
    )


class TestLocalAutonomyFlow(unittest.TestCase):
    def _queue(self) -> tuple[tempfile.TemporaryDirectory, OrchestratorQueue]:
        td = tempfile.TemporaryDirectory()
        storage = SQLiteTaskStorage(Path(td.name) / "jobs.sqlite")
        q = OrchestratorQueue(storage=storage, role_profiles=_role_profiles())
        return td, q

    def test_funnel_closes_only_with_all_quality_gates(self) -> None:
        td, q = self._queue()
        self.addCleanup(td.cleanup)
        order_id = "11111111-1111-1111-1111-111111111111"
        slice_id = "slice_alpha"

        impl = _new_task(role="implementer_local", parent_job_id=order_id, key=f"local_impl_guard_{slice_id}")
        q.submit_task(impl)
        q.update_state(
            impl.job_id,
            "done",
            result_summary="implemented and validated",
            slice_id=slice_id,
            slice_status="validated",
            quality_gate_status="validated",
            failure_class="retriable",
            attempt_n=1,
            slice_patch_applied=True,
            slice_validation_ok=True,
        )

        rev = _new_task(role="reviewer_local", parent_job_id=order_id, key=f"local_review_guard_{slice_id}")
        q.submit_task(rev)
        q.update_state(
            rev.job_id,
            "done",
            result_summary="READY: validation pass",
            slice_id=slice_id,
            slice_status="reviewed_ready",
            quality_gate_status="reviewed_ready",
            failure_class="retriable",
            attempt_n=1,
        )

        ctl = _new_task(role="skynet", parent_job_id=order_id, key="final_sweep")
        q.submit_task(ctl)
        q.update_state(
            ctl.job_id,
            "done",
            result_summary="VERIFIED_IMPROVEMENT",
            slice_id=slice_id,
            slice_status="closed",
            quality_gate_status="closed",
            failure_class="retriable",
            improvement_verified=True,
        )

        funnel = bot._collect_order_local_autonomy_funnel(orch_q=q, root_ticket=order_id, now=time.time())
        self.assertEqual(funnel["slices_started"], 1)
        self.assertEqual(funnel["slices_applied"], 1)
        self.assertEqual(funnel["slices_validated"], 1)
        self.assertEqual(funnel["slices_closed"], 1)
        self.assertTrue(funnel["improvement_verified"])
        self.assertEqual(funnel["quality_gate_status"], "closed")

    def test_funnel_does_not_close_when_reviewer_ready_without_validated_slice(self) -> None:
        td, q = self._queue()
        self.addCleanup(td.cleanup)
        order_id = "22222222-2222-2222-2222-222222222222"
        slice_id = "slice_beta"

        impl = _new_task(role="implementer_local", parent_job_id=order_id, key=f"local_impl_guard_{slice_id}")
        q.submit_task(impl)
        q.update_state(
            impl.job_id,
            "done",
            result_summary="generated diff only",
            slice_id=slice_id,
            slice_status="applied",
            quality_gate_status="applied",
            failure_class="retriable",
            attempt_n=1,
            slice_patch_applied=True,
            slice_validation_ok=False,
        )

        rev = _new_task(role="reviewer_local", parent_job_id=order_id, key=f"local_review_guard_{slice_id}")
        q.submit_task(rev)
        q.update_state(
            rev.job_id,
            "done",
            result_summary="READY",
            slice_id=slice_id,
            slice_status="reviewed_ready",
            quality_gate_status="reviewed_ready",
            failure_class="retriable",
            attempt_n=1,
        )

        ctl = _new_task(role="skynet", parent_job_id=order_id, key="final_sweep")
        q.submit_task(ctl)
        q.update_state(
            ctl.job_id,
            "done",
            result_summary="VERIFIED_IMPROVEMENT",
            slice_id=slice_id,
            slice_status="closed",
            quality_gate_status="closed",
            failure_class="retriable",
            improvement_verified=True,
        )

        funnel = bot._collect_order_local_autonomy_funnel(orch_q=q, root_ticket=order_id, now=time.time())
        self.assertEqual(funnel["slices_applied"], 1)
        self.assertEqual(funnel["slices_validated"], 0)
        self.assertEqual(funnel["slices_closed"], 0)
        self.assertFalse(funnel["improvement_verified"])

    def test_funnel_maps_controller_verification_without_slice_to_latest_ready_slice(self) -> None:
        td, q = self._queue()
        self.addCleanup(td.cleanup)
        order_id = "33333333-3333-3333-3333-333333333333"
        slice_id = "slice_gamma"

        impl = _new_task(role="implementer_local", parent_job_id=order_id, key=f"local_impl_guard_{slice_id}")
        q.submit_task(impl)
        q.update_state(
            impl.job_id,
            "done",
            result_summary="validated implementation",
            slice_id=slice_id,
            slice_status="validated",
            quality_gate_status="validated",
            failure_class="retriable",
            attempt_n=1,
            slice_patch_applied=True,
            slice_validation_ok=True,
        )
        rev = _new_task(role="reviewer_local", parent_job_id=order_id, key=f"local_review_guard_{slice_id}")
        q.submit_task(rev)
        q.update_state(
            rev.job_id,
            "done",
            result_summary="READY with evidence",
            slice_id=slice_id,
            slice_status="reviewed_ready",
            quality_gate_status="reviewed_ready",
            failure_class="retriable",
            attempt_n=1,
        )

        ctl = _new_task(role="skynet", parent_job_id=order_id, key="final_sweep")
        q.submit_task(ctl)
        q.update_state(
            ctl.job_id,
            "done",
            result_summary="VERIFIED_IMPROVEMENT",
            improvement_verified=True,
        )

        funnel = bot._collect_order_local_autonomy_funnel(orch_q=q, root_ticket=order_id, now=time.time())
        self.assertEqual(funnel["slices_closed"], 1)
        self.assertTrue(funnel["improvement_verified"])

    def test_failure_class_patch_apply_errors_retry_once_then_terminal(self) -> None:
        msg = "patch apply failed: hunk error"
        first = bot._classify_local_slice_failure(
            role_norm="implementer_local",
            orch_state="failed",
            summary=msg,
            attempt_n=1,
        )
        second = bot._classify_local_slice_failure(
            role_norm="implementer_local",
            orch_state="failed",
            summary=msg,
            attempt_n=2,
        )
        self.assertEqual(first, "retriable")
        self.assertEqual(second, "terminal")

    def test_failure_class_no_valid_patches_is_terminal_immediately(self) -> None:
        msg = 'patch rejected by git apply --check: error: No valid patches in input (allow with "--allow-empty")'
        klass = bot._classify_local_slice_failure(
            role_norm="implementer_local",
            orch_state="failed",
            summary=msg,
            attempt_n=1,
        )
        self.assertEqual(klass, "terminal")

    def test_failure_class_no_worktree_changes_after_apply_is_terminal_immediately(self) -> None:
        msg = "implementer_local produced no worktree changes after apply; nothing to commit"
        klass = bot._classify_local_slice_failure(
            role_norm="implementer_local",
            orch_state="failed",
            summary=msg,
            attempt_n=1,
        )
        self.assertEqual(klass, "terminal")

    def test_failure_class_patch_apply_noop_markers_are_terminal_immediately(self) -> None:
        cases = (
            'patch apply failed: allow with "--allow-empty"',
            "patch apply failed: patch is empty",
            "patch apply failed: nothing to apply",
        )
        for msg in cases:
            with self.subTest(msg=msg):
                klass = bot._classify_local_slice_failure(
                    role_norm="implementer_local",
                    orch_state="failed",
                    summary=msg,
                    attempt_n=1,
                )
                self.assertEqual(klass, "terminal")

    def test_failure_class_local_env_blockers_are_terminal_immediately(self) -> None:
        cases = (
            "error: read-only file system",
            "error: filesystem is read-only",
            "error: read only file system",
            "bwrap: failed to create namespace",
            "bubblewrap execution failed",
        )
        for msg in cases:
            with self.subTest(msg=msg):
                klass = bot._classify_local_slice_failure(
                    role_norm="implementer_local",
                    orch_state="failed",
                    summary=msg,
                    attempt_n=1,
                )
                self.assertEqual(klass, "terminal")

    def test_failure_class_patch_apply_eperm_retries_once_then_terminal(self) -> None:
        msg = "patch apply failed: operation not permitted"
        first = bot._classify_local_slice_failure(
            role_norm="implementer_local",
            orch_state="failed",
            summary=msg,
            attempt_n=1,
        )
        second = bot._classify_local_slice_failure(
            role_norm="implementer_local",
            orch_state="failed",
            summary=msg,
            attempt_n=2,
        )
        self.assertEqual(first, "retriable")
        self.assertEqual(second, "terminal")

    def test_failure_class_blocker_text_is_blocked(self) -> None:
        msg = "BLOCKER: missing requirement for evidence artifact path"
        klass = bot._classify_local_slice_failure(
            role_norm="implementer_local",
            orch_state="failed",
            summary=msg,
            attempt_n=1,
        )
        self.assertEqual(klass, "blocked")

    def test_prune_local_specs_blocks_reviewer_without_applied_evidence(self) -> None:
        specs = [
            bot.TaskSpec(key="review", role="reviewer_local", text="review", mode_hint="ro", priority=1),
            bot.TaskSpec(key="impl", role="implementer_local", text="impl", mode_hint="rw", priority=1),
        ]
        pruned = bot._prune_local_specs_against_active_backlog(specs=specs, existing_children=[])
        roles = [bot._coerce_orchestrator_role(str(s.role or "")) for s in pruned]
        self.assertNotIn("reviewer_local", roles)
        self.assertIn("implementer_local", roles)

    def test_prune_allows_replan_when_only_blocked_implementer_exists(self) -> None:
        order_id = "44444444-4444-4444-4444-444444444444"
        blocked_impl = _new_task(
            role="implementer_local",
            parent_job_id=order_id,
            state="blocked",
            key="local_impl_guard_slice_blocked",
            trace={
                "slice_id": "slice_blocked",
                "slice_status": "blocked",
                "quality_gate_status": "blocked",
                "failure_class": "blocked",
                "attempt_n": 1,
            },
        )
        specs = [
            bot.TaskSpec(key="arch", role="architect_local", text="replan", mode_hint="ro", priority=1),
        ]
        pruned = bot._prune_local_specs_against_active_backlog(specs=specs, existing_children=[blocked_impl])
        roles = [bot._coerce_orchestrator_role(str(s.role or "")) for s in pruned]
        self.assertIn("architect_local", roles)

    def test_apply_local_first_policy_directs_grounded_excerpt_blocker_to_implementer(self) -> None:
        td, q = self._queue()
        self.addCleanup(td.cleanup)
        order_id = "55555555-5555-5555-5555-555555555555"
        summary = (
            "BLOCKER: Missing current excerpt for `bot.py` around `_classify_local_slice_failure` "
            "(full function body). Please provide the exact current function body so I can add the "
            "terminal classification predicate."
        )
        impl = _new_task(
            role="implementer_local",
            parent_job_id=order_id,
            key="local_impl_guard_slice_excerpt",
            trace={
                "slice_id": "slice_excerpt",
                "slice_status": "blocked",
                "quality_gate_status": "blocked",
                "failure_class": "blocked",
                "attempt_n": 1,
            },
        )
        q.submit_task(impl)
        q.update_state(
            impl.job_id,
            "blocked",
            result_summary=summary,
            slice_id="slice_excerpt",
            slice_status="blocked",
            quality_gate_status="blocked",
            failure_class="blocked",
            attempt_n=1,
        )

        specs = [
            bot.TaskSpec(
                key="backend_followup",
                role="backend",
                text="Investigate the local autonomy blocker.",
                mode_hint="ro",
                priority=2,
                acceptance_criteria=["Produce one next step."],
                definition_of_done=["One next step is identified."],
                eta_minutes=15,
                sla_tier="medium",
            )
        ]
        with tempfile.TemporaryDirectory() as worktree_td:
            worktree = Path(worktree_td)
            (worktree / "bot.py").write_text(
                "def _classify_local_slice_failure(*, role_norm: str, orch_state: str, summary: str, attempt_n: int) -> str:\n"
                "    return 'blocked'\n",
                encoding="utf-8",
            )
            with patch.object(bot, "_local_role_worktree_dir", return_value=worktree):
                planned = bot._apply_autonomous_local_first_policy(
                    specs=specs,
                    orch_q=q,
                    root_ticket=order_id,
                    now=time.time(),
                )

        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].role, "implementer_local")
        self.assertIn("LATEST_IMPLEMENTER_BLOCKER", planned[0].text)
        self.assertIn("- bot.py", planned[0].text)
        self.assertIn("python3 -m py_compile bot.py", planned[0].text)

    def test_structured_handoff_actionable_requires_sections(self) -> None:
        good = (
            "FILES:\n"
            "- tools/backend_done_evidence_guard.py\n"
            "CHANGE:\n"
            "- add missing evidence_artifact_exists check.\n"
            "VALIDATION:\n"
            "- pytest -q test_backend_done_evidence_guard.py\n"
            "RISK:\n"
            "- false negatives if artifact path is wrong.\n"
        )
        bad = (
            "FILES:\n"
            "- tools/backend_done_evidence_guard.py\n"
            "CHANGE:\n"
            "- tweak behavior.\n"
        )
        self.assertTrue(bot._structured_handoff_is_actionable(good))
        self.assertFalse(bot._structured_handoff_is_actionable(bad))

    def test_structured_handoff_artifact_contract_gate(self) -> None:
        without_contract = (
            "FILES:\n"
            "- tools/backend_done_evidence_guard.py\n"
            "CHANGE:\n"
            "- add evidence existence validation.\n"
            "VALIDATION:\n"
            "- pytest -q test_backend_done_evidence_guard.py\n"
            "RISK:\n"
            "- path mismatch.\n"
        )
        with_contract = without_contract + "ARTIFACT_CONTRACT:\n- data/artifacts/<ticket>/final_evidence.json\n"
        self.assertFalse(
            bot._structured_handoff_is_actionable(
                without_contract,
                require_artifact_contract=True,
            )
        )
        self.assertTrue(
            bot._structured_handoff_is_actionable(
                with_contract,
                require_artifact_contract=True,
            )
        )

    def test_extract_changed_files_from_patch_text(self) -> None:
        patch = (
            "diff --git a/orchestrator/workspaces.py b/orchestrator/workspaces.py\n"
            "--- a/orchestrator/workspaces.py\n"
            "+++ b/orchestrator/workspaces.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/tools/example.py b/tools/example.py\n"
            "--- a/tools/example.py\n"
            "+++ b/tools/example.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        )
        self.assertEqual(
            bot._extract_changed_files_from_patch_text(patch),
            ["orchestrator/workspaces.py", "tools/example.py"],
        )

    def test_finalize_local_implementer_change_validates_only_current_slice_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            env = {
                **dict(os.environ),
                "GIT_AUTHOR_NAME": "Codex",
                "GIT_AUTHOR_EMAIL": "codex@example.com",
                "GIT_COMMITTER_NAME": "Codex",
                "GIT_COMMITTER_EMAIL": "codex@example.com",
            }
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True, env=env)
            good = repo / "good.py"
            bad = repo / "bad.py"
            good.write_text("value = 1\n", encoding="utf-8")
            bad.write_text("value = 2\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True, env=env)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True, env=env)

            # Simulate stale invalid residue from a previous failed slice.
            bad.write_text("{\n", encoding="utf-8")
            # Simulate the current slice touching only good.py.
            good.write_text("value = 3\n", encoding="utf-8")

            artifacts_dir = Path(td) / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifacts, info = bot._finalize_local_implementer_change(
                artifacts_dir=artifacts_dir,
                worktree_dir=repo,
                artifacts=[],
                apply_mode="rewrite",
                rewrite_files=["good.py"],
                changed_files_hint=["good.py"],
            )

            self.assertTrue(artifacts)
            self.assertEqual(info["changed_files"], ["good.py"])
            self.assertTrue(info["validation_ok"])

    def test_augment_implementer_failure_summary_includes_validation_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            (artifacts_dir / "local_ollama_validation.txt").write_text(
                "$ python3 -m py_compile tools/example.py\n"
                "File \"tools/example.py\", line 9\n"
                "SyntaxError: invalid syntax\n",
                encoding="utf-8",
            )
            (artifacts_dir / "local_ollama_git_status.txt").write_text(
                " M tools/example.py\n M tools/other.py\n",
                encoding="utf-8",
            )
            summary = bot._augment_implementer_failure_summary(
                summary="Local Ollama execution failed: implementer_local change touched Python files but py_compile failed",
                job_id="job-test",
                artifacts_dir=str(artifacts_dir),
            )
            self.assertIn("FAILED_VALIDATION_OUTPUT", summary)
            self.assertIn("tools/example.py", summary)
            self.assertIn("FAILED_CHANGED_FILES", summary)

    def test_workspace_context_large_exact_target_includes_symbol_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            worktree = Path(td)
            filler = 'x = "' + ("a" * 70) + '"'
            lines = [f"{filler}  # {i}" for i in range(1300)]
            lines.extend(
                [
                    "",
                    "def _classify_local_slice_failure(role_norm: str) -> str:",
                    "    return role_norm",
                    "",
                ]
            )
            (worktree / "bot.py").write_text("\n".join(lines), encoding="utf-8")
            task = SimpleNamespace(
                input_text="Please patch bot.py around _classify_local_slice_failure.",
                trace={},
            )
            prompt = bot._augment_local_specialist_prompt_with_workspace_context(
                task=task,
                user_prompt="Modify bot.py for _classify_local_slice_failure reliability.",
                worktree_dir=worktree,
                role="implementer_local",
            )
            self.assertIn("FILE: bot.py", prompt)
            self.assertIn("def _classify_local_slice_failure", prompt)
            self.assertIn("# [FOCUSED_EXACT_TARGET_SYMBOL_CONTEXT]", prompt)

    def test_workspace_context_large_exact_target_symbol_not_found_uses_truncated_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            worktree = Path(td)
            filler = 'x = "' + ("b" * 70) + '"'
            lines = [f"{filler}  # {i}" for i in range(1300)]
            lines.extend(
                [
                    "",
                    "def _classify_local_slice_failure(role_norm: str) -> str:",
                    "    return role_norm",
                    "",
                ]
            )
            (worktree / "bot.py").write_text("\n".join(lines), encoding="utf-8")
            task = SimpleNamespace(
                input_text="Please patch bot.py around _symbol_not_present_in_file.",
                trace={},
            )
            prompt = bot._augment_local_specialist_prompt_with_workspace_context(
                task=task,
                user_prompt="Modify bot.py for _symbol_not_present_in_file reliability.",
                worktree_dir=worktree,
                role="implementer_local",
            )
            self.assertIn("FILE: bot.py", prompt)
            self.assertIn("# [TRUNCATED_EXACT_TARGET_CONTEXT]", prompt)
            self.assertNotIn("def _classify_local_slice_failure", prompt)

    def test_workspace_context_large_ranked_python_candidate_includes_symbol_excerpt_without_path_hint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            worktree = Path(td)
            subprocess.run(["git", "init"], cwd=worktree, check=True, capture_output=True, text=True)
            filler = 'x = "' + ("c" * 70) + '"'
            lines = [f"{filler}  # {i}" for i in range(1300)]
            lines.extend(
                [
                    "",
                    "def _classify_local_slice_failure(role_norm: str) -> str:",
                    "    return role_norm",
                    "",
                ]
            )
            (worktree / "bot.py").write_text("\n".join(lines), encoding="utf-8")
            subprocess.run(["git", "add", "bot.py"], cwd=worktree, check=True, capture_output=True, text=True)
            task = SimpleNamespace(
                input_text="Please patch _classify_local_slice_failure reliability.",
                trace={},
            )
            prompt = bot._augment_local_specialist_prompt_with_workspace_context(
                task=task,
                user_prompt="Modify the local failure classifier to reduce implementer blockers.",
                worktree_dir=worktree,
                role="implementer_local",
            )
            self.assertIn("FILE: bot.py", prompt)
            self.assertIn("def _classify_local_slice_failure", prompt)
            self.assertIn("# [FOCUSED_EXACT_TARGET_SYMBOL_CONTEXT]", prompt)


if __name__ == "__main__":
    unittest.main()
