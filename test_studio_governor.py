import json
import os
import sqlite3
import subprocess
from types import SimpleNamespace

import bot
from orchestrator.schemas.task import Task
from orchestrator.storage import SQLiteTaskStorage


def _delivery_failure_memory(now: float) -> dict:
    return {
        "recent_studio_negative_outcomes": [
            {
                "key": "new-project-incubator",
                "repo_id": "",
                "type": "NEW_PROJECT",
                "status": "failed_root_caused",
                "summary": "delivery job claimed progress but left no mergeable branch or validated repo diff",
                "updated_at": now - 60,
            }
        ],
        "recent_studio_positive_outcomes": [],
        "studio_portfolio_recent_count_6h": 0,
        "studio_portfolio_recent_count_24h": 0,
    }


def test_studio_governor_forces_delivery_repair_after_ghost_delivery():
    now = 1_700_000_000.0
    governor = bot._studio_governor_assessment(_delivery_failure_memory(now), now=now)

    assert governor["mode"] == "repair_delivery_contract"
    assert governor["severity"] == "red"
    assert "new-project-incubator" in governor["avoid_keys"]
    assert governor["delivery_failure_count_72h"] == 1


def test_studio_governor_makes_core_repair_beat_incubator():
    now = 1_700_000_000.0
    memory = _delivery_failure_memory(now)
    memory["recent_studio_positive_outcomes"] = [
        {
            "key": "repo-codexbot",
            "repo_id": "codexbot",
            "type": "DEEP_IMPROVEMENT",
            "status": "merged",
            "summary": "Prior core reliability improvement shipped.",
            "updated_at": now - 900,
        },
        {
            "key": "repo-codexbot",
            "repo_id": "codexbot",
            "type": "DEEP_IMPROVEMENT",
            "status": "merged",
            "summary": "Another core reliability improvement shipped.",
            "updated_at": now - 1800,
        },
    ]
    memory["studio_governor"] = bot._studio_governor_assessment(memory, now=now)
    core_repo = {
        "repo_id": "codexbot",
        "path": "/home/aponce/codexbot",
        "priority": 1,
        "status": "active",
        "autonomy_enabled": True,
        "metadata": {},
    }
    dashboard_repo = {
        "repo_id": "executivedashboard",
        "path": "/home/aponce/ExecutiveDashboard",
        "priority": 1,
        "status": "active",
        "autonomy_enabled": True,
        "metadata": {},
    }

    core = bot._studio_opportunity_for_repo(core_repo, now=now, memory=memory)
    dashboard = bot._studio_opportunity_for_repo(dashboard_repo, now=now, memory=memory)
    incubator = bot._studio_incubator_opportunity(cfg=SimpleNamespace(), now=now, memory=memory)

    assert core["score"] >= 140
    assert core["score"] > dashboard["score"]
    assert core["score"] > incubator["score"]
    assert "delivery contract" in core["thesis"].lower()
    assert "temporarily gated" in incubator["thesis"].lower()


def test_studio_core_recent_win_cooldown_prefers_dashboard_without_delivery_failure():
    now = 1_700_000_000.0
    memory = {
        "recent_studio_negative_outcomes": [],
        "recent_studio_positive_outcomes": [
            {
                "key": "repo-codexbot",
                "repo_id": "codexbot",
                "type": "DEEP_IMPROVEMENT",
                "status": "merged",
                "summary": "Core selection controls shipped.",
                "updated_at": now - 900,
            },
            {
                "key": "repo-codexbot",
                "repo_id": "codexbot",
                "type": "DEEP_IMPROVEMENT",
                "status": "merged",
                "summary": "Core validation checks shipped.",
                "updated_at": now - 1800,
            },
        ],
        "studio_portfolio_recent_count_6h": 0,
        "studio_portfolio_recent_count_24h": 0,
    }
    memory["studio_governor"] = bot._studio_governor_assessment(memory, now=now)
    core_repo = {
        "repo_id": "codexbot",
        "path": "/home/aponce/codexbot",
        "priority": 1,
        "status": "active",
        "autonomy_enabled": True,
        "metadata": {},
    }
    dashboard_repo = {
        "repo_id": "executivedashboard",
        "path": "/home/aponce/ExecutiveDashboard",
        "priority": 1,
        "status": "active",
        "autonomy_enabled": True,
        "metadata": {},
    }

    core = bot._studio_opportunity_for_repo(core_repo, now=now, memory=memory)
    dashboard = bot._studio_opportunity_for_repo(dashboard_repo, now=now, memory=memory)
    core_text = " ".join(
        [
            core["risk_summary"],
            core["why_better_than_alternatives"],
            " ".join(core["recent_studio_saturation"]),
        ]
    ).lower()

    assert dashboard["score"] > core["score"]
    assert "codexbot core cooldown" in core_text
    assert "avoid another core deep_improvement" in core_text


def test_studio_governor_breaks_repeated_core_repair_loop():
    now = 1_700_000_000.0
    memory = _delivery_failure_memory(now)
    memory["recent_studio_negative_outcomes"].extend(
        [
            {
                "key": "repo-codexbot",
                "repo_id": "codexbot",
                "type": "DEEP_IMPROVEMENT",
                "status": "failed_root_caused",
                "summary": "Prior core repair failed.",
                "updated_at": now - 120,
            },
            {
                "key": "repo-codexbot",
                "repo_id": "codexbot",
                "type": "DEEP_IMPROVEMENT",
                "status": "rejected_low_value",
                "summary": "Prior core repair had no delta.",
                "updated_at": now - 240,
            },
        ]
    )
    memory["studio_governor"] = bot._studio_governor_assessment(memory, now=now)
    assert memory["studio_governor"]["mode"] == "repair_loop_breaker"
    assert "repo-codexbot" in memory["studio_governor"]["avoid_keys"]

    core_repo = {
        "repo_id": "codexbot",
        "path": "/home/aponce/codexbot",
        "priority": 1,
        "status": "active",
        "autonomy_enabled": True,
        "metadata": {},
    }
    dashboard_repo = {
        "repo_id": "executivedashboard",
        "path": "/home/aponce/ExecutiveDashboard",
        "priority": 1,
        "status": "active",
        "autonomy_enabled": True,
        "metadata": {},
    }

    core = bot._studio_opportunity_for_repo(core_repo, now=now, memory=memory)
    dashboard = bot._studio_opportunity_for_repo(dashboard_repo, now=now, memory=memory)

    assert core["score"] < dashboard["score"]
    assert "Hold this repeated PonceBot core repair" in core["thesis"]


def test_studio_governor_cools_down_after_non_core_ship():
    now = 1_700_000_000.0
    memory = _delivery_failure_memory(now)
    memory["recent_studio_negative_outcomes"].extend(
        [
            {
                "key": "repo-codexbot",
                "repo_id": "codexbot",
                "type": "DEEP_IMPROVEMENT",
                "status": "failed_root_caused",
                "summary": "Prior core repair failed.",
                "updated_at": now - 120,
            },
            {
                "key": "repo-codexbot",
                "repo_id": "codexbot",
                "type": "DEEP_IMPROVEMENT",
                "status": "rejected_low_value",
                "summary": "Prior core repair had no delta.",
                "updated_at": now - 240,
            },
        ]
    )
    memory["recent_studio_positive_outcomes"] = [
        {
            "key": "new-project-incubator",
            "repo_id": "",
            "type": "NEW_PROJECT",
            "status": "published_project",
            "summary": "Published a private project with README and validation.",
            "updated_at": now - 60,
        }
    ]

    governor = bot._studio_governor_assessment(memory, now=now)

    assert governor["mode"] == "repair_loop_cooldown"
    assert governor["severity"] == "yellow"
    assert "new-project-incubator" in governor["avoid_keys"]


def test_studio_prompt_includes_governor_directives():
    now = 1_700_000_000.0
    memory = _delivery_failure_memory(now)
    memory["studio_governor"] = bot._studio_governor_assessment(memory, now=now)
    selected = {
        "type": "DEEP_IMPROVEMENT",
        "repo_name": "codexbot",
        "score": 120,
        "thesis": "Repair delivery contract.",
        "operator_visible_outcome": "No ghost deliveries.",
        "business_model": "Factory leverage.",
        "monetization_path": "Reduce wasted cycles.",
        "commercial_evidence_target": "Validated recovery.",
    }

    prompt = bot._studio_cycle_prompt_packet(selected=selected, opportunities=[selected], memory=memory)

    assert "Studio governor:" in prompt
    assert "repair_delivery_contract" in prompt
    assert "Do not start another new-folder project" in prompt


def test_studio_governor_preempts_active_incubator_cycle_only_in_repair_mode():
    cycle = {
        "selected_key": "new-project-incubator",
        "selected_type": "NEW_PROJECT",
        "selected_lane": "incubator",
    }
    normal_governor = {"mode": "normal", "avoid_keys": []}
    repair_governor = {"mode": "repair_delivery_contract", "avoid_keys": ["new-project-incubator"]}
    loop_governor = {"mode": "repair_loop_breaker", "avoid_keys": ["repo-codexbot"]}
    cooldown_governor = {"mode": "repair_loop_cooldown", "avoid_keys": ["new-project-incubator"]}
    quality_governor = {"mode": "incubator_quality_gate", "avoid_keys": ["new-project-incubator"]}
    loop_cycle = {
        "selected_key": "repo-codexbot",
        "selected_repo_id": "codexbot",
        "selected_type": "DEEP_IMPROVEMENT",
        "selected_lane": "core",
    }
    dashboard_cycle = {
        "selected_key": "repo-executivedashboard",
        "selected_repo_id": "executivedashboard",
        "selected_type": "FEATURE",
        "selected_lane": "dashboard",
    }

    assert not bot._studio_governor_should_preempt_active_cycle(normal_governor, cycle)
    assert bot._studio_governor_should_preempt_active_cycle(repair_governor, cycle)
    assert bot._studio_governor_should_preempt_active_cycle(loop_governor, loop_cycle)
    assert not bot._studio_governor_should_preempt_active_cycle(loop_governor, cycle)
    assert bot._studio_governor_should_preempt_active_cycle(cooldown_governor, cycle)
    assert bot._studio_governor_should_preempt_active_cycle(cooldown_governor, dashboard_cycle)
    assert bot._studio_governor_should_preempt_active_cycle(quality_governor, cycle)


def test_workspace_lease_reaps_terminal_job_holder(tmp_path):
    storage = SQLiteTaskStorage(tmp_path / "jobs.sqlite")
    stale = Task.new(
        job_id="stale-job",
        source="test",
        role="skynet",
        input_text="old",
        request_type="exec",
        priority=1,
        model="gpt-5.5",
        effort="high",
        mode_hint="full",
        requires_approval=False,
        max_cost_window_usd=0,
        chat_id=1,
        state="done",
    )
    fresh = Task.new(
        job_id="fresh-job",
        source="test",
        role="skynet",
        input_text="new",
        request_type="exec",
        priority=1,
        model="gpt-5.5",
        effort="high",
        mode_hint="full",
        requires_approval=False,
        max_cost_window_usd=0,
        chat_id=1,
        state="queued",
    )
    storage.submit_task(stale)
    storage.submit_task(fresh)
    with storage._conn() as conn:
        conn.execute(
            "INSERT INTO workspace_leases(role, slot, job_id, leased_at) VALUES (?, ?, ?, ?)",
            ("skynet", 1, stale.job_id, 1.0),
        )
        conn.commit()

    assert storage.lease_workspace(role="skynet", job_id=fresh.job_id, slots=1) == 1
    assert storage.get_workspace_lease(job_id=fresh.job_id) == ("skynet", 1)


def test_controller_snapshot_delivery_candidate_requires_validated_release_blocker(tmp_path):
    snapshot = tmp_path / "artifacts" / "order" / "controller_snapshot"
    snapshot.mkdir(parents=True)
    patch = snapshot.parent / "changes.patch"
    patch.write_text("diff --git a/static/app.js b/static/app.js\n", encoding="utf-8")

    candidate = bot._controller_snapshot_delivery_candidate(
        {
            "controller_snapshot_workdir": str(snapshot),
            "result_artifacts": [str(patch)],
            "merge_cancelled": True,
            "result_summary": "PASS implementation/review. Outcome: blocked_need_operator. controller-snapshot has no origin.",
            "result_next_action": "Apply the validated changes into the real repository checkout, commit, push, and deploy.",
        }
    )

    assert candidate["snapshot_dir"] == str(snapshot)
    assert candidate["patch_path"] == str(patch)
    assert not bot._controller_snapshot_delivery_candidate(
        {
            "controller_snapshot_workdir": str(snapshot),
            "result_artifacts": [str(patch)],
            "result_summary": "Blocked in controller snapshot without validation evidence.",
        }
    )


def test_controller_snapshot_delivery_candidate_accepts_published_project_summary(tmp_path):
    snapshot = tmp_path / "artifacts" / "order" / "controller_snapshot"
    snapshot.mkdir(parents=True)
    patch = snapshot.parent / "changes.patch"
    patch.write_text("diff --git a/README.md b/README.md\n", encoding="utf-8")

    candidate = bot._controller_snapshot_delivery_candidate(
        {
            "controller_snapshot_workdir": str(snapshot),
            "result_artifacts": [str(patch)],
            "merge_cancelled": True,
            "result_status": "done",
            "result_summary": "PASS. Outcome: `published_project`. Private GitHub repo exists on main with validation evidence.",
            "result_next_action": None,
        }
    )

    assert candidate["snapshot_dir"] == str(snapshot)
    assert candidate["patch_path"] == str(patch)


def test_recent_controller_snapshot_rows_use_studio_updated_at_for_recovery(tmp_path):
    db = tmp_path / "jobs.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE studio_cycles (order_id TEXT, status TEXT, outcome_status TEXT, updated_at REAL)"
        )
        conn.execute(
            "CREATE TABLE ceo_orders (order_id TEXT, chat_id INTEGER, status TEXT, phase TEXT, title TEXT, updated_at REAL)"
        )
        conn.execute("CREATE TABLE jobs (job_id TEXT, trace TEXT)")
        conn.execute(
            "INSERT INTO studio_cycles VALUES (?, ?, ?, ?)",
            ("order-1", "active", "blocked_need_operator", 2_000.0),
        )
        conn.execute(
            "INSERT INTO ceo_orders VALUES (?, ?, ?, ?, ?, ?)",
            ("order-1", 123, "done", "done", "Snapshot order", 100.0),
        )
        conn.execute(
            "INSERT INTO jobs VALUES (?, ?)",
            ("order-1", json.dumps({"controller_snapshot_workdir": "/tmp/snapshot"})),
        )

    orch_q = SimpleNamespace(_storage=SimpleNamespace(path=db))
    rows = bot._recent_controller_snapshot_studio_order_rows(
        orch_q=orch_q,
        now=2_100.0,
        max_age_seconds=300.0,
    )

    assert len(rows) == 1
    assert rows[0]["updated_at"] == 2_000.0


def test_autonomous_commit_identity_ignores_stale_repo_config(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    monkeypatch.delenv("BOT_AUTONOMOUS_GIT_AUTHOR_NAME", raising=False)
    monkeypatch.delenv("BOT_AUTONOMOUS_GIT_AUTHOR_EMAIL", raising=False)

    env = bot._autonomous_commit_identity_env(repo)

    assert env["GIT_AUTHOR_NAME"] == "manolosake"
    assert env["GIT_AUTHOR_EMAIL"] == "manolosake@gmail.com"
    assert env["GIT_COMMITTER_NAME"] == "manolosake"
    assert env["GIT_COMMITTER_EMAIL"] == "manolosake@gmail.com"


def test_autonomous_validation_env_finds_user_jdk(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    java_bin = home / ".local" / "jdks" / "jdk-17" / "bin" / "java"
    java_bin.parent.mkdir(parents=True)
    java_bin.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    java_bin.chmod(0o755)
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.delenv("BOT_AUTONOMOUS_GIT_AUTHOR_NAME", raising=False)
    monkeypatch.delenv("BOT_AUTONOMOUS_GIT_AUTHOR_EMAIL", raising=False)

    env = bot._autonomous_commit_identity_env(repo)

    assert env["JAVA_HOME"] == str(home / ".local" / "jdks" / "jdk-17")
    assert env["PATH"].split(os.pathsep)[0] == str(java_bin.parent)
    assert env["GIT_AUTHOR_NAME"] == "manolosake"


def test_studio_readiness_uses_user_jdk_for_android_repo(tmp_path, monkeypatch):
    home = tmp_path / "home"
    java_bin = home / ".local" / "jdks" / "jdk-17" / "bin" / "java"
    java_bin.parent.mkdir(parents=True)
    java_bin.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    java_bin.chmod(0o755)
    android = tmp_path / "OmniCrewApp.android"
    android.mkdir()
    (android / "gradlew").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(tmp_path / "missing-bin"))
    monkeypatch.delenv("JAVA_HOME", raising=False)

    readiness = bot._studio_operational_readiness(
        cfg=SimpleNamespace(),
        repos=[
            {
                "repo_id": "omnicrewapp.android",
                "path": str(android),
                "priority": 1,
                "status": "active",
                "autonomy_enabled": True,
                "metadata": {},
            }
        ],
        now=1_700_000_000.0,
    )

    assert readiness["status"] == "green"
    assert readiness["repo_checks"][0]["stacks"] == ["android"]
    assert readiness["repo_checks"][0]["missing_tools"] == []


def test_studio_governor_blocks_missing_validation_toolchain():
    now = 1_700_000_000.0
    memory = {
        "recent_studio_negative_outcomes": [],
        "recent_studio_positive_outcomes": [],
        "studio_portfolio_recent_count_6h": 0,
        "studio_portfolio_recent_count_24h": 0,
        "studio_readiness": {
            "status": "red",
            "repo_gaps": [
                {
                    "repo_id": "omnicrewapp.android",
                    "repo_name": "OmniCrewApp.android",
                    "stacks": ["android"],
                    "missing_tools": ["java"],
                }
            ],
        },
    }

    governor = bot._studio_governor_assessment(memory, now=now)

    assert governor["mode"] == "toolchain_readiness_gate"
    assert governor["severity"] == "red"
    assert "repo-omnicrewapp.android" in governor["avoid_keys"]
    assert governor["readiness_gap_count"] == 1


def test_controller_snapshot_untracked_filter_keeps_source_not_evidence():
    assert bot._controller_snapshot_safe_untracked_path("test_agents_sprint_brief_shell.py")
    assert bot._controller_snapshot_safe_untracked_path("static/app.js")
    assert bot._controller_snapshot_safe_untracked_path("samples/sample_requests.csv")
    assert not bot._controller_snapshot_safe_untracked_path("output/playwright/screenshot.png")
    assert not bot._controller_snapshot_safe_untracked_path(".codexbot_tmp/playwright/prefs.js")
    assert not bot._controller_snapshot_safe_untracked_path("../outside.py")


def test_controller_snapshot_validation_commands_include_android_build(tmp_path):
    repo = tmp_path / "android"
    repo.mkdir()
    (repo / "gradlew").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "validate_unit_tests.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    commands = bot._controller_snapshot_validation_commands(
        repo,
        ["app/src/main/java/com/example/MainActivity.kt", "app/src/test/java/com/example/UiLogicTest.kt"],
    )

    assert ["bash", "scripts/validate_unit_tests.sh"] in commands
    assert ["bash", "./gradlew", ":app:assembleDebug"] in commands


def test_controller_snapshot_validation_commands_include_node_workflow(tmp_path):
    repo = tmp_path / "node"
    repo.mkdir()
    (repo / "package.json").write_text('{"scripts":{"test":"node --test","validate":"node scripts/demo.mjs"}}\n', encoding="utf-8")
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "demo.mjs").write_text("console.log('demo')\n", encoding="utf-8")

    commands = bot._controller_snapshot_validation_commands(
        repo,
        ["src/app.js", "scripts/demo.mjs", "src/styles.css"],
    )

    assert ["node", "--check", "src/app.js"] in commands
    assert ["node", "--check", "scripts/demo.mjs"] in commands
    assert ["npm", "test"] in commands
    assert ["npm", "run", "validate"] in commands
    assert ["node", "scripts/demo.mjs"] in commands


def test_controller_snapshot_revert_applied_delta_restores_only_snapshot_delta(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "app.js").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.js"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    (repo / "app.js").write_text("base\nchange\n", encoding="utf-8")
    patch = tmp_path / "changes.patch"
    patch.write_text(subprocess.run(["git", "diff", "HEAD", "--", "."], cwd=repo, check=True, text=True, capture_output=True).stdout, encoding="utf-8")
    subprocess.run(["git", "checkout", "--", "app.js"], cwd=repo, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=repo, check=True)
    (repo / "test_snapshot.py").write_text("print('ok')\n", encoding="utf-8")

    result = bot._controller_snapshot_revert_applied_delta(
        repo_dir=repo,
        patch_path=patch,
        copied_untracked=["test_snapshot.py"],
    )

    assert result["reverse_patch_returncode"] == 0
    assert not result["errors"]
    assert (repo / "app.js").read_text(encoding="utf-8") == "base\n"
    assert not (repo / "test_snapshot.py").exists()


def test_studio_repo_kind_does_not_treat_poncebot_named_portfolio_as_core():
    assert bot._studio_repo_kind({"path": "/home/aponce/codexbot", "repo_id": "codexbot"}) == "Core"
    assert bot._studio_repo_kind({"path": "/home/aponce/poncebot-control-room", "repo_id": "poncebot-control-room"}) == "Portfolio"


def test_studio_finalize_orphaned_active_cycles_keeps_live_work(tmp_path):
    now = 1_700_000_000.0
    db = tmp_path / "jobs.sqlite"
    bot._studio_ensure_schema(db)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE jobs(job_id TEXT PRIMARY KEY, parent_job_id TEXT, state TEXT, trace TEXT)")
        conn.execute("CREATE TABLE ceo_orders(order_id TEXT PRIMARY KEY, status TEXT)")
        conn.executemany(
            """
            INSERT INTO studio_cycles(
                cycle_id, version, ts, status, selected_key, selected_type, selected_repo_id,
                selected_repo_path, selected_lane, thesis, rationale, debate_summary,
                operator_visible_outcome, evidence_target, risk_summary, prompt_packet,
                opportunities_json, order_id, created_at, updated_at
            ) VALUES (?, 1, ?, 'active', ?, 'DEEP_IMPROVEMENT', ?, '/home/aponce/codexbot',
                'core', 'Repair delivery.', 'Because ghost delivery.', 'Critic agrees.',
                'No ghost deliveries.', 'Tests.', 'Low.', 'packet', '[]', ?, ?, ?)
            """,
            [
                ("orphan-cycle", now - 900, "repo-codexbot", "codexbot", "orphan-order", now - 900, now - 900),
                ("live-order-cycle", now - 900, "repo-codexbot", "codexbot", "live-order", now - 900, now - 900),
                ("live-job-cycle", now - 900, "repo-codexbot", "codexbot", "live-job", now - 900, now - 900),
                (
                    "snapshot-cycle",
                    now - 900,
                    "repo-codexbot",
                    "codexbot",
                    "snapshot-order",
                    now - 900,
                    now - 900,
                ),
            ],
        )
        conn.execute("INSERT INTO ceo_orders(order_id, status) VALUES (?, ?)", ("live-order", "active"))
        conn.execute("INSERT INTO jobs(job_id, parent_job_id, state, trace) VALUES (?, ?, ?, ?)", ("live-job", "", "running", "{}"))
        conn.execute(
            "INSERT INTO jobs(job_id, parent_job_id, state, trace) VALUES (?, ?, ?, ?)",
            (
                "snapshot-order",
                "",
                "done",
                json.dumps(
                    {
                        "controller_snapshot_workdir": "/tmp/controller_snapshot",
                        "result_summary": "Outcome: blocked_need_operator. PASS validated controller snapshot.",
                        "result_next_action": "Run the validated patch in a write-enabled source checkout.",
                    }
                ),
            ),
        )
        conn.commit()

    closed = bot._studio_finalize_orphaned_active_cycles(db, now=now, max_age_seconds=300)

    assert closed == 2
    with sqlite3.connect(db) as conn:
        rows = {
            row[0]: row[1:]
            for row in conn.execute(
                "SELECT cycle_id, status, outcome_status FROM studio_cycles ORDER BY cycle_id"
            )
        }
    assert rows["orphan-cycle"] == ("failed", "failed_root_caused")
    assert rows["live-order-cycle"] == ("active", None)
    assert rows["live-job-cycle"] == ("active", None)
    assert rows["snapshot-cycle"] == ("active", "blocked_need_operator")


def test_final_sweep_no_change_close_waits_for_open_nonblocking_children():
    children = [
        SimpleNamespace(role="qa", state="running"),
        SimpleNamespace(role="skynet", state="running"),
    ]
    open_states = {"queued", "waiting_deps", "blocked_approval", "running", "blocked"}

    assert bot._final_sweep_blocker_count(children=children, open_states=open_states) == 0
    assert bot._has_open_child_jobs(children=children, open_states=open_states)


def test_order_branch_has_merge_delta_rejects_no_delta_branch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "branch", "feature/noop"], cwd=repo, check=True)

    has_delta, reason = bot._order_branch_has_merge_delta(
        repo=repo,
        order_branch="feature/noop",
        default_branch="main",
    )

    assert not has_delta
    assert reason == "branch_has_no_delta_vs_main"


def test_order_branch_has_merge_delta_accepts_real_branch_delta(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "feature/delta"], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("base\nchange\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "change"], cwd=repo, check=True, capture_output=True)

    has_delta, reason = bot._order_branch_has_merge_delta(
        repo=repo,
        order_branch="feature/delta",
        default_branch="main",
    )

    assert has_delta
    assert reason == "has_delta"
