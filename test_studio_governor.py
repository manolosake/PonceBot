import sqlite3
import subprocess
from types import SimpleNamespace

import bot


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
    memory["studio_governor"] = bot._studio_governor_assessment(memory, now=now)
    core_repo = {
        "repo_id": "codexbot",
        "path": "/home/aponce/codexbot",
        "priority": 1,
        "status": "active",
        "autonomy_enabled": True,
        "metadata": {},
    }

    core = bot._studio_opportunity_for_repo(core_repo, now=now, memory=memory)
    incubator = bot._studio_incubator_opportunity(cfg=SimpleNamespace(), now=now, memory=memory)

    assert core["score"] > incubator["score"]
    assert "delivery contract" in core["thesis"].lower()
    assert "temporarily gated" in incubator["thesis"].lower()


def test_studio_governor_keeps_core_repair_high_despite_prior_core_failures():
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
    core_repo = {
        "repo_id": "codexbot",
        "path": "/home/aponce/codexbot",
        "priority": 1,
        "status": "active",
        "autonomy_enabled": True,
        "metadata": {},
    }

    core = bot._studio_opportunity_for_repo(core_repo, now=now, memory=memory)

    assert core["score"] >= 140


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

    assert not bot._studio_governor_should_preempt_active_cycle(normal_governor, cycle)
    assert bot._studio_governor_should_preempt_active_cycle(repair_governor, cycle)


def test_studio_repo_kind_does_not_treat_poncebot_named_portfolio_as_core():
    assert bot._studio_repo_kind({"path": "/home/aponce/codexbot", "repo_id": "codexbot"}) == "Core"
    assert bot._studio_repo_kind({"path": "/home/aponce/poncebot-control-room", "repo_id": "poncebot-control-room"}) == "Portfolio"


def test_studio_finalize_orphaned_active_cycles_keeps_live_work(tmp_path):
    now = 1_700_000_000.0
    db = tmp_path / "jobs.sqlite"
    bot._studio_ensure_schema(db)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE jobs(job_id TEXT PRIMARY KEY, parent_job_id TEXT, state TEXT)")
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
            ],
        )
        conn.execute("INSERT INTO ceo_orders(order_id, status) VALUES (?, ?)", ("live-order", "active"))
        conn.execute("INSERT INTO jobs(job_id, parent_job_id, state) VALUES (?, ?, ?)", ("live-job", "", "running"))
        conn.commit()

    closed = bot._studio_finalize_orphaned_active_cycles(db, now=now, max_age_seconds=300)

    assert closed == 1
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
