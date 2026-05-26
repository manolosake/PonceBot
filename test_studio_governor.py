import json
import inspect
import os
import socket
import sqlite3
import subprocess
import urllib.error
import unittest
from pathlib import Path
from types import SimpleNamespace

import bot
from orchestrator.queue import OrchestratorQueue
from orchestrator.schemas.task import Task
from orchestrator.storage import SQLiteTaskStorage


def test_repo_deploy_policy_uses_static_product_page_for_readme_repo(tmp_path):
    codexbot = tmp_path / "codexbot"
    monitor = codexbot / "tools" / "main_deploy_monitor.py"
    monitor.parent.mkdir(parents=True)
    monitor.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    repo = tmp_path / "20260511-studio-cycle-new-product-incubator-0f5a5cc0"
    repo.mkdir()
    (repo / "README.md").write_text("# QuoteKit Studio\n\nA sellable workflow.", encoding="utf-8")

    policy = bot._repo_deploy_policy(
        cfg=SimpleNamespace(codex_workdir=codexbot),
        repo_record={"metadata": {"repo_name": "20260511-studio-cycle-new-product-incubator-0f5a5cc0"}},
        repo_dir=repo,
    )

    assert policy["source"] == "factory_static_product_default"
    assert policy["command"] == ["systemctl", "--user", "start", "codexbot-main-deploy-monitor.service"]
    assert policy["url"] == "http://127.0.0.1:8890/quotekit-studio/"
    assert policy["verify_command"][:2] == ["bash", "-lc"]
    assert "Static product page published" in policy["success_summary"]


def test_studio_repo_display_name_prefers_remote_name_for_generic_incubator(monkeypatch, tmp_path):
    repo = tmp_path / "20260511-studio-cycle-new-product-incubator-0f5a5cc0"
    repo.mkdir()

    monkeypatch.setattr(bot, "_studio_repo_origin_hint", lambda _repo: "git@github.com:manolosake/quotekit-studio.git")

    assert bot._studio_repo_display_name({"path": str(repo), "metadata": {}}) == "quotekit-studio"


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


def _autoship_recovery_memory(now: float) -> dict:
    return {
        "recent_studio_negative_outcomes": [
            {
                "key": "repo-codexbot",
                "repo_id": "codexbot",
                "type": "DEEP_IMPROVEMENT",
                "status": "blocked_need_operator",
                "summary": "PASS validated controller snapshot waiting for autoship recovery.",
                "updated_at": now - 60,
            },
            {
                "key": "repo-codexbot",
                "repo_id": "codexbot",
                "type": "DEEP_IMPROVEMENT",
                "status": "blocked_need_operator",
                "summary": "Validated controller snapshot waiting for autoship; operator recovery needed.",
                "updated_at": now - (2 * 3600),
            },
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


def test_studio_governor_detects_autoship_recovery_backlog():
    now = 1_700_000_000.0
    governor = bot._studio_governor_assessment(_autoship_recovery_memory(now), now=now)

    assert governor["mode"] == "autoship_recovery_gate"
    assert governor["severity"] == "red"
    assert governor["autoship_recovery_block_count_72h"] == 2
    assert governor["autoship_recovery_block_count_24h"] == 2
    assert "validated controller-snapshot autoship/recovery blockers" in governor["trigger"]
    assert "Recover, merge, push, deploy, or root-cause" in governor["force_next_action"]
    assert any("validated controller snapshots waiting" in directive for directive in governor["directives"])


def test_transient_network_exception_detection_handles_dns_cause():
    try:
        try:
            raise urllib.error.URLError(socket.gaierror(-3, "Temporary failure in name resolution"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Telegram URL error calling getUpdates: {exc}") from exc
    except RuntimeError as exc:
        assert bot._is_transient_network_exception(exc)

    assert not bot._is_transient_network_exception(RuntimeError("database schema mismatch"))


def test_studio_governor_does_not_treat_generic_autoship_waiting_as_recovery_backlog():
    now = 1_700_000_000.0
    memory = {
        "recent_studio_negative_outcomes": [
            {
                "key": "repo-codexbot",
                "repo_id": "codexbot",
                "type": "DEEP_IMPROVEMENT",
                "status": "blocked_need_operator",
                "summary": "waiting for autoship",
                "updated_at": now - 60,
            },
            {
                "key": "repo-codexbot",
                "repo_id": "codexbot",
                "type": "DEEP_IMPROVEMENT",
                "status": "blocked_need_operator",
                "summary": "still waiting for autoship",
                "updated_at": now - 120,
            },
        ],
        "recent_studio_positive_outcomes": [],
        "studio_portfolio_recent_count_6h": 0,
        "studio_portfolio_recent_count_24h": 0,
    }

    governor = bot._studio_governor_assessment(memory, now=now)

    assert governor["mode"] == "normal"
    assert governor["autoship_recovery_block_count_72h"] == 0


def test_studio_governor_requires_multiple_fresh_blocked_autoship_recovery_signals():
    now = 1_700_000_000.0
    base = _autoship_recovery_memory(now)

    one_blocker = {
        **base,
        "recent_studio_negative_outcomes": base["recent_studio_negative_outcomes"][:1],
    }
    stale_blockers = {
        **base,
        "recent_studio_negative_outcomes": [
            {**entry, "updated_at": now - (73 * 3600)}
            for entry in base["recent_studio_negative_outcomes"]
        ],
    }
    wrong_status = {
        **base,
        "recent_studio_negative_outcomes": [
            {**entry, "status": "failed_root_caused"}
            for entry in base["recent_studio_negative_outcomes"]
        ],
    }

    assert bot._studio_governor_assessment(one_blocker, now=now)["mode"] == "normal"
    assert bot._studio_governor_assessment(stale_blockers, now=now)["mode"] == "normal"
    assert bot._studio_governor_assessment(wrong_status, now=now)["mode"] == "normal"


def test_studio_governor_autoship_gate_makes_codexbot_beat_portfolio_and_incubator():
    now = 1_700_000_000.0
    memory = _autoship_recovery_memory(now)
    memory["studio_governor"] = bot._studio_governor_assessment(memory, now=now)
    core_repo = {
        "repo_id": "codexbot",
        "path": "/home/aponce/codexbot",
        "priority": 1,
        "status": "active",
        "autonomy_enabled": True,
        "metadata": {},
    }
    portfolio_repo = {
        "repo_id": "portfolio-tool",
        "path": "/home/aponce/portfolio-tool",
        "priority": 1,
        "status": "active",
        "autonomy_enabled": True,
        "metadata": {},
    }

    core = bot._studio_opportunity_for_repo(core_repo, now=now, memory=memory)
    portfolio = bot._studio_opportunity_for_repo(portfolio_repo, now=now, memory=memory)
    incubator = bot._studio_incubator_opportunity(cfg=SimpleNamespace(), now=now, memory=memory)

    assert core["score"] >= 150
    assert core["score"] > portfolio["score"]
    assert core["score"] > incubator["score"]
    assert "Recover the autoship backlog" in core["thesis"]
    assert "autoship recovery gate is active" in incubator["risk_summary"]


def test_studio_prompt_includes_autoship_recovery_governor_force_next_action():
    now = 1_700_000_000.0
    memory = _autoship_recovery_memory(now)
    memory["studio_governor"] = bot._studio_governor_assessment(memory, now=now)
    selected = {
        "type": "DEEP_IMPROVEMENT",
        "repo_name": "codexbot",
        "score": 150,
        "thesis": "Recover the autoship backlog.",
        "operator_visible_outcome": "Validated snapshots ship or receive exact blockers.",
        "business_model": "Factory leverage.",
        "monetization_path": "Turn validated work into shipped outcomes.",
        "commercial_evidence_target": "Merged, pushed, deployed, or root-caused snapshots.",
    }

    prompt = bot._studio_cycle_prompt_packet(selected=selected, opportunities=[selected], memory=memory)

    assert "autoship_recovery_gate" in prompt
    assert "Recover, merge, push, deploy, or root-cause" in prompt
    assert "validated controller snapshots waiting for autoship/recovery" in prompt


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


def test_studio_governor_exposes_incubator_quality_debt_and_prompt_line():
    now = 1_700_000_000.0
    memory = {
        "recent_studio_negative_outcomes": [
            {
                "key": "new-project-incubator",
                "repo_id": "",
                "type": "NEW_PROJECT",
                "status": "rejected_low_value",
                "summary": f"New project failed buyer/demo quality gate {idx}.",
                "updated_at": now - (idx * 60),
            }
            for idx in range(1, 12)
        ],
        "recent_studio_positive_outcomes": [
            {
                "key": "new-project-incubator",
                "repo_id": "",
                "type": "NEW_PROJECT",
                "status": "published_project",
                "summary": "One project published with validation evidence.",
                "updated_at": now - 120,
            }
        ],
        "studio_portfolio_recent_count_6h": 0,
        "studio_portfolio_recent_count_24h": 0,
    }

    governor = bot._studio_governor_assessment(memory, now=now)

    assert governor["mode"] == "incubator_quality_gate"
    assert governor["new_project_negative_24h"] == 11
    assert governor["new_project_positive_24h"] == 1
    assert governor["new_project_cycles_24h"] == 12
    assert governor["new_project_quality_debt_24h"] == 10
    assert governor["new_project_failure_ratio_24h"] == 0.917
    assert "quality debt 10" in governor["trigger"]
    assert "failure ratio 92%" in governor["trigger"]
    assert "factory improvement" in governor["force_next_action"]
    assert "buyer/demo/validation/publication evidence" in governor["force_next_action"]
    assert any("quality debt" in directive for directive in governor["directives"])
    assert any("fresh projects need stronger proof" in directive for directive in governor["directives"])

    memory["studio_governor"] = governor
    selected = {
        "type": "DASHBOARD",
        "repo_name": "ExecutiveDashboard",
        "score": 101,
        "thesis": "Improve selection visibility.",
        "operator_visible_outcome": "Governor debt is visible to delegated specialists.",
        "business_model": "Factory leverage.",
        "monetization_path": "Reduce churn.",
        "commercial_evidence_target": "Prompt packet includes quality debt.",
    }

    prompt = bot._studio_cycle_prompt_packet(selected=selected, opportunities=[selected], memory=memory)

    assert "new-project quality: debt=10; failure_ratio=92%; negatives=11; positives=1; cycles=12" in prompt


def test_studio_governor_counts_incubator_like_product_workflow_debt():
    now = 1_700_000_000.0
    memory = {
        "recent_studio_negative_outcomes": [
            {
                "key": f"20260511-studio-cycle-new-product-incubator-{idx}",
                "repo_id": "",
                "type": "PRODUCT_WORKFLOW",
                "status": "rejected_low_value",
                "summary": f"Rejected product workflow incubator quality gate {idx}.",
                "updated_at": now - (idx * 60),
            }
            for idx in range(1, 3)
        ],
        "recent_studio_positive_outcomes": [],
        "studio_portfolio_recent_count_6h": 0,
        "studio_portfolio_recent_count_24h": 0,
    }

    governor = bot._studio_governor_assessment(memory, now=now)

    assert governor["mode"] == "incubator_quality_gate"
    assert governor["new_project_negative_24h"] == 2
    assert governor["new_project_cycles_24h"] == 2
    assert governor["new_project_quality_debt_24h"] == 2
    assert governor["new_project_failure_ratio_24h"] == 1.0


def test_studio_build_opportunities_excludes_governor_avoided_incubator(monkeypatch, tmp_path):
    now = 1_700_000_000.0
    repo_dir = tmp_path / "ExecutiveDashboard"
    repo_dir.mkdir()
    memory = {
        "recent_studio_negative_outcomes": [
            {
                "key": "new-project-incubator",
                "repo_id": "",
                "type": "NEW_PROJECT",
                "status": "rejected_low_value",
                "summary": f"Rejected new project without material delta {idx}.",
                "updated_at": now - (idx * 60),
            }
            for idx in range(1, 3)
        ],
        "recent_studio_positive_outcomes": [],
        "studio_portfolio_recent_count_6h": 0,
        "studio_portfolio_recent_count_24h": 0,
    }
    cfg = SimpleNamespace(orchestrator_db_path=tmp_path / "jobs.sqlite")
    repos = [
        {
            "repo_id": "executivedashboard-9ab2eb91",
            "path": str(repo_dir),
            "priority": 1,
            "status": "active",
            "autonomy_enabled": True,
            "metadata": {"kind": "Dashboard", "repo_name": "ExecutiveDashboard"},
        }
    ]

    monkeypatch.setattr(bot, "_project_incubator_enabled", lambda: True)
    monkeypatch.setattr(bot, "_project_incubator_due", lambda cfg, occupied_keys: True)

    opportunities = bot._studio_build_opportunities(
        cfg=cfg,
        orch_q=SimpleNamespace(),
        repos=repos,
        occupied_keys=set(),
        occupied_repo_ids=set(),
        now=now,
        memory=memory,
    )

    assert memory["studio_governor"]["mode"] == "incubator_quality_gate"
    assert all(item["key"] != "new-project-incubator" for item in opportunities)
    assert any(item["repo_id"] == "executivedashboard-9ab2eb91" for item in opportunities)


def test_studio_governor_focus_gate_blocks_fresh_repos_when_portfolio_is_too_broad():
    now = 1_700_000_000.0
    memory = {
        "recent_studio_negative_outcomes": [],
        "recent_studio_positive_outcomes": [],
        "studio_portfolio_recent_count_6h": 4,
        "studio_portfolio_recent_count_24h": 8,
        "studio_active_portfolio_repo_count": 54,
        "studio_active_portfolio_focus_cap": 18,
        "studio_unpublished_portfolio_repo_count": 13,
        "studio_unpublished_portfolio_focus_cap": 3,
        "studio_unpublished_portfolio_names": ["focus-forge", "runproof-board"],
    }

    governor = bot._studio_governor_assessment(memory, now=now)

    assert governor["mode"] == "portfolio_focus_gate"
    assert governor["severity"] == "yellow"
    assert "new-project-incubator" in governor["avoid_keys"]
    assert governor["active_portfolio_repo_count"] == 54
    assert governor["unpublished_portfolio_repo_count"] == 13
    assert "Do not create another fresh repo" in governor["force_next_action"]


def test_studio_build_opportunities_focus_gate_excludes_incubator(monkeypatch, tmp_path):
    now = 1_700_000_000.0
    dashboard_dir = tmp_path / "ExecutiveDashboard"
    dashboard_dir.mkdir()
    repos = [
        {
            "repo_id": "executivedashboard-9ab2eb91",
            "path": str(dashboard_dir),
            "priority": 1,
            "status": "active",
            "autonomy_enabled": True,
            "metadata": {"repo_name": "ExecutiveDashboard"},
        }
    ]
    for idx in range(20):
        repo_dir = tmp_path / f"portfolio-{idx}"
        repo_dir.mkdir()
        repos.append(
            {
                "repo_id": f"portfolio-{idx}",
                "path": str(repo_dir),
                "priority": 3,
                "status": "active",
                "autonomy_enabled": True,
                "metadata": {"repo_name": f"portfolio-{idx}"},
            }
        )
    memory = {
        "recent_studio_negative_outcomes": [],
        "recent_studio_positive_outcomes": [],
        "studio_portfolio_recent_count_6h": 0,
        "studio_portfolio_recent_count_24h": 0,
    }
    cfg = SimpleNamespace(orchestrator_db_path=tmp_path / "jobs.sqlite")

    monkeypatch.setattr(bot, "_project_incubator_enabled", lambda: True)
    monkeypatch.setattr(bot, "_project_incubator_due", lambda cfg, occupied_keys: True)

    opportunities = bot._studio_build_opportunities(
        cfg=cfg,
        orch_q=SimpleNamespace(),
        repos=repos,
        occupied_keys=set(),
        occupied_repo_ids=set(),
        now=now,
        memory=memory,
    )

    assert memory["studio_governor"]["mode"] == "portfolio_focus_gate"
    assert all(item["key"] != "new-project-incubator" for item in opportunities)
    assert any(item["lane"] in {"dashboard", "portfolio"} for item in opportunities)


def test_late_deploy_success_recovery_clears_deploy_failed_order(monkeypatch, tmp_path):
    state_path = tmp_path / "main_deploy_state.json"
    state_path.write_text(
        json.dumps(
            {
                "repos": {
                    "omnicrewapp.android-ba7b4a67": {
                        "status": "ok",
                        "deployed_head": "4c046e4ff4ed257c5da6887db18eae1af10ecc1d",
                        "last_deploy_at": 1_700_000_100.0,
                        "url": "http://127.0.0.1:8890/omnicrewapp-android/",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    completed = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Queue:
        _storage = SimpleNamespace(path=tmp_path / "jobs.sqlite")

        def __init__(self):
            self.status = None
            self.phase = None
            self.state_call = None
            self.audit_events = []

        def set_order_status(self, *args, **kwargs):
            self.status = (args, kwargs)

        def set_order_phase(self, *args, **kwargs):
            self.phase = (args, kwargs)

        def update_state(self, *args, **kwargs):
            self.state_call = (args, kwargs)

        def append_audit_event(self, *args, **kwargs):
            self.audit_events.append((args, kwargs))

    monkeypatch.setattr(bot.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(bot, "_studio_complete_cycle_for_order_from_queue", lambda **kwargs: completed.update(kwargs))
    queue = Queue()

    assert bot._recover_late_successful_deploy(
        orch_q=queue,
        order_id="order-1",
        chat_id=123,
        repo_record={"repo_id": "omnicrewapp.android-ba7b4a67"},
        root_trace={"merge_commit": "4c046e4", "deploy_status": "failed"},
        now=1_700_000_200.0,
    )

    assert queue.status[1]["status"] == "done"
    assert queue.phase[1]["phase"] == "done"
    assert queue.state_call[0][1] == "done"
    assert queue.state_call[1]["blocked_reason"] is None
    assert queue.state_call[1]["deploy_status"] == "ok"
    assert queue.state_call[1]["deploy_result"] == "deploy_ok_recovered"
    assert completed["outcome_status"] == "shipped_to_main"
    assert queue.audit_events[-1][1]["event_type"] == "order.late_deploy_success_recovered"


def test_sync_order_phase_recovers_late_deploy_before_failed_terminal(monkeypatch, tmp_path):
    recovered = []

    class Queue:
        def get_order(self, *_args, **_kwargs):
            return {
                "status": "active",
                "phase": "review",
                "title": "AUTONOMOUS PROACTIVE SPRINT",
                "body": "[proactive:studio-repo-omnicrewapp]",
            }

        def get_job(self, *_args, **_kwargs):
            return SimpleNamespace(
                trace={
                    "merged_to_main": False,
                    "deploy_status": "failed",
                    "merge_commit": "4c046e4",
                },
                state="blocked",
                blocked_reason="deploy_failed",
            )

    monkeypatch.setattr(
        bot,
        "_repo_context_for_order",
        lambda **_kwargs: ({"repo_id": "omnicrewapp.android-ba7b4a67"}, tmp_path, "main"),
    )
    monkeypatch.setattr(bot, "_order_trace_requires_merge", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(bot, "_trace_has_operator_verified_deployment", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(bot, "_trace_has_project_incubator_delivery_evidence", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(bot, "_order_has_verified_no_change_resolution", lambda **_kwargs: False)
    monkeypatch.setattr(
        bot,
        "_recover_late_successful_deploy",
        lambda **kwargs: recovered.append(kwargs) or True,
    )
    monkeypatch.setattr(
        bot,
        "_studio_terminal_outcome_for_order",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("terminal outcome should not win before recovery")),
    )

    bot._sync_order_phase_from_runtime(orch_q=Queue(), root_ticket="order-1", chat_id=123)

    assert recovered
    assert recovered[0]["order_id"] == "order-1"


def test_studio_governor_ignores_ordinary_product_workflow_debt():
    now = 1_700_000_000.0
    memory = {
        "recent_studio_negative_outcomes": [
            {
                "key": f"20260511-studio-cycle-product-workflow-{idx}",
                "repo_id": "executivedashboard",
                "type": "PRODUCT_WORKFLOW",
                "status": "rejected_low_value",
                "summary": f"Rejected ordinary workflow quality gate {idx}.",
                "updated_at": now - (idx * 60),
            }
            for idx in range(1, 3)
        ],
        "recent_studio_positive_outcomes": [],
        "studio_portfolio_recent_count_6h": 0,
        "studio_portfolio_recent_count_24h": 0,
    }

    governor = bot._studio_governor_assessment(memory, now=now)

    assert governor["mode"] == "normal"
    assert governor["new_project_negative_24h"] == 0
    assert governor["new_project_cycles_24h"] == 0
    assert governor["new_project_quality_debt_24h"] == 0
    assert governor["new_project_failure_ratio_24h"] is None


def test_studio_governor_preempts_active_incubator_cycle_only_in_repair_mode():
    cycle = {
        "selected_key": "new-project-incubator",
        "selected_type": "NEW_PROJECT",
        "selected_lane": "incubator",
    }
    normal_governor = {"mode": "normal", "avoid_keys": []}
    repair_governor = {"mode": "repair_delivery_contract", "avoid_keys": ["new-project-incubator"]}
    autoship_governor = {"mode": "autoship_recovery_gate", "avoid_keys": ["new-project-incubator"]}
    loop_governor = {"mode": "repair_loop_breaker", "avoid_keys": ["repo-codexbot"]}
    cooldown_governor = {"mode": "repair_loop_cooldown", "avoid_keys": ["new-project-incubator"]}
    quality_governor = {"mode": "incubator_quality_gate", "avoid_keys": ["new-project-incubator"]}
    loop_cycle = {
        "selected_key": "repo-codexbot",
        "selected_repo_id": "codexbot",
        "selected_type": "DEEP_IMPROVEMENT",
        "selected_lane": "core",
    }
    autoship_recovery_cycle = {
        "selected_key": "repo-codexbot",
        "selected_repo_id": "codexbot",
        "selected_type": "DEEP_IMPROVEMENT",
        "selected_lane": "core",
        "thesis": "Recover the autoship backlog by converting validated controller snapshots into shipped outcomes.",
        "operator_visible_outcome": "Validated controller snapshots are merged, pushed, deployed, or root-caused.",
    }
    dashboard_cycle = {
        "selected_key": "repo-executivedashboard",
        "selected_repo_id": "executivedashboard",
        "selected_type": "FEATURE",
        "selected_lane": "dashboard",
    }

    assert not bot._studio_governor_should_preempt_active_cycle(normal_governor, cycle)
    assert bot._studio_governor_should_preempt_active_cycle(repair_governor, cycle)
    assert bot._studio_governor_should_preempt_active_cycle(autoship_governor, cycle)
    assert bot._studio_governor_should_preempt_active_cycle(autoship_governor, loop_cycle)
    assert not bot._studio_governor_should_preempt_active_cycle(autoship_governor, autoship_recovery_cycle)
    assert bot._studio_governor_should_preempt_active_cycle(loop_governor, loop_cycle)
    assert not bot._studio_governor_should_preempt_active_cycle(loop_governor, cycle)
    assert bot._studio_governor_should_preempt_active_cycle(cooldown_governor, cycle)
    assert bot._studio_governor_should_preempt_active_cycle(cooldown_governor, dashboard_cycle)
    assert bot._studio_governor_should_preempt_active_cycle(quality_governor, cycle)


def test_studio_governor_preempts_incubator_like_product_workflow_cycle():
    quality_governor = {"mode": "incubator_quality_gate", "avoid_keys": ["new-project-incubator"]}
    incubator_like_cycle = {
        "selected_key": "20260511-studio-cycle-new-product-incubator-1",
        "selected_repo_id": "",
        "selected_type": "PRODUCT_WORKFLOW",
        "selected_lane": "studio",
    }
    ordinary_cycle = {
        "selected_key": "20260511-studio-cycle-product-workflow-1",
        "selected_repo_id": "executivedashboard",
        "selected_type": "PRODUCT_WORKFLOW",
        "selected_lane": "portfolio",
    }

    assert bot._studio_governor_should_preempt_active_cycle(quality_governor, incubator_like_cycle)
    assert not bot._studio_governor_should_preempt_active_cycle(quality_governor, ordinary_cycle)


def test_studio_recent_cycle_outcome_memory_preserves_selected_lane(tmp_path):
    now = 1_700_000_000.0
    db_path = tmp_path / "studio.sqlite"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE studio_cycles (
                cycle_id TEXT PRIMARY KEY,
                selected_key TEXT,
                selected_repo_id TEXT,
                selected_type TEXT,
                selected_lane TEXT,
                outcome_status TEXT,
                outcome_summary TEXT,
                updated_at REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO studio_cycles (
                cycle_id, selected_key, selected_repo_id, selected_type, selected_lane,
                outcome_status, outcome_summary, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cycle-1",
                "20260511-studio-cycle-product-workflow-1",
                "",
                "PRODUCT_WORKFLOW",
                "incubator",
                "rejected_low_value",
                "Incubator lane workflow was rejected.",
                now - 60,
            ),
        )
        conn.execute(
            """
            CREATE TABLE studio_portfolio_projects (
                project_key TEXT PRIMARY KEY,
                project_name TEXT,
                project_path TEXT,
                github_repo TEXT,
                latest_head TEXT,
                latest_summary TEXT,
                status TEXT,
                updated_at REAL
            )
            """
        )
        conn.commit()

    memory = bot._studio_recent_cycle_outcome_memory(db_path, now=now)

    assert memory["recent_studio_negative_outcomes"][0]["lane"] == "incubator"


def test_studio_reconciles_incomplete_published_portfolio_projects(tmp_path):
    now = 1_700_000_000.0
    db_path = tmp_path / "studio.sqlite"
    bot._studio_ensure_schema(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO studio_portfolio_projects(
                project_key, project_name, project_path, github_repo, github_url,
                default_branch, latest_head, private, status, source_order_id,
                latest_order_id, latest_outcome_status, latest_summary,
                validation_summary, monetization_summary, next_milestone,
                first_seen_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "/home/aponce/delay-notice-desk",
                "Delay Notice Desk",
                "/home/aponce/delay-notice-desk",
                "",
                "",
                "main",
                "",
                1,
                "published_private",
                "order-1",
                "order-1",
                "published_project",
                "Local project was validated.",
                "",
                "",
                "",
                now - 10,
                now - 10,
            ),
        )
        conn.commit()

    changed = bot._studio_reconcile_portfolio_publication_contract(db_path, now=now)

    assert changed == 1
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT status, latest_summary FROM studio_portfolio_projects WHERE project_key = ?",
            ("/home/aponce/delay-notice-desk",),
        ).fetchone()
    assert row[0] == "needs_publication"
    assert "missing github_repo, github_url, latest_head" in row[1]

    memory = bot._studio_recent_cycle_outcome_memory(db_path, now=now)

    assert memory["studio_portfolio_total"] == 0


def test_studio_quality_audit_flags_incomplete_published_project(tmp_path):
    now = 1_700_000_000.0
    db_path = tmp_path / "studio.sqlite"
    order_id = "order-quality-1"
    bot._studio_ensure_schema(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO studio_cycles(
                cycle_id, version, ts, status, selected_key, selected_type,
                selected_repo_id, selected_repo_path, selected_lane, thesis,
                rationale, debate_summary, operator_visible_outcome, evidence_target,
                risk_summary, prompt_packet, opportunities_json, outcome_status,
                outcome_summary, order_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cycle-quality-1",
                1,
                now,
                "active",
                "new-project-incubator",
                "NEW_PROJECT",
                "",
                "",
                "incubator",
                "Create a sellable buyer workflow.",
                "rationale",
                "debate",
                "A private GitHub project with demo and buyer evidence.",
                "tests, GitHub publication, and demo evidence",
                "bounded risk",
                "prompt",
                "[]",
                None,
                None,
                order_id,
                now,
                now,
            ),
        )
        conn.commit()

    bot._studio_complete_cycle_for_order_db(
        db_path=db_path,
        order_id=order_id,
        outcome_status="published_project",
        outcome_summary="PASS. Outcome: `published_project`. Local project /home/aponce/delay-notice-desk has tests and buyer demo evidence.",
        now=now + 5,
    )

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        audit = conn.execute(
            "SELECT quality_status, score, summary, checks_json FROM studio_quality_audits WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        portfolio = conn.execute(
            "SELECT status FROM studio_portfolio_projects WHERE project_key = ?",
            ("/home/aponce/delay-notice-desk",),
        ).fetchone()

    assert audit["quality_status"] == "needs_publication"
    assert audit["score"] < 60
    assert "github_repo" in audit["summary"]
    assert json.loads(audit["checks_json"])["publication_complete"] is False
    assert portfolio["status"] == "needs_publication"

    memory = bot._studio_recent_cycle_outcome_memory(db_path, now=now + 10)

    assert memory["studio_quality_low_24h"] == 1
    assert any("needs_publication" in item for item in memory["studio_quality_warnings"])


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


def test_controller_snapshot_delivery_candidate_reports_expected_missing_patch(tmp_path):
    snapshot = tmp_path / "artifacts" / "order" / "controller_snapshot"
    snapshot.mkdir(parents=True)
    patch = snapshot.parent / "changes.patch"

    candidate = bot._controller_snapshot_delivery_candidate(
        {
            "controller_snapshot_workdir": str(snapshot),
            "merge_cancelled": True,
            "result_summary": "PASS implementation/review. Outcome: blocked_need_operator.",
            "result_next_action": "Apply the validated changes into the real repository checkout, commit, push, and deploy.",
        }
    )

    assert candidate["snapshot_dir"] == str(snapshot)
    assert candidate["patch_path"] == str(patch)


def test_controller_snapshot_delivery_candidate_accepts_nested_structured_digest_metadata(tmp_path):
    snapshot = tmp_path / "artifacts" / "order" / "controller_snapshot"
    snapshot.mkdir(parents=True)
    recovery_dir = tmp_path / "recovery"
    recovery_dir.mkdir()
    patch = recovery_dir / "changes.patch"
    patch.write_text("diff --git a/static/app.js b/static/app.js\n", encoding="utf-8")

    candidate = bot._controller_snapshot_delivery_candidate(
        {
            "structured_digest": {
                "controller_snapshot": {"workdir": str(snapshot)},
                "controller_recovery_artifacts": [str(patch), str(patch)],
            },
            "result_summary": "PASS implementation/review. Outcome: blocked_need_operator.",
            "result_next_action": "Apply the validated changes into the real repository checkout, commit, push, and deploy.",
        }
    )

    assert candidate["snapshot_dir"] == str(snapshot)
    assert candidate["patch_path"] == str(patch)


def test_controller_snapshot_autoship_closes_empty_published_project_patch_as_no_delta(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    patch = tmp_path / "changes.patch"
    patch.write_text("", encoding="utf-8")
    completed = {}

    class RecordingQueue:
        def __init__(self):
            self.status_calls = []
            self.phase_calls = []
            self.state_call = None
            self.audit_events = []

        def set_order_status(self, *args, **kwargs):
            self.status_calls.append((args, kwargs))

        def set_order_phase(self, *args, **kwargs):
            self.phase_calls.append((args, kwargs))

        def update_state(self, *args, **kwargs):
            self.state_call = (args, kwargs)

        def append_audit_event(self, *args, **kwargs):
            self.audit_events.append((args, kwargs))

    def fake_run_git(_repo, args, **kwargs):
        if args[:2] == ["apply", "--check"]:
            return SimpleNamespace(returncode=128, stdout="", stderr="error: No valid patches in input")
        if args[:2] == ["rev-parse", "--short"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(bot, "_sync_repo_checkout_to_default_branch", lambda **kwargs: (True, "", None, None))
    monkeypatch.setattr(bot, "_git_status_porcelain", lambda _repo: "")
    monkeypatch.setattr(bot, "_run_git", fake_run_git)
    monkeypatch.setattr(bot, "_controller_snapshot_copy_safe_untracked_files", lambda **kwargs: [])
    monkeypatch.setattr(bot, "_deploy_after_order_merge", lambda **kwargs: {"status": "skipped", "reason": "no_policy", "summary": "Deploy skipped."})
    monkeypatch.setattr(bot, "_studio_complete_cycle_for_order", lambda **kwargs: completed.update(kwargs))

    queue = RecordingQueue()
    result = bot._auto_ship_controller_snapshot_order(
        cfg=SimpleNamespace(),
        orch_q=queue,
        order_id="snapshot-order",
        chat_id=123,
        trace={
            "controller_snapshot_workdir": str(snapshot),
            "result_artifacts": [str(patch)],
            "result_status": "done",
            "result_summary": "PASS. Outcome: `published_project`. Private GitHub repo exists on main with validation evidence.",
        },
        repo_record=None,
        repo_dir=repo,
        default_branch="main",
        now=2_000.0,
    )

    assert result["status"] == "ok"
    assert result["reason"] == "snapshot_no_delta_rejected_low_value"
    assert queue.status_calls[-1][1]["status"] == "done"
    assert queue.phase_calls[-1][1]["phase"] == "done"
    assert queue.state_call[0][1] == "done"
    assert queue.state_call[1]["result_status"] == "rejected_low_value"
    assert queue.state_call[1]["controller_snapshot_autoship_no_delta"] is True
    assert completed["outcome_status"] == "rejected_low_value"
    assert queue.audit_events[-1][1]["event_type"] == "order.controller_snapshot_autoship_no_delta"


def test_controller_snapshot_autoship_rejects_empty_non_published_patch_as_low_value(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    patch = tmp_path / "changes.patch"
    patch.write_text("", encoding="utf-8")
    completed = {}

    class RecordingQueue:
        def __init__(self):
            self.state_call = None

        def set_order_status(self, *args, **kwargs):
            return None

        def set_order_phase(self, *args, **kwargs):
            return None

        def update_state(self, *args, **kwargs):
            self.state_call = (args, kwargs)

        def append_audit_event(self, *args, **kwargs):
            return None

    def fake_run_git(_repo, args, **kwargs):
        if args[:2] == ["apply", "--check"]:
            return SimpleNamespace(returncode=128, stdout="", stderr="error: No valid patches in input")
        if args[:2] == ["rev-parse", "--short"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(bot, "_sync_repo_checkout_to_default_branch", lambda **kwargs: (True, "", None, None))
    monkeypatch.setattr(bot, "_git_status_porcelain", lambda _repo: "")
    monkeypatch.setattr(bot, "_run_git", fake_run_git)
    monkeypatch.setattr(bot, "_controller_snapshot_copy_safe_untracked_files", lambda **kwargs: [])
    monkeypatch.setattr(bot, "_deploy_after_order_merge", lambda **kwargs: {"status": "skipped", "reason": "no_policy", "summary": "Deploy skipped."})
    monkeypatch.setattr(bot, "_studio_complete_cycle_for_order", lambda **kwargs: completed.update(kwargs))

    queue = RecordingQueue()
    result = bot._auto_ship_controller_snapshot_order(
        cfg=SimpleNamespace(),
        orch_q=queue,
        order_id="snapshot-order",
        chat_id=123,
        trace={
            "controller_snapshot_workdir": str(snapshot),
            "result_artifacts": [str(patch)],
            "result_status": "done",
            "result_summary": "PASS validation. Ready to commit, push and deploy, but no material repo change was produced.",
        },
        repo_record=None,
        repo_dir=repo,
        default_branch="main",
        now=2_000.0,
    )

    assert result["status"] == "ok"
    assert result["reason"] == "snapshot_no_delta_rejected_low_value"
    assert queue.state_call[1]["result_status"] == "rejected_low_value"
    assert queue.state_call[1]["merged_to_main"] is False
    assert queue.state_call[1]["controller_snapshot_autoship_commit"] is None
    assert completed["outcome_status"] == "rejected_low_value"


def test_controller_snapshot_delivery_candidate_nested_metadata_still_requires_release_and_validation(tmp_path):
    snapshot = tmp_path / "artifacts" / "order" / "controller_snapshot"
    snapshot.mkdir(parents=True)
    recovery_dir = tmp_path / "recovery"
    recovery_dir.mkdir()
    patch = recovery_dir / "changes.patch"
    patch.write_text("diff --git a/static/app.js b/static/app.js\n", encoding="utf-8")
    nested_metadata = {
        "structured_digest": {
            "controller_snapshot": {"workdir": str(snapshot)},
            "controller_recovery_artifacts": [str(patch)],
        }
    }

    assert not bot._controller_snapshot_delivery_candidate(
        {
            **nested_metadata,
            "result_summary": "PASS implementation/review completed with tests.",
        }
    )
    assert not bot._controller_snapshot_delivery_candidate(
        {
            **nested_metadata,
            "result_summary": "Outcome: blocked_need_operator. Operator must apply changes.",
            "result_next_action": "Ship to main is blocked.",
        }
    )


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


def test_recent_controller_snapshot_rows_include_nested_snapshot_metadata(tmp_path):
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
            (
                "order-1",
                json.dumps(
                    {
                        "structured_digest": {
                            "controller_snapshot": {"workdir": "/tmp/snapshot"},
                            "controller_recovery_artifacts": ["/tmp/changes.patch"],
                        }
                    }
                ),
            ),
        )

    orch_q = SimpleNamespace(_storage=SimpleNamespace(path=db))
    rows = bot._recent_controller_snapshot_studio_order_rows(
        orch_q=orch_q,
        now=2_100.0,
        max_age_seconds=300.0,
    )

    assert len(rows) == 1
    assert rows[0]["order_id"] == "order-1"


def test_recent_controller_snapshot_rows_skip_terminal_autoship_failure(tmp_path):
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
            ("order-1", "failed", "failed_root_caused", 2_000.0),
        )
        conn.execute(
            "INSERT INTO ceo_orders VALUES (?, ?, ?, ?, ?, ?)",
            ("order-1", 123, "failed", "failed", "Snapshot order", 2_000.0),
        )
        conn.execute(
            "INSERT INTO jobs VALUES (?, ?)",
            (
                "order-1",
                json.dumps(
                    {
                        "controller_snapshot_workdir": "/tmp/snapshot",
                        "controller_snapshot_autoship_error": {
                            "status": "failed",
                            "reason": "snapshot_patch_check_failed",
                        },
                        "blocked_reason": "controller_snapshot_autoship_failed:snapshot_patch_check_failed",
                    }
                ),
            ),
        )

    orch_q = SimpleNamespace(_storage=SimpleNamespace(path=db))
    rows = bot._recent_controller_snapshot_studio_order_rows(
        orch_q=orch_q,
        now=2_100.0,
        max_age_seconds=300.0,
    )

    assert rows == []


def _seed_snapshot_autoship_recovery_order(tmp_path, *, now: float = 2_000.0, updated_at: float | None = None):
    db = tmp_path / "jobs.sqlite"
    storage = SQLiteTaskStorage(db)
    orch_q = OrchestratorQueue(storage)
    chat_id = 123
    order_id = "snapshot-order"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    patch = tmp_path / "changes.patch"
    patch.write_text("diff --git a/app.py b/app.py\n", encoding="utf-8")
    trace = {
        "controller_snapshot_workdir": str(snapshot),
        "result_artifacts": [str(patch)],
        "result_status": "blocked_need_operator",
        "result_summary": "Outcome: blocked_need_operator. PASS validated controller snapshot.",
        "result_next_action": "Run the validated patch in a write-enabled source checkout.",
    }
    orch_q.submit_task(
        Task.new(
            job_id=order_id,
            source="test",
            role="skynet",
            input_text="recover snapshot",
            request_type="exec",
            priority=1,
            model="gpt-5.5",
            effort="high",
            mode_hint="full",
            requires_approval=False,
            max_cost_window_usd=0,
            chat_id=chat_id,
            state="done",
            trace=trace,
        )
    )
    orch_q.upsert_order(
        order_id=order_id,
        chat_id=chat_id,
        title="Snapshot order",
        body="recover",
        status="done",
        phase="done",
    )
    bot._studio_ensure_schema(db)
    cycle_updated_at = float(now - 60 if updated_at is None else updated_at)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO studio_cycles(
                cycle_id, version, ts, status, selected_key, selected_type, selected_repo_id,
                selected_repo_path, selected_lane, thesis, rationale, debate_summary,
                operator_visible_outcome, evidence_target, risk_summary, prompt_packet,
                opportunities_json, outcome_status, outcome_summary, order_id, created_at, updated_at
            ) VALUES (?, 1, ?, 'active', 'repo-codexbot', 'DEEP_IMPROVEMENT', 'codexbot',
                ?, 'core', 'Repair delivery.', 'Because autoship backlog.', 'Critic agrees.',
                'No autoship backlog.', 'Tests.', 'Low.', 'packet', '[]',
                'blocked_need_operator', 'PASS validated controller snapshot waiting for autoship recovery.',
                ?, ?, ?)
            """,
            ("cycle-1", cycle_updated_at, str(tmp_path), order_id, cycle_updated_at, cycle_updated_at),
        )
        conn.commit()
    return db, orch_q, chat_id, order_id


def test_auto_merge_tick_root_causes_pre_deploy_controller_snapshot_autoship_failure(tmp_path, monkeypatch):
    now = 2_000.0
    db, orch_q, chat_id, order_id = _seed_snapshot_autoship_recovery_order(tmp_path, now=now)

    autoship_result = {
        "status": "failed",
        "reason": "snapshot_patch_check_failed",
        "detail": "error: patch failed: app.py:12",
    }
    monkeypatch.setattr(bot, "_repo_context_for_order", lambda **kwargs: (None, tmp_path, "main"))
    monkeypatch.setattr(bot, "_auto_ship_controller_snapshot_order", lambda **kwargs: autoship_result)

    merged = bot._auto_merge_ready_orders_tick(
        cfg=SimpleNamespace(orchestrator_db_path=db, codex_workdir=tmp_path),
        api=SimpleNamespace(),
        orch_q=orch_q,
        now=now,
    )

    assert merged == 0
    order = orch_q.get_order(order_id, chat_id=chat_id)
    job = orch_q.get_job(order_id)
    assert order["status"] == "failed"
    assert order["phase"] == "failed"
    assert job.state == "failed"
    assert job.trace["result_status"] == "failed_root_caused"
    assert "snapshot patch check failed" in job.trace["result_summary"]
    assert "app.py:12" in job.trace["result_summary"]
    assert "waiting for autoship" not in job.trace["result_summary"].lower()
    assert job.trace["controller_snapshot_autoship_error"] == autoship_result
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT status, outcome_status, outcome_summary FROM studio_cycles WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    assert row == (
        "failed",
        "failed_root_caused",
        "Controller snapshot autoship failed_root_caused: snapshot patch check failed (reason=snapshot_patch_check_failed). Detail: error: patch failed: app.py:12",
    )


def test_auto_merge_tick_does_not_retry_terminal_controller_snapshot_autoship_failure(tmp_path, monkeypatch):
    now = 2_000.0
    db, orch_q, chat_id, order_id = _seed_snapshot_autoship_recovery_order(tmp_path, now=now)
    autoship_result = {
        "status": "failed",
        "reason": "snapshot_patch_check_failed",
        "detail": "error: patch failed: app.py:12",
    }
    monkeypatch.setattr(bot, "_repo_context_for_order", lambda **kwargs: (None, tmp_path, "main"))
    monkeypatch.setattr(bot, "_auto_ship_controller_snapshot_order", lambda **kwargs: autoship_result)

    bot._auto_merge_ready_orders_tick(
        cfg=SimpleNamespace(orchestrator_db_path=db, codex_workdir=tmp_path),
        api=SimpleNamespace(),
        orch_q=orch_q,
        now=now,
    )

    seen = {"called": False}

    def fail_if_retried(**kwargs):
        seen["called"] = True
        raise AssertionError("terminal autoship failure should not be retried")

    monkeypatch.setattr(bot, "_auto_ship_controller_snapshot_order", fail_if_retried)
    merged = bot._auto_merge_ready_orders_tick(
        cfg=SimpleNamespace(orchestrator_db_path=db, codex_workdir=tmp_path),
        api=SimpleNamespace(),
        orch_q=orch_q,
        now=now + 30,
    )

    assert merged == 0
    assert seen["called"] is False
    order = orch_q.get_order(order_id, chat_id=chat_id)
    job = orch_q.get_job(order_id)
    assert order["status"] == "failed"
    assert order["phase"] == "failed"
    assert job.trace["controller_snapshot_autoship_error"] == autoship_result
    assert job.trace["blocked_reason"] == "controller_snapshot_autoship_failed:snapshot_patch_check_failed"


def test_auto_merge_tick_root_causes_missing_controller_snapshot_patch(tmp_path, monkeypatch):
    now = 2_000.0
    db, orch_q, chat_id, order_id = _seed_snapshot_autoship_recovery_order(tmp_path, now=now)
    missing_patch = tmp_path / "changes.patch"
    missing_patch.unlink()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    trace = dict(orch_q.get_job(order_id).trace or {})
    candidate = bot._controller_snapshot_delivery_candidate(trace)
    assert candidate["patch_path"] == str(missing_patch)

    monkeypatch.setattr(bot, "_repo_context_for_order", lambda **kwargs: (None, repo, "main"))

    merged = bot._auto_merge_ready_orders_tick(
        cfg=SimpleNamespace(orchestrator_db_path=db, codex_workdir=tmp_path),
        api=SimpleNamespace(),
        orch_q=orch_q,
        now=now,
    )

    assert merged == 0
    order = orch_q.get_order(order_id, chat_id=chat_id)
    job = orch_q.get_job(order_id)
    assert order["status"] == "failed"
    assert order["phase"] == "failed"
    assert job.state == "failed"
    assert job.trace["result_status"] == "failed_root_caused"
    assert "missing controller snapshot artifacts" in job.trace["result_summary"]
    assert job.trace["result_next_action"] == "Regenerate the controller snapshot artifacts and ensure changes.patch is present."
    assert job.trace["controller_snapshot_autoship_error"] == {
        "status": "failed",
        "reason": "missing_snapshot_artifacts",
    }
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT status, outcome_status, outcome_summary FROM studio_cycles WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    assert row == (
        "failed",
        "failed_root_caused",
        "Controller snapshot autoship failed_root_caused: missing controller snapshot artifacts (reason=missing_snapshot_artifacts).",
    )


def test_auto_merge_tick_preserves_recorded_controller_snapshot_deploy_failure(tmp_path, monkeypatch):
    now = 2_000.0
    db, orch_q, chat_id, order_id = _seed_snapshot_autoship_recovery_order(tmp_path, now=now)
    deploy_summary = "Auto-shipped validated controller snapshot to main commit=abc123. Deploy failed: service restart failed."
    autoship_result = {
        "status": "failed",
        "reason": "snapshot_autoship_deploy_failed",
        "summary": deploy_summary,
        "commit": "abc123",
        "deploy": {
            "status": "failed",
            "reason": "deploy_script_failed",
            "detail": "service restart failed",
            "summary": "Deploy failed: service restart failed.",
        },
    }

    def record_deploy_failure(**kwargs):
        orch_q.set_order_status(order_id, chat_id=chat_id, status="active")
        orch_q.set_order_phase(order_id, chat_id=chat_id, phase="review")
        orch_q.update_state(
            order_id,
            "blocked",
            blocked_reason="deploy_failed",
            merge_ready=False,
            merge_required=False,
            deploy_status="failed",
            deploy_result="deploy_script_failed",
            deploy_summary="Deploy failed: service restart failed.",
            deployed_commit=None,
            deploy_error="service restart failed",
            result_status="deploy_failed",
            result_summary=deploy_summary,
            result_next_action="Inspect deployment failure and complete rollout.",
        )
        bot._studio_complete_cycle_for_order_from_queue(
            orch_q=orch_q,
            order_id=order_id,
            outcome_status="failed_root_caused",
            outcome_summary=deploy_summary,
            now=now,
        )
        return autoship_result

    monkeypatch.setattr(bot, "_repo_context_for_order", lambda **kwargs: (None, tmp_path, "main"))
    monkeypatch.setattr(bot, "_auto_ship_controller_snapshot_order", record_deploy_failure)

    merged = bot._auto_merge_ready_orders_tick(
        cfg=SimpleNamespace(orchestrator_db_path=db, codex_workdir=tmp_path),
        api=SimpleNamespace(),
        orch_q=orch_q,
        now=now,
    )

    assert merged == 0
    order = orch_q.get_order(order_id, chat_id=chat_id)
    job = orch_q.get_job(order_id)
    assert order["status"] == "active"
    assert order["phase"] == "review"
    assert job.state == "blocked"
    assert job.blocked_reason == "deploy_failed"
    assert job.trace["deploy_status"] == "failed"
    assert job.trace["deploy_result"] == "deploy_script_failed"
    assert job.trace["deploy_summary"] == "Deploy failed: service restart failed."
    assert job.trace["deploy_error"] == "service restart failed"
    assert job.trace["result_status"] == "deploy_failed"
    assert job.trace["result_summary"] == deploy_summary
    assert job.trace["result_next_action"] == "Inspect deployment failure and complete rollout."
    assert job.trace["controller_snapshot_autoship_error"] == autoship_result
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT status, outcome_status, outcome_summary FROM studio_cycles WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    assert row == ("failed", "failed_root_caused", deploy_summary)


def test_auto_merge_tick_does_not_overwrite_recorded_controller_snapshot_deploy_failure(tmp_path, monkeypatch):
    now = 2_000.0
    db, orch_q, chat_id, order_id = _seed_snapshot_autoship_recovery_order(tmp_path, now=now)
    deploy_summary = "Auto-shipped validated controller snapshot to main commit=abc123. Deploy failed: service restart failed."
    autoship_result = {
        "status": "failed",
        "reason": "snapshot_autoship_deploy_failed",
        "summary": deploy_summary,
        "commit": "abc123",
        "deploy": {
            "status": "failed",
            "reason": "deploy_script_failed",
            "detail": "service restart failed",
            "summary": "Deploy failed: service restart failed.",
        },
    }
    orch_q.set_order_status(order_id, chat_id=chat_id, status="active")
    orch_q.set_order_phase(order_id, chat_id=chat_id, phase="review")
    orch_q.update_state(
        order_id,
        "blocked",
        blocked_reason="deploy_failed",
        merge_ready=False,
        merge_required=False,
        deploy_status="failed",
        deploy_result="deploy_script_failed",
        deploy_summary="Deploy failed: service restart failed.",
        deploy_error="service restart failed",
        result_status="deploy_failed",
        result_summary=deploy_summary,
        result_next_action="Inspect deployment failure and complete rollout.",
        controller_snapshot_autoship_error=autoship_result,
    )
    bot._studio_complete_cycle_for_order_from_queue(
        orch_q=orch_q,
        order_id=order_id,
        outcome_status="failed_root_caused",
        outcome_summary=deploy_summary,
        now=now,
    )

    seen = {"called": False}

    def fail_if_retried(**kwargs):
        seen["called"] = True
        raise AssertionError("recorded deploy failure should not be overwritten")

    monkeypatch.setattr(bot, "_repo_context_for_order", lambda **kwargs: (None, tmp_path, "main"))
    monkeypatch.setattr(bot, "_auto_ship_controller_snapshot_order", fail_if_retried)

    merged = bot._auto_merge_ready_orders_tick(
        cfg=SimpleNamespace(orchestrator_db_path=db, codex_workdir=tmp_path),
        api=SimpleNamespace(),
        orch_q=orch_q,
        now=now + 30,
    )

    assert merged == 0
    assert seen["called"] is False
    order = orch_q.get_order(order_id, chat_id=chat_id)
    job = orch_q.get_job(order_id)
    assert order["status"] == "active"
    assert order["phase"] == "review"
    assert job.state == "blocked"
    assert job.blocked_reason == "deploy_failed"
    assert job.trace["result_status"] == "deploy_failed"
    assert job.trace["result_summary"] == deploy_summary
    assert job.trace["controller_snapshot_autoship_error"] == autoship_result


def test_auto_merge_tick_recovers_aged_controller_snapshot_blocker_within_72h(tmp_path, monkeypatch):
    now = 400_000.0
    updated_at = now - (7 * 3600)
    db, orch_q, chat_id, order_id = _seed_snapshot_autoship_recovery_order(tmp_path, now=now, updated_at=updated_at)
    seen = {}
    autoship_result = {
        "status": "failed",
        "reason": "snapshot_patch_check_failed",
        "detail": "error: patch aged but recoverable",
    }

    def record_autoship(**kwargs):
        seen["order_id"] = kwargs["order_id"]
        return autoship_result

    monkeypatch.setattr(bot, "_repo_context_for_order", lambda **kwargs: (None, tmp_path, "main"))
    monkeypatch.setattr(bot, "_auto_ship_controller_snapshot_order", record_autoship)

    merged = bot._auto_merge_ready_orders_tick(
        cfg=SimpleNamespace(orchestrator_db_path=db, codex_workdir=tmp_path),
        api=SimpleNamespace(),
        orch_q=orch_q,
        now=now,
    )

    assert merged == 0
    assert seen["order_id"] == order_id
    job = orch_q.get_job(order_id)
    assert job.trace["result_status"] == "failed_root_caused"
    assert "patch aged but recoverable" in job.trace["result_summary"]


def test_controller_snapshot_autoship_success_reports_persistence_failure_after_push(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    patch = tmp_path / "changes.patch"
    patch.write_text("diff --git a/app.py b/app.py\n", encoding="utf-8")
    commands = []

    class FailingUpdateQueue:
        def set_order_status(self, *args, **kwargs):
            return None

        def set_order_phase(self, *args, **kwargs):
            return None

        def update_state(self, *args, **kwargs):
            raise RuntimeError("sqlite is locked")

        def append_audit_event(self, *args, **kwargs):
            raise AssertionError("audit should not run after failed state persistence")

    def fake_run_git(_repo, args, **kwargs):
        commands.append(list(args))
        stdout = "abc123\n" if args[:2] == ["rev-parse", "--short"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(bot, "_sync_repo_checkout_to_default_branch", lambda **kwargs: (True, "", None, None))
    monkeypatch.setattr(bot, "_git_status_porcelain", lambda _repo: "" if len(commands) < 2 else " M app.py\n")
    monkeypatch.setattr(bot, "_run_git", fake_run_git)
    monkeypatch.setattr(bot, "_controller_snapshot_copy_safe_untracked_files", lambda **kwargs: [])
    monkeypatch.setattr(bot, "_controller_snapshot_validation_commands", lambda *args, **kwargs: [])
    monkeypatch.setattr(bot, "_deploy_after_order_merge", lambda **kwargs: {"status": "ok", "reason": "noop", "summary": "Deploy skipped."})
    monkeypatch.setattr(bot, "_studio_complete_cycle_for_order", lambda **kwargs: None)

    result = bot._auto_ship_controller_snapshot_order(
        cfg=SimpleNamespace(),
        orch_q=FailingUpdateQueue(),
        order_id="snapshot-order",
        chat_id=123,
        trace={
            "controller_snapshot_workdir": str(snapshot),
            "result_artifacts": [str(patch)],
            "result_status": "blocked_need_operator",
            "result_summary": "Outcome: blocked_need_operator. PASS validated controller snapshot.",
        },
        repo_record=None,
        repo_dir=repo,
        default_branch="main",
        now=2_000.0,
    )

    assert result["status"] == "failed"
    assert result["reason"] == "snapshot_autoship_persistence_failed"
    assert result["original_reason"] == "snapshot_autoship_complete"
    assert "sqlite is locked" in result["detail"]
    assert result["commit"] == "abc123"
    assert sum(1 for args in commands if args[:2] == ["push", "origin"]) == 1


def test_controller_snapshot_autoship_retries_transient_push_failure(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    patch = tmp_path / "changes.patch"
    patch.write_text("diff --git a/app.py b/app.py\n", encoding="utf-8")
    commands = []
    completed = {}
    status_calls = {"count": 0}

    class RecordingQueue:
        def __init__(self):
            self.state_call = None

        def set_order_status(self, *args, **kwargs):
            return None

        def set_order_phase(self, *args, **kwargs):
            return None

        def update_state(self, *args, **kwargs):
            self.state_call = (args, kwargs)

        def append_audit_event(self, *args, **kwargs):
            return None

    def fake_run_git(_repo, args, **kwargs):
        commands.append(list(args))
        if args[:2] == ["rev-parse", "--short"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if args[:2] == ["push", "origin"]:
            push_count = sum(1 for command in commands if command[:2] == ["push", "origin"])
            if push_count == 1:
                return SimpleNamespace(returncode=128, stdout="", stderr="ssh: Could not resolve hostname github.com: Temporary failure in name resolution")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_status(_repo):
        status_calls["count"] += 1
        return "" if status_calls["count"] == 1 else " M app.py\n"

    monkeypatch.setattr(bot, "_sync_repo_checkout_to_default_branch", lambda **kwargs: (True, "", None, None))
    monkeypatch.setattr(bot, "_git_status_porcelain", fake_status)
    monkeypatch.setattr(bot, "_run_git", fake_run_git)
    monkeypatch.setattr(bot, "_controller_snapshot_copy_safe_untracked_files", lambda **kwargs: [])
    monkeypatch.setattr(bot, "_controller_snapshot_validation_commands", lambda *args, **kwargs: [])
    monkeypatch.setattr(bot, "_deploy_after_order_merge", lambda **kwargs: {"status": "skipped", "reason": "no_policy", "summary": "Deploy skipped."})
    monkeypatch.setattr(bot, "_studio_complete_cycle_for_order", lambda **kwargs: completed.update(kwargs))
    monkeypatch.setattr(bot.time, "sleep", lambda _seconds: None)

    queue = RecordingQueue()
    result = bot._auto_ship_controller_snapshot_order(
        cfg=SimpleNamespace(),
        orch_q=queue,
        order_id="snapshot-order",
        chat_id=123,
        trace={
            "controller_snapshot_workdir": str(snapshot),
            "result_artifacts": [str(patch)],
            "result_status": "blocked_need_operator",
            "result_summary": "Outcome: blocked_need_operator. PASS validated controller snapshot.",
        },
        repo_record=None,
        repo_dir=repo,
        default_branch="main",
        now=2_000.0,
    )

    assert result["status"] == "ok"
    assert result["reason"] == "snapshot_autoship_complete"
    assert sum(1 for args in commands if args[:2] == ["push", "origin"]) == 2
    assert queue.state_call[1]["controller_snapshot_autoship_done"] is True
    assert completed["outcome_status"] == "shipped_to_main"


def test_controller_snapshot_autoship_failure_reports_persistence_failure(tmp_path):
    class FailingFailureQueue:
        def set_order_status(self, *args, **kwargs):
            return None

        def set_order_phase(self, *args, **kwargs):
            return None

        def update_state(self, *args, **kwargs):
            raise RuntimeError("cannot persist root cause")

        def append_audit_event(self, *args, **kwargs):
            raise AssertionError("audit should not run after failed state persistence")

    autoship_result = {
        "status": "failed",
        "reason": "snapshot_patch_check_failed",
        "detail": "patch failed",
    }

    result = bot._finalize_controller_snapshot_autoship_failure(
        orch_q=FailingFailureQueue(),
        order_id="snapshot-order",
        chat_id=123,
        autoship=autoship_result,
        now=2_000.0,
    )

    assert result["status"] == "failed"
    assert result["reason"] == "snapshot_autoship_persistence_failed"
    assert result["original_reason"] == "snapshot_patch_check_failed"
    assert "cannot persist root cause" in result["detail"]
    assert result["original_result"] == autoship_result


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


def test_unwritable_android_build_outputs_trigger_fresh_autoship_worktree(tmp_path, monkeypatch):
    repo = tmp_path / "android"
    repo.mkdir()
    (repo / "gradlew").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    app_build = repo / "app" / "build"
    app_build.mkdir(parents=True)
    app_build.chmod(0o555)
    monkeypatch.setenv("BOT_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))

    try:
        assert bot._repo_has_unwritable_build_outputs(repo)
        fresh_dir, fresh_branch = bot._fresh_autoship_worktree_dir(repo, "main")
        assert str(fresh_dir).startswith(str(tmp_path / "artifacts" / "autoship_worktrees"))
        assert "poncebot/autoship/android-main-" in fresh_branch
    finally:
        app_build.chmod(0o755)


def test_default_branch_tracked_changes_use_fresh_autoship_worktree(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    dirty_file = repo / "bot.py"
    dirty_file.write_text("local dirty source checkout\n", encoding="utf-8")
    fresh_dir = tmp_path / "autoship" / "repo-main-fresh"
    calls = []

    def fake_run_git(target_repo, args, **kwargs):
        calls.append((Path(target_repo), list(args)))
        if args == ["fetch", "origin", "--prune"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if args == ["status", "--porcelain", "--untracked-files=no"]:
            stdout = " M bot.py\n" if Path(target_repo) == repo else ""
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        if args[:4] == ["worktree", "add", "-B", "poncebot/autoship/repo-main-fresh"]:
            assert Path(args[4]) == fresh_dir
            fresh_dir.mkdir(parents=True)
            (fresh_dir / ".git").write_text("gitdir: fake\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args == ["merge", "--ff-only", "origin/main"]:
            assert Path(target_repo) == fresh_dir
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args == ["rev-parse", "--short", "HEAD"]:
            assert Path(target_repo) == fresh_dir
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        raise AssertionError(f"unexpected git call: {target_repo} {args}")

    monkeypatch.setattr(bot, "_run_git", fake_run_git)
    monkeypatch.setattr(bot, "_git_fetch_remote_branch", lambda _repo, _branch: (True, ""))
    monkeypatch.setattr(bot, "_repo_has_unwritable_build_outputs", lambda _repo: False)
    monkeypatch.setattr(
        bot,
        "_fresh_autoship_worktree_dir",
        lambda _repo, _branch: (fresh_dir, "poncebot/autoship/repo-main-fresh"),
    )

    synced, reason, deployed_commit, checkout = bot._sync_repo_checkout_to_default_branch(repo=repo, default_branch="main")

    assert synced is True
    assert reason == "synced"
    assert deployed_commit == "abc123"
    assert checkout == fresh_dir
    assert dirty_file.read_text(encoding="utf-8") == "local dirty source checkout\n"
    assert not any(target == repo and args[:2] == ["merge", "--ff-only"] for target, args in calls)
    assert any(target == repo and args[:2] == ["worktree", "add"] for target, args in calls)


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
    assert rows["snapshot-cycle"] == ("blocked", "blocked_need_operator")


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


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    current_module = globals()
    for name in sorted(current_module):
        if not name.startswith("test_"):
            continue
        candidate = current_module[name]
        if not callable(candidate):
            continue
        if inspect.signature(candidate).parameters:
            continue
        suite.addTest(unittest.FunctionTestCase(candidate))
    return suite
