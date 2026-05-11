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
