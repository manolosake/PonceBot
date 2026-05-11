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
