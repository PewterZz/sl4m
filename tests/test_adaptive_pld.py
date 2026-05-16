from __future__ import annotations

from sl4m.adaptive_pld import AdaptivePldConfig, decide_adaptive_pld, resolve_agent_mode


def test_adaptive_pld_waits_for_enough_signal() -> None:
    decision = decide_adaptive_pld(
        [{"num_draft": 0, "num_accept": 0}] * 4,
        tokens=12,
    )

    assert decision.action == "continue"
    assert decision.reason == "warming-up"


def test_adaptive_pld_falls_back_when_matches_are_rare() -> None:
    rounds = [{"num_draft": 0, "num_accept": 0}] * 16

    decision = decide_adaptive_pld(rounds, tokens=64)

    assert decision.action == "fallback"
    assert decision.reason == "few-pld-matches"
    assert decision.stats.rounds_with_draft_rate == 0.0


def test_adaptive_pld_keeps_healthy_prompt_lookup() -> None:
    rounds = [{"num_draft": 4, "num_accept": 3}] * 16

    decision = decide_adaptive_pld(rounds, tokens=64)

    assert decision.action == "continue"
    assert decision.reason == "pld-healthy"
    assert decision.stats.per_round_accept == 0.75


def test_adaptive_pld_uses_recent_window() -> None:
    rounds = (
        [{"num_draft": 0, "num_accept": 0}] * 8
        + [{"num_draft": 4, "num_accept": 2}] * 8
    )
    config = AdaptivePldConfig(min_rounds=8, min_tokens=32, window_rounds=8)

    decision = decide_adaptive_pld(rounds, tokens=40, config=config)

    assert decision.action == "continue"
    assert decision.stats.rounds == 16
    assert decision.stats.rounds_with_draft_rate == 1.0


def test_resolve_agent_mode_routes_auto_to_adaptive_pld_for_edits() -> None:
    assert resolve_agent_mode("auto", "pld") == "adaptive-pld"
    assert resolve_agent_mode("agent-auto", "pld") == "adaptive-pld"
    assert resolve_agent_mode("agent-auto", "baseline") == "baseline"
    assert resolve_agent_mode("pld", "baseline") == "pld"
