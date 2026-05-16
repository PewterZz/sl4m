from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AdaptivePldAction = Literal["continue", "fallback"]


@dataclass(frozen=True)
class AdaptivePldConfig:
    min_rounds: int = 16
    min_tokens: int = 64
    window_rounds: int = 12
    min_rounds_with_draft_rate: float = 0.25
    min_from_draft_rate: float = 0.10
    min_accept_per_round: float = 0.25


@dataclass(frozen=True)
class AdaptivePldStats:
    rounds: int
    tokens: int
    draft_tokens: int
    accepted_tokens: int
    rounds_with_draft: int
    rounds_with_draft_rate: float
    from_draft_rate: float
    per_round_accept: float
    mean_accept_per_round: float


@dataclass(frozen=True)
class AdaptivePldDecision:
    action: AdaptivePldAction
    reason: str
    stats: AdaptivePldStats


def summarize_pld_rounds(
    rounds: list[dict],
    *,
    tokens: int,
    window_rounds: int,
) -> AdaptivePldStats:
    sample = rounds[-window_rounds:] if window_rounds > 0 else rounds
    draft_tokens = sum(int(r.get("num_draft", 0)) for r in sample)
    accepted_tokens = sum(int(r.get("num_accept", 0)) for r in sample)
    rounds_with_draft = sum(1 for r in sample if int(r.get("num_draft", 0)) > 0)
    rounds_count = len(sample)
    return AdaptivePldStats(
        rounds=len(rounds),
        tokens=tokens,
        draft_tokens=draft_tokens,
        accepted_tokens=accepted_tokens,
        rounds_with_draft=rounds_with_draft,
        rounds_with_draft_rate=(rounds_with_draft / rounds_count) if rounds_count else 0.0,
        from_draft_rate=(accepted_tokens / tokens) if tokens else 0.0,
        per_round_accept=(accepted_tokens / draft_tokens) if draft_tokens else 0.0,
        mean_accept_per_round=(accepted_tokens / rounds_count) if rounds_count else 0.0,
    )


def decide_adaptive_pld(
    rounds: list[dict],
    *,
    tokens: int,
    config: AdaptivePldConfig = AdaptivePldConfig(),
) -> AdaptivePldDecision:
    stats = summarize_pld_rounds(
        rounds,
        tokens=tokens,
        window_rounds=config.window_rounds,
    )
    if stats.rounds < config.min_rounds or stats.tokens < config.min_tokens:
        return AdaptivePldDecision("continue", "warming-up", stats)

    if stats.rounds_with_draft_rate < config.min_rounds_with_draft_rate:
        return AdaptivePldDecision("fallback", "few-pld-matches", stats)
    if stats.from_draft_rate < config.min_from_draft_rate:
        return AdaptivePldDecision("fallback", "low-draft-token-rate", stats)
    if stats.per_round_accept < config.min_accept_per_round:
        return AdaptivePldDecision("fallback", "low-accept-rate", stats)
    return AdaptivePldDecision("continue", "pld-healthy", stats)


def resolve_agent_mode(requested: str, route_mode: str) -> str:
    """Map user-facing agent modes to the concrete decoder mode."""
    if requested in {"auto", "agent-auto"}:
        return "adaptive-pld" if route_mode == "pld" else "baseline"
    return requested
