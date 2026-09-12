"""Reliability budget checks.

All reliability mechanisms must respect the episode-level budget; exceeding
it means controlled abort (BUDGET_EXCEEDED), never an unbounded loop.
Per-call retry limits live in RetryPolicy; this module owns the cross-step
episode budget.
"""

from __future__ import annotations

from web_harness.core.reliability import ReliabilityBudget, ReliabilityState


def extra_model_calls_remaining(
    state: ReliabilityState, budget: ReliabilityBudget
) -> int:
    return max(0, budget.max_extra_model_calls_per_episode - state.extra_model_calls)


def episode_budget_exhausted(
    state: ReliabilityState, budget: ReliabilityBudget
) -> bool:
    return state.extra_model_calls >= budget.max_extra_model_calls_per_episode
