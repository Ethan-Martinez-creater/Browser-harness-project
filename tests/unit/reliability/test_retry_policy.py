"""Unit tests for RetryPolicy (Phase 1B).

max_retries=2 means 1 initial attempt + 2 retries (3 calls total).
"""


from web_harness.core.reliability import (
    FailureKind,
    ReliabilityBudget,
    ReliabilityState,
)
from web_harness.reliability.retry import RetryPolicy

BUDGET = ReliabilityBudget()


def decide(kind, retries_already_done, state=None, policy=None, budget=BUDGET):
    p = policy or RetryPolicy()
    return p.decide(
        failure_kind=kind,
        retry_index_for_kind=retries_already_done,
        reliability_state=state or ReliabilityState(),
        budget=budget,
    )


def test_api_retry_budget_allows_two_retries():
    kind = FailureKind.MODEL_API_TRANSIENT
    assert decide(kind, 0).retry
    assert decide(kind, 1).retry
    d3 = decide(kind, 2)
    assert not d3.retry
    assert d3.reason == "retry_exhausted"


def test_api_backoff_is_fixed_and_deterministic():
    d1 = decide(FailureKind.MODEL_API_TRANSIENT, 0)
    d2 = decide(FailureKind.MODEL_API_TRANSIENT, 1)
    assert d1.backoff_ms == 500
    assert d2.backoff_ms == 1000
    # no jitter: identical decisions every time
    assert decide(FailureKind.MODEL_API_TRANSIENT, 0).backoff_ms == 500


def test_parse_retry_allows_one_repair():
    kind = FailureKind.MODEL_OUTPUT_INVALID
    assert decide(kind, 0).retry
    d2 = decide(kind, 1)
    assert not d2.retry and d2.reason == "retry_exhausted"


def test_parse_repair_has_no_backoff():
    assert decide(FailureKind.MODEL_OUTPUT_INVALID, 0).backoff_ms == 0


def test_environment_failures_never_retry():
    for kind in (
        FailureKind.ACTION_ERROR,
        FailureKind.NO_PROGRESS,
        FailureKind.LOOP_DETECTED,
        FailureKind.TASK_FAILED,
    ):
        d = decide(kind, 0)
        assert not d.retry
        assert "not retryable" in d.reason


def test_episode_budget_blocks_retry():
    state = ReliabilityState(extra_model_calls=6)
    d = decide(
        FailureKind.MODEL_API_TRANSIENT, 0,
        state=state,
        budget=ReliabilityBudget(max_extra_model_calls_per_episode=6),
    )
    assert not d.retry
    assert "budget exhausted" in d.reason


def test_budget_one_remaining_allows_one_retry():
    state = ReliabilityState(extra_model_calls=5)
    d = decide(
        FailureKind.MODEL_API_TRANSIENT, 0,
        state=state,
        budget=ReliabilityBudget(max_extra_model_calls_per_episode=6),
    )
    assert d.retry


def test_custom_limits():
    policy = RetryPolicy(api_max_retries=1, api_backoff_ms=[250], parse_max_retries=0)
    assert decide(FailureKind.MODEL_API_TRANSIENT, 0, policy=policy).retry
    assert decide(FailureKind.MODEL_API_TRANSIENT, 0, policy=policy).backoff_ms == 250
    assert not decide(FailureKind.MODEL_API_TRANSIENT, 1, policy=policy).retry
    # parse_max_retries=0: parse failures are terminal immediately
    assert not decide(FailureKind.MODEL_OUTPUT_INVALID, 0, policy=policy).retry
