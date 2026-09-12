"""Controlled retry policy (Phase 1B).

The policy handles ONLY model-side failures that have not touched the
environment:

    FailureKind.MODEL_API_TRANSIENT   (MODEL_API_ERROR)
    FailureKind.MODEL_OUTPUT_INVALID  (MODEL_OUTPUT_PARSE_ERROR)

Environment-side failures (ACTION_ERROR, NO_PROGRESS, LOOP_DETECTED) never
enter the retry branch — they belong to Recovery (Phase 1C). Browser actions
are never retried blindly: a retry here re-runs a model call, never an
environment step.

Terminology: `max_retries = 2` means 1 initial attempt + 2 retry attempts
(3 model calls maximum).
"""

from __future__ import annotations

from pydantic import BaseModel

from web_harness.core.reliability import (
    FailureKind,
    ReliabilityBudget,
    ReliabilityState,
)
from web_harness.reliability.budget import episode_budget_exhausted

RETRYABLE_KINDS = {
    FailureKind.MODEL_API_TRANSIENT,
    FailureKind.MODEL_OUTPUT_INVALID,
}


class RetryDecision(BaseModel):
    retry: bool
    reason: str
    backoff_ms: int = 0


class RetryPolicy:
    """Deterministic retry policy with fixed (no-jitter) backoff."""

    def __init__(
        self,
        *,
        api_max_retries: int = 2,
        api_backoff_ms: list[int] | None = None,
        parse_max_retries: int = 1,
    ):
        if api_max_retries < 0 or parse_max_retries < 0:
            raise ValueError("retry limits must be >= 0")
        self.api_max_retries = api_max_retries
        # NOTE: when api_max_retries > len(api_backoff_ms), the LAST value is
        # reused for all remaining retries (deliberate, deterministic).
        self.api_backoff_ms = list(api_backoff_ms or [500, 1000])
        self.parse_max_retries = parse_max_retries

    def _max_retries(self, kind: FailureKind) -> int:
        if kind == FailureKind.MODEL_API_TRANSIENT:
            return self.api_max_retries
        if kind == FailureKind.MODEL_OUTPUT_INVALID:
            return self.parse_max_retries
        return 0

    def _backoff_ms(self, kind: FailureKind, retry_index: int) -> int:
        """Backoff before the given retry (1-based); fixed, deterministic."""
        if kind != FailureKind.MODEL_API_TRANSIENT:
            return 0  # parse repair retries immediately
        if 1 <= retry_index <= len(self.api_backoff_ms):
            return self.api_backoff_ms[retry_index - 1]
        return self.api_backoff_ms[-1] if self.api_backoff_ms else 0

    def decide(
        self,
        *,
        failure_kind: FailureKind,
        retry_index_for_kind: int,
        reliability_state: ReliabilityState,
        budget: ReliabilityBudget,
    ) -> RetryDecision:
        """Decide whether a failed attempt of `failure_kind` may be retried.

        `retry_index_for_kind` is the number of retries ALREADY executed for
        this failure kind within the current decision cycle (0-based). Retry
        allowances are independent per kind; the episode-level
        extra-model-call budget is global across kinds.
        """
        if failure_kind not in RETRYABLE_KINDS:
            return RetryDecision(
                retry=False,
                reason=f"failure kind {failure_kind.value} is not retryable "
                "(environment-side failures are handled by Recovery, Phase 1C)",
            )
        if episode_budget_exhausted(reliability_state, budget):
            return RetryDecision(
                retry=False,
                reason="episode extra-model-call budget exhausted",
            )
        if retry_index_for_kind >= self._max_retries(failure_kind):
            return RetryDecision(
                retry=False,
                reason="retry_exhausted",
            )
        return RetryDecision(
            retry=True,
            reason=failure_kind.value,
            backoff_ms=self._backoff_ms(failure_kind, retry_index_for_kind + 1),
        )
