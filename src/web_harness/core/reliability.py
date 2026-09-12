"""Cross-layer reliability data contracts (Phase 1).

These models describe HOW the harness understands failures (semantic layer),
complementary to `ErrorType` which describes concrete technical errors.
Detection, policy, retry, recovery and replanning all share these contracts;
Phase 1A only populates and traces them (shadow mode) without changing the
control flow.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FailureKind(StrEnum):
    """Semantic classification of a failure (what happened)."""

    MODEL_API_TRANSIENT = "MODEL_API_TRANSIENT"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    ACTION_ERROR = "ACTION_ERROR"
    OBSERVATION_INVALID = "OBSERVATION_INVALID"
    NO_PROGRESS = "NO_PROGRESS"
    LOOP_DETECTED = "LOOP_DETECTED"
    TASK_FAILED = "TASK_FAILED"
    UNKNOWN = "UNKNOWN"


class FailureSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FailureSignal(BaseModel):
    """One detected failure, produced by a detector during verification.

    `signature` is deterministic and normalized (never a raw exception
    string) so repeated failures of the same kind can be counted.
    """

    kind: FailureKind
    severity: FailureSeverity
    source: str
    signature: str
    retryable: bool = False
    recoverable: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)


class RecoveryKind(StrEnum):
    """Recovery strategies (data contract only in Phase 1A; behavior in 1C)."""

    REDECIDE_WITH_FEEDBACK = "redecide_with_feedback"
    BLOCK_REPEATED_ACTION = "block_repeated_action"
    WAIT_AND_REOBSERVE = "wait_and_reobserve"


class RecoveryDirective(BaseModel):
    """Data contract for a recovery directive (executed in Phase 1C)."""

    kind: RecoveryKind
    reason: str
    feedback: str | None = None
    blocked_actions: list[str] = Field(default_factory=list)
    wait_ms: int | None = None
    expires_after_agent_steps: int = 1


class RecoveryPlan(BaseModel):
    """Data contract for a structured replan (produced in Phase 1D)."""

    diagnosis: str
    immediate_subgoal: str
    strategy_steps: list[str] = Field(default_factory=list)
    avoid_actions: list[str] = Field(default_factory=list)
    horizon_steps: int = 1
    created_at_step: int = 0


class ReliabilityState(BaseModel):
    """Reliability bookkeeping for one episode, kept inside RunState."""

    recent_state_fingerprints: list[str] = Field(default_factory=list)
    recent_actions: list[str] = Field(default_factory=list)
    recent_transition_signatures: list[str] = Field(default_factory=list)
    failure_counts: dict[str, int] = Field(default_factory=dict)
    consecutive_failure_count: int = 0
    retry_count: int = 0
    recovery_count: int = 0
    replan_count: int = 0
    extra_model_calls: int = 0
    active_recovery_directive: RecoveryDirective | None = None
    active_recovery_plan: RecoveryPlan | None = None
    # Phase 1C: local recovery success evaluation (bounded, deterministic).
    # Each entry: {"signature", "steps_observed", "pre_fingerprint"}.
    pending_recovery_evaluations: list[dict] = Field(default_factory=list)
    last_recovery_failure_signature: str | None = None

    def record_failure(self, signal: FailureSignal) -> None:
        self.failure_counts[signal.signature] = (
            self.failure_counts.get(signal.signature, 0) + 1
        )

    @property
    def total_failure_signals(self) -> int:
        return sum(self.failure_counts.values())


class ReliabilityBudget(BaseModel):
    """Hard limits for reliability mechanisms; exceeding means controlled
    abort, never an unbounded loop. All mechanisms must respect this budget."""

    max_model_api_retries_per_call: int = 2
    max_parse_retries_per_call: int = 1
    max_recoveries_per_episode: int = 3
    max_replans_per_episode: int = 1
    max_extra_model_calls_per_episode: int = 6


def default_budget_from_config(reliability_cfg: dict[str, Any]) -> ReliabilityBudget:
    """Build the budget from the `reliability:` config section."""
    retry_cfg = reliability_cfg.get("retry") or {}
    recovery_cfg = reliability_cfg.get("recovery") or {}
    replan_cfg = reliability_cfg.get("replanning") or {}
    budget_cfg = reliability_cfg.get("budget") or {}
    return ReliabilityBudget(
        max_model_api_retries_per_call=int(
            retry_cfg.get("model_api", {}).get("max_retries", 2)
        ),
        max_parse_retries_per_call=int(
            retry_cfg.get("model_output", {}).get("max_retries", 1)
        ),
        max_recoveries_per_episode=int(
            recovery_cfg.get("max_recoveries_per_episode", 3)
        ),
        max_replans_per_episode=int(replan_cfg.get("max_replans_per_episode", 1)),
        max_extra_model_calls_per_episode=int(
            budget_cfg.get("max_extra_model_calls_per_episode", 6)
        ),
    )
