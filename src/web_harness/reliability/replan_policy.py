"""ReplanTriggerPolicy: deterministic escalation decision (Phase 1D).

Sits AFTER the Phase 1C FailurePolicyEngine and answers exactly one question:
"the current failure was going to be RECOVERED — is it severe enough to
escalate to a Replan intervention instead?" It never uses an LLM and never
changes the base policy's CONTINUE/ABORT decisions.

Fixed trigger rules (per the Phase 1D plan):

    base action must be RECOVER
    failure kind must be ACTION_ERROR or LOOP_DETECTED
    replan budget must have room (replan_count < max_replans_per_episode)

    D1 REPEATED_LOOP_AFTER_RECOVERY:
        LOOP_DETECTED and the current failure signature equals the
        signature of the most recent FAILED recovery
    D2 RECOVERY_FAILURE_STREAK:
        consecutive REAL recovery failures (FAILED outcomes only;
        UNRESOLVED is neutral) >= recovery_failures_before_replan

Everything else stays with the existing Phase 1C Recovery.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from web_harness.core.reliability import (
    FailureKind,
    ReliabilityBudget,
    ReliabilityState,
)
from web_harness.reliability.policy import PolicyAction


class ReplanTriggerReason(StrEnum):
    REPEATED_LOOP_AFTER_RECOVERY = "repeated_loop_after_recovery"
    RECOVERY_FAILURE_STREAK = "recovery_failure_streak"


class ReplanTriggerDecision(BaseModel):
    trigger: bool
    reason: ReplanTriggerReason | None = None
    failure_kind: FailureKind | None = None
    failure_signature: str | None = None


_REPLANABLE_KINDS = (FailureKind.ACTION_ERROR, FailureKind.LOOP_DETECTED)


class ReplanTriggerPolicy:
    """Deterministic escalation policy over verified failure signals."""

    def decide(
        self,
        *,
        base_action: PolicyAction,
        failure_kind: FailureKind | None,
        failure_signature: str | None,
        reliability_state: ReliabilityState,
        budget: ReliabilityBudget,
    ) -> ReplanTriggerDecision:
        no_trigger = ReplanTriggerDecision(
            trigger=False, failure_kind=failure_kind,
            failure_signature=failure_signature,
        )

        # escalation only replaces an intended RECOVER; PASS/ABORT/CONTINUE
        # decisions are never upgraded or downgraded
        if base_action != PolicyAction.RECOVER:
            return no_trigger
        # observation-invalid failures recover the observation first; model
        # side failures belong to the DecisionExecutor (never replan here)
        if failure_kind not in _REPLANABLE_KINDS:
            return no_trigger
        # replan budget: no NEW plan once exhausted (fallback stays RECOVER)
        if reliability_state.replan_count >= budget.max_replans_per_episode:
            return no_trigger

        # D1: the same loop that the last FAILED recovery failed to break
        if (
            failure_kind == FailureKind.LOOP_DETECTED
            and reliability_state.last_failed_recovery_signature
            and failure_signature == reliability_state.last_failed_recovery_signature
        ):
            return ReplanTriggerDecision(
                trigger=True,
                reason=ReplanTriggerReason.REPEATED_LOOP_AFTER_RECOVERY,
                failure_kind=failure_kind,
                failure_signature=failure_signature,
            )

        # D2: a streak of consecutive REAL recovery failures
        if (
            reliability_state.consecutive_recovery_failures
            >= budget.recovery_failures_before_replan
        ):
            return ReplanTriggerDecision(
                trigger=True,
                reason=ReplanTriggerReason.RECOVERY_FAILURE_STREAK,
                failure_kind=failure_kind,
                failure_signature=failure_signature,
            )

        return no_trigger
