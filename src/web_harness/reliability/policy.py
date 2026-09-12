"""FailurePolicyEngine: deterministic rule-based recovery decisions.

The policy maps verified FailureSignals to a PolicyAction. It never uses an
LLM and never retries browser actions: model-side failures stay with the
DecisionExecutor (Phase 1B); environment-side failures move to the
RecoveryManager (Phase 1C). REPLAN exists only as a future contract
(Phase 1D) and is never executed in Phase 1C.

Fixed rules (per the Phase 1C plan, no extra heuristics):

    TASK_FAILED         -> ABORT    (environment terminal is final)
    ACTION_ERROR        -> RECOVER  (REDECIDE_WITH_FEEDBACK + block 1 step)
    OBSERVATION_INVALID -> RECOVER  (WAIT_AND_REOBSERVE)
    LOOP_DETECTED       -> RECOVER  (BLOCK_REPEATED_ACTION)
    NO_PROGRESS alone   -> CONTINUE (single unchanged state is not a failure)
    recovery budget exhausted -> ABORT (RECOVERY_FAILED)
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from web_harness.core.reliability import (
    FailureKind,
    FailureSignal,
    RecoveryDirective,
    RecoveryKind,
    ReliabilityBudget,
    ReliabilityState,
)


class PolicyAction(StrEnum):
    CONTINUE = "continue"
    RETRY = "retry"  # reserved: only ever means existing Phase 1B model retry
    RECOVER = "recover"
    REPLAN = "replan"  # reserved for Phase 1D; never executed in Phase 1C
    ABORT = "abort"


class PolicyDecision(BaseModel):
    action: PolicyAction
    reason: str
    failure_kind: FailureKind | None = None
    directive: RecoveryDirective | None = None


class FailurePolicyEngine:
    """Deterministic, rule-based policy over verified failure signals."""

    def __init__(self, *, wait_ms: int = 500, block_steps: int = 1):
        self.wait_ms = wait_ms
        self.block_steps = block_steps

    def decide(
        self,
        *,
        signals: list[FailureSignal],
        failed_action: str | None,
        reliability_state: ReliabilityState,
        budget: ReliabilityBudget,
    ) -> PolicyDecision:
        kinds = {s.kind for s in signals}

        # environment terminal is final: no recovery, no reset
        if FailureKind.TASK_FAILED in kinds:
            return PolicyDecision(
                action=PolicyAction.ABORT,
                reason="task terminated without success; environment terminal "
                "is final",
                failure_kind=FailureKind.TASK_FAILED,
            )

        # recovery budget: no new recovery once exhausted (controlled abort)
        if reliability_state.recovery_count >= budget.max_recoveries_per_episode:
            return PolicyDecision(
                action=PolicyAction.ABORT,
                reason="recovery budget exhausted "
                f"({reliability_state.recovery_count} >= "
                f"{budget.max_recoveries_per_episode})",
                failure_kind=_priority_kind(kinds),
            )

        if FailureKind.ACTION_ERROR in kinds:
            return PolicyDecision(
                action=PolicyAction.RECOVER,
                reason="browser action failed; re-decide with feedback",
                failure_kind=FailureKind.ACTION_ERROR,
                directive=RecoveryDirective(
                    kind=RecoveryKind.REDECIDE_WITH_FEEDBACK,
                    reason="previous browser action failed",
                    feedback=(
                        "Previous browser action failed.\n\n"
                        f"Failure:\n{_failure_summary(signals, FailureKind.ACTION_ERROR)}\n\n"
                        "Do not assume the action succeeded.\n"
                        "Use the current page state and choose a different "
                        "valid action if appropriate.\n\n"
                        "Do not repeat the blocked action."
                    ),
                    blocked_actions=[failed_action] if failed_action else [],
                    expires_after_agent_steps=self.block_steps,
                ),
            )

        if FailureKind.OBSERVATION_INVALID in kinds:
            return PolicyDecision(
                action=PolicyAction.RECOVER,
                reason="observation unusable; wait and re-observe",
                failure_kind=FailureKind.OBSERVATION_INVALID,
                directive=RecoveryDirective(
                    kind=RecoveryKind.WAIT_AND_REOBSERVE,
                    reason="observation invalid; harness waits and re-observes",
                    wait_ms=self.wait_ms,
                ),
            )

        if FailureKind.LOOP_DETECTED in kinds:
            return PolicyDecision(
                action=PolicyAction.RECOVER,
                reason="same state/action transition repeated without progress",
                failure_kind=FailureKind.LOOP_DETECTED,
                directive=RecoveryDirective(
                    kind=RecoveryKind.BLOCK_REPEATED_ACTION,
                    reason="previous transition repeated without progress",
                    feedback=(
                        "The previous transition repeated without progress.\n\n"
                        "Do not repeat the blocked action.\n"
                        "Use the current page state and choose a different "
                        "valid strategy."
                    ),
                    blocked_actions=[failed_action] if failed_action else [],
                    expires_after_agent_steps=self.block_steps,
                ),
            )

        # single NO_PROGRESS (or clean pass): continue
        return PolicyDecision(
            action=PolicyAction.CONTINUE,
            reason="no recoverable environment-side failure",
            failure_kind=_priority_kind(kinds),
        )


def _failure_summary(signals: list[FailureSignal], kind: FailureKind) -> str:
    for signal in signals:
        if signal.kind == kind:
            evidence = signal.evidence or {}
            error = evidence.get("action_error") or signal.signature
            return str(error)[:200]
    return kind.value


def _priority_kind(kinds: set[FailureKind]) -> FailureKind | None:
    for kind in (
        FailureKind.TASK_FAILED,
        FailureKind.ACTION_ERROR,
        FailureKind.OBSERVATION_INVALID,
        FailureKind.LOOP_DETECTED,
        FailureKind.NO_PROGRESS,
    ):
        if kind in kinds:
            return kind
    return next(iter(kinds), None)
