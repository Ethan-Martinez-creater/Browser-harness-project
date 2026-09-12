"""Detector: the state did not change and nothing was gained.

Fires when pre-fingerprint == post-fingerprint, reward == 0, there was no
action error and the task did not terminate. The first occurrence is a
WARNING; a repeated no-progress state (already seen in recent fingerprints)
escalates to ERROR so the policy engine can react.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from web_harness.core.reliability import FailureKind, FailureSeverity, FailureSignal

if TYPE_CHECKING:
    from web_harness.reliability.verifier import VerificationContext


class NoProgressDetector:
    source = "no_progress_detector"

    def detect(self, ctx: VerificationContext) -> list[FailureSignal]:
        step = ctx.env_step
        no_change = (
            ctx.pre_fingerprint is not None
            and ctx.post_fingerprint is not None
            and ctx.pre_fingerprint == ctx.post_fingerprint
        )
        if not (
            no_change
            and step.reward == 0
            and not step.action_error
            and not step.terminated
            and not step.truncated
        ):
            return []
        seen_before = (
            ctx.reliability_state.recent_state_fingerprints.count(
                ctx.pre_fingerprint or ""
            )
            > 0
        )
        return [
            FailureSignal(
                kind=FailureKind.NO_PROGRESS,
                severity=FailureSeverity.ERROR if seen_before else FailureSeverity.WARNING,
                source=self.source,
                signature="no_progress:state_unchanged_reward_zero",
                recoverable=True,
                evidence={
                    "fingerprint": ctx.post_fingerprint,
                    "seen_before": seen_before,
                },
            )
        ]
