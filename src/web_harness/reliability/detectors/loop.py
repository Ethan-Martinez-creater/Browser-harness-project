"""Detector: the same state -> action -> state transition repeats.

A loop is only declared when the identical transition signature appears
consecutively >= `consecutive_threshold` times (default 2). Identical actions
alone are NOT a loop: the same click on a state that changed is progress.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from web_harness.core.reliability import FailureKind, FailureSeverity, FailureSignal

if TYPE_CHECKING:
    from web_harness.reliability.verifier import VerificationContext


class LoopDetector:
    source = "loop_detector"

    def __init__(self, consecutive_threshold: int = 2):
        if consecutive_threshold < 2:
            raise ValueError("consecutive_threshold must be >= 2")
        self.consecutive_threshold = consecutive_threshold

    def detect(self, ctx: VerificationContext) -> list[FailureSignal]:
        recent = ctx.reliability_state.recent_transition_signatures
        current = ctx.transition_signature
        if not current:
            return []
        # consecutive occurrences of the same transition signature
        streak = 0
        for sig in reversed(recent):
            if sig != current:
                break
            streak += 1
        # current transition is appended after verification; count it too
        streak += 1
        if streak < self.consecutive_threshold:
            return []
        return [
            FailureSignal(
                kind=FailureKind.LOOP_DETECTED,
                severity=FailureSeverity.ERROR,
                source=self.source,
                signature=f"loop:{current}",
                recoverable=True,
                evidence={
                    "transition_signature": current,
                    "consecutive_occurrences": streak,
                },
            )
        ]
