"""Detector: browser action reported an environment-side error.

An action error is a step-level failure: it never blind-retries the browser
action (retryable=false) because the action may already have had side
effects; it is handled by Recovery (Phase 1C).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from web_harness.core.reliability import FailureKind, FailureSeverity, FailureSignal
from web_harness.reliability.fingerprint import (
    extract_action_type,
    normalize_error_signature,
)

if TYPE_CHECKING:
    from web_harness.reliability.verifier import VerificationContext


class ActionErrorDetector:
    source = "action_error_detector"

    def detect(self, ctx: VerificationContext) -> list[FailureSignal]:
        error = ctx.env_step.action_error
        if not error:
            return []
        action_type = extract_action_type(ctx.action)
        return [
            FailureSignal(
                kind=FailureKind.ACTION_ERROR,
                severity=FailureSeverity.ERROR,
                source=self.source,
                signature=(
                    f"action_error:{action_type}:{normalize_error_signature(error)}"
                ),
                retryable=False,  # browser actions may have had side effects
                recoverable=True,
                evidence={
                    "action": ctx.action,
                    "action_type": action_type,
                    "action_error": error[:500],
                },
            )
        ]
