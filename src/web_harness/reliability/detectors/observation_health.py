"""Detector: the post-action observation is unusable (no URL and no content).

The adapter may also explicitly report an observation failure through the
observation's `truncated` + empty content combination; this detector treats a
missing URL together with missing axtree/dom as an invalid observation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from web_harness.core.reliability import FailureKind, FailureSeverity, FailureSignal

if TYPE_CHECKING:
    from web_harness.reliability.verifier import VerificationContext


class ObservationHealthDetector:
    source = "observation_health_detector"

    def detect(self, ctx: VerificationContext) -> list[FailureSignal]:
        obs = ctx.env_step.observation
        url_empty = not (obs.url or "").strip()
        content_empty = not ((obs.axtree or "").strip() or (obs.dom or "").strip())
        adapter_failed = bool(ctx.env_step.action_error) and "observation" in (
            ctx.env_step.action_error or ""
        ).lower()
        if not (url_empty and content_empty) and not adapter_failed:
            return []
        return [
            FailureSignal(
                kind=FailureKind.OBSERVATION_INVALID,
                severity=FailureSeverity.ERROR,
                source=self.source,
                signature="observation_invalid:url_and_content_empty",
                recoverable=True,
                evidence={
                    "url_empty": url_empty,
                    "content_empty": content_empty,
                    "adapter_reported": adapter_failed,
                },
            )
        ]
