"""Detector: the environment terminated the task without success.

Records TASK_FAILED for observability. It must NOT cause a task reset or any
continuation logic: episode-level termination stays with the runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from web_harness.core.reliability import FailureKind, FailureSeverity, FailureSignal

if TYPE_CHECKING:
    from web_harness.reliability.verifier import VerificationContext


class TaskFailureDetector:
    source = "task_failure_detector"

    def detect(self, ctx: VerificationContext) -> list[FailureSignal]:
        step = ctx.env_step
        if not (step.terminated and step.reward <= 0):
            return []
        return [
            FailureSignal(
                kind=FailureKind.TASK_FAILED,
                severity=FailureSeverity.ERROR,
                source=self.source,
                signature=f"task_failed:reward_{step.reward:g}",
                recoverable=False,
                retryable=False,
                evidence={"reward": step.reward, "truncated": step.truncated},
            )
        ]
