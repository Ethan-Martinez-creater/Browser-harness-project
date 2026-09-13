"""Step verification engine.

Runs deterministic detectors over (pre observation, action, environment
result) and produces a VerificationResult with FailureSignals. In Phase 1A
the engine runs in SHADOW mode: its output is traced and counted but does not
change the control flow. State updates (fingerprints, transition history,
failure counts) are maintained here so detectors stay stateless.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from web_harness.core.models import EnvironmentStep, Observation, StepRecord, TaskSpec
from web_harness.core.reliability import FailureSeverity, FailureSignal, ReliabilityState
from web_harness.reliability.fingerprint import (
    fingerprint_of,
    transition_signature,
)

MAX_RECENT_ITEMS = 10


class VerificationStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class VerificationResult(BaseModel):
    status: VerificationStatus
    signals: list[FailureSignal]
    pre_fingerprint: str | None
    post_fingerprint: str | None
    state_changed: bool | None


class VerificationContext(BaseModel):
    """Everything a detector may look at for one verified step."""

    task: TaskSpec
    pre_observation: Observation
    action: str
    env_step: EnvironmentStep
    history: list[StepRecord]
    reliability_state: ReliabilityState
    pre_fingerprint: str | None = None
    post_fingerprint: str | None = None
    transition_signature: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class StepVerifier(Protocol):
    def verify(
        self,
        *,
        task: TaskSpec,
        pre_observation: Observation,
        action: str,
        env_step: EnvironmentStep,
        history: list[StepRecord],
        reliability_state: ReliabilityState,
    ) -> VerificationResult: ...


class DefaultStepVerifier:
    """Detector-composite verifier with deterministic, stateless detectors."""

    def __init__(
        self,
        *,
        detectors: list | None = None,
        loop_consecutive_threshold: int = 2,
        detect_no_progress: bool = True,
        detect_loop: bool = True,
    ):
        if detectors is None:
            from web_harness.reliability.detectors.action_error import (
                ActionErrorDetector,
            )
            from web_harness.reliability.detectors.loop import LoopDetector
            from web_harness.reliability.detectors.observation_health import (
                ObservationHealthDetector,
            )
            from web_harness.reliability.detectors.progress import NoProgressDetector
            from web_harness.reliability.detectors.task_failure import (
                TaskFailureDetector,
            )

            custom_composition = False
            detectors = [
                ActionErrorDetector(),
                ObservationHealthDetector(),
                NoProgressDetector() if detect_no_progress else None,
                LoopDetector(consecutive_threshold=loop_consecutive_threshold)
                if detect_loop
                else None,
                TaskFailureDetector(),
            ]
        else:
            custom_composition = True
        self.detectors = [d for d in detectors if d is not None]
        # whether a custom detector composition was injected: such a verifier
        # must NOT publish a spec that looks like the rebuildable standard
        # composition (Phase 2A1 closure: offline replay must not guess)
        self._custom_detector_composition = custom_composition
        # effective behavior (Phase 2A1): persisted into the run manifest so
        # offline semantic replay can rebuild the verifier the live run used
        self.detect_no_progress = detect_no_progress
        self.detect_loop = detect_loop
        self.loop_consecutive_threshold = loop_consecutive_threshold

    def spec(self) -> dict:
        """Recorded effective verifier specification for trace provenance.

        Only the standard built-in detector composition publishes a
        rebuildable spec; a custom composition is marked unsupported so
        semantic replay reports it instead of silently rebuilding it as the
        standard verifier.
        """
        if self._custom_detector_composition:
            return {
                "implementation": "custom_detector_composition",
                "rebuildable": False,
            }
        return {
            "implementation": "DefaultStepVerifier",
            "detect_no_progress": self.detect_no_progress,
            "detect_loop": self.detect_loop,
            "loop_consecutive_threshold": self.loop_consecutive_threshold,
        }

    def verify(
        self,
        *,
        task: TaskSpec,
        pre_observation: Observation,
        action: str,
        env_step: EnvironmentStep,
        history: list[StepRecord],
        reliability_state: ReliabilityState,
    ) -> VerificationResult:
        pre_fp = fingerprint_of(pre_observation)
        post_fp = fingerprint_of(env_step.observation)
        trans_sig = transition_signature(pre_fp, action, post_fp)
        ctx = VerificationContext(
            task=task,
            pre_observation=pre_observation,
            action=action,
            env_step=env_step,
            history=history,
            reliability_state=reliability_state,
            pre_fingerprint=pre_fp,
            post_fingerprint=post_fp,
            transition_signature=trans_sig,
        )

        signals = []
        for detector in self.detectors:
            signals.extend(detector.detect(ctx))

        # update reliability bookkeeping (bounded recent history)
        state = reliability_state
        state.recent_state_fingerprints.append(post_fp)
        del state.recent_state_fingerprints[:-MAX_RECENT_ITEMS]
        state.recent_actions.append(action)
        del state.recent_actions[:-MAX_RECENT_ITEMS]
        state.recent_transition_signatures.append(trans_sig)
        del state.recent_transition_signatures[:-MAX_RECENT_ITEMS]
        for signal in signals:
            state.record_failure(signal)
        # consecutive failure tracking: reset on a clean step
        if signals:
            state.consecutive_failure_count += 1
        else:
            state.consecutive_failure_count = 0

        if any(s.severity == FailureSeverity.ERROR for s in signals):
            status = VerificationStatus.FAIL
        elif any(s.severity == FailureSeverity.WARNING for s in signals):
            status = VerificationStatus.WARNING
        else:
            status = VerificationStatus.PASS

        return VerificationResult(
            status=status,
            signals=signals,
            pre_fingerprint=pre_fp,
            post_fingerprint=post_fp,
            state_changed=pre_fp != post_fp,
        )
