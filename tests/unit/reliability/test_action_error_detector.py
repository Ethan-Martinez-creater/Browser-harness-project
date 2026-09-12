"""Unit tests for ActionErrorDetector."""

from web_harness.core.models import EnvironmentStep
from web_harness.core.reliability import FailureKind, FailureSeverity
from web_harness.reliability.detectors.action_error import ActionErrorDetector

from .conftest import make_ctx


def test_no_signal_on_clean_step():
    ctx = make_ctx()
    assert ActionErrorDetector().detect(ctx) == []


def test_signal_on_action_error():
    step = EnvironmentStep(
        observation=make_ctx().env_step.observation,
        action_error="TimeoutError: Locator.click: Timeout 500ms exceeded",
    )
    ctx = make_ctx(env_step=step)
    signals = ActionErrorDetector().detect(ctx)
    assert len(signals) == 1
    s = signals[0]
    assert s.kind == FailureKind.ACTION_ERROR
    assert s.severity == FailureSeverity.ERROR
    assert s.recoverable is True
    assert s.retryable is False  # never blind-retry a browser action
    assert s.signature.startswith("action_error:")
    assert s.evidence["action"] == ctx.action


def test_signature_is_normalized_not_raw():
    err_a = "TimeoutError: click Timeout 500ms exceeded"
    err_b = "timeouterror:   CLICK   timeout 900ms exceeded"
    sigs = set()
    for err in (err_a, err_b):
        step = EnvironmentStep(
            observation=make_ctx().env_step.observation, action_error=err
        )
        ctx = make_ctx(env_step=step)
        (signal,) = ActionErrorDetector().detect(ctx)
        sigs.add(signal.signature)
    assert len(sigs) == 1  # numbers/spacing normalized away
    assert "500" not in next(iter(sigs))
