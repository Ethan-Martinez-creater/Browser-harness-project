"""Unit tests for NoProgressDetector (single no-progress = WARNING)."""

from web_harness.core.models import EnvironmentStep
from web_harness.core.reliability import FailureKind, FailureSeverity
from web_harness.env.fake import make_fake_observation
from web_harness.reliability.detectors.progress import NoProgressDetector

from .conftest import make_ctx


def test_progress_is_clean():
    # pre != post by default in make_ctx
    assert NoProgressDetector().detect(make_ctx()) == []


def test_single_no_progress_is_warning():
    same = make_fake_observation(url="http://fake.local/same")
    step = EnvironmentStep(observation=same.model_copy(deep=True))
    ctx = make_ctx(pre=same, post=same, env_step=step)
    signals = NoProgressDetector().detect(ctx)
    assert len(signals) == 1
    s = signals[0]
    assert s.kind == FailureKind.NO_PROGRESS
    assert s.severity == FailureSeverity.WARNING  # first occurrence
    assert s.recoverable is True


def test_repeated_no_progress_escalates_to_error():
    same = make_fake_observation(url="http://fake.local/same")
    step = EnvironmentStep(observation=same.model_copy(deep=True))
    ctx = make_ctx(pre=same, post=same, env_step=step)
    # fingerprint already seen in recent history => repeated no-progress
    ctx.reliability_state.recent_state_fingerprints.append(
        ctx.pre_fingerprint
    )
    (signal,) = NoProgressDetector().detect(ctx)
    assert signal.severity == FailureSeverity.ERROR


def test_terminated_step_is_not_no_progress():
    same = make_fake_observation(url="http://fake.local/same")
    step = EnvironmentStep(observation=same.model_copy(deep=True), terminated=True)
    ctx = make_ctx(pre=same, post=same, env_step=step)
    assert NoProgressDetector().detect(ctx) == []


def test_action_error_step_is_not_no_progress():
    same = make_fake_observation(url="http://fake.local/same")
    step = EnvironmentStep(
        observation=same.model_copy(deep=True), action_error="boom"
    )
    ctx = make_ctx(pre=same, post=same, env_step=step)
    assert NoProgressDetector().detect(ctx) == []


def test_signature_distinguishes_states():
    """No-progress on different states must produce different signatures (R1)."""
    s1 = make_fake_observation(url="http://fake.local/stuck1")
    s2 = make_fake_observation(url="http://fake.local/stuck2")
    sig1 = NoProgressDetector().detect(
        make_ctx(pre=s1, post=s1, env_step=EnvironmentStep(observation=s1.model_copy(deep=True)))
    )[0].signature
    sig2 = NoProgressDetector().detect(
        make_ctx(pre=s2, post=s2, env_step=EnvironmentStep(observation=s2.model_copy(deep=True)))
    )[0].signature
    assert sig1.startswith("no_progress:")
    assert sig2.startswith("no_progress:")
    assert sig1 != sig2


def test_rewarded_step_is_not_no_progress():
    same = make_fake_observation(url="http://fake.local/same")
    step = EnvironmentStep(observation=same.model_copy(deep=True), reward=1.0)
    ctx = make_ctx(pre=same, post=same, env_step=step)
    assert NoProgressDetector().detect(ctx) == []
