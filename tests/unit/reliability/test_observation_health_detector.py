"""Unit tests for ObservationHealthDetector."""

from web_harness.core.models import EnvironmentStep, Observation
from web_harness.core.reliability import FailureKind, FailureSeverity
from web_harness.reliability.detectors.observation_health import (
    ObservationHealthDetector,
)

from .conftest import make_ctx


def test_no_signal_on_healthy_observation():
    ctx = make_ctx()
    assert ObservationHealthDetector().detect(ctx) == []


def test_signal_on_empty_url_and_content():
    step = EnvironmentStep(observation=Observation(goal="g", url=""))
    signals = ObservationHealthDetector().detect(make_ctx(env_step=step))
    assert len(signals) == 1
    s = signals[0]
    assert s.kind == FailureKind.OBSERVATION_INVALID
    assert s.severity == FailureSeverity.ERROR
    assert s.recoverable is True
    assert s.signature == "observation_invalid:url_and_content_empty"


def test_healthy_url_but_empty_axtree_is_not_invalid():
    # axtree may legitimately be empty on sparse pages; URL presence keeps it valid
    step = EnvironmentStep(
        observation=Observation(goal="g", url="http://fake.local/x", axtree=None)
    )
    assert ObservationHealthDetector().detect(make_ctx(env_step=step)) == []
