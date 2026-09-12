"""Unit tests for the DefaultStepVerifier composite (Phase 1A coverage:
normal progress, action error, empty observation, single no-progress,
consecutive loop, task failure)."""

from web_harness.core.models import EnvironmentStep, TaskSpec
from web_harness.core.reliability import (
    FailureKind,
    ReliabilityState,
)
from web_harness.env.fake import make_fake_observation
from web_harness.reliability.verifier import (
    DefaultStepVerifier,
    VerificationStatus,
)

TASK = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=5)


def verify(pre, action, env_step, state=None):
    verifier = DefaultStepVerifier()
    return verifier.verify(
        task=TASK,
        pre_observation=pre,
        action=action,
        env_step=env_step,
        history=[],
        reliability_state=state or ReliabilityState(),
    )


def test_normal_progress_passes():
    pre = make_fake_observation(url="http://x/1")
    post = make_fake_observation(url="http://x/2")
    result = verify(pre, "click(bid='1')", EnvironmentStep(observation=post))
    assert result.status == VerificationStatus.PASS
    assert result.signals == []
    assert result.state_changed is True


def test_action_error_fails_verification():
    pre = make_fake_observation(url="http://x/1")
    post = make_fake_observation(url="http://x/2")
    step = EnvironmentStep(observation=post, action_error="TimeoutError: click")
    result = verify(pre, "click(bid='1')", step)
    assert result.status == VerificationStatus.FAIL
    kinds = {s.kind for s in result.signals}
    assert FailureKind.ACTION_ERROR in kinds


def test_empty_observation_fails_verification():
    pre = make_fake_observation(url="http://x/1")
    from web_harness.core.models import Observation

    step = EnvironmentStep(observation=Observation(goal="g", url=""))
    result = verify(pre, "click(bid='1')", step)
    assert result.status == VerificationStatus.FAIL
    assert FailureKind.OBSERVATION_INVALID in {s.kind for s in result.signals}


def test_single_no_progress_is_warning():
    same = make_fake_observation(url="http://x/same")
    result = verify(
        same, "click(bid='1')", EnvironmentStep(observation=same.model_copy(deep=True))
    )
    assert result.status == VerificationStatus.WARNING
    assert FailureKind.NO_PROGRESS in {s.kind for s in result.signals}


def test_consecutive_loop_fails():
    verifier = DefaultStepVerifier()
    state = ReliabilityState()
    same = make_fake_observation(url="http://x/stuck")

    # step 1: click from state S back to state S
    r1 = verifier.verify(
        task=TASK,
        pre_observation=same,
        action="click(bid='7')",
        env_step=EnvironmentStep(observation=same.model_copy(deep=True)),
        history=[],
        reliability_state=state,
    )
    # first identical transition: no-progress warning only
    assert r1.status == VerificationStatus.WARNING
    assert FailureKind.LOOP_DETECTED not in {s.kind for s in r1.signals}

    # step 2: identical transition again (same pre state, same action, same post)
    r2 = verifier.verify(
        task=TASK,
        pre_observation=same,
        action="click(bid='7')",
        env_step=EnvironmentStep(observation=same.model_copy(deep=True)),
        history=[],
        reliability_state=state,
    )
    assert r2.status == VerificationStatus.FAIL
    assert FailureKind.LOOP_DETECTED in {s.kind for s in r2.signals}


def test_task_failure_signal_on_terminated_without_reward():
    pre = make_fake_observation(url="http://x/1")
    post = make_fake_observation(url="http://x/2")
    step = EnvironmentStep(observation=post, reward=0.0, terminated=True)
    result = verify(pre, "click(bid='1')", step)
    assert FailureKind.TASK_FAILED in {s.kind for s in result.signals}
    assert result.status == VerificationStatus.FAIL


def test_state_updates_are_bounded_and_recorded():
    verifier = DefaultStepVerifier()
    state = ReliabilityState()
    for i in range(15):
        pre = make_fake_observation(url=f"http://x/{i}")
        post = make_fake_observation(url=f"http://x/{i + 1}")
        verifier.verify(
            task=TASK,
            pre_observation=pre,
            action=f"click(bid='{i}')",
            env_step=EnvironmentStep(observation=post),
            history=[],
            reliability_state=state,
        )
    assert len(state.recent_state_fingerprints) == 10
    assert len(state.recent_actions) == 10
    assert len(state.recent_transition_signatures) == 10
