"""Shared helpers for reliability detector tests."""

from web_harness.core.models import EnvironmentStep, Observation, StepRecord, TaskSpec
from web_harness.core.reliability import ReliabilityState
from web_harness.env.fake import make_fake_observation
from web_harness.reliability.fingerprint import (
    fingerprint_of,
    transition_signature,
)
from web_harness.reliability.verifier import VerificationContext

TASK = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=5)


def make_ctx(
    *,
    pre: Observation | None = None,
    post: Observation | None = None,
    action: str = "click(bid='1')",
    env_step: EnvironmentStep | None = None,
    reliability_state: ReliabilityState | None = None,
    history: list[StepRecord] | None = None,
) -> VerificationContext:
    pre = pre or make_fake_observation(url="http://fake.local/pre")
    post = post or make_fake_observation(url="http://fake.local/post")
    env_step = env_step or EnvironmentStep(
        observation=post, reward=0.0, terminated=False, truncated=False
    )
    state = reliability_state or ReliabilityState()
    pre_fp = fingerprint_of(pre)
    post_fp = fingerprint_of(post)
    return VerificationContext(
        task=TASK,
        pre_observation=pre,
        action=action,
        env_step=env_step,
        history=history or [],
        reliability_state=state,
        pre_fingerprint=pre_fp,
        post_fingerprint=post_fp,
        transition_signature=transition_signature(pre_fp, action, post_fp),
    )
