"""Phase 1A compatibility regression (Phase 1B contract).

With retry disabled, Phase 1A behavior must be byte-identical: one model call
per step, no retry events, no extra tokens. With retry enabled but no faults,
normal episodes must behave exactly like Phase 1A as well.
"""

from pathlib import Path

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.models import RunStatus, TaskSpec
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.reliability.verifier import DefaultStepVerifier
from web_harness.runtime.decision_executor import DecisionExecutor
from web_harness.runtime.episode_runner import EpisodeRunner

TASK = TaskSpec(benchmark="fake", task_id="compat", seed=0, max_steps=10)
SCRIPT = [
    {"observation": make_fake_observation(url="http://s.local/1")},
    {
        "observation": make_fake_observation(url="http://s.local/2"),
        "reward": 1.0,
        "terminated": True,
    },
]
ACTIONS = ["click(bid='1')", "noop()"]


def run_episode(tmp_path: Path, run_id: str, *, verifier=None, executor=None):
    env = FakeEnvironment(
        reset_observation=make_fake_observation(url="http://s.local/0"),
        script=SCRIPT,
    )
    model = MockModelAdapter(ACTIONS)
    agent = BaselineAgent(model_adapter=model)
    runner = EpisodeRunner(
        agent=agent, env=env, trace_root=tmp_path,
        verifier=verifier, decision_executor=executor,
    )
    return runner.run(TASK, run_id=run_id), env, model


def test_phase1a_behavior_preserved_with_executor_default(tmp_path):
    """Default (retry-disabled) executor: identical to Phase 1A."""
    result, env, model = run_episode(tmp_path, "compat", verifier=DefaultStepVerifier())

    assert model.call_count == 2
    assert env.step_calls == 2
    assert env.executed_actions == ACTIONS
    assert result.status == RunStatus.SUCCESS
    assert result.retry_count == 0
    assert result.extra_model_calls == 0
    assert result.retry_input_tokens == 0
    assert result.retry_output_tokens == 0
    steps = TraceRecorder.read_steps(Path(result.trace_path))
    assert len(steps) == 2
    assert all(s.retry_count == 0 for s in steps)
    events = TraceRecorder.read_events(Path(result.trace_path))
    assert [e for e in events if e.event_type == "retry"] == []
    # verification events unchanged
    assert len([e for e in events if e.event_type == "verification"]) == 2


def test_retry_enabled_without_faults_changes_nothing(tmp_path):
    from web_harness.reliability.retry import RetryPolicy

    executor = DecisionExecutor(
        retry_policy=RetryPolicy(),
        sleep=lambda _s: None,
    )
    result, env, model = run_episode(
        tmp_path, "compat-retry", verifier=DefaultStepVerifier(), executor=executor
    )
    assert model.call_count == 2
    assert env.step_calls == 2
    assert env.executed_actions == ACTIONS
    assert result.status == RunStatus.SUCCESS
    assert result.retry_count == 0
    assert result.extra_model_calls == 0
    events = TraceRecorder.read_events(Path(result.trace_path))
    assert [e for e in events if e.event_type == "retry"] == []
