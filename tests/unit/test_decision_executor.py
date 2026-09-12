"""Unit tests for DecisionExecutor + the six deterministic fault scenarios
(B1-B6) at the episode level, plus fault-injection adapter tests."""

from pathlib import Path

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.errors import ErrorType
from web_harness.core.models import Observation, RunStatus, TaskSpec
from web_harness.core.reliability import ReliabilityBudget, ReliabilityState
from web_harness.env.action_contract import ActionContract, ActionSpec
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.evaluation.fault_injection import FaultInjectingModelAdapter
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.reliability.retry import RetryPolicy
from web_harness.runtime.decision_executor import DecisionExecutor
from web_harness.runtime.episode_runner import EpisodeRunner

TASK = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=5)
CONTRACT = ActionContract(
    benchmark="fake",
    actions=[ActionSpec(name="click", signature="click(bid: str)", description="click"),
             ActionSpec(name="noop", signature="noop()", description="wait")],
)
OBS = Observation(goal="g", url="http://fake.local/", axtree="[1] button 'Go'")
SUCCESS_SCRIPT = [{"reward": 1.0, "terminated": True}]


def make_executor(max_api_retries=2, budget_extra=6, enabled=True) -> DecisionExecutor:
    return DecisionExecutor(
        retry_policy=RetryPolicy(api_max_retries=max_api_retries) if enabled else None,
        budget=ReliabilityBudget(max_extra_model_calls_per_episode=budget_extra),
        sleep=lambda _s: None,  # deterministic tests: no real waiting
    )


def make_agent(api_error_calls=(), parse_error_calls=(), actions=("click(bid='1')",)):
    model = MockModelAdapter(list(actions))
    agent = BaselineAgent(
        model_adapter=FaultInjectingModelAdapter(
            model,
            api_error_on_calls=set(api_error_calls),
            parse_error_on_calls=set(parse_error_calls),
        )
    )
    return agent, model


def run_episode(tmp_path, *, api_error_calls=(), parse_error_calls=(),
                max_api_retries=2, budget_extra=6, retry_enabled=True,
                script=None, run_id="r"):
    agent, wrapped = make_agent(api_error_calls, parse_error_calls)
    env = FakeEnvironment(
        reset_observation=make_fake_observation(url="http://fake.local/0"),
        script=script if script is not None else SUCCESS_SCRIPT,
    )
    runner = EpisodeRunner(
        agent=agent,
        env=env,
        trace_root=tmp_path,
        decision_executor=make_executor(max_api_retries, budget_extra, retry_enabled),
    )
    return runner.run(TASK, run_id=run_id), env, wrapped


# -- executor unit behavior ----------------------------------------------------


def test_retry_disabled_single_call():
    agent, _ = make_agent(api_error_calls=(0,))
    result = make_executor(enabled=False).execute(
        agent=agent, task=TASK, observation=OBS, history=[],
        action_contract=CONTRACT, reliability_state=ReliabilityState(),
    )
    assert not result.success
    assert result.attempts == 1
    assert result.retry_count == 0
    assert result.terminal_error_type == ErrorType.MODEL_API_ERROR


def test_parse_error_tokens_accounted_on_failure():
    # retry disabled: the injected parse failure is terminal, and its usage
    # must still be accounted (never zero)
    agent, _ = make_agent(parse_error_calls=(0,))
    result = make_executor(enabled=False).execute(
        agent=agent, task=TASK, observation=OBS, history=[],
        action_contract=CONTRACT, reliability_state=ReliabilityState(),
    )
    assert not result.success
    assert result.terminal_error_type == ErrorType.MODEL_OUTPUT_PARSE_ERROR
    assert result.total_input_tokens == FaultInjectingModelAdapter.PARSE_FAIL_INPUT_TOKENS
    assert result.total_output_tokens == FaultInjectingModelAdapter.PARSE_FAIL_OUTPUT_TOKENS


def test_state_counters_track_retries():
    agent, _ = make_agent(api_error_calls=(0,))
    state = ReliabilityState()
    result = make_executor().execute(
        agent=agent, task=TASK, observation=OBS, history=[],
        action_contract=CONTRACT, reliability_state=state,
    )
    assert result.success
    assert state.retry_count == 1
    assert state.extra_model_calls == 1


def test_repair_retry_reaches_the_model_twice():
    agent, wrapped = make_agent(parse_error_calls=(0,))
    result = make_executor().execute(
        agent=agent, task=TASK, observation=OBS, history=[],
        action_contract=CONTRACT, reliability_state=ReliabilityState(),
    )
    assert result.success
    assert result.attempts == 2
    assert wrapped.call_count == 1  # only the non-injected call delegated


# -- episode-level fault scenarios (B1-B6) --------------------------------------


def test_b1_api_error_once_then_success(tmp_path):
    result, env, _ = run_episode(tmp_path, api_error_calls=(0,), run_id="b1")
    assert result.status == RunStatus.SUCCESS
    assert result.retry_count == 1
    assert result.extra_model_calls == 1
    assert result.num_steps == 1
    assert env.step_calls == 1
    # one agent action only: retry never touched the environment
    assert env.executed_actions == ["click(bid='1')"]
    events = TraceRecorder.read_events(Path(result.trace_path))
    retry_events = [e for e in events if e.event_type == "retry"]
    assert [e.attempt_index for e in retry_events if e.outcome == "retry"] == [1]
    assert any(e.outcome == "retry_succeeded" for e in retry_events)


def test_b2_two_api_errors_then_success(tmp_path):
    result, env, _ = run_episode(tmp_path, api_error_calls=(0, 1), run_id="b2")
    assert result.status == RunStatus.SUCCESS
    assert result.retry_count == 2
    assert result.extra_model_calls == 2
    assert env.step_calls == 1
    retry_events = [
        e for e in TraceRecorder.read_events(Path(result.trace_path))
        if e.event_type == "retry" and e.outcome == "retry"
    ]
    assert [e.attempt_index for e in retry_events] == [1, 2]


def test_b3_persistent_api_errors_exhaust_retries(tmp_path):
    result, env, _ = run_episode(tmp_path, api_error_calls=(0, 1, 2), run_id="b3")
    assert result.status == RunStatus.ERROR
    assert result.error_type == ErrorType.RETRY_EXHAUSTED
    # zero browser actions after exhaustion
    assert env.step_calls == 0
    assert result.retry_count == 2
    assert result.retry_exhausted_count == 1
    events = TraceRecorder.read_events(Path(result.trace_path))
    assert any(e.outcome == "exhausted" for e in events)
    steps = TraceRecorder.read_steps(Path(result.trace_path))
    assert len(steps) == 1 and steps[0].action is None


def test_b4_parse_repair_retry_success(tmp_path):
    result, env, _ = run_episode(tmp_path, parse_error_calls=(0,), run_id="b4")
    assert result.status == RunStatus.SUCCESS
    assert result.retry_count == 1
    assert result.num_steps == 1
    assert env.step_calls == 1
    # failed attempt tokens retained in episode totals (never zero): the
    # parse failure happened on the initial attempt, so its usage lands in
    # the episode totals, while retry tokens cover the repair attempt
    assert result.input_tokens >= FaultInjectingModelAdapter.PARSE_FAIL_INPUT_TOKENS
    assert result.output_tokens >= FaultInjectingModelAdapter.PARSE_FAIL_OUTPUT_TOKENS
    assert result.input_tokens >= result.retry_input_tokens
    assert result.retry_success_count == 1


def test_b5_retry_disabled_no_retry(tmp_path):
    result, env, _ = run_episode(
        tmp_path, api_error_calls=(0,), retry_enabled=False, run_id="b5"
    )
    assert result.status == RunStatus.ERROR
    assert result.error_type == ErrorType.MODEL_API_ERROR
    assert result.retry_count == 0
    assert env.step_calls == 0
    events = TraceRecorder.read_events(Path(result.trace_path))
    assert [e for e in events if e.event_type == "retry"] == []


def test_b6_extra_model_call_budget_exhausted(tmp_path):
    # budget allows 1 extra call; the faults need 2 retries
    result, env, _ = run_episode(
        tmp_path, api_error_calls=(0, 1, 2), budget_extra=1, run_id="b6"
    )
    assert result.status == RunStatus.ERROR
    assert result.error_type == ErrorType.BUDGET_EXCEEDED
    assert result.extra_model_calls == 1
    assert env.step_calls == 0
    steps = TraceRecorder.read_steps(Path(result.trace_path))
    assert steps[0].budget_exhausted is True
