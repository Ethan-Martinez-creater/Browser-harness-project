"""Phase 1B remediation tests: mixed-failure retries (B1/M1-M3), Agent
Protocol conformance (B2), retry-cycle metrics (R1), backoff latency (R2),
config validation (R3) and failed-attempt artifacts (R4)."""

import json
from pathlib import Path

import pytest

from web_harness.agents.base import AgentTurn
from web_harness.agents.baseline import BaselineAgent
from web_harness.core.errors import ErrorType, ModelApiError
from web_harness.core.models import (
    ActionDecision,
    Observation,
    PromptBundle,
    RunStatus,
    TaskSpec,
)
from web_harness.core.reliability import ReliabilityBudget, ReliabilityState
from web_harness.env.action_contract import ActionContract, ActionSpec
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.evaluation.fault_injection import FaultInjectingModelAdapter
from web_harness.models.base import ModelOutput
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.reliability.retry import RetryPolicy
from web_harness.runtime.decision_executor import DecisionExecutor
from web_harness.runtime.episode_runner import EpisodeRunner

TASK = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=5)
CONTRACT = ActionContract(
    benchmark="fake",
    actions=[ActionSpec(name="click", signature="click(bid: str)", description="click")],
)
OBS = Observation(goal="g", url="http://fake.local/", axtree="[1] button 'Go'")
SUCCESS_SCRIPT = [{"reward": 1.0, "terminated": True}]


def make_executor(max_api_retries=2, budget_extra=6, **kwargs) -> DecisionExecutor:
    kwargs.setdefault("sleep", lambda _s: None)
    return DecisionExecutor(
        retry_policy=RetryPolicy(api_max_retries=max_api_retries),
        budget=ReliabilityBudget(max_extra_model_calls_per_episode=budget_extra),
        **kwargs,
    )


def run_episode(tmp_path, *, api_error_calls=(), parse_error_calls=(),
                max_api_retries=2, budget_extra=6, run_id="r",
                non_transient_api_calls=()):
    model = MockModelAdapter(["click(bid='1')"])
    faults = FaultInjectingModelAdapter(
        model,
        api_error_on_calls=set(api_error_calls),
        parse_error_on_calls=set(parse_error_calls),
    )
    # patch non-transient faults: wrap generate_action to raise a
    # non-transient ModelApiError on the requested call indices
    if non_transient_api_calls:
        original_generate = faults.generate_action

        def patched(**kwargs):
            idx = faults.call_count
            if idx in set(non_transient_api_calls):
                faults.call_count += 1
                raise ModelApiError("injected auth failure", transient=False)
            return original_generate(**kwargs)

        faults.generate_action = patched

    agent = BaselineAgent(model_adapter=faults)
    env = FakeEnvironment(
        reset_observation=make_fake_observation(url="http://fake.local/0"),
        script=SUCCESS_SCRIPT,
    )
    runner = EpisodeRunner(
        agent=agent,
        env=env,
        trace_root=tmp_path,
        decision_executor=make_executor(
            max_api_retries=max_api_retries, budget_extra=budget_extra
        ),
    )
    return runner.run(TASK, run_id=run_id), env


# -- B1: per-kind retry allowances (mixed failure sequences) --------------------


def test_m1_api_then_parse_then_success(tmp_path):
    result, env = run_episode(tmp_path, api_error_calls=(0,), parse_error_calls=(1,),
                              run_id="m1")
    assert result.status == RunStatus.SUCCESS
    assert result.retry_count == 2
    assert env.step_calls == 1  # one environment action
    assert result.num_steps == 1  # one StepRecord
    events = TraceRecorder.read_events(Path(result.trace_path))
    retry_events = [e for e in events if e.event_type == "retry" and e.outcome == "retry"]
    reasons = [e.data.get("reason") for e in retry_events]
    assert reasons == ["MODEL_API_ERROR", "MODEL_OUTPUT_PARSE_ERROR"]
    # per-kind allowance: 1 API retry + 1 parse retry, no cross-consumption
    succeeded = [e for e in events if e.outcome == "retry_succeeded"][0]
    assert succeeded.data["api_retries"] == 1
    assert succeeded.data["parse_retries"] == 1


def test_m2_parse_then_two_api_errors_then_success(tmp_path):
    result, _ = run_episode(tmp_path, parse_error_calls=(0,), api_error_calls=(1, 2),
                            run_id="m2")
    assert result.status == RunStatus.SUCCESS
    # parse repair = 1, API retries = 2: mixed sequence works both directions
    assert result.retry_count == 3
    assert result.extra_model_calls == 3


def test_m3_episode_budget_cuts_mixed_path(tmp_path):
    # budget allows 2 extra calls; the mixed faults need 3 retries
    result, env = run_episode(
        tmp_path, parse_error_calls=(0,), api_error_calls=(1, 2, 3),
        budget_extra=2, run_id="m3",
    )
    assert result.status == RunStatus.ERROR
    assert result.error_type == ErrorType.BUDGET_EXCEEDED
    assert result.extra_model_calls == 2
    assert env.step_calls == 0  # controlled stop, no environment action


def test_non_transient_api_error_is_terminal(tmp_path):
    """Authentication-class failures must never be retried (review suggestion)."""
    result, env = run_episode(tmp_path, non_transient_api_calls=(0,), run_id="nt")
    assert result.status == RunStatus.ERROR
    assert result.error_type == ErrorType.MODEL_API_ERROR
    assert result.retry_count == 0
    assert env.step_calls == 0
    events = TraceRecorder.read_events(Path(result.trace_path))
    assert [e for e in events if e.event_type == "retry"] == []


# -- B2: Agent Protocol conformance ---------------------------------------------


class MinimalProtocolAgent:
    """A minimal Agent implemented ONLY against the documented protocol
    (no BaselineAgent inheritance, no PromptBuilder, no ModelAdapter)."""

    def __init__(self):
        self.calls = 0

    def decide(self, *, task, observation, history, action_contract,
               repair_feedback=None):
        self.calls += 1
        decision = ActionDecision(action="click(bid='1')", short_reason="minimal")
        return (
            AgentTurn(
                decision=decision,
                model_output=ModelOutput(decision=decision, model_name="minimal"),
            ),
            PromptBundle(system="minimal", user=f"repair={repair_feedback!r}"),
        )


def test_executor_works_with_protocol_conforming_agent(tmp_path):
    agent = MinimalProtocolAgent()
    env = FakeEnvironment(
        reset_observation=make_fake_observation(url="http://fake.local/0"),
        script=SUCCESS_SCRIPT,
    )
    runner = EpisodeRunner(
        agent=agent, env=env, trace_root=tmp_path, decision_executor=make_executor()
    )
    result = runner.run(TASK, run_id="proto")
    assert result.status == RunStatus.SUCCESS
    assert agent.calls == 1
    assert result.retry_count == 0


def test_protocol_agent_receives_repair_feedback(tmp_path):
    """A protocol-conforming agent must receive the repair feedback on the
    retry attempt without any BaselineAgent-specific machinery."""
    seen_feedback = []

    class FaultyThenGood(MinimalProtocolAgent):
        def decide(self, *, task, observation, history, action_contract,
                   repair_feedback=None):
            seen_feedback.append(repair_feedback)
            if self.calls == 0:
                self.calls += 1
                raise ModelApiError("transient failure", transient=True)
            return super().decide(
                task=task, observation=observation, history=history,
                action_contract=action_contract, repair_feedback=repair_feedback,
            )

    agent = FaultyThenGood()
    executor = make_executor()
    result = executor.execute(
        agent=agent, task=TASK, observation=OBS, history=[],
        action_contract=CONTRACT, reliability_state=ReliabilityState(),
    )
    assert result.success
    assert seen_feedback == [None, None]  # API retry does not add repair text


def test_parse_repair_feedback_reaches_protocol_agent():
    seen_feedback = []

    class FaultyParseAgent(MinimalProtocolAgent):
        def decide(self, *, task, observation, history, action_contract,
                   repair_feedback=None):
            seen_feedback.append(repair_feedback)
            if self.calls == 0:
                self.calls += 1
                from web_harness.core.errors import ModelOutputParseError

                raise ModelOutputParseError("bad json", raw_text="nope")
            return super().decide(
                task=task, observation=observation, history=history,
                action_contract=action_contract, repair_feedback=repair_feedback,
            )

    from web_harness.agents.prompt import REPAIR_FEEDBACK

    result = make_executor().execute(
        agent=FaultyParseAgent(), task=TASK, observation=OBS, history=[],
        action_contract=CONTRACT, reliability_state=ReliabilityState(),
    )
    assert result.success
    assert seen_feedback[0] is None
    assert seen_feedback[1] == REPAIR_FEEDBACK


# -- R2: retry latency includes backoff ------------------------------------------


def test_retry_latency_includes_backoff():
    """One API retry with backoff=500ms must produce retry_latency_s >= 0.5."""
    agent, _ = BaselineAgent(model_adapter=FaultInjectingModelAdapter(
        MockModelAdapter(["click(bid='1')"]), api_error_on_calls={0}
    )), None
    agent = BaselineAgent(model_adapter=FaultInjectingModelAdapter(
        MockModelAdapter(["click(bid='1')"]), api_error_on_calls={0}
    ))
    import time as real_time

    executor = DecisionExecutor(
        retry_policy=RetryPolicy(api_max_retries=2, api_backoff_ms=[500, 1000]),
        budget=ReliabilityBudget(),
        sleep=real_time.sleep,  # REAL sleep: backoff must be measurable
    )
    result = executor.execute(
        agent=agent, task=TASK, observation=OBS, history=[],
        action_contract=CONTRACT, reliability_state=ReliabilityState(),
    )
    assert result.success
    assert result.retry_count == 1
    assert result.retry_latency_s >= 0.5


def test_parse_repair_adds_no_backoff():
    agent = BaselineAgent(model_adapter=FaultInjectingModelAdapter(
        MockModelAdapter(["click(bid='1')"]), parse_error_on_calls={0}
    ))
    sleeps = []
    executor = DecisionExecutor(
        retry_policy=RetryPolicy(),
        budget=ReliabilityBudget(),
        sleep=lambda s: sleeps.append(s),
    )
    result = executor.execute(
        agent=agent, task=TASK, observation=OBS, history=[],
        action_contract=CONTRACT, reliability_state=ReliabilityState(),
    )
    assert result.success
    assert sleeps == []  # parse repair is immediate: no scheduled backoff


# -- R3: retry config fail-fast validation ----------------------------------------


def make_config_data(reliability: dict) -> dict:
    return {
        "model": {"provider": "mock", "mock_actions": ["noop()"]},
        "agent": {"type": "baseline"},
        "runtime": {"max_steps": 5},
        "trace": {"root_dir": "runs"},
        "environment": {},
        "reliability": reliability,
    }


@pytest.mark.parametrize(
    "reliability",
    [
        # retry.enabled not a bool
        {"enabled": True, "retry": {"enabled": "yes"}},
        # negative max_retries
        {"enabled": True, "retry": {"enabled": True, "model_api": {"max_retries": -1}}},
        # backoff_ms not a list of ints
        {"enabled": True, "retry": {"enabled": True, "model_api": {"backoff_ms": [500, "1s"]}}},
        # negative backoff
        {"enabled": True, "retry": {"enabled": True, "model_api": {"backoff_ms": [-5]}}},
        # negative parse retries
        {"enabled": True, "retry": {"enabled": True, "model_output": {"max_retries": -1}}},
        # negative episode budget
        {"enabled": True, "budget": {"max_extra_model_calls_per_episode": -1}},
    ],
)
def test_invalid_retry_config_fails_fast(reliability):
    from web_harness.config.loader import HarnessConfig
    from web_harness.core.errors import ConfigError

    with pytest.raises(ConfigError):
        HarnessConfig(make_config_data(reliability))


def test_valid_retry_config_accepted():
    from web_harness.config.loader import HarnessConfig

    cfg = HarnessConfig(
        make_config_data(
            {
                "enabled": True,
                "retry": {
                    "enabled": True,
                    "model_api": {"max_retries": 2, "backoff_ms": [500, 1000]},
                    "model_output": {"max_retries": 1},
                },
                "budget": {"max_extra_model_calls_per_episode": 6},
            }
        )
    )
    assert cfg.reliability_enabled


# -- R4: failed parse attempts are auditable --------------------------------------


def test_failed_parse_attempt_artifact_written(tmp_path):
    """malformed output -> repair success: BOTH the failed attempt artifact
    and the final successful model artifact must exist in the run trace."""
    result, _ = run_episode(tmp_path, parse_error_calls=(0,), run_id="artifact")
    run_dir = Path(result.trace_path)

    failed_artifact = run_dir / "artifacts" / "attempt_000_00.json"
    assert failed_artifact.exists(), "failed attempt artifact missing"
    payload = json.loads(failed_artifact.read_text(encoding="utf-8"))
    assert payload["failure_type"] == "MODEL_OUTPUT_PARSE_ERROR"
    assert payload["raw_text"]
    assert payload["input_tokens"] == FaultInjectingModelAdapter.PARSE_FAIL_INPUT_TOKENS
    assert payload["output_tokens"] == FaultInjectingModelAdapter.PARSE_FAIL_OUTPUT_TOKENS

    # final successful model response artifact still present
    assert (run_dir / "artifacts" / "model_000.json").exists()

    # the retry event points at the failed attempt artifact
    events = TraceRecorder.read_steps(run_dir)  # ensure steps reload fine
    assert len(events) == 1
    retry_events = [
        e for e in TraceRecorder.read_events(run_dir)
        if e.event_type == "retry" and e.outcome == "retry"
    ]
    assert retry_events[0].data.get("artifact_ref") == "artifacts/attempt_000_00.json"

    # steps.jsonl still has exactly one step (one-step semantics untouched)
    steps = TraceRecorder.read_steps(run_dir)
    assert len(steps) == 1


def test_attempt_artifacts_disabled_with_save_model_responses_off(tmp_path):
    agent = BaselineAgent(model_adapter=FaultInjectingModelAdapter(
        MockModelAdapter(["click(bid='1')"]), parse_error_on_calls={0}
    ))
    env = FakeEnvironment(
        reset_observation=make_fake_observation(url="http://fake.local/0"),
        script=SUCCESS_SCRIPT,
    )
    runner = EpisodeRunner(
        agent=agent, env=env, trace_root=tmp_path,
        save_model_responses=False, decision_executor=make_executor(),
    )
    result = runner.run(TASK, run_id="noart")
    run_dir = Path(result.trace_path)
    assert not (run_dir / "artifacts" / "attempt_000_00.json").exists()
