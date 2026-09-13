"""Phase 1D closure remediation tests (B1-B4, R1-R3, D11-D15).

Cost / trace / reporting contracts:
- every real replan model call consumes one global extra-model-call slot
  BEFORE the call (budget=0 -> zero planner calls);
- episode total tokens include Replanner tokens;
- replan parse retries use a real format-repair prompt;
- decision and replan failed-attempt artifacts never collide.
"""

import json
from pathlib import Path

import pytest

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.models import (
    ActionDecision,
    ModelOutput,
    RunResult,
    RunStatus,
    TaskSpec,
)
from web_harness.core.reliability import (
    ReliabilityBudget,
)
from web_harness.env.fake import make_fake_observation
from web_harness.evaluation.fault_injection import (
    FaultInjectingEnvironmentAdapter,
    FaultInjectingModelAdapter,
)
from web_harness.evaluation.metrics import aggregate_metrics, episode_metrics
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.reliability.policy import FailurePolicyEngine
from web_harness.reliability.replan_policy import ReplanTriggerPolicy
from web_harness.reliability.replanner import (
    ReplanExecutor,
)
from web_harness.reliability.retry import RetryPolicy
from web_harness.reliability.verifier import DefaultStepVerifier
from web_harness.runtime.decision_executor import DecisionExecutor
from web_harness.runtime.episode_runner import EpisodeRunner

RESET_OBS = make_fake_observation(url="http://fake.local/start")
OBS_B = make_fake_observation(url="http://fake.local/B")
TASK = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=8)
PLAN_JSON = json.dumps({
    "diagnosis": "Repeated interaction with the same control is not "
                 "changing the page.",
    "immediate_subgoal": "Find an alternative control that advances the "
                         "task.",
    "strategy_steps": [
        "Inspect nearby interactive elements.",
        "Choose an alternative control.",
        "Confirm the page state changes.",
    ],
    "avoid_actions": ["click(bid='1')"],
    "horizon_steps": 3,
})


class ScriptedStructuredModel:
    """Deterministic replan model: scripted raw texts / exceptions."""

    def __init__(self, plan_jsons):
        self.plan_jsons = list(plan_jsons)
        self.call_count = 0

    def generate_structured(self, *, prompt):
        item = self.plan_jsons[min(self.call_count, len(self.plan_jsons) - 1)]
        self.call_count += 1
        if isinstance(item, Exception):
            raise item
        return ModelOutput(
            decision=ActionDecision(action=item),
            model_name="mock-replan",
            input_tokens=50,
            output_tokens=20,
            raw_text=item,
        )


def make_env(script, **injection):
    from web_harness.env.fake import FakeEnvironment

    return FaultInjectingEnvironmentAdapter(
        FakeEnvironment(reset_observation=RESET_OBS, script=script),
        **injection,
    )

D1_SCRIPT = [
    {}, {}, {}, {}, {}, {"observation": OBS_B},
    {"reward": 1.0, "terminated": True, "observation": OBS_B},
]
D1_ACTIONS = [
    "click(bid='1')",  # no_progress
    "click(bid='1')",  # loop1 -> BLOCK recovery#1
    "click(bid='2')",  # recovery#1 signature repeats -> FAILED
    "click(bid='1')",  # state unchanged (no loop)
    "click(bid='1')",  # loop1 again -> D1 escalation -> replan
    "click(bid='3')",  # plan guides: page changes -> replan SUCCESS
    "click(bid='9')",  # task success
]


def make_runner_ex(
    tmp_path,
    env,
    actions,
    *,
    plan_jsons=None,
    max_extra_model_calls=6,
    max_replans=1,
    api_max_retries=1,
    decision_api_error_calls=(),
    decision_parse_error_calls=(),
):
    """Runner with an explicit global extra-model-call budget for G tests."""
    budget = ReliabilityBudget(
        max_recoveries_per_episode=3,
        max_replans_per_episode=max_replans,
        recovery_failures_before_replan=2,
        plan_horizon_steps=3,
        recent_steps=6,
        max_extra_model_calls_per_episode=max_extra_model_calls,
    )
    structured = ScriptedStructuredModel(
        plan_jsons if plan_jsons is not None else [PLAN_JSON]
    )
    model = MockModelAdapter(list(actions))
    agent = BaselineAgent(
        model_adapter=FaultInjectingModelAdapter(
            model,
            api_error_on_calls=set(decision_api_error_calls),
            parse_error_on_calls=set(decision_parse_error_calls),
        )
    )
    return EpisodeRunner(
        agent=agent,
        env=env,
        trace_root=tmp_path,
        verifier=DefaultStepVerifier(),
        failure_policy=FailurePolicyEngine(wait_ms=500),
        recovery_budget=budget,
        decision_executor=DecisionExecutor(
            retry_policy=RetryPolicy(
                api_max_retries=1, api_backoff_ms=[0], parse_max_retries=1,
            ),
            budget=budget,
            sleep=lambda _s: None,
        ),
        replan_trigger_policy=ReplanTriggerPolicy(),
        replan_executor=ReplanExecutor(
            model_adapter=structured,
            retry_policy=RetryPolicy(
                api_max_retries=api_max_retries,
                api_backoff_ms=[0],
                parse_max_retries=1,
            ),
            budget=budget,
            sleep=lambda _s: None,
        ),
    )


def assert_replan_outcome_invariant(result):
    assert (
        result.replan_success_count
        + result.replan_failed_count
        + result.replan_unresolved_count
        == result.replan_count
    )


def assert_recovery_outcome_invariant(result):
    assert (
        result.recovery_success_count
        + result.recovery_failed_count
        + result.recovery_unresolved_count
        == result.recovery_count
    )


# -- B1 / G1: budget=0 blocks the initial replan call entirely ------------------


def test_g1_budget_zero_blocks_initial_replan_call(tmp_path):
    env = make_env(D1_SCRIPT)
    result = make_runner_ex(
        tmp_path, env, D1_ACTIONS, max_extra_model_calls=0
    ).run(TASK, run_id="g1")

    # zero planner model calls: the trigger fires but no call is made
    assert result.replan_count == 1
    assert result.replan_model_calls == 0
    assert result.replan_failed_count == 1
    assert_replan_outcome_invariant(result)
    events = TraceRecorder.read_events(Path(result.trace_path))
    exhausted = [
        e for e in events
        if e.event_type.value == "replan" and e.outcome == "exhausted"
    ]
    assert exhausted and "budget" in exhausted[0].data.get("reason", "")
    # execution fell back to Phase 1C recovery instead of dying
    assert result.success or result.status.value in ("error", "failed")


# -- B1 / G2: budget=1 lets the initial call through but no repair retry --------


def test_g2_budget_one_blocks_parse_repair(tmp_path):
    env = make_env(D1_SCRIPT)
    result = make_runner_ex(
        tmp_path, env, D1_ACTIONS,
        max_extra_model_calls=1,
        plan_jsons=["not json at all", PLAN_JSON],
    ).run(TASK, run_id="g2")

    # initial call consumed the only slot; the parse repair never ran
    assert result.replan_model_calls == 1
    assert result.replan_count == 1
    assert result.replan_failed_count == 1
    assert_replan_outcome_invariant(result)


# -- B1 / G3: decision retries and planner calls share one global budget --------


def test_g3_decision_retry_shares_budget_with_replan(tmp_path):
    # 1 agent-decision retry (api error on the first decision call) consumes
    # one slot; with budget=2 only ONE planner call can happen afterwards
    env = make_env(D1_SCRIPT)
    result = make_runner_ex(
        tmp_path, env, D1_ACTIONS,
        max_extra_model_calls=2,
        decision_api_error_calls=(0,),
        plan_jsons=["still bad", PLAN_JSON],
    ).run(TASK, run_id="g3")

    assert result.retry_count == 1  # agent decision retry happened
    assert result.replan_count == 1
    assert result.replan_model_calls == 1  # budget left: exactly one slot
    assert_replan_outcome_invariant(result)


# -- B2: episode total tokens include Replanner tokens ---------------------------


def test_b2_total_tokens_include_replanner(tmp_path):
    env = make_env(D1_SCRIPT)
    result = make_runner_ex(
        tmp_path, env, D1_ACTIONS, max_extra_model_calls=6
    ).run(TASK, run_id="b2")

    # ScriptedStructuredModel: 50 in / 20 out per call, 1 call
    assert result.replan_input_tokens == 50
    assert result.replan_output_tokens == 20
    # agent decision tokens (mock) + planner tokens = episode total
    assert result.input_tokens >= result.replan_input_tokens
    m = episode_metrics(result)
    agent_in = m["input_tokens"] - result.replan_input_tokens
    assert agent_in + result.replan_input_tokens == m["input_tokens"]
    assert agent_in + result.replan_output_tokens - result.replan_output_tokens \
        + result.replan_output_tokens == m["output_tokens"]


# -- B4: decision and replan attempt artifacts never collide ---------------------


def test_b4_artifacts_do_not_collide_on_same_step(tmp_path):
    # same step: agent decision parse failure AND replan parse failure
    env = make_env(D1_SCRIPT)
    result = make_runner_ex(
        tmp_path, env, D1_ACTIONS,
        decision_parse_error_calls=(0,),
        plan_jsons=["bad replan json", PLAN_JSON],
    ).run(TASK, run_id="b4")

    run_dir = Path(result.trace_path)
    decision_artifacts = list(run_dir.glob("artifacts/attempt_*.json"))
    replan_artifacts = list(run_dir.glob("artifacts/replan_attempt_*.json"))
    assert decision_artifacts, "decision attempt artifact must exist"
    assert replan_artifacts, "replan attempt artifact must exist"
    # distinct files, both present
    assert {a.name for a in decision_artifacts}.isdisjoint(
        {a.name for a in replan_artifacts}
    )
    # event artifact_refs point at the right files
    events = TraceRecorder.read_events(run_dir)
    refs = [
        e.data.get("artifact_ref") for e in events
        if e.event_type.value in ("retry", "replan")
        and e.data.get("artifact_ref")
    ]
    assert any("replan_attempt_" in r for r in refs)
    assert any("replan_attempt_" not in r for r in refs)


# -- B3 / D13: replan parse repair prompt actually changes ------------------------


def test_d13_replan_repair_prompt_changes(tmp_path):
    captured_prompts = []


    env = make_env(D1_SCRIPT)
    runner = make_runner_ex(
        tmp_path, env, D1_ACTIONS,
        plan_jsons=["bad replan json", PLAN_JSON],
    )
    # wrap the executor's model instead of rebuilding the runner
    inner = runner.replan_executor.model_adapter

    class Spy:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def generate_structured(self, *, prompt):
            captured_prompts.append(prompt.user)
            return self.wrapped.generate_structured(prompt=prompt)

    runner.replan_executor.model_adapter = Spy(inner)
    result = runner.run(TASK, run_id="d13")

    assert result.replan_model_calls == 2
    assert len(captured_prompts) == 2
    assert "# Format correction" not in captured_prompts[0]
    assert "# Format correction" in captured_prompts[1]
    assert "Previous response did not match" in captured_prompts[1]
    assert result.replan_success_count == 1


# -- R3 / D15: empty plan fields go through structured parse repair --------------


def test_d15_invalid_plan_fields_repaired(tmp_path):
    bad_plan = json.dumps({
        "diagnosis": "",
        "immediate_subgoal": "   ",
        "strategy_steps": [],
        "horizon_steps": 0,
    })
    env = make_env(D1_SCRIPT)
    result = make_runner_ex(
        tmp_path, env, D1_ACTIONS, plan_jsons=[bad_plan, PLAN_JSON]
    ).run(TASK, run_id="d15")

    assert result.replan_count == 1
    assert result.replan_model_calls == 2  # invalid plan -> repair -> valid
    assert result.replan_success_count == 1
    assert_replan_outcome_invariant(result)


# -- R2: cost metric contract tests (M1-M4) ---------------------------------------


def _result(**kwargs) -> RunResult:
    base = dict(
        run_id="r", task_spec=TaskSpec(benchmark="b", task_id="t", seed=0),
        status=RunStatus.SUCCESS,
    )
    base.update(kwargs)
    return RunResult(**base)


def test_m1_replan_calls_only():
    agg = aggregate_metrics([
        _result(replan_model_calls=2),
    ])
    assert agg["reliability_extra_model_calls"] == 2
    assert agg["total_extra_model_calls"] == 0


def test_m2_decision_retry_plus_replan_no_double_count():
    agg = aggregate_metrics([
        _result(extra_model_calls=1, replan_model_calls=2),
    ])
    assert agg["reliability_extra_model_calls"] == 3
    # retry tokens are not double counted into reliability_extra_tokens
    assert agg["total_extra_model_calls"] == 1


def test_m3_token_relationships():
    r = _result(
        input_tokens=1000, output_tokens=500,
        retry_input_tokens=100, retry_output_tokens=50,
        replan_input_tokens=200, replan_output_tokens=80,
    )
    agg = aggregate_metrics([r])
    # total >= retry tokens + replan tokens (retry/replan are subsets)
    assert agg["total_input_tokens"] == 1000
    assert agg["total_output_tokens"] == 500
    assert agg["total_retry_input_tokens"] == 100
    assert agg["total_replan_input_tokens"] == 200
    assert agg["reliability_extra_tokens"] == 100 + 50 + 200 + 80


def test_m4_latency_relationship():
    r = _result(
        retry_latency_s=0.5, recovery_latency_s=0.2, replan_latency_s=1.0,
    )
    agg = aggregate_metrics([r])
    assert agg["reliability_extra_latency_s"] == pytest.approx(1.7)


# -- R1: episode_metrics exposes replan fields (inspect-run safety) ---------------


def test_r1_episode_metrics_has_replan_fields():
    m = episode_metrics(_result(replan_count=1, replan_model_calls=2))
    for key in (
        "replan_count", "replan_success_count", "replan_failed_count",
        "replan_unresolved_count", "replan_model_calls",
        "replan_input_tokens", "replan_output_tokens", "replan_latency_s",
        "reliability_extra_model_calls", "reliability_extra_tokens",
        "reliability_extra_latency_s",
    ):
        assert key in m


def test_r1_inspect_run_shows_replan_summary(tmp_path, capsys):
    """inspect-run must render a run WITH replans without KeyError."""
    from web_harness.cli import inspect_run

    env = make_env(D1_SCRIPT)
    result = make_runner_ex(
        tmp_path, env, D1_ACTIONS, max_extra_model_calls=6
    ).run(TASK, run_id="insp")
    # result.trace_path is inside tmp_path; run inspect-run against it
    inspect_run(
        run_id=Path(result.trace_path).name,
        runs_root=str(Path(result.trace_path).parent),
    )
    out = capsys.readouterr().out
    assert "Replans: 1" in out
    assert "Replan model calls: 1" in out
