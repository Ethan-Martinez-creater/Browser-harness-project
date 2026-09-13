"""Phase 1D controlled replanning tests (D1-D10 semantics).

Replanning is escalation-only: the deterministic ReplanTriggerPolicy runs
AFTER the Phase 1C policy on its RECOVER path, the Replanner produces a
bounded RecoveryPlan as ADVISORY prompt context (never executed), and the
plan horizon is consumed only by real Agent StepRecords.
"""

import json
from pathlib import Path

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.errors import ModelApiError
from web_harness.core.models import ActionDecision, ModelOutput, TaskSpec
from web_harness.core.reliability import (
    FailureKind,
    FailureSeverity,
    FailureSignal,
    ReliabilityBudget,
)
from web_harness.env.fake import make_fake_observation
from web_harness.evaluation.fault_injection import (
    FaultInjectingEnvironmentAdapter,
)
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.reliability.policy import FailurePolicyEngine
from web_harness.reliability.replan_policy import (
    ReplanTriggerPolicy,
    ReplanTriggerReason,
)
from web_harness.reliability.replanner import ReplanExecutor
from web_harness.reliability.retry import RetryPolicy
from web_harness.runtime.episode_runner import EpisodeRunner

TASK = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=8)
RESET_OBS = make_fake_observation(url="http://fake.local/start")
OBS_B = make_fake_observation(url="http://fake.local/B")

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

BAD_PLAN_JSON = "sorry, no json here"


def signal(kind, sig, severity=FailureSeverity.ERROR):
    return FailureSignal(
        kind=kind, severity=severity, source="test", signature=sig
    )


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


def make_runner(
    tmp_path,
    env,
    actions,
    *,
    replanning=True,
    plan_jsons=None,
    max_steps=8,
    max_replans=1,
    horizon=3,
    api_max_retries=1,
    max_recoveries=3,
):
    budget = ReliabilityBudget(
        max_recoveries_per_episode=max_recoveries,
        max_replans_per_episode=max_replans,
        recovery_failures_before_replan=2,
        plan_horizon_steps=horizon,
        recent_steps=6,
    )
    return _make_runner_flexible(
        tmp_path, env, actions, replanning=replanning,
        plan_jsons=plan_jsons, max_steps=max_steps,
        max_replans=max_replans, horizon=horizon,
        api_max_retries=api_max_retries, budget=budget,
    )


def _make_runner_flexible(
    tmp_path, env, actions, *, replanning, plan_jsons, max_steps,
    max_replans, horizon, api_max_retries, budget,
):
    from web_harness.reliability.verifier import DefaultStepVerifier

    return EpisodeRunner(
        agent=BaselineAgent(model_adapter=MockModelAdapter(list(actions))),
        env=env,
        trace_root=tmp_path,
        verifier=DefaultStepVerifier(),
        failure_policy=FailurePolicyEngine(wait_ms=500),
        recovery_budget=budget,
        replan_trigger_policy=ReplanTriggerPolicy() if replanning else None,
        replan_executor=ReplanExecutor(
            model_adapter=ScriptedStructuredModel(
                plan_jsons if plan_jsons is not None else [PLAN_JSON]
            ),
            retry_policy=RetryPolicy(
                api_max_retries=api_max_retries,
                api_backoff_ms=[0],
                parse_max_retries=1,
            ),
            budget=budget,
            sleep=lambda _s: None,
        ),
    )


def assert_replan_outcome_invariant(result) -> None:
    assert (
        result.replan_success_count
        + result.replan_failed_count
        + result.replan_unresolved_count
        == result.replan_count
    )


def assert_recovery_outcome_invariant(result) -> None:
    assert (
        result.recovery_success_count
        + result.recovery_failed_count
        + result.recovery_unresolved_count
        == result.recovery_count
    )


def events_of(result):
    return TraceRecorder.read_events(Path(result.trace_path))


def outcomes(events, event_type):
    return [e.outcome for e in events if e.event_type.value == event_type]


# -- trigger policy unit rules --------------------------------------------------


def test_trigger_policy_fixed_rules():
    policy = ReplanTriggerPolicy()
    state = type(signal("MODEL_API_TRANSIENT", "x")).__mro__  # noqa: F841
    from web_harness.core.reliability import ReliabilityState

    st = ReliabilityState()
    budget = ReliabilityBudget(
        max_replans_per_episode=1, recovery_failures_before_replan=2
    )
    from web_harness.reliability.policy import PolicyAction

    # base PASS/ABORT never escalates
    assert not policy.decide(
        base_action=PolicyAction.CONTINUE, failure_kind=FailureKind.LOOP_DETECTED,
        failure_signature="s", reliability_state=st, budget=budget,
    ).trigger
    # OBSERVATION_INVALID never escalates even on RECOVER
    assert not policy.decide(
        base_action=PolicyAction.RECOVER,
        failure_kind=FailureKind.OBSERVATION_INVALID,
        failure_signature="s", reliability_state=st, budget=budget,
    ).trigger
    # D1: loop matching the last FAILED recovery signature
    st.last_failed_recovery_signature = "loop:sig"
    d = policy.decide(
        base_action=PolicyAction.RECOVER, failure_kind=FailureKind.LOOP_DETECTED,
        failure_signature="loop:sig", reliability_state=st, budget=budget,
    )
    assert d.trigger and d.reason == ReplanTriggerReason.REPEATED_LOOP_AFTER_RECOVERY
    # single failure (empty state) cannot replan
    st2 = ReliabilityState()
    assert not policy.decide(
        base_action=PolicyAction.RECOVER, failure_kind=FailureKind.ACTION_ERROR,
        failure_signature="s", reliability_state=st2, budget=budget,
    ).trigger
    # D2: streak >= threshold triggers
    st2.consecutive_recovery_failures = 2
    d2 = policy.decide(
        base_action=PolicyAction.RECOVER, failure_kind=FailureKind.ACTION_ERROR,
        failure_signature="s", reliability_state=st2, budget=budget,
    )
    assert d2.trigger and d2.reason == ReplanTriggerReason.RECOVERY_FAILURE_STREAK
    # budget exhausted blocks NEW plans (no trigger; caller falls back)
    st2.replan_count = 1
    assert not policy.decide(
        base_action=PolicyAction.RECOVER, failure_kind=FailureKind.ACTION_ERROR,
        failure_signature="s", reliability_state=st2, budget=budget,
    ).trigger


# -- D1: repeated loop after failed recovery -> replan -> success ---------------


def test_d1_repeated_loop_after_failed_recovery(tmp_path):
    env = make_env(
        [{}, {}, {}, {}, {}, {"observation": OBS_B},
         {"reward": 1.0, "terminated": True, "observation": OBS_B}],
    )
    actions = [
        "click(bid='1')",  # no_progress (warning)
        "click(bid='1')",  # loop1 (twice) -> BLOCK recovery#1 (pending)
        "click(bid='2')",  # same state: recovery#1 signature repeats -> FAILED
        "click(bid='1')",  # state unchanged (no loop: different transition)
        "click(bid='1')",  # loop1 again (twice) -> D1 escalation -> replan
        "click(bid='3')",  # plan guides: page changes -> replan SUCCESS
        "click(bid='9')",  # task success
    ]
    result = make_runner(tmp_path, env, actions).run(TASK, run_id="d1")

    assert result.success
    assert result.replan_count == 1
    assert result.replan_success_count == 1
    assert result.replan_failed_count == 0
    assert result.replan_model_calls == 1
    assert result.recovery_failed_count == 1
    assert_recovery_outcome_invariant(result)
    assert_replan_outcome_invariant(result)
    events = events_of(result)
    assert "triggered" in outcomes(events, "replan")
    assert "created" in outcomes(events, "replan")
    # the plan artifact exists
    artifacts = list(Path(result.trace_path).glob("artifacts/replan_*.json"))
    assert len(artifacts) == 1
    plan_data = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert plan_data["trigger_reason"] == "repeated_loop_after_recovery"
    assert plan_data["plan"]["immediate_subgoal"]


# -- D2: two consecutive recovery failures trigger replan -----------------------


def test_d2_recovery_failure_streak_triggers_replan(tmp_path):
    env = make_env(
        [{}] * 7 + [{"reward": 1.0, "terminated": True, "observation": OBS_B}],
        action_error_on_steps={0, 2, 5},
    )
    actions = [f"click(bid='{i}')" for i in range(1, 9)]
    result = make_runner(tmp_path, env, actions).run(TASK, run_id="d2")

    assert result.success
    assert result.replan_count == 1
    assert result.replan_success_count == 1
    assert result.recovery_failed_count == 2  # two real recovery failures
    assert_recovery_outcome_invariant(result)
    assert_replan_outcome_invariant(result)
    events = events_of(result)
    triggered = [
        e for e in events
        if e.event_type.value == "replan" and e.outcome == "triggered"
    ]
    assert triggered[0].data["reason"] == "recovery_failure_streak"
    assert triggered[0].data["recovery_failure_streak"] == 2


# -- D3: replanning disabled leaves Phase 1C behavior unchanged -----------------


def test_d3_replanning_disabled_causal_control(tmp_path):
    env = make_env(
        [{}, {}, {}, {}, {}, {"observation": OBS_B},
         {"reward": 1.0, "terminated": True, "observation": OBS_B}],
    )
    actions = [
        "click(bid='1')",
        "click(bid='1')",
        "click(bid='2')",
        "click(bid='1')",
        "click(bid='1')",
        "click(bid='3')",
        "click(bid='9')",
    ]
    result = make_runner(tmp_path, env, actions, replanning=False).run(
        TASK, run_id="d3"
    )

    assert result.replan_count == 0
    assert outcomes(events_of(result), "replan") == []
    assert not list(Path(result.trace_path).glob("artifacts/replan_*.json"))
    assert_recovery_outcome_invariant(result)


# -- D4: replan budget exhausted never kills execution ---------------------------


def test_d4_budget_exhausted_falls_back_and_continues(tmp_path):
    # after the one replan is used, clean steps CONTINUE and a further
    # recoverable failure falls back to Phase 1C recovery
    env = make_env(
        [{}] * 8 + [{}, {"reward": 1.0, "terminated": True,
                        "observation": OBS_B}],
        action_error_on_steps={0, 2, 5, 8},
    )
    actions = [f"click(bid='{i}')" for i in range(1, 11)]
    task = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=10)
    result = make_runner(
        tmp_path, env, actions, max_steps=10, max_replans=1,
        max_recoveries=5,
    ).run(task, run_id="d4")

    assert result.success
    assert result.replan_count == 1
    assert "abort" not in outcomes(events_of(result), "policy_decision")
    # the post-budget ACTION_ERROR was handled by a plain recovery
    assert result.recovery_count >= 3
    assert_replan_outcome_invariant(result)


# -- D5: plan horizon consumed only by real agent steps --------------------------


def test_d5_plan_horizon_expires(tmp_path):
    # plan horizon = 2: the WAIT recovery noop and blocked re-decisions do
    # NOT consume it; only real StepRecords do
    env = make_env(
        [{}] * 8
        + [{"reward": 1.0, "terminated": True, "observation": OBS_B}],
        empty_observation_on_steps={5},
    )
    actions = [
        "click(bid='1')",  # no_progress
        "click(bid='1')",  # loop1 -> BLOCK recovery#1
        "click(bid='2')",  # recovery#1 signature repeats -> FAILED
        "click(bid='1')",  # state unchanged
        "click(bid='1')",  # loop1 again -> D1 escalation -> replan (horizon 2)
        "click(bid='3')",  # plan step 1; empty obs -> WAIT recovery (no consume)
        "click(bid='4')",  # plan step 2 -> horizon expires after this
        "click(bid='9')",  # no plan anymore; task success
    ]
    prompts: list[str] = []

    runner = _make_runner_flexible(
        tmp_path, env, actions, replanning=True, plan_jsons=[PLAN_JSON],
        max_steps=8, max_replans=1, horizon=2, api_max_retries=1,
        budget=ReliabilityBudget(
            max_recoveries_per_episode=3, max_replans_per_episode=1,
            recovery_failures_before_replan=2, plan_horizon_steps=2,
            recent_steps=6,
        ),
    )
    # wrap the agent to capture prompts
    agent_decide = runner.agent.decide

    def spy_decide(**kwargs):
        turn, prompt = agent_decide(**kwargs)
        prompts.append(prompt.user)
        return turn, prompt

    runner.agent.decide = spy_decide
    result = runner.run(TASK, run_id="d5")

    assert result.success
    assert result.replan_count == 1
    plan_prompts = ["# Recovery plan" in p for p in prompts]
    assert any(plan_prompts), "plan must reach at least one prompt"
    # the LAST decision (after expiry) must not carry the plan
    assert not plan_prompts[-1]
    assert_recovery_outcome_invariant(result)
    assert_replan_outcome_invariant(result)


# -- D6: same trigger failure under active plan -> replan FAILED -----------------


def test_d6_same_failure_repeats_under_plan(tmp_path):
    env = make_env(
        [{}] * 7
        + [{"reward": 1.0, "terminated": True, "observation": OBS_B}],
    )
    actions = [
        "click(bid='1')",  # no_progress
        "click(bid='1')",  # loop1 -> BLOCK recovery#1
        "click(bid='2')",  # recovery#1 signature repeats -> FAILED
        "click(bid='1')",  # state unchanged
        "click(bid='1')",  # loop1 again -> D1 replan
        "click(bid='1')",  # same trigger signature repeats -> replan FAILED
        "click(bid='3')",  # different action -> recovery#2 resolves, success
        "click(bid='9')",
    ]
    result = make_runner(tmp_path, env, actions).run(TASK, run_id="d6")

    assert result.success
    assert result.replan_count == 1
    assert result.replan_failed_count == 1
    assert result.replan_success_count == 0
    # max_replans=1: no second replan intervention
    assert outcomes(events_of(result), "replan").count("triggered") == 1
    assert_replan_outcome_invariant(result)


# -- D7: replan generation parse repair ------------------------------------------


def test_d7_replan_parse_repair(tmp_path):
    env = make_env(
        [{}, {}, {}, {}, {}, {"observation": OBS_B},
         {"reward": 1.0, "terminated": True, "observation": OBS_B}],
    )
    actions = [
        "click(bid='1')",
        "click(bid='1')",
        "click(bid='2')",
        "click(bid='1')",
        "click(bid='1')",
        "click(bid='3')",
        "click(bid='9')",
    ]
    result = make_runner(
        tmp_path, env, actions, plan_jsons=[BAD_PLAN_JSON, PLAN_JSON]
    ).run(TASK, run_id="d7")

    assert result.success
    assert result.replan_count == 1
    assert result.replan_model_calls == 2  # parse fail + repair
    assert result.replan_success_count == 1
    assert result.replan_input_tokens >= 100  # failed attempt usage retained
    assert_replan_outcome_invariant(result)
    artifacts = list(
        Path(result.trace_path).glob("artifacts/*attempt_*.json")
    )
    assert any(
        json.loads(a.read_text(encoding="utf-8"))["failure_type"]
        == "REPLAN_OUTPUT_PARSE_ERROR"
        for a in artifacts
    )


# -- D8: replan generation API failure paths --------------------------------------


def test_d8a_transient_api_error_then_success(tmp_path):
    env = make_env(
        [{}, {}, {}, {}, {}, {"observation": OBS_B},
         {"reward": 1.0, "terminated": True, "observation": OBS_B}],
    )
    actions = [
        "click(bid='1')",
        "click(bid='1')",
        "click(bid='2')",
        "click(bid='1')",
        "click(bid='1')",
        "click(bid='3')",
        "click(bid='9')",
    ]
    result = make_runner(
        tmp_path, env, actions,
        plan_jsons=[ModelApiError("transient", transient=True), PLAN_JSON],
    ).run(TASK, run_id="d8a")

    assert result.success
    assert result.replan_count == 1
    assert result.replan_model_calls == 2
    assert result.replan_success_count == 1
    assert_replan_outcome_invariant(result)


def test_d8b_persistent_api_failure_falls_back_to_recovery(tmp_path):
    env = make_env(
        [{}, {}, {}, {}, {}, {}, {"reward": 1.0, "terminated": True,
                                   "observation": OBS_B}],
        action_error_on_steps={0, 2, 5},
    )
    actions = [f"click(bid='{i}')" for i in range(1, 8)]
    result = make_runner(
        tmp_path, env, actions,
        plan_jsons=[ModelApiError("down", transient=True)],
        api_max_retries=1,
    ).run(TASK, run_id="d8b")

    # generation failed -> replan FAILED outcome; execution falls back to
    # the Phase 1C recovery instead of crashing or executing browser actions
    assert result.replan_count == 1
    assert result.replan_failed_count == 1
    assert result.replan_success_count == 0
    assert result.replan_model_calls == 2  # initial + 1 api retry
    assert "generation_failed" in outcomes(events_of(result), "replan")
    assert_replan_outcome_invariant(result)
    assert_recovery_outcome_invariant(result)


# -- D9: terminal task never replans ----------------------------------------------


def test_d9_terminal_never_replans(tmp_path):
    env = make_env([{"terminated": True, "reward": 0.0}])
    result = make_runner(
        tmp_path, env, ["click(bid='1')", "click(bid='2')"]
    ).run(TASK, run_id="d9")

    assert result.status.value == "failed"
    assert result.replan_count == 0
    assert result.replan_model_calls == 0
    assert outcomes(events_of(result), "replan") == []


# -- D10: unresolved recovery is neutral for the streak ---------------------------


def test_d10_unresolved_recovery_neutral(tmp_path):
    # a recovery on the LAST step stays unresolved and must not raise the
    # failure streak nor trigger any replan
    env = make_env([{}, {}])
    task = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=2)
    result = make_runner(tmp_path, env, ["click(bid='1')", "click(bid='1')"],
                         max_steps=2).run(task, run_id="d10")

    assert result.recovery_unresolved_count == 1
    assert result.recovery_failed_count == 0
    assert result.replan_count == 0
    assert_recovery_outcome_invariant(result)
    assert_replan_outcome_invariant(result)
