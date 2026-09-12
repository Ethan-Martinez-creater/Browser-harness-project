"""Phase 1C controlled fault suite (C1-C8) + policy rules + regression.

Recovery is exercised through the real EpisodeRunner pipeline with the
deterministic rule policy (FailurePolicyEngine) and RecoveryManager wired in;
environment-side faults are injected with FaultInjectingEnvironmentAdapter.
Guarantees under test:

- a blocked action NEVER reaches env.step;
- a recovery environment action (WAIT_AND_REOBSERVE noop) is never an Agent
  step and never produces a StepRecord;
- the real environment result of a recovery (incl. termination) is kept;
- recovery is bounded by max_recoveries_per_episode (controlled abort);
- with recovery disabled the Phase 1B behavior is unchanged.
"""

import json
from pathlib import Path

import pytest

from web_harness.agents.baseline import BaselineAgent
from web_harness.config.loader import HarnessConfig
from web_harness.core.errors import ErrorType
from web_harness.core.models import RunStatus, TaskSpec
from web_harness.core.reliability import (
    FailureKind,
    FailureSeverity,
    FailureSignal,
    RecoveryKind,
    ReliabilityBudget,
    ReliabilityState,
)
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.evaluation.benchmark_runner import run_benchmark
from web_harness.evaluation.fault_injection import FaultInjectingEnvironmentAdapter
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.reliability.policy import FailurePolicyEngine, PolicyAction
from web_harness.reliability.verifier import DefaultStepVerifier
from web_harness.runtime.episode_runner import EpisodeRunner

TASK = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=6)
RESET_OBS = make_fake_observation(url="http://fake.local/start")
OBS_B = make_fake_observation(url="http://fake.local/B")


def signal(kind: FailureKind, sig: str, severity=FailureSeverity.ERROR):
    return FailureSignal(
        kind=kind, severity=severity, source="test", signature=sig
    )


def make_env(script, **injection):
    return FaultInjectingEnvironmentAdapter(
        FakeEnvironment(reset_observation=RESET_OBS, script=script),
        **injection,
    )


def make_runner(tmp_path, env, actions, *, max_steps=6, recovery=True):
    return EpisodeRunner(
        agent=BaselineAgent(model_adapter=MockModelAdapter(list(actions))),
        env=env,
        trace_root=tmp_path,
        verifier=DefaultStepVerifier(),
        verification_mode="shadow",
        failure_policy=FailurePolicyEngine(wait_ms=500) if recovery else None,
        recovery_budget=ReliabilityBudget(max_recoveries_per_episode=3),
    )


def events_of(result) -> list:
    return TraceRecorder.read_events(Path(result.trace_path))


def steps_of(result) -> list:
    return TraceRecorder.read_steps(Path(result.trace_path))


def outcomes(events, event_type) -> list[str]:
    return [
        e.outcome for e in events if e.event_type.value == event_type
    ]


# -- policy unit rules ---------------------------------------------------------


def test_policy_rules_are_fixed():
    policy = FailurePolicyEngine(wait_ms=500, block_steps=1)
    state, budget = ReliabilityState(), ReliabilityBudget(max_recoveries_per_episode=3)

    # TASK_FAILED -> ABORT, no directive, no reset
    d = policy.decide(
        signals=[signal(FailureKind.TASK_FAILED, "task_failed:reward_0")],
        failed_action="click(bid='1')",
        reliability_state=state, budget=budget,
    )
    assert d.action == PolicyAction.ABORT
    assert d.directive is None

    # ACTION_ERROR -> RECOVER(REDECIDE_WITH_FEEDBACK) blocking the failed action
    d = policy.decide(
        signals=[signal(FailureKind.ACTION_ERROR, "action_error:click:timeout")],
        failed_action="click(bid='1')",
        reliability_state=state, budget=budget,
    )
    assert d.action == PolicyAction.RECOVER
    assert d.directive.kind == RecoveryKind.REDECIDE_WITH_FEEDBACK
    assert d.directive.blocked_actions == ["click(bid='1')"]

    # OBSERVATION_INVALID -> RECOVER(WAIT_AND_REOBSERVE), no blocked action
    d = policy.decide(
        signals=[signal(FailureKind.OBSERVATION_INVALID, "observation_invalid:x")],
        failed_action="click(bid='1')",
        reliability_state=state, budget=budget,
    )
    assert d.directive.kind == RecoveryKind.WAIT_AND_REOBSERVE
    assert d.directive.wait_ms == 500
    assert d.directive.blocked_actions == []

    # LOOP_DETECTED -> RECOVER(BLOCK_REPEATED_ACTION)
    d = policy.decide(
        signals=[signal(FailureKind.LOOP_DETECTED, "loop:x")],
        failed_action="click(bid='1')",
        reliability_state=state, budget=budget,
    )
    assert d.directive.kind == RecoveryKind.BLOCK_REPEATED_ACTION
    assert d.directive.blocked_actions == ["click(bid='1')"]

    # single NO_PROGRESS -> CONTINUE
    d = policy.decide(
        signals=[signal(FailureKind.NO_PROGRESS, "no_progress:x",
                        severity=FailureSeverity.WARNING)],
        failed_action="click(bid='1')",
        reliability_state=state, budget=budget,
    )
    assert d.action == PolicyAction.CONTINUE

    # budget exhausted + ACTION_ERROR -> ABORT (no new recovery possible)
    state.recovery_count = 3
    d = policy.decide(
        signals=[signal(FailureKind.ACTION_ERROR, "action_error:click:timeout")],
        failed_action="click(bid='1')",
        reliability_state=state, budget=budget,
    )
    assert d.action == PolicyAction.ABORT

    # budget exhausted must NEVER kill normal execution (B2)
    state.recovery_count = budget.max_recoveries_per_episode

    # budget full + clean PASS -> CONTINUE
    d = policy.decide(
        signals=[], failed_action=None,
        reliability_state=state, budget=budget,
    )
    assert d.action == PolicyAction.CONTINUE

    # budget full + single NO_PROGRESS -> CONTINUE
    d = policy.decide(
        signals=[signal(FailureKind.NO_PROGRESS, "no_progress:x",
                        severity=FailureSeverity.WARNING)],
        failed_action="click(bid='1')",
        reliability_state=state, budget=budget,
    )
    assert d.action == PolicyAction.CONTINUE

    # budget full + OBSERVATION_INVALID -> ABORT
    d = policy.decide(
        signals=[signal(FailureKind.OBSERVATION_INVALID, "observation_invalid:x")],
        failed_action="click(bid='1')",
        reliability_state=state, budget=budget,
    )
    assert d.action == PolicyAction.ABORT

    # budget full + LOOP_DETECTED -> ABORT
    d = policy.decide(
        signals=[signal(FailureKind.LOOP_DETECTED, "loop:x")],
        failed_action="click(bid='1')",
        reliability_state=state, budget=budget,
    )
    assert d.action == PolicyAction.ABORT

    # recovery directives carry the trigger failure identity (R2)
    state.recovery_count = 0
    d = policy.decide(
        signals=[signal(FailureKind.ACTION_ERROR, "action_error:click:timeout")],
        failed_action="click(bid='1')",
        reliability_state=state, budget=budget,
    )
    assert d.directive.failure_kind == FailureKind.ACTION_ERROR
    assert d.directive.failure_signature == "action_error:click:timeout"


# -- C1: ACTION_ERROR once -> REDECIDE + block -> alternate action -> success --


def test_c1_action_error_redecide_with_feedback(tmp_path):
    env = make_env(
        [{}, {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        action_error_on_steps={0},
    )
    runner = make_runner(
        tmp_path, env, ["click(bid='1')", "click(bid='2')"]
    )
    result = runner.run(TASK, run_id="c1")

    assert result.success
    assert result.recovery_count == 1
    # REDECIDE_WITH_FEEDBACK never touches the environment
    assert result.recovery_environment_actions == 0
    assert result.recovery_success_count == 1
    assert result.recovered_episode
    assert result.num_steps == 2
    # the blocked action executed exactly once (before recovery); the agent
    # switched to an alternate action afterwards
    assert env.wrapped.executed_actions == ["click(bid='1')", "click(bid='2')"]

    events = events_of(result)
    assert "redecide_with_feedback" in outcomes(events, "recovery")
    assert "recover" in outcomes(events, "policy_decision")

    # the next decision received the typed recovery directive (separate from
    # repair_feedback) in its prompt
    steps = steps_of(result)
    assert steps[0].action_error is not None


def test_c1b_recovery_directive_reaches_prompt_not_normal_path(tmp_path):
    model = MockModelAdapter(["click(bid='1')", "click(bid='2')"])
    prompts: list[str] = []
    original = model.generate_action

    def spy(**kwargs):
        prompts.append(kwargs["prompt"].user)
        return original(**kwargs)

    model.generate_action = spy
    env = make_env(
        [{}, {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        action_error_on_steps={0},
    )
    runner = EpisodeRunner(
        agent=BaselineAgent(model_adapter=model),
        env=env,
        trace_root=tmp_path,
        verifier=DefaultStepVerifier(),
        failure_policy=FailurePolicyEngine(wait_ms=500),
        recovery_budget=ReliabilityBudget(max_recoveries_per_episode=3),
    )
    result = runner.run(TASK, run_id="c1b")
    assert result.success
    assert len(prompts) == 2
    # first decision (no directive yet): normal path untouched
    assert "# Reliability recovery feedback" not in prompts[0]
    # second decision received the typed recovery directive naming the block
    assert "# Reliability recovery feedback" in prompts[1]
    assert "click(bid='1')" in prompts[1]


# -- C2: EMPTY_OBSERVATION -> WAIT_AND_REOBSERVE (recovery env op, no step) ----


def test_c2_empty_observation_wait_and_reobserve(tmp_path):
    env = make_env(
        [{}, {}, {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        empty_observation_on_steps={0},
    )
    runner = make_runner(
        tmp_path, env, ["click(bid='1')", "click(bid='2')"]
    )
    result = runner.run(TASK, run_id="c2")

    assert result.success
    assert result.recovery_count == 1
    # exactly one harness-owned environment operation for the recovery
    assert result.recovery_environment_actions == 1
    assert result.recovery_latency_s >= 0.0
    # the recovery noop is NOT an agent step: 2 decisions -> 2 StepRecords
    assert result.num_steps == 2
    steps = steps_of(result)
    assert [s.action for s in steps] == ["click(bid='1')", "click(bid='2')"]
    # the recovery observation (fresh noop result) drove the next decision
    assert env.wrapped.executed_actions == [
        "click(bid='1')", "noop(wait_ms=500)", "click(bid='2')",
    ]

    events = events_of(result)
    recovery_events = [e for e in events if e.event_type.value == "recovery"]
    assert recovery_events[0].outcome == "wait_and_reobserve"
    for field in ("wait_ms", "action_error", "reward", "terminated", "truncated"):
        assert field in recovery_events[0].data
    assert recovery_events[0].data["wait_ms"] == 500


# -- C3: LOOP -> BLOCK_REPEATED_ACTION -> different action succeeds ------------


def test_c3_loop_blocked_repeated_action(tmp_path):
    # two identical transitions (same pre/post fingerprint + same action)
    # trigger LOOP_DETECTED; then a different action succeeds
    env = make_env(
        [{}, {}, {"reward": 1.0, "terminated": True, "observation": OBS_B}],
    )
    runner = make_runner(
        tmp_path, env,
        ["click(bid='1')", "click(bid='1')", "click(bid='2')"],
    )
    result = runner.run(TASK, run_id="c3")

    assert result.success
    assert result.recovery_count == 1
    assert result.recovered_episode
    assert result.num_steps == 3
    assert result.recovery_environment_actions == 0
    assert env.wrapped.executed_actions == [
        "click(bid='1')", "click(bid='1')", "click(bid='2')",
    ]
    events = events_of(result)
    assert "block_repeated_action" in outcomes(events, "recovery")


# -- C4: persistent ACTION_ERROR -> budget exhausted -> controlled abort -------


def test_c4_recovery_budget_exhausted_controlled_abort(tmp_path):
    env = make_env([{}], action_error_on_steps=set(range(20)))
    task = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=8)
    runner = make_runner(
        tmp_path, env,
        [f"click(bid='{i}')" for i in range(1, 9)], max_steps=8,
    )
    result = runner.run(task, run_id="c4")

    # deterministic abort at the budget boundary, never an unbounded loop
    assert not result.success
    assert result.error_type == ErrorType.RECOVERY_FAILED
    assert result.recovery_count == 3
    assert result.recovery_environment_actions == 0
    assert result.num_steps == 4
    # outcome accounting invariant (B3): the abort that refuses a 4th
    # recovery adds no phantom failure; success+failed+unresolved == count
    assert result.recovery_failed_count == 3
    assert result.recovery_unresolved_count == 0
    assert (
        result.recovery_success_count
        + result.recovery_failed_count
        + result.recovery_unresolved_count
        == result.recovery_count
    )


# -- C5: single NO_PROGRESS -> CONTINUE (no recovery) --------------------------


def test_c5_single_no_progress_continues(tmp_path):
    env = make_env([{}, {"reward": 1.0, "terminated": True, "observation": OBS_B}])
    runner = make_runner(tmp_path, env, ["click(bid='1')", "click(bid='2')"])
    result = runner.run(TASK, run_id="c5")

    assert result.success
    assert result.recovery_count == 0
    assert result.recovery_success_count == 0
    assert result.recovery_failed_count == 0
    assert result.recovery_environment_actions == 0
    assert not result.recovered_episode
    events = events_of(result)
    assert outcomes(events, "recovery") == []
    assert "continue" in outcomes(events, "policy_decision")


# -- C6: terminated + reward=0 -> final, no recovery, no reset -----------------


def test_c6_task_failed_is_final_no_recovery(tmp_path):
    env = make_env([{"terminated": True, "reward": 0.0}])
    runner = make_runner(tmp_path, env, ["click(bid='1')", "click(bid='2')"])
    result = runner.run(TASK, run_id="c6")

    assert result.status == RunStatus.FAILED
    assert not result.success
    assert result.error_type == ErrorType.TASK_TERMINATED
    # environment terminal is final: zero recoveries, zero extra env actions
    assert result.recovery_count == 0
    assert result.recovery_environment_actions == 0
    assert env.wrapped.executed_actions == ["click(bid='1')"]


# -- C7: blocked action selected again -> never reaches env.step ---------------


def test_c7_blocked_action_reselected_never_executed(tmp_path):
    env = make_env(
        [{}, {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        action_error_on_steps={0},
    )
    runner = make_runner(
        tmp_path, env,
        ["click(bid='1')", "click(bid='1')", "click(bid='2')"],
    )
    result = runner.run(TASK, run_id="c7")

    assert result.success
    # one actual recovery (the action-error directive); re-selecting the
    # blocked action is a separate re-decision, not a new recovery (R3)
    assert result.recovery_count == 1
    assert result.blocked_action_redecision_count == 1
    # outcome invariant: the single recovery resolved successfully
    assert result.recovery_success_count + result.recovery_failed_count \
        + result.recovery_unresolved_count == result.recovery_count
    assert result.num_steps == 2
    # the blocked action entered the environment exactly once: before it was
    # blocked. The re-selected blocked action NEVER reached env.step.
    assert env.wrapped.executed_actions.count("click(bid='1')") == 1
    assert env.wrapped.executed_actions == ["click(bid='1')", "click(bid='2')"]
    events = events_of(result)
    assert "blocked_action_selected" in outcomes(events, "recovery")
    steps = steps_of(result)
    assert steps[1].action == "click(bid='2')"


# -- C8: recovery noop terminates the task -> environment result is kept -------


def test_c8_recovery_noop_termination_accepted(tmp_path):
    env = make_env(
        [{}, {"reward": 1.0, "terminated": True}],
        empty_observation_on_steps={0},
    )
    runner = make_runner(tmp_path, env, ["click(bid='1')", "click(bid='2')"])
    result = runner.run(TASK, run_id="c8")

    # the harness never overrides the environment's terminal semantics
    assert result.success
    assert result.final_reward == 1.0
    assert result.recovery_count == 1
    assert result.recovery_environment_actions == 1
    assert result.recovered_episode
    # one agent decision only; the terminating noop was the recovery action
    assert result.num_steps == 1
    assert steps_of(result)[0].action == "click(bid='1')"


# -- C12: pending recovery finalized (as unresolved) when the episode ends -----


def test_c12_pending_recovery_finalized_not_silently_lost(tmp_path):
    # recovery on the LAST step: the 2-step outcome window outlives the
    # episode, so the outcome must surface as explicitly unresolved (B3)
    env = make_env(
        [{}],
        action_error_on_steps={0},
    )
    task = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=1)
    runner = make_runner(tmp_path, env, ["click(bid='1')"], max_steps=1)
    result = runner.run(task, run_id="c12")

    assert result.recovery_count == 1
    assert result.recovery_success_count == 0
    assert result.recovery_failed_count == 0
    assert result.recovery_unresolved_count == 1
    assert (
        result.recovery_success_count
        + result.recovery_failed_count
        + result.recovery_unresolved_count
        == result.recovery_count
    )


# -- C13 / R1: WAIT recovery itself restores state -> outcome correct ----------


def test_c13_wait_recovery_fingerprint_reference(tmp_path):
    # invalid observation -> WAIT_AND_REOBSERVE -> the recovery noop itself
    # restores a valid page; the NEXT agent action changes nothing. The
    # outcome must still be success because the fingerprint moved away from
    # the recovery-start fingerprint (R1), not because of the last action.
    env = make_env(
        [{}, {"observation": OBS_B}, {"reward": 1.0, "terminated": True,
                                     "observation": OBS_B}],
        empty_observation_on_steps={0},
    )
    runner = make_runner(
        tmp_path, env, ["click(bid='1')", "click(bid='2')"]
    )
    result = runner.run(TASK, run_id="c13")

    assert result.success
    assert result.recovery_count == 1
    assert result.recovery_environment_actions == 1
    # recovery-start fingerprint (invalid obs) != current fingerprint
    # (valid obs), even though the last agent action changed nothing
    assert result.recovery_success_count == 1
    assert result.recovery_failed_count == 0
    assert result.recovery_unresolved_count == 0
    assert (
        result.recovery_success_count
        + result.recovery_failed_count
        + result.recovery_unresolved_count
        == result.recovery_count
    )


# -- R4: TASK_FAILED short-circuit produces a deterministic policy event -------


def test_c6b_task_failed_short_circuit_policy_event(tmp_path):
    env = make_env([{"terminated": True, "reward": 0.0}])
    runner = make_runner(tmp_path, env, ["click(bid='1')"])
    result = runner.run(TASK, run_id="c6b")

    assert result.status == RunStatus.FAILED
    events = events_of(result)
    aborts = [
        e for e in events
        if e.event_type.value == "policy_decision" and e.outcome == "abort"
    ]
    assert len(aborts) == 1
    assert aborts[0].data.get("short_circuited") is True
    assert aborts[0].data.get("failure_kind") == "TASK_FAILED"


# -- B1: canonical recovery budget path used by runtime/config/renderer --------


def test_b1_recovery_budget_single_config_path():
    from web_harness.core.reliability import default_budget_from_config

    # canonical: reliability.recovery.max_recoveries_per_episode
    budget = default_budget_from_config({
        "recovery": {"max_recoveries_per_episode": 7},
        "budget": {"max_recoveries_per_episode": 99},
    })
    assert budget.max_recoveries_per_episode == 7


def test_b1_recovery_config_fail_fast():
    from web_harness.core.errors import ConfigError

    base = {
        "model": {"provider": "mock"},
        "agent": {"type": "baseline"},
        "runtime": {"max_steps": 5},
        "trace": {"root_dir": "runs"},
        "environment": {},
        "reliability": {
            "enabled": True,
            "recovery": {"enabled": True},
        },
    }
    HarnessConfig(base)  # valid

    bad_cases = [
        ("reliability.recovery.enabled must be a boolean",
         {"recovery": {"enabled": "yes"}}),
        ("max_recoveries_per_episode must be an int >= 0",
         {"recovery": {"enabled": True, "max_recoveries_per_episode": -1}}),
        ("wait_ms must be an int >= 0",
         {"recovery": {"enabled": True, "wait_ms": -5}}),
        ("block_steps must be an int >= 1",
         {"recovery": {"enabled": True, "block_steps": 0}}),
    ]
    for _msg, recovery in bad_cases:
        data = {**base, "reliability": {"enabled": True, **recovery}}
        with pytest.raises(ConfigError):
            HarnessConfig(data)


# -- regression: recovery disabled -> Phase 1B behavior unchanged --------------


def test_regression_recovery_disabled_phase1b_unchanged(tmp_path):
    # same C1 fault injection, but no failure policy: identical to Phase 1B
    env = make_env(
        [{}, {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        action_error_on_steps={0},
    )
    runner = make_runner(
        tmp_path, env, ["click(bid='1')", "click(bid='2')"], recovery=False
    )
    result = runner.run(TASK, run_id="reg")

    assert result.success
    assert result.num_steps == 2
    assert result.recovery_count == 0
    assert result.recovery_success_count == 0
    assert result.recovery_failed_count == 0
    assert result.recovery_environment_actions == 0
    assert not result.recovered_episode
    events = events_of(result)
    assert outcomes(events, "recovery") == []
    assert outcomes(events, "policy_decision") == []
    assert env.wrapped.executed_actions == ["click(bid='1')", "click(bid='2')"]


def test_regression_benchmark_runner_recovery_wiring(tmp_path):
    # config-level: recovery disabled -> no recovery in results; recovery
    # enabled config runs cleanly on a failure-free environment
    def make_cfg(recovery_enabled: bool) -> HarnessConfig:
        data = {
            "model": {"provider": "mock", "mock_actions": ["click(bid='1')"]},
            "agent": {"type": "baseline", "max_history_steps": 4},
            "runtime": {"max_steps": 5},
            "trace": {"root_dir": str(tmp_path / "runs")},
            "environment": {"bootstrap_action": None},
            "reliability": {
                "enabled": True,
                "verification": {"enabled": True, "mode": "shadow"},
                "recovery": {"enabled": recovery_enabled, "wait_ms": 500},
                "budget": {"max_recoveries_per_episode": 3},
            },
        }
        return HarnessConfig(data)

    def success_env_factory():
        return make_env(
            [{"reward": 1.0, "terminated": True, "observation": OBS_B}]
        )

    def agent_factory():
        return BaselineAgent(model_adapter=MockModelAdapter(["click(bid='1')"]))

    _, results_off = run_benchmark(
        make_cfg(False), benchmark_name="fake", tasks=["t"], seeds=[0],
        experiments_root=tmp_path / "exp-off",
        environment_factory=success_env_factory, agent_factory=agent_factory,
    )
    assert results_off[0].recovery_count == 0
    summary_off = json.loads(
        (next((tmp_path / "exp-off").glob("*/summary.json"))).read_text(encoding="utf-8")
    )
    assert summary_off["config"]["reliability"]["recovery"]["enabled"] is False
    assert summary_off["aggregate"]["total_recovery_count"] == 0

    _, results_on = run_benchmark(
        make_cfg(True), benchmark_name="fake", tasks=["t"], seeds=[0],
        experiments_root=tmp_path / "exp-on",
        environment_factory=success_env_factory, agent_factory=agent_factory,
    )
    # failure-free environment: policy never triggers, plumbing stays silent
    assert results_on[0].recovery_count == 0
    summary_on = json.loads(
        (next((tmp_path / "exp-on").glob("*/summary.json"))).read_text(encoding="utf-8")
    )
    assert summary_on["config"]["reliability"]["recovery"]["enabled"] is True
    assert summary_on["aggregate"]["total_recovery_count"] == 0
