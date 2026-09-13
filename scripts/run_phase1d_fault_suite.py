"""Run the Phase 1D controlled-fault suite (D1-D10) and emit machine results.

Usage:
    uv run python scripts/run_phase1d_fault_suite.py --output-dir reports/phase1

Each scenario runs a full episode with FakeEnvironment, fault injection and a
ScriptedStructuredModel as the provider-neutral Replanner model. The JSON
output records per-scenario assertions; the Markdown report is generated from
that JSON (never hand-written). These fault results are NOT mixed with the
natural MiniWoB smoke success rate.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.errors import ModelApiError
from web_harness.core.models import ActionDecision, ModelOutput, TaskSpec
from web_harness.core.reliability import ReliabilityBudget
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.evaluation.fault_injection import (
    FaultInjectingEnvironmentAdapter,
)
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.reliability.policy import FailurePolicyEngine
from web_harness.reliability.replan_policy import ReplanTriggerPolicy
from web_harness.reliability.replanner import ReplanExecutor
from web_harness.reliability.retry import RetryPolicy
from web_harness.reliability.verifier import DefaultStepVerifier
from web_harness.runtime.decision_executor import DecisionExecutor
from web_harness.runtime.episode_runner import EpisodeRunner

RESET_OBS = make_fake_observation(url="http://fault.local/start")
OBS_B = make_fake_observation(url="http://fault.local/B")

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


class ScriptedStructuredModel:
    """Deterministic replan model: scripted raw texts / exceptions."""

    def __init__(self, plan_jsons):
        self.plan_jsons = list(plan_jsons)
        self.call_count = 0
        self.prompts: list[str] = []

    def generate_structured(self, *, prompt):
        self.prompts.append(prompt.user)
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
    return FaultInjectingEnvironmentAdapter(
        FakeEnvironment(reset_observation=RESET_OBS, script=script),
        **injection,
    )


def run_episode(
    *,
    name: str,
    script: list[dict],
    actions: list[str],
    injection: dict | None = None,
    max_steps: int = 8,
    max_replans: int = 1,
    max_recoveries: int = 3,
    horizon: int = 3,
    replanning: bool = True,
    plan_jsons: list | None = None,
    api_max_retries: int = 1,
    max_extra_model_calls: int = 6,
    decision_parse_error_calls: tuple[int, ...] = (),
    tmp_root: Path,
):
    task = TaskSpec(benchmark="fake", task_id="fault", seed=0,
                    max_steps=max_steps)
    env = make_env(script, **(injection or {}))
    budget = ReliabilityBudget(
        max_recoveries_per_episode=max_recoveries,
        max_replans_per_episode=max_replans,
        recovery_failures_before_replan=2,
        plan_horizon_steps=horizon,
        recent_steps=6,
        max_extra_model_calls_per_episode=max_extra_model_calls,
    )
    structured = ScriptedStructuredModel(
        plan_jsons if plan_jsons is not None else [PLAN_JSON]
    )
    agent_model = MockModelAdapter(list(actions))
    if decision_parse_error_calls:
        from web_harness.evaluation.fault_injection import (
            FaultInjectingModelAdapter,
        )

        agent_model = FaultInjectingModelAdapter(
            MockModelAdapter(list(actions)),
            parse_error_on_calls=set(decision_parse_error_calls),
        )
    runner = EpisodeRunner(
        agent=BaselineAgent(model_adapter=agent_model),
        env=env,
        trace_root=tmp_root,
        verifier=DefaultStepVerifier(),
        failure_policy=FailurePolicyEngine(wait_ms=500),
        recovery_budget=budget,
        decision_executor=DecisionExecutor(
            retry_policy=RetryPolicy(
                api_max_retries=api_max_retries,
                api_backoff_ms=[0],
                parse_max_retries=1,
            ),
            budget=budget,
            sleep=lambda _s: None,
        ),
        replan_trigger_policy=ReplanTriggerPolicy() if replanning else None,
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
    with tempfile.TemporaryDirectory() as td:
        runner.trace_root = Path(td)
        result = runner.run(task, run_id=f"fault-{name.lower()}")
        events = TraceRecorder.read_events(Path(result.trace_path))
        steps = TraceRecorder.read_steps(Path(result.trace_path))
        # collect artifact names BEFORE the tmp dir disappears
        artifact_names = sorted(
            a.name for a in Path(result.trace_path).glob("artifacts/*.json")
        )
        attempt_types = [
            json.loads(a.read_text(encoding="utf-8")).get("failure_type")
            for a in Path(result.trace_path).glob("artifacts/*attempt_*.json")
        ]
    return (result, env.wrapped.executed_actions, events, steps,
            artifact_names, attempt_types, structured.prompts)


def outcomes(events, event_type):
    return [e.outcome for e in events if e.event_type.value == event_type]


def replan_invariant(result):
    return (
        result.replan_success_count
        + result.replan_failed_count
        + result.replan_unresolved_count
        == result.replan_count
    )


def recovery_invariant(result):
    return (
        result.recovery_success_count
        + result.recovery_failed_count
        + result.recovery_unresolved_count
        == result.recovery_count
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/phase1")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []

    def add(name, expected, assertions, **extra):
        data = {"scenario": name, "expected": expected,
                "assertions": assertions, **extra}
        scenarios.append(data)

    # D1: repeated loop after failed recovery -> replan -> success
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D1",
        script=[{}, {}, {}, {}, {}, {"observation": OBS_B},
                {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='3')",
                 "click(bid='9')"],
        tmp_root=out_dir,
    )
    add(
        "D1 repeated loop after failed recovery",
        "replan_count=1, replan succeeds, one plan artifact, one REPLAN event",
        {
            "success": result.success,
            "replan_count_is_1": result.replan_count == 1,
            "replan_success": result.replan_success_count == 1,
            "replan_model_calls_is_1": result.replan_model_calls == 1,
            "one_artifact": sum(
                1 for n in artifact_names if n.startswith("replan_")
            ) == 1,
            "triggered_event": "triggered" in outcomes(events, "replan"),
            "replan_invariant": replan_invariant(result),
            "recovery_invariant": recovery_invariant(result),
        },
        replan_count=result.replan_count,
        replan_success_count=result.replan_success_count,
        replan_model_calls=result.replan_model_calls,
    )

    # D2: two consecutive recovery failures trigger replan (streak)
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D2",
        script=[{}] * 7
        + [{"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=[f"click(bid='{i}')" for i in range(1, 9)],
        injection={"action_error_on_steps": {0, 2, 5}},
        tmp_root=out_dir,
    )
    triggered = [
        e for e in events
        if e.event_type.value == "replan" and e.outcome == "triggered"
    ]
    add(
        "D2 recovery failure streak triggers replan",
        "streak>=2 plus a new recoverable failure escalates deterministically",
        {
            "success": result.success,
            "replan_count_is_1": result.replan_count == 1,
            "replan_success": result.replan_success_count == 1,
            "streak_reason": triggered
            and triggered[0].data.get("reason") == "recovery_failure_streak",
            "streak_value_2": triggered
            and triggered[0].data.get("recovery_failure_streak") == 2,
            "recovery_failed_is_2": result.recovery_failed_count == 2,
            "replan_invariant": replan_invariant(result),
        },
        replan_count=result.replan_count,
    )

    # D3: replanning disabled -> no REPLAN anything (causal control)
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D3",
        script=[{}, {}, {}, {}, {}, {"observation": OBS_B},
                {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='3')",
                 "click(bid='9')"],
        replanning=False,
        tmp_root=out_dir,
    )
    add(
        "D3 replanning disabled causal control",
        "identical trajectory without the replanner: no REPLAN events, no plan",
        {
            "no_replan_events": outcomes(events, "replan") == [],
            "replan_count_is_0": result.replan_count == 0,
            "no_plan_artifacts": not any(
                n.startswith("replan_") for n in artifact_names
            ),
            "recovery_invariant": recovery_invariant(result),
        },
        replan_count=result.replan_count,
    )

    # D4: replan budget exhausted does not kill execution
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D4",
        script=[{}] * 9
        + [{"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=[f"click(bid='{i}')" for i in range(1, 11)],
        injection={"action_error_on_steps": {0, 2, 5, 8}},
        max_steps=10,
        max_replans=1,
        max_recoveries=5,
        tmp_root=out_dir,
    )
    add(
        "D4 replan budget exhausted falls back and continues",
        "after the single replan: clean steps CONTINUE, recoverable failures "
        "fall back to Phase 1C recovery",
        {
            "success": result.success,
            "replan_count_is_1": result.replan_count == 1,
            "no_policy_abort": "abort" not in outcomes(events, "policy_decision"),
            "fallback_recoveries": result.recovery_count == 3,
            "replan_invariant": replan_invariant(result),
        },
        replan_count=result.replan_count,
        recovery_count=result.recovery_count,
    )

    # D5: plan horizon consumed only by real agent steps
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D5",
        script=[{}] * 8
        + [{"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='3')",
                 "click(bid='4')", "click(bid='9')"],
        injection={"empty_observation_on_steps": {5}},
        max_steps=8,
        horizon=2,
        tmp_root=out_dir,
    )
    created = [
        e for e in events
        if e.event_type.value == "replan" and e.outcome == "created"
    ]
    add(
        "D5 plan horizon expires",
        "horizon=2 expires after two real agent steps; the WAIT recovery "
        "noop inside the horizon does not consume it",
        {
            "success": result.success,
            "replan_count_is_1": result.replan_count == 1,
            "replan_success": result.replan_success_count == 1,
            "wait_recovery_present": "wait_and_reobserve"
            in outcomes(events, "recovery"),
            "plan_created_once": len(created) == 1,
            "replan_invariant": replan_invariant(result),
        },
        replan_count=result.replan_count,
    )

    # D6: same trigger failure repeats under active plan -> replan FAILED
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D6",
        script=[{}] * 7
        + [{"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='1')",
                 "click(bid='3')", "click(bid='9')"],
        tmp_root=out_dir,
    )
    add(
        "D6 same failure repeats under active plan",
        "the trigger signature reappearing fails the replan and deactivates "
        "the plan; max_replans=1 prevents a second intervention",
        {
            "success": result.success,
            "replan_count_is_1": result.replan_count == 1,
            "replan_failed_is_1": result.replan_failed_count == 1,
            "one_triggered_event": outcomes(events, "replan").count(
                "triggered"
            )
            == 1,
            "replan_invariant": replan_invariant(result),
        },
        replan_count=result.replan_count,
    )

    # D7: replan generation parse repair
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D7",
        script=[{}, {}, {}, {}, {}, {"observation": OBS_B},
                {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='3')",
                 "click(bid='9')"],
        plan_jsons=[BAD_PLAN_JSON, PLAN_JSON],
        tmp_root=out_dir,
    )
    add(
        "D7 replan generation parse repair",
        "attempt 0 malformed JSON -> one repair retry -> plan activated; "
        "failed attempt artifact + token usage retained",
        {
            "success": result.success,
            "replan_count_is_1": result.replan_count == 1,
            "model_calls_is_2": result.replan_model_calls == 2,
            "replan_success": result.replan_success_count == 1,
            "tokens_retained": result.replan_input_tokens >= 100,
            "failed_attempt_artifact": any(
                t == "REPLAN_OUTPUT_PARSE_ERROR" for t in attempt_types
            ),
            "repair_prompt_used": len(planner_prompts) == 2
            and "# Format correction" not in planner_prompts[0]
            and "# Format correction" in planner_prompts[1],
            "replan_invariant": replan_invariant(result),
        },
        replan_count=result.replan_count,
        replan_model_calls=result.replan_model_calls,
    )

    # D11: budget=0 blocks the initial replan call
    result, executed, events, steps, artifact_names, attempt_types, \
        planner_prompts = run_episode(
        name="D11",
        script=[{}] * 6
        + [{"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='3')",
                 "click(bid='9')"],
        max_extra_model_calls=0,
        tmp_root=out_dir,
    )
    add(
        "D11 global budget=0 blocks the initial replan call",
        "trigger fires but zero planner model calls happen; execution falls "
        "back to Phase 1C recovery",
        {
            "replan_count_is_1": result.replan_count == 1,
            "zero_planner_calls": result.replan_model_calls == 0,
            "replan_failed_is_1": result.replan_failed_count == 1,
            "no_replan_prompt_sent": len(planner_prompts) == 0,
            "replan_invariant": replan_invariant(result),
        },
        replan_count=result.replan_count,
        replan_model_calls=result.replan_model_calls,
    )

    # D12: planner initial call consumes budget before parse retry
    result, executed, events, steps, artifact_names, attempt_types, \
        planner_prompts = run_episode(
        name="D12",
        script=[{}] * 6
        + [{"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='3')",
                 "click(bid='9')"],
        max_extra_model_calls=1,
        plan_jsons=["not json at all", PLAN_JSON],
        tmp_root=out_dir,
    )
    add(
        "D12 planner initial call consumes budget before parse retry",
        "with budget=1 the initial call runs but the parse repair cannot",
        {
            "replan_count_is_1": result.replan_count == 1,
            "model_calls_is_1": result.replan_model_calls == 1,
            "replan_failed_is_1": result.replan_failed_count == 1,
            "replan_invariant": replan_invariant(result),
        },
        replan_count=result.replan_count,
        replan_model_calls=result.replan_model_calls,
    )

    # D13: replan parse repair prompt actually changes
    result, executed, events, steps, artifact_names, attempt_types, \
        planner_prompts = run_episode(
        name="D13",
        script=[{}, {}, {}, {}, {}, {"observation": OBS_B},
                {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='3')",
                 "click(bid='9')"],
        plan_jsons=["bad replan json", PLAN_JSON],
        tmp_root=out_dir,
    )
    add(
        "D13 replan parse repair prompt actually changes",
        "the second planner call carries the typed format-repair section",
        {
            "two_prompts": len(planner_prompts) == 2,
            "first_prompt_clean": planner_prompts
            and "# Format correction" not in planner_prompts[0],
            "second_prompt_has_repair": planner_prompts
            and "# Format correction" in planner_prompts[1],
            "replan_success": result.replan_success_count == 1,
            "replan_invariant": replan_invariant(result),
        },
        replan_count=result.replan_count,
    )

    # D14: decision + replan failed attempts on the same step do not collide
    result, executed, events, steps, artifact_names, attempt_types, \
        planner_prompts = run_episode(
        name="D14",
        script=[{}, {}, {}, {}, {}, {"observation": OBS_B},
                {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='3')",
                 "click(bid='9')"],
        decision_parse_error_calls=(0,),
        plan_jsons=["bad replan json", PLAN_JSON],
        tmp_root=out_dir,
    )
    add(
        "D14 decision+replan attempt artifacts do not collide",
        "same-step decision and replan parse failures produce two distinct "
        "artifacts with correct event refs",
        {
            "decision_artifact_present": any(
                n.startswith("attempt_") for n in artifact_names
            ),
            "replan_artifact_present": any(
                n.startswith("replan_attempt_") for n in artifact_names
            ),
            "both_parse_failures_recorded": attempt_types.count(
                "REPLAN_OUTPUT_PARSE_ERROR"
            )
            >= 1
            and any(t == "MODEL_OUTPUT_PARSE_ERROR" for t in attempt_types),
            "success": result.success,
        },
        replan_count=result.replan_count,
    )

    # D15: empty plan fields go through structured parse repair
    result, executed, events, steps, artifact_names, attempt_types, \
        planner_prompts = run_episode(
        name="D15",
        script=[{}, {}, {}, {}, {}, {"observation": OBS_B},
                {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='3')",
                 "click(bid='9')"],
        plan_jsons=[
            json.dumps({
                "diagnosis": "",
                "immediate_subgoal": "   ",
                "strategy_steps": [],
                "horizon_steps": 0,
            }),
            PLAN_JSON,
        ],
        tmp_root=out_dir,
    )
    add(
        "D15 invalid plan fields go through parse repair",
        "empty diagnosis/subgoal or horizon<1 fails validation and the "
        "repair retry produces a valid plan",
        {
            "model_calls_is_2": result.replan_model_calls == 2,
            "replan_success_is_1": result.replan_success_count == 1,
            "repair_prompt_used": len(planner_prompts) == 2
            and "# Format correction" in planner_prompts[1],
            "replan_invariant": replan_invariant(result),
        },
        replan_count=result.replan_count,
        replan_model_calls=result.replan_model_calls,
    )

    # D8a: transient API failure then success
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D8a",
        script=[{}, {}, {}, {}, {}, {"observation": OBS_B},
                {"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')",
                 "click(bid='1')", "click(bid='1')", "click(bid='3')",
                 "click(bid='9')"],
        plan_jsons=[ModelApiError("transient", transient=True), PLAN_JSON],
        tmp_root=out_dir,
    )
    add(
        "D8a transient API failure retries then succeeds",
        "one transient API error -> controlled retry -> plan created",
        {
            "success": result.success,
            "replan_count_is_1": result.replan_count == 1,
            "model_calls_is_2": result.replan_model_calls == 2,
            "replan_success": result.replan_success_count == 1,
            "replan_invariant": replan_invariant(result),
        },
        replan_count=result.replan_count,
    )

    # D8b: persistent API failure -> generation failed -> fallback recovery
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D8b",
        script=[{}] * 6
        + [{"reward": 1.0, "terminated": True, "observation": OBS_B}],
        actions=[f"click(bid='{i}')" for i in range(1, 8)],
        injection={"action_error_on_steps": {0, 2, 5}},
        plan_jsons=[ModelApiError("down", transient=True)],
        api_max_retries=1,
        tmp_root=out_dir,
    )
    add(
        "D8b persistent replan API failure falls back",
        "generation fails (no browser action from replanner), replan FAILED "
        "outcome, execution falls back to Phase 1C recovery",
        {
            "generation_failed_event": "generation_failed"
            in outcomes(events, "replan"),
            "replan_count_is_1": result.replan_count == 1,
            "replan_failed_is_1": result.replan_failed_count == 1,
            "model_calls_is_2": result.replan_model_calls == 2,
            "no_plan_artifact": not any(
                n.startswith("replan_") for n in artifact_names
            ),
            "replan_invariant": replan_invariant(result),
            "recovery_invariant": recovery_invariant(result),
        },
        replan_count=result.replan_count,
    )

    # D9: terminal task never replans
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D9",
        script=[{"terminated": True, "reward": 0.0}],
        actions=["click(bid='1')", "click(bid='2')"],
        tmp_root=out_dir,
    )
    add(
        "D9 terminal task never replans",
        "terminated=true reward<=0: zero replans, zero planner model calls",
        {
            "failed": result.status.value == "failed",
            "replan_count_is_0": result.replan_count == 0,
            "no_replan_events": outcomes(events, "replan") == [],
        },
        replan_count=result.replan_count,
    )

    # D10: unresolved recovery is neutral for the streak
    result, executed, events, steps, artifact_names, attempt_types, planner_prompts = run_episode(
        name="D10",
        script=[{}, {}],
        actions=["click(bid='1')", "click(bid='1')"],
        max_steps=2,
        tmp_root=out_dir,
    )
    add(
        "D10 unresolved recovery is neutral",
        "an episode-outlived recovery outcome is UNRESOLVED: no streak "
        "increment, no replan",
        {
            "recovery_unresolved_is_1": result.recovery_unresolved_count == 1,
            "recovery_failed_is_0": result.recovery_failed_count == 0,
            "replan_count_is_0": result.replan_count == 0,
            "recovery_invariant": recovery_invariant(result),
        },
        replan_count=result.replan_count,
    )

    for s in scenarios:
        s["pass"] = all(s["assertions"].values())
        s["failed_assertions"] = [
            k for k, v in s["assertions"].items() if not v
        ]

    payload = {
        "suite": "phase1d_fault_suite",
        "scenarios": scenarios,
        "all_pass": all(s["pass"] for s in scenarios),
    }
    json_path = out_dir / "phase1d_fault_suite.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Phase 1D Controlled Fault Suite Report",
        "",
        "Auto-generated from `phase1d_fault_suite.json` — deterministic",
        "replanning scenarios on FakeEnvironment + MockModelAdapter + a",
        "ScriptedStructuredModel (provider-neutral, no real model calls).",
        "These results are NOT mixed with natural MiniWoB success rates.",
        "",
        "| scenario | pass | status | replans | replan model calls |",
        "|---|---|---|---:|---:|",
    ]
    for s in scenarios:
        lines.append(
            f"| {s['scenario']} | {'PASS' if s['pass'] else 'FAIL'} "
            f"| {s.get('status', '-')} | {s.get('replan_count', 0)} "
            f"| {s.get('replan_model_calls', 0)} |"
        )
    lines.append("")
    lines.append(
        f"**Suite result: "
        f"{'ALL PASS' if payload['all_pass'] else 'FAILURES PRESENT'}**"
    )
    lines.append("")
    lines.append(
        "Guarantees exercised: escalation is deterministic and happens only "
        "on the Phase 1C RECOVER path for ACTION_ERROR/LOOP_DETECTED; the "
        "RecoveryPlan is advisory prompt context (never executed); the plan "
        "horizon is consumed only by real Agent StepRecords; replan model "
        "calls reuse Phase 1B retry protections and cost accounting; "
        "outcome invariant success + failed + unresolved == replan_count."
    )
    lines.append("")
    for s in scenarios:
        if not s["pass"]:
            lines.append(f"- {s['scenario']} failed: {s['failed_assertions']}")
    (out_dir / "phase1d_fault_suite_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"fault suite written to {json_path} (all_pass={payload['all_pass']})")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
