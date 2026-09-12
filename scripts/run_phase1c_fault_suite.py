"""Run the Phase 1C controlled-fault suite (C1-C8) and emit machine results.

Usage:
    uv run python scripts/run_phase1c_fault_suite.py --output-dir reports/phase1

Each scenario runs a full episode with FaultInjectingEnvironmentAdapter
(environment-side faults) against a FakeEnvironment, with the rule-based
FailurePolicyEngine and RecoveryManager wired into the EpisodeRunner. The
JSON output records per-scenario assertions; the Markdown report is
generated from that JSON (never hand-written). These fault results are NOT
mixed with the natural MiniWoB smoke success rate.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.errors import ErrorType
from web_harness.core.models import RunStatus, TaskSpec
from web_harness.core.reliability import ReliabilityBudget
from web_harness.env.fake import make_fake_observation
from web_harness.evaluation.fault_injection import (
    FaultInjectingEnvironmentAdapter,
)
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.reliability.policy import FailurePolicyEngine
from web_harness.reliability.verifier import DefaultStepVerifier
from web_harness.runtime.episode_runner import EpisodeRunner

RESET_OBS = make_fake_observation(url="http://fault.local/start")
OBS_B = make_fake_observation(url="http://fault.local/B")


def make_env(script, **injection) -> FaultInjectingEnvironmentAdapter:
    from web_harness.env.fake import FakeEnvironment

    return FaultInjectingEnvironmentAdapter(
        FakeEnvironment(reset_observation=RESET_OBS, script=script),
        **injection,
    )


def run_recovery_episode(
    *,
    name: str,
    script: list[dict],
    actions: list[str],
    injection: dict | None = None,
    max_steps: int = 6,
    budget: int = 3,
    tmp_root: Path,
) -> tuple[dict, list[str], list]:
    task = TaskSpec(benchmark="fake", task_id="fault", seed=0, max_steps=max_steps)
    env = make_env(script, **(injection or {}))
    runner = EpisodeRunner(
        agent=BaselineAgent(model_adapter=MockModelAdapter(list(actions))),
        env=env,
        trace_root=tmp_root,
        verifier=DefaultStepVerifier(),
        verification_mode="shadow",
        failure_policy=FailurePolicyEngine(wait_ms=500),
        recovery_budget=ReliabilityBudget(max_recoveries_per_episode=budget),
    )
    with tempfile.TemporaryDirectory() as td:
        runner.trace_root = Path(td)
        result = runner.run(task, run_id=f"fault-{name.lower()}")
        events = TraceRecorder.read_events(Path(result.trace_path))
        steps = TraceRecorder.read_steps(Path(result.trace_path))
    return result, env.wrapped.executed_actions, (events, steps)


def event_outcomes(events, event_type: str) -> list[str]:
    return [
        e.outcome for e in events if e.event_type.value == event_type
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/phase1")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    success_script = [{}, {"reward": 1.0, "terminated": True, "observation": OBS_B}]

    # C1: ACTION_ERROR once -> REDECIDE_WITH_FEEDBACK -> alternate action
    result, executed, (events, steps) = run_recovery_episode(
        name="C1", script=success_script,
        actions=["click(bid='1')", "click(bid='2')"],
        injection={"action_error_on_steps": {0}},
        tmp_root=out_dir,
    )
    data = {
        "scenario": "C1 action_error -> redecide_with_feedback",
        "status": result.status.value,
        "error_type": result.error_type.value if result.error_type else None,
        "recovery_count": result.recovery_count,
        "recovery_success_count": result.recovery_success_count,
        "recovery_environment_actions": result.recovery_environment_actions,
        "recovered_episode": result.recovered_episode,
        "steps": result.num_steps,
        "executed_actions": executed,
        "expected": "success; 1 recovery (no env action); blocked action "
                    "executed once, then alternate action",
        "assertions": {
            "success": result.success,
            "recovery_count_is_1": result.recovery_count == 1,
            "no_recovery_env_actions": result.recovery_environment_actions == 0,
            "recovery_success_counted": result.recovery_success_count == 1,
            "recovered_episode": result.recovered_episode,
            "blocked_executed_once": executed.count("click(bid='1')") == 1,
            "redecide_event_recorded":
                "redecide_with_feedback" in event_outcomes(events, "recovery"),
        },
    }
    scenarios.append(data)

    # C2: EMPTY_OBSERVATION -> WAIT_AND_REOBSERVE (recovery env op, no step)
    result, executed, (events, steps) = run_recovery_episode(
        name="C2", script=[{}, {}, {"reward": 1.0, "terminated": True,
                                    "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='2')"],
        injection={"empty_observation_on_steps": {0}},
        tmp_root=out_dir,
    )
    recovery_events = [e for e in events if e.event_type.value == "recovery"]
    data = {
        "scenario": "C2 empty_observation -> wait_and_reobserve",
        "status": result.status.value,
        "recovery_count": result.recovery_count,
        "recovery_environment_actions": result.recovery_environment_actions,
        "steps": result.num_steps,
        "executed_actions": executed,
        "recovery_event_data_keys": sorted(recovery_events[0].data.keys())
        if recovery_events else [],
        "expected": "success; recovery noop counts as env action but NOT as "
                    "an agent step; recovery observation drives next decision",
        "assertions": {
            "success": result.success,
            "one_recovery": result.recovery_count == 1,
            "one_recovery_env_action": result.recovery_environment_actions == 1,
            "noop_is_not_a_step": result.num_steps == 2,
            "step_actions_are_agent_actions": [s.action for s in steps] == [
                "click(bid='1')", "click(bid='2')",
            ],
            "recovery_observation_used": "noop(wait_ms=500)" in executed,
            "event_has_env_semantics": all(
                k in recovery_events[0].data
                for k in ("wait_ms", "action_error", "reward",
                          "terminated", "truncated")
            ) if recovery_events else False,
        },
    }
    scenarios.append(data)

    # C3: LOOP -> BLOCK_REPEATED_ACTION -> different action succeeds
    result, executed, (events, steps) = run_recovery_episode(
        name="C3", script=[{}, {}, {"reward": 1.0, "terminated": True,
                                    "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')"],
        tmp_root=out_dir,
    )
    data = {
        "scenario": "C3 loop -> block_repeated_action",
        "status": result.status.value,
        "recovery_count": result.recovery_count,
        "recovered_episode": result.recovered_episode,
        "steps": result.num_steps,
        "executed_actions": executed,
        "expected": "success; loop detected on the repeated transition; the "
                    "agent switches to a different action",
        "assertions": {
            "success": result.success,
            "one_recovery": result.recovery_count == 1,
            "recovered_episode": result.recovered_episode,
            "no_recovery_env_actions": result.recovery_environment_actions == 0,
            "block_event_recorded":
                "block_repeated_action" in event_outcomes(events, "recovery"),
            "repeated_action_allowed_once_then_switch":
                executed == ["click(bid='1')", "click(bid='1')", "click(bid='2')"],
        },
    }
    scenarios.append(data)

    # C4: persistent ACTION_ERROR -> recovery budget exhausted -> abort
    result, executed, (events, steps) = run_recovery_episode(
        name="C4", script=[{}],
        actions=[f"click(bid='{i}')" for i in range(1, 9)],
        injection={"action_error_on_steps": set(range(20))},
        max_steps=8, budget=3,
        tmp_root=out_dir,
    )
    data = {
        "scenario": "C4 persistent action_error -> budget exhausted",
        "status": result.status.value,
        "error_type": result.error_type.value if result.error_type else None,
        "recovery_count": result.recovery_count,
        "steps": result.num_steps,
        "expected": "controlled RECOVERY_FAILED abort at the budget boundary "
                    "(3 recoveries), never an unbounded loop",
        "assertions": {
            "not_success": not result.success,
            "recovery_failed_error":
                result.error_type == ErrorType.RECOVERY_FAILED,
            "budget_respected": result.recovery_count == 3,
            "bounded_steps": result.num_steps == 4,
            "no_recovery_env_actions": result.recovery_environment_actions == 0,
        },
    }
    scenarios.append(data)

    # C5: single NO_PROGRESS -> CONTINUE (no recovery)
    result, executed, (events, steps) = run_recovery_episode(
        name="C5", script=[{}, {"reward": 1.0, "terminated": True,
                                "observation": OBS_B}],
        actions=["click(bid='1')", "click(bid='2')"],
        tmp_root=out_dir,
    )
    data = {
        "scenario": "C5 single no_progress -> continue",
        "status": result.status.value,
        "recovery_count": result.recovery_count,
        "expected": "success without any recovery: one unchanged state is "
                    "not a recoverable failure",
        "assertions": {
            "success": result.success,
            "zero_recoveries": result.recovery_count == 0,
            "no_recovery_events": event_outcomes(events, "recovery") == [],
            "continue_policy_recorded":
                "continue" in event_outcomes(events, "policy_decision"),
        },
    }
    scenarios.append(data)

    # C6: terminated + reward=0 -> final, no recovery, no reset
    result, executed, (events, steps) = run_recovery_episode(
        name="C6", script=[{"terminated": True, "reward": 0.0}],
        actions=["click(bid='1')", "click(bid='2')"],
        tmp_root=out_dir,
    )
    data = {
        "scenario": "C6 task_failed is final",
        "status": result.status.value,
        "error_type": result.error_type.value if result.error_type else None,
        "recovery_count": result.recovery_count,
        "executed_actions": executed,
        "expected": "episode FAILED with no recovery and no task reset",
        "assertions": {
            "failed": result.status == RunStatus.FAILED,
            "task_terminated_error":
                result.error_type == ErrorType.TASK_TERMINATED,
            "zero_recoveries": result.recovery_count == 0,
            "zero_recovery_env_actions":
                result.recovery_environment_actions == 0,
            "no_extra_env_action": executed == ["click(bid='1')"],
        },
    }
    scenarios.append(data)

    # C7: blocked action selected again -> never reaches env.step
    result, executed, (events, steps) = run_recovery_episode(
        name="C7", script=success_script,
        actions=["click(bid='1')", "click(bid='1')", "click(bid='2')"],
        injection={"action_error_on_steps": {0}},
        tmp_root=out_dir,
    )
    data = {
        "scenario": "C7 blocked action re-selected",
        "status": result.status.value,
        "recovery_count": result.recovery_count,
        "steps": result.num_steps,
        "executed_actions": executed,
        "expected": "success; the re-selected blocked action never enters "
                    "env.step; re-decision consumes recovery budget",
        "assertions": {
            "success": result.success,
            "blocked_never_reexecuted": executed.count("click(bid='1')") == 1,
            "two_recoveries": result.recovery_count == 2,
            "redecide_event_recorded": "blocked_action_selected"
            in event_outcomes(events, "recovery"),
            "final_step_action": steps[-1].action == "click(bid='2')",
        },
    }
    scenarios.append(data)

    # C8: recovery noop terminates the task -> environment result kept
    result, executed, (events, steps) = run_recovery_episode(
        name="C8", script=[{}, {"reward": 1.0, "terminated": True}],
        actions=["click(bid='1')", "click(bid='2')"],
        injection={"empty_observation_on_steps": {0}},
        tmp_root=out_dir,
    )
    data = {
        "scenario": "C8 recovery noop terminates the task",
        "status": result.status.value,
        "final_reward": result.final_reward,
        "recovery_count": result.recovery_count,
        "recovery_environment_actions": result.recovery_environment_actions,
        "steps": result.num_steps,
        "expected": "success accepted from the recovery noop's real "
                    "environment termination; one agent step only",
        "assertions": {
            "success": result.success,
            "reward_one": result.final_reward == 1.0,
            "one_recovery": result.recovery_count == 1,
            "one_recovery_env_action": result.recovery_environment_actions == 1,
            "one_agent_step": result.num_steps == 1,
            "recovered_episode": result.recovered_episode,
        },
    }
    scenarios.append(data)

    for s in scenarios:
        s["pass"] = all(s["assertions"].values())
        s["failed_assertions"] = [k for k, v in s["assertions"].items() if not v]

    payload = {
        "suite": "phase1c_fault_suite",
        "scenarios": scenarios,
        "all_pass": all(s["pass"] for s in scenarios),
    }
    json_path = out_dir / "phase1c_fault_suite.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Phase 1C Controlled Fault Suite Report",
        "",
        "Auto-generated from `phase1c_fault_suite.json` — deterministic",
        "environment-side fault-injection scenarios on FakeEnvironment +",
        "MockModelAdapter, with the rule policy and RecoveryManager wired in.",
        "These results are NOT mixed with natural MiniWoB success rates.",
        "",
        "| scenario | pass | status | recoveries | recovery env actions | error type |",
        "|---|---|---|---:|---:|---|",
    ]
    for s in scenarios:
        lines.append(
            f"| {s['scenario']} | {'PASS' if s['pass'] else 'FAIL'} "
            f"| {s['status']} | {s['recovery_count']} "
            f"| {s.get('recovery_environment_actions', 0)} "
            f"| {s.get('error_type') or '-'} |"
        )
    lines.append("")
    lines.append(
        f"**Suite result: {'ALL PASS' if payload['all_pass'] else 'FAILURES PRESENT'}**"
    )
    lines.append("")
    lines.append("Guarantees exercised: a blocked action never reaches"
                 " `env.step`; a recovery environment action is never an Agent"
                 " step and never produces a StepRecord; the recovery's real"
                 " environment result (including termination) is kept;"
                 " recovery is bounded by `max_recoveries_per_episode`"
                 " (controlled abort); TASK_FAILED is final and never reset.")
    lines.append("")
    for s in scenarios:
        if not s["pass"]:
            lines.append(f"- {s['scenario']} failed: {s['failed_assertions']}")
    (out_dir / "phase1c_fault_suite_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"fault suite written to {json_path} (all_pass={payload['all_pass']})")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
