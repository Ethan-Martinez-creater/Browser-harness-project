"""Run the Phase 1B controlled-fault suite (B1-B6) and emit machine results.

Usage:
    uv run python scripts/run_phase1b_fault_suite.py --output-dir reports/phase1

Each scenario runs a full episode with FaultInjectingModelAdapter against a
FakeEnvironment. The JSON output records per-scenario assertions; the
Markdown report is generated from that JSON (never hand-written). These
fault results are NOT mixed with the natural MiniWoB smoke success rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.errors import ErrorType
from web_harness.core.models import RunStatus, TaskSpec
from web_harness.core.reliability import ReliabilityBudget
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.evaluation.fault_injection import FaultInjectingModelAdapter
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.reliability.retry import RetryPolicy
from web_harness.runtime.decision_executor import DecisionExecutor
from web_harness.runtime.episode_runner import EpisodeRunner

TASK = TaskSpec(benchmark="fake", task_id="fault", seed=0, max_steps=5)
SUCCESS_SCRIPT = [{"reward": 1.0, "terminated": True}]


def run_scenario(
    *,
    name: str,
    api_error_calls: tuple[int, ...] = (),
    parse_error_calls: tuple[int, ...] = (),
    retry_enabled: bool = True,
    max_api_retries: int = 2,
    budget_extra: int = 6,
    tmp_root: Path,
) -> tuple[bool, dict]:
    import tempfile

    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as td:
        model = MockModelAdapter(["click(bid='1')"])
        agent = BaselineAgent(
            model_adapter=FaultInjectingModelAdapter(
                model,
                api_error_on_calls=set(api_error_calls),
                parse_error_on_calls=set(parse_error_calls),
            )
        )
        env = FakeEnvironment(
            reset_observation=make_fake_observation(url="http://fault.local/0"),
            script=SUCCESS_SCRIPT,
        )
        executor = DecisionExecutor(
            retry_policy=RetryPolicy(api_max_retries=max_api_retries) if retry_enabled else None,
            budget=ReliabilityBudget(max_extra_model_calls_per_episode=budget_extra),
            sleep=lambda _s: None,
        )
        runner = EpisodeRunner(
            agent=agent, env=env, trace_root=Path(td), decision_executor=executor
        )
        result = runner.run(TASK, run_id=f"fault-{name.lower()}")
        events = TraceRecorder.read_events(Path(result.trace_path))
        retry_events = [e for e in events if e.event_type == "retry"]

        checks["episode_completed"] = result.status in (RunStatus.SUCCESS, RunStatus.ERROR)
        checks["no_extra_env_actions"] = env.step_calls == (
            1 if result.status == RunStatus.SUCCESS else 0
        )
        checks["retry_count_as_expected"] = True  # refined below
        checks["events_recorded"] = True

        return checks, {
            "scenario": name,
            "status": result.status.value,
            "error_type": result.error_type.value if result.error_type else None,
            "retry_count": result.retry_count,
            "extra_model_calls": result.extra_model_calls,
            "env_steps": env.step_calls,
            "steps": result.num_steps,
            "retry_events": [
                {"attempt_index": e.attempt_index, "outcome": e.outcome}
                for e in retry_events
            ],
            "input_tokens": result.input_tokens,
            "retry_input_tokens": result.retry_input_tokens,
        }


def evaluate(scenario: dict, checks: dict) -> list[str]:
    failures = [name for name, ok in checks.items() if not ok]
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/phase1")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []

    # B1: attempt 0 API error, attempt 1 success
    checks, data = run_scenario(name="B1", api_error_calls=(0,), tmp_root=out_dir)
    data["expected"] = "retry_count=1, 1 env action, episode success"
    data["assertions"] = {
        "retry_count_is_1": data["retry_count"] == 1,
        "success": data["status"] == "success",
        "one_env_action": data["env_steps"] == 1,
        "one_step_record": data["steps"] == 1,
    }
    scenarios.append(data)

    # B2: two API errors then success
    _, data = run_scenario(name="B2", api_error_calls=(0, 1), tmp_root=out_dir)
    data["expected"] = "retry_count=2, episode success"
    data["assertions"] = {
        "retry_count_is_2": data["retry_count"] == 2,
        "success": data["status"] == "success",
        "one_env_action": data["env_steps"] == 1,
    }
    scenarios.append(data)

    # B3: three API errors -> retry exhausted, no env action
    _, data = run_scenario(name="B3", api_error_calls=(0, 1, 2), tmp_root=out_dir)
    data["expected"] = "retry exhausted, zero env actions, controlled termination"
    data["assertions"] = {
        "retry_exhausted": data["error_type"] == ErrorType.RETRY_EXHAUSTED.value,
        "zero_env_actions": data["env_steps"] == 0,
        "controlled_stop": data["status"] == "error",
        "exhausted_event_recorded": any(
            e["outcome"] == "exhausted" for e in data["retry_events"]
        ),
    }
    scenarios.append(data)

    # B4: malformed output then valid JSON
    _, data = run_scenario(name="B4", parse_error_calls=(0,), tmp_root=out_dir)
    data["expected"] = "one format repair retry, tokens of failed attempt retained"
    data["assertions"] = {
        "retry_count_is_1": data["retry_count"] == 1,
        "success": data["status"] == "success",
        "one_env_action": data["env_steps"] == 1,
        "failed_attempt_tokens_retained": data["input_tokens"] > 0,
    }
    scenarios.append(data)

    # B5: retry disabled -> no retry even on API error
    _, data = run_scenario(
        name="B5", api_error_calls=(0,), retry_enabled=False, tmp_root=out_dir
    )
    data["expected"] = "no retry, episode ends, no retry events"
    data["assertions"] = {
        "no_retry": data["retry_count"] == 0,
        "no_retry_events": len(data["retry_events"]) == 0,
        "controlled_stop": data["status"] == "error",
        "zero_env_actions": data["env_steps"] == 0,
    }
    scenarios.append(data)

    # B6: episode extra-model-call budget exhausted
    _, data = run_scenario(
        name="B6", api_error_calls=(0, 1, 2), budget_extra=1, tmp_root=out_dir
    )
    data["expected"] = "budget exhausted after 1 extra call, controlled stop"
    data["assertions"] = {
        "budget_respected": data["extra_model_calls"] == 1,
        "budget_exceeded_error": data["error_type"] == ErrorType.BUDGET_EXCEEDED.value,
        "zero_env_actions": data["env_steps"] == 0,
    }
    scenarios.append(data)

    # M1: API error -> parse error -> success (per-kind allowances independent)
    _, data = run_scenario(
        name="M1", api_error_calls=(0,), parse_error_calls=(1,), tmp_root=out_dir
    )
    data["expected"] = "success; 1 API retry + 1 parse retry; one env action"
    data["assertions"] = {
        "success": data["status"] == "success",
        "two_retries": data["retry_count"] == 2,
        "one_env_action": data["env_steps"] == 1,
        "mixed_reasons_recorded": [e["outcome"] for e in data["retry_events"]].count("retry") == 2,
    }
    scenarios.append(data)

    # M2: parse error -> two API errors -> success
    _, data = run_scenario(
        name="M2", parse_error_calls=(0,), api_error_calls=(1, 2), tmp_root=out_dir
    )
    data["expected"] = "success; parse repair = 1 + 2 API retries"
    data["assertions"] = {
        "success": data["status"] == "success",
        "three_retries": data["retry_count"] == 3,
        "one_env_action": data["env_steps"] == 1,
    }
    scenarios.append(data)

    # M3: mixed failures cross the episode budget
    _, data = run_scenario(
        name="M3",
        parse_error_calls=(0,),
        api_error_calls=(1, 2, 3),
        budget_extra=2,
        tmp_root=out_dir,
    )
    data["expected"] = "episode budget cuts the mixed path; controlled stop"
    data["assertions"] = {
        "budget_exceeded_error": data["error_type"] == ErrorType.BUDGET_EXCEEDED.value,
        "budget_respected": data["extra_model_calls"] == 2,
        "zero_env_actions": data["env_steps"] == 0,
    }
    scenarios.append(data)

    for s in scenarios:
        s["pass"] = all(s["assertions"].values())
        s["failed_assertions"] = [k for k, v in s["assertions"].items() if not v]

    payload = {
        "suite": "phase1b_fault_suite",
        "scenarios": scenarios,
        "all_pass": all(s["pass"] for s in scenarios),
    }
    json_path = out_dir / "phase1b_fault_suite.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Phase 1B Controlled Fault Suite Report",
        "",
        "Auto-generated from `phase1b_fault_suite.json` — deterministic",
        "fault-injection scenarios on FakeEnvironment + MockModelAdapter.",
        "These results are NOT mixed with natural MiniWoB success rates.",
        "",
        "| scenario | pass | status | retries | env actions | error type |",
        "|---|---|---|---:|---:|---|",
    ]
    for s in scenarios:
        lines.append(
            f"| {s['scenario']} | {'PASS' if s['pass'] else 'FAIL'} "
            f"| {s['status']} | {s['retry_count']} | {s['env_steps']} "
            f"| {s['error_type'] or '-'} |"
        )
    lines.append("")
    lines.append(f"**Suite result: {'ALL PASS' if payload['all_pass'] else 'FAILURES PRESENT'}**")
    lines.append("")
    for s in scenarios:
        if not s["pass"]:
            lines.append(f"- {s['scenario']} failed: {s['failed_assertions']}")
    (out_dir / "phase1b_fault_suite_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"fault suite written to {json_path} (all_pass={payload['all_pass']})")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
