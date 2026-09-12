"""Unit tests for core data models: JSON round-trip safety."""

import json

from web_harness.core.errors import ErrorType
from web_harness.core.models import (
    ActionDecision,
    EnvironmentStep,
    Observation,
    PromptBundle,
    RunResult,
    RunStatus,
    StepRecord,
    TaskSpec,
)


def test_task_spec_roundtrip():
    spec = TaskSpec(benchmark="miniwob", task_id="click-test", seed=0, max_steps=20)
    raw = spec.model_dump_json()
    assert TaskSpec.model_validate_json(raw) == spec
    payload = json.loads(raw)
    assert payload["task_id"] == "click-test"


def test_observation_defaults_and_roundtrip():
    obs = Observation(goal="Click the button.", url="http://x/", axtree="[1] button 'b'")
    assert obs.open_pages == []
    assert obs.truncated is False
    obs2 = Observation.model_validate_json(obs.model_dump_json())
    assert obs2 == obs


def test_action_decision_short_reason():
    d = ActionDecision(action="click(bid='1')", short_reason="target visible")
    assert ActionDecision.model_validate_json(d.model_dump_json()) == d


def test_environment_step_roundtrip():
    step = EnvironmentStep(
        observation=Observation(goal="g", url="u"),
        reward=1.0,
        terminated=True,
        action_error=None,
    )
    assert EnvironmentStep.model_validate_json(step.model_dump_json()) == step


def test_step_record_roundtrip_with_error_type():
    rec = StepRecord(
        run_id="run-1",
        step_index=3,
        action="click(bid='7')",
        action_error="element not found",
        reward=0.0,
        latency_ms=12.5,
        error_type=ErrorType.ACTION_EXECUTION_ERROR,
    )
    assert StepRecord.model_validate_json(rec.model_dump_json()) == rec


def test_run_result_roundtrip():
    res = RunResult(
        run_id="run-1",
        task_spec=TaskSpec(benchmark="miniwob", task_id="click-test"),
        status=RunStatus.SUCCESS,
        success=True,
        final_reward=1.0,
        num_steps=2,
        error_type=ErrorType.TASK_TERMINATED,
    )
    assert RunResult.model_validate_json(res.model_dump_json()) == res


def test_prompt_bundle_roundtrip():
    p = PromptBundle(system="s", user="u")
    assert PromptBundle.model_validate_json(p.model_dump_json()) == p


def test_error_type_values_match_plan():
    expected = {
        "CONFIG_ERROR",
        "MODEL_API_ERROR",
        "MODEL_OUTPUT_PARSE_ERROR",
        "ENVIRONMENT_INIT_ERROR",
        "ACTION_EXECUTION_ERROR",
        "OBSERVATION_ERROR",
        "MAX_STEPS_EXCEEDED",
        "TASK_TERMINATED",
        "TASK_TRUNCATED",
        "TRACE_WRITE_ERROR",
        "UNKNOWN_ERROR",
    }
    assert expected <= {e.value for e in ErrorType}
