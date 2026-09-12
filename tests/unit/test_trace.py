"""Unit tests for TraceRecorder durability."""

import json

from web_harness.core.models import (
    ActionDecision,
    PromptBundle,
    RunResult,
    RunStatus,
    StepRecord,
    TaskSpec,
)
from web_harness.env.fake import make_fake_observation
from web_harness.models.base import ModelOutput
from web_harness.observability.trace import TraceRecorder


def _record_step(recorder: TraceRecorder, idx: int) -> StepRecord:
    return recorder.record_step(
        StepRecord(run_id="run-1", step_index=idx, action="noop()"),
        observation=make_fake_observation(),
        prompt=PromptBundle(system="s", user="u"),
        model_output=ModelOutput(
            decision=ActionDecision(action="noop()"),
            model_name="mock",
            raw_text='{"action": "noop()"}',
        ),
    )


def test_trace_durability(tmp_path):
    recorder = TraceRecorder(tmp_path / "run-1")
    recorder.write_manifest(
        {"run_id": "run-1", "model_name": "mock", "task_id": "click-test"}
    )
    for i in range(3):
        _record_step(recorder, i)
    recorder.write_result(
        RunResult(
            run_id="run-1",
            task_spec=TaskSpec(benchmark="fake", task_id="t"),
            status=RunStatus.SUCCESS,
            success=True,
            final_reward=1.0,
            num_steps=3,
        )
    )

    run_dir = tmp_path / "run-1"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "result.json").exists()
    steps_file = run_dir / "steps.jsonl"
    assert steps_file.exists()
    lines = [
        line for line in steps_file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 3

    for line in lines:
        json.loads(line)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_id"] == "click-test"

    steps = TraceRecorder.read_steps(run_dir)
    assert len(steps) == 3
    assert steps[1].step_index == 1
    reloaded = TraceRecorder.read_result(run_dir)
    assert reloaded is not None and reloaded.success

    for i in range(3):
        assert (run_dir / "artifacts" / f"obs_{i:03d}.txt").exists()
        assert (run_dir / "artifacts" / f"prompt_{i:03d}.txt").exists()
        assert (run_dir / "artifacts" / f"model_{i:03d}.json").exists()


def test_trace_disable_prompts(tmp_path):
    recorder = TraceRecorder(
        tmp_path / "r2", save_prompts=False, save_model_responses=False
    )
    _record_step(recorder, 0)
    run_dir = tmp_path / "r2"
    assert (run_dir / "artifacts" / "obs_000.txt").exists()
    assert not (run_dir / "artifacts" / "prompt_000.txt").exists()
    assert not (run_dir / "artifacts" / "model_000.json").exists()


def test_no_api_key_leak(tmp_path):
    recorder = TraceRecorder(tmp_path / "r3")
    recorder.write_manifest({"model_provider": "openai_compatible"})
    _record_step(recorder, 0)
    text = (tmp_path / "r3" / "steps.jsonl").read_text(encoding="utf-8")
    assert "sk-" not in text
