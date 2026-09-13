"""Phase 2A1 — offline trace replay tests.

Covers TraceBundleLoader, structural replay, deterministic StepVerifier
semantic replay, ReplayReport behavior on corrupt traces, legacy v1
handling, and the read-only guarantee (replay never mutates the trace).

Replay itself must stay offline: zero model calls, zero environment actions.
The source runs here use FakeEnvironment + MockModelAdapter only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.models import Observation, RunStatus, TaskSpec
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.models.mock import MockModelAdapter
from web_harness.persistence.replay import TraceBundleLoader, TraceReplayEngine
from web_harness.reliability.verifier import DefaultStepVerifier
from web_harness.runtime.episode_runner import EpisodeRunner

TASK = TaskSpec(benchmark="fake", task_id="replay", seed=0, max_steps=5)

# step 0: action changes nothing (no-progress warning); step 1: task success
SCRIPT = [{"reward": 0.0, "terminated": False}, {"reward": 1.0, "terminated": True}]
ACTIONS = ["noop()", "click(bid='1')"]


def build_run(tmp_path: Path, run_id: str = "replay-ok") -> Path:
    env = FakeEnvironment(
        reset_observation=make_fake_observation(url="http://fake.local/start"),
        script=SCRIPT,
    )
    agent = BaselineAgent(model_adapter=MockModelAdapter(ACTIONS))
    runner = EpisodeRunner(
        agent=agent,
        env=env,
        trace_root=tmp_path,
        verifier=DefaultStepVerifier(),
    )
    result = runner.run(TASK, run_id=run_id)
    assert result.status == RunStatus.SUCCESS
    assert result.num_steps == 2
    return tmp_path / run_id


def replay(run_dir: Path, **engine_kwargs):
    bundle = TraceBundleLoader().load(run_dir)
    report = TraceReplayEngine(**engine_kwargs).run(bundle)
    return bundle, report


def edit_jsonl(path: Path, index: int, mutate) -> None:
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mutate(lines[index])
    path.write_text(
        "\n".join(json.dumps(obj, ensure_ascii=False) for obj in lines) + "\n",
        encoding="utf-8",
    )


def edit_result_json(run_dir: Path, mutate) -> None:
    path = run_dir / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def tree_digest(run_dir: Path) -> dict[str, str]:
    return {
        str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
    }


# -- valid traces -----------------------------------------------------------


def test_valid_completed_trace_replays_clean(tmp_path):
    run_dir = build_run(tmp_path)
    bundle, report = replay(run_dir)

    assert bundle.trace_schema_version == 2
    assert bundle.manifest["trace_schema_version"] == 2
    assert report.run_id == "replay-ok"
    assert report.step_count == 2
    assert report.event_count > 0
    assert report.structural_valid is True
    assert report.artifact_missing_count == 0
    assert report.invariant_error_count == 0
    assert report.metric_mismatches == []
    assert report.verification_mismatches == []
    assert report.errors == []
    assert report.replayed_verifications == 2
    assert report.semantic_replay_skipped is False


def test_structured_observation_artifacts_written_and_loadable(tmp_path):
    run_dir = build_run(tmp_path)
    bundle = TraceBundleLoader().load(run_dir)

    for step in bundle.steps:
        assert step.observation_ref == f"artifacts/obs_{step.step_index:03d}.txt"
        assert (
            step.next_observation_ref
            == f"artifacts/next_obs_{step.step_index:03d}.txt"
        )
        # additive JSON refs exist alongside the unchanged .txt artifacts
        assert (
            step.observation_json_ref == f"artifacts/obs_{step.step_index:03d}.json"
        )
        assert (
            step.next_observation_json_ref
            == f"artifacts/next_obs_{step.step_index:03d}.json"
        )
        assert (run_dir / step.observation_ref).exists()
        assert (run_dir / step.next_observation_ref).exists()
        pre = Observation.model_validate_json(
            (run_dir / step.observation_json_ref).read_text(encoding="utf-8")
        )
        post = Observation.model_validate_json(
            (run_dir / step.next_observation_json_ref).read_text(encoding="utf-8")
        )
        assert pre.url == "http://fake.local/start"
        assert post.last_action == ACTIONS[step.step_index]


def test_semantic_verifier_replay_matches_recorded_signals(tmp_path):
    run_dir = build_run(tmp_path)
    _, report = replay(run_dir)

    # step 0 recorded a NO_PROGRESS warning, step 1 recorded a pass; the
    # replayed deterministic verifier must reproduce both exactly
    assert report.replayed_verifications == 2
    assert report.verification_mismatches == []


def test_valid_interrupted_trace_without_result_json(tmp_path):
    run_dir = build_run(tmp_path)
    (run_dir / "result.json").unlink()

    bundle, report = replay(run_dir)
    assert bundle.result is None
    # structural checks that need result.json are skipped; the trace itself
    # is still valid and the TaskSpec is recovered from the manifest
    assert report.structural_valid is True
    assert report.metric_mismatches == []
    assert report.replayed_verifications == 2


def test_replay_does_not_modify_trace(tmp_path):
    run_dir = build_run(tmp_path)
    before = tree_digest(run_dir)

    replay(run_dir)

    assert tree_digest(run_dir) == before


# -- corrupt / inconsistent traces -------------------------------------------


def test_missing_artifact_detected(tmp_path):
    run_dir = build_run(tmp_path)
    (run_dir / "artifacts" / "next_obs_000.txt").unlink()

    _, report = replay(run_dir)
    assert report.artifact_missing_count >= 1
    assert any("next_obs_000.txt" in e for e in report.errors)
    assert report.structural_valid is False


def test_corrupt_steps_jsonl_detected(tmp_path):
    run_dir = build_run(tmp_path)
    with open(run_dir / "steps.jsonl", "a", encoding="utf-8") as f:
        f.write('{"run_id": "replay-ok", "broken' + "\n")

    bundle, report = replay(run_dir)
    assert any("steps.jsonl" in e for e in bundle.load_errors)
    assert report.structural_valid is False


def test_missing_steps_jsonl_detected(tmp_path):
    run_dir = build_run(tmp_path)
    (run_dir / "steps.jsonl").unlink()

    bundle, report = replay(run_dir)
    assert any("steps.jsonl missing" in e for e in bundle.load_errors)
    assert report.structural_valid is False


def test_non_contiguous_step_index_detected(tmp_path):
    run_dir = build_run(tmp_path)

    def break_index(record):
        record["step_index"] = 3

    edit_jsonl(run_dir / "steps.jsonl", 1, break_index)
    _, report = replay(run_dir)
    assert any("not contiguous" in e for e in report.errors)
    assert report.structural_valid is False


def test_run_id_mismatch_detected(tmp_path):
    run_dir = build_run(tmp_path)

    def change_run_id(record):
        record["run_id"] = "run-elsewhere"

    edit_jsonl(run_dir / "steps.jsonl", 1, change_run_id)
    _, report = replay(run_dir)
    assert any("run_id mismatch" in e for e in report.errors)
    assert report.structural_valid is False


def test_event_with_unknown_step_index_detected(tmp_path):
    run_dir = build_run(tmp_path)

    def point_at_missing_step(record):
        if record["event_type"] == "verification":
            record["step_index"] = 99

    edit_jsonl(run_dir / "events.jsonl", 0, point_at_missing_step)
    _, report = replay(run_dir)
    assert any("unknown step_index" in e for e in report.errors)
    assert report.structural_valid is False


# -- metric / invariant recomputation ----------------------------------------


def test_metric_mismatch_detected(tmp_path):
    run_dir = build_run(tmp_path)

    def bump_tokens(payload):
        payload["input_tokens"] = payload["input_tokens"] + 7

    edit_result_json(run_dir, bump_tokens)
    _, report = replay(run_dir)
    assert any("input_tokens" in m for m in report.metric_mismatches)
    assert report.structural_valid is False


def test_num_steps_mismatch_detected(tmp_path):
    run_dir = build_run(tmp_path)

    def bump_steps(payload):
        payload["num_steps"] = 9

    edit_result_json(run_dir, bump_steps)
    _, report = replay(run_dir)
    assert any("num_steps" in m for m in report.metric_mismatches)
    assert report.structural_valid is False


def test_recovery_invariant_violation_detected(tmp_path):
    run_dir = build_run(tmp_path)

    def break_invariant(payload):
        payload["recovery_count"] = 1  # success+failed+unresolved stay 0

    edit_result_json(run_dir, break_invariant)
    _, report = replay(run_dir)
    assert report.invariant_error_count == 1
    assert any("recovery outcome invariant" in e for e in report.errors)
    assert report.structural_valid is False


def test_replan_invariant_violation_detected(tmp_path):
    run_dir = build_run(tmp_path)

    def break_invariant(payload):
        payload["replan_count"] = 2
        payload["replan_success_count"] = 1  # failed+unresolved stay 0

    edit_result_json(run_dir, break_invariant)
    _, report = replay(run_dir)
    assert report.invariant_error_count == 1
    assert any("replan outcome invariant" in e for e in report.errors)
    assert report.structural_valid is False


# -- semantic replay mismatch paths ------------------------------------------


def test_semantic_replay_detects_tampered_recorded_signals(tmp_path):
    run_dir = build_run(tmp_path)

    def drop_recorded_signals(record):
        if record["event_type"] == "verification":
            record["data"]["signals"] = []

    edit_jsonl(run_dir / "events.jsonl", 0, drop_recorded_signals)
    _, report = replay(run_dir)
    assert report.replayed_verifications == 2
    assert len(report.verification_mismatches) == 1
    assert "step 0" in report.verification_mismatches[0]
    # verification mismatches are semantic: they do not flip structural_valid
    assert report.structural_valid is True


def test_semantic_replay_detects_tampered_post_observation(tmp_path):
    run_dir = build_run(tmp_path)
    obs_path = run_dir / "artifacts" / "next_obs_000.json"
    obs = json.loads(obs_path.read_text(encoding="utf-8"))
    # an empty observation (no url, no content) makes the replayed verifier
    # fire OBSERVATION_INVALID where the live run saw a no-progress warning
    obs["url"] = ""
    obs["axtree"] = None
    obs_path.write_text(json.dumps(obs, indent=2, ensure_ascii=False), encoding="utf-8")

    _, report = replay(run_dir)
    # the replayed verifier now sees an action error the live run never saw
    assert len(report.verification_mismatches) >= 1
    assert any("step 0" in m for m in report.verification_mismatches)


# -- legacy schema v1 traces --------------------------------------------------


def test_legacy_trace_without_schema_version_treated_as_v1(tmp_path):
    run_dir = build_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["trace_schema_version"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    bundle, report = replay(run_dir)
    assert bundle.trace_schema_version == 1
    # structural replay covers legacy traces; semantic replay is v2-only
    assert report.structural_valid is True
    assert report.semantic_replay_skipped is True
    assert report.replayed_verifications == 0
    assert any("legacy" in n for n in report.notes)


def test_semantic_replay_can_be_disabled(tmp_path):
    run_dir = build_run(tmp_path)
    _, report = replay(run_dir, semantic_replay=False)

    assert report.semantic_replay_skipped is True
    assert report.replayed_verifications == 0
    assert report.verification_mismatches == []
    assert report.structural_valid is True
