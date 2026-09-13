"""Phase 2A2 — durable checkpoint tests (C1-C16).

Level 1 (pure unit): state roundtrips, atomic checkpoint manager, action
contract hash, resume capability contract, orphan .tmp handling.
Level 2 (runtime fake): journal integration through the real EpisodeRunner
(FakeEnvironment + MockModelAdapter), safe-point checkpoint wiring, secret
sentinel, checkpoint-disabled compatibility.

NO resume is implemented or tested here: Phase 2A2 only persists state.
"""

from __future__ import annotations

import json
from pathlib import Path

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.events import RuntimeEventType
from web_harness.core.ids import now_utc_iso
from web_harness.core.models import RunStatus, StepRecord, TaskSpec
from web_harness.core.reliability import (
    RecoveryDirective,
    RecoveryKind,
    RecoveryPlan,
    ReliabilityBudget,
    ReliabilityState,
)
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.evaluation.fault_injection import (
    FaultInjectingEnvironmentAdapter,
)
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.persistence.checkpoint import (
    CheckpointManager,
    CheckpointTraceOffsets,
    action_contract_hash,
    environment_resume_strategy,
)
from web_harness.persistence.journal import (
    EnvironmentOperationKind,
    EnvironmentOperationRecord,
    read_journal,
)
from web_harness.reliability.fingerprint import fingerprint_of
from web_harness.reliability.policy import FailurePolicyEngine
from web_harness.reliability.verifier import DefaultStepVerifier
from web_harness.runtime.episode_runner import EpisodeRunner
from web_harness.runtime.state import EpisodeCounters, RunState

TASK = TaskSpec(benchmark="fake", task_id="cp", seed=0, max_steps=6)
ACTIONS = ["click(bid='1')", "click(bid='2')"]
SUCCESS_SCRIPT = [{"reward": 0.0, "terminated": False}, {"reward": 1.0, "terminated": True}]

SENTINEL = "sk-PHASE2A2-SENTINEL-SECRET"


def make_env(script=None):
    return FakeEnvironment(
        reset_observation=make_fake_observation(url="http://fake.local/start"),
        script=script or SUCCESS_SCRIPT,
    )


def make_runner(tmp_path: Path, env, actions=None, *, checkpoint=True, **kw):
    return EpisodeRunner(
        agent=BaselineAgent(model_adapter=MockModelAdapter(list(actions or ACTIONS))),
        env=env,
        trace_root=tmp_path,
        verifier=DefaultStepVerifier(),
        checkpoint_enabled=checkpoint,
        checkpoint_every_agent_steps=1,
        **kw,
    )


def full_state() -> RunState:
    """RunState with every checkpoint-relevant field populated."""
    return RunState(
        run_id="run-cp",
        task=TASK,
        status=RunStatus.RUNNING,
        current_observation=make_fake_observation(url="http://fake.local/x"),
        steps=[
            StepRecord(
                run_id="run-cp",
                step_index=0,
                action="click(bid='1')",
                reward=0.0,
            )
        ],
        reliability=ReliabilityState(
            active_recovery_directive=RecoveryDirective(
                kind=RecoveryKind.WAIT_AND_REOBSERVE,
                reason="test directive",
                wait_ms=250,
                expires_after_agent_steps=1,
            ),
            active_recovery_plan=RecoveryPlan(
                diagnosis="stuck on popup",
                immediate_subgoal="dismiss the popup",
                horizon_steps=3,
            ),
            remaining_plan_steps=2,
            pending_recovery_evaluations=[
                {"signature": "sig-a", "steps_observed": 1, "pre_fingerprint": "fp1"}
            ],
            pending_replan_evaluation={
                "signature": "sig-b",
                "steps_observed": 0,
                "pre_fingerprint": "fp2",
            },
            consecutive_recovery_failures=1,
            last_failed_recovery_signature="sig-a",
        ),
        counters=EpisodeCounters(
            retry_count=1,
            recovery_count=2,
            recovery_success_count=1,
            replan_count=1,
            replan_input_tokens=111,
        ),
    )


def save_state(manager: CheckpointManager, state: RunState):
    return manager.save(
        run_id=state.run_id,
        created_at=now_utc_iso(),
        reason="test",
        task=state.task,
        state=state,
        next_step_index=1,
        environment_op_count=3,
        current_environment_fingerprint="fp-current",
        action_contract_hash="hash-abc",
        environment_adapter="FakeEnvironment",
        resume_strategy="deterministic_replay",
        trace_offsets=CheckpointTraceOffsets(
            step_count=state.num_steps,
            event_count=4,
            environment_op_count=3,
        ),
    )


# -- C1-C3: state roundtrips ---------------------------------------------------


def test_c1_run_state_roundtrip():
    state = full_state()
    restored = RunState.model_validate_json(state.model_dump_json())
    assert restored == state


def test_c2_episode_counters_roundtrip():
    counters = EpisodeCounters(
        retry_count=3,
        retry_cycle_count=2,
        retry_success_count=1,
        retry_exhausted_count=1,
        extra_model_calls=3,
        retry_input_tokens=100,
        retry_output_tokens=200,
        retry_latency_s=1.5,
        recovery_count=2,
        recovery_success_count=1,
        recovery_failed_count=1,
        recovery_unresolved_count=0,
        recovery_environment_actions=1,
        recovery_latency_s=0.5,
        recovered_episode=True,
        blocked_action_redecision_count=1,
        replan_count=1,
        replan_failed_count=1,
        replan_model_calls=1,
        replan_input_tokens=50,
        replan_output_tokens=60,
        replan_latency_s=2.0,
    )
    restored = EpisodeCounters.model_validate_json(counters.model_dump_json())
    assert restored == counters


def test_c3_reliability_state_roundtrip():
    reliability = full_state().reliability
    restored = ReliabilityState.model_validate_json(reliability.model_dump_json())
    assert restored == reliability


# -- C4-C6: checkpoint carries active reliability state -------------------------


def make_manager(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(tmp_path / "run-cp")


def test_c4_active_recovery_directive_checkpoint_roundtrip(tmp_path):
    manager = make_manager(tmp_path)
    envelope = save_state(manager, full_state())
    loaded = manager.latest()
    assert loaded is not None
    directive = loaded.state.reliability.active_recovery_directive
    assert directive == full_state().reliability.active_recovery_directive
    assert directive.kind == RecoveryKind.WAIT_AND_REOBSERVE
    assert envelope.next_step_index == 1


def test_c5_active_recovery_plan_and_horizon_checkpoint_roundtrip(tmp_path):
    manager = make_manager(tmp_path)
    save_state(manager, full_state())
    loaded = manager.latest()
    plan = loaded.state.reliability.active_recovery_plan
    assert plan is not None
    assert plan.immediate_subgoal == "dismiss the popup"
    assert loaded.state.reliability.remaining_plan_steps == 2


def test_c6_pending_recovery_and_replan_evaluation_checkpoint_roundtrip(tmp_path):
    manager = make_manager(tmp_path)
    save_state(manager, full_state())
    loaded = manager.latest()
    reliability = loaded.state.reliability
    assert reliability.pending_recovery_evaluations == [
        {"signature": "sig-a", "steps_observed": 1, "pre_fingerprint": "fp1"}
    ]
    assert reliability.pending_replan_evaluation == {
        "signature": "sig-b",
        "steps_observed": 0,
        "pre_fingerprint": "fp2",
    }
    assert reliability.consecutive_recovery_failures == 1
    assert reliability.last_failed_recovery_signature == "sig-a"


# -- C7-C9: environment operation journal ---------------------------------------


def test_c7_journal_records_agent_actions(tmp_path):
    run_dir = tmp_path / "run"
    result = make_runner(tmp_path, make_env()).run(TASK, run_id="run")
    records = read_journal(run_dir)

    assert len(records) == result.num_steps == 2
    for i, record in enumerate(records):
        assert record.kind == EnvironmentOperationKind.AGENT_ACTION
        assert record.action == ACTIONS[i]
        assert record.source_step_index == i
        assert record.run_id == "run"
        assert record.schema_version == 1


def test_c8_journal_records_recovery_actions(tmp_path):
    # empty observation on env step 0 -> OBSERVATION_INVALID ->
    # WAIT_AND_REOBSERVE noop: a real, Harness-owned environment action.
    # The script keeps two no-change entries before termination so the
    # recovery noop does not itself terminate the task.
    script = [
        {"reward": 0.0, "terminated": False},
        {"reward": 0.0, "terminated": False},
        {"reward": 1.0, "terminated": True},
    ]
    env = FaultInjectingEnvironmentAdapter(
        make_env(script), empty_observation_on_steps={0}
    )
    runner = EpisodeRunner(
        agent=BaselineAgent(model_adapter=MockModelAdapter(list(ACTIONS))),
        env=env,
        trace_root=tmp_path,
        verifier=DefaultStepVerifier(),
        failure_policy=FailurePolicyEngine(wait_ms=500),
        recovery_budget=ReliabilityBudget(max_recoveries_per_episode=3),
        checkpoint_enabled=True,
    )
    result = runner.run(TASK, run_id="run")
    records = read_journal(tmp_path / "run")

    kinds = [r.kind for r in records]
    assert kinds == [
        EnvironmentOperationKind.AGENT_ACTION,
        EnvironmentOperationKind.RECOVERY_ACTION,
        EnvironmentOperationKind.AGENT_ACTION,
    ]
    recovery_record = records[1]
    assert recovery_record.action.startswith("noop(wait_ms=")
    # recovery actions are never agent steps: step count stays agent-only
    agent_records = [r for r in records if r.kind == EnvironmentOperationKind.AGENT_ACTION]
    assert result.num_steps == len(agent_records) == 2


def test_c9_op_index_is_monotonic(tmp_path):
    env = FaultInjectingEnvironmentAdapter(
        make_env(), empty_observation_on_steps={0}
    )
    runner = EpisodeRunner(
        agent=BaselineAgent(model_adapter=MockModelAdapter(list(ACTIONS))),
        env=env,
        trace_root=tmp_path,
        verifier=DefaultStepVerifier(),
        failure_policy=FailurePolicyEngine(wait_ms=500),
        recovery_budget=ReliabilityBudget(max_recoveries_per_episode=3),
    )
    runner.run(TASK, run_id="run")
    records = read_journal(tmp_path / "run")

    assert [r.op_index for r in records] == list(range(len(records)))


# -- C10: action contract hash --------------------------------------------------


def test_c10_action_contract_hash_deterministic():
    from web_harness.env.action_contract import ActionContract, ActionSpec

    contract = ActionContract(
        benchmark="fake",
        actions=[
            ActionSpec(name="click", signature="click(bid: str)", description="d"),
            ActionSpec(name="noop", signature="noop()", description="d"),
        ],
    )
    assert action_contract_hash(contract) == action_contract_hash(contract)
    assert len(action_contract_hash(contract)) == 16
    drifted = contract.model_copy(deep=True)
    drifted.actions[0].signature = "click(bid: str, extra: str)"
    assert action_contract_hash(contract) != action_contract_hash(drifted)


# -- C11-C13: atomic CheckpointManager ------------------------------------------


def test_c11_atomic_checkpoint_write(tmp_path):
    manager = make_manager(tmp_path)
    envelope = save_state(manager, full_state())

    cp_path = tmp_path / "run-cp" / "checkpoints" / "cp_000000.json"
    latest_path = tmp_path / "run-cp" / "checkpoints" / "latest.json"
    assert cp_path.exists() and latest_path.exists()
    # no orphan .tmp files remain after a successful save
    assert not list((tmp_path / "run-cp" / "checkpoints").glob("*.tmp"))
    # the file is valid strict JSON with the envelope contract
    payload = json.loads(cp_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_schema_version"] == 1
    assert payload["checkpoint_id"] == envelope.checkpoint_id
    assert payload["trace_offsets"]["environment_op_count"] == 3
    assert payload["environment_resume_strategy"] == "deterministic_replay"
    loaded_ok = manager.load("cp_000000.json")
    assert loaded_ok is not None
    assert loaded_ok.state.counters.recovery_count == 2


def test_c12_orphan_tmp_files_ignored(tmp_path):
    manager = make_manager(tmp_path)
    save_state(manager, full_state())
    checkpoints_dir = tmp_path / "run-cp" / "checkpoints"
    # simulate a crash mid-write: orphan .tmp files must never be valid
    (checkpoints_dir / "cp_000042.json.tmp").write_text("{broken", encoding="utf-8")
    (checkpoints_dir / "latest.json.tmp").write_text("{broken", encoding="utf-8")

    assert manager.load("cp_000042.json.tmp") is None
    latest = manager.latest()
    assert latest is not None
    assert latest.checkpoint_id == "cp-000000"


def test_c13_latest_checkpoint_selection(tmp_path):
    manager = make_manager(tmp_path)
    first = save_state(manager, full_state())
    second = save_state(manager, full_state())
    assert first.checkpoint_id == "cp-000000"
    assert second.checkpoint_id == "cp-000001"

    latest = manager.latest()
    assert latest is not None
    assert latest.checkpoint_id == "cp-000001"
    assert latest.next_step_index == 1

    # a corrupt latest.json pointer falls back to a directory scan
    (tmp_path / "run-cp" / "checkpoints" / "latest.json").write_text(
        "{corrupt", encoding="utf-8"
    )
    assert manager.latest().checkpoint_id == "cp-000001"


# -- C14-C16: runner integration -------------------------------------------------


def test_c14_checkpoint_trace_offsets_and_safe_point_semantics(tmp_path):
    # 2 agent steps, terminating on the second: the terminating step breaks
    # the loop, so exactly ONE safe-point checkpoint exists (after step 0)
    result = make_runner(tmp_path, make_env()).run(TASK, run_id="run")
    assert result.status == RunStatus.SUCCESS
    manager = CheckpointManager(tmp_path / "run")
    envelope = manager.latest()
    assert envelope is not None
    assert envelope.checkpoint_id == "cp-000000"
    assert envelope.next_step_index == 1
    assert envelope.reason == "stable_safe_point"
    assert envelope.trace_offsets.step_count == 1
    assert envelope.trace_offsets.environment_op_count == 1
    events = TraceRecorder.read_events(tmp_path / "run")
    # events.jsonl is append-only: the checkpoint event sits exactly at the
    # offset recorded at save time (events before it == recorded event_count)
    cp_index = next(
        i
        for i, e in enumerate(events)
        if e.event_type == RuntimeEventType.CHECKPOINT
    )
    assert envelope.trace_offsets.event_count == cp_index
    assert envelope.current_environment_fingerprint == fingerprint_of(
        envelope.state.current_observation
    )
    # the checkpoint event records ids/counts, never the full state
    cp_events = [e for e in events if e.event_type == RuntimeEventType.CHECKPOINT]
    assert len(cp_events) == 1
    data = cp_events[0].data
    assert data["checkpoint_id"] == "cp-000000"
    assert data["next_step_index"] == 1
    assert data["environment_op_count"] == 1
    assert envelope.state.num_steps == 1
    # resume capability contract on the recorded envelope
    assert envelope.environment_adapter == "FakeEnvironment"
    assert envelope.environment_resume_strategy == "deterministic_replay"


def test_c14b_every_n_steps_control(tmp_path):
    runner = EpisodeRunner(
        agent=BaselineAgent(model_adapter=MockModelAdapter(list(ACTIONS))),
        env=make_env(),
        trace_root=tmp_path,
        checkpoint_enabled=True,
        checkpoint_every_agent_steps=5,
    )
    runner.run(TASK, run_id="run")
    # step 1 terminates the episode -> no checkpoint ever reaches the
    # every_agent_steps boundary
    assert CheckpointManager(tmp_path / "run").latest() is None


def test_c15_no_secret_leak_into_checkpoints_or_journal(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MODEL_API_KEY", SENTINEL)
    result = make_runner(tmp_path, make_env()).run(TASK, run_id="run")
    run_dir = Path(result.trace_path)
    assert (run_dir / "checkpoints").exists()
    scanned = 0
    for path in run_dir.rglob("*"):
        if path.is_file():
            content = path.read_text(encoding="utf-8", errors="ignore")
            assert SENTINEL not in content, f"sentinel leaked into {path}"
            scanned += 1
    assert scanned > 0
    assert (run_dir / "environment_ops.jsonl").exists()
    assert (run_dir / "checkpoints" / "cp_000000.json").exists()


def test_c16_checkpoint_disabled_behavior_unchanged(tmp_path):
    # disabled (default): no checkpoints/ directory, no checkpoint events,
    # and identical RunResult metrics semantics to Phase 1
    result = make_runner(tmp_path, make_env(), checkpoint=False).run(
        TASK, run_id="run"
    )
    run_dir = Path(result.trace_path)
    assert not (run_dir / "checkpoints").exists()
    events = TraceRecorder.read_events(run_dir)
    assert not any(e.event_type == RuntimeEventType.CHECKPOINT for e in events)
    # the journal is unconditional durable trace, but behavior is unchanged
    assert len(read_journal(run_dir)) == 2
    assert result.status == RunStatus.SUCCESS
    assert result.retry_count == 0
    assert result.recovery_count == 0


# -- resume capability contract ---------------------------------------------------


def test_resume_strategy_contract():
    # deterministic fake: replayable
    assert environment_resume_strategy(
        make_env(), benchmark="fake"
    ) == environment_resume_strategy(make_env(), benchmark="fake")
    from web_harness.persistence.checkpoint import EnvironmentResumeStrategy

    assert environment_resume_strategy(
        make_env(), benchmark="fake"
    ) is EnvironmentResumeStrategy.DETERMINISTIC_REPLAY
    # unknown adapter: unsupported (never claim more than implemented)
    class UnknownEnv:
        def step(self, action):
            ...

    assert environment_resume_strategy(
        UnknownEnv(), benchmark="miniwob"
    ) is EnvironmentResumeStrategy.UNSUPPORTED


def test_journal_schema_version_recorded():
    record = EnvironmentOperationRecord(
        run_id="r",
        op_index=0,
        kind=EnvironmentOperationKind.AGENT_ACTION,
        action="noop()",
        expected_post_fingerprint="fp",
    )
    assert record.schema_version == 1
