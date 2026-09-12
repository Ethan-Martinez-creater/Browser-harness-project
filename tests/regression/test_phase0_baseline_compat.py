"""Phase 0 baseline compatibility regression tests (Phase 1 contract).

With every reliability mechanism disabled, the runtime must behave exactly as
the frozen Phase 0 baseline (commit 63d928e): same number of model calls,
same number of environment steps, same action sequence, same termination,
same StepRecord count and trace semantics.

Additionally, with the verifier enabled in SHADOW mode, the control flow must
be byte-identical: no extra model calls, no extra environment actions, same
actions, same termination — only added observation (events.jsonl, summary
fields).
"""

from pathlib import Path

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.models import RunStatus, TaskSpec
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.reliability.verifier import DefaultStepVerifier
from web_harness.runtime.episode_runner import EpisodeRunner

TASK = TaskSpec(benchmark="fake", task_id="compat", seed=0, max_steps=10)

# deterministic episode: 2 actions then terminate successfully
SCRIPT = [
    {"observation": make_fake_observation(url="http://s.local/1")},
    {
        "observation": make_fake_observation(url="http://s.local/2"),
        "reward": 1.0,
        "terminated": True,
    },
]
ACTIONS = ["click(bid='1')", "noop()"]


def run_episode(tmp_path: Path, run_id: str, *, verifier=None):
    env = FakeEnvironment(
        reset_observation=make_fake_observation(url="http://s.local/0"),
        script=SCRIPT,
    )
    model = MockModelAdapter(ACTIONS)
    agent = BaselineAgent(model_adapter=model)
    runner = EpisodeRunner(
        agent=agent, env=env, trace_root=tmp_path, verifier=verifier
    )
    result = runner.run(TASK, run_id=run_id)
    return result, env, model


def test_phase0_baseline_contract_preserved(tmp_path):
    result, env, model = run_episode(tmp_path, "baseline")

    # model call count: exactly one per step, no extras
    assert model.call_count == 2
    # environment steps: exactly one per step, no extras
    assert env.step_calls == 2
    # action sequence unchanged
    assert env.executed_actions == ACTIONS
    # termination unchanged
    assert result.status == RunStatus.SUCCESS
    assert result.success and result.final_reward == 1.0
    # StepRecord count matches env steps
    steps = TraceRecorder.read_steps(Path(result.trace_path))
    assert len(steps) == 2
    # no verification happened when disabled
    assert all(s.verification_status is None for s in steps)
    assert TraceRecorder.read_events(Path(result.trace_path)) == []
    assert result.verification_count == 0
    assert result.failure_signal_count == 0


def test_shadow_verifier_does_not_change_control_flow(tmp_path):
    result, env, model = run_episode(
        tmp_path, "shadow", verifier=DefaultStepVerifier()
    )

    # identical control flow vs Phase 0 baseline
    assert model.call_count == 2
    assert env.step_calls == 2
    assert env.executed_actions == ACTIONS
    assert result.status == RunStatus.SUCCESS
    assert result.success and result.final_reward == 1.0

    # added observation only
    steps = TraceRecorder.read_steps(Path(result.trace_path))
    assert len(steps) == 2
    assert all(s.verification_status == "pass" for s in steps)
    events = TraceRecorder.read_events(Path(result.trace_path))
    assert len(events) == 2
    assert all(e.event_type == "verification" for e in events)
    assert result.verification_count == 2
    assert result.failure_signal_count == 0


def test_shadow_records_failure_signals_without_flow_change(tmp_path):
    # episode with an action error: verifier must observe it (event) while
    # the episode continues exactly as it would without the verifier
    env = FakeEnvironment(
        reset_observation=make_fake_observation(url="http://s.local/0"),
        script=[
            {"action_error": "TimeoutError: click", "observation": make_fake_observation(url="http://s.local/1")},
            {
                "observation": make_fake_observation(url="http://s.local/2"),
                "reward": 1.0,
                "terminated": True,
            },
        ],
    )
    model = MockModelAdapter(ACTIONS)
    agent = BaselineAgent(model_adapter=model)

    def run(env, tmp):
        runner = EpisodeRunner(
            agent=agent,
            env=env,
            trace_root=tmp,
        )
        return runner.run(TASK, run_id="plain")

    result_plain = run(
        FakeEnvironment(
            reset_observation=make_fake_observation(url="http://s.local/0"),
            script=[
                {"action_error": "TimeoutError: click", "observation": make_fake_observation(url="http://s.local/1")},
                {
                    "observation": make_fake_observation(url="http://s.local/2"),
                    "reward": 1.0,
                    "terminated": True,
                },
            ],
        ),
        tmp_path,
    )
    runner = EpisodeRunner(
        agent=agent, env=env, trace_root=tmp_path, verifier=DefaultStepVerifier()
    )
    result_shadow = runner.run(TASK, run_id="shadow")

    # identical outcome and control flow
    assert result_plain.status == result_shadow.status
    assert result_plain.num_steps == result_shadow.num_steps
    assert result_plain.success == result_shadow.success
    # shadow adds failure observation
    assert result_shadow.failure_signal_count > 0
    events = TraceRecorder.read_events(Path(result_shadow.trace_path))
    assert any(
        any(s["kind"] == "ACTION_ERROR" for s in e.data.get("signals", []))
        for e in events
    )
    # StepRecord summary fields populated
    steps = TraceRecorder.read_steps(Path(result_shadow.trace_path))
    assert steps[0].failure_kinds == ["ACTION_ERROR"]
    assert steps[0].verification_status == "fail"
