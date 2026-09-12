"""Unit tests for EpisodeRunner: the four Phase 0 termination paths."""


from web_harness.agents.baseline import BaselineAgent
from web_harness.core.models import RunStatus, TaskSpec
from web_harness.env.fake import FakeEnvironment
from web_harness.models.mock import MockModelAdapter


def make_agent(actions: list[str], **kwargs) -> BaselineAgent:
    return BaselineAgent(model_adapter=MockModelAdapter(actions, **kwargs))


def make_env(script: list[dict] | None = None, **kwargs) -> FakeEnvironment:
    if script is None:
        script = [{"reward": 1.0, "terminated": True}]
    return FakeEnvironment(script=script, **kwargs)


TASK = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=20)


def test_normal_termination(tmp_path):
    env = make_env(
        script=[
            {"reward": 0.0},
            {"reward": 1.0, "terminated": True},
        ]
    )
    agent = make_agent(["click(bid='1')", "noop()"])
    from web_harness.runtime.episode_runner import EpisodeRunner

    result = EpisodeRunner(agent=agent, env=env, trace_root=tmp_path).run(TASK)
    assert result.status == RunStatus.SUCCESS
    assert result.success
    assert result.final_reward == 1.0
    assert result.num_steps == 2
    assert result.error_type is not None  # TASK_TERMINATED
    assert env.close_calls == 1


def test_max_steps_exceeded(tmp_path):
    env = make_env(script=[{"reward": 0.0}])  # never terminates
    agent = make_agent(["noop()"])
    from web_harness.runtime.episode_runner import EpisodeRunner

    task = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=3)
    result = EpisodeRunner(agent=agent, env=env, trace_root=tmp_path).run(task)
    assert result.status == RunStatus.MAX_STEPS_REACHED
    assert result.num_steps == 3
    assert result.error_type is not None
    assert result.error_type.value == "MAX_STEPS_EXCEEDED"
    assert env.close_calls == 1


def test_environment_step_exception_normalized(tmp_path):
    env = make_env(
        script=[{"reward": 1.0, "terminated": True}],
        raise_on_step_index={0},
    )
    agent = make_agent(["click(bid='1')"])
    from web_harness.runtime.episode_runner import EpisodeRunner

    result = EpisodeRunner(agent=agent, env=env, trace_root=tmp_path).run(TASK)
    # step 0 raised -> normalized as action error; step 1 succeeded
    assert result.status == RunStatus.SUCCESS
    assert result.action_error_count >= 1
    assert result.num_steps == 2
    assert env.close_calls == 1


def test_model_parse_error(tmp_path):
    env = make_env()
    agent = make_agent(["noop()"], raise_parse_error_on_step={1})
    from web_harness.runtime.episode_runner import EpisodeRunner

    result = EpisodeRunner(agent=agent, env=env, trace_root=tmp_path).run(TASK)
    assert result.status == RunStatus.ERROR
    assert result.error_type is not None
    assert result.error_type.value == "MODEL_OUTPUT_PARSE_ERROR"
    assert env.close_calls == 1


def test_environment_init_error(tmp_path):
    class BrokenEnv(FakeEnvironment):
        def reset(self, task):
            raise RuntimeError("no browser")

    from web_harness.runtime.episode_runner import EpisodeRunner

    runner_result = EpisodeRunner(
        agent=make_agent(["noop()"]), env=BrokenEnv(), trace_root=tmp_path
    ).run(TASK)
    assert runner_result.status == RunStatus.ERROR
    assert runner_result.error_type is not None
    assert runner_result.error_type.value == "ENVIRONMENT_INIT_ERROR"


def test_env_reset_failure_still_closes(tmp_path):
    class BrokenEnv(FakeEnvironment):
        def reset(self, task):
            super().reset(task)
            raise RuntimeError("boom")

    from web_harness.runtime.episode_runner import EpisodeRunner

    env = BrokenEnv()
    EpisodeRunner(agent=make_agent(["noop()"]), env=env, trace_root=tmp_path).run(TASK)
    assert env.close_calls == 1
