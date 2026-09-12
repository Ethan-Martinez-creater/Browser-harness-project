"""Unit tests for step-trace semantics (B2): obs_N = decision input,
next_obs_N = action result; next_obs_N ≡ obs_(N+1)."""

from pathlib import Path

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.models import RunStatus, TaskSpec
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.models.mock import MockModelAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.runtime.episode_runner import EpisodeRunner

TASK = TaskSpec(benchmark="fake", task_id="t", seed=0, max_steps=5)


def make_env(pre_url: str, post_url: str) -> FakeEnvironment:
    """Environment whose post-action observation differs from its reset view."""
    post = make_fake_observation(url=post_url)
    return FakeEnvironment(
        reset_observation=make_fake_observation(url=pre_url),
        script=[
            {"observation": post, "reward": 1.0, "terminated": True},
        ],
    )


def run_episode(tmp_path: Path, env: FakeEnvironment, actions: list[str], **kw):
    agent = BaselineAgent(model_adapter=MockModelAdapter(actions))
    runner = EpisodeRunner(agent=agent, env=env, trace_root=tmp_path)
    return runner.run(TASK, run_id="r"), env


def test_obs_is_pre_action_and_next_obs_is_post_action(tmp_path):
    env = make_env("http://pre.local/a", "http://post.local/b")
    result, _ = run_episode(tmp_path, env, ["click(bid='1')"])

    run_dir = Path(result.trace_path)
    obs_text = (run_dir / "artifacts" / "obs_000.txt").read_text(encoding="utf-8")
    next_text = (run_dir / "artifacts" / "next_obs_000.txt").read_text(encoding="utf-8")

    # obs_000 = what the model decided on (pre-action)
    assert "http://pre.local/a" in obs_text
    # next_obs_000 = what the action produced (post-action)
    assert "http://post.local/b" in next_text

    steps = TraceRecorder.read_steps(run_dir)
    assert steps[0].observation_ref == "artifacts/obs_000.txt"
    assert steps[0].next_observation_ref == "artifacts/next_obs_000.txt"


def test_prompt_matches_pre_action_observation(tmp_path):
    env = make_env("http://pre.local/a", "http://post.local/b")
    agent = BaselineAgent(model_adapter=MockModelAdapter(["click(bid='1')"]))
    runner = EpisodeRunner(agent=agent, env=env, trace_root=tmp_path)
    result = runner.run(TASK, run_id="r")

    run_dir = Path(result.trace_path)
    prompt_text = (run_dir / "artifacts" / "prompt_000.txt").read_text(encoding="utf-8")
    obs_text = (run_dir / "artifacts" / "obs_000.txt").read_text(encoding="utf-8")

    # the prompt must embed the same pre-action URL that obs_000 records
    assert "http://pre.local/a" in prompt_text
    assert "http://post.local/b" not in prompt_text
    assert "http://pre.local/a" in obs_text


def test_next_obs_n_equals_obs_n_plus_one(tmp_path):
    env = FakeEnvironment(
        reset_observation=make_fake_observation(url="http://s.local/0"),
        script=[
            {"observation": make_fake_observation(url="http://s.local/1")},
            {
                "observation": make_fake_observation(url="http://s.local/2"),
                "reward": 1.0,
                "terminated": True,
            },
        ],
    )
    agent = BaselineAgent(model_adapter=MockModelAdapter(["click(bid='1')"]))
    runner = EpisodeRunner(agent=agent, env=env, trace_root=tmp_path)
    result = runner.run(TASK, run_id="r")

    run_dir = Path(result.trace_path)
    # for every non-final step, next_obs_N must equal obs_(N+1) semantically
    for n in (0,):
        nxt = (run_dir / "artifacts" / f"next_obs_{n:03d}.txt").read_text(encoding="utf-8")
        obs_next = (run_dir / "artifacts" / f"obs_{n + 1:03d}.txt").read_text(encoding="utf-8")
        assert f"http://s.local/{n + 1}" in nxt
        assert f"http://s.local/{n + 1}" in obs_next
    # the final (terminating) step still records its post-action observation
    final = (run_dir / "artifacts" / "next_obs_001.txt").read_text(encoding="utf-8")
    assert "http://s.local/2" in final


def test_parse_error_has_pre_obs_and_no_fabricated_next_obs(tmp_path):
    env = FakeEnvironment(
        reset_observation=make_fake_observation(url="http://pre.local/a"),
        script=[
            {"observation": make_fake_observation(url="http://post.local/b")},
            {
                "observation": make_fake_observation(url="http://post.local/c"),
                "reward": 1.0,
                "terminated": True,
            },
        ],
    )
    agent = BaselineAgent(
        # fail on the SECOND decision (call_count is 1-based)
        model_adapter=MockModelAdapter(["click(bid='1')"], raise_parse_error_on_step={2}),
    )
    runner = EpisodeRunner(agent=agent, env=env, trace_root=tmp_path)
    result = runner.run(TASK, run_id="r")
    assert result.status == RunStatus.ERROR

    run_dir = Path(result.trace_path)
    # step 0 executed normally
    assert (run_dir / "artifacts" / "next_obs_000.txt").exists()
    # step 1 failed at decision time: pre obs exists, next obs must NOT
    assert (run_dir / "artifacts" / "obs_001.txt").exists()
    assert not (run_dir / "artifacts" / "next_obs_001.txt").exists()
    steps = TraceRecorder.read_steps(run_dir)
    assert steps[1].next_observation_ref is None
    assert steps[1].action is None
