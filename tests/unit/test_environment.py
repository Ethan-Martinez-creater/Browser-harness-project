"""Unit tests for ObservationNormalizer and FakeEnvironment."""

from web_harness.core.models import TaskSpec
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.env.observation import ObservationNormalizer, clickable_bids, truncate_text


def test_truncate_text_deterministic():
    text = "a" * 100
    cut, was = truncate_text(text, 50)
    assert was and len(cut) <= 50
    cut2, was2 = truncate_text(text, 50)
    assert cut == cut2 and was2
    full, was3 = truncate_text(text, 100)
    assert not was3 and full == text


def test_truncate_text_no_limit():
    _, was = truncate_text("abc", 0)
    assert not was


def test_normalizer_axtree_priority_and_truncation():
    n = ObservationNormalizer(observation_char_limit=30)
    obs = n.from_axtree_parts(
        goal="g",
        url="http://x/",
        axtree_text="x" * 100,
        open_pages=["http://x/"],
    )
    assert obs.truncated
    assert len(obs.axtree) <= 30
    assert obs.dom is None  # DOM disabled by default


def test_clickable_bids_order():
    axtree = "[3] button 'A'\n[1] link 'B'\n[3] button 'A'"
    assert clickable_bids(axtree) == ["3", "1"]


def test_fake_environment_scripted():
    env = FakeEnvironment(
        script=[
            {"observation": make_fake_observation(url="http://fake.local/2"), "reward": 0.0},
            {
                "observation": make_fake_observation(url="http://fake.local/3"),
                "reward": 1.0,
                "terminated": True,
            },
        ]
    )
    obs = env.reset(TaskSpec(benchmark="fake", task_id="t1"))
    assert obs.goal == "fake goal"
    s1 = env.step("click(bid='1')")
    assert s1.reward == 0.0 and not s1.terminated
    s2 = env.step("click(bid='1')")
    assert s2.reward == 1.0 and s2.terminated
    # script exhausted: last entry repeats
    s3 = env.step("noop()")
    assert s3.terminated
    env.close()
    assert env.close_calls == 1
    assert env.executed_actions == ["click(bid='1')", "click(bid='1')", "noop()"]


def test_fake_environment_raise_on_step():
    env = FakeEnvironment(raise_on_step_index={1})
    env.reset(TaskSpec(benchmark="fake", task_id="t1"))
    env.step("noop()")
    try:
        env.step("noop()")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
    env.close()
