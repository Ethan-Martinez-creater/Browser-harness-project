"""Integration test: the advertised action contract must be executable.

For every action the prompt contract exposes, this test verifies that
BrowserGym's action parser and the live environment accept it (no
last_action_error). The goal is contract validation, not task completion.
"""

import re

import pytest

from web_harness.env.browsergym_adapter import BrowserGymAdapter

pytestmark = pytest.mark.integration

pytest.importorskip("browsergym.miniwob")

BOOTSTRAP = "noop(wait_ms=500)"


def make_adapter() -> BrowserGymAdapter:
    return BrowserGymAdapter(bootstrap_action=BOOTSTRAP, headless=True)


def first_bid(obs_axtree: str, roles: tuple[str, ...]) -> str | None:
    for line in (obs_axtree or "").splitlines():
        m = re.search(r"\[([^\]\s]+)\]\s*(\w+)", line)
        if m and m.group(2) in roles:
            return m.group(1)
    return None


def element_label(obs_axtree: str, bid: str) -> str | None:
    """The quoted name text of an axtree element, if any."""
    m = re.search(rf"\[{re.escape(bid)}\]\s*\w+\s+'([^']*)'", obs_axtree or "")
    return m.group(1) if m else None


def run_action(task_id: str, build_action):
    """Reset the task, derive an action from the axtree, execute it.

    Returns (env, env_step); asserts the action was accepted by the env.
    """
    from web_harness.core.models import TaskSpec

    env = make_adapter()
    try:
        obs = env.reset(TaskSpec(benchmark="miniwob", task_id=task_id, seed=0))
        action = build_action(env, obs)
        step = env.step(action)
        return env, obs, step, action
    finally:
        env.close()


def check_accepted(env, obs, step, action):
    """The action must parse and execute without an environment error."""
    assert step.action_error is None, (
        f"action {action!r} rejected by environment: {step.action_error!r}"
    )


@pytest.fixture(scope="module")
def miniwob_available():
    from web_harness.env.browsergym_adapter import resolve_miniwob_url

    try:
        resolve_miniwob_url(None)
    except Exception:
        pytest.skip("MiniWoB HTML source not available (set MINIWOB_URL)")


def test_contract_matches_installed_action_mapping(miniwob_available):
    """The contract must list exactly what the env's action mapping accepts."""
    env = make_adapter()
    try:
        contract = env.action_contract()
        action_set = env._action_set
        assert set(contract.allowed_names()) == set(action_set.action_set.keys())
    finally:
        env.close()


def test_click_action_accepted(miniwob_available):
    def build(env, obs):
        bid = first_bid(obs.axtree, ("button", "link"))
        assert bid, "no clickable element found"
        return f"click(bid={bid!r})"

    check_accepted(*run_action("click-test", build))


def test_fill_action_accepted(miniwob_available):
    def build(env, obs):
        bid = first_bid(obs.axtree, ("textbox", "searchbox", "combobox"))
        assert bid, "no text input found"
        return f"fill(bid={bid!r}, value='hello')"

    check_accepted(*run_action("enter-text", build))


def test_select_option_action_accepted(miniwob_available):
    def build(env, obs):
        bid = first_bid(obs.axtree, ("combobox", "listbox"))
        assert bid, "no select element found"
        # selecting the currently-selected value must always be accepted
        label = element_label(obs.axtree, bid) or ""
        return f"select_option(bid={bid!r}, options={label!r})"

    check_accepted(*run_action("choose-list", build))


def test_check_uncheck_actions_accepted(miniwob_available):
    from web_harness.core.models import TaskSpec

    env = make_adapter()
    try:
        obs = env.reset(TaskSpec(benchmark="miniwob", task_id="click-checkboxes", seed=0))
        bid = first_bid(obs.axtree, ("checkbox",))
        assert bid, "no checkbox found"
        step = env.step(f"check(bid={bid!r})")
        assert step.action_error is None, step.action_error
        step = env.step(f"uncheck(bid={bid!r})")
        assert step.action_error is None, step.action_error
    finally:
        env.close()


def test_scroll_action_accepted(miniwob_available):
    check_accepted(*run_action("click-test", lambda env, obs: "scroll(0, 100)"))


def test_press_action_accepted(miniwob_available):
    def build(env, obs):
        bid = first_bid(obs.axtree, ("button", "textbox", "link"))
        assert bid, "no focusable element found"
        return f"press(bid={bid!r}, key_comb='Enter')"

    check_accepted(*run_action("click-test", build))


def test_noop_and_new_tab_accepted(miniwob_available):
    env, obs, step, action = run_action("click-test", lambda env, obs: "noop()")
    check_accepted(env, obs, step, action)
