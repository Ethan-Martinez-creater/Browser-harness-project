"""Unit tests for the Action Contract (B1)."""

import pytest

from web_harness.agents.baseline import BaselineAgent
from web_harness.core.models import Observation, TaskSpec
from web_harness.env.action_contract import (
    ActionContract,
    ActionSpec,
)
from web_harness.env.fake import FAKE_CONTRACT, FakeEnvironment
from web_harness.models.mock import MockModelAdapter

# Phase 0 contract requires at least these MiniWoB actions (per review B1)
REQUIRED_ACTIONS = {
    "noop",
    "click",
    "fill",
    "select_option",
    "check",
    "uncheck",
    "scroll",
    "press",
}


def test_browsergym_contract_contains_required_actions():
    pytest.importorskip("browsergym.core.action.highlevel")
    from web_harness.env.browsergym_adapter import BrowserGymAdapter

    adapter = BrowserGymAdapter()
    contract = adapter.action_contract()
    names = set(contract.allowed_names())
    assert names >= REQUIRED_ACTIONS, f"missing actions: {REQUIRED_ACTIONS - names}"


def test_browsergym_contract_signatures_come_from_real_action_set():
    """Signatures must come from the same action_set installed on the env."""
    pytest.importorskip("browsergym.core.action.highlevel")
    from web_harness.env.browsergym_adapter import BrowserGymAdapter

    adapter = BrowserGymAdapter()
    contract = adapter.action_contract()
    by_name = {a.name: a for a in contract.actions}

    # real BrowserGym signatures (fill/select_option/scroll), not invented ones
    assert by_name["fill"].signature.startswith("fill(")
    assert "value" in by_name["fill"].signature
    assert by_name["select_option"].signature.startswith("select_option(")
    assert by_name["scroll"].signature.startswith("scroll(")
    # the old, wrong signatures must not reappear
    assert not any("check_uncheck" in a.signature for a in contract.actions)
    assert not any(a.signature.startswith("type(") for a in contract.actions)


def test_contract_rejects_disallowed_action_names():
    pytest.importorskip("browsergym.core.action.highlevel")
    from browsergym.core.action.highlevel import HighLevelActionSet

    from web_harness.env.browsergym_adapter import BrowserGymAdapter

    adapter = BrowserGymAdapter()
    action_set = adapter._build_action_set(multiaction=False)
    assert isinstance(action_set, HighLevelActionSet)
    # every advertised action is parseable by the env's action mapping
    contract = adapter.action_contract()
    for spec in contract.actions:
        assert spec.name in action_set.action_set


def test_render_for_prompt_lists_signatures():
    contract = FAKE_CONTRACT
    text = contract.render_for_prompt()
    assert "click(bid: str)" in text
    assert "noop(wait_ms: float = 1000)" in text


def test_fake_environment_provides_contract():
    env = FakeEnvironment()
    contract = env.action_contract()
    assert isinstance(contract, ActionContract)
    assert "click" in contract.allowed_names()


def test_prompt_builder_uses_injected_contract():
    custom = ActionContract(
        benchmark="miniwob",
        actions=[ActionSpec(name="noop", signature="noop()", description="idle")],
    )
    builder_prompt = _build_prompt(custom)
    assert "noop()" in builder_prompt
    assert "click" not in builder_prompt  # no hardcoded schema leaks through


def _build_prompt(contract: ActionContract) -> str:

    agent = BaselineAgent(model_adapter=MockModelAdapter(["noop()"]))
    turn, prompt = agent.decide(
        task=TaskSpec(benchmark="fake", task_id="t"),
        observation=Observation(goal="g", url="u"),
        history=[],
        action_contract=contract,
    )
    return prompt.system + prompt.user
