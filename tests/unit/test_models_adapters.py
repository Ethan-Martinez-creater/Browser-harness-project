"""Unit tests for model adapters (mock + structured parsing)."""

import pytest

from web_harness.core.errors import ModelApiError, ModelOutputParseError
from web_harness.core.models import Observation, PromptBundle, TaskSpec
from web_harness.models.base import select_history
from web_harness.models.mock import MockModelAdapter
from web_harness.models.openai_compatible import parse_structured_action

TASK = TaskSpec(benchmark="miniwob", task_id="t")
OBS = Observation(goal="g", url="u")
PROMPT = PromptBundle(system="s", user="u")


def test_mock_sequential_actions():
    m = MockModelAdapter(["click(bid='1')", "type(bid='2', value='x')"])
    out1 = m.generate_action(task=TASK, observation=OBS, history=[], prompt=PROMPT)
    out2 = m.generate_action(task=TASK, observation=OBS, history=[], prompt=PROMPT)
    out3 = m.generate_action(task=TASK, observation=OBS, history=[], prompt=PROMPT)
    assert out1.decision.action == "click(bid='1')"
    assert out2.decision.action == "type(bid='2', value='x')"
    assert out3.decision.action == "type(bid='2', value='x')"  # last repeats
    assert out3.model_name == "mock"


def test_mock_records_prompt():
    m = MockModelAdapter(["noop()"])
    m.generate_action(task=TASK, observation=OBS, history=[], prompt=PROMPT)
    assert m.last_prompt == PROMPT


def test_mock_parse_error_simulation():
    m = MockModelAdapter(["noop()"], raise_parse_error_on_step={2})
    m.generate_action(task=TASK, observation=OBS, history=[], prompt=PROMPT)
    with pytest.raises(ModelOutputParseError):
        m.generate_action(task=TASK, observation=OBS, history=[], prompt=PROMPT)


def test_parse_structured_bare_json():
    d = parse_structured_action('{"action": "click(bid=\'13\')", "short_reason": "ok"}')
    assert d.action == "click(bid='13')"
    assert d.short_reason == "ok"


def test_parse_structured_json_embedded():
    text = 'Sure! Here is my choice:\n{"action": "noop()"}\nDone.'
    assert parse_structured_action(text).action == "noop()"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "no json at all",
        '{"no_action_field": true}',
        '{"action": ""}',
        '{"action": 42}',
    ],
)
def test_parse_structured_failures(bad):
    with pytest.raises(ModelOutputParseError):
        parse_structured_action(bad)


def test_openai_adapter_requires_config(monkeypatch):
    from web_harness.models.openai_compatible import OpenAICompatibleModelAdapter

    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ModelApiError):
        OpenAICompatibleModelAdapter(model="m")


def test_select_history_bound():
    from web_harness.core.models import StepRecord

    hist = [StepRecord(run_id="r", step_index=i) for i in range(10)]
    sel = select_history(hist, 4)
    assert [s.step_index for s in sel] == [6, 7, 8, 9]
    assert select_history(hist, 0) == []
