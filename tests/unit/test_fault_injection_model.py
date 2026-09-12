"""Unit tests for FaultInjectingModelAdapter determinism."""

import pytest

from web_harness.core.errors import ModelApiError, ModelOutputParseError
from web_harness.core.models import Observation, PromptBundle, TaskSpec
from web_harness.env.action_contract import ActionContract
from web_harness.evaluation.fault_injection import FaultInjectingModelAdapter
from web_harness.models.mock import MockModelAdapter

TASK = TaskSpec(benchmark="fake", task_id="t")
OBS = Observation(goal="g", url="u")
PROMPT = PromptBundle(system="s", user="u")
CONTRACT = ActionContract(benchmark="fake", actions=[])


def call(adapter):
    return adapter.generate_action(task=TASK, observation=OBS, history=[], prompt=PROMPT)


def test_passthrough_without_faults():
    adapter = FaultInjectingModelAdapter(MockModelAdapter(["click(bid='1')"]))
    out1 = call(adapter)
    out2 = call(adapter)
    assert out1.decision.action == "click(bid='1')"
    assert out2.decision.action == "click(bid='1')"


def test_api_error_injection_deterministic():
    adapter = FaultInjectingModelAdapter(
        MockModelAdapter(["noop()"]), api_error_on_calls={0, 2}
    )
    with pytest.raises(ModelApiError):
        call(adapter)
    call(adapter)  # call 1 passes through
    with pytest.raises(ModelApiError):
        call(adapter)  # call 2 injected again
    call(adapter)  # call 3 passes through
    assert adapter.call_count == 4


def test_parse_error_injection_carries_usage():
    adapter = FaultInjectingModelAdapter(
        MockModelAdapter(["noop()"]), parse_error_on_calls={0}
    )
    with pytest.raises(ModelOutputParseError) as exc_info:
        call(adapter)
    assert exc_info.value.raw_text
    assert exc_info.value.input_tokens > 0
    assert exc_info.value.output_tokens > 0
    # identical on re-run (deterministic)
    adapter2 = FaultInjectingModelAdapter(
        MockModelAdapter(["noop()"]), parse_error_on_calls={0}
    )
    with pytest.raises(ModelOutputParseError) as exc_info2:
        call(adapter2)
    assert exc_info.value.raw_text == exc_info2.value.raw_text


def test_on_call_callback_fires_for_every_call():
    seen = []
    adapter = FaultInjectingModelAdapter(
        MockModelAdapter(["noop()"]), on_call=seen.append
    )
    call(adapter)
    call(adapter)
    assert seen == [0, 1]
