"""Unit tests for LoopDetector (transition-based, not action-based)."""

import pytest

from web_harness.core.reliability import FailureKind, FailureSeverity, ReliabilityState
from web_harness.reliability.detectors.loop import LoopDetector

from .conftest import make_ctx


def test_single_identical_action_is_not_a_loop():
    # same action, but the state changed => no loop
    ctx = make_ctx(action="click(bid='1')")
    ctx.reliability_state.recent_actions.append("click(bid='1')")
    assert LoopDetector().detect(ctx) == []


def test_same_transition_repeated_twice_is_loop():
    # step 1: A -> click -> B (recorded in state)
    # step 2: B -> click -> B? no: loop requires same pre AND post state
    state = ReliabilityState()
    sig = "trans1"
    state.recent_transition_signatures = [sig]
    ctx = make_ctx(reliability_state=state)
    ctx.reliability_state.recent_transition_signatures = [sig]
    # force the current transition signature to match
    ctx = ctx.model_copy(update={"transition_signature": sig})
    signals = LoopDetector().detect(ctx)
    assert len(signals) == 1
    s = signals[0]
    assert s.kind == FailureKind.LOOP_DETECTED
    assert s.severity == FailureSeverity.ERROR
    assert s.recoverable is True


def test_alternating_transitions_are_not_loop():
    state = ReliabilityState(recent_transition_signatures=["t1", "t2"])
    ctx = make_ctx(reliability_state=state)
    ctx = ctx.model_copy(update={"transition_signature": "t1"})
    assert LoopDetector().detect(ctx) == []


def test_three_consecutive_same_transitions_is_loop():
    state = ReliabilityState(recent_transition_signatures=["t9", "t9"])
    ctx = make_ctx(reliability_state=state)
    ctx = ctx.model_copy(update={"transition_signature": "t9"})
    (signal,) = LoopDetector().detect(ctx)
    assert signal.evidence["consecutive_occurrences"] == 3


def test_threshold_configurable():
    detector = LoopDetector(consecutive_threshold=3)
    state = ReliabilityState(recent_transition_signatures=["t9"])
    ctx = make_ctx(reliability_state=state)
    ctx = ctx.model_copy(update={"transition_signature": "t9"})
    assert detector.detect(ctx) == []  # streak = 2 < 3


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        LoopDetector(consecutive_threshold=1)
