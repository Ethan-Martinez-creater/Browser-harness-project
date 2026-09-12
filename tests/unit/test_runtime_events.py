"""Unit tests for the runtime event stream (events.jsonl)."""

from web_harness.core.events import RuntimeEvent, RuntimeEventType, new_event_id
from web_harness.observability.trace import TraceRecorder


def test_event_model_roundtrip():
    event = RuntimeEvent(
        event_id=new_event_id(),
        run_id="run-1",
        event_type=RuntimeEventType.VERIFICATION,
        step_index=3,
        attempt_index=None,
        component="verification_engine",
        outcome="pass",
        data={"signals": []},
    )
    assert RuntimeEvent.model_validate_json(event.model_dump_json()) == event


def test_event_ids_unique():
    assert len({new_event_id() for _ in range(100)}) == 100


def test_events_persisted_and_reloadable(tmp_path):
    recorder = TraceRecorder(tmp_path / "run-1")
    for i in range(3):
        recorder.record_event(
            RuntimeEvent(
                event_id=new_event_id(),
                run_id="run-1",
                event_type=RuntimeEventType.VERIFICATION,
                step_index=i,
                component="verification_engine",
                outcome="pass" if i == 0 else "fail",
                data={"i": i},
            )
        )
    events = TraceRecorder.read_events(tmp_path / "run-1")
    assert len(events) == 3
    assert events[0].step_index == 0
    assert events[2].outcome == "fail"
    # steps.jsonl untouched by the event stream
    assert not (tmp_path / "run-1" / "steps.jsonl").exists()


def test_read_events_missing_file(tmp_path):
    assert TraceRecorder.read_events(tmp_path / "nope") == []
