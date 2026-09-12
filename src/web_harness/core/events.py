"""Runtime event model for the reliability event stream (`events.jsonl`).

Events record what verifier/policy/retry/recovery/replan/budget components
did, WITHOUT changing the meaning of `steps.jsonl` (the agent trajectory).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from web_harness.core.ids import now_utc_iso


def new_event_id() -> str:
    import uuid

    return f"evt-{uuid.uuid4().hex[:12]}"


class RuntimeEventType(StrEnum):
    VERIFICATION = "verification"
    POLICY_DECISION = "policy_decision"
    RETRY = "retry"
    RECOVERY = "recovery"
    REPLAN = "replan"
    BUDGET = "budget"


class RuntimeEvent(BaseModel):
    event_id: str
    run_id: str
    timestamp: str = Field(default_factory=now_utc_iso)
    event_type: RuntimeEventType
    step_index: int | None = None
    attempt_index: int | None = None
    component: str
    outcome: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
