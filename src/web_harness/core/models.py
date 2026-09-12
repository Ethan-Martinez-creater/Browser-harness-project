"""Core data models shared across the harness.

These Pydantic models are the stable contract between agent, runtime,
environment adapter, model adapter, trace and evaluation layers. BrowserGym
dicts must never leak beyond the environment adapter.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .errors import ErrorType
from .ids import now_utc_iso


class TaskSpec(BaseModel):
    """A single benchmark task to execute."""

    benchmark: str
    task_id: str
    seed: int | None = None
    max_steps: int = 20
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Normalized environment observation (text-only in Phase 0).

    Screenshots are never sent to the model in Phase 0; saving them as
    artifacts is optional and off by default.
    """

    goal: str
    url: str
    axtree: str | None = None
    dom: str | None = None
    screenshot_path: str | None = None
    open_pages: list[str] = Field(default_factory=list)
    last_action: str | None = None
    last_action_error: str | None = None
    elapsed_time_s: float = 0.0
    raw_artifact_refs: dict[str, str] = Field(default_factory=dict)
    truncated: bool = False  # True if axtree/dom was deterministically cut


class ActionDecision(BaseModel):
    """The single browser action the agent wants to execute.

    `short_reason` is a brief debug-oriented summary only; full model
    chain-of-thought is never persisted.
    """

    action: str
    short_reason: str | None = None


class EnvironmentStep(BaseModel):
    """Result of applying one action to the environment."""

    observation: Observation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    action_error: str | None = None


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"  # terminated by the environment with positive reward
    FAILED = "failed"  # environment terminated the task without success
    TRUNCATED = "truncated"  # environment time limit hit
    MAX_STEPS_REACHED = "max_steps_reached"
    ERROR = "error"  # fatal exception; see error_type


class PromptBundle(BaseModel):
    """Fully materialized prompt for one model call.

    Built by the agent's PromptBuilder; consumed by the ModelAdapter. Keeping
    it as data makes prompts traceable and reproducible.
    """

    system: str
    user: str


class ModelOutput(BaseModel):
    """Everything the harness wants to keep from one model call."""

    decision: ActionDecision
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_text: str | None = None  # persisted as artifact, not sent anywhere else


class StepRecord(BaseModel):
    """One observe-decide-act cycle. All fields must be JSON-serializable."""

    run_id: str
    step_index: int
    timestamp: str = Field(default_factory=now_utc_iso)
    url: str | None = None
    observation_ref: str | None = None  # pre-action observation (decision input)
    next_observation_ref: str | None = None  # post-action observation (result)
    prompt_ref: str | None = None
    model_response_ref: str | None = None
    action: str | None = None
    action_error: str | None = None
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_name: str | None = None
    short_reason: str | None = None
    error_type: ErrorType | None = None


class RunResult(BaseModel):
    """Final outcome of one episode."""

    run_id: str
    task_spec: TaskSpec
    status: RunStatus
    success: bool = False
    final_reward: float = 0.0
    num_steps: int = 0
    started_at: str = Field(default_factory=now_utc_iso)
    ended_at: str = Field(default_factory=now_utc_iso)
    duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float | None = None  # null unless a reliable price table exists
    action_error_count: int = 0
    trace_path: str | None = None
    error_type: ErrorType | None = None
    error_message: str | None = None


def json_dump(model: BaseModel) -> str:
    """Canonical JSON serialization for all core models."""
    return model.model_dump_json(exclude_none=False)


def json_load(model_cls: type[BaseModel], raw: str | bytes) -> BaseModel:
    """Canonical JSON deserialization for all core models."""
    return model_cls.model_validate_json(raw)
