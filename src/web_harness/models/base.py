"""Model adapter abstraction.

A ModelAdapter turns a fully materialized prompt into an ActionDecision. The
provider lives behind this boundary: agent code never imports `openai` or
knows which vendor is in use.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from web_harness.core.models import (
    ModelOutput,  # re-exported: the cross-layer model call contract
    Observation,
    PromptBundle,
    StepRecord,
    TaskSpec,
)


class StructuredModelAdapter(Protocol):
    """Provider-neutral raw/structured generation (Phase 1D).

    The adapter returns the model text UNPARSED in `decision.action`; JSON
    validation and schema interpretation belong to the caller. Implemented by
    the OpenAI-compatible and Mock adapters — no provider SDK may leak past
    this boundary."""

    def generate_structured(
        self,
        *,
        prompt: PromptBundle,
    ) -> ModelOutput: ...


__all__ = ["ModelAdapter", "ModelOutput", "StructuredModelAdapter", "select_history"]

HistoryRule = Callable[[list[StepRecord]], bool]


class ModelAdapter(Protocol):
    def generate_action(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        prompt: PromptBundle,
    ) -> ModelOutput: ...


def select_history(history: list[StepRecord], max_steps: int) -> list[StepRecord]:
    """Most recent `max_steps` records, in chronological order."""
    if max_steps <= 0:
        return []
    return list(history[-max_steps:])
