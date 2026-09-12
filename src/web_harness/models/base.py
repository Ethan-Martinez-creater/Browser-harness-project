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

__all__ = ["ModelAdapter", "ModelOutput", "select_history"]

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
