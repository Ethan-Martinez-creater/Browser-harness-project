"""Deterministic mock model for unit tests and offline verification."""

from __future__ import annotations

import re

from web_harness.core.errors import ModelOutputParseError
from web_harness.core.models import (
    ActionDecision,
    ModelOutput,
    Observation,
    PromptBundle,
    StepRecord,
    TaskSpec,
)


class MockModelAdapter:
    """Scripted model.

    - `actions`: exact action strings returned in order; the last one repeats.
    - `until_action`: when provided, an observation whose goal/url/axtree
      matches this regex keeps returning the first action; this allows
      "act only on state A" style tests.
    - `raise_parse_error_on_step`: steps where the adapter should simulate an
      unparseable model response (raises ModelOutputParseError).
    """

    def __init__(
        self,
        actions: list[str],
        *,
        short_reasons: list[str] | None = None,
        until_action: str | None = None,
        raise_parse_error_on_step: set[int] | None = None,
    ):
        if not actions:
            raise ValueError("MockModelAdapter needs at least one action")
        self.actions = actions
        self.short_reasons = short_reasons or []
        self.until_action = re.compile(until_action) if until_action else None
        self.raise_parse_error_on_step = raise_parse_error_on_step or set()
        self.call_count = 0
        self.last_prompt: PromptBundle | None = None

    def generate_action(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        prompt: PromptBundle,
    ) -> ModelOutput:
        self.call_count += 1
        self.last_prompt = prompt
        if self.call_count in self.raise_parse_error_on_step:
            raise ModelOutputParseError("mock model produced unparseable output")

        idx = 0
        if self.until_action is not None and self.call_count > 1:
            haystack = f"{observation.goal}\n{observation.url}\n{observation.axtree or ''}"
            if self.until_action.search(haystack):
                idx = 0
            else:
                idx = min(self.call_count - 1, len(self.actions) - 1)
        else:
            idx = min(self.call_count - 1, len(self.actions) - 1)

        reason = (
            self.short_reasons[min(idx, len(self.short_reasons) - 1)]
            if self.short_reasons
            else None
        )
        return ModelOutput(
            decision=ActionDecision(action=self.actions[idx], short_reason=reason),
            model_name="mock",
        )

    def generate_structured(
        self,
        *,
        prompt: PromptBundle,
    ) -> ModelOutput:
        """Structured generation for Phase 1D: returns the scripted text
        (a RecoveryPlan JSON) verbatim, unparsed, in `decision.action`."""
        self.call_count += 1
        idx = min(self.call_count - 1, len(self.actions) - 1)
        text = self.actions[idx]
        return ModelOutput(
            decision=ActionDecision(action=text),
            model_name="mock",
            raw_text=text,
        )
