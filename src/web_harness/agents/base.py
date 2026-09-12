"""Agent abstraction.

An agent turns (task, current observation, short history) into exactly one
action decision per step, using the action contract provided by the current
environment. Everything else — planning, memory, verification — is
deliberately out of scope for the Phase 0 baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from web_harness.core.reliability import RecoveryDirective

from pydantic import BaseModel

from web_harness.core.models import ActionDecision, Observation, PromptBundle, StepRecord, TaskSpec
from web_harness.env.action_contract import ActionContract
from web_harness.models.base import ModelOutput


class AgentTurn(BaseModel):
    """What an agent returns for one decision point."""

    decision: ActionDecision
    model_output: ModelOutput | None = None


class Agent(Protocol):
    def decide(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        action_contract: ActionContract,
        repair_feedback: str | None = None,
        recovery_directive: RecoveryDirective | None = None,
    ) -> tuple[AgentTurn, PromptBundle]:
        """Return the decision plus the exact prompt used (for tracing).

        `repair_feedback` is set only by format-repair retries (Phase 1B).
        `recovery_directive` is set only by environment-side recovery
        (Phase 1C); the two feedback channels are intentionally separate.
        """
        ...
