"""Phase 0 baseline agent: observe -> build prompt -> model -> one action.

Deliberately minimal: no planning, no retry, no recovery, no long-term
memory, no verification. It only selects the next single browser action from
the current goal, a bounded history and the current observation.
"""

from __future__ import annotations

from web_harness.agents.base import AgentTurn
from web_harness.agents.prompt import PromptBuilder
from web_harness.core.models import (
    Observation,
    PromptBundle,
    StepRecord,
    TaskSpec,
)
from web_harness.env.action_contract import ActionContract
from web_harness.models.base import ModelAdapter, select_history


class BaselineAgent:
    def __init__(
        self,
        *,
        model_adapter: ModelAdapter,
        max_history_steps: int = 4,
    ):
        self.model_adapter = model_adapter
        self.max_history_steps = max_history_steps
        self.prompt_builder = PromptBuilder(max_history_steps=max_history_steps)

    def decide(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        action_contract: ActionContract,
        repair_feedback: str | None = None,
        recovery_directive=None,
    ) -> tuple[AgentTurn, PromptBundle]:
        """Return the decision plus the exact prompt used (for tracing).

        `repair_feedback` is set only by format-repair retries (Phase 1B);
        `recovery_directive` only by environment recovery (Phase 1C). The
        normal path stays identical to Phase 0/1A/1B.
        """
        prompt = self.prompt_builder.build(
            task=task,
            observation=observation,
            history=history,
            action_contract=action_contract,
            repair_feedback=repair_feedback,
            recovery_directive=recovery_directive,
        )
        model_output = self.model_adapter.generate_action(
            task=task,
            observation=observation,
            history=select_history(history, self.max_history_steps),
            prompt=prompt,
        )
        return (
            AgentTurn(decision=model_output.decision, model_output=model_output),
            prompt,
        )
