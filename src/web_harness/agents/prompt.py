"""Prompt building for the baseline agent.

The prompt intentionally contains no planner, no memory instructions and no
verification hints — it is the pure Phase 0 baseline. It includes: task goal,
current URL, axtree view, last action and error, a short recent-history
digest, the available action schema and the required output format.
"""

from __future__ import annotations

from web_harness.core.models import Observation, PromptBundle, StepRecord, TaskSpec
from web_harness.core.reliability import RecoveryDirective
from web_harness.env.action_contract import ActionContract

OUTPUT_FORMAT = """\
Respond with ONE JSON object and nothing else:

{"action": "<one action from the list above>", "short_reason": "<max ~15 words, why>"}

The `action` value must be ONE complete, executable action call copied from
the list above, with ALL of its required arguments inline (an element action
must include the target element id directly in the call). Never split
arguments into separate JSON fields, and never output a bare action name."""


REPAIR_FEEDBACK = """\
Previous response did not match the required JSON action format.

Return exactly one valid JSON object using the current ActionContract.
Do not add markdown fences or extra prose."""


def summarize_history(history: list[StepRecord], max_steps: int) -> str:
    """Short deterministic digest of the last steps (no model output text)."""
    if max_steps <= 0 or not history:
        return "(no previous steps)"
    recent = history[-max_steps:]
    lines = []
    for rec in recent:
        parts = [f"step {rec.step_index}: action={rec.action or 'NONE'}"]
        if rec.action_error:
            parts.append(f"error={rec.action_error}")
        if rec.reward is not None:
            parts.append(f"reward={rec.reward:g}")
        lines.append("; ".join(parts))
    return "\n".join(lines)


class PromptBuilder:
    def __init__(self, *, max_history_steps: int = 4):
        self.max_history_steps = max_history_steps

    def build(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        action_contract: ActionContract,
        repair_feedback: str | None = None,
        recovery_directive: RecoveryDirective | None = None,
    ) -> PromptBundle:
        system = (
            "You are a careful web automation agent. You complete the given "
            "task by choosing exactly one browser action at a time, based on "
            "the current accessibility tree. Prefer the simplest action that "
            "makes progress toward the goal.\n\n"
            f"{action_contract.render_for_prompt()}\n\n{OUTPUT_FORMAT}"
        )
        history_digest = summarize_history(history, self.max_history_steps)
        view = observation.axtree or "(accessibility tree unavailable)"
        if observation.truncated:
            view += "\n[NOTE: observation was truncated]"
        user_parts = [
            f"# Task goal\n{observation.goal or task.task_id}",
            f"# Current URL\n{observation.url or '(unknown)'}",
            "# Open pages\n" + ("\n".join(observation.open_pages) or "(one page)"),
            "# Last action\n"
            + (observation.last_action or "(none yet)")
            + "\n# Last action error\n"
            + (observation.last_action_error or "(none)"),
            f"# Recent history (last {self.max_history_steps} steps)\n{history_digest}",
            f"# Current page accessibility tree\n{view}",
            "Choose the next action now. Respond with the JSON object only.",
        ]
        user = "\n\n".join(user_parts)
        if repair_feedback:
            # minimal format-repair feedback only (no reflection, no replanning)
            user += f"\n\n# Format correction\n{repair_feedback}"
        if recovery_directive is not None:
            # environment-recovery feedback is a separate, typed channel
            lines = ["# Reliability recovery feedback"]
            if recovery_directive.feedback:
                lines.append(recovery_directive.feedback)
            if recovery_directive.blocked_actions:
                lines.append(
                    "Blocked actions (do NOT execute these this step): "
                    + ", ".join(recovery_directive.blocked_actions)
                )
            user += "\n\n".join(["\n".join(lines)])
        return PromptBundle(system=system, user=user)
