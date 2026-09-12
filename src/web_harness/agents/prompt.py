"""Prompt building for the baseline agent.

The prompt intentionally contains no planner, no memory instructions and no
verification hints — it is the pure Phase 0 baseline. It includes: task goal,
current URL, axtree view, last action and error, a short recent-history
digest, the available action schema and the required output format.
"""

from __future__ import annotations

from web_harness.core.models import Observation, PromptBundle, StepRecord, TaskSpec

ACTION_SCHEMA = """\
You control a browser through ONE high-level action per step. Available actions:

  click(bid='ID')            click the element with the given bid
  type(bid='ID', value='TEXT')   focus element and type text (replaces content)
  select_option(bid='ID', option='TEXT')  pick an option in a <select>
  check_uncheck(bid='ID')    toggle a checkbox
  scroll(x=0, y=300)         scroll the page
  noop()                     do nothing this step
  new_tab()                  open a new empty tab

`bid` values are the bracketed ids in the accessibility tree, e.g. `[42] button 'OK'`
means you can `click(bid='42')`."""

OUTPUT_FORMAT = """\
Respond with ONE JSON object and nothing else:

{"action": "<one action string>", "short_reason": "<max ~15 words, why>"}

Example: {"action": "click(bid='13')", "short_reason": "target button found"}"""


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
    ) -> PromptBundle:
        system = (
            "You are a careful web automation agent. You complete the given "
            "task by choosing exactly one browser action at a time, based on "
            "the current accessibility tree. Prefer the simplest action that "
            "makes progress toward the goal.\n\n"
            f"{ACTION_SCHEMA}\n\n{OUTPUT_FORMAT}"
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
        return PromptBundle(system=system, user="\n\n".join(user_parts))
