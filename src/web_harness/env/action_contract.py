"""Action Contract: the single source of truth for agent-visible actions.

The contract is generated FROM the environment's actual action set (for
BrowserGym: the exact `HighLevelActionSet` instance whose `to_python_code` is
installed as the env's action mapping). The prompt can therefore never drift
from what the environment accepts — every action advertised to the agent is
guaranteed parseable and executable by the current EnvironmentAdapter.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ActionSpec(BaseModel):
    name: str
    signature: str
    description: str
    examples: list[str] = Field(default_factory=list)


class ActionContract(BaseModel):
    """The set of actions the current environment actually accepts."""

    benchmark: str
    actions: list[ActionSpec]

    def allowed_names(self) -> list[str]:
        return [a.name for a in self.actions]

    def render_for_prompt(self) -> str:
        """Deterministic prompt-ready description of the action space."""
        lines = [
            "You control the browser with ONE action per step.",
            "Available actions (use exactly these signatures):",
            "",
        ]
        for spec in self.actions:
            lines.append(f"  {spec.signature}")
            if spec.description:
                lines.append(f"      {spec.description}")
            for example in spec.examples[:2]:
                lines.append(f"      e.g. {example}")
        lines.append("")
        lines.append("Element references (`bid`) are the bracketed ids in the")
        lines.append("accessibility tree, e.g. `[42] button 'OK'` refers to the")
        lines.append("element you would target with its id '42'.")
        return "\n".join(lines)


def action_contract_from_browsergym(action_set, *, benchmark: str) -> ActionContract:
    """Build an ActionContract from a BrowserGym HighLevelActionSet instance.

    `action_set` must be the same object (or built with the same configuration)
    that is installed as the environment's action_mapping, so advertised
    actions == executable actions.
    """
    specs: list[ActionSpec] = []
    for name, hl_action in action_set.action_set.items():
        examples = [e for e in (getattr(hl_action, "examples", None) or [])]
        specs.append(
            ActionSpec(
                name=name,
                signature=hl_action.signature,
                description=hl_action.description or "",
                examples=examples,
            )
        )
    specs.sort(key=lambda s: s.name)
    return ActionContract(benchmark=benchmark, actions=specs)
