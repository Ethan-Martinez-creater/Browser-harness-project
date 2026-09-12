"""Deterministic fake environment for unit tests and offline pipeline checks.

No browser, no network: state transitions are driven by a scripted list of
(EnvironmentStep inputs) so runtime behavior is fully predictable.
"""

from __future__ import annotations

from web_harness.core.models import EnvironmentStep, Observation, TaskSpec
from web_harness.env.action_contract import ActionContract, ActionSpec
from web_harness.env.observation import ObservationNormalizer

FAKE_CONTRACT = ActionContract(
    benchmark="fake",
    actions=[
        ActionSpec(
            name="click",
            signature="click(bid: str)",
            description="Click the element with the given bid.",
            examples=["click(bid='1')"],
        ),
        ActionSpec(
            name="noop",
            signature="noop(wait_ms: float = 1000)",
            description="Do nothing for a short wait.",
            examples=["noop()"],
        ),
    ],
)


class FakeEnvironment:
    """Scripted environment.

    `script` is a list of step outcomes. Step i returns script[i]; when the
    script is exhausted it keeps returning the last entry. Each entry may be
    an EnvironmentStep or a dict of its fields (reward/terminated/truncated/
    action_error/observation overrides).
    """

    def __init__(
        self,
        *,
        script: list[EnvironmentStep | dict] | None = None,
        reset_observation: Observation | None = None,
        raise_on_step_index: set[int] | None = None,
    ):
        from web_harness.core.models import EnvironmentStep as _ES

        self.script: list[_ES] = []
        default_obs = reset_observation or Observation(goal="fake goal", url="http://fake.local")
        for entry in script or []:
            if isinstance(entry, _ES):
                self.script.append(entry)
            else:
                fields = dict(entry)
                if "observation" not in fields:
                    fields["observation"] = default_obs.model_copy(deep=True)
                self.script.append(_ES(**fields))
        if not self.script:
            self.script = [_ES(observation=Observation(goal="", url=""))]
        self.reset_observation = reset_observation or Observation(
            goal="fake goal", url="http://fake.local/start"
        )
        self.raise_on_step_index = raise_on_step_index or set()
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.executed_actions: list[str] = []
        self.closed_while_active = False

    def reset(self, task: TaskSpec) -> Observation:
        self.reset_calls += 1
        self.step_calls = 0
        return self.reset_observation.model_copy(deep=True)

    def step(self, action: str) -> EnvironmentStep:
        if self.step_calls in self.raise_on_step_index:
            self.step_calls += 1
            raise RuntimeError(f"fake environment failure on step {self.step_calls - 1}")
        entry = self.script[min(self.step_calls, len(self.script) - 1)]
        self.step_calls += 1
        self.executed_actions.append(action)
        out = entry.model_copy(deep=True)
        out.observation.last_action = action
        return out

    def close(self) -> None:
        self.close_calls += 1

    def action_contract(self) -> ActionContract:
        return FAKE_CONTRACT.model_copy(deep=True)


def make_fake_observation(goal: str = "fake goal", url: str = "http://fake.local") -> Observation:
    return ObservationNormalizer().from_axtree_parts(
        goal=goal,
        url=url,
        axtree_text="[1] button 'Go'",
        open_pages=[url],
    )
