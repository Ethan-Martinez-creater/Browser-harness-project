"""Fault injection for controlled reliability testing.

`FaultInjectingModelAdapter` wraps a real/mock ModelAdapter and injects
deterministic model-side failures by call index. It is for tests and
evaluation only — production configs never enable it — and it never touches
the environment path (environment fault injection belongs to Phase 1C).
"""

from __future__ import annotations

from collections.abc import Callable

from web_harness.core.errors import ModelApiError, ModelOutputParseError
from web_harness.core.models import (
    EnvironmentStep,
    ModelOutput,
    Observation,
    PromptBundle,
    StepRecord,
    TaskSpec,
)
from web_harness.env.action_contract import ActionContract
from web_harness.models.base import ModelAdapter


class FaultInjectingModelAdapter:
    """Deterministic model-side fault wrapper.

    Call indices are 0-based over ALL generate_action calls handled by this
    instance. Injected failures raise before delegating; non-injected calls
    are forwarded to the wrapped adapter unchanged. To emulate a provider
    that consumed tokens before failing, parse-error injections carry usage
    (PARSE_FAIL_INPUT_TOKENS / PARSE_FAIL_OUTPUT_TOKENS).
    """

    PARSE_FAIL_INPUT_TOKENS = 111
    PARSE_FAIL_OUTPUT_TOKENS = 22

    def __init__(
        self,
        wrapped: ModelAdapter,
        *,
        api_error_on_calls: set[int] | None = None,
        parse_error_on_calls: set[int] | None = None,
        parse_error_text: str = 'sorry, I cannot comply. no json here',
        on_call: Callable[[int], None] | None = None,
    ):
        self.wrapped = wrapped
        self.api_error_on_calls = api_error_on_calls or set()
        self.parse_error_on_calls = parse_error_on_calls or set()
        self.parse_error_text = parse_error_text
        self.on_call = on_call
        self.call_count = 0

    def generate_action(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        prompt: PromptBundle,
    ) -> ModelOutput:
        call_index = self.call_count
        self.call_count += 1
        if self.on_call:
            self.on_call(call_index)
        if call_index in self.api_error_on_calls:
            raise ModelApiError(
                f"injected MODEL_API_ERROR on call {call_index}"
            )
        if call_index in self.parse_error_on_calls:
            raise ModelOutputParseError(
                f"injected MODEL_OUTPUT_PARSE_ERROR on call {call_index}",
                raw_text=self.parse_error_text,
                input_tokens=self.PARSE_FAIL_INPUT_TOKENS,
                output_tokens=self.PARSE_FAIL_OUTPUT_TOKENS,
                model_name="fault-injected",
            )
        return self.wrapped.generate_action(
            task=task, observation=observation, history=history, prompt=prompt
        )


class FaultInjectingEnvironmentAdapter:
    """Environment-side fault wrapper (Phase 1C). See module docstring."""

    def __init__(
        self,
        wrapped,
        *,
        action_error_on_steps: set[int] | None = None,
        empty_observation_on_steps: set[int] | None = None,
        stale_observation_on_steps: set[int] | None = None,
        action_error_text: str = "TimeoutError: injected environment fault",
    ):
        self.wrapped = wrapped
        self.action_error_on_steps = action_error_on_steps or set()
        self.empty_observation_on_steps = empty_observation_on_steps or set()
        self.stale_observation_on_steps = stale_observation_on_steps or set()
        self.action_error_text = action_error_text
        self.step_count = 0
        self.last_observation: Observation | None = None

    def reset(self, task: TaskSpec) -> Observation:
        obs = self.wrapped.reset(task)
        self.last_observation = obs
        self.step_count = 0
        return obs

    def close(self) -> None:
        self.wrapped.close()

    def action_contract(self) -> ActionContract:
        return self.wrapped.action_contract()

    @property
    def bootstrap_action_executed(self):
        return getattr(self.wrapped, "bootstrap_action_executed", None)

    def step(self, action: str) -> EnvironmentStep:
        step_index = self.step_count
        self.step_count += 1
        env_step = self.wrapped.step(action)

        if step_index in self.action_error_on_steps:
            env_step.action_error = self.action_error_text
            # state unchanged: the failed action had no effect
            if self.last_observation is not None:
                env_step.observation = self.last_observation.model_copy(deep=True)
                env_step.observation.last_action = action
                env_step.observation.last_action_error = self.action_error_text

        if step_index in self.empty_observation_on_steps:
            env_step.observation = Observation(goal=env_step.observation.goal, url="")

        if (
            step_index in self.stale_observation_on_steps
            and self.last_observation is not None
        ):
            stale = self.last_observation.model_copy(deep=True)
            stale.last_action = action
            env_step.observation = stale

        self.last_observation = env_step.observation
        return env_step
