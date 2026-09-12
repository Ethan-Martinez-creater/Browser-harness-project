"""RecoveryManager: executes RecoveryDirectives (Phase 1C).

Only WAIT_AND_REOBSERVE performs a Harness-owned environment action
(`noop(wait_ms=...)`); it is NOT an agent step, does not produce a
StepRecord, and its real environment result (reward/termination/error) is
never discarded. REDECIDE_WITH_FEEDBACK and BLOCK_REPEATED_ACTION only
install a directive on the ReliabilityState to shape the next agent
decision — they never touch the environment.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel

from web_harness.core.errors import ErrorType
from web_harness.core.models import EnvironmentStep, Observation, TaskSpec
from web_harness.core.reliability import (
    RecoveryDirective,
    RecoveryKind,
    ReliabilityState,
)
from web_harness.env.base import EnvironmentAdapter


class RecoveryResult(BaseModel):
    success: bool
    observation: Observation
    directive: RecoveryDirective
    environment_step: EnvironmentStep | None = None
    # WAIT_AND_REOBSERVE accounting (extra environment operation, not a step)
    environment_actions: int = 0
    latency_s: float = 0.0
    terminal: bool = False
    error_type: ErrorType | None = None
    error_message: str | None = None


class RecoveryManager:
    def __init__(self, *, event_sink: Callable[[dict], None] | None = None):
        self.event_sink = event_sink or (lambda event: None)

    def _emit(self, *, step_index, outcome: str, data: dict) -> None:
        self.event_sink(
            {
                "event_type": "recovery",
                "step_index": step_index,
                "component": "recovery_manager",
                "outcome": outcome,
                "data": data,
            }
        )

    def recover(
        self,
        *,
        directive: RecoveryDirective,
        task: TaskSpec,
        observation: Observation,
        failed_action: str | None,
        env: EnvironmentAdapter,
        reliability_state: ReliabilityState,
        step_index: int,
    ) -> RecoveryResult:
        started = time.monotonic()
        reliability_state.recovery_count += 1
        reliability_state.active_recovery_directive = directive

        base_event = {
            "failure_kind": directive.failure_kind.value
            if directive.failure_kind
            else None,
            "failure_signature": directive.failure_signature,
            "directive_kind": directive.kind.value,
            "reason": directive.reason,
            "blocked_actions": list(directive.blocked_actions),
            "expires_after_agent_steps": directive.expires_after_agent_steps,
        }

        if directive.kind in (
            RecoveryKind.REDECIDE_WITH_FEEDBACK, RecoveryKind.BLOCK_REPEATED_ACTION,
        ):
            # no environment action: shape the next decision only
            self._emit(
                step_index=step_index,
                outcome=directive.kind.value,
                data={**base_event, "blocked_action": (
                    directive.blocked_actions[0]
                    if directive.blocked_actions else None
                )},
            )
            return RecoveryResult(
                success=True,
                observation=observation,
                directive=directive,
                latency_s=time.monotonic() - started,
            )

        if directive.kind == RecoveryKind.WAIT_AND_REOBSERVE:
            # wait_ms=0 is a legal configuration and must be executed as 0;
            # only an unset directive falls back to the default (R1)
            wait_ms = 500 if directive.wait_ms is None else directive.wait_ms
            action = f"noop(wait_ms={wait_ms})"
            env_step = env.step(action)
            latency = time.monotonic() - started
            self._emit(
                step_index=step_index,
                outcome=directive.kind.value,
                data={
                    **base_event,
                    "wait_ms": wait_ms,
                    "environment_action": action,
                    "action_error": env_step.action_error,
                    "reward": env_step.reward,
                    "terminated": env_step.terminated,
                    "truncated": env_step.truncated,
                },
            )
            terminal = env_step.terminated or env_step.truncated
            return RecoveryResult(
                success=not env_step.action_error,
                observation=env_step.observation,
                directive=directive,
                environment_step=env_step,
                environment_actions=1,
                latency_s=latency,
                terminal=terminal,
                error_type=(
                    ErrorType.ACTION_EXECUTION_ERROR if env_step.action_error else None
                ),
                error_message=env_step.action_error,
            )

        return RecoveryResult(
            success=False,
            observation=observation,
            directive=directive,
            latency_s=time.monotonic() - started,
            error_type=ErrorType.UNKNOWN_ERROR,
            error_message=f"unknown recovery kind: {directive.kind}",
        )
