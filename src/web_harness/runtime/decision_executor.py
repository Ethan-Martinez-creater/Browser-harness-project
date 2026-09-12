"""DecisionExecutor: one agent step, possibly several model attempts.

Phase 1B introduces controlled retry for model-side failures that have NOT
touched the environment:

    MODEL_API_ERROR          -> FailureKind.MODEL_API_TRANSIENT  (retryable)
    MODEL_OUTPUT_PARSE_ERROR -> FailureKind.MODEL_OUTPUT_INVALID (repairable)

Hard guarantees:
- the executor never performs an environment action; browser actions are
  never retried blindly;
- `max_retries = N` means 1 initial attempt + N retries;
- every retry decision, its attempt_index and its outcome are emitted as
  RuntimeEventType.RETRY events through the provided event sink;
- episode-level extra-model-call budget is enforced (BUDGET_EXCEEDED);
  per-call limits produce RETRY_EXHAUSTED;
- tokens of failed attempts (parse failures carry usage on the exception)
  are aggregated and never recorded as zero;
- a failed decision cycle performs zero environment steps (the runner
  terminates the episode without env.step()).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from web_harness.agents.base import Agent, AgentTurn
from web_harness.agents.prompt import REPAIR_FEEDBACK
from web_harness.core.errors import (
    ErrorType,
    HarnessError,
    ModelApiError,
    ModelOutputParseError,
)
from web_harness.core.models import (
    ModelOutput,
    Observation,
    PromptBundle,
    StepRecord,
    TaskSpec,
)
from web_harness.core.reliability import (
    FailureKind,
    FailureSeverity,
    FailureSignal,
    ReliabilityBudget,
    ReliabilityState,
)
from web_harness.env.action_contract import ActionContract
from web_harness.reliability.fingerprint import normalize_error_signature
from web_harness.reliability.retry import RetryPolicy


class DecisionExecutionResult(BaseModel):
    success: bool
    turn: AgentTurn | None = None
    prompt: PromptBundle | None = None

    attempts: int = 0
    retry_count: int = 0
    api_retry_count: int = 0
    parse_retry_count: int = 0

    terminal_error_type: ErrorType | None = None
    terminal_error_message: str | None = None

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    retry_input_tokens: int = 0
    retry_output_tokens: int = 0
    retry_latency_s: float = 0.0

    retry_success: bool = False
    retry_exhausted: bool = False
    budget_exhausted: bool = False


class DecisionExecutor:
    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        budget: ReliabilityBudget | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        # None policy = retry disabled: exactly one call, failures are terminal
        self.retry_policy = retry_policy
        self.budget = budget or ReliabilityBudget()
        self.sleep = sleep

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _api_signal(exc: ModelApiError) -> FailureSignal:
        return FailureSignal(
            kind=FailureKind.MODEL_API_TRANSIENT,
            severity=FailureSeverity.ERROR,
            source="decision_executor",
            signature=f"model_api:{normalize_error_signature(exc.message, limit=80)}",
            retryable=True,
            recoverable=False,
            evidence={"error": exc.message[:300]},
        )

    @staticmethod
    def _parse_signal(exc: ModelOutputParseError) -> FailureSignal:
        return FailureSignal(
            kind=FailureKind.MODEL_OUTPUT_INVALID,
            severity=FailureSeverity.ERROR,
            source="decision_executor",
            signature="model_output_invalid:json_action_parse",
            retryable=True,
            recoverable=False,
            evidence={"raw_text_chars": len(exc.raw_text or "")},
        )

    def _record_usage(
        self,
        result: DecisionExecutionResult,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        is_retry: bool,
        started: float,
    ) -> None:
        result.total_input_tokens += input_tokens or 0
        result.total_output_tokens += output_tokens or 0
        if is_retry:
            result.retry_input_tokens += input_tokens or 0
            result.retry_output_tokens += output_tokens or 0
            result.retry_latency_s += time.monotonic() - started

    def execute(
        self,
        *,
        agent: Agent,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        action_contract: ActionContract,
        reliability_state: ReliabilityState,
        event_sink: Callable[[dict], None] | None = None,
        attempt_sink: Callable[..., str | None] | None = None,
        step_index: int | None = None,
    ) -> DecisionExecutionResult:
        """Run one decision cycle: initial attempt + controlled model-side
        retries. Emits retry events when a sink is provided; persists failed
        parse attempts through `attempt_sink` when given; updates the
        reliability state (retry_count, extra_model_calls).

        Retry limits are counted PER FAILURE KIND (api_retry_count and
        parse_retry_count are independent); the episode-level extra-model-call
        budget is global across kinds.
        """

        result = DecisionExecutionResult(success=False)
        emit = event_sink or (lambda event: None)

        def emit_retry_event(*, attempt_index: int, outcome: str, **data) -> None:
            emit(
                {
                    "event_type": "retry",
                    "step_index": step_index,
                    "attempt_index": attempt_index,
                    "component": "decision_executor",
                    "outcome": outcome,
                    "data": data,
                }
            )

        # per-kind retry counters inside this decision cycle (B1)
        api_retry_count = 0
        parse_retry_count = 0
        repair_feedback: str | None = None
        attempt_index = 0  # 0 = initial attempt
        while True:
            started = time.monotonic()
            try:
                turn, prompt = agent.decide(
                    task=task,
                    observation=observation,
                    history=history,
                    action_contract=action_contract,
                    repair_feedback=repair_feedback,
                )
            except ModelApiError as exc:
                result.attempts += 1
                self._record_usage(
                    result, input_tokens=None, output_tokens=None,
                    is_retry=attempt_index > 0, started=started,
                )
                # non-transient API errors (auth/permission/bad request) are
                # terminal before any retry accounting: a retry cannot fix
                # them and must not consume budget
                if not getattr(exc, "transient", True):
                    decision = RetryDecisionShim(
                        retry=False, reason="non-transient api error"
                    )
                else:
                    decision = self._retry_decision(
                        result, self._api_signal(exc), api_retry_count,
                        reliability_state,
                    )
                if decision.retry:
                    api_retry_count += 1
                    emit_retry_event(
                        attempt_index=attempt_index + 1,
                        outcome="retry",
                        reason="MODEL_API_ERROR",
                        backoff_ms=decision.backoff_ms,
                    )
                    # backoff is part of the retry overhead (R2)
                    result.retry_latency_s += decision.backoff_ms / 1000.0
                    self.sleep(decision.backoff_ms / 1000.0)
                    repair_feedback = None
                    attempt_index += 1
                    continue
                self._emit_terminal_event(
                    emit_retry_event, attempt_index, decision, exc
                )
                return self._terminal(
                    result,
                    exc,
                    _terminal_error_type(exc, decision.reason),
                )
            except ModelOutputParseError as exc:
                result.attempts += 1
                # parse failures cost real tokens: account them (never zero)
                self._record_usage(
                    result,
                    input_tokens=exc.input_tokens,
                    output_tokens=exc.output_tokens,
                    is_retry=attempt_index > 0,
                    started=started,
                )
                # keep the failed attempt auditable (R4): artifact via sink
                artifact_ref = None
                if attempt_sink is not None:
                    artifact_ref = attempt_sink(
                        step_index=step_index,
                        attempt_index=attempt_index,
                        failure_type="MODEL_OUTPUT_PARSE_ERROR",
                        raw_text=exc.raw_text,
                        input_tokens=exc.input_tokens,
                        output_tokens=exc.output_tokens,
                        model_name=exc.model_name,
                    )
                decision = self._retry_decision(
                    result, self._parse_signal(exc), parse_retry_count,
                    reliability_state,
                )
                if decision.retry:
                    parse_retry_count += 1
                    emit_retry_event(
                        attempt_index=attempt_index + 1,
                        outcome="retry",
                        reason="MODEL_OUTPUT_PARSE_ERROR",
                        backoff_ms=0,
                        artifact_ref=artifact_ref,
                    )
                    repair_feedback = REPAIR_FEEDBACK
                    attempt_index += 1
                    continue
                self._emit_terminal_event(
                    emit_retry_event, attempt_index, decision, exc
                )
                return self._terminal(
                    result,
                    exc,
                    _terminal_error_type(exc, decision.reason),
                )
            except HarnessError as exc:
                result.attempts += 1
                return self._terminal(result, exc, exc.error_type)

            result.attempts += 1
            mo: ModelOutput = turn.model_output
            self._record_usage(
                result,
                input_tokens=mo.input_tokens if mo else None,
                output_tokens=mo.output_tokens if mo else None,
                is_retry=attempt_index > 0,
                started=started,
            )
            result.turn = turn
            result.prompt = prompt
            result.success = True
            if attempt_index > 0:
                result.retry_count = attempt_index
                result.retry_success = True
                result.api_retry_count = api_retry_count
                result.parse_retry_count = parse_retry_count
                emit_retry_event(
                    attempt_index=attempt_index,
                    outcome="retry_succeeded",
                    reason="decision produced a valid ActionDecision",
                    api_retries=api_retry_count,
                    parse_retries=parse_retry_count,
                )
            return result

    def _retry_decision(
        self,
        result: DecisionExecutionResult,
        signal: FailureSignal,
        retry_index_for_kind: int,
        reliability_state: ReliabilityState,
    ):
        """Consult the policy with the PER-KIND retry count; return its
        decision, updating state when the failure is terminal."""
        if self.retry_policy is None:
            # retry disabled: model-side failures stay terminal (Phase 0 path)
            return RetryDecisionShim(retry=False, reason="retry_disabled")
        decision = self.retry_policy.decide(
            failure_kind=signal.kind,
            retry_index_for_kind=retry_index_for_kind,
            reliability_state=reliability_state,
            budget=self.budget,
        )
        if decision.retry:
            # a retry is one extra model call (global across kinds)
            reliability_state.extra_model_calls += 1
            reliability_state.retry_count += 1
            result.retry_count += 1
        return decision

    @staticmethod
    def _emit_terminal_event(emit_retry_event, attempt_index, decision, exc) -> None:
        """Exhaustion must always be visible in the event stream."""
        if attempt_index == 0:
            return  # no retry happened: no retry events at all
        if decision.reason in ("retry_exhausted",) or "budget" in decision.reason:
            emit_retry_event(
                attempt_index=attempt_index,
                outcome="exhausted",
                reason=decision.reason,
                error=type(exc).__name__,
            )

    @staticmethod
    def _terminal(
        result: DecisionExecutionResult,
        exc: HarnessError,
        error_type: ErrorType,
    ) -> DecisionExecutionResult:
        result.terminal_error_type = error_type
        result.terminal_error_message = exc.message
        if error_type == ErrorType.RETRY_EXHAUSTED:
            result.retry_exhausted = True
        elif error_type == ErrorType.BUDGET_EXCEEDED:
            result.budget_exhausted = True
        if result.retry_count > 0:
            result.retry_success = False
        return result


class RetryDecisionShim(BaseModel):
    """Minimal decision returned when retry is disabled."""

    retry: bool
    reason: str
    backoff_ms: int = Field(default=0)


def _terminal_error_type(exc: HarnessError, decision_reason: str) -> ErrorType:
    """Map a terminal decision failure to its structured error type.

    Retry disabled (Phase 0 path): the original model-side error type is
    preserved. Policy exhausted: RETRY_EXHAUSTED. Episode budget exhausted:
    BUDGET_EXCEEDED.
    """
    if decision_reason == "retry_exhausted":
        return ErrorType.RETRY_EXHAUSTED
    if decision_reason == "episode extra-model-call budget exhausted":
        return ErrorType.BUDGET_EXCEEDED
    return exc.error_type
