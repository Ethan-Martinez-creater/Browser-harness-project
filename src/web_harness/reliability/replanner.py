"""Replanner: exception-path short-horizon plan generation (Phase 1D).

The Replanner is NOT an always-on planner. It is invoked only when the
deterministic ReplanTriggerPolicy escalates an intended Phase 1C RECOVER,
and it produces a bounded RecoveryPlan that is injected as CONTEXT into the
next agent decisions — it never returns or executes browser actions.

Model access is provider-neutral: the ReplanExecutor talks to a
StructuredModelAdapter (generate_structured) and never imports a provider
SDK. Replan model calls reuse the Phase 1B protections (RetryPolicy,
ReliabilityBudget, attempt artifacts, token/latency accounting) instead of
inventing a second retry system.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable

from pydantic import BaseModel

from web_harness.agents.prompt import OUTPUT_FORMAT  # noqa: F401  (doc anchor)
from web_harness.core.errors import (
    ErrorType,
    ModelApiError,
    ModelOutputParseError,
)
from web_harness.core.events import RuntimeEvent
from web_harness.core.models import (
    ModelOutput,
    Observation,
    PromptBundle,
    StepRecord,
    TaskSpec,
)
from web_harness.core.reliability import (
    FailureKind,
    FailureSignal,
    RecoveryPlan,
    ReliabilityBudget,
    ReliabilityState,
)
from web_harness.env.action_contract import ActionContract
from web_harness.reliability.retry import RetryPolicy

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_REPLAN_SYSTEM = (
    "You are a reliability assistant inside a browser automation harness. "
    "You are NOT completing the task yourself. You only produce a SHORT "
    "term RecoveryPlan describing how the reactive agent should escape a "
    "persistent failure. Respond with the JSON object only."
)

_OUTPUT_CONTRACT = (
    "Respond with ONE strict JSON object, no extra text:\n"
    "{\n"
    '  "diagnosis": "<one short sentence: why the failure persists>",\n'
    '  "immediate_subgoal": "<the next concrete intermediate goal>",\n'
    '  "strategy_steps": ["<step 1>", "<step 2>", "<max 4 steps>"],\n'
    '  "avoid_actions": ["<action string to avoid, if any>"],\n'
    '  "horizon_steps": <small integer, <= max horizon>\n'
    "}\n"
    "No hidden reasoning, no chain-of-thought, no browser action sequences: "
    "the reactive agent chooses its own actions from these hints."
)


def build_replan_prompt(
    *,
    task: TaskSpec,
    observation: Observation,
    history: list[StepRecord],
    failure_signals: list[FailureSignal],
    recent_recovery_events: list[RuntimeEvent],
    action_contract: ActionContract,
    recent_steps: int,
    horizon_steps: int,
) -> PromptBundle:
    """Bounded replanner prompt: only the task goal, the current observation,
    the most recent agent steps, the current failure evidence, a summary of
    recent recovery outcomes and the action contract summary."""
    recent = history[-recent_steps:] if recent_steps > 0 else []
    recent_lines = [
        f"{s.step_index}: action={s.action!r} "
        f"error={s.action_error or 'none'} "
        f"reward={s.reward:g} kinds={','.join(s.failure_kinds) or 'none'}"
        for s in recent
    ] or ["(no previous steps)"]

    signal_lines = [
        f"- kind={s.kind.value} signature={s.signature} severity={s.severity.value}"
        for s in failure_signals
    ] or ["- (none)"]

    recovery_lines = [
        f"- step={e.step_index} outcome={e.outcome} "
        f"kind={e.data.get('failure_kind') or e.data.get('directive_kind') or 'n/a'}"
        for e in recent_recovery_events
    ] or ["- (none yet)"]

    contract_lines = [f"- {a.name}{a.signature}" for a in action_contract.actions]

    user_parts = [
        f"# Task goal\n{task.task_id}: {observation.goal or task.task_id}",
        f"# Current state\nURL: {observation.url or '(unknown)'}\n"
        f"{(observation.axtree or '(accessibility tree unavailable)')[:4000]}",
        "# Recent failed execution\n" + "\n".join(recent_lines),
        "# Failure evidence\n" + "\n".join(signal_lines),
        "# Recent recovery outcomes\n" + "\n".join(recovery_lines),
        "# Available browser action types\n" + "\n".join(contract_lines),
        f"# Output schema\nPlan horizon must be <= {horizon_steps} agent steps.\n"
        + _OUTPUT_CONTRACT,
    ]
    return PromptBundle(system=_REPLAN_SYSTEM, user="\n\n".join(user_parts))


class ReplanGenerationResult(BaseModel):
    """Outcome of one replan generation intervention (model calls included)."""

    success: bool
    plan: RecoveryPlan | None = None

    error_type: ErrorType | None = None
    error_message: str | None = None

    # replan_model_calls = real model calls for this intervention (incl.
    # API/parse retries) — replan_count counts interventions, not calls
    model_calls: int = 0
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    api_retry_count: int = 0
    parse_retry_count: int = 0


class ReplanExecutor:
    """One replan generation intervention with Phase 1B protections.

    Initial call + transient API retries (fixed backoff) + one structured
    JSON repair retry, all through the shared RetryPolicy and the global
    episode extra-model-call budget. Failed parse attempts are persisted
    through `attempt_sink` exactly like decision-cycle attempts.
    """

    def __init__(
        self,
        *,
        model_adapter,
        retry_policy: RetryPolicy | None = None,
        budget: ReliabilityBudget | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model_adapter = model_adapter
        self.retry_policy = retry_policy
        self.budget = budget or ReliabilityBudget()
        self.sleep = sleep

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _parse_plan(raw_text: str, *, step_index: int) -> RecoveryPlan:
        candidates = [raw_text.strip()]
        candidates.extend(m.group(0) for m in _JSON_BLOCK.finditer(raw_text))
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            try:
                return RecoveryPlan(**{
                    "diagnosis": str(data.get("diagnosis", ""))[:500],
                    "immediate_subgoal": str(data.get("immediate_subgoal", ""))[:500],
                    "strategy_steps": [
                        str(step)[:300] for step in
                        (data.get("strategy_steps") or [])
                    ],
                    "avoid_actions": [
                        str(a)[:200] for a in (data.get("avoid_actions") or [])
                    ],
                    "horizon_steps": int(data.get("horizon_steps", 1)),
                })
            except (TypeError, ValueError):
                continue
        raise ModelOutputParseError(
            "replan output is not a valid RecoveryPlan JSON",
            raw_text=raw_text,
        )

    def _retry_decision(self, result: ReplanGenerationResult, kind, index, state):
        if self.retry_policy is None:
            from web_harness.runtime.decision_executor import RetryDecisionShim

            return RetryDecisionShim(retry=False, reason="retry_disabled")
        decision = self.retry_policy.decide(
            failure_kind=kind,
            retry_index_for_kind=index,
            reliability_state=state,
            budget=self.budget,
        )
        if decision.retry:
            state.extra_model_calls += 1
        return decision

    def generate(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        failure_signals: list[FailureSignal],
        recent_recovery_events: list[RuntimeEvent],
        action_contract: ActionContract,
        step_index: int | None,
        reliability_state: ReliabilityState,
        event_sink: Callable[[dict], None] | None = None,
        attempt_sink: Callable[..., str | None] | None = None,
    ) -> ReplanGenerationResult:
        from web_harness.runtime.decision_executor import RetryDecisionShim

        prompt = build_replan_prompt(
            task=task,
            observation=observation,
            history=history,
            failure_signals=failure_signals,
            recent_recovery_events=recent_recovery_events,
            action_contract=action_contract,
            recent_steps=self.budget.recent_steps,
            horizon_steps=self.budget.plan_horizon_steps,
        )

        result = ReplanGenerationResult(success=False)
        emit = event_sink or (lambda event: None)

        def emit_replan_event(*, outcome: str, **data) -> None:
            emit(
                {
                    "event_type": "replan",
                    "step_index": step_index,
                    "component": "replan_executor",
                    "outcome": outcome,
                    "data": data,
                }
            )

        api_retry_count = 0
        parse_retry_count = 0
        attempt_index = 0
        while True:
            started = time.monotonic()
            try:
                mo: ModelOutput = self.model_adapter.generate_structured(
                    prompt=prompt,
                )
            except ModelApiError as exc:
                result.attempts += 1
                result.latency_s += time.monotonic() - started
                if not getattr(exc, "transient", True):
                    decision = RetryDecisionShim(
                        retry=False, reason="non-transient api error"
                    )
                else:
                    decision = self._retry_decision(
                        result, FailureKind.MODEL_API_TRANSIENT,
                        api_retry_count, reliability_state,
                    )
                if decision.retry:
                    api_retry_count += 1
                    emit_replan_event(
                        outcome="retry",
                        attempt_index=attempt_index + 1,
                        reason="MODEL_API_ERROR",
                        backoff_ms=decision.backoff_ms,
                    )
                    result.latency_s += decision.backoff_ms / 1000.0
                    self.sleep(decision.backoff_ms / 1000.0)
                    attempt_index += 1
                    continue
                emit_replan_event(
                    outcome="exhausted",
                    reason=decision.reason,
                    error=type(exc).__name__,
                )
                result.error_type = self._terminal_type(decision.reason, exc)
                result.error_message = exc.message
                return result
            except ModelOutputParseError as exc:
                result.attempts += 1
                result.latency_s += time.monotonic() - started
                result.input_tokens += exc.input_tokens or 0
                result.output_tokens += exc.output_tokens or 0
                artifact_ref = None
                if attempt_sink is not None:
                    artifact_ref = attempt_sink(
                        step_index=step_index,
                        attempt_index=attempt_index,
                        failure_type="REPLAN_OUTPUT_PARSE_ERROR",
                        raw_text=exc.raw_text,
                        input_tokens=exc.input_tokens,
                        output_tokens=exc.output_tokens,
                        model_name=exc.model_name,
                    )
                decision = self._retry_decision(
                    result, FailureKind.MODEL_OUTPUT_INVALID,
                    parse_retry_count, reliability_state,
                )
                if decision.retry:
                    parse_retry_count += 1
                    emit_replan_event(
                        outcome="retry",
                        attempt_index=attempt_index + 1,
                        reason="REPLAN_OUTPUT_PARSE_ERROR",
                        backoff_ms=0,
                        artifact_ref=artifact_ref,
                    )
                    attempt_index += 1
                    continue
                emit_replan_event(
                    outcome="exhausted",
                    reason=decision.reason,
                    error=type(exc).__name__,
                )
                result.error_type = self._terminal_type(decision.reason, exc)
                result.error_message = exc.message
                return result

            result.attempts += 1
            result.latency_s += time.monotonic() - started
            result.input_tokens += mo.input_tokens or 0
            result.output_tokens += mo.output_tokens or 0
            try:
                plan = self._parse_plan(mo.decision.action, step_index=step_index)
            except ModelOutputParseError as exc:
                artifact_ref = None
                if attempt_sink is not None:
                    artifact_ref = attempt_sink(
                        step_index=step_index,
                        attempt_index=attempt_index,
                        failure_type="REPLAN_OUTPUT_PARSE_ERROR",
                        raw_text=exc.raw_text,
                        input_tokens=mo.input_tokens,
                        output_tokens=mo.output_tokens,
                        model_name=mo.model_name,
                    )
                decision = self._retry_decision(
                    result, FailureKind.MODEL_OUTPUT_INVALID,
                    parse_retry_count, reliability_state,
                )
                if decision.retry:
                    parse_retry_count += 1
                    emit_replan_event(
                        outcome="retry",
                        attempt_index=attempt_index + 1,
                        reason="REPLAN_OUTPUT_PARSE_ERROR",
                        backoff_ms=0,
                        artifact_ref=artifact_ref,
                    )
                    attempt_index += 1
                    continue
                emit_replan_event(
                    outcome="exhausted",
                    reason=decision.reason,
                    error="RecoveryPlanParseError",
                )
                result.error_type = self._terminal_type(decision.reason, exc)
                result.error_message = exc.message
                return result

            # clamp the plan horizon to the configured maximum
            plan.horizon_steps = max(1, min(
                plan.horizon_steps, self.budget.plan_horizon_steps
            ))
            plan.created_at_step = step_index or 0
            result.plan = plan
            result.success = True
            result.api_retry_count = api_retry_count
            result.parse_retry_count = parse_retry_count
            return result

    @staticmethod
    def _terminal_type(decision_reason: str, exc) -> ErrorType:
        if decision_reason == "retry_exhausted":
            return ErrorType.RETRY_EXHAUSTED
        if decision_reason == "episode extra-model-call budget exhausted":
            return ErrorType.BUDGET_EXCEEDED
        return exc.error_type
