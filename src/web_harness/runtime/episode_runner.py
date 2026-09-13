"""EpisodeRunner: the self-owned harness runtime loop.

This module is the heart of Phase 0. It implements, explicitly and without
any third-party agent framework:

    reset -> (observe -> decide -> act -> record -> update -> terminate?)*

Termination rules in Phase 0 (nothing more):
    1. environment termination (terminated/truncated)
    2. max_steps reached
    3. fatal exception (model API / parse / trace / unexpected)

Action-level errors from the environment do NOT terminate the episode; they
are recorded on the StepRecord and the next observation lets the model react.
The environment is always closed (finally), even on failures.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from web_harness.reliability.policy import FailurePolicyEngine
    from web_harness.reliability.replan_policy import ReplanTriggerPolicy
    from web_harness.reliability.verifier import StepVerifier

from web_harness.agents.base import Agent
from web_harness.core.errors import ErrorType, HarnessError
from web_harness.core.events import RuntimeEvent, RuntimeEventType, new_event_id
from web_harness.core.ids import new_run_id, now_utc_iso
from web_harness.core.models import RunResult, RunStatus, StepRecord, TaskSpec
from web_harness.core.reliability import FailureKind, ReliabilityBudget
from web_harness.env.base import EnvironmentAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.persistence.schema import TRACE_SCHEMA_VERSION
from web_harness.reliability.fingerprint import fingerprint_of
from web_harness.reliability.policy import PolicyAction
from web_harness.reliability.recovery import RecoveryManager
from web_harness.runtime.decision_executor import DecisionExecutor
from web_harness.runtime.state import RunState


def _priority_signature(signals) -> str | None:
    """Most important failure signature of a verification pass."""
    for signal in signals:
        if signal.severity.value == "error":
            return signal.signature
    return signals[0].signature if signals else None


def _verifier_spec(verifier) -> dict | None:
    """Effective verifier specification for manifest provenance.

    Verifiers that expose `spec()` record their full effective behavior;
    other implementations record identity only (offline semantic replay
    cannot rebuild them and must say so instead of guessing).
    """
    if verifier is None:
        return None
    spec_method = getattr(verifier, "spec", None)
    if spec_method is not None:
        return spec_method()
    return {"implementation": type(verifier).__name__}

logger = logging.getLogger(__name__)


def git_commit() -> str | None:
    """Best-effort current git SHA; None when unavailable (never fatal)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class EpisodeRunner:
    def __init__(
        self,
        *,
        agent: Agent,
        env: EnvironmentAdapter,
        trace_root: Path,
        save_prompts: bool = True,
        save_model_responses: bool = True,
        model_provider: str | None = None,
        model_name: str | None = None,
        manifest_extra: dict | None = None,
        verifier: StepVerifier | None = None,
        verification_mode: str = "shadow",
        decision_executor: DecisionExecutor | None = None,
        failure_policy: FailurePolicyEngine | None = None,
        recovery_budget: ReliabilityBudget | None = None,
        replan_trigger_policy: ReplanTriggerPolicy | None = None,
        replan_executor=None,
    ):
        self.agent = agent
        self.env = env
        self.trace_root = Path(trace_root)
        self.save_prompts = save_prompts
        self.save_model_responses = save_model_responses
        self.model_provider = model_provider
        self.model_name = model_name
        self.manifest_extra = manifest_extra or {}
        # Phase 1A: verifier runs in shadow mode only — it observes and
        # records, it never changes the control flow. None = fully off.
        self.verifier = verifier
        self.verification_mode = verification_mode
        # Phase 1B: the executor wraps agent.decide with controlled model-side
        # retries. Default executor has retry disabled → identical to Phase 0.
        self.decision_executor = decision_executor or DecisionExecutor()
        # Phase 1C: rule-based policy + RecoveryManager for environment-side
        # failures. None = recovery off (Phase 1B behavior unchanged).
        self.failure_policy = failure_policy
        self.recovery_budget = recovery_budget or ReliabilityBudget()
        # Phase 1D: deterministic escalation policy + bounded Replanner.
        # None = replanning off (Phase 1C behavior unchanged).
        self.replan_trigger_policy = replan_trigger_policy
        self.replan_executor = replan_executor

    # -- main entry ---------------------------------------------------------

    def run(self, task: TaskSpec, *, run_id: str | None = None) -> RunResult:
        run_id = run_id or new_run_id()
        run_dir = self.trace_root / run_id
        recorder = TraceRecorder(
            run_dir,
            save_prompts=self.save_prompts,
            save_model_responses=self.save_model_responses,
        )

        state = RunState(run_id=run_id, task=task)
        started = time.monotonic()
        started_at = now_utc_iso()
        status = RunStatus.ERROR
        error_type: ErrorType | None = None
        error_message: str | None = None
        final_reward = 0.0

        def write_manifest() -> None:
            bootstrap = getattr(self.env, "bootstrap_action_executed", None)
            recorder.write_manifest(
                {
                    "run_id": run_id,
                    "timestamp": now_utc_iso(),
                    "trace_schema_version": TRACE_SCHEMA_VERSION,
                    "git_commit": git_commit(),
                    "config_hash": self.manifest_extra.get("config_hash"),
                    "python_version": sys.version.split()[0],
                    "platform": platform.platform(),
                    "model_provider": self.model_provider,
                    "model_name": self.model_name,
                    "benchmark": task.benchmark,
                    "task_id": task.task_id,
                    "seed": task.seed,
                    "max_steps": task.max_steps,
                    "environment_bootstrap_action": bootstrap,
                    # recorded effective verifier behavior (Phase 2A1
                    # provenance) so offline semantic replay can rebuild the
                    # verifier the live run actually used
                    "verification_spec": _verifier_spec(self.verifier),
                    **{
                        k: v
                        for k, v in self.manifest_extra.items()
                        if k != "config_hash"
                    },
                }
            )

        try:
            try:
                observation = self.env.reset(task)
            except HarnessError as exc:
                write_manifest()
                error_type, error_message = exc.error_type, exc.message
                logger.error("env reset failed: %s", exc.message)
                return self._finish(
                    recorder, state, RunStatus.ERROR, error_type, error_message,
                    started, started_at, run_dir,
                )
            except Exception as exc:  # noqa: BLE001 - normalized, never escapes
                write_manifest()
                error_type, error_message = ErrorType.ENVIRONMENT_INIT_ERROR, str(exc)
                logger.exception("unexpected env reset failure")
                return self._finish(
                    recorder, state, RunStatus.ERROR, error_type, error_message,
                    started, started_at, run_dir,
                )
            write_manifest()
            state.current_observation = observation
            # the environment is the single source of truth for the action space
            action_contract = self.env.action_contract()

            # episode-level retry accounting (Phase 1B)
            episode_retry_count = 0
            episode_retry_cycle_count = 0
            episode_retry_success_count = 0
            episode_retry_exhausted_count = 0
            episode_extra_model_calls = 0
            episode_retry_input_tokens = 0
            episode_retry_output_tokens = 0
            episode_retry_latency_s = 0.0
            # Phase 1C recovery accounting
            episode_recovery_count = 0
            episode_recovery_success_count = 0
            episode_recovery_failed_count = 0
            episode_recovery_env_actions = 0
            episode_recovery_latency_s = 0.0
            episode_recovered = False  # recovery triggered AND episode success
            # blocked-action re-selections are counted separately from
            # recoveries (R3): no directive is created, env.step is skipped
            episode_blocked_action_redecisions = 0
            # Phase 1D replan accounting. Invariant: replan_success + failed
            # + unresolved == replan_count. replan_model_calls counts real
            # model calls per intervention (incl. retries) — replan_count
            # counts interventions.
            episode_replan_count = 0
            episode_replan_success_count = 0
            episode_replan_failed_count = 0
            episode_replan_unresolved_count = 0
            episode_replan_model_calls = 0
            episode_replan_input_tokens = 0
            episode_replan_output_tokens = 0
            episode_replan_latency_s = 0.0

            def emit_runtime_event(fields: dict) -> None:
                recorder.record_event(
                    RuntimeEvent(
                        event_id=new_event_id(),
                        run_id=run_id,
                        timestamp=now_utc_iso(),
                        event_type=RuntimeEventType(fields["event_type"]),
                        step_index=fields.get("step_index"),
                        attempt_index=fields.get("attempt_index"),
                        component=fields.get("component", "decision_executor"),
                        outcome=fields.get("outcome"),
                        data=fields.get("data", {}),
                    )
                )
                if fields["event_type"] == "recovery":
                    recovery_events_log.append(
                        RuntimeEvent(
                            event_id=new_event_id(),
                            run_id=run_id,
                            event_type=RuntimeEventType.RECOVERY,
                            step_index=fields.get("step_index"),
                            component=fields.get("component", "recovery_manager"),
                            outcome=fields.get("outcome"),
                            data=fields.get("data", {}),
                        )
                    )

            recovery_events_log: list[RuntimeEvent] = []

            def attempt_sink(**kwargs) -> str | None:
                # failed model attempts become trace artifacts (R4); ref is
                # attached to the matching retry event
                if not self.save_model_responses:
                    return None
                return recorder.write_failed_attempt(**kwargs)

            recovery_manager = (
                RecoveryManager(event_sink=emit_runtime_event)
                if self.failure_policy is not None
                else None
            )

            def normalize_action(action: str | None) -> str:
                return (action or "").strip()

            def action_is_blocked(action: str | None) -> bool:
                directive = state.reliability.active_recovery_directive
                if directive is None:
                    return False
                normalized = normalize_action(action)
                return any(
                    normalize_action(blocked) == normalized
                    for blocked in directive.blocked_actions
                )

            def expire_recovery_directive() -> None:
                directive = state.reliability.active_recovery_directive
                if directive is None:
                    return
                directive.expires_after_agent_steps -= 1
                if directive.expires_after_agent_steps <= 0:
                    state.reliability.active_recovery_directive = None

            def evaluate_pending_recoveries(
                *,
                current_fingerprint: str,
                task_success: bool,
                current_signatures: list[str],
            ) -> None:
                """Local recovery success evaluation (deterministic, bounded):
                within 2 agent steps after a recovery, moving away from the
                recovery-start fingerprint OR task success AND no repeat of
                the same failure signature -> success; otherwise -> failure.
                The recovery-start fingerprint (not the last action's own
                pre/post delta) is the reference so a recovery that itself
                restored page state is judged correctly (R1)."""
                nonlocal episode_recovery_success_count
                nonlocal episode_recovery_failed_count
                still_pending = []
                for pending in state.reliability.pending_recovery_evaluations:
                    pending["steps_observed"] += 1
                    signature = pending["signature"]
                    repeated = any(
                        s.signature == signature for s in current_signatures
                    )
                    state_changed = current_fingerprint != pending["pre_fingerprint"]
                    if (state_changed or task_success) and not repeated:
                        episode_recovery_success_count += 1
                        note_recovery_outcome(success=True, signature=signature)
                    elif pending["steps_observed"] >= 2 or repeated:
                        episode_recovery_failed_count += 1
                        note_recovery_outcome(success=False, signature=signature)
                    else:
                        still_pending.append(pending)
                state.reliability.pending_recovery_evaluations = still_pending

            def note_recovery_outcome(
                *, success: bool, signature: str | None
            ) -> None:
                """Recovery outcomes drive the replan streak (Phase 1D):
                FAILED +1, SUCCESS resets, UNRESOLVED never touches it."""
                if success:
                    state.reliability.consecutive_recovery_failures = 0
                    state.reliability.last_failed_recovery_signature = None
                else:
                    state.reliability.consecutive_recovery_failures += 1
                    state.reliability.last_failed_recovery_signature = signature

            def evaluate_pending_replan(
                *,
                current_fingerprint: str,
                task_success: bool,
                current_signatures: list[str],
            ) -> None:
                """Replan local outcome (deterministic, horizon-bounded):
                leaving the replan-start fingerprint OR task success AND no
                immediate repeat of the trigger signature -> SUCCESS; the
                trigger signature reappearing -> FAILED (plan deactivated);
                a horizon fully consumed without progress -> FAILED."""
                nonlocal episode_replan_success_count
                nonlocal episode_replan_failed_count
                entry = state.reliability.pending_replan_evaluation
                if entry is None:
                    return
                entry["steps_observed"] += 1
                repeated = any(
                    s.signature == entry["signature"] for s in current_signatures
                )
                state_changed = current_fingerprint != entry["pre_fingerprint"]
                if (state_changed or task_success) and not repeated:
                    episode_replan_success_count += 1
                    state.reliability.pending_replan_evaluation = None
                    # replan success resets the recovery failure streak
                    state.reliability.consecutive_recovery_failures = 0
                    state.reliability.last_failed_recovery_signature = None
                elif repeated:
                    episode_replan_failed_count += 1
                    state.reliability.pending_replan_evaluation = None
                    state.reliability.active_recovery_plan = None
                    state.reliability.remaining_plan_steps = None

            def consume_plan_horizon() -> None:
                """Only a real Agent StepRecord consumes plan horizon."""
                nonlocal episode_replan_failed_count
                if state.reliability.active_recovery_plan is None:
                    return
                state.reliability.remaining_plan_steps -= 1
                if state.reliability.remaining_plan_steps <= 0:
                    # horizon exhausted: an unresolved outcome means FAILED
                    if state.reliability.pending_replan_evaluation is not None:
                        episode_replan_failed_count += 1
                    state.reliability.active_recovery_plan = None
                    state.reliability.remaining_plan_steps = None
                    state.reliability.pending_replan_evaluation = None

            # -- explicit harness loop ------------------------------------
            for step_idx in range(task.max_steps):
                t0 = time.monotonic()
                recovery_budget_exhausted = False

                # decide (on the pre-action observation); the executor wraps
                # the model call with controlled retries for model-side
                # failures only — it never performs environment actions.
                # A blocked action from an active recovery directive can
                # NEVER reach env.step: re-decide within recovery budget.
                while True:
                    recovery_directive = state.reliability.active_recovery_directive
                    decision_result = self.decision_executor.execute(
                        agent=self.agent,
                        task=task,
                        observation=observation,
                        history=state.steps,
                        action_contract=action_contract,
                        reliability_state=state.reliability,
                        event_sink=emit_runtime_event,
                        attempt_sink=attempt_sink,
                        step_index=step_idx,
                        recovery_directive=recovery_directive,
                        recovery_plan=state.reliability.active_recovery_plan,
                    )
                    if not decision_result.success:
                        break
                    if action_is_blocked(decision_result.turn.decision.action):
                        # controlled recovery outcome: no env.step for the
                        # blocked action; one re-decision. This is NOT a new
                        # recovery (no directive is created/executed) — it is
                        # counted separately as a blocked-action re-decision
                        # and bounded by the same recovery budget as an
                        # anti-loop guard.
                        if (
                            state.reliability.recovery_count
                            >= self.recovery_budget.max_recoveries_per_episode
                            or episode_blocked_action_redecisions
                            >= self.recovery_budget.max_recoveries_per_episode
                        ):
                            recovery_budget_exhausted = True
                            break
                        episode_blocked_action_redecisions += 1
                        emit_runtime_event(
                            {
                                "event_type": "recovery",
                                "step_index": step_idx,
                                "component": "recovery_manager",
                                "outcome": "blocked_action_selected",
                                "data": {
                                    "blocked_action": decision_result.turn.decision.action,
                                    "reason": "agent selected a blocked action; "
                                    "re-deciding (counted separately from "
                                    "recovery_count)",
                                },
                            }
                        )
                        continue
                    break

                if recovery_budget_exhausted:
                    # a blocked action must never reach env.step; the guard
                    # is exhausted so re-deciding is no longer possible. The
                    # refusal to re-decide is not itself a recovery failure.
                    step = self._failed_decision_step(
                        state, step_idx, observation,
                        ErrorType.RECOVERY_FAILED,
                        HarnessError("blocked-action re-decision guard "
                                     "exhausted"),
                    )
                    recorder.record_step(step, observation=observation)
                    state.steps.append(step)
                    error_type = ErrorType.RECOVERY_FAILED
                    error_message = "recovery budget exhausted with a blocked " \
                        "action selected; refusing to execute it"
                    break

                episode_retry_count += decision_result.retry_count
                if decision_result.retry_count > 0:
                    episode_retry_cycle_count += 1
                episode_extra_model_calls += decision_result.retry_count
                episode_retry_input_tokens += decision_result.retry_input_tokens
                episode_retry_output_tokens += decision_result.retry_output_tokens
                episode_retry_latency_s += decision_result.retry_latency_s

                if not decision_result.success:
                    # terminal model-side failure: zero environment actions
                    mo_exc_input = decision_result.total_input_tokens
                    step = self._failed_decision_step(
                        state,
                        step_idx,
                        observation,
                        decision_result.terminal_error_type or ErrorType.UNKNOWN_ERROR,
                        HarnessError(
                            decision_result.terminal_error_message or "decision failed"
                        ),
                        attempts=decision_result.attempts,
                        retry_count=decision_result.retry_count,
                        input_tokens=mo_exc_input,
                        output_tokens=decision_result.total_output_tokens,
                        retry_input_tokens=decision_result.retry_input_tokens,
                        retry_output_tokens=decision_result.retry_output_tokens,
                        retry_latency_s=decision_result.retry_latency_s,
                        retry_exhausted=decision_result.retry_exhausted,
                        budget_exhausted=decision_result.budget_exhausted,
                    )
                    recorder.record_step(step, observation=observation)
                    state.steps.append(step)
                    if decision_result.retry_exhausted or decision_result.budget_exhausted:
                        episode_retry_exhausted_count += 1
                    error_type = decision_result.terminal_error_type
                    error_message = decision_result.terminal_error_message
                    break

                turn = decision_result.turn
                prompt = decision_result.prompt
                if decision_result.retry_success:
                    episode_retry_success_count += 1

                # act: any environment exception is normalized here so the
                # episode can be traced and the model can react next step
                try:
                    env_step = self.env.step(turn.decision.action)
                except Exception as exc:  # noqa: BLE001 - normalized on purpose
                    logger.warning("env step exception: %s", exc)
                    fallback = observation.model_copy(deep=True)
                    fallback.last_action = turn.decision.action
                    fallback.last_action_error = str(exc)
                    from web_harness.core.models import EnvironmentStep

                    env_step = EnvironmentStep(
                        observation=fallback, action_error=str(exc)
                    )
                step_latency_ms = (time.monotonic() - t0) * 1000.0

                # verify (shadow) BEFORE recording so the verification summary
                # lands inside the StepRecord: observe and record only — no
                # control flow changes, no extra model calls, no extra actions
                verification_summary = None
                failure_kinds: list[str] = []
                if self.verifier is not None:
                    verification = self.verifier.verify(
                        task=task,
                        pre_observation=observation,
                        action=turn.decision.action,
                        env_step=env_step,
                        history=state.steps,
                        reliability_state=state.reliability,
                    )
                    verification_summary = verification.status.value
                    failure_kinds = sorted({s.kind.value for s in verification.signals})

                # record: obs_N = what the model saw, next_obs_N = what the
                # action produced (step-level failure, not episode failure)
                mo = turn.model_output
                step = StepRecord(
                    run_id=run_id,
                    step_index=step_idx,
                    url=observation.url,
                    action=turn.decision.action,
                    action_error=env_step.action_error,
                    reward=env_step.reward,
                    terminated=env_step.terminated,
                    truncated=env_step.truncated,
                    latency_ms=step_latency_ms,
                    # token totals cover ALL model attempts of this step
                    input_tokens=decision_result.total_input_tokens or None,
                    output_tokens=decision_result.total_output_tokens or None,
                    model_name=mo.model_name if mo else None,
                    short_reason=turn.decision.short_reason,
                    error_type=(
                        ErrorType.ACTION_EXECUTION_ERROR
                        if env_step.action_error
                        else None
                    ),
                    attempts=decision_result.attempts,
                    retry_count=decision_result.retry_count,
                    retry_input_tokens=decision_result.retry_input_tokens,
                    retry_output_tokens=decision_result.retry_output_tokens,
                    retry_latency_s=decision_result.retry_latency_s,
                    retry_success=decision_result.retry_success,
                    verification_status=verification_summary,
                    failure_kinds=failure_kinds,
                )
                step = recorder.record_step(
                    step,
                    observation=observation,
                    next_observation=env_step.observation,
                    prompt=prompt,
                    model_output=mo,
                )

                # event stream: detailed verification evidence (events.jsonl)
                if verification_summary is not None:
                    recorder.record_event(
                        RuntimeEvent(
                            event_id=new_event_id(),
                            run_id=run_id,
                            event_type=RuntimeEventType.VERIFICATION,
                            step_index=step_idx,
                            component="verification_engine",
                            outcome=verification_summary,
                            data={
                                "mode": self.verification_mode,
                                "signals": [
                                    s.model_dump(mode="json")
                                    for s in verification.signals
                                ],
                                "pre_fingerprint": verification.pre_fingerprint,
                                "post_fingerprint": verification.post_fingerprint,
                                "state_changed": verification.state_changed,
                            },
                        )
                    )

                state.steps.append(step)
                state.current_observation = env_step.observation
                # the next decision must see what this action produced
                observation = env_step.observation
                final_reward = env_step.reward

                # evaluate pending local-recovery outcomes against this step
                if (
                    self.failure_policy is not None
                    and state.reliability.pending_recovery_evaluations
                ):
                    evaluate_pending_recoveries(
                        current_fingerprint=fingerprint_of(observation),
                        task_success=bool(
                            env_step.terminated and env_step.reward > 0
                        ),
                        current_signatures=verification.signals
                        if verification_summary is not None
                        else [],
                    )
                # an active directive governs exactly one agent step
                expire_recovery_directive()

                # Phase 1D: evaluate the active plan outcome, then consume
                # plan horizon (ONLY this real Agent StepRecord consumes it)
                if self.replan_trigger_policy is not None:
                    evaluate_pending_replan(
                        current_fingerprint=fingerprint_of(env_step.observation),
                        task_success=bool(
                            env_step.terminated and env_step.reward > 0
                        ),
                        current_signatures=verification.signals
                        if verification_summary is not None
                        else [],
                    )
                    consume_plan_horizon()

                # terminate? Environment terminal has the HIGHEST priority:
                # it short-circuits the policy entirely. TASK_FAILED is
                # therefore terminal-short-circuited by the runner before the
                # policy could decide — record the deterministic ABORT
                # decision as an event for a consistent trace, but never
                # enter recovery for a terminal environment state (R4).
                if env_step.terminated:
                    if (
                        self.failure_policy is not None
                        and verification_summary is not None
                        and any(
                            s.kind == FailureKind.TASK_FAILED
                            for s in verification.signals
                        )
                    ):
                        emit_runtime_event(
                            {
                                "event_type": "policy_decision",
                                "step_index": step_idx,
                                "component": "failure_policy_engine",
                                "outcome": "abort",
                                "data": {
                                    "reason": "TASK_FAILED is "
                                    "terminal-short-circuited by "
                                    "EpisodeRunner; environment terminal "
                                    "is final",
                                    "failure_kind": "TASK_FAILED",
                                    "short_circuited": True,
                                },
                            }
                        )
                    status = RunStatus.SUCCESS if env_step.reward > 0 else RunStatus.FAILED
                    error_type = ErrorType.TASK_TERMINATED
                    if status == RunStatus.SUCCESS and episode_recovery_count > 0:
                        episode_recovered = True
                    break
                if env_step.truncated:
                    status = RunStatus.TRUNCATED
                    error_type = ErrorType.TASK_TRUNCATED
                    break

                # policy: deterministic rules over verified failure signals;
                # CONTINUE keeps the loop untouched, RECOVER/ABORT only run
                # when the policy layer is configured (Phase 1C)
                if self.failure_policy is not None and verification_summary is not None:
                    policy_decision = self.failure_policy.decide(
                        signals=verification.signals,
                        failed_action=turn.decision.action,
                        reliability_state=state.reliability,
                        budget=self.recovery_budget,
                    )
                    emit_runtime_event(
                        {
                            "event_type": "policy_decision",
                            "step_index": step_idx,
                            "component": "failure_policy_engine",
                            "outcome": policy_decision.action.value,
                            "data": {
                                "reason": policy_decision.reason,
                                "failure_kind": policy_decision.failure_kind.value
                                if policy_decision.failure_kind
                                else None,
                            },
                        }
                    )
                    if policy_decision.action == PolicyAction.ABORT:
                        # refusing to START another recovery is not itself a
                        # recovery failure: no phantom outcome is added (B3)
                        error_type = ErrorType.RECOVERY_FAILED
                        error_message = policy_decision.reason
                        break
                    if policy_decision.action == PolicyAction.RECOVER:
                        # Phase 1D escalation check: deterministic, runs AFTER
                        # the base policy and only on its RECOVER path. When
                        # triggered, the Replanner produces a bounded
                        # RecoveryPlan and the current recovery is replaced;
                        # generation failure falls back to Phase 1C recovery.
                        escalated = False
                        if (
                            self.replan_trigger_policy is not None
                            and self.replan_executor is not None
                        ):
                            trigger_signature = _priority_signature(
                                verification.signals
                            )
                            escalation = self.replan_trigger_policy.decide(
                                base_action=policy_decision.action,
                                failure_kind=policy_decision.failure_kind,
                                failure_signature=trigger_signature,
                                reliability_state=state.reliability,
                                budget=self.recovery_budget,
                            )
                            if escalation.trigger:
                                escalated = True
                                episode_replan_count += 1
                                # keep the state in sync: the trigger policy
                                # reads replan_count from ReliabilityState
                                state.reliability.replan_count += 1
                                emit_runtime_event(
                                    {
                                        "event_type": "replan",
                                        "step_index": step_idx,
                                        "component": "replan_trigger_policy",
                                        "outcome": "triggered",
                                        "data": {
                                            "reason": escalation.reason.value
                                            if escalation.reason else None,
                                            "failure_kind":
                                            escalation.failure_kind.value
                                            if escalation.failure_kind else None,
                                            "failure_signature":
                                            escalation.failure_signature,
                                            "recovery_failure_streak":
                                            state.reliability
                                            .consecutive_recovery_failures,
                                            "replan_count":
                                            episode_replan_count,
                                        },
                                    }
                                )
                                gen = self.replan_executor.generate(
                                    task=task,
                                    observation=observation,
                                    history=state.steps,
                                    failure_signals=verification.signals,
                                    recent_recovery_events=recovery_events_log[
                                        -self.recovery_budget.recent_steps :
                                    ],
                                    action_contract=action_contract,
                                    step_index=step_idx,
                                    reliability_state=state.reliability,
                                    event_sink=emit_runtime_event,
                                    attempt_sink=attempt_sink,
                                )
                                episode_replan_model_calls += gen.attempts
                                episode_replan_input_tokens += gen.input_tokens
                                episode_replan_output_tokens += gen.output_tokens
                                episode_replan_latency_s += gen.latency_s
                                if gen.success:
                                    state.reliability.active_recovery_plan = (
                                        gen.plan
                                    )
                                    state.reliability.remaining_plan_steps = (
                                        gen.plan.horizon_steps
                                    )
                                    state.reliability.pending_replan_evaluation = {
                                        "signature": trigger_signature,
                                        "steps_observed": 0,
                                        "pre_fingerprint": fingerprint_of(
                                            observation
                                        ),
                                    }
                                    artifact_ref = recorder.write_replan_plan(
                                        replan_index=episode_replan_count,
                                        plan=gen.plan,
                                        trigger_reason=escalation.reason.value
                                        if escalation.reason else None,
                                        failure_kind=(
                                            escalation.failure_kind.value
                                            if escalation.failure_kind else None
                                        ),
                                        failure_signature=trigger_signature,
                                        recovery_failure_streak=state.reliability
                                        .consecutive_recovery_failures,
                                        model_calls=gen.attempts,
                                        input_tokens=gen.input_tokens or None,
                                        output_tokens=gen.output_tokens or None,
                                        latency_s=gen.latency_s,
                                    )
                                    emit_runtime_event(
                                        {
                                            "event_type": "replan",
                                            "step_index": step_idx,
                                            "component": "replan_executor",
                                            "outcome": "created",
                                            "data": {
                                                "artifact_ref": artifact_ref,
                                                "horizon_steps":
                                                gen.plan.horizon_steps,
                                                "immediate_subgoal":
                                                gen.plan.immediate_subgoal,
                                                "model_calls": gen.attempts,
                                                "input_tokens": gen.input_tokens,
                                                "output_tokens":
                                                gen.output_tokens,
                                                "latency_s": round(
                                                    gen.latency_s, 3
                                                ),
                                            },
                                        }
                                    )
                                else:
                                    episode_replan_failed_count += 1
                                    emit_runtime_event(
                                        {
                                            "event_type": "replan",
                                            "step_index": step_idx,
                                            "component": "replan_executor",
                                            "outcome": "generation_failed",
                                            "data": {
                                                "error_type":
                                                gen.error_type.value
                                                if gen.error_type else None,
                                                "error_message":
                                                gen.error_message,
                                                "model_calls": gen.attempts,
                                            },
                                        }
                                    )
                                    # fallback to the Phase 1C recovery below
                                    escalated = False
                        if not escalated:
                            recovery_result = recovery_manager.recover(
                                directive=policy_decision.directive,
                                task=task,
                                observation=observation,
                                failed_action=turn.decision.action,
                                env=self.env,
                                reliability_state=state.reliability,
                                step_index=step_idx,
                            )
                            episode_recovery_count += 1
                            episode_recovery_env_actions += recovery_result.environment_actions
                            episode_recovery_latency_s += recovery_result.latency_s
                            if recovery_result.terminal:
                                # the recovery noop itself ended the task: accept
                                # the environment's real terminal semantics and
                                # record exactly ONE local outcome for this
                                # recovery (terminal outcomes never enter the
                                # pending evaluation path — closure B1)
                                env_step = recovery_result.environment_step
                                final_reward = env_step.reward
                                if env_step.truncated:
                                    status = RunStatus.TRUNCATED
                                    error_type = ErrorType.TASK_TRUNCATED
                                    episode_recovery_failed_count += 1
                                    note_recovery_outcome(
                                        success=False,
                                        signature=_priority_signature(
                                            verification.signals
                                        ),
                                    )
                                elif env_step.terminated and env_step.reward > 0:
                                    status = RunStatus.SUCCESS
                                    error_type = ErrorType.TASK_TERMINATED
                                    episode_recovery_success_count += 1
                                    episode_recovered = True
                                    note_recovery_outcome(success=True, signature=None)
                                else:
                                    status = RunStatus.FAILED
                                    error_type = ErrorType.TASK_TERMINATED
                                    episode_recovery_failed_count += 1
                                    note_recovery_outcome(
                                        success=False,
                                        signature=_priority_signature(
                                            verification.signals
                                        ),
                                    )
                                break
                            if recovery_result.error_type:
                                # the recovery operation itself failed: exactly
                                # one immediate failed outcome, and the recovery
                                # must NOT also enter pending evaluation (the
                                # three outcome paths are mutually exclusive —
                                # closure B2)
                                episode_recovery_failed_count += 1
                                note_recovery_outcome(
                                    success=False,
                                    signature=_priority_signature(
                                        verification.signals
                                    ),
                                )
                            else:
                                # remember this recovery for local success
                                # evaluation (only if it had no immediate outcome)
                                state.reliability.pending_recovery_evaluations.append(
                                    {
                                        "signature": _priority_signature(
                                            verification.signals
                                        ),
                                        "steps_observed": 0,
                                        "pre_fingerprint": fingerprint_of(observation),
                                    }
                                )
                                state.reliability.last_recovery_failure_signature = (
                                    _priority_signature(verification.signals)
                                )
                            # the recovery observation drives the next decision
                            observation = recovery_result.observation
                            state.current_observation = observation
            else:
                status = RunStatus.MAX_STEPS_REACHED
                error_type = ErrorType.MAX_STEPS_EXCEEDED

            # finalize pending recovery outcomes: an episode that ends before
            # its 2-step evaluation window completes must not lose them
            # silently (B3) — they are counted as explicitly unresolved.
            episode_recovery_unresolved_count = len(
                state.reliability.pending_recovery_evaluations
            )
            state.reliability.pending_recovery_evaluations = []

            # a plan outcome that the episode outlived is explicitly
            # unresolved, never silently dropped (Phase 1D)
            if state.reliability.pending_replan_evaluation is not None:
                episode_replan_unresolved_count += 1
            state.reliability.pending_replan_evaluation = None
            state.reliability.active_recovery_plan = None
            state.reliability.remaining_plan_steps = None

            if error_type is None and status == RunStatus.ERROR:
                error_type = ErrorType.UNKNOWN_ERROR

            return self._finish(
                recorder, state, status, error_type, error_message,
                started, started_at, run_dir, final_reward=final_reward,
                retry_count=episode_retry_count,
                retry_cycle_count=episode_retry_cycle_count,
                retry_success_count=episode_retry_success_count,
                retry_exhausted_count=episode_retry_exhausted_count,
                extra_model_calls=episode_extra_model_calls,
                retry_input_tokens=episode_retry_input_tokens,
                retry_output_tokens=episode_retry_output_tokens,
                retry_latency_s=episode_retry_latency_s,
                recovery_count=episode_recovery_count,
                recovery_success_count=episode_recovery_success_count,
                recovery_failed_count=episode_recovery_failed_count,
                recovery_unresolved_count=episode_recovery_unresolved_count,
                recovered_episode=episode_recovered,
                recovery_environment_actions=episode_recovery_env_actions,
                recovery_latency_s=episode_recovery_latency_s,
                blocked_action_redecisions=episode_blocked_action_redecisions,
                replan_count=episode_replan_count,
                replan_success_count=episode_replan_success_count,
                replan_failed_count=episode_replan_failed_count,
                replan_unresolved_count=episode_replan_unresolved_count,
                replan_model_calls=episode_replan_model_calls,
                replan_input_tokens=episode_replan_input_tokens,
                replan_output_tokens=episode_replan_output_tokens,
                replan_latency_s=episode_replan_latency_s,
            )
        finally:
            # the environment must be closed on every path
            self.env.close()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _failed_decision_step(
        state: RunState,
        step_idx: int,
        observation,
        error_type: ErrorType,
        exc: HarnessError,
        *,
        attempts: int = 1,
        retry_count: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        retry_input_tokens: int = 0,
        retry_output_tokens: int = 0,
        retry_latency_s: float = 0.0,
        retry_exhausted: bool = False,
        budget_exhausted: bool = False,
    ) -> StepRecord:
        return StepRecord(
            run_id=state.run_id,
            step_index=step_idx,
            url=observation.url,
            action=None,
            error_type=error_type,
            attempts=attempts,
            retry_count=retry_count,
            input_tokens=input_tokens or None,
            output_tokens=output_tokens or None,
            retry_input_tokens=retry_input_tokens,
            retry_output_tokens=retry_output_tokens,
            retry_latency_s=retry_latency_s,
            retry_exhausted=retry_exhausted,
            budget_exhausted=budget_exhausted,
        )

    def _finish(
        self,
        recorder: TraceRecorder,
        state: RunState,
        status: RunStatus,
        error_type: ErrorType | None,
        error_message: str | None,
        started: float,
        started_at: str,
        run_dir: Path,
        final_reward: float = 0.0,
        recovery_count: int = 0,
        recovery_success_count: int = 0,
        recovery_failed_count: int = 0,
        recovery_unresolved_count: int = 0,
        recovered_episode: bool = False,
        recovery_environment_actions: int = 0,
        recovery_latency_s: float = 0.0,
        blocked_action_redecisions: int = 0,
        replan_count: int = 0,
        replan_success_count: int = 0,
        replan_failed_count: int = 0,
        replan_unresolved_count: int = 0,
        replan_model_calls: int = 0,
        replan_input_tokens: int = 0,
        replan_output_tokens: int = 0,
        replan_latency_s: float = 0.0,
        retry_count: int = 0,
        retry_cycle_count: int = 0,
        retry_success_count: int = 0,
        retry_exhausted_count: int = 0,
        extra_model_calls: int = 0,
        retry_input_tokens: int = 0,
        retry_output_tokens: int = 0,
        retry_latency_s: float = 0.0,
    ) -> RunResult:
        state.status = status
        duration = time.monotonic() - started
        kind_counts: dict[str, int] = {}
        for step in state.steps:
            for kind in step.failure_kinds:
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
        result = RunResult(
            run_id=state.run_id,
            task_spec=state.task,
            status=status,
            success=status == RunStatus.SUCCESS,
            final_reward=final_reward,
            num_steps=state.num_steps,
            started_at=started_at,
            ended_at=now_utc_iso(),
            duration_s=duration,
            # episode total tokens = agent decision-path tokens + Replanner
            # tokens (closure B2); the retry/replan breakdown stays separate
            input_tokens=state.input_tokens + replan_input_tokens,
            output_tokens=state.output_tokens + replan_output_tokens,
            action_error_count=state.action_error_count,
            trace_path=str(run_dir),
            error_type=error_type,
            error_message=error_message,
            verification_count=sum(
                1 for s in state.steps if s.verification_status is not None
            ),
            verifications_with_signal=sum(
                1
                for s in state.steps
                if s.verification_status not in (None, "pass")
            ),
            failure_signal_count=state.reliability.total_failure_signals,
            failure_kind_counts=kind_counts,
            retry_count=retry_count,
            retry_cycle_count=retry_cycle_count,
            retry_success_count=retry_success_count,
            retry_exhausted_count=retry_exhausted_count,
            extra_model_calls=extra_model_calls,
            retry_input_tokens=retry_input_tokens,
            retry_output_tokens=retry_output_tokens,
            retry_latency_s=retry_latency_s,
            recovery_count=recovery_count,
            recovery_success_count=recovery_success_count,
            recovery_failed_count=recovery_failed_count,
            recovery_unresolved_count=recovery_unresolved_count,
            recovered_episode=recovered_episode,
            recovery_environment_actions=recovery_environment_actions,
            recovery_latency_s=recovery_latency_s,
            blocked_action_redecision_count=blocked_action_redecisions,
            replan_count=replan_count,
            replan_success_count=replan_success_count,
            replan_failed_count=replan_failed_count,
            replan_unresolved_count=replan_unresolved_count,
            replan_model_calls=replan_model_calls,
            replan_input_tokens=replan_input_tokens,
            replan_output_tokens=replan_output_tokens,
            replan_latency_s=replan_latency_s,
        )
        recorder.write_result(result)
        return result
