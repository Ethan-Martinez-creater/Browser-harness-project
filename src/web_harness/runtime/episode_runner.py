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
    from web_harness.reliability.verifier import StepVerifier

from web_harness.agents.base import Agent
from web_harness.core.errors import ErrorType, HarnessError
from web_harness.core.events import RuntimeEvent, RuntimeEventType, new_event_id
from web_harness.core.ids import new_run_id, now_utc_iso
from web_harness.core.models import RunResult, RunStatus, StepRecord, TaskSpec
from web_harness.env.base import EnvironmentAdapter
from web_harness.observability.trace import TraceRecorder
from web_harness.runtime.decision_executor import DecisionExecutor
from web_harness.runtime.state import RunState

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

            def attempt_sink(**kwargs) -> str | None:
                # failed model attempts become trace artifacts (R4); ref is
                # attached to the matching retry event
                if not self.save_model_responses:
                    return None
                return recorder.write_failed_attempt(**kwargs)

            # -- explicit harness loop ------------------------------------
            for step_idx in range(task.max_steps):
                t0 = time.monotonic()

                # decide (on the pre-action observation); the executor wraps
                # the model call with controlled retries for model-side
                # failures only — it never performs environment actions
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
                )
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

                # terminate?
                if env_step.terminated:
                    status = RunStatus.SUCCESS if env_step.reward > 0 else RunStatus.FAILED
                    error_type = ErrorType.TASK_TERMINATED
                    break
                if env_step.truncated:
                    status = RunStatus.TRUNCATED
                    error_type = ErrorType.TASK_TRUNCATED
                    break
            else:
                status = RunStatus.MAX_STEPS_REACHED
                error_type = ErrorType.MAX_STEPS_EXCEEDED

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
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
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
        )
        recorder.write_result(result)
        return result
