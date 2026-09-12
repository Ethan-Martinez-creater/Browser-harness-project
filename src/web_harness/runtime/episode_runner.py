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

from web_harness.agents.base import Agent
from web_harness.core.errors import (
    ErrorType,
    HarnessError,
    ModelApiError,
    ModelOutputParseError,
)
from web_harness.core.ids import new_run_id, now_utc_iso
from web_harness.core.models import RunResult, RunStatus, StepRecord, TaskSpec
from web_harness.env.base import EnvironmentAdapter
from web_harness.observability.trace import TraceRecorder
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
    ):
        self.agent = agent
        self.env = env
        self.trace_root = Path(trace_root)
        self.save_prompts = save_prompts
        self.save_model_responses = save_model_responses
        self.model_provider = model_provider
        self.model_name = model_name
        self.manifest_extra = manifest_extra or {}

    # -- main entry ---------------------------------------------------------

    def run(self, task: TaskSpec, *, run_id: str | None = None) -> RunResult:
        run_id = run_id or new_run_id()
        run_dir = self.trace_root / run_id
        recorder = TraceRecorder(
            run_dir,
            save_prompts=self.save_prompts,
            save_model_responses=self.save_model_responses,
        )
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
                **{
                    k: v
                    for k, v in self.manifest_extra.items()
                    if k != "config_hash"
                },
            }
        )

        state = RunState(run_id=run_id, task=task)
        started = time.monotonic()
        started_at = now_utc_iso()
        status = RunStatus.ERROR
        error_type: ErrorType | None = None
        error_message: str | None = None
        final_reward = 0.0

        try:
            try:
                observation = self.env.reset(task)
            except HarnessError as exc:
                error_type, error_message = exc.error_type, exc.message
                logger.error("env reset failed: %s", exc.message)
                return self._finish(
                    recorder, state, RunStatus.ERROR, error_type, error_message,
                    started, started_at, run_dir,
                )
            except Exception as exc:  # noqa: BLE001 - normalized, never escapes
                error_type, error_message = ErrorType.ENVIRONMENT_INIT_ERROR, str(exc)
                logger.exception("unexpected env reset failure")
                return self._finish(
                    recorder, state, RunStatus.ERROR, error_type, error_message,
                    started, started_at, run_dir,
                )
            state.current_observation = observation

            # -- explicit harness loop ------------------------------------
            for step_idx in range(task.max_steps):
                t0 = time.monotonic()

                # decide
                try:
                    turn, prompt = self.agent.decide(
                        task=task, observation=observation, history=state.steps
                    )
                except ModelOutputParseError as exc:
                    recorder.write_failed_model_output(step_idx, exc.raw_text)
                    step = self._failed_decision_step(
                        state, step_idx, observation, ErrorType.MODEL_OUTPUT_PARSE_ERROR, exc
                    )
                    recorder.record_step(step, observation=observation)
                    state.steps.append(step)
                    error_type, error_message = ErrorType.MODEL_OUTPUT_PARSE_ERROR, exc.message
                    break
                except ModelApiError as exc:
                    step = self._failed_decision_step(
                        state, step_idx, observation, ErrorType.MODEL_API_ERROR, exc
                    )
                    recorder.record_step(step, observation=observation)
                    state.steps.append(step)
                    error_type, error_message = ErrorType.MODEL_API_ERROR, exc.message
                    break
                except HarnessError as exc:
                    step = self._failed_decision_step(
                        state, step_idx, observation, exc.error_type, exc
                    )
                    recorder.record_step(step, observation=observation)
                    state.steps.append(step)
                    error_type, error_message = exc.error_type, exc.message
                    break

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

                # record
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
                    input_tokens=mo.input_tokens if mo else None,
                    output_tokens=mo.output_tokens if mo else None,
                    model_name=mo.model_name if mo else None,
                    short_reason=turn.decision.short_reason,
                )
                step = recorder.record_step(
                    step,
                    observation=env_step.observation,
                    prompt=prompt,
                    model_output=mo,
                )
                state.steps.append(step)
                state.current_observation = env_step.observation
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
    ) -> StepRecord:
        return StepRecord(
            run_id=state.run_id,
            step_index=step_idx,
            url=observation.url,
            action=None,
            error_type=error_type,
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
    ) -> RunResult:
        state.status = status
        duration = time.monotonic() - started
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
        )
        recorder.write_result(result)
        return result
