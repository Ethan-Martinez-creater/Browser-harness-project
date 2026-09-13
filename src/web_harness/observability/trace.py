"""Episode trace recorder.

Every episode gets a directory:

    runs/<run_id>/
        manifest.json     run metadata (git commit, config hash, model, task)
        steps.jsonl       one StepRecord per line, flushed after every step
        result.json       final RunResult
        artifacts/        deterministic per-step artifacts

Trace failures are never silent: they raise TraceWriteError. API keys must
never reach any trace content.
"""

from __future__ import annotations

import json
from pathlib import Path

from web_harness.core.errors import TraceWriteError
from web_harness.core.events import RuntimeEvent
from web_harness.core.models import ModelOutput, Observation, PromptBundle, RunResult, StepRecord


def _write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise TraceWriteError(f"cannot write {path}: {exc}") from exc


def _write_json(path: Path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _observation_text(obs: Observation) -> str:
    parts = [
        f"# goal\n{obs.goal}",
        f"# url\n{obs.url}",
        "# open_pages\n" + "\n".join(obs.open_pages),
        f"# last_action\n{obs.last_action or ''}",
        f"# last_action_error\n{obs.last_action_error or ''}",
        f"# truncated\n{obs.truncated}",
        f"# axtree\n{obs.axtree or ''}",
    ]
    if obs.dom:
        parts.append(f"# dom\n{obs.dom}")
    if obs.screenshot_path:
        parts.append(f"# screenshot: {obs.screenshot_path}")
    return "\n\n".join(parts)


class TraceRecorder:
    def __init__(
        self,
        run_dir: Path,
        *,
        save_prompts: bool = True,
        save_model_responses: bool = True,
    ):
        self.run_dir = Path(run_dir)
        self.artifact_dir = self.run_dir / "artifacts"
        self.save_prompts = save_prompts
        self.save_model_responses = save_model_responses
        self._steps_path = self.run_dir / "steps.jsonl"
        self._events_path = self.run_dir / "events.jsonl"
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TraceWriteError(f"cannot create {self.artifact_dir}: {exc}") from exc

    # -- metadata ----------------------------------------------------------

    def write_manifest(self, fields: dict) -> None:
        _write_json(self.run_dir / "manifest.json", fields)

    def write_result(self, result: RunResult) -> None:
        _write_json(self.run_dir / "result.json", result.model_dump(mode="json"))

    def write_failed_model_output(self, step_index: int, raw_text: str | None) -> None:
        """Persist raw model output for a step whose parsing failed."""
        if raw_text:
            _write_text(
                self.run_dir / f"artifacts/raw_failed_{step_index:03d}.txt", raw_text
            )

    def write_failed_attempt(
        self,
        *,
        step_index: int | None,
        attempt_index: int,
        failure_type: str,
        raw_text: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        model_name: str | None = None,
        component: str = "decision",
    ) -> str | None:
        """Persist one failed model attempt as a deterministic artifact (R4).

        `component` namespaces the file so decision-cycle and replan-cycle
        attempts on the same step never overwrite each other (closure B4).
        Returns the artifact reference (relative to the run dir) so retry
        events can point at it, or None when there is nothing to store.
        steps.jsonl one-step semantics are untouched.
        """
        if not raw_text:
            return None
        pad_step = f"{step_index:03d}" if step_index is not None else "xxx"
        prefix = "attempt" if component == "decision" else f"{component}_attempt"
        ref = f"artifacts/{prefix}_{pad_step}_{attempt_index:02d}.json"
        _write_json(
            self.run_dir / ref,
            {
                "step_index": step_index,
                "attempt_index": attempt_index,
                "failure_type": failure_type,
                "raw_text": raw_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model_name": model_name,
            },
        )
        return ref

    # -- steps -------------------------------------------------------------

    def write_replan_plan(
        self,
        *,
        replan_index: int,
        plan,
        trigger_reason: str | None,
        failure_kind: str | None,
        failure_signature: str | None,
        recovery_failure_streak: int,
        model_calls: int,
        input_tokens: int | None,
        output_tokens: int | None,
        latency_s: float,
    ) -> str:
        """Persist one created RecoveryPlan as artifacts/replan_XXX.json (1D).

        The REPLAN event stores only the artifact ref + summary fields; the
        full plan lives here. Returns the artifact reference."""
        ref = f"artifacts/replan_{replan_index:03d}.json"
        _write_json(
            self.run_dir / ref,
            {
                "replan_index": replan_index,
                "plan": plan.model_dump(mode="json") if plan else None,
                "trigger_reason": trigger_reason,
                "failure_kind": failure_kind,
                "failure_signature": failure_signature,
                "recovery_failure_streak": recovery_failure_streak,
                "model_calls": model_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_s": latency_s,
            },
        )
        return ref

    def record_step(
        self,
        step: StepRecord,
        *,
        observation: Observation | None = None,
        next_observation: Observation | None = None,
        prompt: PromptBundle | None = None,
        model_output: ModelOutput | None = None,
    ) -> StepRecord:
        """Persist one step with its artifacts; returns the updated record.

        Trace semantics (B2): `observation` is the PRE-action observation the
        model actually decided on (matches prompt_N / model_N);
        `next_observation` is the POST-action result of executing the action.
        obs_(N+1) must always equal next_obs_N semantically. When a step fails
        before an action executes, next_observation is left unset — it is
        never fabricated.

        Artifact names are deterministic (zero-padded step index). Reference
        fields on the StepRecord are filled in before writing to steps.jsonl.
        """
        idx = step.step_index
        pad = f"{idx:03d}"

        if observation is not None:
            obs_ref = f"artifacts/obs_{pad}.txt"
            _write_text(self.run_dir / obs_ref, _observation_text(observation))
            step.observation_ref = obs_ref
            # structured JSON twin (Phase 2A1): machine-readable observation
            # for offline replay; the .txt artifact stays the human view
            obs_json_ref = f"artifacts/obs_{pad}.json"
            _write_json(
                self.run_dir / obs_json_ref, observation.model_dump(mode="json")
            )
            step.observation_json_ref = obs_json_ref

        if next_observation is not None:
            next_ref = f"artifacts/next_obs_{pad}.txt"
            _write_text(self.run_dir / next_ref, _observation_text(next_observation))
            step.next_observation_ref = next_ref
            next_json_ref = f"artifacts/next_obs_{pad}.json"
            _write_json(
                self.run_dir / next_json_ref, next_observation.model_dump(mode="json")
            )
            step.next_observation_json_ref = next_json_ref

        if prompt is not None and self.save_prompts:
            prompt_ref = f"artifacts/prompt_{pad}.txt"
            content = f"# system\n{prompt.system}\n\n# user\n{prompt.user}"
            _write_text(self.run_dir / prompt_ref, content)
            step.prompt_ref = prompt_ref

        if model_output is not None and self.save_model_responses:
            resp_ref = f"artifacts/model_{pad}.json"
            _write_json(
                self.run_dir / resp_ref,
                model_output.model_dump(mode="json", exclude={"raw_text"}),
            )
            step.model_response_ref = resp_ref
            if model_output.raw_text:
                _write_text(self.run_dir / f"artifacts/raw_{pad}.txt", model_output.raw_text)

        line = json.dumps(step.model_dump(mode="json"), ensure_ascii=False, default=str)
        try:
            self._steps_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._steps_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                # flush to disk: os.fsync so a crash never loses recorded steps
                import os

                os.fsync(f.fileno())
        except OSError as exc:
            raise TraceWriteError(f"cannot append to {self._steps_path}: {exc}") from exc
        return step

    # -- reliability event stream (events.jsonl) ---------------------------

    def record_event(self, event: RuntimeEvent) -> RuntimeEvent:
        """Append one runtime event (verification/policy/retry/...), fsynced."""
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False, default=str)
        try:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._events_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                import os

                os.fsync(f.fileno())
        except OSError as exc:
            raise TraceWriteError(f"cannot append to {self._events_path}: {exc}") from exc
        return event

    @staticmethod
    def read_events(run_dir: Path) -> list[RuntimeEvent]:
        path = Path(run_dir) / "events.jsonl"
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(RuntimeEvent.model_validate_json(line))
        return events

    # -- read helpers (used by inspect-run and evaluation) -----------------

    @staticmethod
    def read_steps(run_dir: Path) -> list[StepRecord]:
        path = Path(run_dir) / "steps.jsonl"
        if not path.exists():
            return []
        steps = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                steps.append(StepRecord.model_validate_json(line))
        return steps

    @staticmethod
    def read_result(run_dir: Path) -> RunResult | None:
        path = Path(run_dir) / "result.json"
        if not path.exists():
            return None
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))
