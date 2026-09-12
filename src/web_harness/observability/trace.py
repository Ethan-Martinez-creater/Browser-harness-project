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
from web_harness.core.models import ModelOutput, Observation, PromptBundle, RunResult, StepRecord


def _write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise TraceWriteError(f"cannot write {path}: {exc}") from exc


def _write_json(path: Path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


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

    # -- steps -------------------------------------------------------------

    def record_step(
        self,
        step: StepRecord,
        *,
        observation: Observation | None = None,
        prompt: PromptBundle | None = None,
        model_output: ModelOutput | None = None,
    ) -> StepRecord:
        """Persist one step with its artifacts; returns the updated record.

        Artifact names are deterministic (zero-padded step index). Reference
        fields on the StepRecord are filled in before writing to steps.jsonl.
        """
        idx = step.step_index
        pad = f"{idx:03d}"

        if observation is not None:
            obs_ref = f"artifacts/obs_{pad}.txt"
            parts = [
                f"# goal\n{observation.goal}",
                f"# url\n{observation.url}",
                "# open_pages\n" + "\n".join(observation.open_pages),
                f"# last_action\n{observation.last_action or ''}",
                f"# last_action_error\n{observation.last_action_error or ''}",
                f"# truncated\n{observation.truncated}",
                f"# axtree\n{observation.axtree or ''}",
            ]
            if observation.dom:
                parts.append(f"# dom\n{observation.dom}")
            if observation.screenshot_path:
                parts.append(f"# screenshot: {observation.screenshot_path}")
            _write_text(self.run_dir / obs_ref, "\n\n".join(parts))
            step.observation_ref = obs_ref

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
