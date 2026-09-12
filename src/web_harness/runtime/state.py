"""In-memory run state for one episode.

Phase 0 keeps state in memory only (no database). This schema is the seed for
later checkpoint/persistence phases, so it stays JSON-serializable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from web_harness.core.models import Observation, RunStatus, StepRecord, TaskSpec


class RunState(BaseModel):
    run_id: str
    task: TaskSpec
    status: RunStatus = RunStatus.RUNNING
    current_observation: Observation | None = None
    steps: list[StepRecord] = Field(default_factory=list)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens or 0 for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens or 0 for s in self.steps)

    @property
    def action_error_count(self) -> int:
        return sum(1 for s in self.steps if s.action_error)
