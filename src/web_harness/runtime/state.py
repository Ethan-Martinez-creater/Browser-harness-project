"""In-memory run state for one episode.

Phase 0 keeps state in memory only (no database). This schema is the seed for
later checkpoint/persistence phases, so it stays JSON-serializable.

Phase 2A2: all mutable episode accounting lives in `RunState.counters`
(EpisodeCounters) — the SINGLE authoritative source. EpisodeRunner no longer
keeps parallel local counters, so a checkpoint captures every reliability
metric a future Resume needs. RunResult semantics are unchanged: `_finish`
reads exactly these counters.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from web_harness.core.models import Observation, RunStatus, StepRecord, TaskSpec
from web_harness.core.reliability import ReliabilityState


class EpisodeCounters(BaseModel):
    """Episode-level reliability accounting (single authoritative copy).

    Groups the counters that previously lived as EpisodeRunner.run() local
    variables. Values and semantics are identical to the Phase 0-1 fields on
    RunResult; recovery/replan outcome invariants apply exactly as before.
    """

    # Phase 1B retry accounting
    retry_count: int = 0
    retry_cycle_count: int = 0  # decision cycles with >= 1 retry
    retry_success_count: int = 0
    retry_exhausted_count: int = 0
    extra_model_calls: int = 0
    retry_input_tokens: int = 0
    retry_output_tokens: int = 0
    retry_latency_s: float = 0.0

    # Phase 1C recovery accounting.
    # recovery_count = created/executed RecoveryDirectives; success + failed
    # + unresolved must always sum to recovery_count.
    recovery_count: int = 0
    recovery_success_count: int = 0
    recovery_failed_count: int = 0
    recovery_unresolved_count: int = 0
    recovery_environment_actions: int = 0
    recovery_latency_s: float = 0.0
    recovered_episode: bool = False  # recovery triggered AND episode success
    # blocked-action re-decisions: counted separately from recovery_count
    blocked_action_redecision_count: int = 0

    # Phase 1D replan accounting: success + failed + unresolved == replan_count
    replan_count: int = 0
    replan_success_count: int = 0
    replan_failed_count: int = 0
    replan_unresolved_count: int = 0
    replan_model_calls: int = 0
    replan_input_tokens: int = 0
    replan_output_tokens: int = 0
    replan_latency_s: float = 0.0


class RunState(BaseModel):
    run_id: str
    task: TaskSpec
    status: RunStatus = RunStatus.RUNNING
    current_observation: Observation | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    reliability: ReliabilityState = Field(default_factory=ReliabilityState)
    counters: EpisodeCounters = Field(default_factory=EpisodeCounters)

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
