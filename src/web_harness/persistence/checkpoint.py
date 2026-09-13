"""Durable checkpoints (Phase 2A2).

A checkpoint is created ONLY at a stable safe point:

    no model call in flight, no env.step in flight, the current Observation
    matches the live environment, StepRecords / RuntimeEvents / environment
    operations are already persisted (fsynced), ReliabilityState is settled,
    and the next agent decision has not started.

Checkpoint semantics: `next_step_index` is the step at which the next agent
decision will START. Phase 2A2 persists the state; it does NOT resume.
"""

from __future__ import annotations

import hashlib
import json
import os
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from web_harness.core.errors import TraceWriteError
from web_harness.core.models import TaskSpec
from web_harness.persistence.schema import TRACE_SCHEMA_VERSION
from web_harness.runtime.state import RunState

CHECKPOINT_SCHEMA_VERSION = 1
ENV_JOURNAL_SCHEMA_VERSION = 1


class EnvironmentResumeStrategy(StrEnum):
    """Capability contract: HOW a future Resume can restore the environment.

    Declared capability only in Phase 2A2 — nothing here executes a resume.
    MiniWoB BrowserGymAdapter (and the deterministic FakeEnvironment used by
    tests) declare DETERMINISTIC_REPLAY; everything else is UNSUPPORTED.
    Never claim resume support for all BrowserGym benchmarks.
    """

    DETERMINISTIC_REPLAY = "deterministic_replay"
    NATIVE_SNAPSHOT = "native_snapshot"
    UNSUPPORTED = "unsupported"


def environment_resume_strategy(env, *, benchmark: str) -> EnvironmentResumeStrategy:
    """Resolve the resume capability of the given environment adapter."""
    from web_harness.env.browsergym_adapter import BrowserGymAdapter
    from web_harness.env.fake import FakeEnvironment

    if isinstance(env, BrowserGymAdapter) and benchmark == "miniwob":
        return EnvironmentResumeStrategy.DETERMINISTIC_REPLAY
    if isinstance(env, FakeEnvironment):
        return EnvironmentResumeStrategy.DETERMINISTIC_REPLAY
    return EnvironmentResumeStrategy.UNSUPPORTED


def action_contract_hash(contract) -> str:
    """Deterministic SHA256 over the ActionContract's canonical JSON.

    A future Resume (Phase 2A3) rejects an environment whose action contract
    drifted from the checkpointed one.
    """
    canonical = json.dumps(
        contract.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class CheckpointTraceOffsets(BaseModel):
    """Counts that tie a checkpoint to the durable trace files.

    Phase 2A3 will refuse a checkpoint that these offsets no longer match
    (checkpoint superseded by later durable execution).
    """

    step_count: int
    event_count: int
    environment_op_count: int


class CheckpointEnvelope(BaseModel):
    """Complete, self-describing snapshot of one safe point."""

    checkpoint_schema_version: int = CHECKPOINT_SCHEMA_VERSION

    checkpoint_id: str
    run_id: str

    created_at: str
    reason: str

    git_commit: str | None = None
    config_hash: str | None = None

    trace_schema_version: int = TRACE_SCHEMA_VERSION
    env_journal_schema_version: int = ENV_JOURNAL_SCHEMA_VERSION

    task: TaskSpec
    next_step_index: int

    state: RunState

    environment_op_count: int
    current_environment_fingerprint: str

    action_contract_hash: str

    environment_adapter: str
    environment_resume_strategy: str

    trace_offsets: CheckpointTraceOffsets


def _atomic_write_json(path: Path, payload: dict) -> None:
    """serialize -> write .tmp -> flush -> fsync -> os.replace."""
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        if os.name == "posix":
            # best-effort directory fsync on POSIX; Windows cannot open
            # directory handles this way and relies on os.replace alone
            try:
                dir_fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
    except OSError as exc:
        raise TraceWriteError(f"cannot write {path}: {exc}") from exc


class CheckpointManager:
    """Atomic checkpoint writer/reader for one run directory.

    Layout: runs/<run_id>/checkpoints/cp_NNNNNN.json + latest.json.
    Orphan `.tmp` files (crash mid-write) are never valid checkpoints.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.checkpoints_dir = self.run_dir / "checkpoints"

    def _next_index(self) -> int:
        if not self.checkpoints_dir.exists():
            return 0
        indices = [
            int(p.stem.split("_")[1])
            for p in self.checkpoints_dir.glob("cp_*.json")
        ]
        return max(indices) + 1 if indices else 0

    def save(
        self,
        *,
        run_id: str,
        created_at: str,
        reason: str,
        task: dict,
        state: RunState,
        next_step_index: int,
        environment_op_count: int,
        current_environment_fingerprint: str,
        action_contract_hash: str,
        environment_adapter: str,
        resume_strategy: str,
        trace_offsets: CheckpointTraceOffsets,
        git_commit: str | None = None,
        config_hash: str | None = None,
    ) -> CheckpointEnvelope:
        """Atomically persist one checkpoint and update latest.json."""
        index = self._next_index()
        envelope = CheckpointEnvelope(
            checkpoint_id=f"cp-{index:06d}",
            run_id=run_id,
            created_at=created_at,
            reason=reason,
            git_commit=git_commit,
            config_hash=config_hash,
            task=task,
            next_step_index=next_step_index,
            state=state,
            environment_op_count=environment_op_count,
            current_environment_fingerprint=current_environment_fingerprint,
            action_contract_hash=action_contract_hash,
            environment_adapter=environment_adapter,
            environment_resume_strategy=resume_strategy,
            trace_offsets=trace_offsets,
        )
        filename = f"cp_{index:06d}.json"
        _atomic_write_json(
            self.checkpoints_dir / filename,
            envelope.model_dump(mode="json"),
        )
        _atomic_write_json(
            self.checkpoints_dir / "latest.json",
            {"checkpoint_id": envelope.checkpoint_id, "file": filename},
        )
        return envelope

    def load(self, filename: str) -> CheckpointEnvelope | None:
        """Load one checkpoint by file name; `.tmp` files are never valid."""
        if filename.endswith(".tmp") or not filename.startswith("cp_"):
            return None
        path = self.checkpoints_dir / filename
        if not path.exists():
            return None
        try:
            return CheckpointEnvelope.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise TraceWriteError(f"cannot load {path}: {exc}") from exc

    def latest(self) -> CheckpointEnvelope | None:
        """Return the most recent valid checkpoint, or None when there is
        none. latest.json is authoritative; a missing/corrupt pointer falls
        back to the highest cp_ index. Orphan `.tmp` files are ignored."""
        pointer_path = self.checkpoints_dir / "latest.json"
        if pointer_path.exists():
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                envelope = self.load(pointer.get("file", ""))
                if envelope is not None:
                    return envelope
            except (OSError, ValueError):
                pass  # corrupt pointer: fall through to a directory scan
        return self._scan_latest()

    def _scan_latest(self) -> CheckpointEnvelope | None:
        if not self.checkpoints_dir.exists():
            return None
        valid = sorted(self.checkpoints_dir.glob("cp_*.json"))
        return self.load(valid[-1].name) if valid else None
