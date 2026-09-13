"""Environment operation journal (Phase 2A2).

Every REAL `EnvironmentAdapter.step()` call — agent actions AND Harness
recovery actions (e.g. WAIT_AND_REOBSERVE noop) — is appended here, fsynced,
before verification/checkpoint work continues. The journal is the durable
answer to "which environment operations produced the current page state";
Phase 2A3 will reconstruct the environment from it.

NOT recorded (they never touch the environment): model retries, blocked-action
re-decisions, verifier/policy runs, replanner model calls.

MiniWoB bootstrap runs inside `env.reset()` and is re-executed by any future
reconstruction that calls `reset()` again — it is deliberately NOT journaled
to avoid double execution.
"""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from web_harness.core.errors import TraceWriteError
from web_harness.reliability.fingerprint import normalize_error_signature

ENV_JOURNAL_SCHEMA_VERSION = 1


class EnvironmentOperationKind(StrEnum):
    AGENT_ACTION = "agent_action"
    RECOVERY_ACTION = "recovery_action"


class EnvironmentOperationRecord(BaseModel):
    """One real environment-affecting operation, in durable order."""

    schema_version: int = ENV_JOURNAL_SCHEMA_VERSION
    run_id: str
    op_index: int  # monotonic over the whole run, 0-based
    source_step_index: int | None = None  # agent step this op belongs to
    kind: EnvironmentOperationKind
    action: str
    expected_post_fingerprint: str
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    action_error_signature: str | None = None


class EnvironmentOperationJournal:
    """Append-only, fsynced JSONL journal for one run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "environment_ops.jsonl"
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def append(
        self,
        *,
        run_id: str,
        kind: EnvironmentOperationKind,
        action: str,
        post_observation_fingerprint: str,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        action_error: str | None = None,
        source_step_index: int | None = None,
    ) -> EnvironmentOperationRecord:
        """Record one executed env.step and durably flush it (append+fsync)."""
        record = EnvironmentOperationRecord(
            run_id=run_id,
            op_index=self._count,
            source_step_index=source_step_index,
            kind=kind,
            action=action,
            expected_post_fingerprint=post_observation_fingerprint,
            reward=float(reward or 0.0),
            terminated=bool(terminated),
            truncated=bool(truncated),
            action_error_signature=(
                normalize_error_signature(action_error)
                if action_error
                else None
            ),
        )
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise TraceWriteError(
                f"cannot append to {self.path}: {exc}"
            ) from exc
        self._count += 1
        return record

    def read_all(self) -> list[EnvironmentOperationRecord]:
        """Read all journaled operations in recorded order."""
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(EnvironmentOperationRecord.model_validate_json(line))
        return records


def read_journal(run_dir: Path) -> list[EnvironmentOperationRecord]:
    """Standalone read helper (no writer state needed)."""
    path = Path(run_dir) / "environment_ops.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(EnvironmentOperationRecord.model_validate_json(line))
    return records


def count_journal_records(run_dir: Path) -> int:
    return len(read_journal(run_dir))
