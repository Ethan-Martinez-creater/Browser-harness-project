"""Trace persistence schema versions.

`TRACE_SCHEMA_VERSION` describes the on-disk run trace layout:

- v1: the Phase 0-1D layout (text observation artifacts only, no version
  field in manifest.json).
- v2: adds structured Observation JSON artifacts (`obs_NNN.json` /
  `next_obs_NNN.json`) plus the corresponding additive `StepRecord`
  references; the `.txt` artifacts keep their existing semantics.

Legacy runs are never modified: a manifest without `trace_schema_version` is
treated as v1 by offline tooling (replay treats it as structural-only).
"""

from __future__ import annotations

TRACE_SCHEMA_VERSION = 2


def manifest_schema_version(manifest: dict) -> int:
    """Schema version of a loaded manifest; a missing field means legacy v1."""
    value = manifest.get("trace_schema_version")
    if value is None:
        return 1
    return int(value)
