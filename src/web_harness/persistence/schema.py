"""Trace persistence schema versions.

`TRACE_SCHEMA_VERSION` describes the on-disk run trace layout:

- v1: the Phase 0-1D layout (text observation artifacts only, no version
  field in manifest.json).
- v2: adds structured Observation JSON artifacts (`obs_NNN.json` /
  `next_obs_NNN.json`) plus the corresponding additive `StepRecord`
  references and the recorded `verification_spec`; the `.txt` artifacts
  keep their existing semantics.

Legacy runs are never modified: a manifest without `trace_schema_version` is
treated as v1 by offline tooling (replay treats it as structural-only).
Version parsing is fail-closed: an invalid or unsupported version becomes a
structured error for the replay report, never an unhandled exception.
"""

from __future__ import annotations

TRACE_SCHEMA_VERSION = 2

# versions this harness can load and validate
SUPPORTED_TRACE_SCHEMA_VERSIONS = {1, 2}


def parse_manifest_schema_version(manifest) -> tuple[int, list[str]]:
    """Fail-closed schema version resolution for a loaded manifest.

    Returns (version, errors). `version` is 0 when the value is invalid
    (callers must treat a 0 as structurally invalid, never replayable); the
    actual stored value is kept for unsupported-but-well-formed versions so
    reports can name it. A missing field means legacy v1.
    """
    if not isinstance(manifest, dict):
        return 0, ["manifest.json top level is not a JSON object"]
    value = manifest.get("trace_schema_version")
    if value is None:
        return 1, []
    if isinstance(value, bool) or not isinstance(value, int):
        return 0, [
            f"manifest trace_schema_version invalid: {value!r} "
            f"(expected integer, one of {sorted(SUPPORTED_TRACE_SCHEMA_VERSIONS)})"
        ]
    if value not in SUPPORTED_TRACE_SCHEMA_VERSIONS:
        return value, [
            f"unsupported trace schema version: {value} "
            f"(supported: {sorted(SUPPORTED_TRACE_SCHEMA_VERSIONS)})"
        ]
    return value, []
