"""Deterministic-enough identifiers for runs and experiments."""

from __future__ import annotations

import datetime as _dt
import uuid


def now_utc_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision, Z suffix)."""
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    """Unique run id: run-<utc timestamp>-<short uuid>."""
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{stamp}-{uuid.uuid4().hex[:8]}"


def new_experiment_id(prefix: str = "exp") -> str:
    """Unique experiment id: <prefix>-<utc timestamp>-<short uuid>."""
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"
