"""Environment abstraction.

The harness core only ever talks to `EnvironmentAdapter`; BrowserGym (or any
other environment) is reached through an adapter so benchmarks stay swappable.
"""

from __future__ import annotations

from typing import Protocol

from web_harness.core.models import EnvironmentStep, Observation, TaskSpec


class EnvironmentAdapter(Protocol):
    """Lifecycle: reset() -> step()* -> close(). close() must be idempotent."""

    def reset(self, task: TaskSpec) -> Observation: ...

    def step(self, action: str) -> EnvironmentStep: ...

    def close(self) -> None: ...
