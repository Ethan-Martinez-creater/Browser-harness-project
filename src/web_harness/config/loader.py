"""Configuration loading (YAML -> typed pydantic models).

Rules enforced here:
- secrets only ever come from environment variables (never config files)
- every parameter that can influence results is part of the config
- a stable hash of the resolved config can be embedded in manifests
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from web_harness.core.errors import ConfigError


def load_env_file(path: Path | str = ".env") -> bool:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Existing environment variables always win (the file never overrides a
    value already set in the shell). Values are not logged. Returns True when
    a file was loaded.
    """
    path = Path(path)
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def config_hash(data: dict[str, Any]) -> str:
    """Stable hash of a config mapping (for experiment manifests)."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class ModelConfig(dict):
    """Typed accessors over the `model:` config section."""


class HarnessConfig:
    """Typed wrapper over the harness baseline config (no secret values)."""

    # keys that may never carry a literal secret in YAML (R3 fail-fast)
    FORBIDDEN_SECRET_KEYS = ("api_key", "token", "secret", "password")

    def __init__(self, data: dict[str, Any], *, source_path: Path | None = None):
        self.data = data
        self.source_path = source_path
        self._validate_no_literal_secrets(data)
        try:
            self.model = data["model"]
            self.agent = data.get("agent", {})
            self.runtime = data.get("runtime", {})
            self.trace = data.get("trace", {})
            self.environment = data.get("environment", {})
        except KeyError as exc:
            raise ConfigError(f"missing config section: {exc}") from exc

        if self.model.get("provider") not in (None, "openai_compatible", "mock"):
            raise ConfigError(f"unknown model provider: {self.model.get('provider')}")
        if self.model.get("provider") == "openai_compatible" and not self.model.get("model"):
            raise ConfigError("model.model is required for provider openai_compatible")

    @classmethod
    def _validate_no_literal_secrets(cls, data: dict[str, Any]) -> None:
        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_l = str(key).lower()
                    if key_l in cls.FORBIDDEN_SECRET_KEYS and isinstance(value, str) and value:
                        raise ConfigError(
                            f"literal secret in config at '{path}{key}': secrets must "
                            "only come from environment variables (use *_env keys)"
                        )
                    walk(value, f"{path}{key}.")

        walk(data, "")

    @classmethod
    def from_yaml(cls, path: Path | str) -> HarnessConfig:
        path = Path(path)
        return cls(_load_yaml(path), source_path=path)

    @property
    def hash(self) -> str:
        return config_hash(self.data)

    @property
    def trace_root(self) -> Path:
        return Path(self.trace.get("root_dir", "runs"))

    # common accessors -----------------------------------------------------

    @property
    def max_steps(self) -> int:
        return int(self.runtime.get("max_steps", 20))

    @property
    def max_history_steps(self) -> int:
        return int(self.agent.get("max_history_steps", 4))

    @property
    def observation_char_limit(self) -> int:
        return int(self.agent.get("observation_char_limit", 30000))

    @property
    def save_prompts(self) -> bool:
        return bool(self.trace.get("save_prompts", True))

    @property
    def save_model_responses(self) -> bool:
        return bool(self.trace.get("save_model_responses", True))

    @property
    def save_screenshots(self) -> bool:
        return bool(self.trace.get("save_screenshots", False))

    @property
    def bootstrap_action(self) -> str | None:
        """Explicit environment bootstrap action (see ADR-004); None = disabled."""
        return self.environment.get("bootstrap_action")

    @property
    def headless(self) -> bool:
        return bool(self.environment.get("headless", True))
