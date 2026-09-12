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
            self.reliability = data.get("reliability", {})
        except KeyError as exc:
            raise ConfigError(f"missing config section: {exc}") from exc

        if not isinstance(self.reliability, dict):
            raise ConfigError("reliability config must be a mapping")
        reliability_enabled = bool(self.reliability.get("enabled", False))
        if self.reliability and not reliability_enabled:
            # explicit disabled switch wins; ignore sub-sections entirely
            self.reliability = {"enabled": False}
        verifier_cfg = self.reliability.get("verification") or {}
        if reliability_enabled and verifier_cfg:
            mode = verifier_cfg.get("mode", "shadow")
            if mode == "active":
                raise ConfigError(
                    "verification mode 'active' is not available in Phase 1A "
                    "(only 'shadow'); active verification arrives in later phases"
                )
            if mode != "shadow":
                raise ConfigError(f"unknown verification mode: {mode}")
            if not isinstance(verifier_cfg.get("enabled", True), bool):
                raise ConfigError("verification.enabled must be a boolean")
        if reliability_enabled:
            self._validate_retry_config(self.reliability.get("retry") or {})
            budget_cfg = self.reliability.get("budget") or {}
            if "max_extra_model_calls_per_episode" in budget_cfg:
                value = budget_cfg["max_extra_model_calls_per_episode"]
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise ConfigError(
                        "reliability.budget.max_extra_model_calls_per_episode "
                        "must be an int >= 0"
                    )

        if self.model.get("provider") not in (None, "openai_compatible", "mock"):
            raise ConfigError(f"unknown model provider: {self.model.get('provider')}")
        if self.model.get("provider") == "openai_compatible" and not self.model.get("model"):
            raise ConfigError("model.model is required for provider openai_compatible")

    @classmethod
    def _validate_retry_config(cls, retry_cfg: dict[str, Any]) -> None:
        """Fail-fast validation of the retry section (R3)."""
        if not retry_cfg:
            return
        enabled = retry_cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError("reliability.retry.enabled must be a boolean")

        def _validate_non_negative_int(value: Any, name: str) -> None:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ConfigError(f"reliability.retry.{name} must be an int >= 0")

        api_cfg = retry_cfg.get("model_api") or {}
        if "max_retries" in api_cfg:
            _validate_non_negative_int(api_cfg["max_retries"], "model_api.max_retries")
        backoff = api_cfg.get("backoff_ms")
        if backoff is not None and (
            not isinstance(backoff, list)
            or not all(
                isinstance(ms, int) and not isinstance(ms, bool) and ms >= 0
                for ms in backoff
            )
        ):
            raise ConfigError(
                "reliability.retry.model_api.backoff_ms must be a list of ints >= 0"
            )
        output_cfg = retry_cfg.get("model_output") or {}
        if "max_retries" in output_cfg:
            _validate_non_negative_int(
                output_cfg["max_retries"], "model_output.max_retries"
            )
        # NOTE: when api_max_retries > len(backoff_ms), the LAST backoff value
        # is reused for all remaining retries (deliberate, deterministic).

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

    @property
    def reliability_enabled(self) -> bool:
        return bool(self.reliability.get("enabled", False))

    @property
    def verification_enabled(self) -> bool:
        """Verification runs only when reliability AND verification are on."""
        verifier_cfg = self.reliability.get("verification") or {}
        return self.reliability_enabled and bool(verifier_cfg.get("enabled", True))

    @property
    def verification_mode(self) -> str:
        verifier_cfg = self.reliability.get("verification") or {}
        return str(verifier_cfg.get("mode", "shadow"))
