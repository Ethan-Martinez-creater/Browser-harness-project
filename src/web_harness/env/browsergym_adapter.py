"""BrowserGym adapter.

This is the ONLY module allowed to import BrowserGym/Gymnasium. Everything
above the environment layer works with `TaskSpec`, `Observation` and
`EnvironmentStep` exclusively.

Action contract: the adapter builds ONE HighLevelActionSet and installs its
`to_python_code` as the environment's action mapping; the prompt-facing
ActionContract is generated from that same instance, so every action shown to
the agent is guaranteed to be accepted by this environment.

Bootstrap action: MiniWoB pages render their task HTML after DOM-load, so the
first observation right after reset() can have an empty accessibility tree.
If a bootstrap action is configured, the adapter executes it once after
reset(); it is verified to produce no reward and no termination, exposed in
provenance, and must stay disabled for other benchmarks (see ADR-004).
"""

from __future__ import annotations

import os
from pathlib import Path

from web_harness.core.errors import EnvironmentInitError
from web_harness.core.models import EnvironmentStep, Observation, TaskSpec
from web_harness.env.action_contract import (
    ActionContract,
    action_contract_from_browsergym,
)
from web_harness.env.observation import ObservationNormalizer

DEFAULT_MINIWOB_LOCAL_PATH = (
    Path("third_party") / "miniwob-plusplus" / "miniwob" / "html" / "miniwob"
)


def resolve_miniwob_url(configured: str | None = None) -> str:
    """Resolve the MiniWoB HTML base URL.

    Priority: explicit config value > MINIWOB_URL env var > local
    third_party/miniwob-plusplus checkout (file:// URL).
    """
    if configured:
        return configured
    if os.environ.get("MINIWOB_URL"):
        return os.environ["MINIWOB_URL"]
    local = Path(os.environ.get("MINIWOB_LOCAL_PATH", str(DEFAULT_MINIWOB_LOCAL_PATH)))
    if local.is_dir():
        return local.resolve().as_uri().rstrip("/") + "/"
    raise EnvironmentInitError(
        "MiniWoB HTML source not found. Set MINIWOB_URL, or clone "
        "https://github.com/Farama-Foundation/miniwob-plusplus to "
        f"{DEFAULT_MINIWOB_LOCAL_PATH}."
    )


class BrowserGymAdapter:
    """Adapter around a single BrowserGym environment instance.

    Lifecycle: one adapter instance per episode. `close()` is idempotent and
    must be called even when the episode fails (the runner guarantees this via
    try/finally).
    """

    def __init__(
        self,
        *,
        observation_char_limit: int = 30000,
        headless: bool = True,
        save_screenshots: bool = False,
        artifact_dir: Path | None = None,
        miniwob_url: str | None = None,
        bootstrap_action: str | None = None,
        multiaction: bool = False,
    ):
        self._normalizer = ObservationNormalizer(
            observation_char_limit=observation_char_limit
        )
        self._headless = headless
        self._save_screenshots = save_screenshots
        self._artifact_dir = artifact_dir
        self._miniwob_url = miniwob_url
        self._bootstrap_action = bootstrap_action
        self._env = None
        self._last_observation: Observation | None = None
        self._closed = False
        self._contract: ActionContract | None = None
        self._action_set = self._build_action_set(multiaction=multiaction)
        # provenance: set during reset() when a bootstrap action was executed
        self.bootstrap_action_executed: str | None = None

    @staticmethod
    def _build_action_set(*, multiaction: bool):
        """Build the env's action set.

        One instance serves both execution (env action_mapping) and the prompt
        contract. `check`/`uncheck` are valid BrowserGym actions but are
        excluded from the upstream default bid subset; they are re-added as
        custom actions so the contract can expose them.
        """
        from browsergym.core.action.functions import check, uncheck
        from browsergym.core.action.highlevel import HighLevelActionSet

        return HighLevelActionSet(
            subsets=["bid", "tab", "custom"],
            custom_actions=[check, uncheck],
            multiaction=multiaction,
            strict=False,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def gym_task_id(task: TaskSpec) -> str:
        """Map TaskSpec to a BrowserGym task id, e.g. browsergym/miniwob.click-test."""
        return f"browsergym/{task.benchmark}.{task.task_id}"

    def action_contract(self) -> ActionContract:
        if self._contract is None:
            self._contract = action_contract_from_browsergym(
                self._action_set, benchmark="browsergym"
            )
        return self._contract

    def _normalize(self, raw_obs: dict) -> Observation:
        # BrowserGym 0.14 exposes objects; text must be extracted explicitly.
        from browsergym.utils.obs import flatten_axtree_to_str

        try:
            axtree_text = flatten_axtree_to_str(raw_obs.get("axtree_object")) or ""
        except Exception as exc:  # observation extraction must not crash the run
            axtree_text = ""
            last_error = raw_obs.get("last_action_error") or f"axtree extraction failed: {exc}"
        else:
            last_error = raw_obs.get("last_action_error")

        elapsed = raw_obs.get("elapsed_time")
        try:
            elapsed_s = float(elapsed)
        except (TypeError, ValueError):
            # BrowserGym may return a numpy array; take its scalar value
            try:
                elapsed_s = float(elapsed.flatten()[-1])
            except Exception:
                elapsed_s = 0.0

        open_pages = list(raw_obs.get("open_pages_urls") or [])
        screenshot_path = self._maybe_save_screenshot(raw_obs)
        obs = self._normalizer.from_axtree_parts(
            goal=str(raw_obs.get("goal") or ""),
            url=str(raw_obs.get("url") or ""),
            axtree_text=axtree_text,
            open_pages=[str(p) for p in open_pages],
            last_action=raw_obs.get("last_action"),
            last_action_error=last_error,
            elapsed_time_s=elapsed_s,
        )
        obs.screenshot_path = screenshot_path
        return obs

    def _maybe_save_screenshot(self, raw_obs: dict) -> str | None:
        if not self._save_screenshots or not self._artifact_dir:
            return None
        shot = raw_obs.get("screenshot")
        if shot is None:
            return None
        import datetime as dt

        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        name = f"screenshot_{dt.datetime.now().strftime('%H%M%S%f')}.png"
        path = self._artifact_dir / name
        path.write_bytes(shot)
        return str(path)

    # -- EnvironmentAdapter protocol ---------------------------------------

    def reset(self, task: TaskSpec) -> Observation:
        import gymnasium as gym

        try:
            if task.benchmark == "miniwob":
                os.environ.setdefault("MINIWOB_URL", resolve_miniwob_url(self._miniwob_url))
            import browsergym.miniwob  # noqa: F401  (registers miniwob tasks)

            env = gym.make(
                self.gym_task_id(task),
                headless=self._headless,
                action_mapping=self._action_set.to_python_code,
            )
            self._env = env
            raw_obs, _info = env.reset(seed=task.seed)
            obs = self._normalize(raw_obs)

            # Explicit, verified, provenance-tracked bootstrap: MiniWoB task
            # HTML renders after DOM-load, so the very first observation can
            # have an empty axtree. The bootstrap action must not produce
            # reward or termination, otherwise the task state is unusable.
            if self._bootstrap_action:
                raw_obs, reward, terminated, truncated, _info = self._env.step(
                    self._bootstrap_action
                )
                if terminated or truncated or float(reward or 0.0) != 0.0:
                    raise EnvironmentInitError(
                        f"bootstrap action {self._bootstrap_action!r} changed the "
                        f"task state (reward={reward}, terminated={terminated}, "
                        f"truncated={truncated})"
                    )
                obs = self._normalize(raw_obs)
                self.bootstrap_action_executed = self._bootstrap_action
        except Exception as exc:
            self.close()
            if isinstance(exc, EnvironmentInitError):
                raise
            raise EnvironmentInitError(
                f"failed to init {self.gym_task_id(task)}: {exc}"
            ) from exc
        self._last_observation = obs
        return obs

    def step(self, action: str) -> EnvironmentStep:
        """Execute one action.

        Action exceptions are normalized: they never escape as raw Python
        exceptions; they are reported on EnvironmentStep.action_error and the
        runner decides about termination.
        """
        if self._env is None:
            raise EnvironmentInitError("environment not reset before step()")
        try:
            raw_obs, reward, terminated, truncated, _info = self._env.step(action)
            obs = self._normalize(raw_obs)
            error = raw_obs.get("last_action_error") or None
            env_step = EnvironmentStep(
                observation=obs,
                reward=float(reward or 0.0),
                terminated=bool(terminated),
                truncated=bool(truncated),
                action_error=error,
            )
        except Exception as exc:
            # The environment state is unknown after a raised exception; reuse
            # the last good observation so the episode can be traced and the
            # model can react on the next step.
            fallback = (
                self._last_observation.model_copy(deep=True)
                if self._last_observation
                else Observation(goal="", url="")
            )
            fallback.last_action = action
            fallback.last_action_error = str(exc)
            env_step = EnvironmentStep(
                observation=fallback,
                action_error=str(exc),
            )
        self._last_observation = env_step.observation
        return env_step

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
