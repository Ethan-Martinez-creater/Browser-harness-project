"""BrowserGym adapter.

This is the ONLY module allowed to import BrowserGym/Gymnasium. Everything
above the environment layer works with `TaskSpec`, `Observation` and
`EnvironmentStep` exclusively.
"""

from __future__ import annotations

import os
from pathlib import Path

from web_harness.core.errors import EnvironmentInitError
from web_harness.core.models import EnvironmentStep, Observation, TaskSpec
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
        wait_ms_after_reset: int = 0,
    ):
        self._normalizer = ObservationNormalizer(
            observation_char_limit=observation_char_limit
        )
        self._headless = headless
        self._save_screenshots = save_screenshots
        self._artifact_dir = artifact_dir
        self._miniwob_url = miniwob_url
        self._wait_ms_after_reset = wait_ms_after_reset
        self._env = None
        self._last_observation: Observation | None = None
        self._closed = False

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def gym_task_id(task: TaskSpec) -> str:
        """Map TaskSpec to a BrowserGym task id, e.g. browsergym/miniwob.click-test."""
        return f"browsergym/{task.benchmark}.{task.task_id}"

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

            env = gym.make(self.gym_task_id(task), headless=self._headless)
            self._env = env
            raw_obs, _info = env.reset(seed=task.seed)
            # The first frame may render with an empty axtree; one internal
            # noop refreshes the observation. This does not consume a model step.
            raw_obs, _r, _t, _tr, _i = env.step("noop()")
            if self._wait_ms_after_reset:
                import time

                time.sleep(self._wait_ms_after_reset / 1000.0)
        except Exception as exc:
            self.close()
            raise EnvironmentInitError(
                f"failed to init {self.gym_task_id(task)}: {exc}"
            ) from exc
        obs = self._normalize(raw_obs)
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
