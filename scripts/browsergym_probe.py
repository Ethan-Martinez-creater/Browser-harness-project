"""Phase 0 environment probe: verify BrowserGym + MiniWoB + Chromium work end to end.

Runs `browsergym/miniwob.click-test` with a scripted action extracted from the
accessibility tree, and reports what worked. Not a performance benchmark.

Usage:
    uv run python scripts/browsergym_probe.py
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import re
import sys

RESULTS: list[tuple[str, str]] = []


def record(label: str, value: str) -> None:
    RESULTS.append((label, value))
    print(f"[probe] {label}: {value}")


def find_clickable_bid(axtree: str) -> str | None:
    """Return the bid of the first button/checkbox/link in the axtree text."""
    for line in axtree.splitlines():
        m = re.search(r"\[([^\]\s]+)\]\s*(button|link|checkbox|tab)", line)
        if m:
            return m.group(1)
    return None


def axtree_to_str(obs: dict) -> str:
    from browsergym.utils.obs import flatten_axtree_to_str

    return flatten_axtree_to_str(obs["axtree_object"]) or ""


def main() -> int:
    import browsergym.miniwob  # noqa: F401
    import gymnasium as gym

    record("python", sys.version.split()[0])
    for pkg in (
        "browsergym-core",
        "browsergym-miniwob",
        "gymnasium",
        "playwright",
        "openai",
        "pydantic",
    ):
        try:
            record(pkg, md.version(pkg))
        except Exception:
            record(pkg, "NOT INSTALLED")
    record("platform", platform.platform())

    env = gym.make("browsergym/miniwob.click-test", headless=True)
    try:
        obs, info = env.reset(seed=0)
        record("reset", "OK")
        record("observation keys", ", ".join(sorted(obs.keys())))
        record("goal", (obs.get("goal") or "")[:120])
        # axtree needs one rendered frame; a noop step refreshes the observation
        obs, _, _, _, _ = env.step("noop()")
        axtree: str = axtree_to_str(obs)
        record("axtree head", axtree[:200].replace("\n", " | "))

        bid = find_clickable_bid(axtree)
        record("found bid", bid or "NONE")

        action = f"click(bid={bid!r})" if bid else "noop()"
        obs2, reward, terminated, truncated, info2 = env.step(action)
        record("action", action)
        record("last_action_error", (obs2.get("last_action_error") or "none")[:200])
        record("reward", str(reward))
        record("terminated", str(terminated))
        record("truncated", str(truncated))
    finally:
        env.close()
        record("close", "OK")

    ok = (
        bid is not None
        and not obs2.get("last_action_error")
        and (terminated or reward == 1)
    )
    record("PROBE RESULT", "SUCCESS" if ok else "PARTIAL/FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
