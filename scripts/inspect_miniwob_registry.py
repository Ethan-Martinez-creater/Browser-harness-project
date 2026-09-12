"""List registered MiniWoB tasks available through BrowserGym.

Usage:
    uv run python scripts/inspect_miniwob_registry.py [name substring]
"""

from __future__ import annotations

import sys

import browsergym.miniwob  # noqa: F401  (registers tasks)


def main() -> int:
    substring = sys.argv[1] if len(sys.argv) > 1 else ""

    import gymnasium as gym

    env_specs = [
        spec.id
        for spec in gym.registry.values()
        if spec.id and spec.id.startswith("browsergym_miniwob")
    ]
    # BrowserGym also registers "browsergym/miniwob.*" aliases; list task ids directly.
    from browsergym.miniwob import ALL_MINIWOB_TASKS

    tasks = sorted(t for t in ALL_MINIWOB_TASKS if substring in t)
    print(f"miniwob tasks matching '{substring}': {len(tasks)}")
    for t in tasks:
        print(" ", t)
    print(f"gym-registered miniwob env ids: {len(env_specs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
