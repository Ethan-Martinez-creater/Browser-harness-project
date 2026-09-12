"""Integration test: real MiniWoB environment through BrowserGymAdapter.

Purpose is environment-interface validation, not model capability: a
deterministic scripted agent clicks the target extracted from the axtree.
No LLM API is involved, so this test runs fully offline (Chromium required).
"""

from pathlib import Path

import pytest

from web_harness.agents.base import AgentTurn
from web_harness.core.models import Observation, PromptBundle, StepRecord, TaskSpec
from web_harness.env.browsergym_adapter import BrowserGymAdapter
from web_harness.runtime.episode_runner import EpisodeRunner

pytestmark = pytest.mark.integration

# allow skipping in constrained environments
pytest.importorskip("browsergym.miniwob")


class ScriptedClickAgent:
    """Extracts the first button/link bid from the axtree and clicks it."""

    def __init__(self):
        self.prompts: list[PromptBundle] = []

    def decide(self, *, task, observation: Observation, history: list[StepRecord]):
        import re

        from web_harness.core.models import ActionDecision
        from web_harness.models.base import ModelOutput

        bid = None
        for line in (observation.axtree or "").splitlines():
            m = re.search(r"\[([^\]\s]+)\]\s*(button|link|checkbox|tab)", line)
            if m:
                bid = m.group(1)
                break
        action = f"click(bid={bid!r})" if bid else "noop()"
        decision = ActionDecision(action=action, short_reason="scripted click")
        return (
            AgentTurn(
                decision=decision,
                model_output=ModelOutput(decision=decision, model_name="scripted"),
            ),
            PromptBundle(system="scripted", user="scripted"),
        )


@pytest.fixture()
def miniwob_available():
    from web_harness.env.browsergym_adapter import resolve_miniwob_url

    try:
        resolve_miniwob_url(None)
    except Exception:
        pytest.skip("MiniWoB HTML source not available (set MINIWOB_URL)")


def test_miniwob_click_test_episode(tmp_path, miniwob_available):
    task = TaskSpec(benchmark="miniwob", task_id="click-test", seed=0, max_steps=5)
    env = BrowserGymAdapter(observation_char_limit=30000, headless=True)
    runner = EpisodeRunner(
        agent=ScriptedClickAgent(),
        env=env,
        trace_root=tmp_path,
        model_provider="scripted",
        model_name="scripted-click",
    )
    result = runner.run(task, run_id="it-miniwob-click-test")

    assert result.status.value == "success"
    assert result.success
    assert result.final_reward == 1.0
    assert result.num_steps >= 1

    # trace must be complete and reloadable
    run_dir = Path(result.trace_path)
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "steps.jsonl").exists()
    assert (run_dir / "result.json").exists()
    manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert "miniwob" in manifest_text and "click-test" in manifest_text
