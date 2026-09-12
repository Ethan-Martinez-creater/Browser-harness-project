"""Contract tests for BenchmarkRunner (R6) and the secret contract (R3)."""

import json
from pathlib import Path

import pytest

from web_harness.agents.baseline import BaselineAgent
from web_harness.config.loader import HarnessConfig
from web_harness.core.errors import ConfigError
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.evaluation.benchmark_runner import run_benchmark, sanitize_config
from web_harness.models.mock import MockModelAdapter

SENTINEL = "sk-PHASE0-SENTINEL-SECRET"


def make_config(tmp_path: Path, **overrides) -> HarnessConfig:
    data = {
        "model": {"provider": "mock", "mock_actions": ["click(bid='1')"]},
        "agent": {"type": "baseline", "max_history_steps": 4},
        "runtime": {"max_steps": 5},
        "trace": {"root_dir": str(tmp_path / "runs")},
        "environment": {"bootstrap_action": None},
    }
    data.update(overrides)
    return HarnessConfig(data)


def fake_env_factory():
    return FakeEnvironment(
        reset_observation=make_fake_observation(url="http://fake.local/start"),
        script=[{"reward": 1.0, "terminated": True}],
    )


def agent_factory():
    return BaselineAgent(model_adapter=MockModelAdapter(["click(bid='1')"]))


def test_benchmark_runner_contract(tmp_path):
    cfg = make_config(tmp_path)
    exp_dir, results = run_benchmark(
        cfg,
        benchmark_name="fake",
        tasks=["t1", "t2"],
        seeds=[0, 1],
        experiments_root=tmp_path / "experiments",
        environment_factory=fake_env_factory,
        agent_factory=agent_factory,
    )

    # 1. task x seed episode count
    assert len(results) == 4

    # 2. every episode has a distinct run_id
    assert len({r.run_id for r in results}) == 4

    # 3. manifest.json is strict JSON
    manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_episodes"] == 4

    # 4. summary.json is strict JSON
    summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_id"] == exp_dir.name

    # 5. episodes.csv row count
    csv_lines = [
        line
        for line in (exp_dir / "episodes.csv").read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(csv_lines) == 5  # header + 4 episodes

    # 6. aggregate metrics are consistent with episode rows
    assert summary["aggregate"]["num_episodes"] == 4
    assert summary["aggregate"]["num_successes"] == 4
    assert summary["aggregate"]["success_rate"] == 1.0
    total_in = sum(r.input_tokens for r in results)
    assert summary["aggregate"]["total_input_tokens"] == total_in
    total_steps = sum(r.num_steps for r in results)
    assert summary["aggregate"]["mean_steps"] == total_steps / 4

    # 7. run_ids.txt matches
    run_ids = [
        line
        for line in (exp_dir / "run_ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sorted(run_ids) == sorted(r.run_id for r in results)

    # 8. no secret anywhere in the experiment artifacts
    for path in exp_dir.rglob("*"):
        if path.is_file():
            assert SENTINEL not in path.read_text(encoding="utf-8", errors="ignore")

    # 9. a failing episode must not lose sibling results
    def failing_env_factory():
        class BrokenEnv(FakeEnvironment):
            def reset(self, task):
                raise RuntimeError("boom")

        return BrokenEnv()

    cfg2 = make_config(tmp_path)
    exp_dir2, results2 = run_benchmark(
        cfg2,
        benchmark_name="fake",
        tasks=["t1", "t2"],
        seeds=[0],
        experiments_root=tmp_path / "experiments",
        environment_factory=failing_env_factory,
        agent_factory=agent_factory,
    )
    assert len(results2) == 2  # both episodes recorded (as errors)
    assert all(r.status.value == "error" for r in results2)
    summary2 = json.loads((exp_dir2 / "summary.json").read_text(encoding="utf-8"))
    assert summary2["aggregate"]["num_episodes"] == 2


def test_config_rejects_literal_api_key():
    with pytest.raises(ConfigError):
        HarnessConfig(
            {"model": {"provider": "openai_compatible", "model": "m", "api_key": SENTINEL}}
        )


def test_config_rejects_nested_secret_keys():
    with pytest.raises(ConfigError):
        HarnessConfig(
            {
                "model": {"provider": "mock"},
                "credentials": {"password": "hunter2"},
            }
        )


def test_sanitize_config_strips_secret_keys():
    cleaned = sanitize_config({"model": {"api_key": "x", "model": "m"}, "runtime": {}})
    assert "api_key" not in cleaned["model"]
    assert cleaned["model"]["model"] == "m"


def test_no_sentinel_leak_across_runs_and_experiments(tmp_path, monkeypatch):
    """Even with a secret present in the environment, no trace/config file
    under runs/ or experiments/ may ever contain it (R3 sentinel test)."""
    monkeypatch.setenv("MODEL_API_KEY", SENTINEL)
    monkeypatch.setenv("MODEL_BASE_URL", "https://sentinel-endpoint.example")

    cfg = make_config(tmp_path)
    exp_dir, results = run_benchmark(
        cfg,
        benchmark_name="fake",
        tasks=["t1"],
        seeds=[0],
        experiments_root=tmp_path / "experiments",
        environment_factory=fake_env_factory,
        agent_factory=agent_factory,
    )
    assert len(results) == 1

    scanned = 0
    for root in (tmp_path / "runs", tmp_path / "experiments"):
        for path in Path(root).rglob("*"):
            if path.is_file():
                content = path.read_text(encoding="utf-8", errors="ignore")
                assert SENTINEL not in content, f"sentinel leaked into {path}"
                scanned += 1
    assert scanned > 0
