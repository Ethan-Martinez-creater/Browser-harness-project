"""Remediation tests: config semantics (B2) and canonical config provenance (B1)."""

import json
from pathlib import Path

import pytest

from web_harness.agents.baseline import BaselineAgent
from web_harness.config.loader import HarnessConfig, config_hash
from web_harness.core.errors import ConfigError
from web_harness.env.fake import FakeEnvironment, make_fake_observation
from web_harness.evaluation.benchmark_runner import run_benchmark
from web_harness.models.mock import MockModelAdapter

SENTINEL_ENV = FakeEnvironment(
    reset_observation=make_fake_observation(url="http://fake.local/start"),
    script=[{"reward": 1.0, "terminated": True}],
)


def fake_env_factory():
    return FakeEnvironment(
        reset_observation=make_fake_observation(url="http://fake.local/start"),
        script=[{"reward": 1.0, "terminated": True}],
    )


def agent_factory():
    return BaselineAgent(model_adapter=MockModelAdapter(["click(bid='1')"]))


def base_data(reliability: dict | None = None) -> dict:
    return {
        "model": {"provider": "mock", "mock_actions": ["click(bid='1')"]},
        "agent": {"type": "baseline"},
        "runtime": {"max_steps": 5},
        "trace": {"root_dir": "runs"},
        "environment": {},
        "reliability": reliability if reliability is not None else {"enabled": False},
    }


# -- B2: verification switch semantics ----------------------------------------


def test_reliability_disabled_means_no_verification():
    cfg = HarnessConfig(base_data({"enabled": False, "verification": {"enabled": True}}))
    assert cfg.reliability_enabled is False
    assert cfg.verification_enabled is False


def test_verification_disabled_overrides_reliability():
    cfg = HarnessConfig(
        base_data({"enabled": True, "verification": {"enabled": False, "mode": "shadow"}})
    )
    assert cfg.reliability_enabled is True
    assert cfg.verification_enabled is False


def test_verification_shadow_enabled():
    cfg = HarnessConfig(
        base_data({"enabled": True, "verification": {"enabled": True, "mode": "shadow"}})
    )
    assert cfg.verification_enabled is True
    assert cfg.verification_mode == "shadow"


def test_active_mode_rejected_in_phase1a():
    with pytest.raises(ConfigError, match="active"):
        HarnessConfig(
            base_data({"enabled": True, "verification": {"enabled": True, "mode": "active"}})
        )


def test_unknown_mode_rejected():
    with pytest.raises(ConfigError, match="unknown verification mode"):
        HarnessConfig(
            base_data({"enabled": True, "verification": {"enabled": True, "mode": "turbo"}})
        )


def test_benchmark_runner_wiring_disabled_vs_shadow(tmp_path):
    """Config-level wiring: the runner must create the verifier only when
    reliability AND verification are enabled (not via manual injection)."""
    # reliability off -> zero verifications
    cfg_off = HarnessConfig(base_data({"enabled": False}))
    _, results_off = run_benchmark(
        cfg_off,
        benchmark_name="fake",
        tasks=["t1"],
        seeds=[0],
        experiments_root=tmp_path / "exp-off",
        environment_factory=fake_env_factory,
        agent_factory=agent_factory,
    )
    assert results_off[0].verification_count == 0

    # reliability on but verification off -> zero verifications
    cfg_verif_off = HarnessConfig(
        base_data({"enabled": True, "verification": {"enabled": False}})
    )
    _, results_voff = run_benchmark(
        cfg_verif_off,
        benchmark_name="fake",
        tasks=["t1"],
        seeds=[0],
        experiments_root=tmp_path / "exp-verif-off",
        environment_factory=fake_env_factory,
        agent_factory=agent_factory,
    )
    assert results_voff[0].verification_count == 0

    # reliability on + verification on + shadow -> one verification per step
    cfg_shadow = HarnessConfig(
        base_data({"enabled": True, "verification": {"enabled": True, "mode": "shadow"}})
    )
    exp_dir, results_on = run_benchmark(
        cfg_shadow,
        benchmark_name="fake",
        tasks=["t1"],
        seeds=[0],
        experiments_root=tmp_path / "exp-shadow",
        environment_factory=fake_env_factory,
        agent_factory=agent_factory,
    )
    assert results_on[0].verification_count == results_on[0].num_steps == 1
    # event mode reflects the real configured mode
    run_id = results_on[0].run_id
    from web_harness.observability.trace import TraceRecorder

    events = TraceRecorder.read_events(Path(cfg_shadow.trace_root) / run_id)
    assert events and events[0].data["mode"] == "shadow"


# -- B1: canonical config provenance ------------------------------------------


def test_canonical_config_persisted_with_reliability(tmp_path):
    cfg = HarnessConfig(
        base_data({"enabled": True, "verification": {"enabled": True, "mode": "shadow"}})
    )
    exp_dir, results = run_benchmark(
        cfg,
        benchmark_name="fake",
        tasks=["t1"],
        seeds=[0],
        experiments_root=tmp_path / "exp",
        environment_factory=fake_env_factory,
        agent_factory=agent_factory,
    )
    persisted = yaml_safe_load(exp_dir / "config.yaml")
    assert "reliability" in persisted
    assert persisted["reliability"]["enabled"] is True

    summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
    assert "reliability" in summary["config"]

    exp_manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    canonical_hash = config_hash(persisted)
    assert exp_manifest["config_hash"] == canonical_hash

    # run-level manifest uses the SAME canonical hash
    run_manifest = json.loads(
        (Path(cfg.trace_root) / results[0].run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_manifest["config_hash"] == canonical_hash


def test_reliability_changes_change_config_hash(tmp_path):
    def hash_for(reliability: dict) -> str:
        cfg = HarnessConfig(base_data(reliability))
        exp_dir, _ = run_benchmark(
            cfg,
            benchmark_name="fake",
            tasks=["t1"],
            seeds=[0],
            experiments_root=tmp_path / "exp",
            environment_factory=fake_env_factory,
            agent_factory=agent_factory,
        )
        persisted = yaml_safe_load(exp_dir / "config.yaml")
        return config_hash(persisted)

    h_shadow = hash_for({"enabled": True, "verification": {"enabled": True, "mode": "shadow"}})
    h_verif_off = hash_for({"enabled": True, "verification": {"enabled": False, "mode": "shadow"}})
    h_threshold = hash_for(
        {
            "enabled": True,
            "verification": {
                "enabled": True,
                "mode": "shadow",
                "loop": {"consecutive_threshold": 3},
            },
        }
    )
    assert len({h_shadow, h_verif_off, h_threshold}) == 3


def yaml_safe_load(path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
