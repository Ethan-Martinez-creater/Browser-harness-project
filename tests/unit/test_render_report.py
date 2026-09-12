"""Tests for the phase-aware benchmark report renderer (R3)."""

import importlib.util
import json
from pathlib import Path

import pytest

RENDERER_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "render_benchmark_report.py"
)


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_benchmark_report", RENDERER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_summary() -> dict:
    return {
        "experiment_id": "miniwob-smoke-test",
        "config": {
            "benchmark": "miniwob",
            "tasks": ["click-test", "click-button"],
            "seeds": [0],
            "model": {"provider": "openai_compatible", "model": "m", "temperature": 0.0},
            "agent": {"type": "baseline"},
            "runtime": {"max_steps": 20},
            "environment": {"bootstrap_action": "noop(wait_ms=500)"},
            "reliability": {
                "enabled": True,
                "verification": {
                    "enabled": True,
                    "mode": "shadow",
                    "loop": {"consecutive_threshold": 2},
                },
            },
        },
        "aggregate": {
            "num_episodes": 2,
            "num_successes": 2,
            "success_rate": 1.0,
            "mean_reward": 1.0,
            "mean_steps": 1.5,
            "median_steps": 1.5,
            "mean_duration_s": 10.0,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "action_error_rate": 0.0,
            "verification_count": 3,
            "verifications_with_signal": 1,
            "verification_signal_rate": 1 / 3,
            "mean_failure_signals_per_verification": 0.5,
            "failure_signal_count": 1,
            "episodes_with_failure_signal": 1,
            "failure_kind_counts": {"ACTION_ERROR": 1},
        },
        "episodes": [],
    }


def make_episodes() -> list[dict]:
    return [
        {
            "task_id": "click-test",
            "seed": 0,
            "status": "success",
            "success": True,
            "final_reward": 1.0,
            "steps": 1,
            "duration_s": 9.0,
            "input_tokens": 60,
            "output_tokens": 30,
            "action_error_count": 0,
            "verification_count": 1,
            "verifications_with_signal": 0,
            "failure_signal_count": 0,
            "error_type": "TASK_TERMINATED",
        },
        {
            "task_id": "click-button",
            "seed": 0,
            "status": "success",
            "success": True,
            "final_reward": 1.0,
            "steps": 2,
            "duration_s": 11.0,
            "input_tokens": 40,
            "output_tokens": 20,
            "action_error_count": 1,
            "verification_count": 2,
            "verifications_with_signal": 1,
            "failure_signal_count": 1,
            "error_type": "TASK_TERMINATED",
        },
    ]


def test_phase1a_profile_renders_reliability_metrics():
    r = load_renderer()
    report = r.render(make_summary(), make_episodes(), "phase1a")
    assert report.startswith("# Phase 1A Verification (Shadow Mode) Report")
    assert "verification_count | 3" in report
    assert "verification_signal_rate | 0.333" in report
    assert "mean_failure_signals_per_verification | 0.500" in report
    assert "failure_signal_count | 1" in report
    assert "episodes_with_failure_signal | 1" in report
    assert "| ACTION_ERROR | 1 |" in report
    assert "verification mode: shadow" in report
    # no Phase 0 scope text may leak into a Phase 1A report
    assert "Phase 0" not in report


def test_phase0_profile_unchanged_shape():
    r = load_renderer()
    report = r.render(make_summary(), make_episodes(), "phase0")
    assert report.startswith("# Phase 0 MiniWoB Smoke Benchmark Report")
    # phase0 profile has no reliability section
    assert "Verification layer metrics" not in report


def test_rendered_report_matches_committed_report(tmp_path):
    """The committed Phase 1A report must be rebuildable from machine results."""
    repo_root = Path(__file__).resolve().parents[2]
    summary_path = repo_root / "reports" / "phase1" / "phase1a_verifier_summary.json"
    episodes_path = repo_root / "reports" / "phase1" / "phase1a_verifier_episodes.csv"
    committed = repo_root / "reports" / "phase1" / "phase1a_verifier_report.md"
    notes_path = repo_root / "reports" / "phase1" / "phase1a_verifier_notes.md"
    if not summary_path.exists():
        pytest.skip("Phase 1A summary not committed")

    r = load_renderer()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = r.load_episodes(episodes_path)
    rebuilt = r.render(summary, episodes, "phase1a")
    if notes_path.exists():
        rebuilt += "\n" + notes_path.read_text(encoding="utf-8").strip() + "\n"
    assert rebuilt == committed.read_text(encoding="utf-8")
