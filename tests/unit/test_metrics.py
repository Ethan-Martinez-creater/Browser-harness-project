"""Unit tests for metrics aggregation."""

from web_harness.core.models import RunResult, RunStatus, TaskSpec
from web_harness.evaluation.metrics import aggregate_metrics, episode_metrics


def make_result(
    run_id: str, *, success: bool, steps: int, reward: float, errors: int = 0
) -> RunResult:
    return RunResult(
        run_id=run_id,
        task_spec=TaskSpec(benchmark="miniwob", task_id="t"),
        status=RunStatus.SUCCESS if success else RunStatus.FAILED,
        success=success,
        final_reward=reward,
        num_steps=steps,
        duration_s=steps * 1.5,
        input_tokens=100 * steps,
        output_tokens=10 * steps,
        action_error_count=errors,
    )


def test_episode_metrics_fields():
    m = episode_metrics(make_result("r1", success=True, steps=3, reward=1.0, errors=1))
    assert m["run_id"] == "r1"
    assert m["success"] is True
    assert m["steps"] == 3
    assert m["action_error_count"] == 1
    assert m["status"] == "success"


def test_aggregate_metrics():
    results = [
        make_result("r1", success=True, steps=2, reward=1.0),
        make_result("r2", success=False, steps=5, reward=0.0, errors=2),
        make_result("r3", success=True, steps=4, reward=1.0, errors=1),
    ]
    agg = aggregate_metrics(results)
    assert agg["num_episodes"] == 3
    assert abs(agg["success_rate"] - 2 / 3) < 1e-9
    assert agg["mean_steps"] == (2 + 5 + 4) / 3
    assert agg["median_steps"] == 4
    assert agg["total_input_tokens"] == 100 * (2 + 5 + 4)
    assert agg["total_output_tokens"] == 10 * (2 + 5 + 4)
    assert abs(agg["action_error_rate"] - 2 / 3) < 1e-9


def test_aggregate_metrics_empty():
    assert aggregate_metrics([]) == {"num_episodes": 0}
