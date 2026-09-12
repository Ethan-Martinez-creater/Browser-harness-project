"""Unit tests for retry metrics aggregation (Phase 1B)."""

from web_harness.core.models import RunResult, RunStatus, TaskSpec
from web_harness.evaluation.metrics import aggregate_metrics, episode_metrics


def make_result(run_id: str, *, retries: int = 0, retry_cycles: int = 0,
                retry_success: int = 0, exhausted: int = 0, extra_calls: int = 0,
                retry_in: int = 0, retry_out: int = 0, retry_latency: float = 0.0) -> RunResult:
    return RunResult(
        run_id=run_id,
        task_spec=TaskSpec(benchmark="miniwob", task_id="t"),
        status=RunStatus.SUCCESS,
        success=True,
        final_reward=1.0,
        num_steps=2,
        duration_s=5.0,
        input_tokens=100 + retry_in,
        output_tokens=20 + retry_out,
        retry_count=retries,
        retry_cycle_count=retry_cycles if retry_cycles else (1 if retries else 0),
        retry_success_count=retry_success,
        retry_exhausted_count=exhausted,
        extra_model_calls=extra_calls,
        retry_input_tokens=retry_in,
        retry_output_tokens=retry_out,
        retry_latency_s=retry_latency,
    )


def test_episode_metrics_include_retry_fields():
    m = episode_metrics(make_result("r1", retries=2, retry_success=1, extra_calls=2,
                                    retry_in=50, retry_out=10, retry_latency=1.5))
    assert m["retry_count"] == 2
    assert m["retry_success_count"] == 1
    assert m["extra_model_calls"] == 2
    assert m["retry_input_tokens"] == 50
    assert m["retry_output_tokens"] == 10
    assert m["retry_latency_s"] == 1.5


def test_aggregate_retry_metrics():
    results = [
        # one decision cycle with 2 retry attempts, eventually successful
        make_result("r1", retries=2, retry_cycles=1, retry_success=1, extra_calls=2,
                    retry_in=50, retry_out=10, retry_latency=1.0),
        make_result("r2", retries=0),
        # one decision cycle with 1 retry, exhausted
        make_result("r3", retries=1, retry_cycles=1, exhausted=1, extra_calls=1,
                    retry_in=30, retry_out=5, retry_latency=0.5),
    ]
    agg = aggregate_metrics(results)
    assert agg["total_retry_count"] == 3  # retry ATTEMPTS
    assert agg["retry_cycle_count"] == 2  # decision cycles with >= 1 retry
    assert agg["episodes_with_retry"] == 2
    assert agg["retry_success_count"] == 1
    assert agg["retry_exhausted_count"] == 1
    # retry_success_rate denominator = retry-bearing CYCLES (1/2), not attempts
    assert abs(agg["retry_success_rate"] - 1 / 2) < 1e-9
    assert agg["total_extra_model_calls"] == 3
    assert agg["total_retry_input_tokens"] == 80
    assert agg["total_retry_output_tokens"] == 15
    assert agg["total_retry_latency_s"] == 1.5


def test_retry_tokens_are_subset_of_total_tokens():
    r = make_result("r1", retries=1, retry_in=40, retry_out=8)
    assert r.input_tokens >= r.retry_input_tokens
    assert r.output_tokens >= r.retry_output_tokens
