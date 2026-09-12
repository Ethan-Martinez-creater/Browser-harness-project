"""Episode and aggregate metrics.

Phase 0 metrics are deliberately limited to what a baseline needs. Later
phases add verification/recovery/checkpoint metrics on top of the same
aggregation layer.
"""

from __future__ import annotations

import statistics

from web_harness.core.models import RunResult


def episode_metrics(result: RunResult) -> dict:
    return {
        "run_id": result.run_id,
        "benchmark": result.task_spec.benchmark,
        "task_id": result.task_spec.task_id,
        "seed": result.task_spec.seed,
        "status": result.status.value,
        "success": result.success,
        "final_reward": result.final_reward,
        "steps": result.num_steps,
        "duration_s": round(result.duration_s, 3),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "action_error_count": result.action_error_count,
        "error_type": result.error_type.value if result.error_type else None,
        "verification_count": result.verification_count,
        "verifications_with_signal": result.verifications_with_signal,
        "failure_signal_count": result.failure_signal_count,
        "recovery_count": result.recovery_count,
        "recovery_success_count": result.recovery_success_count,
        "recovery_failed_count": result.recovery_failed_count,
        "recovery_unresolved_count": result.recovery_unresolved_count,
        "recovery_environment_actions": result.recovery_environment_actions,
        "recovery_latency_s": round(result.recovery_latency_s, 3),
        "blocked_action_redecision_count": result.blocked_action_redecision_count,
        "retry_count": result.retry_count,
        "retry_cycle_count": result.retry_cycle_count,
        "retry_success_count": result.retry_success_count,
        "retry_exhausted_count": result.retry_exhausted_count,
        "extra_model_calls": result.extra_model_calls,
        "retry_input_tokens": result.retry_input_tokens,
        "retry_output_tokens": result.retry_output_tokens,
        "retry_latency_s": round(result.retry_latency_s, 3),
    }


def aggregate_metrics(results: list[RunResult]) -> dict:
    if not results:
        return {"num_episodes": 0}
    successes = [r for r in results if r.success]
    steps = [r.num_steps for r in results]
    durations = [r.duration_s for r in results]
    action_errors = [r.action_error_count for r in results]
    verification_counts = [r.verification_count for r in results]
    signal_counts = [r.failure_signal_count for r in results]
    kind_counts: dict[str, int] = {}
    for r in results:
        for kind, count in r.failure_kind_counts.items():
            kind_counts[kind] = kind_counts.get(kind, 0) + count
    total_verifications = sum(verification_counts)
    total_with_signal = sum(r.verifications_with_signal for r in results)
    total_signals = sum(signal_counts)
    return {
        "num_episodes": len(results),
        "success_rate": len(successes) / len(results),
        "num_successes": len(successes),
        "mean_reward": statistics.fmean(r.final_reward for r in results),
        "mean_steps": statistics.fmean(steps),
        "median_steps": statistics.median(steps),
        "mean_duration_s": statistics.fmean(durations),
        "total_input_tokens": sum(r.input_tokens for r in results),
        "total_output_tokens": sum(r.output_tokens for r in results),
        "action_error_rate": (
            statistics.fmean(1.0 if e > 0 else 0.0 for e in action_errors)
        ),
        # reliability (Phase 1A verification, shadow mode)
        "verification_count": total_verifications,
        "verifications_with_signal": total_with_signal,
        "verification_signal_rate": (
            total_with_signal / total_verifications if total_verifications else 0.0
        ),
        "mean_failure_signals_per_verification": (
            total_signals / total_verifications if total_verifications else 0.0
        ),
        "failure_signal_count": total_signals,
        "episodes_with_failure_signal": sum(1 for c in signal_counts if c > 0),
        "failure_kind_counts": dict(sorted(kind_counts.items())),
        # reliability (Phase 1B controlled retry). Distinctions: an ATTEMPT
        # is one model call, a CYCLE is one decision point (>= 1 retry), an
        # EPISODE bundles many cycles. retry_success_rate uses cycles as the
        # denominator (per the retry_success definition).
        "total_retry_count": sum(r.retry_count for r in results),
        "retry_cycle_count": sum(r.retry_cycle_count for r in results),
        "episodes_with_retry": sum(1 for r in results if r.retry_count > 0),
        "retry_success_count": sum(r.retry_success_count for r in results),
        "retry_exhausted_count": sum(r.retry_exhausted_count for r in results),
        "retry_success_rate": (
            (lambda s, c: s / c if c else 0.0)(
                sum(r.retry_success_count for r in results),
                sum(r.retry_cycle_count for r in results),
            )
        ),
        "total_extra_model_calls": sum(r.extra_model_calls for r in results),
        "total_retry_input_tokens": sum(r.retry_input_tokens for r in results),
        "total_retry_output_tokens": sum(r.retry_output_tokens for r in results),
        "total_retry_latency_s": round(
            sum(r.retry_latency_s for r in results), 3
        ),
        # reliability (Phase 1C recovery). Invariant: success + failed +
        # unresolved == total_recovery_count (no phantom outcomes).
        "total_recovery_count": sum(r.recovery_count for r in results),
        "episodes_with_recovery": sum(1 for r in results if r.recovery_count > 0),
        "recovery_success_count": sum(r.recovery_success_count for r in results),
        "recovery_failed_count": sum(r.recovery_failed_count for r in results),
        "recovery_unresolved_count": sum(
            r.recovery_unresolved_count for r in results
        ),
        "recovered_episode_count": sum(1 for r in results if r.recovered_episode),
        "recovery_environment_actions": sum(
            r.recovery_environment_actions for r in results
        ),
        "blocked_action_redecision_count": sum(
            r.blocked_action_redecision_count for r in results
        ),
        "total_recovery_latency_s": round(
            sum(r.recovery_latency_s for r in results), 3
        ),
    }
