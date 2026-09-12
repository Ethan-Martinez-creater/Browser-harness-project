"""Render benchmark Markdown reports from machine results (phase-aware).

Usage:
    uv run python scripts/render_benchmark_report.py \
        --summary experiments/<id>/summary.json \
        --episodes experiments/<id>/episodes.csv \
        --output report.md \
        [--profile phase0|phase1a] \
        [--notes path/to/notes.md]

All benchmark numbers (per-episode table, aggregate metrics, verification
statistics) are generated from summary.json / episodes.csv — never
hand-written. The profile controls title, scope warning and the reliability
metrics section so committed reports can be rebuilt byte-identically from
their machine results.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
from pathlib import Path


def fmt_num(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def is_truthy(value) -> bool:
    """Tolerate bool/str/int forms coming from CSV."""
    return str(value).strip().lower() in ("true", "1", "yes")


PROFILES = {
    "phase0": {
        "title": "# Phase 0 MiniWoB Smoke Benchmark Report",
        "scope": [
            "> **Scope warning**: this is the Phase 0 engineering smoke run.",
            "It validates the measurement system (runtime, adapter, tracing,",
            "benchmark runner, metrics) and establishes the baseline reference.",
            "It is not a model-capability conclusion.",
        ],
        "reliability": False,
        "retry": False,
    },
    "phase1a": {
        "title": "# Phase 1A Verification (Shadow Mode) Report",
        "scope": [
            "> **Scope warning**: this is the Phase 1A engineering smoke run.",
            "It validates the failure-observation layer (deterministic",
            "fingerprinting, detectors, verification events) in shadow mode and",
            "establishes the verification reference. It is not a",
            "model-capability conclusion, and shadow mode does not change the",
            "control flow.",
        ],
        "reliability": True,
        "retry": False,
    },
    "phase1b": {
        "title": "# Phase 1B Controlled Retry Report",
        "scope": [
            "> **Scope warning**: this is the Phase 1B engineering smoke run.",
            "It validates controlled model-side retry (API retry with fixed",
            "backoff, one parse-repair retry, budget limits) on the natural",
            "MiniWoB workload. It is not a model-capability conclusion, and",
            "browser actions are never retried.",
        ],
        "reliability": True,
        "retry": True,
    },
    "phase1c": {
        "title": "# Phase 1C Recovery Policy Report",
        "scope": [
            "> **Scope warning**: this is the Phase 1C engineering smoke run.",
            "It validates environment-side recovery (rule-based policy over",
            "verified failure signals, budget-bounded WAIT_AND_REOBSERVE /",
            "REDECIDE_WITH_FEEDBACK / BLOCK_REPEATED_ACTION) on the natural",
            "MiniWoB workload. It is not a model-capability conclusion, and",
            "no browser action is ever blindly retried.",
        ],
        "reliability": True,
        "retry": True,
        "recovery": True,
    },
}


def render_reliability_metrics(summary: dict) -> list[str]:
    agg = summary["aggregate"]
    config = summary.get("config", {})
    reliability_cfg = config.get("reliability") or {}
    verification_cfg = reliability_cfg.get("verification") or {}
    loop_cfg = verification_cfg.get("loop") or {}
    lines = [
        "## Verification layer metrics (Phase 1A)",
        "",
        f"- reliability enabled: {reliability_cfg.get('enabled', False)}",
        f"- verification enabled: {verification_cfg.get('enabled', False)}",
        f"- verification mode: {verification_cfg.get('mode', 'shadow')}",
        f"- no-progress detector: "
        f"{(verification_cfg.get('no_progress') or {}).get('enabled', True)}",
        f"- loop detector: {loop_cfg.get('enabled', True)} "
        f"(consecutive_threshold={loop_cfg.get('consecutive_threshold', 2)})",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| verification_count | {agg.get('verification_count', 0)} |",
        f"| verifications_with_signal | {agg.get('verifications_with_signal', 0)} |",
        f"| verification_signal_rate | {fmt_num(agg.get('verification_signal_rate', 0.0))} |",
        f"| mean_failure_signals_per_verification | "
        f"{fmt_num(agg.get('mean_failure_signals_per_verification', 0.0))} |",
        f"| failure_signal_count | {agg.get('failure_signal_count', 0)} |",
        f"| episodes_with_failure_signal | {agg.get('episodes_with_failure_signal', 0)} |",
        "",
        "Failure kinds:",
        "",
    ]
    kind_counts = agg.get("failure_kind_counts") or {}
    if kind_counts:
        lines.append("| failure kind | count |")
        lines.append("|---|---:|")
        for kind, count in kind_counts.items():
            lines.append(f"| {kind} | {count} |")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append(
        "Shadow mode guarantees: verification_count == total agent steps "
        "(one verification per step), zero extra model calls, zero extra "
        "environment actions. Detection only — recovery and replanning "
        "do not exist in this phase."
    )
    lines.append("")
    return lines


def render_retry_metrics(summary: dict) -> list[str]:
    agg = summary["aggregate"]
    config = summary.get("config", {})
    retry_cfg = ((config.get("reliability") or {}).get("retry")) or {}
    api_cfg = retry_cfg.get("model_api") or {}
    output_cfg = retry_cfg.get("model_output") or {}
    budget_cfg = (config.get("reliability") or {}).get("budget") or {}
    lines = [
        "## Retry layer metrics (Phase 1B)",
        "",
        f"- retry enabled: {retry_cfg.get('enabled', False)}",
        f"- model_api.max_retries: {api_cfg.get('max_retries', 2)} "
        f"(1 initial attempt + N retries), backoff_ms={api_cfg.get('backoff_ms', [500, 1000])}",
        f"- model_output.max_retries (format repair): {output_cfg.get('max_retries', 1)}",
        f"- budget.max_extra_model_calls_per_episode: "
        f"{budget_cfg.get('max_extra_model_calls_per_episode', 6)}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total_retry_count | {agg.get('total_retry_count', 0)} |",
        f"| retry_cycle_count | {agg.get('retry_cycle_count', 0)} |",
        f"| episodes_with_retry | {agg.get('episodes_with_retry', 0)} |",
        f"| retry_success_count | {agg.get('retry_success_count', 0)} |",
        f"| retry_exhausted_count | {agg.get('retry_exhausted_count', 0)} |",
        f"| retry_success_rate | {fmt_num(agg.get('retry_success_rate', 0.0))} |",
        f"| total_extra_model_calls | {agg.get('total_extra_model_calls', 0)} |",
        f"| total_retry_input_tokens | {agg.get('total_retry_input_tokens', 0)} |",
        f"| total_retry_output_tokens | {agg.get('total_retry_output_tokens', 0)} |",
        f"| total_retry_latency_s | {fmt_num(agg.get('total_retry_latency_s', 0.0))} |",
        "",
        "Retry scope guarantees: only MODEL_API_ERROR and MODEL_OUTPUT_PARSE_ERROR",
        "are retried; browser actions are never retried; retries never add an",
        "Agent step (one Agent action = one StepRecord, retries live in",
        "events.jsonl); tokens of failed attempts are included in episode totals.",
        "",
    ]
    return lines


def render_recovery_metrics(summary: dict) -> list[str]:
    agg = summary["aggregate"]
    config = summary.get("config", {})
    reliability_cfg = config.get("reliability") or {}
    recovery_cfg = reliability_cfg.get("recovery") or {}
    budget_cfg = reliability_cfg.get("budget") or {}
    lines = [
        "## Recovery layer metrics (Phase 1C)",
        "",
        f"- recovery enabled: {recovery_cfg.get('enabled', False)}",
        f"- recovery.max_recoveries_per_episode: "
        f"{recovery_cfg.get('max_recoveries_per_episode', 3)} "
        f"(canonical recovery budget path)",
        f"- recovery.wait_ms (WAIT_AND_REOBSERVE noop wait): "
        f"{recovery_cfg.get('wait_ms', 500)}",
        f"- recovery.block_steps (REDECIDE_WITH_FEEDBACK block): "
        f"{recovery_cfg.get('block_steps', 1)}",
        f"- budget.max_extra_model_calls_per_episode: "
        f"{budget_cfg.get('max_extra_model_calls_per_episode', 6)}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total_recovery_count | {agg.get('total_recovery_count', 0)} |",
        f"| recovery_success_count | {agg.get('recovery_success_count', 0)} |",
        f"| recovery_failed_count | {agg.get('recovery_failed_count', 0)} |",
        f"| recovery_unresolved_count | {agg.get('recovery_unresolved_count', 0)} |",
        f"| episodes_with_recovery | {agg.get('episodes_with_recovery', 0)} |",
        f"| recovered_episode_count | {agg.get('recovered_episode_count', 0)} |",
        f"| recovery_environment_actions | "
        f"{agg.get('recovery_environment_actions', 0)} |",
        f"| blocked_action_redecision_count | "
        f"{agg.get('blocked_action_redecision_count', 0)} |",
        f"| total_recovery_latency_s | "
        f"{fmt_num(agg.get('total_recovery_latency_s', 0.0))} |",
        "",
        "Recovery scope guarantees: recovery is triggered only by the",
        "deterministic rule policy over verified failure signals; an",
        "exhausted budget never kills a PASS / single-NO_PROGRESS step; a",
        "recovery environment action (WAIT_AND_REOBSERVE noop) is never",
        "counted as an Agent step and never becomes a StepRecord; blocked",
        "actions never reach env.step (re-selections are counted separately",
        "as blocked_action_redecision_count); success + failed + unresolved",
        "sum to total_recovery_count; recovery is bounded by",
        "recovery.max_recoveries_per_episode and independent of the Phase 1B",
        "extra-model-call budget.",
        "",
    ]
    return lines


def render(summary: dict, episodes: list[dict], profile: str) -> str:
    p = PROFILES[profile]
    agg = summary["aggregate"]
    config = summary.get("config", {})
    lines: list[str] = []

    lines.append(p["title"])
    lines.append("")
    lines.append(
        f"- Experiment: `{summary['experiment_id']}` "
        f"(auto-generated from summary.json / episodes.csv)"
    )
    lines.append(
        f"- Tasks: {len(config.get('tasks', []))} × seeds "
        f"{config.get('seeds', [])}, serial"
    )
    lines.append(
        f"- Model: `{(config.get('model') or {}).get('model', '?')}` "
        f"(provider `{(config.get('model') or {}).get('provider', '?')}`, "
        f"temperature {(config.get('model') or {}).get('temperature', '?')})"
    )
    lines.append(
        f"- Agent: `{(config.get('agent') or {}).get('type', 'baseline')}`, "
        f"max_steps {(config.get('runtime') or {}).get('max_steps', '?')}"
    )
    env_cfg = config.get("environment") or {}
    lines.append(
        f"- Environment bootstrap: {env_cfg.get('bootstrap_action') or 'disabled'}"
    )
    lines.extend(p["scope"])
    lines.append("")

    lines.append("## Per-episode results")
    lines.append("")
    lines.append(
        "| task | seed | status | success | reward | steps | duration (s) "
        "| input tokens | output tokens | action errors | verifications | "
        "failure signals | error type |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for e in episodes:
        verif = e.get("verification_count", "-")
        signals = e.get("failure_signal_count", "-")
        lines.append(
            f"| {e['task_id']} | {e['seed']} | {e['status']} "
            f"| {'✅' if is_truthy(e['success']) else '❌'} | {fmt_num(e['final_reward'], 1)} "
            f"| {e['steps']} | {fmt_num(e['duration_s'])} "
            f"| {e['input_tokens']} | {e['output_tokens']} "
            f"| {e['action_error_count']} | {verif} | {signals} "
            f"| {e['error_type'] or '-'} |"
        )
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| success_rate | {fmt_num(agg['success_rate'])} "
                 f"({agg['num_successes']}/{agg['num_episodes']}) |")
    lines.append(f"| mean_reward | {fmt_num(agg['mean_reward'])} |")
    lines.append(f"| mean_steps | {fmt_num(agg['mean_steps'])} |")
    lines.append(f"| median_steps | {fmt_num(agg['median_steps'], 1)} |")
    lines.append(f"| mean_duration_s | {fmt_num(agg['mean_duration_s'])} |")
    lines.append(f"| total_input_tokens | {agg['total_input_tokens']} |")
    lines.append(f"| total_output_tokens | {agg['total_output_tokens']} |")
    lines.append(f"| action_error_rate | {fmt_num(agg['action_error_rate'])} |")
    lines.append("")
    if p["reliability"]:
        lines.extend(render_reliability_metrics(summary))
    if p.get("retry"):
        lines.extend(render_retry_metrics(summary))
    if p.get("recovery"):
        lines.extend(render_recovery_metrics(summary))
    lines.append(
        "`estimated_cost` is null by design: the harness never guesses prices "
        "without a reliable price table; token counts above are the "
        "authoritative usage record."
    )
    return "\n".join(lines) + "\n"


NUMERIC_FIELDS = {
    "final_reward",
    "duration_s",
    "steps",
    "input_tokens",
    "output_tokens",
    "action_error_count",
    "verification_count",
    "verifications_with_signal",
    "failure_signal_count",
    "seed",
    "success",
}


def load_episodes(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        episodes = list(csv.DictReader(f))
    # csv gives strings; restore numeric fields used by the renderer
    for row in episodes:
        for key in NUMERIC_FIELDS & row.keys():
            with contextlib.suppress(ValueError):
                row[key] = (
                    float(row[key]) if key == "duration_s" else int(float(row[key]))
                )
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="path to summary.json")
    parser.add_argument("--episodes", required=True, help="path to episodes.csv")
    parser.add_argument("--output", required=True, help="path to output report.md")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="phase0",
        help="report profile (title, scope warning, reliability section)",
    )
    parser.add_argument(
        "--notes",
        help="optional qualitative notes (markdown, no benchmark numbers) "
        "appended to the report",
    )
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    episodes = load_episodes(Path(args.episodes))

    report = render(summary, episodes, args.profile)
    if args.notes:
        notes = Path(args.notes).read_text(encoding="utf-8")
        report += "\n" + notes.strip() + "\n"
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
