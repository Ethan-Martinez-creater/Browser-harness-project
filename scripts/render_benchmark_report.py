"""Render the Phase 0 benchmark Markdown report from machine results.

Usage:
    uv run python scripts/render_benchmark_report.py \
        --summary experiments/<id>/summary.json \
        --episodes experiments/<id>/episodes.csv \
        --output reports/phase0/miniwob_smoke_report.md \
        [--notes path/to/notes.md]

All benchmark numbers (per-episode table, aggregate metrics, success count)
are generated from summary.json / episodes.csv — never hand-written. The
optional notes file may add qualitative observations but must not contain
manually copied benchmark values.
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


def render(summary: dict, episodes: list[dict]) -> str:
    agg = summary["aggregate"]
    lines: list[str] = []

    config = summary.get("config", {})
    lines.append("# Phase 0 MiniWoB Smoke Benchmark Report")
    lines.append("")
    lines.append(f"- Experiment: `{summary['experiment_id']}` "
                 f"(auto-generated from summary.json / episodes.csv)")
    lines.append(f"- Tasks: {len(config.get('tasks', []))} × seeds "
                 f"{config.get('seeds', [])}, serial")
    lines.append(f"- Model: `{(config.get('model') or {}).get('model', '?')}` "
                 f"(provider `{(config.get('model') or {}).get('provider', '?')}`, "
                 f"temperature {(config.get('model') or {}).get('temperature', '?')})")
    lines.append(f"- Agent: `{(config.get('agent') or {}).get('type', 'baseline')}`, "
                 f"max_steps {(config.get('runtime') or {}).get('max_steps', '?')}")
    lines.append(f"- Environment bootstrap: "
                 f"{(config.get('environment') or {}).get('bootstrap_action') or 'disabled'}")
    lines.append("")
    lines.append("> **Scope warning**: this is the Phase 0 engineering smoke run.")
    lines.append("It validates the measurement system (runtime, adapter, tracing,")
    lines.append("benchmark runner, metrics) and establishes the baseline reference.")
    lines.append("It is not a model-capability conclusion.")
    lines.append("")

    lines.append("## Per-episode results")
    lines.append("")
    lines.append("| task | seed | status | success | reward | steps | duration (s) "
                 "| input tokens | output tokens | action errors | error type |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for e in episodes:
        lines.append(
            f"| {e['task_id']} | {e['seed']} | {e['status']} "
            f"| {'✅' if is_truthy(e['success']) else '❌'} | {fmt_num(e['final_reward'], 1)} "
            f"| {e['steps']} | {fmt_num(e['duration_s'])} "
            f"| {e['input_tokens']} | {e['output_tokens']} "
            f"| {e['action_error_count']} | {e['error_type'] or '-'} |"
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
    lines.append("`estimated_cost` is null by design: the harness never guesses")
    lines.append("prices without a reliable price table; token counts above are")
    lines.append("the authoritative usage record.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="path to summary.json")
    parser.add_argument("--episodes", required=True, help="path to episodes.csv")
    parser.add_argument("--output", required=True, help="path to output report.md")
    parser.add_argument(
        "--notes", help="optional qualitative notes (markdown) appended to the report"
    )
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    with open(args.episodes, encoding="utf-8", newline="") as f:
        episodes = list(csv.DictReader(f))
    # csv gives strings; restore numeric fields used by the renderer
    numeric = {"final_reward", "duration_s", "steps", "input_tokens",
               "output_tokens", "action_error_count", "seed", "success"}
    for row in episodes:
        for key in numeric & row.keys():
            with contextlib.suppress(ValueError):
                row[key] = float(row[key]) if key == "duration_s" else int(float(row[key]))

    report = render(summary, episodes)
    if args.notes:
        notes = Path(args.notes).read_text(encoding="utf-8")
        report += "\n" + notes.strip() + "\n"
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
