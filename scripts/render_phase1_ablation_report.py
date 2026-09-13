"""Render the Final Phase 1 ablation report from machine results.

Usage:
    uv run python scripts/render_phase1_ablation_report.py \
        --config-dir configs/phase1/final \
        --experiments-root experiments \
        --output-dir reports/phase1/final

Reads the five variant summaries (A0-A4), aggregates them (12 tasks x 3
seeds x 5 variants = 180 episodes) and generates:
    phase1_ablation_summary.json   machine-readable aggregate
    phase1_ablation_episodes.csv   one row per episode (all variants)
    phase1_ablation_report.md      auto-generated report

All numbers come from the machine results — nothing is hand-written.
tokens_per_success = total tokens / successful episodes (null when 0).
Task success is NEVER mixed with local retry/recovery/replan success.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

VARIANTS = [
    ("a0_baseline", "A0", "Baseline (reliability disabled)"),
    ("a1_detection", "A1", "Detection only"),
    ("a2_retry", "A2", "Detection + Retry"),
    ("a3_recovery", "A3", "Detection + Retry + Recovery"),
    ("a4_replanning", "A4", "Full Phase 1"),
]

# aggregate fields copied verbatim from each variant summary
_AGG_FIELDS = [
    "num_episodes", "num_successes", "success_rate", "mean_reward",
    "mean_steps", "median_steps", "mean_duration_s",
    "total_input_tokens", "total_output_tokens",
    "action_error_rate",
    "verification_count", "failure_signal_count",
    "episodes_with_failure_signal",
    "total_retry_count", "retry_cycle_count", "retry_success_count",
    "retry_exhausted_count", "retry_success_rate",
    "total_extra_model_calls",
    "total_recovery_count", "recovery_success_count",
    "recovery_failed_count", "recovery_unresolved_count",
    "recovered_episode_count", "recovery_environment_actions",
    "total_recovery_latency_s",
    "total_replan_count", "episodes_with_replan",
    "replan_success_count", "replan_failed_count",
    "replan_unresolved_count", "total_replan_model_calls",
    "total_replan_input_tokens", "total_replan_output_tokens",
    "total_replan_latency_s", "replan_success_rate",
    "reliability_extra_model_calls", "reliability_extra_tokens",
    "reliability_extra_latency_s",
]


def fmt(value, digits=3):
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_variant(experiments_root: Path, name: str):
    """Find the newest experiment whose persisted config matches the variant
    file (matched via the benchmark task list + seeds + reliability flags)."""
    import yaml

    cfg = yaml.safe_load(
        Path(f"configs/phase1/final/{name}.yaml").read_text(encoding="utf-8")
    )
    for exp_dir in sorted(experiments_root.glob("*/"), reverse=True):
        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = summary.get("config") or {}
        if config.get("tasks") != cfg["benchmark"]["tasks"]:
            continue
        if config.get("seeds") != cfg["benchmark"]["seeds"]:
            continue
        if (config.get("model") or {}).get("model") != cfg["model"]["model"]:
            continue
        if config.get("reliability") != cfg.get("reliability"):
            continue
        return exp_dir, summary
    raise FileNotFoundError(f"no experiment found for variant {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs/phase1/final")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--output-dir", default="reports/phase1/final")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments_root = Path(args.experiments_root)

    variants = {}
    all_episodes = []
    for name, label, description in VARIANTS:
        exp_dir, summary = load_variant(experiments_root, name)
        variants[label] = {
            "name": name,
            "description": description,
            "experiment_id": summary["experiment_id"],
            "config_hash": None,
            "aggregate": {k: summary["aggregate"].get(k) for k in _AGG_FIELDS},
            "failure_kind_counts": summary["aggregate"].get(
                "failure_kind_counts", {}
            ),
            "episodes": summary["episodes"],
        }
        manifest = json.loads(
            (exp_dir / "manifest.json").read_text(encoding="utf-8")
        )
        variants[label]["config_hash"] = manifest.get("config_hash")
        for e in summary["episodes"]:
            row = dict(e)
            row["variant"] = label
            all_episodes.append(row)
        print(f"{label}: {exp_dir.name} ({summary['aggregate']['num_episodes']} episodes)")

    # tokens_per_success (null when no successes)
    for _label, v in variants.items():
        agg = v["aggregate"]
        total_tokens = (agg["total_input_tokens"] or 0) + (
            agg["total_output_tokens"] or 0
        )
        successes = agg["num_successes"] or 0
        agg["total_tokens"] = total_tokens
        agg["tokens_per_success"] = (
            total_tokens / successes if successes else None
        )

    # per-seed breakdown
    per_seed = {}
    for label, v in variants.items():
        per_seed[label] = {}
        for seed in (0, 1, 2):
            eps = [e for e in v["episodes"] if e["seed"] == seed]
            per_seed[label][seed] = {
                "episodes": len(eps),
                "successes": sum(1 for e in eps if e["success"]),
                "mean_steps": (
                    statistics.fmean(e["steps"] for e in eps) if eps else 0
                ),
                "total_input_tokens": sum(e["input_tokens"] or 0 for e in eps),
                "total_output_tokens": sum(
                    e["output_tokens"] or 0 for e in eps
                ),
            }

    payload = {
        "suite": "phase1_final_ablation",
        "frozen_commit": "1436cb2207ec53a050a656d01bd4775f93e919ba",
        "variants": variants,
        "per_seed": per_seed,
        "total_episodes": sum(
            v["aggregate"]["num_episodes"] or 0 for v in variants.values()
        ),
    }
    summary_path = out_dir / "phase1_ablation_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # episodes.csv (all variants)
    fieldnames = ["variant"] + list(all_episodes[0].keys()) if all_episodes else []
    fieldnames = [f for f in dict.fromkeys(fieldnames)]
    with open(out_dir / "phase1_ablation_episodes.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_episodes)

    # ---- markdown report -------------------------------------------------
    lines = [
        "# Phase 1 Final Ablation Report (A0-A4)",
        "",
        f"- Frozen implementation commit: "
        f"`{payload['frozen_commit']}`",
        "- 12 MiniWoB tasks x seeds [0, 1, 2] x 5 variants = "
        f"{payload['total_episodes']} episodes, serial",
        "- Model: `deepseek-flash`, temperature 0.0, identical endpoint, "
        "tasks, seeds, ActionContract, bootstrap and runtime across variants",
        "- The ONLY difference between variants is the reliability "
        "capability switches",
        "- Auto-generated from `phase1_ablation_summary.json` — no "
        "hand-written numbers",
        "",
        "## 1. Main metrics comparison",
        "",
        "| metric | " + " | ".join(v[1] for v in VARIANTS) + " |",
        "|---|" + "---:|" * len(VARIANTS),
    ]

    def row(label, key, fmt_digits=3, multiply=None):
        cells = []
        for _, label_, _ in VARIANTS:
            value = variants[label_]["aggregate"].get(key)
            if multiply and value is not None:
                value = value * multiply
            cells.append(fmt(value, fmt_digits))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("episodes", "num_episodes", 0)
    row("successes", "num_successes", 0)
    row("task success rate", "success_rate")
    row("mean reward", "mean_reward")
    row("mean steps", "mean_steps", 2)
    row("median steps", "median_steps", 1)
    row("mean duration (s)", "mean_duration_s", 1)
    row("input tokens", "total_input_tokens", 0)
    row("output tokens", "total_output_tokens", 0)
    row("total tokens", "total_tokens", 0)
    row("tokens per success", "tokens_per_success", 0)
    row("action error rate", "action_error_rate")
    lines.append("")

    lines.append("## 2. Reliability overhead comparison")
    lines.append("")
    lines.append("| metric | " + " | ".join(v[1] for v in VARIANTS) + " |")
    lines.append("|---|" + "---:|" * len(VARIANTS))
    row("verification count", "verification_count", 0)
    row("failure signal count", "failure_signal_count", 0)
    row("episodes with failure signal", "episodes_with_failure_signal", 0)
    row("retry attempts", "total_retry_count", 0)
    row("retry cycles", "retry_cycle_count", 0)
    row("retry success rate (cycles)", "retry_success_rate")
    row("retry exhausted", "retry_exhausted_count", 0)
    row("extra model calls (retry)", "total_extra_model_calls", 0)
    row("recovery count", "total_recovery_count", 0)
    row("recovery success", "recovery_success_count", 0)
    row("recovery failed", "recovery_failed_count", 0)
    row("recovery unresolved", "recovery_unresolved_count", 0)
    row("recovered episodes", "recovered_episode_count", 0)
    row("recovery env actions", "recovery_environment_actions", 0)
    row("recovery latency (s)", "total_recovery_latency_s", 1)
    row("replan count", "total_replan_count", 0)
    row("replan success", "replan_success_count", 0)
    row("replan failed", "replan_failed_count", 0)
    row("replan unresolved", "replan_unresolved_count", 0)
    row("replan model calls", "total_replan_model_calls", 0)
    row("replan tokens", "total_replan_input_tokens", 0)
    row("replan latency (s)", "total_replan_latency_s", 1)
    row("reliability extra model calls", "reliability_extra_model_calls", 0)
    row("reliability extra tokens", "reliability_extra_tokens", 0)
    row("reliability extra latency (s)", "reliability_extra_latency_s", 1)
    lines.append("")

    lines.append(
        "Task success, local retry success, local recovery success and "
        "local replan success are DIFFERENT metrics and are never merged "
        "into one success rate."
    )
    lines.append("")

    lines.append("## 3. FailureKind distribution")
    lines.append("")
    kinds = sorted({
        k for v in variants.values() for k in v["failure_kind_counts"]
    })
    lines.append("| failure kind | " + " | ".join(v[1] for v in VARIANTS) + " |")
    lines.append("|---|" + "---:|" * len(VARIANTS))
    for kind in kinds:
        cells = [
            fmt(variants[l_]["failure_kind_counts"].get(kind, 0), 0)
            for _, l_, _ in VARIANTS
        ]
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 4. Mechanism triggers (actual firings)")
    lines.append("")
    lines.append("| mechanism | " + " | ".join(v[1] for v in VARIANTS) + " |")
    lines.append("|---|" + "---:|" * len(VARIANTS))
    row("retry attempts fired", "total_retry_count", 0)
    row("recoveries fired", "total_recovery_count", 0)
    row("replans fired", "total_replan_count", 0)
    lines.append("")
    never = []
    for label, name, _ in [
        ("retry", "a2_retry", None), ("recovery", "a3_recovery", None),
        ("replan", "a4_replanning", None),
    ]:
        key = {
            "retry": "total_retry_count", "recovery": "total_recovery_count",
            "replan": "total_replan_count",
        }[label]
        active_variants = [
            v[1] for v in VARIANTS
            if name <= v[0]
        ]
        zero = [
            l_ for l_ in active_variants
            if (variants[l_]["aggregate"].get(key) or 0) == 0
        ]
        if zero:
            never.append(f"{label}: never fired in {', '.join(zero)}")
    if never:
        lines.append("Variants where a mechanism never triggered:")
        lines.append("")
        for n in never:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## 5. Per-seed results")
    lines.append("")
    lines.append("| variant | seed | episodes | successes | mean steps | "
                 "input tokens | output tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, label, _ in VARIANTS:
        for seed in (0, 1, 2):
            s = per_seed[label][seed]
            lines.append(
                f"| {label} | {seed} | {s['episodes']} | {s['successes']} "
                f"| {fmt(s['mean_steps'], 2)} | {s['total_input_tokens']} "
                f"| {s['total_output_tokens']} |"
            )
    lines.append("")

    lines.append("## 6. 3-seed aggregate")
    lines.append("")
    lines.append(
        "The main comparison table above IS the 3-seed aggregate "
        "(seeds 0+1+2 pooled per variant)."
    )
    lines.append("")

    lines.append("## 7. Interpretation guardrails")
    lines.append("")
    lines.append(
        "- Single-episode fluctuations at this scale are NOT evidence of a "
        "mechanism's benefit; the ablation quantifies cost and observable "
        "behavior, not statistical significance.",
    )
    lines.append(
        "- A mechanism with zero firings in a variant contributed zero "
        "cost AND zero benefit there.",
    )
    lines.append(
        "- tokens_per_success is undefined (null) for variants with zero "
        "successful episodes.",
    )
    lines.append("")

    lines.append("## Experiment IDs")
    lines.append("")
    for _, label, description in VARIANTS:
        lines.append(
            f"- {label} ({description}): "
            f"`{variants[label]['experiment_id']}` "
            f"(config_hash `{variants[label]['config_hash']}`)"
        )
    lines.append("")

    (out_dir / "phase1_ablation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"report written to {out_dir / 'phase1_ablation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
