"""web-harness command line interface.

Commands:
    run          execute a single task episode
    benchmark    execute a benchmark config (task x seeds, serial)
    inspect-run  print a text summary of one recorded run
    replay       offline replay/validate of a recorded run (no model, no browser)
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from web_harness.config.loader import HarnessConfig, load_env_file
from web_harness.evaluation.benchmark_runner import run_benchmark
from web_harness.evaluation.metrics import episode_metrics
from web_harness.runtime.episode_runner import EpisodeRunner, git_commit

app = typer.Typer(help="Reliable Web Workflow Agent Harness (Phase 0)")
console = Console()


@app.callback()
def _main():
    # pick up .env values (shell environment always wins)
    load_env_file()


def _load_config(path: str) -> HarnessConfig:
    try:
        return HarnessConfig.from_yaml(Path(path))
    except Exception as exc:
        typer.secho(f"config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def run(
    task: str = typer.Option(..., help="Task id, e.g. miniwob.click-test"),
    seed: int = typer.Option(0, help="Random seed"),
    config: str = typer.Option("configs/baseline.yaml", help="Harness config file"),
    max_steps: int | None = typer.Option(None, help="Override task max steps"),
):
    """Run one episode of one task."""
    cfg = _load_config(config)
    benchmark_name, _, task_id = task.partition(".")
    if not task_id:
        benchmark_name, task_id = "miniwob", task

    from web_harness.core.models import TaskSpec
    from web_harness.env.browsergym_adapter import BrowserGymAdapter
    from web_harness.evaluation.benchmark_runner import build_agent

    agent, model_adapter = build_agent(cfg)
    env = BrowserGymAdapter(
        observation_char_limit=cfg.observation_char_limit,
        save_screenshots=cfg.save_screenshots,
        bootstrap_action=cfg.bootstrap_action,
    )
    runner = EpisodeRunner(
        agent=agent,
        env=env,
        trace_root=cfg.trace_root,
        save_prompts=cfg.save_prompts,
        save_model_responses=cfg.save_model_responses,
        model_provider=cfg.model.get("provider", "openai_compatible"),
        model_name=cfg.model.get("model"),
        manifest_extra={"config_hash": cfg.hash, "git_commit": git_commit()},
    )
    task_spec = TaskSpec(
        benchmark=benchmark_name,
        task_id=task_id,
        seed=seed,
        max_steps=max_steps if max_steps is not None else cfg.max_steps,
    )
    result = runner.run(task_spec)
    _print_result(result)


def _print_result(result) -> None:
    table = Table(title=f"Run {result.run_id}")
    table.add_column("field", style="cyan")
    table.add_column("value")
    m = episode_metrics(result)
    for key, value in m.items():
        table.add_row(key, str(value))
    table.add_row("trace", result.trace_path or "")
    console.print(table)


@app.command()
def benchmark(
    config: str = typer.Option(..., help="Benchmark config file"),
    tasks: str | None = typer.Option(None, help="Comma-separated task override"),
    seeds: str | None = typer.Option(None, help="Comma-separated seed override"),
    experiments_root: str = typer.Option("experiments", help="Output root"),
):
    """Run a benchmark (serial, task x seed)."""
    cfg = _load_config(config)
    task_list = [t.strip() for t in tasks.split(",")] if tasks else None
    seed_list = [int(s.strip()) for s in seeds.split(",")] if seeds else None
    exp_dir, results = run_benchmark(
        cfg,
        tasks=task_list,
        seeds=seed_list,
        experiments_root=experiments_root,
    )
    console.print(f"[green]experiment written to[/green] {exp_dir}")
    table = Table(title=f"Experiment {exp_dir.name}")
    for col in ["task_id", "seed", "status", "success", "steps", "action_error_count"]:
        table.add_column(col)
    for r in results:
        table.add_row(
            r.task_spec.task_id,
            str(r.task_spec.seed),
            r.status.value,
            str(r.success),
            str(r.num_steps),
            str(r.action_error_count),
        )
    console.print(table)


@app.command()
def inspect_run(run_id: str, runs_root: str = typer.Option("runs")):
    """Print a text summary of a recorded run (no Web UI in Phase 0)."""
    from web_harness.observability.trace import TraceRecorder

    run_dir = Path(runs_root) / run_id
    result = TraceRecorder.read_result(run_dir)
    if result is None:
        typer.secho(f"no result.json under {run_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    m = episode_metrics(result)
    console.print(f"Run ID: {m['run_id']}")
    console.print(f"Task: {m['benchmark']}.{m['task_id']} (seed={m['seed']})")
    console.print(f"Status: {m['status']}")
    console.print(f"Reward: {m['final_reward']}")
    console.print(f"Steps: {m['steps']}")
    console.print(f"Duration: {m['duration_s']}s")
    console.print(f"Tokens: in={m['input_tokens']} out={m['output_tokens']}")
    console.print(f"Action errors: {m['action_error_count']}")
    console.print(f"Trace directory: {result.trace_path}")
    # reliability summary (Phase 1B/1C/1D)
    events = TraceRecorder.read_events(run_dir)
    replan_events = [e for e in events if e.event_type.value == "replan"]
    plan_summaries = [
        {
            "step": e.step_index,
            "subgoal": (e.data.get("immediate_subgoal") or "")[:80],
            "horizon": e.data.get("horizon_steps"),
        }
        for e in replan_events if e.outcome == "created"
    ]
    console.print(
        f"Retries: {m['retry_count']} (cycles {m['retry_cycle_count']})"
    )
    console.print(
        f"Recoveries: {m['recovery_count']} "
        f"(success {m['recovery_success_count']} / "
        f"failed {m['recovery_failed_count']} / "
        f"unresolved {m['recovery_unresolved_count']})"
    )
    console.print(f"Replans: {m['replan_count']}")
    console.print(f"Replan successes: {m['replan_success_count']}")
    console.print(f"Replan failures: {m['replan_failed_count']}")
    console.print(f"Replan unresolved: {m['replan_unresolved_count']}")
    console.print(f"Replan model calls: {m['replan_model_calls']}")
    if plan_summaries:
        for ps in plan_summaries:
            console.print(
                f"  plan@step {ps['step']}: horizon={ps['horizon']} "
                f"subgoal={ps['subgoal']}"
            )
    steps = TraceRecorder.read_steps(run_dir)
    if steps:
        table = Table(title="Steps")
        for col in ["#", "action", "error", "reward", "in_tok", "out_tok"]:
            table.add_column(col)
        for s in steps:
            table.add_row(
                str(s.step_index),
                (s.action or "-")[:60],
                (s.action_error or "-")[:40],
                str(s.reward if s.reward is not None else "-"),
                str(s.input_tokens if s.input_tokens is not None else "-"),
                str(s.output_tokens if s.output_tokens is not None else "-"),
            )
        console.print(table)


@app.command()
def replay(
    run_id: str,
    runs_root: str = typer.Option("runs", help="Directory containing run folders"),
    semantic: bool = typer.Option(
        True, "--semantic/--no-semantic",
        help="Also replay the deterministic StepVerifier (schema v2 traces)",
    ),
):
    """Offline replay/validate of a recorded run.

    Pure offline analysis: zero model calls, zero environment actions, and
    the original trace is never modified. Exits 1 when the trace is
    structurally invalid or a verifier replay mismatch is found.
    """
    from web_harness.persistence.replay import TraceBundleLoader, TraceReplayEngine

    run_dir = Path(runs_root) / run_id
    if not run_dir.is_dir():
        typer.secho(f"run directory not found: {run_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    bundle = TraceBundleLoader().load(run_dir)
    report = TraceReplayEngine(semantic_replay=semantic).run(bundle)

    console.print(f"Run ID: {report.run_id or '-'}")
    console.print(f"Trace schema version: {report.trace_schema_version}")
    console.print(f"Steps: {report.step_count}")
    console.print(f"Events: {report.event_count}")
    console.print(
        "Structural validation: "
        + ("VALID" if report.structural_valid else "INVALID")
    )
    console.print(
        "Artifact validation: "
        + ("OK" if report.artifact_missing_count == 0
           else f"{report.artifact_missing_count} missing")
    )
    console.print(
        "Metric validation: "
        + ("OK" if not report.metric_mismatches
           else f"{len(report.metric_mismatches)} mismatches")
    )
    console.print(f"Verifier replay: {report.replayed_verifications} verifications")
    console.print(
        "Verification mismatches: "
        + ("0" if not report.verification_mismatches
           else str(len(report.verification_mismatches)))
    )
    for note in report.notes:
        console.print(f"[dim]note: {note}[/dim]")
    for mismatch in report.metric_mismatches + report.verification_mismatches:
        console.print(f"[yellow]mismatch: {mismatch}[/yellow]")
    for error in report.errors:
        console.print(f"[red]error: {error}[/red]")
    if not report.structural_valid or report.verification_mismatches:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
