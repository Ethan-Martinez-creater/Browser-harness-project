"""Benchmark runner: executes task x seed episodes serially.

For each experiment it produces:

    experiments/<experiment_id>/
        config.yaml     resolved benchmark config copy
        manifest.json   provenance (git commit, config hash, versions)
        episodes.csv    one row per episode
        summary.json    aggregate metrics
        run_ids.txt     run ids (trace directories under runs/)

Phase 0 is strictly serial; parallelism is a later-phase concern.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import yaml

from web_harness.agents.baseline import BaselineAgent
from web_harness.config.loader import HarnessConfig, config_hash
from web_harness.core.errors import ConfigError
from web_harness.core.ids import new_experiment_id, new_run_id, now_utc_iso
from web_harness.core.models import RunResult, TaskSpec
from web_harness.evaluation.metrics import aggregate_metrics, episode_metrics
from web_harness.runtime.episode_runner import EpisodeRunner, git_commit

logger = logging.getLogger(__name__)


def build_model_adapter(model_cfg: dict):
    provider = model_cfg.get("provider", "openai_compatible")
    if provider == "mock":
        from web_harness.models.mock import MockModelAdapter

        return MockModelAdapter(actions=model_cfg.get("mock_actions") or ["noop()"])
    if provider == "openai_compatible":
        from web_harness.models.openai_compatible import OpenAICompatibleModelAdapter

        return OpenAICompatibleModelAdapter(
            model=model_cfg["model"],
            base_url_env=model_cfg.get("base_url_env", "MODEL_BASE_URL"),
            api_key_env=model_cfg.get("api_key_env", "MODEL_API_KEY"),
            base_url=model_cfg.get("base_url"),
            api_key=model_cfg.get("api_key"),
            temperature=float(model_cfg.get("temperature", 0.0)),
            timeout_s=float(model_cfg.get("timeout_s", 120.0)),
            max_tokens=model_cfg.get("max_tokens"),
        )
    raise ConfigError(f"unknown model provider: {provider}")


def build_agent(cfg: HarnessConfig):
    model_adapter = build_model_adapter(cfg.model)
    if cfg.agent.get("type", "baseline") != "baseline":
        raise ConfigError(f"unknown agent type: {cfg.agent.get('type')}")
    return BaselineAgent(
        model_adapter=model_adapter,
        max_history_steps=cfg.max_history_steps,
    ), model_adapter


def run_benchmark(
    cfg: HarnessConfig,
    *,
    benchmark_name: str | None = None,
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
    experiments_root: Path | str = "experiments",
    headless: bool = True,
) -> tuple[Path, list[RunResult]]:
    """Run task x seed episodes; returns (experiment_dir, results)."""
    bench = cfg.data.get("benchmark", {})
    if isinstance(bench, dict):
        benchmark_name = benchmark_name or bench.get("benchmark", "miniwob")
        tasks = tasks or bench.get("tasks") or []
        seeds = seeds if seeds is not None else bench.get("seeds", [0])
    else:
        # `benchmark: <name>` string form with top-level tasks/seeds
        # (configs/benchmarks/*.yaml layout)
        benchmark_name = benchmark_name or bench or "miniwob"
        tasks = tasks or cfg.data.get("tasks") or []
        seeds = seeds if seeds is not None else cfg.data.get("seeds", [0])
    if not tasks:
        raise ConfigError("no tasks configured for benchmark run")

    resolved = {
        "benchmark": benchmark_name,
        "tasks": tasks,
        "seeds": seeds,
        "model": cfg.model,
        "agent": cfg.agent,
        "runtime": cfg.runtime,
        "trace": cfg.trace,
    }

    experiment_id = new_experiment_id(f"{benchmark_name}-smoke")
    exp_dir = Path(experiments_root) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # config copy + manifest (provenance)
    (exp_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "experiment_id": experiment_id,
        "timestamp": now_utc_iso(),
        "git_commit": git_commit(),
        "config_hash": config_hash(resolved),
        "python_version": sys.version.split()[0],
        "model_provider": cfg.model.get("provider"),
        "model_name": cfg.model.get("model"),
        "num_episodes": len(tasks) * len(seeds),
    }
    (exp_dir / "manifest.json").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    agent, model_adapter = build_agent(cfg)
    provider = cfg.model.get("provider", "openai_compatible")
    results: list[RunResult] = []

    for task_id in tasks:
        for seed in seeds:
            task = TaskSpec(
                benchmark=benchmark_name,
                task_id=task_id,
                seed=seed,
                max_steps=cfg.max_steps,
            )
            from web_harness.env.browsergym_adapter import BrowserGymAdapter

            env = BrowserGymAdapter(
                observation_char_limit=cfg.observation_char_limit,
                headless=headless,
                save_screenshots=cfg.save_screenshots,
            )
            runner = EpisodeRunner(
                agent=agent,
                env=env,
                trace_root=Path(cfg.trace_root),
                save_prompts=cfg.save_prompts,
                save_model_responses=cfg.save_model_responses,
                model_provider=provider,
                model_name=cfg.model.get("model"),
                manifest_extra={"config_hash": cfg.hash},
            )
            logger.info("episode start: %s seed=%s", task.task_id, seed)
            result = runner.run(task, run_id=new_run_id())
            results.append(result)
            logger.info(
                "episode done: %s status=%s steps=%s",
                result.task_spec.task_id,
                result.status.value,
                result.num_steps,
            )

    # episodes.csv
    rows = [episode_metrics(r) for r in results]
    fieldnames = list(rows[0].keys()) if rows else []
    with open(exp_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # summary.json (strict JSON, machine-readable)
    summary = {
        "experiment_id": experiment_id,
        "config": resolved,
        "aggregate": aggregate_metrics(results),
        "episodes": rows,
    }
    (exp_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # run_ids.txt
    (exp_dir / "run_ids.txt").write_text(
        "\n".join(r.run_id for r in results) + "\n", encoding="utf-8"
    )
    return exp_dir, results
