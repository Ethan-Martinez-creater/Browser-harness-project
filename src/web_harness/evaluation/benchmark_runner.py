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
from collections.abc import Callable
from pathlib import Path

import yaml

from web_harness.agents.base import Agent
from web_harness.agents.baseline import BaselineAgent
from web_harness.config.loader import HarnessConfig, config_hash
from web_harness.core.errors import ConfigError
from web_harness.core.ids import new_experiment_id, new_run_id, now_utc_iso
from web_harness.core.models import RunResult, TaskSpec
from web_harness.env.base import EnvironmentAdapter
from web_harness.evaluation.metrics import aggregate_metrics, episode_metrics
from web_harness.runtime.episode_runner import EpisodeRunner, git_commit

logger = logging.getLogger(__name__)

# config keys that must never hold a literal secret value (R3)
_FORBIDDEN_SECRET_KEYS = {"api_key", "token", "secret", "password", "api_key_value"}


def sanitize_config(data: dict) -> dict:
    """Return a copy of the config safe for persistence.

    Rejects literal secrets entirely (fail-fast upstream in HarnessConfig);
    this is a second, defensive layer that strips any *_env-independent
    secret-looking key before a config copy is written to disk.
    """
    sanitized: dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            sub = {
                k: v
                for k, v in value.items()
                if k.lower() not in _FORBIDDEN_SECRET_KEYS
            }
            sanitized[key] = sub
        else:
            sanitized[key] = value
    return sanitized


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
    headless: bool | None = None,
    environment_factory: Callable[[], EnvironmentAdapter] | None = None,
    agent_factory: Callable[[], Agent] | None = None,
) -> tuple[Path, list[RunResult]]:
    """Run task x seed episodes; returns (experiment_dir, results).

    `environment_factory` / `agent_factory` allow fully offline contract
    tests (FakeEnvironment + MockModelAdapter); defaults build the real
    BrowserGym environment and configured model adapter. Each episode gets a
    fresh environment; one episode failing must never lose sibling results.
    """
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
    headless = cfg.headless if headless is None else headless

    # canonical resolved experiment config: the single source for the config
    # copy, experiment manifest hash AND run manifest hashes (B1)
    resolved = {
        "benchmark": benchmark_name,
        "tasks": tasks,
        "seeds": seeds,
        "model": sanitize_config(cfg.model),
        "agent": cfg.agent,
        "runtime": cfg.runtime,
        "trace": cfg.trace,
        "environment": cfg.environment,
        "reliability": sanitize_config(cfg.reliability),
    }
    canonical_config_hash = config_hash(resolved)

    experiment_id = new_experiment_id(f"{benchmark_name}-smoke")
    exp_dir = Path(experiments_root) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # config copy (sanitized) + manifest (provenance, strict JSON)
    (exp_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "experiment_id": experiment_id,
        "timestamp": now_utc_iso(),
        "git_commit": git_commit(),
        "config_hash": canonical_config_hash,
        "python_version": sys.version.split()[0],
        "model_provider": cfg.model.get("provider"),
        "model_name": cfg.model.get("model"),
        "num_episodes": len(tasks) * len(seeds),
    }
    (exp_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    agent = agent_factory() if agent_factory else build_agent(cfg)[0]
    provider = cfg.model.get("provider", "openai_compatible")
    results: list[RunResult] = []

    if environment_factory is None:
        def environment_factory():
            from web_harness.env.browsergym_adapter import BrowserGymAdapter

            return BrowserGymAdapter(
                observation_char_limit=cfg.observation_char_limit,
                headless=headless,
                save_screenshots=cfg.save_screenshots,
                bootstrap_action=cfg.bootstrap_action,
            )

    # Phase 1A: verifier only when reliability is enabled AND verification is
    # enabled AND mode == shadow (B2); anything else means zero behavioral or
    # cost delta vs the Phase 0 baseline.
    verifier = None
    if cfg.verification_enabled and cfg.verification_mode == "shadow":
        from web_harness.reliability.verifier import DefaultStepVerifier

        verifier_cfg = cfg.reliability.get("verification") or {}
        no_progress_cfg = verifier_cfg.get("no_progress") or {}
        loop_cfg = verifier_cfg.get("loop") or {}
        verifier = DefaultStepVerifier(
            detect_no_progress=bool(no_progress_cfg.get("enabled", True)),
            detect_loop=bool(loop_cfg.get("enabled", True)),
            loop_consecutive_threshold=int(loop_cfg.get("consecutive_threshold", 2)),
        )

    # Phase 1B: controlled retry for model-side failures only; disabled by
    # default (baseline configs never enable it).
    from web_harness.core.reliability import default_budget_from_config

    retry_cfg = cfg.reliability.get("retry") or {}
    decision_executor = None
    if cfg.reliability_enabled and retry_cfg.get("enabled", False):
        from web_harness.reliability.retry import RetryPolicy
        from web_harness.runtime.decision_executor import DecisionExecutor

        api_cfg = retry_cfg.get("model_api") or {}
        output_cfg = retry_cfg.get("model_output") or {}
        decision_executor = DecisionExecutor(
            retry_policy=RetryPolicy(
                api_max_retries=int(api_cfg.get("max_retries", 2)),
                api_backoff_ms=[int(ms) for ms in api_cfg.get("backoff_ms", [500, 1000])],
                parse_max_retries=int(output_cfg.get("max_retries", 1)),
            ),
            budget=default_budget_from_config(cfg.reliability),
        )

    # Phase 1C: rule-based policy + RecoveryManager for environment-side
    # failures; disabled by default (Phase 1B behavior unchanged).
    recovery_cfg = cfg.reliability.get("recovery") or {}
    failure_policy = None
    if cfg.reliability_enabled and recovery_cfg.get("enabled", False):
        from web_harness.reliability.policy import FailurePolicyEngine

        failure_policy = FailurePolicyEngine(
            wait_ms=int(recovery_cfg.get("wait_ms", 500)),
            block_steps=int(recovery_cfg.get("block_steps", 1)),
        )

    # Phase 1D: deterministic escalation + bounded Replanner; disabled by
    # default (Phase 1C behavior unchanged).
    replan_cfg = cfg.reliability.get("replanning") or {}
    replan_trigger_policy = None
    replan_executor = None
    if cfg.reliability_enabled and replan_cfg.get("enabled", False):
        from web_harness.reliability.replan_policy import ReplanTriggerPolicy
        from web_harness.reliability.replanner import ReplanExecutor
        from web_harness.reliability.retry import RetryPolicy as _RetryPolicy

        replan_trigger_policy = ReplanTriggerPolicy()
        replan_model = build_model_adapter(cfg.model)
        api_cfg = retry_cfg.get("model_api") or {}
        output_cfg = retry_cfg.get("model_output") or {}
        replan_executor = ReplanExecutor(
            model_adapter=replan_model,
            retry_policy=_RetryPolicy(
                api_max_retries=int(api_cfg.get("max_retries", 2)),
                api_backoff_ms=[
                    int(ms) for ms in api_cfg.get("backoff_ms", [500, 1000])
                ],
                parse_max_retries=int(output_cfg.get("max_retries", 1)),
            ),
            budget=default_budget_from_config(cfg.reliability),
        )

    for task_id in tasks:
        for seed in seeds:
            task = TaskSpec(
                benchmark=benchmark_name,
                task_id=task_id,
                seed=seed,
                max_steps=cfg.max_steps,
            )
            env = environment_factory()
            runner = EpisodeRunner(
                agent=agent,
                env=env,
                trace_root=Path(cfg.trace_root),
                save_prompts=cfg.save_prompts,
                save_model_responses=cfg.save_model_responses,
                model_provider=provider,
                model_name=cfg.model.get("model"),
                manifest_extra={"config_hash": canonical_config_hash},
                verifier=verifier,
                verification_mode=cfg.verification_mode,
                decision_executor=decision_executor,
                failure_policy=failure_policy,
                recovery_budget=default_budget_from_config(cfg.reliability),
                replan_trigger_policy=replan_trigger_policy,
                replan_executor=replan_executor,
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
