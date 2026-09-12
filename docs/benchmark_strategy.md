# Benchmark Strategy

## Goals

Benchmarks serve the core research question: measuring what each harness
mechanism contributes **with the model held constant**. That requires
environments that are reproducible, automatically scored, and comparable
across harness configurations.

## Environment selection rationale

| Environment | Role | Why |
|---|---|---|
| MiniWoB++ (via BrowserGym) | Phase 0 smoke + Phase 4 multi-seed | local, deterministic, offline-capable, fast |
| WebArena-Verified | Phase 4 primary benchmark | manually audited tasks, deterministic offline evaluation, Hard 258-task subset |
| WorkArena / WorkArena++ | Phase 5 | production-like knowledge-work tasks (L1 atomic + compositional) |

## Phase 0 smoke suite

- 12 fixed tasks (`configs/benchmarks/miniwob_smoke.yaml`):
  `click-test, click-button, enter-text, choose-list, click-checkboxes,
  click-link, login-user, read-table, navigate-tree, use-autocomplete,
  choose-date-easy, order-food`
- Coverage: simple click, button choice, text entry, dropdown, checkbox,
  link, multi-field form, information reading, hierarchical navigation,
  dynamic components, date component, combined interaction.
- **seed 0 only, 12 episodes** — engineering validation of the pipeline
  (runtime, adapter, trace, metrics), *not* a performance result.
- `max_steps = 20`; environment-defined termination is always honored. No
  task-specific logic is allowed anywhere in the harness.

## Experiment identity & reproducibility

Every benchmark run produces `experiments/<experiment_id>/`:

```text
config.yaml     resolved config copy (benchmark, tasks, seeds, model, agent, runtime, trace)
manifest.json   timestamp, git commit, config hash, python version, provider, model name
episodes.csv    one row per episode (metrics)
summary.json    aggregate metrics + per-episode rows
run_ids.txt     run ids → full traces under runs/<run_id>/
```

Reproducibility requirements:
- all result-influencing parameters live in config (hashed into manifests)
- secrets never enter config or traces; configs with literal secret keys are
  rejected at load time and sanitized before persistence
- traces are machine-readable (see `trace_schema.md`), enabling later
  re-analysis and cross-experiment comparison
- Markdown reports are generated from `summary.json` / `episodes.csv` by
  `scripts/render_benchmark_report.py` — benchmark numbers are never
  hand-written into reports

## Scaling path

- **Phase 1** (Reliable Execution): same smoke suite, comparing baseline vs
  +retry / +verifier / +recovery. Single-seed is acceptable for pipeline
  validation; mechanism deltas need multiple seeds.
- **Phase 4**: multi-seed MiniWoB (≥5 seeds), then WebArena-Verified; formal
  ablation matrix across harness configurations.
- **Phase 5**: WorkArena / custom enterprise-style workflow benchmark.

## Known constraints

- Phase 0 runs serially; parallel execution is deliberately postponed.
- Ablation switches (`planner.enabled`, `verification.enabled`, ...) will be
  added only when the corresponding modules exist (Phase 1+), but the config
  schema is designed so they can be toggled without code changes.
- AgentLab compatibility: data models (RunResult/StepRecord/trace layout)
  are designed so results can later be exported or compared against
  AgentLab-style experiment records.
