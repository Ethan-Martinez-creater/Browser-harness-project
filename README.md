# Reliable Web Workflow Agent Harness

A minimal, transparent, extensible and evaluable agent harness for multi-step
web workflows.

This project does not build a feature-rich browser automation product. Its
purpose is to study, module by module, how a modern agent harness is built:
planning, state management, verification, retry/recovery, checkpointing,
context management, skills, human-in-the-loop, tracing and evaluation — and to
measure, through controlled experiments, what each mechanism contributes.

**Phase 0 status**: minimal self-owned harness runtime + BrowserGym
environment layer + MiniWoB smoke benchmark. See
[docs/architecture.md](docs/architecture.md) and the phase plan in the
project documentation.

## Core question

> With the model held constant, how much reliability, efficiency and cost
> benefit do harness mechanisms (Planning, Verification, Recovery, Checkpoint,
> ...) bring to multi-step web workflows?

## Architecture

```text
CLI / Config
    ↓
EpisodeRunner (harness runtime, self-owned loop)
    ├── BaselineAgent → ModelAdapter → ActionDecision
    ├── State Store (RunState)
    └── TraceRecorder (manifest / steps.jsonl / artifacts / result)
    ↓
EnvironmentAdapter → BrowserGym → Playwright Chromium
```

Third-party agent frameworks (LangGraph, Deep Agents, CrewAI, ...) are
deliberately **not** used as the runtime — see
[docs/adr/ADR-001-own-runtime.md](docs/adr/ADR-001-own-runtime.md).

## Installation

Requirements: Python 3.11 is managed automatically by [uv](https://docs.astral.sh/uv/).

```bash
# 1. install dependencies into the project-local .venv (Python 3.11)
uv sync

# 2. install Chromium for Playwright
uv run playwright install chromium

# 3. get the MiniWoB task HTML (local checkout, no network needed at run time)
git clone --depth 1 https://github.com/Farama-Foundation/miniwob-plusplus third_party/miniwob-plusplus
```

`MINIWOB_URL` resolution order: config value → `MINIWOB_URL` env var → the
local `third_party/miniwob-plusplus/miniwob/html/miniwob/` checkout above.

## Environment variables

Secrets only ever come from environment variables (see `.env.example`):

```bash
export MODEL_BASE_URL=https://api.deepseek.com   # any OpenAI-compatible endpoint
export MODEL_API_KEY=sk-...                      # never committed
```

The model name and all other parameters live in `configs/*.yaml`.

## Run a single task

```bash
uv run web-harness run --task miniwob.click-test --seed 0 --config configs/baseline.yaml
```

Offline pipeline check (mock model, no API key, real browser):

```bash
uv run web-harness run --task miniwob.click-test --seed 0 --config configs/mock_local.yaml
```

## Run the smoke benchmark

```bash
uv run web-harness benchmark --config configs/benchmarks/miniwob_smoke.yaml
```

Runs 12 fixed MiniWoB tasks × seed 0, serially. Outputs land in
`experiments/<experiment_id>/` (`config.yaml`, `manifest.json`,
`episodes.csv`, `summary.json`, `run_ids.txt`) and full traces in `runs/`.

## Inspect results

```bash
uv run web-harness inspect-run <run_id>
```

Prints status, reward, steps, duration, tokens, action errors and a per-step
table from the recorded trace.

## Tests

```bash
uv run pytest            # unit tests + MiniWoB integration test
uv run pytest tests/unit # unit tests only (no browser needed)
```

Unit tests use a deterministic mock model and a scripted fake environment; the
integration test drives the real MiniWoB environment with a scripted agent
(no LLM API involved).

## Project structure

```text
configs/            harness and benchmark configs (no secrets)
docs/               architecture, benchmark strategy, trace schema, ADRs
scripts/            environment probe and registry inspection
src/web_harness/
  core/             data models, error taxonomy, ids
  models/           ModelAdapter: mock + OpenAI-compatible
  agents/           BaselineAgent + PromptBuilder
  env/              EnvironmentAdapter: BrowserGym adapter + fake env
  runtime/          EpisodeRunner + RunState (the self-owned loop)
  observability/    TraceRecorder
  evaluation/       metrics + benchmark runner
  config/           YAML config loading
  cli.py            web-harness CLI
tests/unit          unit tests (mock model, fake env)
tests/integration   real MiniWoB environment test
reports/phase0/     probe results and benchmark reports
runs/               per-episode traces (gitignored)
experiments/        benchmark outputs (gitignored)
```

## Phase 0 limitations

- Baseline agent only: no planner, no retry/recovery, no verification, no
  checkpointing, no skills, no human approval, no vision input.
- Smoke benchmark is engineering validation only — 12 tasks × 1 seed is
  **not** a performance result.
- Serial benchmark execution only.
- Cost estimation is left `null` unless a reliable price table is configured.

## Version notes

Python 3.11.16 · browsergym-core/min*iwob 0.14.3 · gymnasium 1.3.0 ·
playwright 1.44.0 (Chromium build v1117) · openai SDK 3.13.0 · pydantic
2.13.5. Exact resolved versions are pinned in `uv.lock`.
