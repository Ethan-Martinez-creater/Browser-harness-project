# ADR-003: Event Trace First

## Status

Accepted (Phase 0)

## Context

Agent systems without reliable traces cannot be debugged or seriously
evaluated: failures are multi-step, model decisions are opaque, and
environment errors hide between steps. Retrofitting tracing after building
features always loses information.

## Decision

Every episode is traced from the first commit, with the same rigor as the
runtime itself:

- `runs/<run_id>/manifest.json` — provenance (git commit, config hash,
  python/platform, provider/model, task, seed, max_steps)
- `runs/<run_id>/steps.jsonl` — one `StepRecord` per line, fsynced after each
  step (observation, action, action error, reward, termination, latency,
  tokens, error_type)
- `runs/<run_id>/artifacts/` — deterministic per-step artifacts (normalized
  observation, exact prompt, model response, raw text)
- `runs/<run_id>/result.json` — final `RunResult`

Rules:
1. Trace write failures raise `TRACE_WRITE_ERROR` — never silent.
2. Traces are JSON/JSONL and reloadable through typed readers.
3. Secrets never enter any trace file (unit-test enforced).
4. Full model chain-of-thought is never persisted; only a short
   debug-oriented `short_reason`.
5. Prompt/response saving is configurable; Phase 0 defaults are on for local
   benchmarks.

## Consequences

- Positive: every phase (verification, recovery, checkpointing) can build on
  complete traces; experiments remain re-analyzable after the fact.
- Positive: crash debugging is possible even for runs that died mid-episode.
- Negative: disk usage grows with episodes; mitigated by gitignoring
  `runs/`/`experiments/` and keeping only curated reports.
