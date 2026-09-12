# ADR-001: Own the Harness Runtime (no third-party agent framework)

## Status

Accepted (Phase 0)

## Context

LangGraph, Deep Agents, OpenAI Agents SDK and CrewAI all ship production-grade
agent runtimes with loop, state, checkpointing, guardrails, tool dispatch,
human-in-the-loop and tracing already implemented. Adopting one would make the
first working demo much faster.

## Decision

We implement our own harness runtime (`EpisodeRunner`, `RunState`,
`TraceRecorder`) and use third-party frameworks only as **design references**
(reading their source/interfaces), never as the core runtime.

Allowed: plain model SDKs (`openai`), BrowserGym as the environment layer.
Forbidden as core architecture: `Deep Agents -> browser tool -> benchmark`.

## Consequences

- Positive: every harness mechanism becomes something we build, measure and
  understand; ablation studies (baseline vs +planning vs +verification ...)
  are only meaningful if the baseline runtime is ours.
- Positive: no framework version churn on the critical path; the runtime loop
  stays thin, transparent and inspectable.
- Negative: slower initial progress; we re-implement things frameworks give
  for free (checkpointing, retries).
- Mitigation: later phases may add compatibility adapters to compare our
  results against framework implementations.
