# ADR-002: BrowserGym as the Environment Layer, not the Agent Runtime

## Status

Accepted (Phase 0)

## Context

Building Playwright-based browser infrastructure (DOM/axtree extraction,
action execution, tab management, task scoring) from scratch is expensive and
would compete with the project's actual focus: harness mechanisms.
BrowserGym already provides a unified environment interface
(`reset/step/close`), rich observations (goal, URL, axtree, DOM, screenshot,
open tabs, last action error) and one integration surface for MiniWoB,
WebArena, WorkArena and more.

## Decision

```text
Our Harness Runtime -> EnvironmentAdapter -> BrowserGym -> Playwright Chromium
```

- `EnvironmentAdapter` is the only door to BrowserGym (implemented in
  `env/browsergym_adapter.py`).
- Harness core works with `TaskSpec` / `Observation` / `EnvironmentStep`
  exclusively; Gymnasium/BrowserGym types must not leak into `runtime/`,
  `agents/` or `evaluation/`.
- Phase 0 uses the MiniWoB backend; the adapter (not the runtime) owns all
  environment-specific details such as `MINIWOB_URL` resolution.

## Consequences

- Positive: benchmarks and environments stay swappable (WebArena, WorkArena,
  future custom/SRE environments) without touching the runtime.
- Positive: high-level action strings (BrowserGym action set) keep the action
  space traceable, comparable and guardrail-friendly.
- Negative: one adapter layer to maintain; observation object→text conversion
  must track BrowserGym versions (0.14 exposes objects, not `axtree_txt`).
- The environment is always closed via `finally`, even on failure paths.
