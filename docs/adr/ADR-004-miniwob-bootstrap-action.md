# ADR-004: MiniWoB Bootstrap Action

## Status

Accepted (Phase 0 remediation, R2)

## Context

MiniWoB task pages render their task HTML *after* the DOM-load event that
BrowserGym waits for during `reset()`. The first observation taken at reset
time can therefore contain an **empty accessibility tree**, leaving the agent
with nothing to act on.

During the initial Phase 0 run, the `BrowserGymAdapter` silently executed a
hidden `noop()` step after `reset()` to refresh the observation. This violated
a harness invariant: an environment-mutating action was executed outside the
trace and outside the step budget.

BrowserGym 0.14.3 provides `pre_observation_delay` (a sleep before
observation extraction) which fully covers *post-action* observation timing,
but the reset-time observation is extracted after `_wait_dom_loaded()` only —
there is no public API to re-extract the reset observation after a delay
without executing an action.

## Decision

The bootstrap becomes **explicit, verified, configured and tracked**:

1. It is a config value, not a hard-coded behavior:
   `environment.bootstrap_action: "noop(wait_ms=500)"` (default: disabled).
2. The adapter verifies the bootstrap step produces **no reward and no
   termination**; otherwise the episode fails fast with
   `ENVIRONMENT_INIT_ERROR`.
3. It is recorded in every run's `manifest.json` as
   `environment_bootstrap_action` (provenance).
4. It does NOT enter the step budget and is NOT recorded as a StepRecord —
   it is part of environment initialization, and its artifacts live only in
   provenance.
5. Other benchmarks (WebArena, WorkArena, future environments) default to
   `bootstrap_action: null` and must not enable it without their own
   verification.

## Consequences

- Positive: MiniWoB observations are reliably populated from step 0 on.
- Positive: the harness no longer hides an untracked action; the cost is a
  one-time, side-effect-free wait, visible in every manifest.
- Negative: episode wall time grows by the bootstrap wait (~0.5–1 s).
- Guard: a unit-level contract (`bootstrap verified`) plus provenance makes
  silent misuse detectable in any future benchmark.
