# Phase 1A Verification (Shadow Mode) Report

- Experiment: `miniwob-smoke-20260912T063506Z-665479dc` (auto-generated from summary.json / episodes.csv)
- Tasks: 12 × seeds [0], serial
- Model: `deepseek-flash` (provider `openai_compatible`, temperature 0.0)
- Agent: `baseline`, max_steps 20
- Environment bootstrap: noop(wait_ms=500)

> **Scope warning**: this is the Phase 0 engineering smoke run.
It validates the measurement system (runtime, adapter, tracing,
benchmark runner, metrics) and establishes the baseline reference.
It is not a model-capability conclusion.

## Per-episode results

| task | seed | status | success | reward | steps | duration (s) | input tokens | output tokens | action errors | error type |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| click-test | 0 | success | ✅ | 1 | 1 | 19.390 | 1405 | 34 | 0 | TASK_TERMINATED |
| click-button | 0 | success | ✅ | 1 | 1 | 14.172 | 1449 | 47 | 0 | TASK_TERMINATED |
| enter-text | 0 | success | ✅ | 1 | 2 | 19.391 | 2866 | 96 | 0 | TASK_TERMINATED |
| choose-list | 0 | success | ✅ | 1 | 1 | 12.140 | 1498 | 119 | 0 | TASK_TERMINATED |
| click-checkboxes | 0 | success | ✅ | 1 | 2 | 16.609 | 2989 | 104 | 0 | TASK_TERMINATED |
| click-link | 0 | failed | ❌ | 0 | 5 | 102.391 | 7636 | 15061 | 1 | TASK_TERMINATED |
| login-user | 0 | success | ✅ | 1 | 3 | 17.500 | 4518 | 142 | 0 | TASK_TERMINATED |
| read-table | 0 | success | ✅ | 1 | 2 | 21.266 | 3160 | 130 | 0 | TASK_TERMINATED |
| navigate-tree | 0 | success | ✅ | 1 | 1 | 10.953 | 1481 | 70 | 0 | TASK_TERMINATED |
| use-autocomplete | 0 | success | ✅ | 1 | 3 | 34.907 | 4481 | 761 | 0 | TASK_TERMINATED |
| choose-date-easy | 0 | success | ✅ | 1 | 4 | 23.797 | 7734 | 325 | 1 | TASK_TERMINATED |
| order-food | 0 | success | ✅ | 1 | 3 | 149.766 | 5643 | 19727 | 0 | TASK_TERMINATED |

## Aggregate metrics

| metric | value |
|---|---:|
| success_rate | 0.917 (11/12) |
| mean_reward | 0.917 |
| mean_steps | 2.333 |
| median_steps | 2.0 |
| mean_duration_s | 36.857 |
| total_input_tokens | 44860 |
| total_output_tokens | 36616 |
| action_error_rate | 0.167 |

`estimated_cost` is null by design: the harness never guesses
prices without a reliable price table; token counts above are
the authoritative usage record.


## Verification layer results (Phase 1A)

- Every agent step received exactly one shadow verification: for all episodes,
  `verification_count == steps` (28 verifications over 28 steps) — the
  verifier added zero agent steps, zero environment actions and zero model
  calls. This is enforced structurally and by
  `tests/regression/test_phase0_baseline_compat.py`.
- `events.jsonl` contains exactly one verification event per step for every
  episode (12/12 episodes checked), with deterministic pre/post state
  fingerprints and the full FailureSignal list.
- Detected failure kinds across the run: ACTION_ERROR (2), NO_PROGRESS (3),
  TASK_FAILED (1). The one failed episode (click-link, task terminated with
  reward 0) was correctly observed by the TaskFailureDetector; a
  choose-date-easy no-progress step was correctly flagged as a WARNING.
- Detection is observation only: no control-flow change, no retry, no
  recovery, no replanning exists in this phase.

## Phase 0 baseline compatibility

- Reliability disabled = Phase 0 behavior (unit + regression tests: model
  call count, env step count, action sequence, termination, StepRecord count
  and trace semantics all preserved).
- Reliability enabled in shadow mode = identical control flow (regression
  tests assert equal model calls, env steps, actions and outcomes; only
  added observation).
