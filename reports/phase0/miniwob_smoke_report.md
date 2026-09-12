# Phase 0 MiniWoB Smoke Benchmark Report

- Date: 2026-09-12
- Experiment: `miniwob-smoke-20260912T025451Z-bf9a9d90` (12 tasks × seed 0, serial)
- Model: `deepseek-flash` via OpenAI-compatible endpoint (temperature 0)
- Agent: Phase 0 baseline (text-only, axtree input, single action per step)
- Config: `configs/benchmarks/miniwob_smoke.yaml` (`max_steps = 20`)

> **Scope warning**: this is the Phase 0 engineering smoke run. 12 episodes ×
> 1 seed say nothing definitive about model or harness capability. It
> validates that the runtime, adapter, tracing, benchmark runner and metrics
> work end to end, and establishes the baseline reference point.

## Per-episode results

| task | success | reward | steps | duration (s) | input tokens | output tokens | action errors | error type |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| click-test | ✅ | 1.0 | 1 | 14.9 | 459 | 68 | 0 | TASK_TERMINATED |
| click-button | ✅ | 1.0 | 1 | 13.3 | 503 | 52 | 0 | TASK_TERMINATED |
| enter-text | ❌ | 0.0 | 4 | 34.4 | 1508 | 793 | 2 | MODEL_OUTPUT_PARSE_ERROR |
| choose-list | ✅ | 1.0 | 1 | 13.8 | 552 | 35 | 0 | TASK_TERMINATED |
| click-checkboxes | ❌ | 0.0 | 7 | 51.7 | 3533 | 622 | 1 | MODEL_OUTPUT_PARSE_ERROR |
| click-link | ✅ | 1.0 | 5 | 48.4 | 2919 | 12313 | 1 | TASK_TERMINATED |
| login-user | ❌ | 0.0 | 7 | 54.7 | 3589 | 1790 | 5 | MODEL_OUTPUT_PARSE_ERROR |
| read-table | ❌ | 0.0 | 20 | 118.8 | 14457 | 8279 | 17 | MAX_STEPS_EXCEEDED |
| navigate-tree | ✅ | 1.0 | 1 | 14.5 | 535 | 145 | 0 | TASK_TERMINATED |
| use-autocomplete | ❌ | 0.0 | 20 | 118.5 | 12741 | 8147 | 15 | MAX_STEPS_EXCEEDED |
| choose-date-easy | ❌ | 0.0 | 7 | 54.5 | 3394 | 1560 | 5 | MODEL_OUTPUT_PARSE_ERROR |
| order-food | ❌ | 0.0 | 2 | 36.3 | 921 | 10854 | 0 | MODEL_API_ERROR |

## Aggregate metrics

| metric | value |
|---|---:|
| success_rate | 5/12 = 0.417 |
| mean_reward | 0.417 |
| mean_steps | 6.33 |
| median_steps | 4.5 |
| mean_duration_s | 59.4 |
| total_input_tokens | 45,111 |
| total_output_tokens | 44,658 |
| action_error_rate (episodes with ≥1 action error) | 0.583 |

## Observations (engineering, not capability conclusions)

1. **Pipeline is healthy end to end**: all 12 episodes produced complete
   traces (manifest, steps.jsonl, artifacts, result.json) and structured
   error classification; failures were cleanly recorded rather than crashing
   the benchmark runner.
2. **Output format drift is the dominant failure (4/12)**:
   `MODEL_OUTPUT_PARSE_ERROR`. In multi-step tasks the model increasingly
   deviates from the required JSON action object (partly correlated with
   action errors in the previous step). This is exactly the kind of failure
   the Phase 1 mechanisms (verification/retry) are designed to address —
   recorded here as the baseline behavior, *not* patched away.
3. **Looping behavior**: `read-table` and `use-autocomplete` exhausted 20
   steps with many action errors, suggesting the model retries the same
   failing action — future ablation targets for verification/recovery.
4. **One infrastructure failure**: `order-food` died on a provider
   `InternalServerError` (after a very large completion of 10.8k tokens).
   Recorded as `MODEL_API_ERROR`; no retry logic exists in Phase 0 by design.
5. **Cost note**: `estimated_cost` is null (no reliable price table for the
   endpoint); token counts above are the authoritative usage record.

## Reproduction

```bash
export MODEL_BASE_URL=...   # OpenAI-compatible endpoint
export MODEL_API_KEY=...    # or fill in .env
uv run web-harness benchmark --config configs/benchmarks/miniwob_smoke.yaml
```

Full traces: `runs/run-20260912T0255xx*`..`runs/run-20260912T0302xx*` (see
`run_ids.txt` in the experiment directory).
