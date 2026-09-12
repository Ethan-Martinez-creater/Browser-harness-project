# Phase 1B Controlled Retry Report

- Experiment: `miniwob-smoke-20260912T081900Z-4fbb1c9f` (auto-generated from summary.json / episodes.csv)
- Tasks: 12 × seeds [0], serial
- Model: `deepseek-flash` (provider `openai_compatible`, temperature 0.0)
- Agent: `baseline`, max_steps 20
- Environment bootstrap: noop(wait_ms=500)
> **Scope warning**: this is the Phase 1B engineering smoke run.
It validates controlled model-side retry (API retry with fixed
backoff, one parse-repair retry, budget limits) on the natural
MiniWoB workload. It is not a model-capability conclusion, and
browser actions are never retried.

## Per-episode results

| task | seed | status | success | reward | steps | duration (s) | input tokens | output tokens | action errors | verifications | failure signals | error type |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| click-test | 0 | success | ✅ | 1 | 1 | 15.000 | 1405 | 28 | 0 | 1 | 0 | TASK_TERMINATED |
| click-button | 0 | success | ✅ | 1 | 1 | 9.891 | 1449 | 26 | 0 | 1 | 0 | TASK_TERMINATED |
| enter-text | 0 | success | ✅ | 1 | 2 | 11.687 | 2866 | 116 | 0 | 2 | 0 | TASK_TERMINATED |
| choose-list | 0 | success | ✅ | 1 | 1 | 10.282 | 1498 | 41 | 0 | 1 | 0 | TASK_TERMINATED |
| click-checkboxes | 0 | success | ✅ | 1 | 2 | 14.625 | 2989 | 74 | 0 | 2 | 0 | TASK_TERMINATED |
| click-link | 0 | success | ✅ | 1 | 4 | 48.079 | 6076 | 6182 | 1 | 4 | 3 | TASK_TERMINATED |
| login-user | 0 | success | ✅ | 1 | 3 | 16.250 | 4518 | 134 | 0 | 3 | 0 | TASK_TERMINATED |
| read-table | 0 | success | ✅ | 1 | 2 | 22.968 | 3160 | 157 | 0 | 2 | 0 | TASK_TERMINATED |
| navigate-tree | 0 | success | ✅ | 1 | 1 | 9.593 | 1481 | 117 | 0 | 1 | 0 | TASK_TERMINATED |
| use-autocomplete | 0 | success | ✅ | 1 | 3 | 22.000 | 4481 | 609 | 0 | 3 | 0 | TASK_TERMINATED |
| choose-date-easy | 0 | success | ✅ | 1 | 4 | 21.609 | 7734 | 398 | 1 | 4 | 1 | TASK_TERMINATED |
| order-food | 0 | success | ✅ | 1 | 3 | 201.203 | 5643 | 13001 | 0 | 3 | 0 | TASK_TERMINATED |

## Aggregate metrics

| metric | value |
|---|---:|
| success_rate | 1.000 (12/12) |
| mean_reward | 1.000 |
| mean_steps | 2.250 |
| median_steps | 2.0 |
| mean_duration_s | 33.599 |
| total_input_tokens | 43300 |
| total_output_tokens | 20883 |
| action_error_rate | 0.167 |

## Verification layer metrics (Phase 1A)

- reliability enabled: True
- verification enabled: True
- verification mode: shadow
- no-progress detector: True
- loop detector: True (consecutive_threshold=2)

| metric | value |
|---|---:|
| verification_count | 27 |
| verifications_with_signal | 4 |
| verification_signal_rate | 0.148 |
| mean_failure_signals_per_verification | 0.148 |
| failure_signal_count | 4 |
| episodes_with_failure_signal | 2 |

Failure kinds:

| failure kind | count |
|---|---:|
| ACTION_ERROR | 2 |
| NO_PROGRESS | 2 |

Shadow mode guarantees: verification_count == total agent steps (one verification per step), zero extra model calls, zero extra environment actions. Detection only — recovery and replanning do not exist in this phase.

## Retry layer metrics (Phase 1B)

- retry enabled: True
- model_api.max_retries: 2 (1 initial attempt + N retries), backoff_ms=[500, 1000]
- model_output.max_retries (format repair): 1
- budget.max_extra_model_calls_per_episode: 6

| metric | value |
|---|---:|
| total_retry_count | 0 |
| episodes_with_retry | 0 |
| retry_success_count | 0 |
| retry_exhausted_count | 0 |
| retry_success_rate | 0.000 |
| total_extra_model_calls | 0 |
| total_retry_input_tokens | 0 |
| total_retry_output_tokens | 0 |
| total_retry_latency_s | 0.000 |

Retry scope guarantees: only MODEL_API_ERROR and MODEL_OUTPUT_PARSE_ERROR
are retried; browser actions are never retried; retries never add an
Agent step (one Agent action = one StepRecord, retries live in
events.jsonl); tokens of failed attempts are included in episode totals.

`estimated_cost` is null by design: the harness never guesses prices without a reliable price table; token counts above are the authoritative usage record.

## Qualitative observations (all numbers above are renderer-generated)

1. **Normal tasks are not disturbed by retry logic**: every episode ran the
   standard observe → decide → act → verify → record flow; the natural
   workload produced zero provider errors this run, so zero retries were
   triggered — exactly as a disabled-by-default mechanism should behave when
   enabled but not needed.
2. **Retry behavior itself is validated separately** in the controlled fault
   suite (`phase1b_fault_suite.json` / `phase1b_fault_suite_report.md`):
   deterministic B1-B6 scenarios cover first-attempt API recovery, retry
   exhaustion, format repair, retry-disabled compatibility and the
   episode-level extra-model-call budget. Fault-suite results are never
   mixed with this natural success rate.
3. **Retry never touches the environment**: a retry is a model call inside
   one decision cycle; one Agent action always equals one StepRecord, and
   retry details live only in `events.jsonl` and metrics.
4. **Token accounting includes all attempts**: failed parse attempts carry
   their real usage on the exception (never zero), and retry tokens are a
   subset of episode totals.
5. `estimated_cost` is null by design; token counts above are the
   authoritative usage record.

## Reproduction

```bash
export MODEL_BASE_URL=...   # OpenAI-compatible endpoint (or fill .env)
export MODEL_API_KEY=...
uv run web-harness benchmark --config configs/phase1/retry.yaml
uv run python scripts/render_benchmark_report.py \
    --profile phase1b \
    --summary <experiment>/summary.json \
    --episodes <experiment>/episodes.csv \
    --output reports/phase1/phase1b_retry_report.md \
    --notes reports/phase1/phase1b_retry_notes.md
uv run python scripts/run_phase1b_fault_suite.py --output-dir reports/phase1
```
