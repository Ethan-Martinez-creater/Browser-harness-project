# Phase 1A Verification (Shadow Mode) Report

- Experiment: `miniwob-smoke-20260912T072424Z-35a49375` (auto-generated from summary.json / episodes.csv)
- Tasks: 12 × seeds [0], serial
- Model: `deepseek-flash` (provider `openai_compatible`, temperature 0.0)
- Agent: `baseline`, max_steps 20
- Environment bootstrap: noop(wait_ms=500)
> **Scope warning**: this is the Phase 1A engineering smoke run.
It validates the failure-observation layer (deterministic
fingerprinting, detectors, verification events) in shadow mode and
establishes the verification reference. It is not a
model-capability conclusion, and shadow mode does not change the
control flow.

## Per-episode results

| task | seed | status | success | reward | steps | duration (s) | input tokens | output tokens | action errors | verifications | failure signals | error type |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| click-test | 0 | success | ✅ | 1 | 1 | 18.360 | 1405 | 51 | 0 | 1 | 0 | TASK_TERMINATED |
| click-button | 0 | success | ✅ | 1 | 1 | 9.078 | 1449 | 44 | 0 | 1 | 0 | TASK_TERMINATED |
| enter-text | 0 | success | ✅ | 1 | 2 | 17.875 | 2866 | 99 | 0 | 2 | 0 | TASK_TERMINATED |
| choose-list | 0 | success | ✅ | 1 | 1 | 9.625 | 1498 | 41 | 0 | 1 | 0 | TASK_TERMINATED |
| click-checkboxes | 0 | success | ✅ | 1 | 2 | 13.140 | 2989 | 115 | 0 | 2 | 0 | TASK_TERMINATED |
| click-link | 0 | success | ✅ | 1 | 5 | 166.734 | 7636 | 16054 | 1 | 5 | 4 | TASK_TERMINATED |
| login-user | 0 | success | ✅ | 1 | 3 | 16.110 | 4518 | 191 | 0 | 3 | 0 | TASK_TERMINATED |
| read-table | 0 | success | ✅ | 1 | 2 | 12.110 | 3160 | 155 | 0 | 2 | 0 | TASK_TERMINATED |
| navigate-tree | 0 | success | ✅ | 1 | 1 | 10.875 | 1481 | 86 | 0 | 1 | 0 | TASK_TERMINATED |
| use-autocomplete | 0 | success | ✅ | 1 | 3 | 17.500 | 4481 | 423 | 0 | 3 | 0 | TASK_TERMINATED |
| choose-date-easy | 0 | success | ✅ | 1 | 4 | 22.734 | 7734 | 321 | 1 | 4 | 1 | TASK_TERMINATED |
| order-food | 0 | failed | ❌ | 0 | 3 | 332.016 | 5625 | 21251 | 0 | 3 | 3 | TASK_TERMINATED |

## Aggregate metrics

| metric | value |
|---|---:|
| success_rate | 0.917 (11/12) |
| mean_reward | 0.917 |
| mean_steps | 2.333 |
| median_steps | 2.0 |
| mean_duration_s | 53.846 |
| total_input_tokens | 44842 |
| total_output_tokens | 38831 |
| action_error_rate | 0.167 |

## Verification layer metrics (Phase 1A)

- reliability enabled: True
- verification enabled: True
- verification mode: shadow
- no-progress detector: True
- loop detector: True (consecutive_threshold=2)

| metric | value |
|---|---:|
| verification_count | 28 |
| verifications_with_signal | 8 |
| verification_signal_rate | 0.286 |
| mean_failure_signals_per_verification | 0.286 |
| failure_signal_count | 8 |
| episodes_with_failure_signal | 3 |

Failure kinds:

| failure kind | count |
|---|---:|
| ACTION_ERROR | 2 |
| NO_PROGRESS | 5 |
| TASK_FAILED | 1 |

Shadow mode guarantees: verification_count == total agent steps (one verification per step), zero extra model calls, zero extra environment actions. Detection only — recovery and replanning do not exist in this phase.

`estimated_cost` is null by design: the harness never guesses prices without a reliable price table; token counts above are the authoritative usage record.

## Qualitative observations (all numbers above are renderer-generated)

1. **Measurement system validated end to end**: every episode produced a
   complete trace whose `events.jsonl` contains exactly one verification
   event per agent step, with deterministic pre/post state fingerprints and
   the full FailureSignal list; `steps.jsonl` semantics are unchanged.
2. **Detection is observation only**: shadow mode added no agent steps, no
   environment actions and no model calls relative to the Phase 0 baseline —
   this is enforced structurally and by
   `tests/regression/test_phase0_baseline_compat.py`, and visible in the
   per-episode table (verification count equals step count everywhere).
3. **Detectors behaved as designed in natural trajectories**: the failed
   episode was observed by the TaskFailureDetector, an environment-side
   action error produced an ACTION_ERROR signal whose signature identifies
   both the action type and the normalized error, and state-unchanged steps
   produced state-specific NO_PROGRESS signatures (WARNING on first
   occurrence).
4. **No reliability actions exist in this phase**: retry, recovery and
   replanning are not implemented; every signal was recorded and left for the
   policy layer of later phases.
5. `estimated_cost` is null by design; token counts above are the
   authoritative usage record.

## Reproduction

```bash
export MODEL_BASE_URL=...   # OpenAI-compatible endpoint (or fill .env)
export MODEL_API_KEY=...
uv run web-harness benchmark --config configs/phase1/verifier_shadow.yaml
uv run python scripts/render_benchmark_report.py \
    --profile phase1a \
    --summary <experiment>/summary.json \
    --episodes <experiment>/episodes.csv \
    --output reports/phase1/phase1a_verifier_report.md \
    --notes reports/phase1/phase1a_verifier_notes.md
```
