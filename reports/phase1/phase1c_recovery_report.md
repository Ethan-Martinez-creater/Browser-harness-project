# Phase 1C Recovery Policy Report

- Experiment: `miniwob-smoke-20260912T121811Z-544730a8` (auto-generated from summary.json / episodes.csv)
- Tasks: 12 × seeds [0], serial
- Model: `deepseek-flash` (provider `openai_compatible`, temperature 0.0)
- Agent: `baseline`, max_steps 20
- Environment bootstrap: noop(wait_ms=500)
> **Scope warning**: this is the Phase 1C engineering smoke run.
It validates environment-side recovery (rule-based policy over
verified failure signals, budget-bounded WAIT_AND_REOBSERVE /
REDECIDE_WITH_FEEDBACK / BLOCK_REPEATED_ACTION) on the natural
MiniWoB workload. It is not a model-capability conclusion, and
no browser action is ever blindly retried.

## Per-episode results

| task | seed | status | success | reward | steps | duration (s) | input tokens | output tokens | action errors | verifications | failure signals | error type |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| click-test | 0 | success | ✅ | 1 | 1 | 48.594 | 1405 | 29 | 0 | 1 | 0 | TASK_TERMINATED |
| click-button | 0 | success | ✅ | 1 | 1 | 19.657 | 1449 | 48 | 0 | 1 | 0 | TASK_TERMINATED |
| enter-text | 0 | success | ✅ | 1 | 2 | 14.750 | 2866 | 110 | 0 | 2 | 0 | TASK_TERMINATED |
| choose-list | 0 | success | ✅ | 1 | 1 | 8.781 | 1498 | 44 | 0 | 1 | 0 | TASK_TERMINATED |
| click-checkboxes | 0 | success | ✅ | 1 | 2 | 26.110 | 2989 | 75 | 0 | 2 | 0 | TASK_TERMINATED |
| click-link | 0 | success | ✅ | 1 | 6 | 146.375 | 9365 | 13854 | 2 | 6 | 5 | TASK_TERMINATED |
| login-user | 0 | success | ✅ | 1 | 3 | 20.594 | 4518 | 193 | 0 | 3 | 0 | TASK_TERMINATED |
| read-table | 0 | success | ✅ | 1 | 2 | 13.954 | 3160 | 124 | 0 | 2 | 0 | TASK_TERMINATED |
| navigate-tree | 0 | success | ✅ | 1 | 1 | 8.719 | 1481 | 96 | 0 | 1 | 0 | TASK_TERMINATED |
| use-autocomplete | 0 | success | ✅ | 1 | 4 | 25.437 | 6044 | 521 | 0 | 4 | 1 | TASK_TERMINATED |
| choose-date-easy | 0 | success | ✅ | 1 | 4 | 22.062 | 7860 | 309 | 1 | 4 | 1 | TASK_TERMINATED |
| order-food | 0 | failed | ❌ | 0 | 3 | 626.578 | 5625 | 16960 | 0 | 3 | 3 | TASK_TERMINATED |

## Aggregate metrics

| metric | value |
|---|---:|
| success_rate | 0.917 (11/12) |
| mean_reward | 0.917 |
| mean_steps | 2.500 |
| median_steps | 2.0 |
| mean_duration_s | 81.801 |
| total_input_tokens | 48260 |
| total_output_tokens | 32363 |
| action_error_rate | 0.167 |

## Verification layer metrics (Phase 1A)

- reliability enabled: True
- verification enabled: True
- verification mode: shadow
- no-progress detector: True
- loop detector: True (consecutive_threshold=2)

| metric | value |
|---|---:|
| verification_count | 30 |
| verifications_with_signal | 10 |
| verification_signal_rate | 0.333 |
| mean_failure_signals_per_verification | 0.333 |
| failure_signal_count | 10 |
| episodes_with_failure_signal | 4 |

Failure kinds:

| failure kind | count |
|---|---:|
| ACTION_ERROR | 3 |
| NO_PROGRESS | 6 |
| TASK_FAILED | 1 |

Shadow mode guarantees: verification_count == total agent steps (one verification per step), zero extra model calls, zero extra environment actions. Detection only — recovery and replanning do not exist in this phase.

## Retry layer metrics (Phase 1B)

- retry enabled: True
- model_api.max_retries: 2 (1 initial attempt + N retries), backoff_ms=[500, 1000]
- model_output.max_retries (format repair): 1
- budget.max_extra_model_calls_per_episode: 6

| metric | value |
|---|---:|
| total_retry_count | 2 |
| retry_cycle_count | 1 |
| episodes_with_retry | 1 |
| retry_success_count | 1 |
| retry_exhausted_count | 0 |
| retry_success_rate | 1.000 |
| total_extra_model_calls | 2 |
| total_retry_input_tokens | 1867 |
| total_retry_output_tokens | 10225 |
| total_retry_latency_s | 329.672 |

Retry scope guarantees: only MODEL_API_ERROR and MODEL_OUTPUT_PARSE_ERROR
are retried; browser actions are never retried; retries never add an
Agent step (one Agent action = one StepRecord, retries live in
events.jsonl); tokens of failed attempts are included in episode totals.

## Recovery layer metrics (Phase 1C)

- recovery enabled: True
- recovery.max_recoveries_per_episode: 3 (canonical recovery budget path)
- recovery.wait_ms (WAIT_AND_REOBSERVE noop wait): 500
- recovery.block_steps (REDECIDE_WITH_FEEDBACK block): 1
- budget.max_extra_model_calls_per_episode: 6

| metric | value |
|---|---:|
| total_recovery_count | 3 |
| recovery_success_count | 2 |
| recovery_failed_count | 1 |
| recovery_unresolved_count | 0 |
| episodes_with_recovery | 2 |
| recovered_episode_count | 2 |
| recovery_environment_actions | 0 |
| blocked_action_redecision_count | 0 |
| total_recovery_latency_s | 0.219 |

Recovery scope guarantees: recovery is triggered only by the
deterministic rule policy over verified failure signals; an
exhausted budget never kills a PASS / single-NO_PROGRESS step; a
recovery environment action (WAIT_AND_REOBSERVE noop) is never
counted as an Agent step and never becomes a StepRecord; blocked
actions never reach env.step (re-selections are counted separately
as blocked_action_redecision_count); success + failed + unresolved
sum to total_recovery_count; recovery is bounded by
recovery.max_recoveries_per_episode and independent of the Phase 1B
extra-model-call budget.

`estimated_cost` is null by design: the harness never guesses prices without a reliable price table; token counts above are the authoritative usage record.

## Qualitative observations (all numbers above are renderer-generated)

1. **Normal tasks are not disturbed by recovery logic**: every episode ran the
   standard observe → decide → act → verify → policy → record flow; with the
   rule policy installed, a step only diverges from Phase 1B behavior when a
   verified failure signal is present. The natural workload is expected to
   produce few or no recoveries this run — exactly as a rule-based mechanism
   should behave when enabled but not needed.
2. **Recovery behavior itself is validated separately** in the controlled
   fault suite (`phase1c_fault_suite.json` / `phase1c_fault_suite_report.md`):
   deterministic C1-C8 scenarios cover action-error re-decision with a
   blocked action, wait-and-reobserve with a harness-owned noop, loop
   blocking, recovery-budget exhaustion, single no-progress tolerance, the
   finality of TASK_FAILED, re-selection of a blocked action, and a recovery
   noop that terminates the task. Fault-suite results are never mixed with
   this natural success rate.
3. **Recovery never blind-retries a browser action**: ACTION_ERROR triggers
   REDECIDE_WITH_FEEDBACK (the failed action is blocked and cannot enter
   `env.step` again); retries remain a model-side concern (Phase 1B), and
   recovery means a changed execution context, never a repeated action.
4. **A recovery environment action is not an agent step**: the
   WAIT_AND_REOBSERVE noop is harness-owned, consumes recovery budget, is
   traced as a RECOVERY event (with wait_ms, reward, terminated, truncated)
   and produces no StepRecord. `steps` above counts agent steps only.
5. **Local recovery outcomes are measured**: within 2 agent steps after a
   recovery, a changed state fingerprint (or task success) without a repeated
   failure signature counts as `recovery_success_count`; otherwise
   `recovery_failed_count`. Bounded and deterministic, no LLM judgment.
6. `estimated_cost` is null by design; token counts above are the
   authoritative usage record.

## Remediation note (Phase 1C closure review)

This smoke run was produced by the pre-remediation runtime. The remediation
changed recovery accounting semantics without changing the real execution
trajectory (identical policy decisions, model calls and environment actions):

- `recovery_count` now counts created/executed RecoveryDirectives only;
  blocked-action re-selections are counted separately as
  `blocked_action_redecision_count` (the machine result above predates this
  split, so its per-episode `recovery_count` may include what is now a
  re-decision).
- local outcomes are evaluated against the recovery-start fingerprint and
  episodes may report `recovery_unresolved_count`; `recovery_success_count +
  recovery_failed_count + recovery_unresolved_count == recovery_count`.
- the recovery budget moved to its canonical config path
  `reliability.recovery.max_recoveries_per_episode`.

The full C1-C13 deterministic fault suite (`phase1c_fault_suite.json`) was
re-run against the remediated runtime and passes, including the new budget /
outcome / fingerprint semantics.

## Reproduction

```bash
export MODEL_BASE_URL=...   # OpenAI-compatible endpoint (or fill .env)
export MODEL_API_KEY=...
uv run web-harness benchmark --config configs/phase1/recovery.yaml
uv run python scripts/render_benchmark_report.py \
    --profile phase1c \
    --summary <experiment>/summary.json \
    --episodes <experiment>/episodes.csv \
    --output reports/phase1/phase1c_recovery_report.md \
    --notes reports/phase1/phase1c_recovery_notes.md
uv run python scripts/run_phase1c_fault_suite.py --output-dir reports/phase1
```
