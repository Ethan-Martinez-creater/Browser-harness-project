# Phase 1D Controlled Replanning Report

- Experiment: `miniwob-smoke-20260912T161444Z-89f98f1e` (auto-generated from summary.json / episodes.csv)
- Tasks: 12 × seeds [0], serial
- Model: `deepseek-flash` (provider `openai_compatible`, temperature 0.0)
- Agent: `baseline`, max_steps 20
- Environment bootstrap: noop(wait_ms=500)
> **Scope warning**: this is the Phase 1D engineering smoke run.
It validates exception-path controlled replanning (deterministic
escalation trigger, budget-bounded short-horizon RecoveryPlan as
advisory prompt context) on the natural MiniWoB workload. It is
not a model-capability conclusion; there is no always-on planner
and no plan executor.

## Per-episode results

| task | seed | status | success | reward | steps | duration (s) | input tokens | output tokens | action errors | verifications | failure signals | error type |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| click-test | 0 | success | ✅ | 1 | 1 | 22.109 | 1405 | 30 | 0 | 1 | 0 | TASK_TERMINATED |
| click-button | 0 | success | ✅ | 1 | 1 | 8.609 | 1449 | 42 | 0 | 1 | 0 | TASK_TERMINATED |
| enter-text | 0 | success | ✅ | 1 | 2 | 21.218 | 2866 | 78 | 0 | 2 | 0 | TASK_TERMINATED |
| choose-list | 0 | success | ✅ | 1 | 1 | 9.688 | 1498 | 42 | 0 | 1 | 0 | TASK_TERMINATED |
| click-checkboxes | 0 | success | ✅ | 1 | 2 | 17.500 | 2989 | 102 | 0 | 2 | 0 | TASK_TERMINATED |
| click-link | 0 | success | ✅ | 1 | 4 | 54.828 | 6142 | 6602 | 1 | 4 | 3 | TASK_TERMINATED |
| login-user | 0 | success | ✅ | 1 | 3 | 19.438 | 4518 | 171 | 0 | 3 | 0 | TASK_TERMINATED |
| read-table | 0 | success | ✅ | 1 | 2 | 17.531 | 3160 | 111 | 0 | 2 | 0 | TASK_TERMINATED |
| navigate-tree | 0 | success | ✅ | 1 | 1 | 11.875 | 1481 | 111 | 0 | 1 | 0 | TASK_TERMINATED |
| use-autocomplete | 0 | success | ✅ | 1 | 3 | 28.735 | 4481 | 743 | 0 | 3 | 0 | TASK_TERMINATED |
| choose-date-easy | 0 | success | ✅ | 1 | 4 | 78.140 | 7860 | 365 | 1 | 4 | 1 | TASK_TERMINATED |
| order-food | 0 | failed | ❌ | 0 | 3 | 429.156 | 5625 | 24623 | 0 | 3 | 3 | TASK_TERMINATED |

## Aggregate metrics

| metric | value |
|---|---:|
| success_rate | 0.917 (11/12) |
| mean_reward | 0.917 |
| mean_steps | 2.250 |
| median_steps | 2.0 |
| mean_duration_s | 59.902 |
| total_input_tokens | 43474 |
| total_output_tokens | 33020 |
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
| verifications_with_signal | 7 |
| verification_signal_rate | 0.259 |
| mean_failure_signals_per_verification | 0.259 |
| failure_signal_count | 7 |
| episodes_with_failure_signal | 3 |

Failure kinds:

| failure kind | count |
|---|---:|
| ACTION_ERROR | 2 |
| NO_PROGRESS | 4 |
| TASK_FAILED | 1 |

Shadow mode guarantees: verification_count == total agent steps (one verification per step), zero extra model calls, zero extra environment actions. Detection only — recovery and replanning do not exist in this phase.

## Retry layer metrics (Phase 1B)

- retry enabled: True
- model_api.max_retries: 2 (1 initial attempt + N retries), backoff_ms=[500, 1000]
- model_output.max_retries (format repair): 1
- budget.max_extra_model_calls_per_episode: 6

| metric | value |
|---|---:|
| total_retry_count | 1 |
| retry_cycle_count | 1 |
| episodes_with_retry | 1 |
| retry_success_count | 1 |
| retry_exhausted_count | 0 |
| retry_success_rate | 1.000 |
| total_extra_model_calls | 1 |
| total_retry_input_tokens | 1867 |
| total_retry_output_tokens | 12938 |
| total_retry_latency_s | 179.235 |

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
| total_recovery_count | 2 |
| recovery_success_count | 2 |
| recovery_failed_count | 0 |
| recovery_unresolved_count | 0 |
| episodes_with_recovery | 2 |
| recovered_episode_count | 2 |
| recovery_environment_actions | 0 |
| blocked_action_redecision_count | 0 |
| total_recovery_latency_s | 0.032 |

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

## Replan layer metrics (Phase 1D)

- replanning enabled: True
- replanning.max_replans_per_episode: 1
- replanning.recovery_failures_before_replan: 2
- replanning.plan_horizon_steps: 3
- replanning.recent_steps (bounded replanner input): 6

| metric | value |
|---|---:|
| total_replan_count | 0 |
| episodes_with_replan | 0 |
| replan_success_count | 0 |
| replan_failed_count | 0 |
| replan_unresolved_count | 0 |
| replan_success_rate | 0.000 |
| total_replan_model_calls | 0 |
| total_replan_input_tokens | 0 |
| total_replan_output_tokens | 0 |
| total_replan_latency_s | 0.000 |

Replan scope guarantees: replanning is escalation-only (never on
PASS / single NO_PROGRESS / model-side failures / first observation
recovery / TASK_FAILED); the RecoveryPlan is advisory prompt context
for the reactive agent, never executed; plan horizon is consumed
only by real Agent StepRecords; outcome invariant success + failed
+ unresolved == replan_count; replan model calls share the Phase 1B
retry protections and the global extra-model-call budget.

## Reliability total overhead (Phase 1)

| metric | value |
|---|---:|
| reliability_extra_model_calls | 1 |
| reliability_extra_tokens | 14805 |
| reliability_extra_latency_s | 179.267 |

Total overhead = retry extra model calls + replan model calls;
tokens = retry + replan tokens; latency = retry + recovery + replan.

`estimated_cost` is null by design: the harness never guesses prices without a reliable price table; token counts above are the authoritative usage record.

## Qualitative observations (all numbers above are renderer-generated)

1. **The normal path is not polluted by a planner**: every episode still runs
   the standard observe → decide → act → verify → record flow with the
   unchanged BaselineAgent (Observation → one ActionDecision). The Replanner
   exists only on the exception escalation path; zero replans in a natural
   run is an acceptable and expected outcome, and no natural failures were
   manufactured for this smoke.
2. **Escalation is deterministic and rare by design**: a replan can only be
   triggered on the Phase 1C RECOVER path, only for ACTION_ERROR or
   LOOP_DETECTED, only by the two fixed rules (repeated loop after a FAILED
   recovery, or a streak of consecutive recovery failures), and at most once
   per episode. MODEL_API_ERROR / MODEL_OUTPUT_PARSE_ERROR / first
   OBSERVATION_INVALID recovery / TASK_FAILED never escalate.
3. **Replan behavior itself is validated separately** in the controlled fault
   suite (`phase1d_fault_suite.json` / `phase1d_fault_suite_report.md`):
   deterministic D1-D10 scenarios cover the two trigger rules, the disabled
   causal control, budget-exhausted fallback, plan horizon consumption, plan
   failure on repeated trigger signature, parse repair, API failure
   fallback, terminal-task neutrality and unresolved-recovery neutrality.
   Fault-suite results are never mixed with this natural success rate.
4. **The RecoveryPlan is advisory context, never executed**: it reaches the
   agent only as a prompt section (`# Recovery plan`), an active recovery
   directive remains the hard constraint, and the plan horizon is consumed
   only by real Agent StepRecords (retries, re-decisions and recovery noops
   never consume it).
5. **Replan cost is fully accounted**: replan model calls reuse the Phase 1B
   retry protections (attempt artifacts, token/latency accounting) and are
   reported both per-layer (replan_*) and in the reliability total overhead
   (reliability_extra_model_calls / tokens / latency).
6. `estimated_cost` is null by design; token counts above are the
   authoritative usage record.

## Reproduction

```bash
export MODEL_BASE_URL=...   # OpenAI-compatible endpoint (or fill .env)
export MODEL_API_KEY=...
uv run web-harness benchmark --config configs/phase1/replanning.yaml
uv run python scripts/render_benchmark_report.py \
    --profile phase1d \
    --summary <experiment>/summary.json \
    --episodes <experiment>/episodes.csv \
    --output reports/phase1/phase1d_replan_report.md \
    --notes reports/phase1/phase1d_replan_notes.md
uv run python scripts/run_phase1d_fault_suite.py --output-dir reports/phase1
```
