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
