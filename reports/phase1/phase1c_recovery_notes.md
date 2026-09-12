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
