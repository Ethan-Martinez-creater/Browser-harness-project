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
