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
