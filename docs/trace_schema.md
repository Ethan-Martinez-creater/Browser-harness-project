# Trace Schema

Tracing exists from day one because **trace is harness infrastructure, not a
UI feature**: it enables debugging, failure analysis, replay, evaluation and
later checkpoint/verification work.

## Schema versioning (Phase 2A1)

`manifest.json` carries `trace_schema_version`:

- **v1** — the Phase 0–1D layout (text observation artifacts only, no
  version field). Legacy traces are never modified; a manifest without the
  field is treated as v1 by offline tooling.
- **v2** — the current layout: adds structured Observation JSON artifacts
  (`obs_NNN.json` / `next_obs_NNN.json`) plus the additive `StepRecord`
  references `observation_json_ref` / `next_observation_json_ref`. The
  `.txt` artifacts keep their existing semantics. v2 is the first version
  consumed by offline semantic replay.

## Directory layout

```text
runs/<run_id>/
  manifest.json        run provenance (incl. trace_schema_version)
  steps.jsonl          one StepRecord per line, fsynced after every step
  result.json          final RunResult
  artifacts/
    obs_000.txt        PRE-action observation (the model's decision input)
    obs_000.json       structured Observation JSON twin (v2, additive)
    prompt_000.txt     exact system+user prompt sent to the model
    model_000.json     ModelOutput (decision, tokens; no raw CoT)
    raw_000.txt        raw model response text (if enabled)
    next_obs_000.txt   POST-action observation (result of the action)
    next_obs_000.json  structured Observation JSON twin (v2, additive)
    raw_failed_NNN.txt raw model output of a step whose parsing failed
    screenshot_*.png   optional (save_screenshots: false by default)
```

Artifact names are deterministic (zero-padded step index). The `.json`
observation twins are written by `TraceRecorder.record_step` from
`Observation.model_dump(mode="json")` — the same normalized observation the
agent actually decided on, in machine-readable form.

## Step trace semantics

```text
obs_N      →  prompt_N  →  model_N  →  action_N  →  next_obs_N
(decision input)                      (executed)   (action result)
```

`obs_(N+1)` must always equal `next_obs_N` semantically. When a step fails
before an action executes (e.g. `MODEL_OUTPUT_PARSE_ERROR`), `next_obs_N` is
left unset — it is never fabricated.

## Reliability event stream (Phase 1A)

`events.jsonl` records what reliability components did, one `RuntimeEvent`
per line (fsynced). It never changes the meaning of `steps.jsonl`.

| Field | Type | Notes |
|---|---|---|
| event_id | str | unique per event |
| run_id | str | matches the run directory |
| timestamp | ISO-8601 UTC | |
| event_type | str | `verification` / `policy_decision` / `retry` / `recovery` / `replan` / `budget` |
| step_index | int? | agent step the event belongs to |
| attempt_index | int? | for retries within one decision cycle |
| component | str | emitting component, e.g. `verification_engine` |
| outcome | str? | e.g. verification `pass` / `warning` / `fail` |
| data | object | event-specific evidence (failure signals, fingerprints, ...) |

Each verification event carries the full FailureSignal list (kind, severity,
signature, evidence) plus pre/post state fingerprints and `state_changed`.
`StepRecord` carries a compact summary (`verification_status`,
`failure_kinds`); the authoritative detail lives here.

### Phase 1C recovery events

- `recovery` events carry the trigger identity of the verified failure that
  produced the directive (`failure_kind`, `failure_signature`) plus
  `directive_kind`, `blocked_actions`; WAIT_AND_REOBSERVE additionally
  records the harness noop's real environment result (`wait_ms`,
  `action_error`, `reward`, `terminated`, `truncated`).
- `policy_decision` `outcome=abort` with `data.short_circuited=true` records
  the deterministic TASK_FAILED abort decision: **TASK_FAILED is
  terminal-short-circuited by EpisodeRunner** (the environment terminal
  check has the highest priority and runs before the policy), so the policy
  ABORT for TASK_FAILED is recorded for a consistent trace but never enters
  recovery.
- A `recovery` event with `outcome=blocked_action_selected` records that the
  agent re-selected a blocked action and re-decided without executing it.
  These re-decisions are NOT new recoveries (no directive is created) and
  are counted separately as `blocked_action_redecision_count`; they remain
  bounded by the recovery budget as an anti-loop guard.

### Recovery metrics semantics (Phase 1C)

`recovery_count` counts created/executed RecoveryDirectives only. The local
outcome invariant is `recovery_success_count + recovery_failed_count +
recovery_unresolved_count == recovery_count`; outcomes are evaluated against
the **recovery-start fingerprint** within a 2-agent-step window, and an
episode that ends before the window completes reports its pending outcomes
as explicitly unresolved (never silently dropped). The recovery budget has a
single canonical config path: `reliability.recovery.max_recoveries_per_episode`.

## manifest.json

```json
{
  "run_id": "run-...",
  "timestamp": "2026-09-12T03:00:00Z",
  "trace_schema_version": 2,
  "git_commit": "9db4d27..." ,
  "config_hash": "sha256[:16] of resolved config",
  "python_version": "3.11.16",
  "platform": "Windows-10-...",
  "model_provider": "openai_compatible",
  "model_name": "deepseek-flash",
  "benchmark": "miniwob",
  "task_id": "click-test",
  "seed": 0,
  "max_steps": 20,
  "environment_bootstrap_action": "noop(wait_ms=500)"
}
```

`environment_bootstrap_action` is provenance for the explicit, verified
environment bootstrap (see `ADR-004`); `null` when disabled.

## steps.jsonl — StepRecord per line

| Field | Type | Notes |
|---|---|---|
| run_id | str | |
| step_index | int | 0-based |
| timestamp | ISO-8601 UTC | |
| url | str? | page URL before the action (pre-action observation) |
| observation_ref | str? | `artifacts/obs_NNN.txt` — pre-action, decision input |
| next_observation_ref | str? | `artifacts/next_obs_NNN.txt` — post-action result |
| observation_json_ref | str? | `artifacts/obs_NNN.json` — structured twin (v2, additive) |
| next_observation_json_ref | str? | `artifacts/next_obs_NNN.json` — structured twin (v2, additive) |
| prompt_ref | str? | `artifacts/prompt_NNN.txt` (configurable) |
| model_response_ref | str? | `artifacts/model_NNN.json` (configurable) |
| action | str? | executed action; null when decision failed |
| action_error | str? | environment-reported error (does not stop the episode) |
| reward | float? | after the action |
| terminated / truncated | bool | environment signals |
| latency_ms | float? | decide+act wall time |
| input_tokens / output_tokens | int? | from model usage when available |
| model_name | str? | |
| short_reason | str? | brief debug summary; full CoT is never stored |
| error_type | str? | unified `ErrorType` value |

## result.json — RunResult

```json
{
  "run_id": "...", "task_spec": {...}, "status": "success|failed|truncated|max_steps_reached|error",
  "success": true, "final_reward": 1.0, "num_steps": 3,
  "started_at": "...", "ended_at": "...", "duration_s": 42.1,
  "input_tokens": 1234, "output_tokens": 567, "estimated_cost": null,
  "action_error_count": 0, "trace_path": "runs/<run_id>",
  "error_type": "TASK_TERMINATED", "error_message": null
}
```

`estimated_cost` stays null unless a reliable price table is configured —
the harness never guesses prices.

## Guarantees

1. **Durability**: every step is flushed (and fsynced) before the next one
   starts; trace write failures raise `TRACE_WRITE_ERROR` instead of being
   swallowed.
2. **Completeness**: failed episodes still produce `steps.jsonl` and
   `result.json` (with structured `error_type`).
3. **Secrets**: API keys never enter manifests, steps or artifacts
   (enforced by a unit test).
4. **Machine readability**: every file is valid JSON/JSONL and reloadable
   through `TraceRecorder.read_steps` / `read_result`.

## Future evolution

Phase 2 (checkpoint/resume) will extend RunState serialization from this
same schema; Phase 1 verifiers will append verification fields to
StepRecord without breaking existing consumers (additive only).

## Offline replay (Phase 2A1)

A recorded run can be validated entirely offline — zero model calls, zero
environment actions, and the original trace is never modified:

```bash
web-harness replay <run_id> [--runs-root runs] [--semantic/--no-semantic]
```

Pipeline: `TraceBundleLoader` → `TraceReplayEngine` → `ReplayReport`
(`src/web_harness/persistence/replay.py`). Structural violations never
raise; they are collected into the structured report.

Two replay levels:

1. **Structural replay** (all trace versions): step_index contiguity,
   run_id consistency across steps/events/result, event step_index
   validity, artifact reference existence, `result.num_steps` vs recorded
   steps, token/metric recomputation (`input_tokens = sum(steps) +
   replan_input_tokens`, action errors, verification counts) and the
   recovery / replan outcome invariants.
2. **Semantic verification replay** (v2 only): for every step with a
   recorded verification event, the recorded pre/post structured
   Observations plus the recorded action are re-fed through the
   deterministic `StepVerifier` with a fresh `ReliabilityState`, and the
   regenerated status + (kind, severity, signature) signal multiset is
   compared against the recorded verification event. Only the StepVerifier
   is replayed — never the LLM, the DecisionExecutor, the environment, the
   RecoveryManager or the Replanner. Legacy v1 traces skip this level
   (reported as a note, not an error).

The replay report counts `replayed_verifications`,
`verification_mismatches`, `artifact_missing_count`,
`invariant_error_count` and `metric_mismatches`; the CLI exits 1 when the
trace is structurally invalid or a verification mismatch is found.
