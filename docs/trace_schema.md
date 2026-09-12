# Trace Schema

Tracing exists from day one because **trace is harness infrastructure, not a
UI feature**: it enables debugging, failure analysis, replay, evaluation and
later checkpoint/verification work.

## Directory layout

```text
runs/<run_id>/
  manifest.json        run provenance
  steps.jsonl          one StepRecord per line, fsynced after every step
  result.json          final RunResult
  artifacts/
    obs_000.txt        PRE-action observation (the model's decision input)
    prompt_000.txt     exact system+user prompt sent to the model
    model_000.json     ModelOutput (decision, tokens; no raw CoT)
    raw_000.txt        raw model response text (if enabled)
    next_obs_000.txt   POST-action observation (result of the action)
    raw_failed_NNN.txt raw model output of a step whose parsing failed
    screenshot_*.png   optional (save_screenshots: false by default)
```

Artifact names are deterministic (zero-padded step index).

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
