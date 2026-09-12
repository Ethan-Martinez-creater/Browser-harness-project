# Architecture

## Overview

Phase 0 implements the smallest slice of the harness that is still fully
traceable and evaluable:

```text
                         ┌───────────────────┐
                         │    CLI / Config   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   EpisodeRunner   │
                         │  Harness Runtime  │
                         └─────────┬─────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
        ┌────────────────┐ ┌────────────────┐ ┌──────────────┐
        │ BaselineAgent  │ │  RunState      │ │TraceRecorder │
        └───────┬────────┘ └────────────────┘ └──────────────┘
                │
                ▼
        ┌────────────────┐
        │  ModelAdapter  │
        └───────┬────────┘
                ▼
        ┌────────────────┐
        │ ActionDecision │
        └───────┬────────┘
                ▼
        ┌────────────────┐
        │Environment     │
        │Adapter         │
        └───────┬────────┘
                ▼
           BrowserGym
                ▼
           Chromium
```

## Layers and import rules

| Layer | Module | May import | Must never import |
|---|---|---|---|
| core | `core/` | stdlib, pydantic | anything in the project |
| models | `models/` | core | env, agents, runtime, evaluation, browsergym |
| agents | `agents/` | core, models | env, runtime, evaluation, browsergym |
| env | `env/` | core | models(adapter ok), agents, runtime, **browsergym outside adapter** |
| runtime | `runtime/` | core, agents, env(base), observability | browsergym, openai |
| observability | `observability/` | core | everything else |
| evaluation | `evaluation/` | core, agents, models, env, runtime, config | — |

Enforced invariants:

1. **Self-owned runtime.** `EpisodeRunner` contains the explicit
   `observe → decide → act → record → update → terminate?` loop. No
   LangGraph / Deep Agents / CrewAI / OpenAI Agents SDK anywhere.
2. **BrowserGym isolation.** `browsergym`/`gymnasium` imports exist only in
   `env/browsergym_adapter.py` (plus the env probe script). Everything else
   works with `TaskSpec` / `Observation` / `EnvironmentStep`.
3. **Model provider isolation.** `openai` imports exist only in
   `models/openai_compatible.py`. Agent code sees a `ModelAdapter` protocol.
4. **Observation normalization.** Raw BrowserGym observations (objects) are
   converted to text by `ObservationNormalizer` with deterministic
   truncation; no LLM-based compression in Phase 0.
5. **Single-source action contract.** The environment adapter builds one
   BrowserGym `HighLevelActionSet` and installs it as the env's action
   mapping; the prompt-facing `ActionContract` is generated from that same
   instance (`env/action_contract.py`), so every action advertised to the
   agent is guaranteed parseable and executable by the environment
   (`ADR-002`).
6. **Step trace semantics.** `obs_N` is the pre-action decision input and
   matches `prompt_N`; `next_obs_N` is the post-action result, and
   `obs_(N+1) ≡ next_obs_N` (`trace_schema.md`).

## Data flow per step

```text
RunState.steps (bounded history)
        ↓
BaselineAgent.decide → PromptBuilder → PromptBundle
        ↓
ModelAdapter.generate_action → ModelOutput(ActionDecision, tokens, raw_text)
        ↓
EnvironmentAdapter.step(action) → EnvironmentStep(observation, reward, terminated, error)
        ↓
TraceRecorder.record_step → steps.jsonl + artifacts (obs/prompt/model/raw)
        ↓
RunState.update → termination check (terminated / truncated / max_steps / fatal)
        ↓
TraceRecorder.write_result → result.json
```

## Termination rules (Phase 0)

| Condition | Status | ErrorType |
|---|---|---|
| environment `terminated`, reward > 0 | `success` | `TASK_TERMINATED` |
| environment `terminated`, reward == 0 | `failed` | `TASK_TERMINATED` |
| environment `truncated` (time limit) | `truncated` | `TASK_TRUNCATED` |
| step budget exhausted | `max_steps_reached` | `MAX_STEPS_EXCEEDED` |
| model output unparseable | `error` | `MODEL_OUTPUT_PARSE_ERROR` |
| model API failure | `error` | `MODEL_API_ERROR` |
| environment init failure | `error` | `ENVIRONMENT_INIT_ERROR` |
| any other exception | `error` | `UNKNOWN_ERROR` / `ACTION_EXECUTION_ERROR` |

Environment-side action errors do **not** terminate the episode: they are
recorded on the StepRecord and the next observation lets the model react. The
environment is always closed in a `finally` block.

## Extension points (later phases)

- `ModelAdapter` protocol → more providers, cost tables
- `EnvironmentAdapter` protocol → WebArena, WorkArena, custom environments
- `RunState` schema → checkpoint/resume (Phase 2)
- `EpisodeRunner` loop → verifier, retry, recovery, budget hooks (Phase 1)
- config sections per mechanism → ablation switches (every later module must
  be toggleable: `planner.enabled: false`, ...)
