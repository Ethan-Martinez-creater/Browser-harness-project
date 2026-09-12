# Phase 1D Controlled Fault Suite Report

Auto-generated from `phase1d_fault_suite.json` — deterministic
replanning scenarios on FakeEnvironment + MockModelAdapter + a
ScriptedStructuredModel (provider-neutral, no real model calls).
These results are NOT mixed with natural MiniWoB success rates.

| scenario | pass | status | replans | replan model calls |
|---|---|---|---:|---:|
| D1 repeated loop after failed recovery | PASS | - | 1 | 1 |
| D2 recovery failure streak triggers replan | PASS | - | 1 | 0 |
| D3 replanning disabled causal control | PASS | - | 0 | 0 |
| D4 replan budget exhausted falls back and continues | PASS | - | 1 | 0 |
| D5 plan horizon expires | PASS | - | 1 | 0 |
| D6 same failure repeats under active plan | PASS | - | 1 | 0 |
| D7 replan generation parse repair | PASS | - | 1 | 2 |
| D8a transient API failure retries then succeeds | PASS | - | 1 | 0 |
| D8b persistent replan API failure falls back | PASS | - | 1 | 0 |
| D9 terminal task never replans | PASS | - | 0 | 0 |
| D10 unresolved recovery is neutral | PASS | - | 0 | 0 |

**Suite result: ALL PASS**

Guarantees exercised: escalation is deterministic and happens only on the Phase 1C RECOVER path for ACTION_ERROR/LOOP_DETECTED; the RecoveryPlan is advisory prompt context (never executed); the plan horizon is consumed only by real Agent StepRecords; replan model calls reuse Phase 1B retry protections and cost accounting; outcome invariant success + failed + unresolved == replan_count.

