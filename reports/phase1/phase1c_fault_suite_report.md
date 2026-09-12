# Phase 1C Controlled Fault Suite Report

Auto-generated from `phase1c_fault_suite.json` — deterministic
environment-side fault-injection scenarios on FakeEnvironment +
MockModelAdapter, with the rule policy and RecoveryManager wired in.
These results are NOT mixed with natural MiniWoB success rates.

| scenario | pass | status | recoveries | recovery env actions | error type |
|---|---|---|---:|---:|---|
| C1 action_error -> redecide_with_feedback | PASS | success | 1 | 0 | TASK_TERMINATED |
| C2 empty_observation -> wait_and_reobserve | PASS | success | 1 | 1 | - |
| C3 loop -> block_repeated_action | PASS | success | 1 | 0 | - |
| C4 persistent action_error -> budget exhausted | PASS | error | 3 | 0 | RECOVERY_FAILED |
| C5 single no_progress -> continue | PASS | success | 0 | 0 | - |
| C6 task_failed is final | PASS | failed | 0 | 0 | TASK_TERMINATED |
| C7 blocked action re-selected | PASS | success | 2 | 0 | - |
| C8 recovery noop terminates the task | PASS | success | 1 | 1 | - |

**Suite result: ALL PASS**

Guarantees exercised: a blocked action never reaches `env.step`; a recovery environment action is never an Agent step and never produces a StepRecord; the recovery's real environment result (including termination) is kept; recovery is bounded by `max_recoveries_per_episode` (controlled abort); TASK_FAILED is final and never reset.

