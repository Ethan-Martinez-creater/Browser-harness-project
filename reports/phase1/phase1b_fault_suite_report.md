# Phase 1B Controlled Fault Suite Report

Auto-generated from `phase1b_fault_suite.json` — deterministic
fault-injection scenarios on FakeEnvironment + MockModelAdapter.
These results are NOT mixed with natural MiniWoB success rates.

| scenario | pass | status | retries | env actions | error type |
|---|---|---|---:|---:|---|
| B1 | PASS | success | 1 | 1 | TASK_TERMINATED |
| B2 | PASS | success | 2 | 1 | TASK_TERMINATED |
| B3 | PASS | error | 2 | 0 | RETRY_EXHAUSTED |
| B4 | PASS | success | 1 | 1 | TASK_TERMINATED |
| B5 | PASS | error | 0 | 0 | MODEL_API_ERROR |
| B6 | PASS | error | 1 | 0 | BUDGET_EXCEEDED |
| M1 | PASS | success | 2 | 1 | TASK_TERMINATED |
| M2 | PASS | success | 3 | 1 | TASK_TERMINATED |
| M3 | PASS | error | 2 | 0 | BUDGET_EXCEEDED |

**Suite result: ALL PASS**

