# Phase 1 Final Ablation Report (A0-A4)

- Frozen implementation commit: `1436cb2207ec53a050a656d01bd4775f93e919ba`
- 12 MiniWoB tasks x seeds [0, 1, 2] x 5 variants = 180 episodes, serial
- Model: `deepseek-flash`, temperature 0.0, identical endpoint, tasks, seeds, ActionContract, bootstrap and runtime across variants
- The ONLY difference between variants is the reliability capability switches
- Auto-generated from `phase1_ablation_summary.json` — no hand-written numbers

## 1. Main metrics comparison

| metric | A0 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| episodes | 36 | 36 | 36 | 36 | 36 |
| successes | 32 | 30 | 34 | 32 | 31 |
| task success rate | 0.889 | 0.833 | 0.944 | 0.889 | 0.861 |
| mean reward | 0.889 | 0.833 | 0.944 | 0.889 | 0.861 |
| mean steps | 2.89 | 2.58 | 2.81 | 2.47 | 2.72 |
| median steps | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| mean duration (s) | 43.9 | 35.1 | 42.5 | 32.4 | 47.6 |
| input tokens | 169738 | 149503 | 170224 | 148607 | 164407 |
| output tokens | 94671 | 74814 | 119519 | 59012 | 114989 |
| total tokens | 264409 | 224317 | 289743 | 207619 | 279396 |
| tokens per success | 8263 | 7477 | 8522 | 6488 | 9013 |
| action error rate | 0.167 | 0.167 | 0.167 | 0.194 | 0.139 |

## 2. Reliability overhead comparison

| metric | A0 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| verification count | 0 | 91 | 101 | 89 | 98 |
| failure signal count | 0 | 26 | 30 | 24 | 32 |
| episodes with failure signal | 0 | 10 | 10 | 10 | 10 |
| retry attempts | 0 | 0 | 1 | 0 | 1 |
| retry cycles | 0 | 0 | 1 | 0 | 1 |
| retry success rate (cycles) | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| retry exhausted | 0 | 0 | 0 | 0 | 0 |
| extra model calls (retry) | 0 | 0 | 1 | 0 | 1 |
| recovery count | 0 | 0 | 0 | 9 | 8 |
| recovery success | 0 | 0 | 0 | 5 | 3 |
| recovery failed | 0 | 0 | 0 | 3 | 5 |
| recovery unresolved | 0 | 0 | 0 | 1 | 0 |
| recovered episodes | 0 | 0 | 0 | 6 | 3 |
| recovery env actions | 0 | 0 | 0 | 0 | 0 |
| recovery latency (s) | 0.0 | 0.0 | 0.0 | 0.3 | 0.2 |
| replan count | 0 | 0 | 0 | 0 | 1 |
| replan success | 0 | 0 | 0 | 0 | 0 |
| replan failed | 0 | 0 | 0 | 0 | 1 |
| replan unresolved | 0 | 0 | 0 | 0 | 0 |
| replan model calls | 0 | 0 | 0 | 0 | 1 |
| replan tokens | 0 | 0 | 0 | 0 | 1598 |
| replan latency (s) | 0.0 | 0.0 | 0.0 | 0.0 | 5.7 |
| reliability extra model calls | 0 | 0 | 1 | 0 | 2 |
| reliability extra tokens | 0 | 0 | 14637 | 0 | 9810 |
| reliability extra latency (s) | 0.0 | 0.0 | 59.7 | 0.3 | 159.8 |

Task success, local retry success, local recovery success and local replan success are DIFFERENT metrics and are never merged into one success rate.

## 3. FailureKind distribution

| failure kind | A0 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| ACTION_ERROR | 0 | 7 | 13 | 9 | 10 |
| LOOP_DETECTED | 0 | 1 | 0 | 0 | 0 |
| NO_PROGRESS | 0 | 14 | 15 | 11 | 18 |
| TASK_FAILED | 0 | 4 | 2 | 4 | 4 |

## 4. Mechanism triggers (actual firings)

| mechanism | A0 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| retry attempts fired | 0 | 0 | 1 | 0 | 1 |
| recoveries fired | 0 | 0 | 0 | 9 | 8 |
| replans fired | 0 | 0 | 0 | 0 | 1 |

Variants where a mechanism never triggered:

- retry: never fired in A3

## 5. Per-seed results

| variant | seed | episodes | successes | mean steps | input tokens | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0 | 12 | 11 | 2.17 | 39203 | 10692 |
| A0 | 1 | 12 | 11 | 4.25 | 89385 | 70119 |
| A0 | 2 | 12 | 10 | 2.25 | 41150 | 13860 |
| A1 | 0 | 12 | 10 | 2.42 | 43970 | 19994 |
| A1 | 1 | 12 | 11 | 3.17 | 62256 | 35169 |
| A1 | 2 | 12 | 9 | 2.17 | 43277 | 19651 |
| A2 | 0 | 12 | 11 | 2.33 | 44916 | 33570 |
| A2 | 1 | 12 | 12 | 3.83 | 81897 | 69778 |
| A2 | 2 | 12 | 11 | 2.25 | 43411 | 16171 |
| A3 | 0 | 12 | 10 | 2.25 | 43474 | 18196 |
| A3 | 1 | 12 | 11 | 3.00 | 62491 | 22121 |
| A3 | 2 | 12 | 11 | 2.17 | 42642 | 18695 |
| A4 | 0 | 12 | 10 | 2.42 | 46697 | 43243 |
| A4 | 1 | 12 | 10 | 3.42 | 72659 | 40103 |
| A4 | 2 | 12 | 11 | 2.33 | 45051 | 31643 |

## 6. 3-seed aggregate

The main comparison table above IS the 3-seed aggregate (seeds 0+1+2 pooled per variant).

## 7. Interpretation guardrails

- Single-episode fluctuations at this scale are NOT evidence of a mechanism's benefit; the ablation quantifies cost and observable behavior, not statistical significance.
- A mechanism with zero firings in a variant contributed zero cost AND zero benefit there.
- tokens_per_success is undefined (null) for variants with zero successful episodes.

## Experiment IDs

- A0 (Baseline (reliability disabled)): `miniwob-smoke-20260913T055448Z-bd51e0fd` (config_hash `8be8c99fa2c4df1a`)
- A1 (Detection only): `miniwob-smoke-20260913T062656Z-c7f599a2` (config_hash `1e91f65f16aa9cad`)
- A2 (Detection + Retry): `miniwob-smoke-20260913T065908Z-cc2f802b` (config_hash `db59386309527437`)
- A3 (Detection + Retry + Recovery): `miniwob-smoke-20260913T073109Z-21b34c66` (config_hash `572f5ddb9a60e821`)
- A4 (Full Phase 1): `miniwob-smoke-20260913T080258Z-c75539d2` (config_hash `3284e94186b61b39`)

