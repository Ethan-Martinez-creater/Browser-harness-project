# Reliable Web Workflow Agent Harness — Phase 1D 审批整改计划

## 0. 审批对象与结论

- Repository: `Ethan-Martinez-creater/Browser-harness-project`
- Branch: `main`
- Reviewed commit: `30c687e6573081e050f6b410826e7cac6904af60`
- Phase 1C approved baseline: `7e73094e04b9975a800d621ad7c6e9b827af3bd8`

当前结论：

**Phase 1D 的核心架构方向通过，但最终 closure 暂不通过。不得运行 Final Phase 1 3-seed Ablation。**

已确认通过、无需重做：

- ReplanTriggerPolicy 为 deterministic escalation layer；
- 只在 Phase 1C 的 RECOVER 路径上考虑 Replan；
- 仅 ACTION_ERROR / LOOP_DETECTED 可升级；
- OBSERVATION_INVALID / TASK_FAILED / model-side failure 不直接 Replan；
- RecoveryPlan 是 advisory prompt context，不直接执行 browser action；
- active RecoveryDirective 继续优先于 RecoveryPlan；
- plan horizon 只由真实 Agent StepRecord 消耗；
- Replan budget exhausted 会 fallback 到 Phase 1C，而不是杀死正常执行；
- provider-neutral StructuredModelAdapter 已建立；
- Replanner 不直接 import OpenAI SDK；
- Replan generation API retry / parse retry 路径存在；
- Recovery failure streak 来自真实 recovery outcome；
- Replan SUCCESS / FAILED / UNRESOLVED invariant 已建立；
- D1–D10 controlled fault suite 当前全部 PASS；
- Natural MiniWoB smoke 已完成（11/12，0 natural replans）；
- GitHub Actions：193 unit/regression PASS；
- Ruff：PASS；
- Final 3-seed ablation 尚未运行，符合要求。

本轮只修 Phase 1D cost / trace / reporting contract，不得开始 Final Ablation。

## 1. Blocker B1 — Replanner 第一次模型调用没有消耗全局 extra-model-call budget

Phase 1B 的 episode-level budget 是 `max_extra_model_calls_per_episode`。Replan model call 对 baseline 本身就是额外模型调用，因此 initial Replan call、Replan API retries、Replan parse retries 都必须消耗该 budget。

当前 `ReplanExecutor` 只有在决定执行 retry 时才增加 `state.extra_model_calls`，第一次 `generate_structured()` 前没有检查或消耗该 budget。因此 `max_extra_model_calls_per_episode: 0` 仍可执行一次 Replanner 模型调用。

### 整改要求

Replanner 的**每一次真实模型调用**都必须先占用一个 global extra-model-call slot。避免与当前 retry 路径中的 `state.extra_model_calls += 1` 重复计数。

最终必须保证：

```text
ReliabilityState.extra_model_calls
= Agent-decision retry calls + all Replanner model calls
```

但 `RunResult.extra_model_calls` 可继续保持 Phase 1B 历史语义（Agent-decision retry calls only），Replanner 调用由 `replan_model_calls` 单独报告。

新增测试：

- G1: budget=0 + Replan trigger -> replan_model_calls=0，generation failed，fallback Recovery。
- G2: budget=1 + first Replan call parse fails -> parse repair不能执行，replan_model_calls=1。
- G3: 先有1次 Agent Decision retry，budget=2，再触发 Replan -> 只剩1次 planner call。

## 2. Blocker B2 — Episode total token usage excludes Replanner tokens

`RunResult.input_tokens/output_tokens` 当前来自 `RunState` 的 StepRecord token 汇总。Replanner 不是 Agent Step，因此 `replan_input_tokens/replan_output_tokens` 没有进入 episode total。

这会使 Final Phase 1 ablation 的 `total tokens` 和后续 `tokens_per_success` 系统性低估 A4 的真实模型成本。

### 整改要求

从 Phase 1D 开始：

```text
RunResult.input_tokens = Agent decision-path total input tokens + Replanner input tokens
RunResult.output_tokens = Agent decision-path total output tokens + Replanner output tokens
```

保留 retry/replan token breakdown。`reliability_extra_tokens` 仍表示 overhead subset，而不是总 token。

新增 deterministic test 验证 Agent tokens + Planner tokens = RunResult total tokens。

## 3. Blocker B3 — Replan parse “repair” 实际使用完全相同的 Prompt

RecoveryPlan JSON parse failure 后，当前再次调用 `generate_structured(prompt=prompt)`，但 prompt 没有任何 format-repair feedback。

D7 之所以通过，是 scripted model 被预设为第一次坏 JSON、第二次好 JSON；它没有证明真实 repair。

### 整改要求

Replan parse retry 必须增加最小 typed feedback，例如：

```text
# Format correction
Previous response did not match the required RecoveryPlan JSON schema.
Return exactly one valid JSON object matching the schema.
Do not add markdown fences or extra prose.
```

无论 parse failure 来自 adapter-raised `ModelOutputParseError`，还是 `_parse_plan()`，下一次 parse retry 都必须使用 repair prompt。

升级 D7：捕获两次 Replanner prompt，断言第二次包含 RecoveryPlan format correction。

## 4. Blocker B4 — Decision retry 与 Replan retry 的 attempt artifact 存在文件名冲突

当前失败 attempt artifact 使用：

```text
artifacts/attempt_<step>_<attempt>.json
```

DecisionExecutor 和 ReplanExecutor 共用同一个 writer。若同一步先发生 Agent decision parse failure，随后环境失败触发 Replan 且 Replanner 也 parse failure，则两者可能写同一个 `attempt_005_00.json`，后写入者覆盖前者。

### 整改要求

增加 component/scope namespace。例如：

```text
Decision: artifacts/attempt_005_00.json
Replan:   artifacts/replan_attempt_005_00.json
```

或统一 component-aware 命名。Retry/Replan event 的 `artifact_ref` 必须指向正确文件。

新增同一步 decision+replan parse failure 场景，确认两份 artifact 同时存在且互不覆盖。

## 5. Required R1 — episode_metrics / episodes.csv 未包含 Phase 1D replan 字段

RunResult 已有 replan fields，但 `episode_metrics()` 未输出，因此 `episodes.csv` 与 `summary.json -> episodes[]` 没有 per-episode replan data。

更直接的是：`inspect-run` 当前访问 `m["replan_count"]` 等字段，而 `episode_metrics(result)` 中不存在这些键，因此当前 `inspect-run` 会在 Replan summary 处触发 `KeyError`。

### 整改要求

`episode_metrics()` 至少增加：

```text
replan_count
replan_success_count
replan_failed_count
replan_unresolved_count
replan_model_calls
replan_input_tokens
replan_output_tokens
replan_latency_s
```

建议同时增加 per-episode reliability overhead 三项。新增 `inspect-run` test，确保含 Replan 的 run 能正确展示 summary。

## 6. Required R2 — Phase 1D cost metrics 增加精确 contract tests

至少新增：

- M1: decision retries=0, replan model calls=2 -> reliability_extra_model_calls=2。
- M2: decision retries=1, replan model calls=2 -> reliability_extra_model_calls=3，不得 double-count planner retry。
- M3: 校验 total tokens / retry tokens / replan tokens / reliability_extra_tokens 的关系。
- M4: `reliability_extra_latency_s = retry_latency + recovery_latency + replan_latency`。

## 7. Required R3 — RecoveryPlan 最小有效性约束建议收紧

当前只硬限制 `strategy_steps <= 4`，但空 diagnosis / empty immediate_subgoal 仍会被接受为成功 plan。

建议至少要求：

```text
diagnosis 非空
immediate_subgoal 非空
horizon_steps >= 1
```

如校验失败，走已有 structured parse repair。

## 8. 非阻断工程债务

当前 `web-harness run --config configs/phase1/replanning.yaml` 构造 bare EpisodeRunner，没有复用 benchmark 的 verifier/retry/recovery/replanning wiring；同一配置在 `run` 与 `benchmark` 下行为不同。

这不阻塞本次 Phase 1D closure，但建议在 Final Phase 1 closure 或下一阶段建立共享 `build_episode_runner()` / RunnerFactory。

## 9. Controlled Fault Suite 扩展

D1–D10 必须继续 PASS。新增至少：

```text
D11 global extra-model-call budget=0 blocks initial Replan call
D12 planner initial call consumes budget before parse/API retry
D13 Replan parse repair prompt actually changes
D14 decision+replan failed-attempt artifacts on same step do not collide
```

如实施 plan minimum validation，再增加 D15。

## 10. Natural MiniWoB smoke

当前 natural smoke `total_replan_count=0`。如果整改只影响 planner budget/token/report/trace 路径，且 no-replan regression 证明正常路径不变，则不强制重新消耗在线模型额度。

必须重新执行扩展后的 deterministic Phase 1D fault suite。

## 11. 禁止事项

不得：

```text
运行 Final Phase 1 3-seed ablation
引入 always-on planner
引入 Plan Executor
实现 Phase 2 checkpoint/resume
实现 memory / skills / multi-agent
```

## 12. Phase 1D Closure Exit Criteria

- [ ] Initial Replan model call consumes global extra-model-call budget；
- [ ] Replan retries 消费同一全局 budget；
- [ ] budget=0 时 zero planner model calls；
- [ ] RunResult total token usage 包含 Replanner tokens；
- [ ] structured parse retry 使用真实 format-repair prompt；
- [ ] Decision / Replan attempt artifacts 不互相覆盖；
- [ ] episode_metrics / episodes.csv 包含 Phase 1D fields；
- [ ] inspect-run Replan summary 不报错；
- [ ] reliability overhead metrics 有精确 contract tests；
- [ ] D1–D10 继续 PASS；
- [ ] D11–D14 PASS；
- [ ] Phase 0–1C regressions PASS；
- [ ] GitHub Actions PASS；
- [ ] Ruff PASS；
- [ ] Final Phase 1 ablation 仍未运行。

通过 closure review 后，才冻结最终 commit 并运行 Phase 1 Final A0–A4 3-seed ablation。
