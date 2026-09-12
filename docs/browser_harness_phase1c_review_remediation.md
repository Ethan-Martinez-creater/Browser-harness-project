# Reliable Web Workflow Agent Harness — Phase 1C 审批整改计划

## 0. 审批结论

审查对象：

- Repository: `Ethan-Martinez-creater/Browser-harness-project`
- Branch: `main`
- Reviewed commit: `26ffa2dbb6dfdbdb7e95501a4a808a27f274aa91`
- Phase 1B approved baseline: `fce8df6b753a365dcafb34d0ac7c03694ce2601b`

当前结论：

**Phase 1C 核心架构通过，但最终 closure 暂不通过。不得进入 Phase 1D。**

已经确认通过、无需重做：

- FailurePolicyEngine 与 RecoveryManager 分层成立；
- Policy 为 deterministic rule-based，不使用 LLM；
- ACTION_ERROR / OBSERVATION_INVALID / LOOP_DETECTED / NO_PROGRESS 主规则已实现；
- Browser action 不存在 blind retry；
- WAIT_AND_REOBSERVE 是 Harness-owned environment action，不产生 Agent StepRecord；
- blocked action 重新选择时不会进入 `env.step()`；
- typed `recovery_directive` 与 Phase 1B `repair_feedback` 分离；
- FaultInjectingEnvironmentAdapter 已实现；
- C1–C8 controlled fault suite 全部 PASS；
- recovery disabled regression 存在；
- machine-generated Phase 1C report 已提交；
- GitHub Actions unit/regression：171 PASS；
- Ruff：PASS。

本轮只修 Phase 1C 的 budget / outcome / trace 契约。不得开始 Phase 1D。

## 1. Blocker B1 — Recovery budget 的配置路径与真实运行路径不一致

当前 `configs/phase1/recovery.yaml` 把 `max_recoveries_per_episode` 写在：

```yaml
reliability:
  budget:
    max_recoveries_per_episode: 3
```

但 `default_budget_from_config()` 实际读取：

```text
reliability.recovery.max_recoveries_per_episode
```

renderer 又读取 `reliability.budget.max_recoveries_per_episode`。

当前两边默认值恰好都是 3，所以 smoke/fault suite 未暴露问题。一旦修改该值，报告与 runtime 可能不一致。

### 整改要求

固定一个 canonical location。推荐按 Phase 1C 原设计使用：

```yaml
reliability:
  recovery:
    enabled: true
    max_recoveries_per_episode: 3
    wait_ms: 500
    block_steps: 1
  budget:
    max_extra_model_calls_per_episode: 6
```

同步修改：

- `default_budget_from_config()`
- `HarnessConfig`
- `configs/phase1/recovery.yaml`
- renderer
- tests
- docs

同时 fail-fast 校验：

```text
recovery.enabled -> bool
max_recoveries_per_episode -> int >= 0
wait_ms -> int >= 0
block_steps -> int >= 1
```

必须测试 runtime / persisted config / renderer 使用同一个值。

## 2. Blocker B2 — budget exhausted 会错误终止后续正常 step

`FailurePolicyEngine.decide()` 当前在判断具体 failure kind 前先检查：

```text
recovery_count >= max_recoveries
```

因此 budget 用尽后，即使当前：

```text
signals=[]
```

或只有：

```text
NO_PROGRESS
```

也会被 ABORT。

正确语义是：

> budget 只禁止新的 Recovery，不应该杀死后续本可继续的正常执行。

### 正确顺序

```text
TASK_FAILED -> ABORT

ACTION_ERROR / OBSERVATION_INVALID / LOOP_DETECTED
    -> budget available: RECOVER
    -> budget exhausted: ABORT

NO_PROGRESS only -> CONTINUE
no signals -> CONTINUE
```

必须新增：

```text
budget full + PASS -> CONTINUE
budget full + NO_PROGRESS -> CONTINUE
budget full + ACTION_ERROR -> ABORT
budget full + OBSERVATION_INVALID -> ABORT
budget full + LOOP_DETECTED -> ABORT
```

## 3. Blocker B3 — Recovery outcome accounting 会产生不可能统计

### 问题 A

EpisodeRunner 在 `PolicyAction.ABORT` 时无条件：

```python
episode_recovery_failed_count += 1
```

persistent failure 场景中，前一个 recovery 已经被 pending evaluator 计为 failed，再因 budget abort 额外 +1，可能出现：

```text
recovery_count = 3
recovery_failed_count = 4
```

Policy 拒绝启动“第 4 次 Recovery”本身不是一个新的 recovery failure，不应额外计数。

### 问题 B

episode 若在 pending recovery 评价完成前结束，例如：

```text
max_steps
model-side terminal failure
recovery abort
```

pending outcome 可能被静默遗留。

### 整改要求

- ABORT 不得凭空新增 recovery failure；
- episode 结束前 finalize pending outcomes；
- 若某类 recovery 明确允许 unresolved，则增加显式 `recovery_unresolved_count`，不得静默丢失。

## 4. Required R1 — local recovery success 要使用 recovery 起点 fingerprint

pending entry 已保存：

```text
pre_fingerprint
```

但当前 evaluator 没有使用。

现在只看“下一 Agent action 自己是否 state_changed”，因此 WAIT_AND_REOBSERVE 本身已经恢复页面状态时可能被漏判。

应使用：

```text
current_fingerprint != pending.pre_fingerprint
```

再结合：

```text
task success
same failure signature repeated
2-step window
```

新增测试：

```text
invalid observation
→ WAIT_AND_REOBSERVE
→ fresh valid observation
→ next agent action不改变页面
```

仍应正确评价 recovery outcome。

## 5. Required R2 — Recovery event 需要 trigger failure identity

当前 RecoveryManager 的 event 中：

```text
failure_kind = None
```

并且没有 `failure_signature`。

至少补齐：

```text
failure_kind
failure_signature
directive_kind
blocked_actions
```

不要从 reason 文本反向解析。

可以把 trigger identity 放入 RecoveryDirective，或显式传给 RecoveryManager。

## 6. Required R3 — 明确 blocked-action re-decision 的计数语义

当前 Agent 再次选择 blocked action 时：

```text
env.step 不执行
```

这是正确的。

但系统同时：

```text
recovery_count += 1
```

却没有为这一新增 count 创建新的 local recovery outcome，所以：

```text
recovery_count
```

与 success/failed outcome 数量可能失配。

推荐固定：

```text
recovery_count = 实际创建/执行 RecoveryDirective 次数
```

blocked-action re-selection 单独统计：

```text
blocked_action_redecision_count
```

它仍需受 guard/budget 约束。

如果保留“它也算 Recovery”的定义，则必须为每次 count 建立 outcome evaluation，并在 docs/metrics 中统一。

## 7. Required R4 — TASK_FAILED policy 与 runtime 实际顺序统一

Policy 定义 `TASK_FAILED -> ABORT`，但 EpisodeRunner 当前在 policy 前就根据 environment terminal 直接结束。

安全结果是正确的，但架构描述/event trace 不一致。

推荐保留：

```text
environment terminal has highest priority
```

并明确文档：

```text
TASK_FAILED is terminal-short-circuited by EpisodeRunner
```

或者在 terminal break 前仅记录 deterministic ABORT policy event，但绝不进入普通 Recovery。

## 8. Controlled Fault Suite 扩展

保留 C1–C8，并增加：

```text
C9  budget exhausted + clean PASS -> CONTINUE
C10 budget exhausted + single NO_PROGRESS -> CONTINUE
C11 budget exhausted + ACTION_ERROR -> controlled abort
C12 pending recovery is finalized when episode ends
C13 WAIT recovery itself changes fingerprint -> local outcome correct
```

## 9. 是否需要重新跑在线 MiniWoB

若整改只影响：

- config wiring
- budget policy ordering
- recovery outcome bookkeeping
- recovery trace metadata
- local outcome calculation

且 regression 证明正常 path / Prompt / Browser action sequence 未改变，则不强制重新消耗在线模型额度。

但必须重新跑 C1–C13 deterministic fault suite。

若真实 recovery trajectory 被改变，则重新跑 Phase 1C smoke。

## 10. 本轮禁止事项

不得实现：

```text
Replanner
RecoveryPlan execution
LLM planning
Phase 1D escalation
Checkpoint
Memory
Skills
```

## 11. Closure Exit Criteria

- [ ] Recovery budget 只有一个 canonical config path；
- [ ] renderer / persisted config / runtime 使用同一值；
- [ ] recovery config fail-fast；
- [ ] exhausted budget 不影响 PASS / NO_PROGRESS；
- [ ] exhausted budget 会阻止新的 recoverable Recovery；
- [ ] `recovery_failed_count` 不重复计数；
- [ ] pending recovery 在 episode end 不静默丢失；
- [ ] local outcome 使用 recovery-start fingerprint；
- [ ] Recovery event 包含 failure kind + signature；
- [ ] blocked-action re-decision 计数语义明确；
- [ ] TASK_FAILED terminal semantics 与 docs 一致；
- [ ] C1–C8 继续 PASS；
- [ ] C9–C13 PASS；
- [ ] Phase 0/1A/1B regressions PASS；
- [ ] GitHub Actions PASS；
- [ ] Ruff PASS；
- [ ] Replanner 未实现。

通过 closure review 后，才进入 **Phase 1D — Controlled Replanning**。
