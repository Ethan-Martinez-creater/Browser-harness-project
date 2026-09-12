# Reliable Web Workflow Agent Harness — Phase 1B 审批整改计划

## 0. 审批结论

审查对象：

- Repository: `Ethan-Martinez-creater/Browser-harness-project`
- Branch: `main`
- Reviewed commit: `1ff35aed48bbf79ea066d240b1e6bdf80c7a4d0c`
- Phase 1A approved baseline: `5afd19026ced06f72f319bda834445c56e13665c`

当前结论：

**Phase 1B 核心架构方向通过，但最终验收暂不通过。不得进入 Phase 1C。**

已确认通过、无需重做：

- `DecisionExecutor` 已独立于 `EpisodeRunner`；
- Retry 只发生在 model-side pre-action path；
- Browser action 没有 blind retry；
- `MODEL_API_ERROR` / `MODEL_OUTPUT_PARSE_ERROR` 可以受控重试；
- 纯 API-error 路径的 `max_retries=2` 正确表现为 1 initial + 2 retries；
- Episode-level extra-model-call budget 已存在；
- Retry 不增加 environment step；
- Retry 不增加 Agent step_index；
- parse failure token usage 能进入 episode 总 token；
- FaultInjectingModelAdapter 已建立；
- B1–B6 controlled fault suite 当前全部通过；
- Phase 0 / Phase 1A compatibility tests 继续存在；
- GitHub Actions unit/regression 与 Ruff 已通过；
- Natural MiniWoB smoke 正常运行，本轮没有自然 model-side retry。

本轮只修 Phase 1B 基础契约，不得实现 Recovery / Replanning。

---

## 1. Blocker B1 — per-type retry limit 被错误绑定到全局 attempt_index

当前 `DecisionExecutor` 只有一个全局 `attempt_index`，每次失败时把 `attempt_index + 1` 作为 `RetryPolicy.retry_index`。这在单一 FailureKind 下正确，但混合失败序列会错误。

例如：

```text
attempt 0 -> MODEL_API_ERROR
              ↓ API retry #1 allowed
attempt 1 -> MODEL_OUTPUT_PARSE_ERROR
```

此时 parse failure 被传入 `retry_index = 2`，而 `parse_max_retries = 1`，因此第一次 parse repair 会被错误判定 exhausted。

真实语义应为：

```text
API retry count   = 1
Parse retry count = 0
```

因此第一次 parse repair仍应允许。

反向序列同样存在问题：

```text
parse failure -> parse retry #1
API failure   -> 应仍拥有 API retry #1/#2
```

### 整改要求

Decision cycle 内分别追踪：

```text
api_retry_count
parse_retry_count
total_retry_count
```

Episode-level：

```text
extra_model_calls
```

仍是所有 retry 的全局总预算。

建议：

```python
RetryPolicy.decide(
    failure_kind=...,
    retry_index_for_kind=...,
    reliability_state=...,
    budget=...,
)
```

### 必须新增 mixed-failure tests

M1：

```text
attempt 0 API error
attempt 1 parse error
attempt 2 success
```

预期：

```text
success
total retry_count = 2
API retries = 1
parse retries = 1
one environment action
one StepRecord
```

M2：

```text
attempt 0 parse error
attempt 1 API error
attempt 2 API error
attempt 3 success
```

预期：

```text
parse repair = 1
API retries = 2
success
```

M3：

验证 episode-level extra-model-call budget 仍能跨 FailureKind 截断混合路径。

---

## 2. Blocker B2 — Agent Protocol 与实际 runtime contract 不一致

`agents/base.py` 当前 `Agent` Protocol 声明：

```python
def decide(
    *,
    task,
    observation,
    history,
    action_contract,
) -> AgentTurn:
    ...
```

但实际 `BaselineAgent.decide()` 已经是：

```python
def decide(
    ...,
    repair_feedback: str | None = None,
) -> tuple[AgentTurn, PromptBundle]:
```

`DecisionExecutor` 无论是否发生 retry，都会调用：

```python
agent.decide(..., repair_feedback=repair_feedback)
```

因此一个真正按照当前 `Agent` Protocol 实现的 Agent 会在 Phase 1B runtime 中因不接受 `repair_feedback` 而出现 `TypeError`。

### 整改要求

做最小修正，使 Protocol 与当前真实 contract 一致：

```python
class Agent(Protocol):
    def decide(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        action_contract: ActionContract,
        repair_feedback: str | None = None,
    ) -> tuple[AgentTurn, PromptBundle]:
        ...
```

新增一个非 BaselineAgent 的最小 protocol-conforming fake agent 测试，确保 DecisionExecutor 不依赖 BaselineAgent 私有实现。

---

## 3. Required R1 — `retry_success_rate` 指标分母错误

Phase 1B 对 `retry_success` 的定义是：

> 一个 decision cycle 发生至少一次 retry，并最终成功产生合法 ActionDecision。

因此 `retry_success_rate` 应为：

```text
成功的 retry-bearing decision cycles
/
所有 retry-bearing decision cycles
```

当前实现却使用：

```python
retry_success_count / total_retry_count
```

其中 `total_retry_count` 是 retry attempt 数。

例如同一个 decision cycle：

```text
API error
API error
success
```

这里：

```text
retry_count = 2
retry_success_count = 1
```

当前会得到 0.5，但这个 decision cycle 的 retry 最终成功率应是 1/1 = 1.0。

### 整改要求

新增：

```text
retry_cycle_count
```

定义为：发生至少一次 retry 的 decision cycle 数量。

EpisodeRunner 中：

```text
每个 decision_result.retry_count > 0
→ retry_cycle_count += 1
```

Aggregate：

```text
retry_success_rate
=
retry_success_count / retry_cycle_count
```

继续保留：

```text
total_retry_count
```

作为 retry attempts 数量。

必须区分：

```text
attempt
cycle
episode
```

---

## 4. Required R2 — `retry_latency_s` 未包含 backoff

设计语义：

```text
retry_latency_s
=
backoff
+
retry model call latency
```

当前实现先 sleep(backoff)，然后下一轮重新设置计时起点，所以 500ms / 1000ms backoff 没有进入 `retry_latency_s`。

### 整改要求

显式累计 scheduled backoff：

```python
result.retry_latency_s += backoff_ms / 1000.0
```

并继续累计 retry attempt model-call elapsed time。

新增测试，至少保证：

```text
一次 API retry，backoff=500ms
→ retry_latency_s >= 0.5
```

parse repair backoff=0，不得增加固定 backoff。

---

## 5. Required R3 — Retry 配置必须 fail-fast validation

当前以下配置主要被直接 `int(...)` 解析：

```text
model_api.max_retries
model_api.backoff_ms
model_output.max_retries
max_extra_model_calls_per_episode
```

需要对非法值做结构化配置错误。

### 整改要求

HarnessConfig 验证：

```text
retry.enabled 必须为 bool
api max_retries >= 0
parse max_retries >= 0
max_extra_model_calls_per_episode >= 0
backoff_ms 必须为 list[int]
每个 backoff_ms >= 0
```

若 `api_max_retries > len(backoff_ms)`，可继续沿用“最后一个 backoff 重复使用”的当前策略，但需要代码注释明确。

---

## 6. Required R4 — Parse retry 的失败 attempt 要保持 Trace 可审计

Phase 0 中 parse failure 会通过 `TraceRecorder.write_failed_model_output()` 保存 raw output。

Phase 1B 后 parse failure 在 `DecisionExecutor` 内被捕获。若 repair 最终成功，主 StepRecord 只保存最后成功 attempt 的 prompt/model response；初始 malformed raw output 丢失。

结果是可以知道“发生过 parse retry”，但无法从 run trace 还原“哪段输出导致 parse retry”。

### 整改要求

不改变 `steps.jsonl` one-step semantics。

为 model attempt 增加最小 artifact，例如：

```text
artifacts/
  attempt_003_00.json/txt
  attempt_003_01.json/txt
```

至少在 `save_model_responses=true` 时，对于 `MODEL_OUTPUT_PARSE_ERROR` 保存：

```text
attempt_index
failure_type
raw_text
input/output token usage
```

Retry event 可以保存 `artifact_ref`。

不要把 raw model text 直接塞进 `events.jsonl`。

如果不希望 TraceRecorder 注入 DecisionExecutor，可以增加轻量 attempt sink/callback，由 EpisodeRunner/TraceRecorder 提供。

新增 trace test：

```text
malformed output -> repair success
```

最终必须能找到 failed attempt artifact 和 final successful model artifact。

---

## 7. 建议项 — API transient classification

当前所有 `ModelApiError` 都被映射成 `MODEL_API_TRANSIENT`，而 OpenAI-compatible adapter 又把所有 SDK exception 统一包装成 `ModelApiError`。

因此理论上 authentication error、bad request、permission denied、invalid model 等也会 retry。

这不是本轮硬性 Blocker，因为 Phase 1B 原计划把 MODEL_API_ERROR 简化为可重试 model-side failure。

但进入正式 benchmark / 生产化之前应进一步区分：

```text
transient:
timeout
connection error
rate limit
5xx

terminal:
authentication
permission
bad request
invalid model/config
```

---

## 8. Controlled Fault Suite 扩展

保留 B1–B6，并增加：

```text
M1 API -> Parse -> Success
M2 Parse -> API -> API -> Success
M3 Mixed failures -> episode budget exhausted
```

机器 JSON 与 generated Markdown 中单独展示。

---

## 9. 是否需要重新跑在线 MiniWoB

如果整改只影响 mixed-failure retry counting、metrics、backoff accounting、Agent Protocol、config validation 和 retry-attempt artifacts，并且 no-failure regression 证明 Prompt / model calls / environment actions 不变，则不强制重新消耗在线模型额度跑 12-task MiniWoB。

但必须重新执行扩展后的 deterministic fault suite，并重新生成 Phase 1B fault report。

---

## 10. 本轮禁止事项

不得实现：

```text
RecoveryDirective
RecoveryManager
FailurePolicyEngine for environment-side failures
WAIT_AND_REOBSERVE
BLOCK_REPEATED_ACTION
REDECIDE_WITH_FEEDBACK for browser action error
Replanner
RecoveryPlan
```

Browser action 仍禁止 blind retry。

---

## 11. Phase 1B Closure Exit Criteria

- [ ] API / Parse retry limit 分开计数；
- [ ] Mixed failure 不交叉消耗 per-type retry allowance；
- [ ] Episode extra-model-call budget 继续跨类型全局生效；
- [ ] Agent Protocol 与 runtime 调用签名一致；
- [ ] DecisionExecutor 不依赖 BaselineAgent 私有实现；
- [ ] retry attempts 与 retry cycles 区分；
- [ ] `retry_success_rate` 使用 retry cycle 为分母；
- [ ] retry latency 包含 backoff；
- [ ] 非法 retry config fail-fast；
- [ ] parse failed attempt 可审计；
- [ ] B1–B6 继续 PASS；
- [ ] M1–M3 PASS；
- [ ] Phase 0/1A compatibility 继续 PASS；
- [ ] GitHub Actions PASS；
- [ ] Ruff PASS；
- [ ] Browser action retry 仍不存在。

通过 closure review 后才进入 **Phase 1C — Recovery Policy**。
