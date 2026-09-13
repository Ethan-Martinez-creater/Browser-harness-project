# Reliable Web Workflow Agent Harness
# Phase 2A — Trace Replay / Checkpoint / Resume 实施计划

> 面向执行智能体的实施文档  
> Repository: `Ethan-Martinez-creater/Browser-harness-project`  
> 开始基线：当前 `main`（Phase 1 Final Ablation 已审批）  
> Phase 1 Final Ablation commit: `6266d47acf532571a45492c0a5516ef3b6bfde27`  
> Phase 1 implementation frozen commit: `1436cb2207ec53a050a656d01bd4775f93e919ba`

---

# 0. 文档目的

Phase 1 已完成：

```text
Failure Detection
→ Controlled Retry
→ Controlled Recovery
→ Controlled Replanning
```

Phase 2A 的目标不是继续提高 Agent 的“智能程度”，而是让当前 Harness 获得生产运行系统必须具备的：

```text
Trace Replay
Checkpoint
Interrupted-run Resume
Replay / Resume Validation
```

本阶段重点回答：

> 一个执行时间较长的 Agent run，如果进程被主动中断，是否可以在不重新调用前序模型、不从头重新完成整个任务的情况下，从一个已验证的 durable checkpoint 恢复？

以及：

> 已完成的 run 是否可以仅依赖持久化 Trace，在不重新调用模型或浏览器的前提下进行离线完整性检查、指标重算和 failure/reliability 行为分析？

Phase 2A 不是浏览器“进程快照”项目，也不是数据库项目。

当前实施范围：

```text
Phase 2A1 — Trace Replay
Phase 2A2 — Durable Checkpoint
Phase 2A3 — Deterministic Resume
Phase 2A4 — Replay / Resume Validation
```

必须严格按顺序实施，每个阶段完成后停止并提交 GitHub 审批。

---

# 1. Phase 2A 的核心边界

## 1.1 Replay、Checkpoint、Resume 必须明确分离

### Replay

```text
已有 trace
↓
读取持久化 records / artifacts
↓
离线验证与重算
```

特点：

```text
不调用 LLM
不操作 BrowserGym
不产生 browser action
不修改原 run
```

---

### Checkpoint

```text
运行到稳定 safe point
↓
序列化 Harness execution state
↓
原子持久化
```

Checkpoint 回答：

> “如果此刻停止，我需要保存哪些信息才能可靠恢复？”

---

### Resume

```text
Checkpoint
+
Environment operation journal
↓
重建环境
↓
验证环境状态
↓
恢复 Harness state
↓
继续真实 Agent execution
```

Resume 会继续真实运行，因此之后可以再次调用：

```text
LLM
Environment
Verifier
Retry
Recovery
Replanner
```

---

## 1.2 Phase 2A 不做“完整浏览器内存快照”

当前 `EnvironmentAdapter` 的正式抽象只有：

```python
reset(task)
step(action)
close()
action_contract()
```

并没有通用：

```text
snapshot_browser()
restore_browser()
```

BrowserGym 的具体 browser/context 又被正确隔离在 adapter 层。

另外，Playwright 的 storage state 主要覆盖：

```text
cookies
localStorage
IndexedDB（版本支持时）
authentication-related storage
```

它并不等同于完整的：

```text
DOM / JS heap / transient page state / sessionStorage / in-flight state
```

因此 Phase 2A 不得把：

```text
browser_context.storage_state()
```

包装一下就宣称实现了“完整 checkpoint”。

Phase 2A 的正式恢复策略为：

> **Deterministic Environment Reconstruction**

即：

```text
TaskSpec + seed
↓
env.reset()
↓
按 checkpoint 记录的 environment operation journal
重新执行已经完成的环境操作
↓
逐操作 fingerprint 验证
↓
到达 checkpoint 环境状态
↓
继续 Agent execution
```

第一版只对：

```text
MiniWoB / deterministic local benchmark
```

声明 Resume 支持。

不得宣称所有 Web benchmark 都可安全 Resume。

---

# 2. Phase 2A 的核心架构

最终目标：

```text
                         ┌─────────────────┐
                         │  EpisodeRunner  │
                         └────────┬────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
      TraceRecorder        CheckpointManager     EnvironmentJournal
            │                     │                     │
            ▼                     ▼                     ▼
      steps/events          checkpoint JSON      env_ops.jsonl
            │                     │                     │
            └──────────────┬──────┴──────────────┘
                           │
                           ▼
                    ResumeManager
                           │
                ┌──────────┴───────────┐
                ▼                      ▼
         restore RunState     reconstruct Environment
                                      │
                                      ▼
                                  validation
                                      │
                                      ▼
                                EpisodeRunner
                                  continues
```

离线 Replay：

```text
run directory
   ↓
TraceBundleLoader
   ↓
TraceReplayEngine
   ↓
ReplayReport
```

---

# 3. Repository 结构建议

Phase 2A 最终新增：

```text
src/web_harness/
├── persistence/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── journal.py
│   ├── replay.py
│   ├── resume.py
│   └── schema.py
│
├── runtime/
│   ├── episode_runner.py
│   └── state.py
│
└── observability/
    └── trace.py
```

允许根据现有代码风格做小调整。

禁止：

```text
SQLAlchemy
Redis
PostgreSQL
external checkpoint service
workflow engine
LangGraph persistence
```

Phase 2A 只使用：

```text
local filesystem
JSON / JSONL
atomic filesystem operations
```

---

# 4. Schema Versioning

Checkpoint/Replay 开始后，必须正式引入 schema version。

建议：

```python
TRACE_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 1
ENV_JOURNAL_SCHEMA_VERSION = 1
```

`manifest.json` 新增：

```json
{
  "trace_schema_version": 2
}
```

Checkpoint：

```json
{
  "checkpoint_schema_version": 1
}
```

Environment journal record：

```json
{
  "schema_version": 1
}
```

旧 Phase 1 trace 没有 version 时：

```text
视为 trace schema v1
```

不要修改旧 committed Phase 1 artifacts。

---

# 5. Phase 2A1 — Trace Replay

## 5.1 目标

建立：

```text
run directory
↓
TraceBundleLoader
↓
offline replay / integrity validation
↓
ReplayReport
```

Phase 2A1 **不实现 Checkpoint，不实现 Resume。**

---

# 6. TraceBundle

建议定义：

```python
class TraceBundle(BaseModel):
    run_dir: str
    manifest: dict
    steps: list[StepRecord]
    events: list[RuntimeEvent]
    result: RunResult | None
```

如需要 artifacts：

不要一次把所有大文件读入内存。

使用：

```text
artifact refs + lazy reader
```

---

# 7. TraceBundleLoader

新增：

```text
persistence/replay.py
```

实现：

```python
class TraceBundleLoader:
    def load(run_dir: Path) -> TraceBundle:
        ...
```

必须检查：

```text
manifest.json
steps.jsonl
events.jsonl（允许不存在于 Phase 0）
result.json（允许 interrupted run 不存在）
artifact references
```

不得修改文件。

---

# 8. Replay Mode

Phase 2A1 定义两种 replay：

## 8.1 Structural Replay

适用于全部历史 trace。

验证：

```text
step_index 连续
run_id 一致
event.step_index 合法
artifact ref 存在
result.num_steps 与 steps 一致
token totals 可重算
retry/recovery/replan outcome invariants
```

不需要 Browser/Model。

---

## 8.2 Semantic Verification Replay

只对具备足够 structured observation 数据的新 trace 启用。

目标：

```text
recorded pre observation
+
recorded action
+
recorded post observation
↓
重新运行 deterministic StepVerifier
↓
得到 FailureSignal
↓
与 recorded verification event 比较
```

Phase 2A1 不要求重放：

```text
LLM
FailurePolicy control flow
Recovery action
Replanner
```

只重放 deterministic verifier。

---

# 9. Structured Observation Artifact

当前 Phase 1：

```text
obs_NNN.txt
next_obs_NNN.txt
```

是面向人的文本 artifact。

Phase 2A1 为未来 replay 增加**附加** JSON artifact：

```text
artifacts/obs_000.json
artifacts/next_obs_000.json
```

内容：

```python
Observation.model_dump(mode="json")
```

保持原有：

```text
obs_000.txt
next_obs_000.txt
```

不删除，保证人类调试体验和兼容性。

`StepRecord` 建议新增：

```python
observation_json_ref: str | None
next_observation_json_ref: str | None
```

Phase 1 consumer 不受影响。

---

# 10. ReplayReport

```python
class ReplayReport(BaseModel):
    run_id: str
    structural_valid: bool

    step_count: int
    event_count: int

    artifact_missing_count: int
    invariant_error_count: int

    metric_mismatches: list[str]
    verification_mismatches: list[str]

    replayed_verifications: int = 0

    errors: list[str]
```

Structural violation 不应该用 Python assertion 直接崩溃。

必须输出 structured result。

---

# 11. Replay CLI

新增：

```bash
web-harness replay <run_id>
```

或：

```bash
web-harness replay --run-dir runs/<run_id>
```

输出至少：

```text
Run ID
Trace schema version
Steps
Events
Structural validation
Artifact validation
Metric validation
Verifier replay
Mismatches
```

不需要 Web UI。

---

# 12. Phase 2A1 Tests

必须覆盖：

```text
valid completed trace
valid interrupted trace without result.json
missing artifact
non-contiguous step index
run_id mismatch
corrupt JSONL
recovery outcome invariant violation
replan outcome invariant violation
metric mismatch
```

Semantic verifier replay：

```text
known pre/post observations
↓
replayed FailureSignal
==
recorded FailureSignal
```

---

# 13. Phase 2A1 Exit Criteria

- [ ] trace schema version 建立；
- [ ] structured observation artifact 建立；
- [ ] TraceBundleLoader；
- [ ] Structural Replay；
- [ ] verifier semantic replay；
- [ ] ReplayReport；
- [ ] replay CLI；
- [ ] 不调用模型；
- [ ] 不启动 BrowserGym；
- [ ] 不修改原 trace；
- [ ] Phase 0–1 regressions PASS。

完成后停止并提交审批。

---

# 14. Phase 2A2 — Durable Checkpoint

Phase 2A1 通过审批后才能实施。

---

# 15. Checkpoint 的 Stable Safe Point

Phase 2A checkpoint **只能在稳定边界创建**。

定义 safe point：

```text
没有 model call in flight
没有 environment action in flight
当前 Observation 与 live environment 一致
此前 StepRecord 已 fsync
此前 RuntimeEvent 已 fsync
此前 EnvironmentOperation 已 fsync
active Recovery/Replan state 已更新完毕
下一动作尚未开始
```

checkpoint 语义：

```text
“下一次将从 next_step_index 开始做 Agent decision”
```

Phase 2A 不支持：

```text
model call 中间 checkpoint
env.step 执行中 checkpoint
```

---

# 16. 当前 EpisodeRunner local state 必须持久化

当前很多 episode accounting 是 `EpisodeRunner.run()` 中的 local variables。

Checkpoint 不能只保存当前 `RunState`，否则恢复后：

```text
retry metrics
recovery metrics
replan metrics
```

会丢失。

Phase 2A2 必须把所有需要跨 checkpoint 存活的 mutable episode state 收敛到 serializable state。

推荐新增：

```python
class EpisodeCounters(BaseModel):
    retry_count: int = 0
    retry_cycle_count: int = 0
    retry_success_count: int = 0
    retry_exhausted_count: int = 0
    extra_model_calls: int = 0
    retry_input_tokens: int = 0
    retry_output_tokens: int = 0
    retry_latency_s: float = 0.0

    recovery_count: int = 0
    recovery_success_count: int = 0
    recovery_failed_count: int = 0
    recovery_unresolved_count: int = 0
    recovery_environment_actions: int = 0
    recovery_latency_s: float = 0.0
    blocked_action_redecision_count: int = 0

    replan_count: int = 0
    replan_success_count: int = 0
    replan_failed_count: int = 0
    replan_unresolved_count: int = 0
    replan_model_calls: int = 0
    replan_input_tokens: int = 0
    replan_output_tokens: int = 0
    replan_latency_s: float = 0.0
```

并加入：

```python
RunState.counters
```

不要为 checkpoint 保留两份不同的权威状态。

EpisodeRunner 应逐步使用：

```text
state.counters
```

替代 local accounting variables。

这是 Phase 2A2 最重要的重构点之一。

---

# 17. Recovery Events Context

Phase 1D Replanner 需要 recent recovery events。

当前 runtime 中存在内存：

```text
recovery_events_log
```

Resume 后不能丢失。

推荐不要在 checkpoint 中无限复制全部 events。

Checkpoint 只保存：

```text
recent_recovery_events
```

数量上限与 Replanner 的：

```text
recent_steps
```

同量级，例如最多 10。

或者 Resume 时从 `events.jsonl` 重建 recent recovery events。

优先：

> 从 durable event trace 重新读取。

避免同一信息双重存储。

---

# 18. Environment Operation Journal

Checkpoint 只能恢复 Harness state 还不够。

必须知道：

```text
环境到达当前页面之前实际发生了哪些环境操作。
```

新增：

```text
runs/<run_id>/environment_ops.jsonl
```

每一个**真实调用 `EnvironmentAdapter.step()`** 的操作写一条。

---

# 19. EnvironmentOperationRecord

建议：

```python
class EnvironmentOperationKind(StrEnum):
    AGENT_ACTION = "agent_action"
    RECOVERY_ACTION = "recovery_action"
```

```python
class EnvironmentOperationRecord(BaseModel):
    schema_version: int = 1

    run_id: str
    op_index: int

    source_step_index: int | None
    kind: EnvironmentOperationKind

    action: str

    expected_post_fingerprint: str

    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False

    action_error_signature: str | None = None
```

`op_index` 是全 run 单调递增序号。

---

# 20. Journal 记录范围

必须记录：

### Agent Action

```text
BaselineAgent / DecisionExecutor
↓
env.step(action)
```

记录。

### Recovery Action

例如：

```text
WAIT_AND_REOBSERVE
→ env.step(noop(...))
```

记录。

### 不记录

```text
blocked-action re-decision
model retry
Replan model call
Verifier
Policy
```

因为它们不修改 environment。

---

# 21. Bootstrap

MiniWoB bootstrap 当前在：

```text
env.reset()
```

内部执行并在 manifest 中记录：

```text
environment_bootstrap_action
```

Resume reconstruction 使用同一 EnvironmentAdapter / config 调用 `reset()`，因此 bootstrap 会自然再次执行。

不要把 bootstrap 再写入：

```text
environment_ops.jsonl
```

否则 reconstruction 会重复 bootstrap。

---

# 22. Operation Journal Durability

Environment operation 的写入顺序必须谨慎。

Phase 2A stable-point语义要求：

```text
env.step()
↓
得到真实 EnvironmentStep
↓
立即 append + fsync EnvironmentOperationRecord
↓
继续 verification / trace
```

这样 durable trace 至少知道“环境操作发生过”。

Checkpoint 只有在对应 operation 已 durable 后才能创建。

---

# 23. Checkpoint Schema

新增：

```python
class CheckpointEnvelope(BaseModel):
    checkpoint_schema_version: int = 1

    checkpoint_id: str
    run_id: str

    created_at: str
    reason: str

    git_commit: str | None
    config_hash: str | None

    trace_schema_version: int
    env_journal_schema_version: int

    task: TaskSpec

    next_step_index: int

    state: RunState

    environment_op_count: int
    current_environment_fingerprint: str

    action_contract_hash: str

    environment_adapter: str
    environment_resume_strategy: str

    trace_offsets: CheckpointTraceOffsets
```

---

# 24. CheckpointTraceOffsets

```python
class CheckpointTraceOffsets(BaseModel):
    step_count: int
    event_count: int
    environment_op_count: int
```

用于 Resume 验证 checkpoint 与 durable trace 的对应关系。

---

# 25. ActionContract Hash

Resume 时如果 ActionContract 改变：

```text
旧 action
```

可能无法再执行。

因此 checkpoint 必须保存：

```text
action_contract_hash
```

要求 deterministic：

```text
canonical JSON of ActionContract
↓
SHA256
```

Resume 前重新获取当前 env ActionContract 并比较。

不一致：

```text
拒绝 Resume
```

---

# 26. Environment Resume Capability

新增明确能力 contract，例如：

```python
class EnvironmentResumeStrategy(StrEnum):
    DETERMINISTIC_REPLAY = "deterministic_replay"
    NATIVE_SNAPSHOT = "native_snapshot"
    UNSUPPORTED = "unsupported"
```

Phase 2A：

```text
MiniWoB BrowserGymAdapter
→ DETERMINISTIC_REPLAY
```

其他 benchmark 默认：

```text
UNSUPPORTED
```

不得因为都使用 BrowserGym 就自动声称支持。

---

# 27. CheckpointManager

新增：

```text
persistence/checkpoint.py
```

```python
class CheckpointManager:
    def save(...)->CheckpointEnvelope:
        ...

    def load(path)->CheckpointEnvelope:
        ...

    def latest(run_dir)->CheckpointEnvelope | None:
        ...
```

目录：

```text
runs/<run_id>/
  checkpoints/
    cp_000000.json
    cp_000001.json
    ...
    latest.json
```

---

# 28. Atomic Checkpoint Write

禁止：

```python
path.write_text(...)
```

直接写最终 checkpoint 文件。

必须：

```text
serialize
↓
write cp_x.tmp
↓
flush
↓
fsync
↓
os.replace(tmp, final)
↓
fsync parent directory（平台允许时 best effort）
```

`latest.json` 同样 atomic。

如果进程在 `.tmp` 阶段死亡：

Resume 不得把 `.tmp` 当合法 checkpoint。

---

# 29. Checkpoint Policy

Phase 2A 默认：

```yaml
persistence:
  checkpoint:
    enabled: false
    every_agent_steps: 1
```

在 Phase 2A test/resume config 中：

```yaml
enabled: true
every_agent_steps: 1
```

即每个 stable Agent step 后 checkpoint。

原因：

Phase 2A 优先验证正确性，而不是降低 checkpoint I/O。

以后复杂 benchmark 再研究：

```text
every N steps
event-based checkpoint
adaptive checkpoint
```

---

# 30. Checkpoint Event

RuntimeEventType 新增：

```text
CHECKPOINT
RESUME
REPLAY
```

Checkpoint event：

```json
{
  "event_type": "checkpoint",
  "step_index": 3,
  "component": "checkpoint_manager",
  "outcome": "saved",
  "data": {
    "checkpoint_id": "...",
    "next_step_index": 4,
    "environment_op_count": 5
  }
}
```

不要在 event 中复制完整 state。

---

# 31. Checkpoint Secrets

Checkpoint 可能保存：

```text
Observation
Task metadata
ReliabilityState
```

必须继续遵守 secret policy。

不得保存：

```text
API key
provider token
password config
```

测试继续使用 sentinel secret 扫描：

```text
runs/<run_id>/
```

包括：

```text
checkpoints/
```

---

# 32. Phase 2A2 Tests

至少：

```text
RunState roundtrip
ReliabilityState roundtrip
active RecoveryDirective checkpoint
active RecoveryPlan + remaining horizon checkpoint
pending recovery outcome checkpoint
pending replan outcome checkpoint
episode counters roundtrip
operation journal roundtrip
action contract hash deterministic
atomic checkpoint
orphan .tmp ignored
latest checkpoint selection
checkpoint secret leak test
```

---

# 33. Phase 2A2 Exit Criteria

- [ ] mutable episode runtime state 不再藏在不可恢复 local variables；
- [ ] Environment operation journal；
- [ ] stable safe-point 定义落实；
- [ ] CheckpointEnvelope；
- [ ] atomic CheckpointManager；
- [ ] action contract hash；
- [ ] environment resume strategy；
- [ ] checkpoint events；
- [ ] checkpoint secret test；
- [ ] 尚未实现真实 Resume。

完成后停止并提交审批。

---

# 34. Phase 2A3 — Deterministic Resume

Phase 2A2 通过审批后实施。

---

# 35. Resume 的正式范围

Phase 2A3 支持：

> **从 durable stable checkpoint 恢复。**

不支持：

```text
env.step 正在执行时的任意进程快照
模型 API 调用中间恢复
操作系统进程级 checkpoint
浏览器进程内存恢复
```

即：

```text
cooperative / safe-point interruption
```

而不是：

```text
arbitrary crash-consistent process snapshot
```

必须在 README/docs 中明确。

---

# 36. ResumeManager

新增：

```text
persistence/resume.py
```

建议：

```python
class ResumeManager:
    def prepare(
        *,
        checkpoint: CheckpointEnvelope,
        env: EnvironmentAdapter,
        run_dir: Path,
        expected_config_hash: str,
    ) -> ResumeContext:
        ...
```

---

# 37. Resume Preflight

在调用 Browser/Model 前必须验证：

```text
checkpoint schema supported
trace schema supported
run_id 一致
TaskSpec 一致
config_hash 一致
git commit 一致
Environment adapter identity 一致
resume strategy supported
ActionContract hash 一致
trace counts 与 checkpoint offsets 一致
checkpoint 尚未被后续 durable execution 超越
```

任何失败：

```text
zero model calls
zero environment reconstruction actions（能提前判断时）
structured Resume error
```

---

# 38. Code Drift Policy

Phase 2A 默认：

```text
checkpoint.git_commit
必须等于
current git commit
```

不提供：

```text
--allow-code-drift
```

原因：

Phase 2A 要先证明 deterministic correctness。

后期再研究迁移 checkpoint schema / code-version compatibility。

---

# 39. Config Drift Policy

同样：

```text
checkpoint.config_hash
==
current resolved config hash
```

否则 Resume 失败。

不能：

```text
改变 retry budget
改变 recovery config
改变 model/provider
改变 action contract
```

后再声称是同一个 Resume。

---

# 40. Environment Reconstruction

算法：

```text
1. env.reset(task)
2. 获取 reset observation
3. 获取 current ActionContract
4. 验证 ActionContract hash
5. 从 environment_ops.jsonl 读取 checkpoint.environment_op_count 条记录
6. 按 op_index 顺序逐条 env.step(record.action)
7. 每一步重新计算 post fingerprint
8. 与 record.expected_post_fingerprint 比较
9. reward / terminal / action error 做一致性检查
10. 最终 fingerprint 必须等于 checkpoint.current_environment_fingerprint
```

完全匹配后：

```text
environment reconstructed
```

---

# 41. Reconstruction 不调用 Agent / Model

重建期间禁止：

```text
DecisionExecutor
ModelAdapter
Verifier-driven new action
Recovery policy
Replanner
```

只允许执行 journal 中已经记录的 environment action。

因此前序模型调用成本不会再次产生。

---

# 42. Reconstruction Divergence

新增 ErrorType，例如：

```text
RESUME_DIVERGENCE
CHECKPOINT_INVALID
RESUME_UNSUPPORTED
```

不要使用 UNKNOWN_ERROR。

如果任一步出现：

```text
fingerprint mismatch
reward mismatch
terminal mismatch
unexpected action error
```

立即：

```text
abort Resume
close environment
zero new Agent model calls
```

写 Resume event：

```text
outcome = divergence
```

原 checkpoint 和 trace 不得修改。

---

# 43. Fingerprint 语义

继续使用 Phase 1 已审批的：

```text
ObservationFingerprint
```

不要为 Resume 发明另一套 page hash。

但需要明确：

```text
elapsed_time
```

不能参与 fingerprint。

当前 Phase 1 fingerprint 已满足这一原则。

---

# 44. Harness State Restore

Environment reconstruction 成功后：

```text
state = checkpoint.state
```

必须确保：

```text
state.current_observation
```

与 reconstructed live observation 的 fingerprint 一致。

随后用 live observation 替换 checkpoint copy：

```text
state.current_observation = reconstructed_observation
```

原因：

live environment 才是继续执行时的真实 source of truth。

---

# 45. Trace Continuation

Phase 2A Resume 继续使用：

```text
同一个 run_id
同一个 run directory
```

而不是创建新 logical run。

Resume event记录：

```text
checkpoint_id
resume_generation
reconstructed_ops
reconstruction_latency
```

---

# 46. Stale Checkpoint

Phase 2A 为避免 trace duplicate：

如果 checkpoint offsets 后已经存在新的 durable：

```text
StepRecord
EnvironmentOperationRecord
```

则该 checkpoint 默认：

```text
STALE
```

不得再次 in-place Resume。

推荐只允许：

```text
latest checkpoint
```

进行 resume。

不要在 Phase 2A 实现 checkpoint branching/fork。

---

# 47. Interrupted Run

新增：

```text
RunStatus.INTERRUPTED
```

或建立明确等价状态。

用户主动在 safe point 中断：

```text
保存 checkpoint
写 result/status = interrupted
关闭 env
```

之后允许 Resume。

不要用：

```text
ERROR
```

表示正常可恢复的 cooperative interruption。

---

# 48. Interrupt Injection

为测试新增：

```text
ControlledInterrupt
```

例如：

```yaml
persistence:
  interrupt_after_step: 3
```

只用于：

```text
tests/evaluation
```

生产默认关闭。

它必须在：

```text
stable checkpoint saved
```

之后中断。

不要用 `os.kill()` 作为主要单元测试机制。

---

# 49. Resume CLI

新增：

```bash
web-harness resume <run_id>
```

默认：

```text
load latest checkpoint
validate
reconstruct
continue
```

以及：

```bash
web-harness inspect-checkpoint <run_id>
```

至少显示：

```text
checkpoint id
next step
task
config hash
git commit
environment ops
active recovery directive
active recovery plan
remaining plan horizon
```

不要打印完整大 Observation。

---

# 50. Phase 2A3 Tests

## FakeEnvironment

必须先使用 deterministic FakeEnvironment。

### R1 — Interrupted vs uninterrupted equivalence

Reference：

```text
run normally to success
```

Resume：

```text
run to checkpoint
interrupt
resume
success
```

比较：

```text
final status
reward
Agent action sequence
Agent StepRecords
Reliability counters
```

---

### R2 — Model calls are not replayed

checkpoint 前：

```text
N model calls
```

Resume reconstruction：

```text
0 model calls
```

继续后只产生剩余真实 decisions。

---

### R3 — Active RecoveryDirective

checkpoint 时：

```text
active recovery directive
```

Resume 后下一 Agent Prompt 必须仍包含该 directive。

---

### R4 — Active RecoveryPlan

checkpoint 时：

```text
remaining_plan_steps = 2
```

Resume 后必须仍为 2。

只由后续真实 Agent StepRecord 消耗。

---

### R5 — Recovery environment action in journal

checkpoint 前包含：

```text
WAIT_AND_REOBSERVE
```

Resume reconstruction 必须重放该 recovery operation。

---

### R6 — Config mismatch

预期：

```text
Resume rejected
zero model calls
```

---

### R7 — Git commit mismatch

同上。

---

### R8 — ActionContract mismatch

同上。

---

### R9 — Environment divergence

Replay 第 N 个 operation 后 fingerprint 不同。

预期：

```text
RESUME_DIVERGENCE
zero post-divergence model calls
```

---

### R10 — stale checkpoint

checkpoint 后存在 durable later step。

旧 checkpoint：

```text
refused
```

---

# 51. MiniWoB Resume Integration

Fake tests 全通过后，使用：

```text
BrowserGym MiniWoB
+
deterministic scripted Agent
```

不要使用在线 LLM 验证 Resume 的因果正确性。

至少 3 类任务：

```text
click-button
enter-text
order-food 或另一组合任务
```

流程：

```text
reference uninterrupted
vs
checkpoint after step N + resume
```

比较：

```text
checkpoint fingerprint
reconstructed fingerprint
final task status/reward
environment op sequence
```

---

# 52. 不要求 Online LLM Resume Benchmark

Phase 2A3 不需要花在线模型额度证明：

```text
resume 后相同模型一定输出相同 action
```

在线 provider 本身存在波动。

Resume 的核心验证是：

```text
前序环境状态正确重建
Harness state 正确恢复
前序模型调用没有重复
后续 execution 可以继续
```

---

# 53. Phase 2A3 Exit Criteria

- [ ] ResumeManager；
- [ ] preflight validation；
- [ ] deterministic environment reconstruction；
- [ ] zero model calls during reconstruction；
- [ ] fingerprint divergence detection；
- [ ] current observation restored from live env；
- [ ] same logical run continuation；
- [ ] stale checkpoint rejection；
- [ ] interrupted status；
- [ ] resume CLI；
- [ ] FakeEnvironment equivalence；
- [ ] MiniWoB scripted integration；
- [ ] 不宣称 arbitrary crash snapshot。

完成后停止提交审批。

---

# 54. Phase 2A4 — Replay / Resume Validation

Phase 2A3 审批通过后实施。

目标不是增加新机制，而是形成正式 benchmark 和工程证据。

---

# 55. Controlled Validation Matrix

至少：

```text
V1 structural replay valid
V2 corrupt trace detected
V3 verifier replay match
V4 checkpoint state roundtrip
V5 interrupted/resumed fake run equivalent
V6 active recovery directive resume
V7 active recovery plan resume
V8 recovery environment operation reconstruction
V9 config drift rejected
V10 git drift rejected
V11 action contract drift rejected
V12 environment divergence rejected
V13 stale checkpoint rejected
V14 MiniWoB simple resume
V15 MiniWoB multi-step resume
```

全部 machine-readable。

---

# 56. Resume Metrics

新增：

```text
checkpoint_count
checkpoint_write_latency_s
checkpoint_bytes

resume_count
resume_reconstructed_environment_ops
resume_reconstruction_latency_s
resume_divergence_count

replayed_model_calls_saved
```

最后一项定义：

```text
replayed_model_calls_saved
=
checkpoint 前已完成、Resume reconstruction 没有重新调用的 model calls
```

不要猜美元 cost。

---

# 57. Replay Metrics

至少：

```text
trace_steps_replayed
trace_events_replayed
artifact_validation_failures
metric_mismatch_count
verification_replay_mismatch_count
```

---

# 58. Phase 2A 报告

生成：

```text
reports/phase2a/
  phase2a_validation_summary.json
  phase2a_validation_report.md
```

由 machine data 自动生成。

必须明确区分：

```text
Offline Replay validation
Checkpoint durability
Resume reconstruction
MiniWoB integration
```

---

# 59. Phase 2A Final Invariants

最终必须成立：

## Trace

```text
Replay 不修改原 trace
```

## Checkpoint

```text
Checkpoint 只在 stable safe point
```

## Environment

```text
每一个 environment-affecting step
都进入 environment operation journal
```

## Resume

```text
Resume reconstruction = zero LLM calls
```

## State

```text
恢复后的 Harness state
与 checkpoint state 一致
```

## Environment validation

```text
reconstructed fingerprint
==
checkpoint fingerprint
```

## Drift

```text
config / code / action contract drift
默认拒绝 Resume
```

## Scope

```text
Phase 2A Resume 支持 deterministic MiniWoB
不宣称所有 BrowserGym benchmark 通用
```

---

# 60. Testing Strategy

继续四层：

### Level 1 Pure Unit

```text
schema
journal
atomic checkpoint
loader
validation
```

### Level 2 Runtime Fake

```text
FakeEnvironment
MockModel
ControlledInterrupt
Resume equivalence
```

### Level 3 Browser Integration

```text
MiniWoB
scripted agent
```

### Level 4 Online

Phase 2A 默认：

```text
不需要
```

---

# 61. CI

Phase 2A unit/regression 继续进入当前 GitHub Actions。

Browser resume integration：

```text
workflow_dispatch
```

即可。

不要把 Chromium integration 强制放到每个 push。

---

# 62. Error Taxonomy 建议

根据实现需要增加：

```text
CHECKPOINT_INVALID
CHECKPOINT_WRITE_ERROR
CHECKPOINT_STALE
RESUME_UNSUPPORTED
RESUME_DIVERGENCE
REPLAY_INVALID_TRACE
```

不要重复已有：

```text
TRACE_WRITE_ERROR
ENVIRONMENT_INIT_ERROR
```

---

# 63. 本阶段明确禁止

```text
PostgreSQL
Redis
remote checkpoint storage
distributed execution
multi-machine resume
browser process memory snapshot
CRIU
always-on native Playwright storage snapshot
checkpoint branching/fork
code-drift migration
WebArena resume
WorkArena resume
long-term memory
skills
human approval
multi-agent
```

这些都不是 Phase 2A。

---

# 64. 实施审批顺序

严格：

```text
Phase 2A1 — Trace Replay
↓
commit
↓
approval

Phase 2A2 — Checkpoint
↓
commit
↓
approval

Phase 2A3 — Resume
↓
commit
↓
approval

Phase 2A4 — Validation
↓
commit
↓
final Phase 2A approval
```

执行智能体当前只实施：

> **Phase 2A1**

不要一次实现 2A1–2A4。

完整计划提前提供，是为了保证 2A1 的 schema/interface 不阻碍后续 Checkpoint/Resume。

---

# 65. Phase 2A1 当前执行范围

本轮仅允许：

```text
trace schema version
structured observation JSON artifacts
TraceBundleLoader
Structural Replay
deterministic verifier semantic replay
ReplayReport
replay CLI
tests
docs
```

禁止本轮创建：

```text
CheckpointManager
EnvironmentOperationJournal
ResumeManager
resume CLI
```

这些属于 2A2/2A3。

---

# 66. Phase 2A1 提交审批时必须提供

```text
Repository:
Branch:
Commit SHA:

Unit/regression:
Ruff:
Browser integration:

Trace schema version:
Structural replay tests:
Semantic verifier replay tests:
Corrupt trace tests:

Checkpoint implemented:
NO

Resume implemented:
NO
```

完成后停止。

---

# 67. Phase 2A 的设计理由

当前 Harness 已经具备非常完整的：

```text
steps.jsonl
events.jsonl
Observation
RunState
ReliabilityState
FailureSignal
RecoveryDirective
RecoveryPlan
```

因此 Phase 2A 应当复用这些数据契约，而不是引入新的 orchestration framework。

最重要的生产工程原则是：

> **Checkpoint 不是“把一个对象 dump 成 JSON”，而是保证 Harness state 和外部 Environment state 能在同一个明确边界被重新对应起来。**

Phase 2A 通过：

```text
serializable Harness state
+
durable environment operation journal
+
deterministic reconstruction
+
fingerprint validation
```

建立第一版真正可验证的 Resume。

---

# 68. Future Compatibility

Phase 2A 完成后，复杂 benchmark 阶段可以针对不同环境实现：

```text
DETERMINISTIC_REPLAY
NATIVE_SNAPSHOT
UNSUPPORTED
```

例如未来某环境可以提供真正可靠的 task snapshot，则：

```text
ResumeManager
```

不需要重写。

只需要新增 environment-specific resume capability。

这正是本阶段保持：

```text
EnvironmentAdapter
Persistence
ResumeManager
```

三层解耦的原因。

---

# Sources

1. Current project `EnvironmentAdapter` and BrowserGym isolation:
   `src/web_harness/env/base.py`
   `src/web_harness/env/browsergym_adapter.py`

2. Current project Trace schema and future checkpoint note:
   `docs/trace_schema.md`

3. Current project serializable `RunState` / `ReliabilityState`:
   `src/web_harness/runtime/state.py`
   `src/web_harness/core/reliability.py`

4. Playwright BrowserContext storage state:
   https://playwright.dev/python/docs/api/class-browsercontext

5. Playwright authentication/storage-state documentation:
   https://playwright.dev/python/docs/auth

Important architectural conclusion from sources 4–5:

Playwright storage state is useful for persisted browser/auth storage, but is not a generic full browser execution snapshot. Phase 2A therefore deliberately uses deterministic environment reconstruction as its first supported Resume strategy rather than claiming complete Browser state serialization.
