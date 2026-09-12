# Reliable Web Workflow Agent Harness
# Phase 1A–1D：Reliable Execution 完整架构与执行计划

> 面向执行智能体的实施文档  
> 基线仓库：`Ethan-Martinez-creater/Browser-harness-project`  
> 冻结基线 Commit：`63d928e358fe31e790b0b06a5c7b801598808b15`  
> Phase 0 状态：PASS / CLOSED  
> 本阶段总目标：构建可观测、可分类、可重试、可恢复、可受控重规划的 Failure Handling Pipeline，并通过可重复实验量化每一层 Harness 机制的贡献。

---

## 0. 文档目的

本文件用于指导执行智能体完成 Phase 1。执行顺序固定为 Phase 1A → 1B → 1C → 1D；每个子阶段完成后必须停止、提交 GitHub 并接受审批。未通过当前阶段，不得提前实现后续机制。

本阶段不是为了单纯提高 MiniWoB 成功率，而是学习生产型 Agent Harness 中最核心的可靠执行链：

```text
Failure Detection
      ↓
Failure Classification
      ↓
Policy Decision
      ↓
Retry / Recovery
      ↓
Escalation
      ↓
Controlled Replanning
```

所有机制必须满足：可开关、可追踪、可测量、可注入故障验证、可与 Phase 0 baseline 对照。

---

# 1. Phase 0 冻结基线

Phase 1 的概念基线固定为：

```text
63d928e358fe31e790b0b06a5c7b801598808b15
```

不得破坏以下契约：

- `EpisodeRunner` 仍是唯一自研主循环。
- BrowserGym 仍只能通过 `EnvironmentAdapter` 访问。
- Action Contract 与 Environment action mapping 保持同一事实来源。
- `obs_N` = action 前、模型实际看到的 observation。
- `next_obs_N` = action 后 observation。
- `steps.jsonl` 仍是主 Agent trajectory。
- Reliability 机制全部关闭时，不得增加额外模型调用、环境 action、prompt 内容或 step 消耗。

新增回归测试：

```text
tests/regression/test_phase0_baseline_compat.py
```

至少验证：模型调用次数、环境 step 数、action sequence、termination、StepRecord 数量和主 trace 语义均与 Phase 0 预期一致。

---

# 2. Phase 1 最终架构

```text
Observation
    │
    ▼
DecisionExecutor
    │
    ├── model call
    └── pre-action Retry（仅 model/API/format）
    │
    ▼
Environment.step(action)
    │
    ▼
VerificationEngine
    │
    ▼
FailureSignal[]
    │
    ▼
FailurePolicyEngine
    │
    ├── CONTINUE
    ├── RETRY
    ├── RECOVER
    ├── REPLAN
    └── ABORT
          │
          ▼
 RecoveryManager / Replanner
          │
          ▼
      next observation
```

必须明确：

```text
Detection ≠ Retry ≠ Recovery ≠ Replanning
```

Detection 只判断发生了什么；Retry 只重试尚未改变环境的 model-side 操作；Recovery 改变执行上下文；Replanning 只在局部恢复持续失败时作为 escalation 使用。

---

# 3. 推荐目录

```text
src/web_harness/
├── core/
│   ├── reliability.py        # NEW：跨层可靠性数据契约
│   └── events.py             # NEW：运行时事件模型
│
├── reliability/              # NEW
│   ├── fingerprint.py
│   ├── verifier.py
│   ├── policy.py
│   ├── retry.py
│   ├── recovery.py
│   ├── replanner.py
│   ├── budget.py
│   └── detectors/
│       ├── action_error.py
│       ├── observation_health.py
│       ├── progress.py
│       └── loop.py
│
├── runtime/
│   ├── episode_runner.py
│   ├── state.py
│   └── decision_executor.py  # Phase 1B
│
├── evaluation/
│   ├── metrics.py
│   ├── benchmark_runner.py
│   └── fault_injection.py    # NEW
│
└── observability/
    └── trace.py
```

不得引入 LangGraph、Deep Agents、CrewAI、AutoGen 等第三方 Agent Runtime。

---

# 4. ErrorType 与 FailureSignal 分离

当前 `ErrorType` 继续用于表示具体技术错误，例如 `MODEL_API_ERROR`、`ACTION_EXECUTION_ERROR`、`MAX_STEPS_EXCEEDED`。

新增 `FailureKind` 表示 Harness 对失败的语义理解：

```python
class FailureKind(StrEnum):
    MODEL_API_TRANSIENT = "MODEL_API_TRANSIENT"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    ACTION_ERROR = "ACTION_ERROR"
    OBSERVATION_INVALID = "OBSERVATION_INVALID"
    NO_PROGRESS = "NO_PROGRESS"
    LOOP_DETECTED = "LOOP_DETECTED"
    TASK_FAILED = "TASK_FAILED"
    UNKNOWN = "UNKNOWN"
```

新增：

```python
class FailureSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
```

```python
class FailureSignal(BaseModel):
    kind: FailureKind
    severity: FailureSeverity
    source: str
    signature: str
    retryable: bool = False
    recoverable: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)
```

`signature` 必须 deterministic，用于识别重复失败。不得直接把完整异常字符串作为 signature。

---

# 5. Observation Fingerprint

实现 `reliability/fingerprint.py`，用于判断页面是否真正发生状态变化。

只使用稳定字段：

```text
URL
AXTree
必要 DOM
open page URLs
```

明确排除：

```text
elapsed_time_s
last_action
last_action_error
screenshot_path
artifact path
```

建议：

```python
class ObservationFingerprint(BaseModel):
    url_hash: str
    content_hash: str
    combined_hash: str
```

只做 deterministic normalization，不使用 embedding 或 LLM semantic diff。

---

# 6. Runtime Event Stream

保留 Phase 0：

```text
steps.jsonl
```

新增：

```text
events.jsonl
```

用于记录 verifier、policy、retry、recovery、replan、budget 事件。

```python
class RuntimeEventType(StrEnum):
    VERIFICATION = "verification"
    POLICY_DECISION = "policy_decision"
    RETRY = "retry"
    RECOVERY = "recovery"
    REPLAN = "replan"
    BUDGET = "budget"
```

```python
class RuntimeEvent(BaseModel):
    event_id: str
    run_id: str
    timestamp: str
    event_type: RuntimeEventType
    step_index: int | None = None
    attempt_index: int | None = None
    component: str
    outcome: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
```

要求：JSONL、每事件 flush/fsync、遵循现有 secret policy，不改变 `steps.jsonl` 的含义。

---

# 7. ReliabilityState 与 ReliabilityBudget

在 `RunState` 中增加独立可靠性状态：

```python
class ReliabilityState(BaseModel):
    recent_state_fingerprints: list[str] = []
    recent_actions: list[str] = []
    failure_counts: dict[str, int] = {}
    consecutive_failure_count: int = 0
    retry_count: int = 0
    recovery_count: int = 0
    replan_count: int = 0
    extra_model_calls: int = 0
    active_recovery_directive: RecoveryDirective | None = None
    active_recovery_plan: RecoveryPlan | None = None
```

所有可靠性机制必须有预算：

```python
class ReliabilityBudget(BaseModel):
    max_model_api_retries_per_call: int = 2
    max_parse_retries_per_call: int = 1
    max_recoveries_per_episode: int = 3
    max_replans_per_episode: int = 1
    max_extra_model_calls_per_episode: int = 6
```

超限必须 controlled abort，不能无限循环。

---

# 8. 配置结构

```yaml
reliability:
  enabled: false

  verification:
    enabled: false
    mode: shadow
    no_progress:
      enabled: true
    loop:
      enabled: true
      consecutive_threshold: 2

  retry:
    enabled: false
    model_api:
      max_retries: 2
      backoff_ms: [500, 1000]
    model_output:
      max_retries: 1

  recovery:
    enabled: false
    max_recoveries_per_episode: 3
    wait_ms: 500
    block_repeated_action_steps: 1

  replanning:
    enabled: false
    max_replans_per_episode: 1
    recovery_failures_before_replan: 2
    plan_horizon_steps: 3

  budget:
    max_extra_model_calls_per_episode: 6
```

`configs/baseline.yaml` 默认只设：

```yaml
reliability:
  enabled: false
```

新增：

```text
configs/phase1/verifier_shadow.yaml
configs/phase1/retry.yaml
configs/phase1/recovery.yaml
configs/phase1/replanning.yaml
```

---

# 9. Phase 1A — Failure Observation / Verification

## 9.1 目标

建立：

```text
pre observation + action + env result
              ↓
       VerificationEngine
              ↓
        FailureSignal[]
```

但默认采用 `shadow` 模式，不改变控制流。

## 9.2 StepVerifier

```python
class StepVerifier(Protocol):
    def verify(
        self,
        *,
        task: TaskSpec,
        pre_observation: Observation,
        action: str,
        env_step: EnvironmentStep,
        history: list[StepRecord],
        reliability_state: ReliabilityState,
    ) -> VerificationResult:
        ...
```

```python
class VerificationStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
```

```python
class VerificationResult(BaseModel):
    status: VerificationStatus
    signals: list[FailureSignal]
    pre_fingerprint: str | None
    post_fingerprint: str | None
    state_changed: bool | None
```

## 9.3 Detectors

### ActionErrorDetector

若 `EnvironmentStep.action_error` 非空：

```text
FailureKind.ACTION_ERROR
severity = ERROR
recoverable = true
retryable = false
```

这里 retryable=false 的含义是：禁止 blind retry 已可能有副作用的 browser action。

### ObservationHealthDetector

若 URL 与 AXTree 同时无有效内容，或 adapter 明确报告 observation failure：

```text
OBSERVATION_INVALID
recoverable = true
```

### NoProgressDetector

条件：

```text
pre_fingerprint == post_fingerprint
AND reward == 0
AND no action_error
AND not terminated
```

第一次只产生 WARNING。

### LoopDetector

建议以：

```text
(pre_state_hash, normalized_action, post_state_hash)
```

作为 transition signature。相同 transition 连续出现 >= 2 次才判定 `LOOP_DETECTED`。不能仅因为 action 相同就判定 loop。

### TaskFailureDetector

`terminated=true` 且 `reward<=0` 时记录 `TASK_FAILED`，但不得 reset 后继续。

## 9.4 Trace

每个 Agent step 后写 `VerificationEvent`。StepRecord 可增加摘要字段：

```text
verification_status
failure_kinds
```

详细 evidence 放在 `events.jsonl`。

## 9.5 Metrics

新增：

```text
verification_count
failure_signal_count
verification_signal_rate
episodes_with_failure_signal
failure_kind_counts
```

## 9.6 Tests

新增：

```text
tests/unit/reliability/test_fingerprint.py
tests/unit/reliability/test_action_error_detector.py
tests/unit/reliability/test_observation_health_detector.py
tests/unit/reliability/test_progress_detector.py
tests/unit/reliability/test_loop_detector.py
tests/unit/reliability/test_verifier.py
tests/unit/test_runtime_events.py
tests/regression/test_phase0_baseline_compat.py
```

必须覆盖 normal progress、action error、empty observation、single no-progress、consecutive loop。

## 9.7 Phase 1A Smoke

运行：

```text
12 MiniWoB × seed 0
baseline + verifier shadow
```

不要求成功率提高。必须证明没有额外模型调用和额外环境 action。

## 9.8 Phase 1A Exit Criteria

- [ ] FailureSignal / VerificationResult 建立。
- [ ] deterministic fingerprint。
- [ ] 4 个核心 detector 完成。
- [ ] events.jsonl 完成。
- [ ] shadow mode 不改变 control flow。
- [ ] reliability metrics 完成。
- [ ] Phase 0 regression pass。
- [ ] MiniWoB smoke 完成。
- [ ] 尚未实现 Retry / Recovery / Replanning。

**Phase 1A 完成后必须停止并提交审批。**

---

# 10. Phase 1B — Controlled Retry

## 10.1 目标

只处理尚未改变环境状态的失败：

```text
MODEL_API_ERROR
MODEL_OUTPUT_PARSE_ERROR
```

禁止默认重试 browser click / submit 等可能有副作用的 action。

## 10.2 DecisionExecutor

新增：

```text
runtime/decision_executor.py
```

```python
class DecisionExecutor:
    def execute(
        self,
        *,
        agent: Agent,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        action_contract: ActionContract,
        reliability_state: ReliabilityState,
    ) -> DecisionExecutionResult:
        ...
```

当 retry 关闭时只能调用一次 `agent.decide()`。

```python
class DecisionExecutionResult(BaseModel):
    success: bool
    turn: AgentTurn | None
    prompt: PromptBundle | None
    attempts: int
    retry_count: int
    terminal_error_type: ErrorType | None
    terminal_error_message: str | None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
```

## 10.3 Model API Retry

默认：

```text
max_retries = 2
backoff = 500ms, 1000ms
```

Phase 1 benchmark 不使用 jitter，保证可复现。

## 10.4 Model Output Repair Retry

默认：

```text
max_retries = 1
```

修复 prompt 仅增加：

```text
Previous response did not match the required JSON action format.
Return exactly one valid JSON object using the current ActionContract.
```

不得引入 planner。

## 10.5 Failed Parse Token Accounting

当前 parse error 发生在模型已输出内容之后，因此必须记录该失败调用消耗的 input/output tokens。推荐扩展 `ModelOutputParseError`，保存 raw_text、input_tokens、output_tokens、model_name。禁止把失败 parse token 记成 0。

## 10.6 Retry Trace / Metrics

Retry 不增加 Agent step_index；每次 retry 写 RuntimeEvent。

新增：

```text
retry_count
episodes_with_retry
retry_success_count
retry_exhausted_count
extra_model_calls
retry_input_tokens
retry_output_tokens
retry_latency_s
```

`retry_success` 定义为同一 decision cycle 经 retry 后最终成功产生有效 `ActionDecision`。

## 10.7 Fault Injection

新增 `evaluation/fault_injection.py` 与 `FaultInjectingModelAdapter`。

至少支持：

```text
MODEL_API_ERROR on call N
MODEL_OUTPUT_PARSE_ERROR on call N
```

必须有四个 deterministic scenario：

1. 首次 API error，第二次成功 → baseline fail / retry success。
2. 前两次 API error，第三次成功 → max_retries=2 时成功。
3. 连续三次 API error → retry exhausted。
4. 首次 malformed output，第二次 valid JSON → format retry success。

## 10.8 Phase 1B Smoke

运行：

```text
12 MiniWoB × seed 0
verification active
retry on
recovery off
replanning off
```

不要求成功率 > 10/12。

## 10.9 Phase 1B Exit Criteria

- [ ] DecisionExecutor 完成。
- [ ] API retry 完成。
- [ ] parse repair retry 完成。
- [ ] 不 blind-retry browser action。
- [ ] retry event / metrics 完成。
- [ ] failed parse token accounting 正确。
- [ ] FaultInjectingModelAdapter 完成。
- [ ] 4 个 deterministic scenarios 全通过。
- [ ] baseline compatibility 继续通过。
- [ ] 尚未实现 Recovery / Replanner。

**完成后停止并提交审批。**

---

# 11. Phase 1C — Recovery Policy

## 11.1 目标

处理：

```text
ACTION_ERROR
OBSERVATION_INVALID
NO_PROGRESS
LOOP_DETECTED
```

Retry 尝试同一个 pre-action computation；Recovery 则改变后续执行条件。

## 11.2 FailurePolicyEngine

新增 `reliability/policy.py`。

```python
class PolicyAction(StrEnum):
    CONTINUE = "continue"
    RETRY = "retry"
    RECOVER = "recover"
    REPLAN = "replan"
    ABORT = "abort"
```

Phase 1C 中 `REPLAN` 只保留 enum，不执行。

## 11.3 RecoveryDirective

```python
class RecoveryKind(StrEnum):
    REDECIDE_WITH_FEEDBACK = "redecide_with_feedback"
    BLOCK_REPEATED_ACTION = "block_repeated_action"
    WAIT_AND_REOBSERVE = "wait_and_reobserve"
```

```python
class RecoveryDirective(BaseModel):
    kind: RecoveryKind
    reason: str
    feedback: str | None
    blocked_actions: list[str]
    wait_ms: int | None
    expires_after_agent_steps: int = 1
```

## 11.4 固定 Recovery Rules

### ACTION_ERROR

使用：

```text
REDECIDE_WITH_FEEDBACK
```

下一轮 prompt 告知上一 browser action 失败，不得假设已成功；将失败 action 临时 block 1 个 Agent step。不得永久 block。

### OBSERVATION_INVALID

使用：

```text
WAIT_AND_REOBSERVE
```

Harness 显式执行 `noop(wait_ms=<config>)`。这是 Recovery Event，不是 Agent Step，但必须消耗 recovery budget 并记录 resulting observation。若 wait 导致 termination/reward，按真实环境结果处理。

### NO_PROGRESS

单次只 `CONTINUE`。达到 loop threshold 后才恢复。

### LOOP_DETECTED

使用：

```text
BLOCK_REPEATED_ACTION
```

下一轮 prompt 明确禁止重复刚才的 transition/action，要求基于当前 state 选择另一种策略。

### TASK_FAILED

`ABORT`。不得 reset task。

## 11.5 Prompt Feedback

`PromptBuilder.build()` 增加 optional：

```text
recovery_directive
recovery_plan
```

仅非空时才增加 `# Reliability feedback`。正常路径 prompt 必须保持 Phase 0 兼容。

## 11.6 Recovery Success

局部 recovery success 定义：Recovery 后最多 2 个 Agent steps 内，状态 fingerprint 改变或 task success，且没有再次出现同一 failure signature。

新增：

```text
recovery_count
episodes_with_recovery
recovery_success_count
recovery_failed_count
recovered_episode_count
recovery_extra_steps
recovery_latency_s
```

`recovered_episode_count` = episode 曾触发 Recovery 且最终 success。

## 11.7 FaultInjectingEnvironmentAdapter

扩展 fault injection，支持：

```text
ACTION_ERROR
STALE_OBSERVATION
EMPTY_OBSERVATION
```

必须代理 `action_contract/reset/step/close` 语义。

## 11.8 Deterministic Scenarios

1. Action error once → recovery directive → alternate action → success。
2. Empty observation once → WAIT_AND_REOBSERVE → observation 恢复。
3. repeated state/action transition → LOOP_DETECTED → block repeated action。
4. persistent failure → recovery budget exhausted → controlled termination，无无限循环。

## 11.9 Phase 1C Smoke

```text
12 MiniWoB × seed 0
verification active
retry on
recovery on
replanning off
```

## 11.10 Phase 1C Exit Criteria

- [ ] FailurePolicyEngine 完成。
- [ ] RecoveryDirective 完成。
- [ ] ACTION_ERROR / OBSERVATION_INVALID / LOOP recovery 完成。
- [ ] recovery budget 生效。
- [ ] recovery trace / metrics 完成。
- [ ] FaultInjectingEnvironmentAdapter 完成。
- [ ] 4 个 deterministic recovery scenarios 通过。
- [ ] baseline compatibility 通过。
- [ ] Replanner 未启用。

**完成后停止并提交审批。**

---

# 12. Phase 1D — Controlled Replanning

## 12.1 目标

只有当 Retry 已失败、局部 Recovery 仍持续失败时，允许有限的 recovery replan。不是 always-on Planner。

## 12.2 Replan Trigger

默认仅允许：

- 同一 `LOOP_DETECTED` 经一次 recovery 后再次出现；
- 连续 recovery failures >= 2；
- 同一 failure signature 超过阈值。

以下情况禁止触发 Replan：

```text
MODEL_API_ERROR
MODEL_OUTPUT_PARSE_ERROR
ENVIRONMENT_INIT_ERROR
成功 termination
```

## 12.3 RecoveryPlan

```python
class RecoveryPlan(BaseModel):
    diagnosis: str
    immediate_subgoal: str
    strategy_steps: list[str]
    avoid_actions: list[str]
    horizon_steps: int
    created_at_step: int
```

限制：

```text
strategy_steps <= 4
horizon_steps <= config.plan_horizon_steps
diagnosis 必须是短摘要，不保存 chain-of-thought
```

## 12.4 Replanner

```python
class Replanner(Protocol):
    def replan(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        failure_signals: list[FailureSignal],
        previous_recoveries: list[RuntimeEvent],
        action_contract: ActionContract,
    ) -> RecoveryPlan:
        ...
```

输入只包含 task goal、当前 observation、最近有限 steps、最近 failure/recovery 和 ActionContract 摘要。默认 recent_steps=6。

## 12.5 Provider-neutral Structured Generation

Replanner 不得直接 import `openai`。

推荐新增：

```python
class StructuredModelAdapter(Protocol):
    def generate_structured(
        self,
        *,
        prompt: PromptBundle,
        schema: type[BaseModel],
    ) -> StructuredModelOutput:
        ...
```

OpenAI-compatible provider 实现该接口。Replan 本身可复用 Phase 1B 的 API retry 与一次 structured-output repair，并计入 extra_model_calls、tokens、latency 和 reliability budget。

## 12.6 Plan 生效方式

保存到：

```text
ReliabilityState.active_recovery_plan
```

最多影响 `plan_horizon_steps` 个 Agent steps。Prompt 增加：

```text
# Recovery plan
Immediate subgoal: ...
Strategy: ...
Avoid: ...
```

Plan 到期或 task success 后自动失效。默认 `max_replans_per_episode=1`。

## 12.7 Replan Trace / Metrics

RecoveryPlan 正文写入：

```text
artifacts/replan_000.json
```

`events.jsonl` 记录 trigger、horizon、result 和 artifact ref。

新增：

```text
replan_count
episodes_with_replan
replan_success_count
replan_failed_count
replan_input_tokens
replan_output_tokens
replan_latency_s
reliability_extra_model_calls
reliability_extra_tokens
reliability_extra_latency_s
```

局部 replan success：plan horizon 内发生 state progress，且没有立即重复同一 failure signature。

## 12.8 Deterministic Scenario

必须构建：初始 Agent 连续选择错误 action → loop → recovery → 仍无 progress → replan → plan 指向 alternate subgoal → scripted model 走正确路径 → success。

要求同一 scenario：

```text
replanning off → failure / budget stop
replanning on  → success
```

这是 Harness causal test，不是模型能力 benchmark。

## 12.9 Phase 1D Exit Criteria

- [ ] Replan trigger 严格受控。
- [ ] structured RecoveryPlan。
- [ ] provider-neutral structured generation。
- [ ] plan horizon 与 max replan budget。
- [ ] replan trace / metrics / tokens。
- [ ] deterministic causal scenario 通过。
- [ ] baseline compatibility 继续通过。

完成后提交审批；通过后才运行 Phase 1 final ablation。

---

# 13. Phase 1 Final Ablation

所有 1A–1D 通过后，使用 **同一个最终 commit**，只改变 config，不允许 baseline 用旧 commit、recovery 用新 commit。

五个 variants：

```text
A0 baseline
  reliability off

A1 detection only
  verifier shadow

A2 retry
  verifier active + retry

A3 recovery stack
  verifier active + retry + recovery

A4 full Phase 1
  verifier active + retry + recovery + replanning
```

MiniWoB：

```text
12 tasks × seeds [0,1,2]
= 36 episodes / variant
= 180 episodes total
```

如果 API 成本/限额确实无法支持，可缩为 8 tasks × 3 seeds × 5 variants，但执行智能体不得自行缩小，必须在提交说明中给出原因。

---

# 14. Final Metrics

主指标：

```text
success_rate
mean_reward
```

效率：

```text
mean_steps
median_steps
mean_duration_s
input_tokens
output_tokens
tokens_per_success
```

Reliability：

```text
episodes_with_failure_signal
failure_kind_counts
retry_count
retry_success_rate
retry_exhausted_count
recovery_count
recovery_success_rate
recovered_episode_count
replan_count
replan_success_rate
```

Overhead：

```text
extra_model_calls
reliability_extra_tokens
reliability_extra_latency_s
```

`tokens_per_success = total_tokens / num_successes`；若成功数为 0，输出 null。

---

# 15. Controlled Fault Suite 与 Natural Benchmark 必须分开

Controlled Fault Suite 回答：

> 指定故障发生时，Harness 机制是否按设计工作？

Natural MiniWoB Benchmark 回答：

> 在真实 trajectory 中，这些机制的净收益与成本是什么？

不得把两者混成一个 success rate。

---

# 16. Trace / Step / Budget Invariants

- 一个 Agent browser action = 一个 `StepRecord`。
- model retry 不增加 Agent step_index。
- Recovery harness action（如 wait/noop）不写成 Agent StepRecord，而写 `RuntimeEvent.RECOVERY`，但必须记录环境结果并消耗 recovery budget。
- Replan 不增加 Agent step。
- `TaskSpec.max_steps` 与 `ReliabilityBudget` 分离。
- 可靠性操作必须计入额外模型调用、token 和 latency。

---

# 17. 最终 EpisodeRunner 应保持可读

目标结构：

```python
observation = env.reset(task)

for step_idx in range(task.max_steps):
    decision_result = decision_executor.execute(...)
    if not decision_result.success:
        terminate(...)

    env_step = env.step(decision.action)

    verification = verifier.verify(
        pre_observation=observation,
        action=decision.action,
        env_step=env_step,
        ...
    )

    record_step(...)
    record_verification(...)

    observation = env_step.observation

    if env_step.terminated or env_step.truncated:
        break

    policy = failure_policy.decide(...)

    if policy == CONTINUE:
        continue
    if policy == RECOVER:
        observation = recovery_manager.recover(...)
    if policy == REPLAN:
        state.reliability.active_recovery_plan = replanner.replan(...)
    if policy == ABORT:
        break
```

允许拆 helper，但 Review 时必须能直接看出 Decision → Action → Verification → Policy → Recovery/Replan 的控制链。禁止构建隐藏 flow 的 callback framework。

---

# 18. Error Taxonomy 扩展

如实现需要，新增：

```text
RETRY_EXHAUSTED
REPLAN_FAILED
```

现有 `VERIFICATION_FAILED`、`RECOVERY_FAILED`、`BUDGET_EXCEEDED` 优先复用，避免重复 enum。

`NO_PROGRESS`、`LOOP_DETECTED` 属于 FailureKind，不属于 ErrorType。

---

# 19. 测试层级

## Level 1：Pure Unit

fingerprint、detectors、policy、budget、recovery directive、replan trigger、metrics。

## Level 2：Runtime Component

FakeEnvironment + MockModel + Fault Injection，验证 control flow、events、budget、state update。

## Level 3：Browser Integration

真实 BrowserGym + MiniWoB，但不要求在线模型。验证 ActionContract、recovery noop、observation、trace。

## Level 4：Online Smoke

真实模型 + 12 MiniWoB，每个子阶段一次。

---

# 20. 最小 CI

Phase 1A 允许增加：

```text
.github/workflows/test.yml
```

默认只运行：

```bash
uv sync
uv run pytest tests/unit tests/regression
uv run ruff check .
```

Browser integration 可通过 `workflow_dispatch` 手动运行，不要求默认 CI 下载浏览器。

---

# 21. 报告结构

```text
reports/phase1/
  phase1a_verifier_report.md
  phase1b_retry_report.md
  phase1c_recovery_report.md
  phase1d_replan_report.md
  fault_suite_summary.json
  phase1_ablation_summary.json
  phase1_ablation_episodes.csv
  phase1_ablation_report.md
```

继续遵循：machine-readable result → renderer → Markdown。数值不得手工抄写。

---

# 22. 本阶段禁止事项

Phase 1A–1D 均禁止：

```text
Checkpoint / resume
persistent DB
long-term memory
skill learning
domain skills
human approval
multi-agent
sub-agent
vision model
WebArena deployment
WorkArena deployment
parallel benchmark
Web UI
RBAC / OAuth
```

这些属于后续阶段。

---

# 23. 每阶段审批提交格式

必须提供：

```text
Repository:
Branch:
Commit SHA:
pytest unit:
pytest regression:
integration:
ruff:
MiniWoB experiment ID:
```

Phase 1A 额外：verification events、failure signals、failure kinds、baseline control flow unchanged yes/no。

Phase 1B 额外：四个 retry fault scenario 结果、retry_count、retry_success、retry_exhausted、extra_model_calls、retry tokens。

Phase 1C 额外：四个 recovery scenario 结果、recovery_count、recovery_success、recovery_failed、recovered episodes。

Phase 1D 额外：replan causal scenario、replan_count、success/fail、token overhead。

---

# 24. Phase 1 Overall Exit Criteria

## Architecture

- [ ] Detection 与 Policy 分离。
- [ ] Retry 与 Recovery 分离。
- [ ] Replanning 只作为 escalation path。
- [ ] EpisodeRunner 仍是唯一主 runtime。
- [ ] Reliability disabled 时 baseline 行为兼容。

## Verification

- [ ] deterministic fingerprint。
- [ ] ActionError / ObservationHealth / NoProgress / Loop detector。
- [ ] shadow mode。

## Retry

- [ ] API retry。
- [ ] output repair retry。
- [ ] retry budget / trace / token accounting。
- [ ] 不 blind retry browser action。

## Recovery

- [ ] REDECIDE_WITH_FEEDBACK。
- [ ] BLOCK_REPEATED_ACTION。
- [ ] WAIT_AND_REOBSERVE。
- [ ] recovery budget / metrics / controlled abort。

## Replanning

- [ ] failure-triggered only。
- [ ] structured RecoveryPlan。
- [ ] plan horizon / max replan budget。
- [ ] replan trace / tokens。

## Observability

- [ ] `steps.jsonl` 保持主轨迹语义。
- [ ] `events.jsonl` 完成。
- [ ] verification/retry/recovery/replan 全部可审计。
- [ ] `inspect-run` 能显示 reliability summary。

## Evaluation

- [ ] controlled fault suite。
- [ ] natural MiniWoB smoke。
- [ ] 同 commit config-based ablation。
- [ ] Phase 1 final 3-seed matrix。
- [ ] machine-generated report。

---

# 25. Phase 1 最终学习目标

完成后项目必须能够回答：

1. Agent 失败时 Harness 能否发现？
2. 能否区分 model failure、action failure、observation failure、no-progress 和 loop？
3. 对 pre-action transient failure，Retry 是否有效，成本是多少？
4. 对 environment/progress failure，Recovery 是否有效，增加多少 step/token/time？
5. 局部 Recovery 失败后，受控 Replanning 是否值得额外模型成本？

最终结论不应是“Agent 更强了”，而应是：

> 在同一模型、同一任务、同一环境下，不同 Harness Failure Handling 机制分别为可靠性带来了多少可量化收益，以及付出了多少执行与推理成本。

---

# 26. 严格执行顺序

```text
Phase 1A
  ↓
Commit + Review Gate
  ↓ PASS
Phase 1B
  ↓
Commit + Review Gate
  ↓ PASS
Phase 1C
  ↓
Commit + Review Gate
  ↓ PASS
Phase 1D
  ↓
Commit + Review Gate
  ↓ PASS
Final Phase 1 Ablation
  ↓
Phase 1 Closure Review
```

**当前执行智能体只允许实施 Phase 1A。**

本文件提前描述 1A–1D，是为了确保 Phase 1A 的数据模型、Trace 和接口设计不会阻碍后续扩展。Phase 1A 审批通过后，再确认是否直接进入本文件的 1B，或基于实际代码状态下发微调版 1B 指令。
