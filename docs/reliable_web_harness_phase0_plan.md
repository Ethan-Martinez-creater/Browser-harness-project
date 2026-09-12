# Reliable Web Workflow Agent Harness — Phase 0 实施计划

## 0. 文档目的

本文件用于指导执行智能体完成 **Reliable Web Workflow Agent Harness** 项目的 Phase 0。

本项目的主要目的不是构建一个功能繁杂的浏览器自动化产品，而是系统学习：在真实生产需求下，一个现代 Agent Harness 应该如何进行模块划分、运行、观测、验证、恢复、评测与迭代。

当前开发模式为：

1. 方案设计、架构选择、实验设计与验收标准由上层研究/评审角色确定。
2. 执行智能体严格按照本文档实施，不自行扩大范围或替换关键架构。
3. Phase 0 完成后，将代码、测试、运行记录与实验结果提交 GitHub。
4. 进入下一阶段前先进行架构和实现审批。

Phase 0 的目标是建立一个**最小、透明、可扩展、可评测的 Harness 骨架和 baseline**。高级能力将在后续阶段逐项加入，并通过消融实验验证其价值。

---

## 1. 项目定位

项目工作名：

**Reliable Web Workflow Agent Harness**

可使用仓库名：

```text
reliable-web-harness
```

核心研究问题：

> 在保持模型不变的前提下，Planning、State Management、Verification、Retry/Recovery、Checkpoint、Context Management、Skills、Human-in-the-loop、Tracing 与 Evaluation 等 Harness 机制，能够在多步骤 Web 工作流中带来多少可靠性、效率和成本收益？

因此，本项目关注的是：

```text
Task
  ↓
Harness Runtime
  ↓
Model Decision
  ↓
Browser Environment
  ↓
Observation
  ↓
State / Trace / Verification
  ↓
Next Step or Termination
```

后续逐步扩展为：

```text
Task
  ↓
Planner
  ↓
Harness Runtime
  ├── Context Manager
  ├── State Manager
  ├── Policy / Guardrail
  ├── Budget Manager
  ├── Skill Manager
  └── Checkpoint Manager
          ↓
       Executor
          ↓
 Browser Environment
          ↓
 Observation
          ↓
   Step Verifier
          ↓
   Retry / Recovery
          ↓
    Task Verifier
          ↓
 Trace / Replay / Metrics
          ↓
 Benchmark / Ablation
```

---

## 2. Phase 0 的核心决策

### 2.1 自己实现 Harness Runtime

**不得直接使用 LangGraph、Deep Agents、OpenAI Agents SDK、CrewAI 等框架作为本项目主 Agent Runtime。**

原因：

这些框架已经封装了不同程度的：

- Agent loop
- state
- checkpoint
- persistence
- guardrail
- tool dispatch
- human-in-the-loop
- tracing
- context management

如果直接采用，虽然开发更快，但会掩盖本项目最需要学习的 Harness 内部机制。

允许：

- 阅读它们的源码、接口和架构作为设计参考。
- 后期增加兼容适配器进行对比。
- 使用普通模型 SDK，例如 `openai` Python SDK。
- 使用 BrowserGym 作为浏览器环境和 benchmark 适配层。

禁止：

```text
Deep Agents -> Browser tool -> Benchmark
```

作为核心架构。

必须采用：

```text
Our Harness Runtime -> EnvironmentAdapter -> BrowserGym
```

---

### 2.2 BrowserGym 作为环境层，不作为 Agent Runtime

BrowserGym 已提供统一的 Web Agent 环境接口，其核心循环为：

```python
obs, info = env.reset()

while not done:
    action = ...
    obs, reward, terminated, truncated, info = env.step(action)
```

其 Observation 可包含：

- goal
- URL
- screenshot
- DOM snapshot
- accessibility tree
- focused element
- last action
- last action error
- elapsed time
- open tabs/pages

因此不应重新开发 Playwright 浏览器基础设施。

本项目实现：

```text
Harness
  ↓
EnvironmentAdapter
  ↓
BrowserGym
  ↓
Playwright / Chromium
```

EnvironmentAdapter 的价值在于：

1. 隔离 BrowserGym 具体 API。
2. 后期可以增加其他环境。
3. Benchmark 和 Harness 解耦。
4. 避免 Agent Runtime 直接依赖 Gymnasium 数据结构。

---

### 2.3 Phase 0 只做最小 Baseline

Baseline 必须故意保持简单。

Phase 0 **不实现**：

- 独立 Planner
- Retry policy
- Recovery policy
- Replanning
- Checkpoint / resume
- Long-term memory
- Skill learning
- Domain skills
- Human approval
- Screenshot/VLM 路径
- Multi-agent
- Sub-agent
- LLM verifier
- Context compression
- 自动 helper 生成

原因：

这些模块后面必须逐个加入，并比较：

```text
Baseline
vs
Baseline + Planning
vs
Baseline + Verification
vs
Baseline + Recovery
...
```

如果 Phase 0 一次实现所有功能，将失去实验基准。

---

## 3. 参考项目与我们需要吸收的设计

### 3.1 BrowserGym

需要吸收：

- 标准化 Environment 接口
- Observation / Action 分离
- benchmark task abstraction
- task-level validation
- benchmark 可扩展性

BrowserGym 当前统一支持 MiniWoB、WebArena、WebArena-Verified、VisualWebArena、WorkArena、AssistantBench、WebLINX、OpenApps、TimeWarp 等 benchmark。

我们不复制 BrowserGym，只通过 adapter 使用。

---

### 3.2 AgentLab

AgentLab 证明 Web Agent 实验系统至少需要：

- Agent 与 Benchmark 分离
- Experiment configuration
- 批量运行
- trace 收集
- result analysis
- reproducibility
- 多模型接口
- benchmark 统一入口

Phase 0 不直接依赖 AgentLab。

但本项目的数据模型和实验目录设计需要保证后期可以与 AgentLab 结果进行对比。

---

### 3.3 WebArena-Verified

后续正式 benchmark 优先采用 WebArena-Verified，而不是原始 WebArena 作为唯一指标。

原因：

WebArena-Verified 提供：

- manually audited tasks
- corrected reference answers/evaluators
- deterministic scoring
- offline evaluation
- network trace replay
- WebArena-Verified Hard 258-task subset

这些属性非常符合 Harness 实验对可重复性的要求。

Phase 0 暂不部署完整 WebArena。

---

### 3.4 WorkArena / WorkArena++

WorkArena 用于后期验证“生产式知识工作”。

当前公开设计包含：

- WorkArena L1：33 类 atomic task，19,912 个实例
- WorkArena++：682 个组合式任务
- planning
- reasoning
- memory
- knowledge work workflow

WorkArena 不作为 Phase 0 环境依赖。

后期作为 production-like benchmark。

---

### 3.5 Browser Harness

Browser Use 的 browser-harness 给本项目两个重要启发：

1. 核心 Harness 应保持薄和透明。
2. Agent workspace / reusable skills 应与 protected core 分离。

后续 Skill 阶段可以参考：

```text
harness core
workspace/
  helpers/
  skills/
```

但 Phase 0 不实现 adaptive skill generation。

---

### 3.6 Deep Agents / OpenAI Agents SDK

它们用于反向检查我们的模块完整度。

Deep Agents 当前覆盖：

- filesystem
- context management
- persistent memory
- subagents
- skills
- human-in-the-loop
- checkpoint/persistence
- tracing/evaluation

OpenAI Agents SDK 当前重点覆盖：

- agent loop
- tools
- guardrails
- sessions
- human-in-the-loop
- tracing
- handoffs

本项目以后需要逐步覆盖类似生产型能力，但以自己的实现为主。

---

## 4. 技术栈

### 4.1 Python

固定使用：

```text
Python 3.11
```

原因：

- BrowserGym 当前要求 Python > 3.10。
- AgentLab 当前支持 Python >=3.11,<3.13。
- Python 3.11 能为以后参考/接入 AgentLab 保留兼容性。

---

### 4.2 依赖管理

使用：

```text
uv
```

必须提交：

```text
pyproject.toml
uv.lock
```

不得只提交 `requirements.txt`。

---

### 4.3 Phase 0 核心依赖

建议：

```text
browsergym-core
browsergym-miniwob
gymnasium
openai
pydantic>=2
typer
rich
pyyaml
pytest
pytest-cov
```

版本要求：

- 不要无理由使用 nightly/dev build。
- 使用当前稳定版本。
- 生成并提交 `uv.lock`。
- 在 README 中记录最终解析出的实际版本。
- BrowserGym 当前稳定 meta package 为 0.14.3，可作为兼容性参考，但以安装时 resolver 的稳定版本为准。

不要在 Phase 0 添加：

```text
langgraph
langchain
deepagents
crewai
autogen
browser-use
```

作为核心依赖。

---

## 5. 系统架构

Phase 0 最终结构：

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
        │ BaselineAgent  │ │  State Store   │ │TraceRecorder │
        └───────┬────────┘ └────────────────┘ └──────────────┘
                │
                ▼
        ┌────────────────┐
        │  ModelAdapter  │
        └───────┬────────┘
                │
                ▼
        ┌────────────────┐
        │ ActionDecision │
        └───────┬────────┘
                │
                ▼
        ┌────────────────┐
        │Environment     │
        │Adapter         │
        └───────┬────────┘
                │
                ▼
           BrowserGym
                │
                ▼
           Chromium
```

Benchmark 侧：

```text
BenchmarkSpec
      │
      ▼
BenchmarkRunner
      │
      ├── Task 1 -> EpisodeRunner
      ├── Task 2 -> EpisodeRunner
      ├── Task 3 -> EpisodeRunner
      └── ...
              │
              ▼
        RunResult[]
              │
              ▼
       MetricsAggregator
              │
              ▼
         summary.json
         episodes.csv
         traces/
```

---

## 6. Phase 0 模块定义

### 6.1 `core/models.py`

使用 Pydantic 定义稳定数据模型。

至少实现：

#### TaskSpec

```python
class TaskSpec(BaseModel):
    benchmark: str
    task_id: str
    seed: int | None = None
    max_steps: int = 20
    metadata: dict = {}
```

---

#### Observation

不要直接把 BrowserGym dict 传播到整个项目。

统一转换成：

```python
class Observation(BaseModel):
    goal: str
    url: str
    axtree: str | None
    dom: str | None
    screenshot_path: str | None
    open_pages: list[str]
    last_action: str | None
    last_action_error: str | None
    elapsed_time_s: float
    raw_artifact_refs: dict[str, str] = {}
```

Phase 0：

- screenshot 不送入模型。
- 可以保存 screenshot artifact，但默认关闭以降低存储。
- AXTree 和必要 DOM 信息作为主要模型 observation。

---

#### ActionDecision

```python
class ActionDecision(BaseModel):
    action: str
    short_reason: str | None = None
```

要求：

- 不保存模型完整私有 chain-of-thought。
- `short_reason` 只允许保存简短、面向调试的决策摘要。
- 真正执行的是 `action`。

---

#### StepRecord

至少包含：

```python
run_id
step_index
timestamp
url
observation_ref
prompt_ref
model_response_ref
action
action_error
reward
terminated
truncated
latency_ms
input_tokens
output_tokens
```

以后 Verification/Retry 字段可以扩展，但 Phase 0 不启用高级逻辑。

---

#### RunResult

至少包含：

```python
run_id
task_spec
status
success
final_reward
num_steps
started_at
ended_at
duration_s
input_tokens
output_tokens
estimated_cost
trace_path
error_type
error_message
```

---

### 6.2 `models/base.py`

定义：

```python
class ModelAdapter(Protocol):
    def generate_action(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
    ) -> ModelOutput:
        ...
```

Phase 0 实现两个 adapter：

```text
MockModelAdapter
OpenAICompatibleModelAdapter
```

`MockModelAdapter` 用于 unit test。

`OpenAICompatibleModelAdapter`：

- 使用标准 SDK。
- endpoint/model/API key 从环境变量或配置读取。
- 不允许把 provider 写死在 Agent 代码中。
- 必须记录 model name。
- 若 SDK 返回 token usage，必须记录。
- cost 估算如果没有可靠价表，可以为 null，不得猜测。

---

### 6.3 `agents/baseline.py`

实现 `BaselineAgent`。

它不是完整 Harness，只负责：

```text
Observation
   ↓
PromptBuilder
   ↓
ModelAdapter
   ↓
ActionDecision
```

Baseline 规则：

1. 每轮只选择一个 browser action。
2. 不预先产生完整 plan。
3. 不自动 retry。
4. 不自动 recovery。
5. 不维护长期记忆。
6. 只使用当前 goal + 精简历史 + 当前 observation。
7. 最大 step 数由 TaskSpec 控制。

Prompt 必须包含：

- task goal
- current URL
- AXTree/DOM view
- last action
- last action error
- BrowserGym 可用 action schema
- 输出格式要求

不要在 prompt 中隐藏额外 planner。

---

### 6.4 `env/base.py`

定义：

```python
class EnvironmentAdapter(Protocol):
    def reset(self, task: TaskSpec) -> Observation:
        ...

    def step(self, action: str) -> EnvironmentStep:
        ...

    def close(self) -> None:
        ...
```

---

### 6.5 `env/browsergym_adapter.py`

职责：

```text
TaskSpec
   ↓
构造 Gym environment
   ↓
env.reset()
   ↓
BrowserGym observation
   ↓
ObservationNormalizer
   ↓
Our Observation
```

执行动作：

```text
ActionDecision.action
   ↓
env.step(action)
   ↓
reward / terminated / error
   ↓
EnvironmentStep
```

要求：

- BrowserGym 类型只能存在于 adapter 层附近。
- Harness core 不允许 import BrowserGym。
- 环境必须在异常情况下通过 `finally` 关闭。
- action error 不应立即让整个进程崩溃；先记录到 EnvironmentStep，再由 Runner 决定 termination。

---

### 6.6 `runtime/episode_runner.py`

这是 Phase 0 最重要的模块。

明确实现自己的 loop：

```python
observation = env.reset(task)

for step_idx in range(task.max_steps):

    decision = agent.decide(...)

    env_step = env.step(decision.action)

    trace.record(...)

    state.update(...)

    if env_step.terminated or env_step.truncated:
        break
```

不要调用第三方 Agent Runner。

职责：

- lifecycle
- step counter
- state
- exception normalization
- timeout/budget hook 预留
- trace
- termination
- final result

Phase 0 只实现：

```text
max_steps
environment termination
fatal exception termination
```

后续再加：

```text
retry
recovery
checkpoint
budget
human interrupt
```

---

### 6.7 `runtime/state.py`

Phase 0 使用简单内存状态：

```python
class RunState(BaseModel):
    run_id: str
    task: TaskSpec
    current_observation: Observation | None
    steps: list[StepRecord]
    status: RunStatus
```

不要使用数据库。

后续 checkpoint 会基于该 schema 演进。

---

### 6.8 `observability/trace.py`

从 Phase 0 开始就必须有 tracing。

原因：

**Trace 不是后期 UI 功能，而是 Harness 调试与评测基础设施。**

目录格式：

```text
runs/
  <run_id>/
    manifest.json
    steps.jsonl
    result.json
    artifacts/
      obs_000.txt
      prompt_000.txt
      model_000.json
      ...
```

`manifest.json` 至少包含：

```text
run_id
timestamp
git_commit
config_hash
python_version
platform
model_provider
model_name
benchmark
task_id
seed
max_steps
```

`steps.jsonl`：

每行一个 StepRecord。

要求：

- Trace 写入失败不能静默忽略。
- 每一步结束立即 flush。
- artifact 文件名必须 deterministic。
- 不把 API key 写入任何 trace。
- prompt/model response 保存行为可配置，但 Phase 0 本地 benchmark 默认开启。

---

### 6.9 `evaluation/metrics.py`

Phase 0 至少输出：

#### Task metrics

```text
success
final_reward
steps
duration_s
input_tokens
output_tokens
action_error_count
```

#### Aggregate metrics

```text
success_rate
mean_reward
mean_steps
median_steps
mean_duration_s
total_input_tokens
total_output_tokens
action_error_rate
```

后续追加：

```text
verification_pass_rate
recovery_success_rate
retry_count
checkpoint_resume_success
cost_per_success
unsafe_action_rate
```

---

### 6.10 `evaluation/benchmark_runner.py`

输入：

```yaml
benchmark: miniwob
tasks:
  - click-test
  - click-button
  ...
seeds:
  - 0
model: ...
max_steps: 20
```

执行：

```text
task × seed
```

每个 episode 使用独立 `run_id`。

输出：

```text
experiments/<experiment_id>/
  config.yaml
  manifest.json
  episodes.csv
  summary.json
  run_ids.txt
```

Phase 0 默认串行执行。

**不要提前实现 Ray 或多进程并行。**

并行属于后期优化。

---

## 7. Repository 目录结构

必须按下述结构创建，允许少量调整，但不得把所有代码堆入几个大文件：

```text
reliable-web-harness/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
│
├── configs/
│   ├── baseline.yaml
│   └── benchmarks/
│       └── miniwob_smoke.yaml
│
├── docs/
│   ├── architecture.md
│   ├── benchmark_strategy.md
│   ├── trace_schema.md
│   └── adr/
│       ├── ADR-001-own-runtime.md
│       ├── ADR-002-browsergym-environment.md
│       └── ADR-003-event-trace-first.md
│
├── src/
│   └── web_harness/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── models.py
│       │   ├── errors.py
│       │   └── ids.py
│       │
│       ├── models/
│       │   ├── base.py
│       │   ├── mock.py
│       │   └── openai_compatible.py
│       │
│       ├── agents/
│       │   ├── base.py
│       │   ├── baseline.py
│       │   └── prompt.py
│       │
│       ├── env/
│       │   ├── base.py
│       │   ├── browsergym_adapter.py
│       │   └── observation.py
│       │
│       ├── runtime/
│       │   ├── episode_runner.py
│       │   └── state.py
│       │
│       ├── observability/
│       │   └── trace.py
│       │
│       ├── evaluation/
│       │   ├── benchmark_runner.py
│       │   └── metrics.py
│       │
│       ├── config/
│       │   └── loader.py
│       │
│       └── cli.py
│
├── tests/
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_episode_runner.py
│   │   ├── test_trace.py
│   │   └── test_metrics.py
│   └── integration/
│       └── test_miniwob_smoke.py
│
├── scripts/
│   └── inspect_miniwob_registry.py
│
├── runs/
│   └── .gitkeep
│
└── experiments/
    └── .gitkeep
```

默认：

```text
runs/*
experiments/*
```

不提交大量运行 artifact。

但必须保留：

```text
reports/phase0/
```

或手工复制的精简结果，用于审批。

---

## 8. MiniWoB Phase 0 Smoke Benchmark

### 8.1 目的

这不是最终性能 benchmark。

只用于验证：

- Runtime 能完整运行。
- BrowserGym adapter 正常。
- action schema 可用。
- Trace 正常。
- benchmark runner 正常。
- metrics 正常。

### 8.2 固定任务

Phase 0 使用以下 12 个任务：

```text
click-test
click-button
enter-text
choose-list
click-checkboxes
click-link
login-user
read-table
navigate-tree
use-autocomplete
choose-date-easy
order-food
```

选择原因：

覆盖：

```text
简单点击
按钮选择
文本输入
下拉列表
checkbox
链接
多字段表单
信息读取
层级导航
动态组件
日期组件
组合式交互
```

### 8.3 Seed

Phase 0 smoke：

```text
seed = 0
```

即：

```text
12 episodes
```

不要根据这 12 个 episode 得出性能结论。

Phase 1 以后至少扩大到多个 seed。

### 8.4 Step limit

默认：

```text
max_steps = 20
```

如果某任务因 BrowserGym 官方 task termination 提前结束，正常记录。

不得为了提高成功率给不同任务手工写特殊逻辑。

---

## 9. Baseline Agent 输入策略

Phase 0 使用：

```text
Text-only
```

主要输入：

```text
Goal
Current URL
Accessibility Tree
必要 DOM 信息
Open pages
Last action
Last action error
最近有限步历史
```

不使用 screenshot 作为模型输入。

理由：

1. 先建立成本较低、可解释 baseline。
2. 后续可以独立评估 vision observation 的增益。
3. 避免 Phase 0 同时混入 VLM 差异。

Observation 必须做长度限制。

不要直接把完整巨大 DOM 无限塞入 prompt。

Phase 0 可采用最简单规则：

```text
优先 AXTree
DOM 只保留必要字段
设置字符上限
超过上限进行 deterministic truncation
```

不要在 Phase 0 使用 LLM summarizer 压缩 observation。

---

## 10. Action 策略

使用 BrowserGym 的 HighLevelActionSet 或其等价稳定 action interface。

不要直接让 Agent 生成任意 Playwright Python 代码。

原因：

- action space 更容易 trace。
- 更容易比较 experiment。
- 更容易在后期加入 guardrail。
- 更容易做 action error taxonomy。

Agent 最终必须输出结构化结果：

```json
{
  "action": "...",
  "short_reason": "..."
}
```

解析失败：

Phase 0 记为：

```text
MODEL_OUTPUT_PARSE_ERROR
```

不要自动让模型无限重试。

---

## 11. Error Taxonomy

Phase 0 即建立错误分类，不要只保存 Python exception 字符串。

至少定义：

```text
CONFIG_ERROR
MODEL_API_ERROR
MODEL_OUTPUT_PARSE_ERROR
ENVIRONMENT_INIT_ERROR
ACTION_EXECUTION_ERROR
OBSERVATION_ERROR
MAX_STEPS_EXCEEDED
TASK_TERMINATED
TASK_TRUNCATED
TRACE_WRITE_ERROR
UNKNOWN_ERROR
```

后期扩展：

```text
VERIFICATION_FAILED
RECOVERY_FAILED
BUDGET_EXCEEDED
CHECKPOINT_ERROR
POLICY_BLOCKED
HUMAN_REJECTED
```

---

## 12. 配置系统

`configs/baseline.yaml` 示例：

```yaml
model:
  provider: openai_compatible
  model: YOUR_MODEL_NAME
  base_url_env: MODEL_BASE_URL
  api_key_env: MODEL_API_KEY
  temperature: 0

agent:
  type: baseline
  max_history_steps: 4
  observation_char_limit: 30000

runtime:
  max_steps: 20

trace:
  root_dir: runs
  save_prompts: true
  save_model_responses: true
  save_screenshots: false

benchmark:
  config: configs/benchmarks/miniwob_smoke.yaml
```

原则：

- secret 只来自环境变量。
- experiment config 必须保存副本。
- 所有影响结果的重要参数都必须进入 config。
- 禁止在代码里散落模型名、step limit 等硬编码。

---

## 13. CLI

至少实现：

```bash
web-harness run --task miniwob.click-test --seed 0 --config configs/baseline.yaml
```

以及：

```bash
web-harness benchmark --config configs/benchmarks/miniwob_smoke.yaml
```

以及：

```bash
web-harness inspect-run <run_id>
```

`inspect-run` Phase 0 只需要文本摘要，不需要 Web UI。

示例：

```text
Run ID:
Task:
Status:
Reward:
Steps:
Duration:
Tokens:
Action errors:
Trace directory:
```

---

## 14. 单元测试

必须避免测试完全依赖真实 LLM。

### 14.1 Mock model

实现确定性 Mock：

```text
Observation A -> Action A
Observation B -> Action B
```

用于验证 runtime。

### 14.2 Fake environment

除 BrowserGym integration test 外，再实现最小 FakeEnvironment。

测试：

#### Episode normal termination

```text
reset
step
step
terminated
```

断言：

```text
RunResult.success/status
step count
trace
```

#### Max steps

模拟永不结束环境。

断言：

```text
MAX_STEPS_EXCEEDED
```

#### Environment error

模拟 step exception。

断言：

```text
错误被规范化
trace 中有失败 step
environment 被 close
```

#### Model parse error

断言：

```text
MODEL_OUTPUT_PARSE_ERROR
```

#### Trace durability

断言：

- manifest 存在
- steps.jsonl 行数正确
- result.json 存在
- JSON 可重新加载

---

## 15. Integration Test

必须包含至少一个真实 MiniWoB integration test：

```text
browsergym/miniwob.click-test
```

测试目的不是模型能力，而是环境接口。

可以使用 deterministic scripted test agent，而不是在线 LLM。

这样 CI 不依赖模型 API。

---

## 16. Phase 0 实施顺序

执行智能体必须按照以下顺序推进。

### Task 0 — 初始化工程

完成：

```text
Python 3.11
uv project
src layout
pytest
lint/type 基础配置
gitignore
```

验收：

```bash
uv sync
uv run pytest
```

能成功运行。

---

### Task 1 — BrowserGym 环境验证

安装：

```text
browsergym-core
browsergym-miniwob
Chromium
```

执行一个最小脚本：

```text
browsergym/miniwob.click-test
```

确认：

```text
reset 成功
observation 可读取
action 可执行
task 可结束
```

保存：

```text
reports/phase0/browsergym_probe.md
```

内容：

- Python version
- BrowserGym version
- Chromium/Playwright 是否正常
- observation keys
- 测试 task
- 是否成功

---

### Task 2 — 建立核心 Schema

完成：

```text
TaskSpec
Observation
ActionDecision
EnvironmentStep
StepRecord
RunState
RunResult
ErrorType
```

要求：

- 全部可 JSON serialize/deserialize。
- 为关键 schema 写 unit test。

---

### Task 3 — Environment abstraction

完成：

```text
EnvironmentAdapter
FakeEnvironment
BrowserGymAdapter
ObservationNormalizer
```

BrowserGym dependency 不得泄漏到：

```text
runtime/
agents/
evaluation/
```

---

### Task 4 — Model abstraction

完成：

```text
ModelAdapter
MockModelAdapter
OpenAICompatibleModelAdapter
```

实现结构化 action parsing。

---

### Task 5 — Baseline Agent

完成：

```text
BaselineAgent
PromptBuilder
history selection
observation truncation
```

不要加入高级 Harness 机制。

---

### Task 6 — Harness Runtime

实现：

```text
EpisodeRunner
RunState
termination rules
exception normalization
```

这是重点审批模块。

代码应清晰体现：

```text
observe -> decide -> act -> record -> update -> terminate
```

不要使用复杂 callback 魔法隐藏 loop。

---

### Task 7 — Trace

完成：

```text
TraceRecorder
manifest
steps.jsonl
artifacts
result.json
```

所有 episode，无论成功还是失败，都应尽可能生成 result。

---

### Task 8 — Evaluation

完成：

```text
MetricsCollector
BenchmarkRunner
ExperimentResult
CSV/JSON summary
```

先串行运行。

---

### Task 9 — MiniWoB smoke benchmark

运行固定 12 tasks × seed 0。

生成：

```text
reports/phase0/miniwob_smoke_summary.json
reports/phase0/miniwob_smoke_episodes.csv
reports/phase0/miniwob_smoke_report.md
```

报告必须列出：

| task | success | reward | steps | duration | input tokens | output tokens | action errors |
|---|---:|---:|---:|---:|---:|---:|---:|

并给 aggregate metrics。

如果模型 API 因预算或配置不可用：

- 不得伪造 benchmark。
- scripted/mock 结果只能用于工程验证。
- 明确标记 `MODEL_BASELINE_NOT_RUN`。
- 代码仍必须可以运行真实模型。

---

### Task 10 — 文档

必须完成：

```text
README.md
docs/architecture.md
docs/benchmark_strategy.md
docs/trace_schema.md
docs/adr/
```

README 至少提供：

```text
安装
环境变量
运行单 task
运行 benchmark
查看结果
测试
项目结构
Phase 0 limitations
```

---

## 17. Phase 0 Exit Criteria

只有同时满足下面条件才能提交审批。

### Architecture

- [ ] Harness Runtime 为自研 loop。
- [ ] BrowserGym 只通过 adapter 访问。
- [ ] Model provider 通过 ModelAdapter 隔离。
- [ ] Agent / Runtime / Environment / Evaluation 分层明确。
- [ ] 没有引入 LangGraph/Deep Agents 等替代自己的 runtime。

### Engineering

- [ ] Python 3.11。
- [ ] `uv.lock` 已提交。
- [ ] `uv sync` 可复现。
- [ ] unit tests 全通过。
- [ ] MiniWoB integration test 通过。
- [ ] env 在失败情况下仍能 close。
- [ ] 配置不包含 secret。

### Harness

- [ ] 有 TaskSpec。
- [ ] 有 RunState。
- [ ] 有 StepRecord。
- [ ] 有 TraceRecorder。
- [ ] 有统一 Error Taxonomy。
- [ ] 有 EpisodeRunner。
- [ ] 有 benchmark runner。
- [ ] 有 metrics aggregator。
- [ ] 能完整追踪 episode。

### Benchmark

- [ ] 12-task smoke suite 配置固定。
- [ ] 每个 task 有独立 run result。
- [ ] experiment 有 summary。
- [ ] 结果可重复读取。
- [ ] 未将 smoke benchmark 夸大为正式能力结论。

### Documentation

- [ ] 架构图存在。
- [ ] trace schema 有说明。
- [ ] benchmark strategy 有说明。
- [ ] ADR-001/002/003 完成。
- [ ] README 可让新环境完成安装和运行。

---

## 18. 提交 GitHub 供审批时必须包含的信息

执行智能体完成后，不要只说“已完成”。

提交内容必须包括：

### 18.1 Git 信息

```text
repository
branch
commit SHA
```

### 18.2 变更摘要

按模块列：

```text
runtime
environment
model
agent
trace
evaluation
tests
docs
```

### 18.3 测试结果

原样记录：

```bash
uv run pytest
```

结果摘要。

### 18.4 Smoke benchmark

提供：

```text
12 tasks
success count
mean reward
mean steps
action errors
token usage
```

### 18.5 已知问题

必须明确列出。

不允许为了通过审批隐藏失败任务或 flaky behavior。

---

## 19. 本阶段明确禁止事项

执行智能体不得自行添加以下内容：

```text
Web dashboard
React/Vue frontend
OAuth
RBAC
database
Redis
Kubernetes
microservices
multi-agent
planner
retry/recovery
checkpoint/resume
long-term memory
skills
human approval
LLM verifier
vision model
WebArena full deployment
WorkArena deployment
parallel benchmark
automatic prompt optimizer
```

除非某个最小实现是解决 Phase 0 blocker 的必要条件。

若发现 blocker：

1. 先记录问题。
2. 选择最小 workaround。
3. 不改变总体架构。
4. 在 GitHub 提交说明中标注。

---

## 20. Phase 0 之后的路线图

Phase 0 审批通过后再确定每阶段具体方案。

当前预期：

### Phase 1 — Reliable Execution

增加：

```text
Step Verifier
Action Retry
Failure Taxonomy enhancement
Recovery policy
Replanning
Budget manager
```

核心实验：

```text
Baseline
vs
+ Retry
vs
+ Verifier
vs
+ Recovery
```

---

### Phase 2 — Durable Harness

增加：

```text
Checkpoint
Resume
Persistent run state
Replay
Crash recovery
Timeout handling
```

核心问题：

> 长任务失败后，是否能够减少重新执行成本？

---

### Phase 3 — Context & Skills

增加：

```text
Context manager
History compression
Workspace
Reusable skills
Domain skills
Skill retrieval
```

参考 Browser Harness 和 Deep Agents，但保持自研模块。

---

### Phase 4 — Formal Evaluation

升级 benchmark：

```text
MiniWoB multi-seed
WebArena-Verified
WebArena-Verified Hard
```

建立正式 ablation matrix。

---

### Phase 5 — Production-like Workflow

引入：

```text
WorkArena / WorkArena++
```

或构建自己的可复现 enterprise-style workflow benchmark。

最终目标：

```text
同一模型
同一任务集合
不同 Harness 配置
→ 对成功率、step、token、latency、failure recovery、cost 的系统比较
```

---

## 21. Phase 0 最重要的原则

### 原则一：不要追求 Baseline 很强

Baseline 的任务是：

> 成为后续所有 Harness 改进的稳定参考点。

---

### 原则二：Trace 从第一天存在

没有 trace 的 Agent 系统无法进行严肃的 Harness 调试和实验分析。

---

### 原则三：Environment 和 Runtime 必须解耦

否则以后无法公平迁移到：

```text
WebArena
WorkArena
其他 Browser Env
甚至 SRE / Coding Env
```

---

### 原则四：每个高级 Harness 机制都必须可关闭

未来所有模块都应该配置化。

例如：

```yaml
planner:
  enabled: false

verification:
  enabled: false

recovery:
  enabled: false
```

这是以后 Ablation Study 的必要条件。

---

### 原则五：优先确定实验可解释性，而不是功能数量

项目成功标准不是：

> 有多少 feature。

而是：

> 是否能严格回答“某个 Harness 机制到底改善了什么”。

---

## 22. Phase 0 审批重点

提交后将重点审查：

1. `EpisodeRunner` 是否真正掌握在项目内部。
2. 数据 schema 是否能支撑后续 checkpoint/replay/verifier。
3. BrowserGym 是否被正确隔离。
4. Trace 是否完整、稳定、可机器读取。
5. Benchmark runner 是否具备实验可重复性。
6. Baseline 是否足够简单，没有偷偷混入后续能力。
7. 错误是否被结构化记录。
8. 模型与环境是否可替换。
9. 后续模块能否通过配置逐项进行消融实验。
10. 当前代码是否已经出现不必要的框架耦合。

---

# Sources

1. ServiceNow, **BrowserGym** — unified browser-agent environment and benchmark ecosystem.  
   https://github.com/ServiceNow/BrowserGym

2. ServiceNow, **AgentLab** — framework for developing, running and evaluating web agents with experiment reproducibility and traces.  
   https://github.com/ServiceNow/AgentLab

3. ServiceNow, **WebArena-Verified** — audited WebArena benchmark with deterministic and offline evaluation.  
   https://github.com/ServiceNow/webarena-verified

4. web-arena-x, **WebArena** — self-hosted realistic web environment for autonomous agents.  
   https://github.com/web-arena-x/webarena

5. ServiceNow, **WorkArena** — browser benchmark for common knowledge-work tasks and compositional workflows.  
   https://github.com/ServiceNow/WorkArena

6. Browser Use, **browser-harness** — thin CDP-based agent harness with editable workspace and reusable domain skills.  
   https://github.com/browser-use/browser-harness

7. LangChain, **Deep Agents** — batteries-included agent harness with context management, skills, persistence, checkpointing and human-in-the-loop.  
   https://github.com/langchain-ai/deepagents

8. OpenAI, **Agents SDK** — agent runtime primitives, sessions, guardrails, human-in-the-loop and tracing.  
   https://openai.github.io/openai-agents-python/

9. PyPI, **BrowserGym 0.14.3 metadata** — Python >3.10.  
   https://pypi.org/project/browsergym/

10. PyPI / ServiceNow, **AgentLab** — Python 3.11/3.12 compatibility and benchmark tooling.  
    https://pypi.org/project/agentlab/
