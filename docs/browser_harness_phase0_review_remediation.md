# Reliable Web Workflow Agent Harness — Phase 0 审批整改计划

## 0. 审批结论

审查对象：

- Repository: `Ethan-Martinez-creater/Browser-harness-project`
- Branch: `main`
- Commit: `8200c4b48712f9b0ca3dd4f120a63ae49752849e`

当前结论：

**Phase 0 架构方向通过，但最终验收暂不通过。不得进入 Phase 1。**

原因不是缺少功能，而是目前存在会污染后续 Harness benchmark / ablation 可信度的基础契约问题。整改只允许修正 Phase 0 基础设施，不得提前实现 Planner、Verifier、Retry、Recovery、Checkpoint、Skills 等 Phase 1+ 能力。

## 1. 已确认通过的部分

以下设计保持不变，不要求重构：

1. 自研 `EpisodeRunner` 显式运行循环，未把 runtime 交给 LangGraph / Deep Agents / CrewAI 等第三方框架。
2. `EnvironmentAdapter`、`ModelAdapter`、Agent、Runtime、Trace、Evaluation 已形成基本分层。
3. BrowserGym 依赖集中在环境适配层附近，整体隔离方向正确。
4. `TaskSpec`、`Observation`、`StepRecord`、`RunState`、`RunResult` 等核心 schema 已建立。
5. `TraceRecorder` 从 Phase 0 即落盘，包含 manifest / steps / artifacts / result。
6. 已实现 FakeEnvironment、MockModelAdapter 和真实 MiniWoB integration test。
7. 已提交 `uv.lock`、README、Architecture、Benchmark Strategy、Trace Schema 和 ADR。
8. 已完成 12-task MiniWoB smoke run，且失败没有导致整批 benchmark 崩溃。
9. 当前 baseline 没有提前加入 planner/retry/recovery/verification 等高级能力。

这些部分只允许做为整改所必需的局部修改。

## 2. Blocker B1 — Prompt Action Contract 与 BrowserGym 实际 Action API 不一致

### 2.1 当前问题

当前 `agents/prompt.py` 手工声明：

```text
type(bid='ID', value='TEXT')
select_option(bid='ID', option='TEXT')
check_uncheck(bid='ID')
scroll(x=0, y=300)
```

但 BrowserGym 0.14 的高层 action 实际 API 使用：

```text
fill(bid, value, enable_autocomplete_menu=False)
check(bid)
uncheck(bid)
select_option(bid, options)
scroll(delta_x, delta_y)
```

这意味着当前 smoke benchmark 中部分 `ACTION_EXECUTION_ERROR` 并不是 Agent 能力失败，而是 Harness 向模型提供了错误工具契约。

因此当前 `5/12` 结果不能作为后续消融实验的正式 Phase 0 baseline。

### 2.2 整改要求

不得继续手工维护一个与真实 BrowserGym action set 无绑定的字符串常量。

实现明确的 **Action Contract**。建议增加：

```text
env/action_contract.py
```

例如：

```python
class ActionSpec(BaseModel):
    name: str
    signature: str
    description: str
    examples: list[str]
```

并由 BrowserGymAdapter 提供：

```python
def action_contract(self) -> ActionContract:
    ...
```

BaselineAgent / PromptBuilder 不再硬编码具体 BrowserGym action 签名，而从注入的 contract 生成工具说明。

Phase 0 MiniWoB 至少允许：

```text
noop
click
fill
select_option
check
uncheck
scroll
press
```

如保留 `new_tab`，则必须确认实际 BrowserGym action 名称/签名一致。

强制要求：**Prompt 中公开给 Agent 的每一个 action 都必须能够被当前 EnvironmentAdapter 接受。**

更优方案是 BrowserGymAdapter 使用明确的 `HighLevelActionSet` / action mapping，使执行环境和 prompt contract 来源一致。

不得通过增加 retry 来掩盖错误 action contract。

### 2.3 新增测试

新增：

```text
tests/integration/test_browsergym_action_contract.py
```

至少验证 click、fill、select_option、check/uncheck、scroll。

测试目的不是完成 benchmark，而是确认 Harness 声明的 action 能被 BrowserGym action parser / environment 正确接受。

## 3. Blocker B2 — Trace 中 Observation 与实际决策输入不一致

### 3.1 当前问题

当前 `EpisodeRunner` 流程是：

```text
pre_observation
    ↓
agent.decide(pre_observation)
    ↓
env.step(action)
    ↓
post_observation
    ↓
TraceRecorder.record_step(observation=post_observation)
```

但 `StepRecord.url` 保存的是 action 前 URL，而 `observation_ref` 指向 action 后 observation。

结果是 `prompt_000.txt` 对应 pre-observation，而 `obs_000.txt` 对应 post-observation。

这破坏了 Trace 的核心语义：给定某一步 trace，必须能够知道模型实际看到了什么，再解释为什么产生该 action。

### 3.2 整改要求

固定 Step Trace 语义：

```python
class StepRecord(BaseModel):
    ...
    observation_ref: str | None
    next_observation_ref: str | None
```

Artifact：

```text
artifacts/
  obs_000.txt
  prompt_000.txt
  model_000.json
  next_obs_000.txt
```

必须满足：

```text
obs_N
   ↓
prompt_N
   ↓
model_N
   ↓
action_N
   ↓
next_obs_N
```

下一轮 `obs_(N+1)` 应与上一轮 `next_obs_N` 语义对应。

### 3.3 新增测试

使用 FakeEnvironment 让 pre/post observation 的 URL 或 AXTree 不同，并断言：

1. `obs_000.txt` 包含 pre-action observation。
2. `next_obs_000.txt` 包含 post-action observation。
3. `prompt_000.txt` 与 `obs_000.txt` 对应。
4. parse error 时存在 pre observation，但不得伪造 next observation。

## 4. Required R1 — Experiment `manifest.json` 实际写成 YAML

当前 `evaluation/benchmark_runner.py` 中对 `manifest.json` 使用 `yaml.safe_dump(...)`。

整改为严格 JSON：

```python
json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
```

新增 BenchmarkRunner unit test，必须能用 `json.loads()` 读取 `manifest.json` 和 `summary.json`。

## 5. Required R2 — 移除未计入 Agent Step 的隐式 `noop()` bootstrap

BrowserGymAdapter 当前在 `env.reset()` 后自动执行 `env.step("noop()")`，但该 action：

- 不计入 Harness step；
- 不进入正常 StepRecord；
- reward/terminated/truncated 被忽略；
- 会改变 benchmark environment 状态。

优先使用 BrowserGym 原生 observation timing 能力，例如 `pre_observation_delay` 或等价配置。

最终必须满足：

> **Harness 不偷偷执行一个未进入 trace / step budget 的任务 action。**

如果 BrowserGym 0.14.3 在本地 MiniWoB 上确实无法避免 bootstrap action，则必须形成 ADR，并将其显式标记为 environment bootstrap，检查它不会终止任务或产生 reward，进入 provenance，而且其他 benchmark 默认关闭。

## 6. Required R3 — Secret Contract 实际并未被有效测试

当前 `test_no_api_key_leak` 没有真正注入 secret，只检查输出中是否出现 `"sk-"`，属于 vacuous test。

整改：

1. Harness YAML 中禁止 literal `api_key`。
2. 对 model/provider 配置中的 `api_key/token/secret/password` 做 fail-fast。
3. Experiment config / manifest 保存前使用 sanitized config。
4. 测试中注入：

```python
monkeypatch.setenv("MODEL_API_KEY", "sk-PHASE0-SENTINEL-SECRET")
```

运行一个 fake/mock episode/experiment 后，递归扫描 `runs/` 与 `experiments/` 所有文本文件，断言 sentinel 一次都不存在。

## 7. Required R4 — Action Error 必须进入结构化 Failure Taxonomy

当：

```python
env_step.action_error is not None
```

时设置：

```python
step.error_type = ErrorType.ACTION_EXECUTION_ERROR
```

这不代表 episode 必须终止。必须区分 step-level failure 与 episode-level terminal failure。

不得在本轮增加 retry。

## 8. Required R5 — Phase 0 Benchmark Report 与机器结果不一致

当前 `miniwob_smoke_summary.json` 与 `miniwob_smoke_episodes.csv` 一致，但 Markdown 报告中多个 episode duration 与机器结果不一致。

例如同一 experiment ID：

```text
click-test
CSV: 10.984s
Markdown: 14.9s
```

```text
order-food
CSV: 241.719s
Markdown: 36.3s
```

整改：

增加 `scripts/render_benchmark_report.py` 或等价函数，由 `summary.json / episodes.csv` 自动生成 Markdown 报告，禁止人工抄写 benchmark 数值。

修复后重新执行 smoke benchmark，并重新生成：

```text
reports/phase0/miniwob_smoke_summary.json
reports/phase0/miniwob_smoke_episodes.csv
reports/phase0/miniwob_smoke_report.md
```

## 9. Required R6 — BenchmarkRunner 缺少关键契约测试

新增：

```text
tests/unit/test_benchmark_runner.py
```

至少验证：

1. `task × seed` episode 数正确。
2. 每个 episode 有独立 run_id。
3. `manifest.json` 是严格 JSON。
4. `summary.json` 是严格 JSON。
5. `episodes.csv` 行数正确。
6. aggregate success/token/steps 正确。
7. `run_ids.txt` 数量正确。
8. experiment config 中没有 secret。
9. 一个 episode 失败不会让整个 experiment 丢失已完成结果。

如现有 BenchmarkRunner 强绑定 BrowserGym，只增加轻量 `environment_factory` / `runner_factory` hook，不引入大型 DI 框架。

## 10. Smoke Benchmark 必须重新运行

由于 Action Contract 会改变模型实际获得的工具说明，旧 `5/12` 不得继续作为正式 Phase 0 baseline。

修复后用同样条件重新运行：

```text
12 tasks
seed = 0
temperature = 0
max_steps = 20
same model/provider if available
```

生成新的 experiment ID。

**不要求成功率高于 5/12。** 审批关注 action contract、trace、artifact 一致性，而不是成功率。

## 11. 本轮禁止事项

整改期间不得增加：

```text
Planner
LLM Verifier
Rule Verifier
Retry
Recovery
Replanning
Checkpoint
Resume
Memory
Context compression
Skills
Human approval
Vision
Multi-agent
WebArena
WorkArena
parallel benchmark
Web UI
database
```

当前任务是修复 baseline measurement system，而不是提高 Agent 成功率。

## 12. 整改后的 Exit Criteria

### Action Contract

- [ ] Prompt action schema 与 BrowserGym 实际 API 一致。
- [ ] 不存在 `type(bid=...)` / `check_uncheck(...)` 等不存在的公开 action。
- [ ] Tool contract 有唯一事实来源或明确同步机制。
- [ ] 有 BrowserGym action-contract integration test。

### Trace

- [ ] `observation_ref` 指向模型决策前 observation。
- [ ] post-action observation 有独立引用。
- [ ] Prompt 与 pre-action observation 对应。
- [ ] action error 使用 `ACTION_EXECUTION_ERROR`。
- [ ] Trace 文档同步更新。

### Environment

- [ ] 不再隐藏未追踪的 `noop()` benchmark action；或提供经过审批的 explicit bootstrap 设计。
- [ ] reset observation 稳定可用。

### Experiment artifacts

- [ ] `manifest.json` 可由 `json.loads()` 读取。
- [ ] `summary.json` 可由 `json.loads()` 读取。
- [ ] Markdown report 由机器结果生成。
- [ ] CSV / summary / report 一致。

### Secrets

- [ ] YAML config 不允许 literal API key。
- [ ] sentinel secret leakage test 有效。
- [ ] runs/experiments 全目录扫描未发现 sentinel。

### Tests

- [ ] 原有 unit/integration test 继续通过。
- [ ] 新增 action contract test。
- [ ] 新增 trace pre/post semantic test。
- [ ] 新增 benchmark runner contract test。
- [ ] 新增 real sentinel secret test。

### Benchmark

- [ ] 修复后重新执行 12-task × seed 0。
- [ ] 新结果作为正式 Phase 0 baseline。
- [ ] 不以成功率是否提高作为通过条件。

## 13. 重新提交审批时必须提供

```text
Repository:
Branch:
Commit SHA:
pytest result:
ruff result:
MiniWoB new experiment ID:
success count:
mean reward:
mean steps:
action error rate:
total tokens:
```

并明确说明：

```text
B1 Action Contract: fixed
B2 Trace semantics: fixed
R1 manifest JSON: fixed
R2 hidden noop: fixed / documented alternative
R3 secret contract: fixed
R4 action error taxonomy: fixed
R5 report consistency: fixed
R6 benchmark runner tests: fixed
```

第二轮审批通过之前，不进入 Phase 1。
