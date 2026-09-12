# Reliable Web Workflow Agent Harness — Phase 1A 审批整改计划

## 0. 审批对象与结论

- Repository: `Ethan-Martinez-creater/Browser-harness-project`
- Branch: `main`
- Reviewed commit: `aaf7e8d71d5b591f6745f1586c16977272aa3e11`
- Frozen Phase 0 baseline: `63d928e358fe31e790b0b06a5c7b801598808b15`

当前结论：

**Phase 1A 的核心架构方向通过，但最终验收暂不通过。不得进入 Phase 1B。**

已经通过且无需重做：FailureSignal/VerificationResult、deterministic fingerprint 基础实现、五类 detector、DefaultStepVerifier、`events.jsonl`、shadow verifier、ReliabilityState、reliability metrics、Phase 0 compatibility regression、GitHub Actions unit/regression/ruff CI、12-task MiniWoB smoke。

本轮只修 Phase 1A 基础契约，不得实现 Retry / Recovery / Replanning。

## 1. Blocker B1 — Experiment provenance 未持久化 reliability 配置

当前 `configs/phase1/verifier_shadow.yaml` 已包含 reliability 配置，但 `evaluation/benchmark_runner.py` 的 persisted `resolved` config 没有 `reliability` 字段。因此当前 `summary.json` / `config.yaml` 不能证明 experiment 使用了 verifier shadow、detector 开关和 loop threshold。

同时 experiment-level `config_hash` 基于缺少 reliability 的 `resolved`，而 run-level manifest 使用 `cfg.hash`，两者语义不一致。

### 整改

建立唯一 canonical resolved experiment config，必须包含：

```text
benchmark
tasks
seeds
model
agent
runtime
trace
environment
reliability
```

以下全部使用同一个 canonical config 及 hash：

```text
experiments/<id>/config.yaml
experiments/<id>/manifest.json
experiments/<id>/summary.json
runs/<run_id>/manifest.json
```

### 测试

- persisted config / summary 都包含 reliability；
- experiment manifest config hash == run manifest config hash；
- 修改 `verification.enabled` 或 `loop.consecutive_threshold` 后 hash 必须变化。

## 2. Blocker B2 — verification.enabled / mode 当前不具有真实语义

BenchmarkRunner 当前只判断 `cfg.reliability_enabled`，因此：

```yaml
reliability:
  enabled: true
  verification:
    enabled: false
```

仍会创建 verifier。

此外 HarnessConfig 允许 `mode: active`，但 EpisodeRunner 仍固定按 shadow 运行并在 event 中写死 `"mode": "shadow"`。

### 整改

Phase 1A 只支持：

```text
mode: shadow
```

`mode: active` 在 Phase 1A 必须 fail-fast。

Verifier 创建条件：

```text
reliability.enabled == true
AND verification.enabled == true
AND verification.mode == shadow
```

新增 accessor：

```python
cfg.verification_enabled
cfg.verification_mode
```

测试：

```text
reliability=false -> verifier off
reliability=true + verification=false -> verifier off
reliability=true + verification=true + shadow -> verifier on
active -> ConfigError
unknown mode -> ConfigError
```

并增加 config-level BenchmarkRunner test，不能只手工注入 verifier。

## 3. Required R1 — FailureSignal.signature 要能识别“同一种失败”

当前 NO_PROGRESS signature 是常量：

```text
no_progress:state_unchanged_reward_zero
```

会把不同页面上的 no-progress 合并成同一个 failure signature。

当前 ACTION_ERROR signature 只有 normalized error，没有 action identity，不同 browser action 的相同错误也会被合并。

### 整改

NO_PROGRESS：

```text
no_progress:<state_fingerprint>
```

ACTION_ERROR：

```text
action_error:<action_type>:<normalized_error>
```

只提取稳定 action function name，不把动态 bid/value/raw action 全部写入 signature。

LOOP 当前 `loop:<transition_signature>` 保持不变。

测试：

- 同 state no-progress signature 相同；
- 不同 state 不同；
- 同 action type + 等价 normalized error 相同；
- 不同 action type + 相同 error text 不同。

## 4. Required R2 — Observation fingerprint 保留 tab/order 语义

当前 fingerprint 对 `open_pages` 使用排序，因此 `[A,B]` 与 `[B,A]` 被认为相同。但 Browser action contract 存在 tab-index 操作，tab 顺序属于可执行环境状态。

### 整改

`open_pages` 按环境原始顺序参与 fingerprint，不排序。

新增测试：

```text
open_pages=[A,B]
open_pages=[B,A]
```

fingerprint 必须不同。

## 5. Required R3 — Phase 1A report 必须由 committed renderer 原样重建

当前 `scripts/render_benchmark_report.py` 仍固定输出 Phase 0 标题和 Phase 0 scope warning，也没有渲染 reliability metrics；但 committed `phase1a_verifier_report.md` 已经是 Phase 1A 标题并包含 verification/failure 数字。

因此当前报告不能由仓库现有 renderer 原样重建。

### 整改

推荐让 renderer phase-aware，例如：

```bash
render_benchmark_report.py --profile phase1a
```

从 summary/config 直接渲染：

```text
Phase
verification mode
verification_count
failure_signal_count
verification_signal_rate
episodes_with_failure_signal
failure_kind_counts
```

也可以增加薄的 `render_phase1a_report.py`，但不要复制大段通用逻辑。

强制要求：

```text
machine result -> committed renderer -> committed Markdown
```

重新生成 Phase 1A report，并增加 renderer test。

## 6. 建议但非阻断 — verification_signal_rate 命名

当前计算是：

```text
failure_signal_count / verification_count
```

严格来说是 `signals per verification`，未来单次 verification 出现多个 signal 时可能大于 1。

建议改为：

```text
mean_failure_signals_per_verification
```

或者真正统计：

```text
verifications_with_signal / verification_count
```

并保留 `verification_signal_rate`。

## 7. 本轮禁止事项

不得新增：

```text
DecisionExecutor
RetryPolicy
Model API retry
parse retry
RecoveryManager
FailurePolicyEngine
Replanner
Phase 1B/1C fault injection
```

## 8. 整改后 Exit Criteria

- [ ] persisted experiment config 包含 reliability；
- [ ] experiment/run 使用同一 canonical config hash；
- [ ] reliability 参数变化会改变 hash；
- [ ] `verification.enabled=false` 真正关闭 verifier；
- [ ] Phase 1A 只允许 shadow；
- [ ] event mode 与真实模式一致；
- [ ] NO_PROGRESS signature 区分 state；
- [ ] ACTION_ERROR signature 区分 action type；
- [ ] open_pages 顺序变化会改变 fingerprint；
- [ ] Phase 1A report 可由 committed renderer 重建；
- [ ] report 不再出现 Phase 0 scope 文案；
- [ ] GitHub Actions unit/regression/ruff 继续通过；
- [ ] Phase 0 compatibility regression 继续通过；
- [ ] Retry/Recovery/Replanning 仍未实现。

## 9. 是否需要重新跑在线 MiniWoB

如果整改只影响 provenance、config wiring、signature、fingerprint 和 renderer，则**不强制重新调用在线模型跑 12-task smoke**。

但必须：

1. 用现有 machine result 重新生成 Phase 1A report；
2. 用 FakeEnvironment/MockModel 做 config/verifier/fingerprint 回归；
3. 若任何改动改变 MiniWoB control flow、Prompt、environment action 或 model call，则必须重新跑 smoke。

## 10. 重新提交审批时提供

```text
Repository:
Branch:
Commit SHA:

GitHub Actions unit:
local/browser integration:
ruff:

B1 provenance: fixed
B2 config semantics: fixed
R1 failure signatures: fixed
R2 tab-order fingerprint: fixed
R3 machine-rebuildable report: fixed
```

并附：

```text
canonical config hash test
verification enabled/disabled config tests
signature distinction tests
fingerprint tab-order test
report renderer test
```

通过 closure review 后，才进入 **Phase 1B — Controlled Retry**。
