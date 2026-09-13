# Browser Harness — Phase 2A1 Trace Replay 审批整改计划

## 0. 审批对象与结论

- Repository: `Ethan-Martinez-creater/Browser-harness-project`
- Branch: `main`
- Reviewed commit: `4cdfd4b2bbb62ae41099e6cd19edf10e70d5f534`
- Parent / Phase 1 Final Ablation commit: `6266d47acf532571a45492c0a5516ef3b6bfde27`

当前结论：

**Phase 2A1 核心架构通过，但 closure 暂不通过。不得进入 Phase 2A2。**

已确认通过、无需重做：

- `TRACE_SCHEMA_VERSION = 2` 已建立；
- legacy trace 缺少 version 时按 v1 处理；
- `obs_NNN.json` / `next_obs_NNN.json` 与原 `.txt` artifact 并存；
- `StepRecord` 只增加 additive JSON refs；
- `TraceBundleLoader` / `TraceReplayEngine` / `ReplayReport` 已建立；
- Structural Replay 已覆盖 step continuity、run id、artifact existence、主要 metric 与 recovery/replan outcome invariants；
- Semantic Replay 只调用 deterministic `StepVerifier`；
- Replay 路径没有 ModelAdapter / EnvironmentAdapter / BrowserGym 调用；
- `web-harness replay` CLI 已建立；
- 原 trace read-only test 已建立；
- Phase 0–1 regression 未被删除；
- GitHub Actions：225 unit/regression PASS；
- Ruff：PASS；
- Checkpoint / Resume 尚未实现，符合阶段边界。

本轮只修 Replay 的 provenance / fail-closed / event-integrity 契约，不得开始 Phase 2A2。

## 1. Blocker B1 — Semantic Replay 没有重建 live run 的真实 Verifier 配置

live benchmark 构造 `DefaultStepVerifier` 时，配置来自：

```text
reliability.verification.no_progress.enabled
reliability.verification.loop.enabled
reliability.verification.loop.consecutive_threshold
```

但 `TraceReplayEngine` 当前直接使用：

```python
DefaultStepVerifier()
```

即默认配置。run manifest 当前只有 `config_hash`，不足以离线重建 verifier 的真实行为。

因此 custom threshold / detector switch 会产生伪 mismatch。

### 整改要求

Trace schema v2 持久化 **effective verifier specification**，推荐：

```json
{
  "verification_spec": {
    "implementation": "DefaultStepVerifier",
    "detect_no_progress": true,
    "detect_loop": true,
    "loop_consecutive_threshold": 2
  }
}
```

保存 resolved effective behavior，不保存整份 YAML 或 secret。

Replay 默认根据 manifest 中的 spec 重建 verifier。

如果 v2 trace 有 verification events 但缺少 verifier spec，不得静默 fallback 到 defaults 并声称 semantic replay 可信。

### 必须测试

- live `detect_no_progress=false`；
- live `detect_loop=false`；
- live `loop_consecutive_threshold=3`；
- v2 verified trace 缺少 spec。

## 2. Blocker B2 — Corrupt / unsupported schema 仍可能直接抛异常

`manifest_schema_version()` 当前直接 `int(value)`，loader 未捕获非法类型/值。

例如：

```json
{"trace_schema_version":"broken"}
```

可能直接 traceback。

未来版本如：

```text
999
```

也不应按 `>=2` 自动进入 semantic replay。

### 整改要求

显式支持：

```text
supported trace schema versions = {1,2}
```

规则：

```text
missing version -> legacy v1
invalid type/value -> structured load error
unsupported future version -> structured error
manifest 顶层非 dict -> structured error
```

JSONL loader 对 read/decode/parse 错误也必须 fail closed 到 `load_errors`。

不得让坏 trace 逃出为未处理异常。

## 3. Blocker B3 — Semantic artifact 错误目前可能被 CLI 当作成功

当前顺序：

```text
structural replay
→ 计算 structural_valid
→ semantic replay
```

structured observation 文件存在但内容损坏时，semantic replay 会往 `report.errors` 追加错误，但 `structural_valid` 不再更新。

CLI 当前只根据：

```text
structural_valid
verification_mismatches
```

决定 exit code。

因此存在：

```text
semantic replay error
+
CLI exit 0
```

的 fail-open 风险。

### 整改要求

ReplayReport 明确区分：

```python
semantic_valid: bool | None
overall_valid: bool
```

建议语义：

```text
legacy / --no-semantic -> semantic_valid=None
semantic replay完整成功 -> True
semantic artifact错误或 mismatch -> False
```

CLI：

```text
overall_valid == false
→ exit 1
```

任何真实 replay error 不得 exit 0。

## 4. Required R1 — Verification event completeness 必须进入 Structural Replay

当前用 dict 按 `step_index` 收集 verification event，会导致：

- verified StepRecord 对应 event 缺失时静默跳过；
- 同一 step 两个 verification events 时后者覆盖前者；
- result verification_count 只和 StepRecord summary 比较，没有验证 authoritative event stream。

### 整改要求

每个：

```text
StepRecord.verification_status != None
```

必须恰好对应一个 verification event。

并至少验证：

```text
event.outcome == StepRecord.verification_status
```

建议继续检查 event failure kinds 与 `StepRecord.failure_kinds` 一致。

必须检测：

```text
missing verification event
duplicate verification event
orphan verification event
summary/event status mismatch
```

并使 replay invalid。

## 5. Required R2 — 比较 deterministic fingerprint 输出

live verification event 已保存：

```text
pre_fingerprint
post_fingerprint
state_changed
```

Semantic Replay 可以确定性重算。

建议比较：

```text
status
signal identity
pre_fingerprint
post_fingerprint
state_changed
```

以提高 trace provenance 校验强度。

## 6. Required R3 — 增加 CLI contract tests

至少覆盖：

```text
valid v2 -> exit 0
legacy v1 -> exit 0 + semantic skipped
structural corruption -> exit 1
semantic mismatch -> exit 1
semantic artifact parse error -> exit 1
```

## 7. 本轮禁止事项

仍禁止实现：

```text
CheckpointManager
CheckpointEnvelope
EnvironmentOperationJournal
ResumeManager
resume CLI
Interrupted Run
Phase 2A2 / 2A3
```

## 8. Phase 2A1 Closure Exit Criteria

- [ ] Semantic Replay 使用 recorded effective verifier spec；
- [ ] custom verifier settings 可忠实 replay；
- [ ] malformed schema 无未处理异常；
- [ ] unsupported schema fail closed；
- [ ] corrupt structured observation 不能 CLI exit 0；
- [ ] ReplayReport 区分 structural / semantic / overall validity；
- [ ] verified StepRecord 与 verification event 一一对应；
- [ ] duplicate/missing/orphan verification events 可检测；
- [ ] event 与 StepRecord summary 一致；
- [ ] CLI contract tests PASS；
- [ ] existing replay tests PASS；
- [ ] Phase 0–1 regression PASS；
- [ ] GitHub Actions PASS；
- [ ] Ruff PASS；
- [ ] Replay model calls = 0；
- [ ] Replay environment actions = 0；
- [ ] Checkpoint = NOT IMPLEMENTED；
- [ ] Resume = NOT IMPLEMENTED。

通过 closure review 后，才进入 **Phase 2A2 — Durable Checkpoint**。
