# Browser Harness — Phase 1C Closure 二次整改

## 审批对象

- Repository: `Ethan-Martinez-creater/Browser-harness-project`
- Branch: `main`
- Reviewed commit: `9011d3aaaa19f074b8f5987aa450352afbf43f7c`

## 结论

Phase 1C 上一轮主要整改已完成，但 closure 仍有少量会影响 Phase 1D escalation 可信度的 outcome 契约问题。

**暂不进入 Phase 1D。**

## B1 — Terminal recovery 没有进入 recovery outcome accounting

`WAIT_AND_REOBSERVE` 自己可能产生真实 environment terminal。当前会增加 `recovery_count`，随后直接结束 episode，但不会增加 success/failed/unresolved，因此可能破坏：

```text
recovery_success_count
+ recovery_failed_count
+ recovery_unresolved_count
== recovery_count
```

修复语义：

- `terminated=True && reward>0` -> `recovery_success_count += 1`
- `terminated=True && reward<=0` -> `recovery_failed_count += 1`
- `truncated=True` -> `recovery_failed_count += 1`

terminal recovery 不进入 pending evaluation。

升级 C8 并新增 terminal-failure / truncation recovery case。

## B2 — Recovery environment action error 可能被重复计 outcome

非 terminal `WAIT_AND_REOBSERVE` 如果产生 `action_error`，Runner 当前会立即 `recovery_failed_count += 1`，但随后仍将同一次 recovery 加入 pending evaluation，后续可能再次计 success/failed。

同一次 recovery 必须只能进入一种 outcome path：

```text
immediate terminal outcome
OR
immediate recovery-operation failure
OR
pending local evaluation
```

三者互斥。

推荐：

```text
recovery_result.error_type != None
-> failed += 1
-> 不 append pending
```

## R1 — `wait_ms=0` 配置没有被真实执行

Config validator 允许 `wait_ms >= 0`，但 RecoveryManager 当前使用：

```python
wait_ms = directive.wait_ms or 500
```

合法的 `wait_ms: 0` 会被改成 500。

改为：

```python
wait_ms = 500 if directive.wait_ms is None else directive.wait_ms
```

并增加 `wait_ms=0 -> noop(wait_ms=0)` 测试。

## R2 — Recovery-owned truncation 的 ErrorType 错误

Recovery noop terminal path 在 `status=TRUNCATED` 时仍统一设置 `TASK_TERMINATED`。

应改为：

```text
terminated -> TASK_TERMINATED
truncated  -> TASK_TRUNCATED
```

增加 recovery noop truncation 测试。

## R3 — 对所有 Recovery 场景强制检查 outcome invariant

增加公共 helper：

```python
def assert_recovery_outcome_invariant(result):
    assert (
        result.recovery_success_count
        + result.recovery_failed_count
        + result.recovery_unresolved_count
        == result.recovery_count
    )
```

至少覆盖 C1/C2/C3/C4/C7/C8/C11/C12/C13，以及新增 recovery-action-error / recovery-truncation 场景。

## 已验收通过

以下无需再修改：

- canonical recovery budget path；
- recovery config fail-fast；
- exhausted budget 不再杀死 PASS / NO_PROGRESS；
- exhausted budget 阻止新的 recoverable recovery；
- policy abort 不再制造 phantom failure；
- pending outcome 显式 unresolved；
- recovery-start fingerprint；
- Recovery event trigger identity；
- blocked-action re-decision 独立计数；
- TASK_FAILED terminal short-circuit trace；
- C1–C13 主场景；
- Phase 0/1A/1B regression；
- CI / Ruff。

## Exit Criteria

- [ ] Terminal recovery 有且只有一个 local outcome。
- [ ] Recovery action error 不会重复计 outcome。
- [ ] 所有 recovery attempts 满足 outcome invariant。
- [ ] `wait_ms=0` 被真实执行为 0。
- [ ] recovery truncation 使用 `TASK_TRUNCATED`。
- [ ] C1–C13 保持 PASS。
- [ ] 新增 terminal-failure/truncation/action-error recovery tests PASS。
- [ ] Existing tests 不回归。
- [ ] GitHub Actions PASS。
- [ ] Ruff PASS。
- [ ] Phase 1D 仍未实现。

通过后 Phase 1C 正式 CLOSED，并进入 Phase 1D。
