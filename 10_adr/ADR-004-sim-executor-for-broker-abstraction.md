# ADR-004: 模拟执行器作为券商接口的降级方案

**状态**: Accepted  
**日期**: 2026-07-16

## 背景 (Context)

Phase 6 需要对接 miniQMT (xtquant) 进行实盘交易。但：
- 实盘接口需要券商账号和授权
- 开发/测试阶段无法连接实盘
- miniQMT 需要特定的客户端环境
- 直接调用 xtquant 失败会导致整个系统不可用

## 决策 (Decision)

使用**执行器策略模式**，提供模拟执行器作为降级方案：

- `services/executor_base.py` — 抽象基类
- `services/sim_executor.py` — 模拟执行器（打印指令到终端）
- `services/xt_executor.py` — miniQMT 实盘执行器
- 环境变量 `AMINQT_BROKER=sim|xt` 控制切换
- 默认 `sim`，安全优先

模拟执行器要求：
- 仅打印买卖指令到终端: `print(f"[SIM] 买入 {code} 100股")`
- 不影响系统其他模块运行
- 必须执行 T+1 限制检查
- 必须记录审计日志（与实盘格式一致）

## 备选方案 (Alternatives Considered)

1. **直接调用 xtquant**: 无券商环境会报错，系统不可用 — 放弃
2. **try-except 降级**: 每次 import xtquant 失败才降级 — 不够清晰 — 放弃
3. **模拟执行器 (选中)**: 策略模式，环境变量切换 — 清晰安全

## 影响 (Consequences)

### 正面影响
- 开发阶段无需券商即可测试完整流程
- 实盘切换只需改环境变量，零代码改动
- 模拟执行器保证 T+1 和审计日志一致性

### 负面影响
- 模拟执行器不验证真实交易逻辑（成交价、滑点、部分成交）
- 切换到实盘时需额外验证

### 缓解措施
- `SimExecutor` 的接口与 `XtExecutor` 完全一致（继承同一基类）
- 实盘前必须通过模拟执行器的完整流程测试
- `XtExecutor` 中所有实盘调用包裹 `try-except`，失败时记录审计日志

## 合规要求 (Compliance)

- 基类: `services/executor_base.py`
- 模拟: `services/sim_executor.py`
- 实盘: `services/xt_executor.py`
- 环境变量: `AMINQT_BROKER` (默认 `sim`)
- 执行模式: `AMINQT_EXEC_MODE` (默认 `manual`)
- T+1 检查: 所有执行器必须实现
- 审计日志: 所有执行器必须记录

## 参考 (References)

- PROMPT_CONTENT §3 miniQMT 占位逻辑
- IMPLEMENTATION PLAN Phase 6
- rules/030-risk-filter-hard-constraints.md §模拟执行器
- 灵感来源: EWM Robot 项目的 ADR-005 Watchdog 降级模式
