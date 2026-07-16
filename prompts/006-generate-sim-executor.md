---
prompt_id: 006
title: 生成模拟执行器
model_tested: Claude Sonnet 4.5
date: 2026-07-16
version: 1.0
input: {broker_config, positions}
output: {sim_executor.py, xt_executor.py, executor_base.py}
known_limitations: "模拟执行器不验证真实成交逻辑；实盘需额外验证"
---

# Prompt 006: 生成模拟执行器

## 指令

### 1. 生成 `services/executor_base.py` — 抽象基类

```python
class ExecutorBase(ABC):
    @abstractmethod
    def buy(self, code: str, volume: int, price: float) -> dict: ...
    @abstractmethod
    def sell(self, code: str, volume: int, price: float) -> dict: ...
    @abstractmethod
    def get_positions(self) -> dict: ...
```

### 2. 生成 `services/sim_executor.py` — 模拟执行器

- 仅打印买卖指令到终端: `print(f"[SIM] 买入 {code} 100股")`
- 不影响系统其他模块运行
- **必须执行 T+1 限制检查**（铁律四）
- **必须记录审计日志**（append-only）

### 3. 生成 `services/xt_executor.py` — miniQMT 实盘执行器

- 引入 `from xtquant import xttrader`
- 实现 `get_positions()` 查询持仓
- 实现 `sync_portfolio(target_holdings)`：对比目标持仓和实际持仓，计算买卖差额
- **包含 A股 T+1 限制逻辑**：当日买入不能当日卖出
- 所有实盘调用包裹 `try-except`，失败时记录审计日志

## 验证标准（模拟盘）

```bash
export AMINQT_BROKER=sim
python -c "
from services.sim_executor import SimExecutor
ex = SimExecutor()
ex.buy('600519', 100, 1680.50)
print(ex.get_positions())
ex.sell('600519', 100, 1700.00)  # 同日卖出 → 应被 T+1 拒绝
"
# 终端打印正确的买入/卖出指令清单，T+1 违规被拒绝，无报错
```

## 合规检查

- [ ] 执行器策略模式（基类 + 模拟 + 实盘）
- [ ] 模拟执行器 T+1 检查
- [ ] 模拟执行器审计日志
- [ ] 实盘执行器 T+1 检查
- [ ] 实盘执行器 try-except
- [ ] 默认 sim + manual（安全优先）
- [ ] 环境变量 `AMINQT_BROKER=sim|xt` 切换
- [ ] 参见 ADR-004 和 `skills/risk-circuit-breaker.md`
