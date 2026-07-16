# 030 — 风控硬约束规则 (Risk Filter Hard Constraints)

> **触发条件**: 编写或修改 `risk_filter.py`、`executor` 层、任何涉及下单决策的代码时

## 三层风控过滤

### 第一层：个股硬约束

| 约束 | 阈值 | 动作 | 配置项 |
|------|------|------|--------|
| 成交额 | < 5000万 | 剔除 | `MIN_AMOUNT = 5_000_000` |
| 涨跌幅 | \|x\| > 9.5% | 剔除 | `PRICE_LIMIT_PCT = 9.5` |

### 第二层：账户级熔断

| 约束 | 阈值 | 动作 | 配置项 |
|------|------|------|--------|
| 账户回撤 | > 3% | 返回空列表，停止下单 | `MAX_ACCOUNT_DRAWDOWN_PCT = 3.0` |

- 熔断后需**人工恢复**，系统不得自动解除
- 恢复接口: `POST /api/v1/recover` (需确认)

### 第三层：T+1 限制

- 当日买入的股票不能在当日卖出
- `executor` 维护 `purchase_date` 字段
- 违反 T+1 的卖单被拒绝并记录审计日志

## 执行模式

```python
class ExecutionMode(str, Enum):
    AUTO = "auto"      # 风控通过后自动下单
    MANUAL = "manual"  # 弹窗推荐，用户确认后执行
```

- AUTO 模式：模型推理 → 风控过滤 → 直接下单
- MANUAL 模式：模型推理 → 风控过滤 → 弹窗推荐 → 用户确认 → 下单
- LLM 不参与下单决策（铁律三）

## 模拟执行器

无法连接券商时，必须使用 `SimExecutor`：

```python
# services/sim_executor.py
class SimExecutor(ExecutorBase):
    def buy(self, code: str, volume: int, price: float) -> dict:
        print(f"[SIM] 买入 {code} {volume}股 @ {price}")
        return {"status": "simulated", "code": code, "volume": volume, "price": price}

    def sell(self, code: str, volume: int, price: float) -> dict:
        # T+1 检查
        if self._is_t0_position(code):
            logger.warning("T+1 violation: cannot sell %s bought today", code)
            return {"status": "rejected", "reason": "T+1"}
        print(f"[SIM] 卖出 {code} {volume}股 @ {price}")
        return {"status": "simulated", "code": code, "volume": volume, "price": price}
```

## 审计日志

每笔交易记录（append-only）：

```json
{
  "timestamp": "2026-07-16T14:50:00+08:00",
  "code": "600519",
  "direction": "buy",
  "volume": 100,
  "price": 1680.50,
  "mode": "auto",
  "score": 0.85,
  "risk_passed": true
}
```
