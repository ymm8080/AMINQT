# 120 — 交易幂等性规则 (Idempotency Patterns)

> **来源**: EWM Robot 项目 `idempotency-patterns.mdc`
> **适配**: 交易重复下单防护、调度器重复执行防护
> **触发**: 编写 executor 下单逻辑、APScheduler 定时任务、API POST 接口时

## 核心原则

**同一交易指令执行一次和多次，结果必须相同。**

> [量化] 重复下单 = 重复花钱。这是资金安全问题（铁律二）。

## 场景 1：重复下单防护

同一选股结果不应导致多次下单。

### 实现：唯一订单 ID

```python
import hashlib
from datetime import datetime

def generate_order_id(code: str, date: str, direction: str) -> str:
    """生成唯一订单 ID（同一代码+同一日+同一方向 = 同一 ID）."""
    raw = f"{code}_{date}_{direction}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]

class OrderDeduplicator:
    """防止重复下单（灵感: EWM Robot idempotency Redis 缓存）."""

    def __init__(self):
        self._processed: set[str] = set()

    def should_process(self, order_id: str) -> bool:
        if order_id in self._processed:
            logger.warning("重复订单拦截: %s", order_id)
            return False
        self._processed.add(order_id)
        return True
```

### 使用

```python
dedup = OrderDeduplicator()

def execute_buy(code: str, volume: int, price: float):
    order_id = generate_order_id(code, today_str(), "buy")
    if not dedup.should_process(order_id):
        return {"status": "skipped", "reason": "duplicate"}
    # 执行下单...
    log_trade(code, "buy", volume, price, ...)
```

## 场景 2：调度器重复执行

APScheduler 可能重复触发（网络分区后补偿执行）。

### 实现：执行锁

```python
from datetime import datetime, timedelta

_last_select_time: datetime | None = None
SELECT_COOLDOWN_MIN = 5  # 5 分钟内不重复执行

def scheduled_select():
    global _last_select_time
    now = datetime.now()
    if _last_select_time and (now - _last_select_time) < timedelta(minutes=SELECT_COOLDOWN_MIN):
        logger.info("调度跳过: 上次执行 %s 分钟内", SELECT_COOLDOWN_MIN)
        return
    _last_select_time = now
    # 执行选股...
```

## 场景 3：API 重复提交

用户快速双击"买入"按钮。

### 实现：幂等键

```python
from fastapi import Header

@app.post("/api/v1/trade/buy")
def buy(code: str, volume: int,
        x_idempotency_key: str = Header(...)):
    """下单接口要求 Idempotency-Key 头."""
    if not dedup.should_process(x_idempotency_key):
        return {"status": "skipped", "reason": "duplicate"}
    # 执行下单...
```

## 检查清单

- [ ] 同一代码+同一日+同一方向的买单只执行一次
- [ ] 调度器 5 分钟内不重复执行选股
- [ ] API 下单接口要求 Idempotency-Key
- [ ] 重复拦截记录审计日志
- [ ] T+1 检查在幂等检查之后执行
- [ ] 熔断状态下不接受新订单（铁律二）
