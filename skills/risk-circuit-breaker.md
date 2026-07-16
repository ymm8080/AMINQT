# Skill: risk-circuit-breaker (风控熔断)

> **来源**: 从 EWM Robot 项目的 Watchdog 熔断模式 + Safety PLC 硬下限适配
> **用途**: 编写或修改 `risk_filter.py`、`executor` 层时的完整规范
> **触发**: 实现风控逻辑、执行模式切换、熔断恢复时

## 熔断架构

```
模型推理 → 候选列表 → 个股硬约束 → 账户级熔断 → T+1检查 → 下单
                            ↓
                     回撤>3% → 空列表 (停止一切)
```

## 三层风控实现

### 第一层：个股硬约束

```python
def apply_filters(candidates: List[dict],
                  account_drawdown_pct: float = 0.0) -> List[dict]:
    # 第二层先检查（短路）
    if account_drawdown_pct > settings.MAX_ACCOUNT_DRAWDOWN_PCT:
        logger.error("账户回撤 %.2f%% > 限制 %.2f%%, 停止下单",
                     account_drawdown_pct, settings.MAX_ACCOUNT_DRAWDOWN_PCT)
        return []

    filtered = []
    for c in candidates:
        # 成交额过滤
        if c.get('amount', 0) < settings.MIN_AMOUNT:
            logger.debug("剔除 %s: 成交额 %.0f < %.0f",
                         c['symbol'], c.get('amount', 0), settings.MIN_AMOUNT)
            continue
        # 涨跌幅过滤
        pct = abs(c.get('pct_change', 0))
        if pct > settings.PRICE_LIMIT_PCT:
            logger.debug("剔除 %s: 涨跌幅 %.2f%% > %.2f%%",
                         c['symbol'], pct, settings.PRICE_LIMIT_PCT)
            continue
        filtered.append(c)

    logger.info("风控过滤: %d → %d (剔除 %d)",
                len(candidates), len(filtered), len(candidates) - len(filtered))
    return filtered
```

### 第二层：账户级熔断

```python
class CircuitBreaker:
    """账户级风控熔断器 (灵感: EWM Robot Watchdog safe-mode)."""

    def __init__(self, threshold: float = 3.0):
        self._threshold = threshold
        self._tripped = False

    def check(self, drawdown_pct: float) -> bool:
        """检查是否应触发熔断。返回 True = 正常, False = 熔断。"""
        if drawdown_pct > self._threshold:
            if not self._tripped:
                logger.error("CIRCUIT BREAKER TRIPPED: 回撤 %.2f%% > %.2f%%",
                             drawdown_pct, self._threshold)
            self._tripped = True
            return False
        return not self._tripped  # 已熔断则保持，需人工恢复

    def manual_recover(self) -> bool:
        """人工恢复 (熔断后不得自动恢复 — 铁律二)."""
        if self._tripped:
            logger.info("CIRCUIT BREAKER 人工恢复")
            self._tripped = False
            return True
        return False

    @property
    def is_tripped(self) -> bool:
        return self._tripped
```

### 第三层：T+1 限制

```python
class T1Checker:
    """A股 T+1 交易限制检查 (灵感: EWM Robot 版本兼容性 N-1)."""

    def __init__(self):
        self._purchase_dates: dict[str, date] = {}

    def record_buy(self, code: str, trade_date: date):
        self._purchase_dates[code] = trade_date

    def can_sell(self, code: str, today: date) -> bool:
        buy_date = self._purchase_dates.get(code)
        if buy_date is None:
            return True  # 非当日买入
        if today <= buy_date:
            logger.warning("T+1 违规: 拒绝卖出 %s (买入日 %s, 今日 %s)",
                           code, buy_date, today)
            return False
        return True
```

## 执行模式安全

```python
# config/settings.py
class ExecutionMode(str, Enum):
    AUTO = "auto"      # 风控通过后自动下单
    MANUAL = "manual"   # 弹窗推荐，用户确认

# 默认 MANUAL (安全优先)
EXECUTION_MODE = ExecutionMode(os.getenv("AMINQT_EXEC_MODE", "manual"))
```

## 审计日志 (灵感: EWM Robot WORM 黑匣子)

每笔交易记录 append-only：

```python
import json, os
from datetime import datetime

def log_trade(code: str, direction: str, volume: int,
              price: float, mode: str, score: float, passed: bool):
    """Append-only 交易审计日志 (严禁 DELETE/UPDATE — 铁律六)."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "code": code,
        "direction": direction,
        "volume": volume,
        "price": price,
        "mode": mode,
        "score": score,
        "risk_passed": passed,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/executor.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())  # 确保写入磁盘
```

## 检查清单

- [ ] 成交额 < 5000万 → 剔除
- [ ] |涨跌幅| > 9.5% → 剔除
- [ ] 账户回撤 > 3% → 返回空列表
- [ ] 熔断后需人工恢复（不得自动解除）
- [ ] T+1 限制检查
- [ ] 审计日志 append-only
- [ ] AUTO 模式下单前经过风控
- [ ] 默认安全 (manual + sim)
