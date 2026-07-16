# 交易应急手册 (Trading Emergency Procedures)

> **位置**: `03_operations/runbooks/trading-emergency-procedures.md`
> **最后更新**: 2026-07-16

## 应急分级

| 级别 | 触发条件 | 响应时间 | 动作 |
|------|----------|----------|------|
| **FATAL** | 账户回撤 > 3% | 立即 | 停止所有下单，人工接管 |
| **CRITICAL** | 模型推理异常 | 5 分钟 | 切换模拟执行器 |
| **WARNING** | 数据下载失败 | 15 分钟 | 使用缓存数据，告警 |
| **INFO** | 单只股票异常 | 下次重试 | 跳过该股票 |

---

## FATAL — 账户回撤熔断

### 触发
```python
# risk_filter.py 中
if account_drawdown_pct > settings.MAX_ACCOUNT_DRAWDOWN_PCT:  # 3%
    logger.error("账户回撤 %.2f%% > 限制, 停止下单", drawdown_pct)
    return []  # 返回空列表，停止一切下单
```

### 恢复步骤
1. **确认回撤原因**: 检查交易日志
   ```bash
   grep "回撤\|drawdown" logs/aminqt.log | tail -20
   ```
2. **人工评估**: 确认是否为真实亏损还是数据错误
3. **人工恢复**: 调用恢复接口（需确认）
   ```bash
   curl -X POST http://127.0.0.1:8000/api/v1/recover -H "Content-Type: application/json" -d '{"confirm": true}'
   ```
4. **记录审计**: 恢复操作记录到审计日志

---

## CRITICAL — 模型推理异常

### 触发
- 模型权重加载失败
- 输入数据格式错误
- 推理结果异常（NaN, inf）

### 步骤
```bash
# 1. 立即切换到模拟执行器
export AMINQT_BROKER=sim
export AMINQT_EXEC_MODE=manual

# 2. 检查模型权重
ls -la app/models/trained/

# 3. 检查模型加载
python -c "
from app.models.lstm_model import LSTMModel
import torch
try:
    m = LSTMModel()
    m.load_state_dict(torch.load('app/models/trained/lstm_best.pth'))
    print('OK')
except Exception as e:
    print(f'FAIL: {e}')
"

# 4. 检查输入数据
python -c "
import pandas as pd
df = pd.read_csv('data/raw/600519.csv')
print(df.dtypes)
print(df.isna().sum())
"

# 5. 如需重新训练
python scripts/train_model.py
```

---

## WARNING — 数据下载失败

### 触发
- akshare 接口异常
- 网络超时
- 反爬封锁

### 步骤
```bash
# 1. 检查 akshare 连接
python -c "
import akshare as ak
try:
    df = ak.stock_zh_a_hist('600519', period='daily', adjust='qfq')
    print(f'OK: {df.shape}')
except Exception as e:
    print(f'FAIL: {e}')
"

# 2. 检查磁盘空间
df -h .                    # Linux
Get-PSDrive .              # Windows

# 3. 使用缓存数据
# 内存缓存仍可用（启动时加载），不影响选股
# 但数据不更新，需尽快修复

# 4. 重试下载（增加延时）
python -c "
import akshare as ak, time
for code in ['000001','000002','600519','000858','600036']:
    try:
        df = ak.stock_zh_a_hist(code, period='daily', adjust='qfq')
        df.to_csv(f'data/raw/{code}.csv', index=False)
        print(f'{code}: OK')
    except Exception as e:
        print(f'{code}: FAIL - {e}')
    time.sleep(1.0)  # 加倍延时
"
```

---

## 审计日志查询

```bash
# 查看所有交易记录
grep "买入\|卖出" logs/executor.log

# 查看拒绝的订单
grep "拒绝\|rejected" logs/executor.log

# 查看 T+1 违规
grep "T+1" logs/executor.log

# 查看风控触发
grep "回撤\|drawdown" logs/aminqt.log

# 查看调度执行
grep "选股\|select" logs/aminqt.log
```

---

## 紧急联系人

| 角色 | 职责 |
|------|------|
| 系统管理员 | 服务恢复、数据修复 |
| 交易员 | 人工交易决策、回撤评估 |
| 模型开发者 | 模型重训练、特征修复 |
