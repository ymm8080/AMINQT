# 日志标准 (Logging Standards)

> **位置**: `03_operations/monitoring/logging-standards.md`
> **最后更新**: 2026-07-16

## 日志库

所有服务使用 Python `logging` 模块，**严禁 `print`**（模拟执行器除外）。

## 日志配置

```python
# app/main.py 中的全局配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

## 日志级别

| 级别 | 用途 | 示例 |
|------|------|------|
| `INFO` | 关键步骤成功 | 数据下载成功、模型加载、选股完成 |
| `WARNING` | 异常但可继续 | 账户回撤接近阈值、单只股票下载失败 |
| `ERROR` | 需要关注 | 模型推理失败、API 请求失败、风控触发 |
| `DEBUG` | 调试信息 | 特征矩阵形状、中间计算值 |

## 关键事件日志

### 数据层
```python
logger.info("数据下载成功: %s (%d rows)", code, len(df))
logger.error("数据下载失败 %s: %s", code, exc)
logger.info("内存缓存加载完成: %d 只股票, %.1f MB", count, size_mb)
```

### 模型层
```python
logger.info("模型加载: %s", model_path)
logger.info("训练完成: epoch=%d, train_loss=%.6f, val_loss=%.6f", epoch, t_loss, v_loss)
logger.error("模型推理失败: %s", exc)
```

### 风控层
```python
logger.warning("账户回撤 %.2f%% 接近限制 %.2f%%", drawdown, limit)
logger.error("账户回撤 %.2f%% > 限制, 停止下单", drawdown)
logger.info("风控过滤: %d → %d (剔除 %d)", input_count, output_count, removed)
```

### 执行层
```python
logger.info("[SIM] 买入 %s %d股 @ %.2f", code, volume, price)
logger.info("[SIM] 卖出 %s %d股 @ %.2f", code, volume, price)
logger.warning("T+1 违规: 拒绝卖出 %s (当日买入)", code)
logger.info("实盘下单: %s %s %d股 @ %.2f", code, direction, volume, price)
```

### 调度层
```python
logger.info("调度器启动 (每日 14:50 Asia/Shanghai)")
logger.info("定时选股触发")
logger.error("调度器初始化失败: %s", exc)
```

## 日志文件

### 日志轮转

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "logs/aminqt.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=3,               # 保留3个轮转文件
)
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s"
))
```

### 日志位置

| 服务 | 日志位置 | 关键事件 |
|------|----------|----------|
| FastAPI | `logs/aminqt.log` | API 请求、选股、调度 |
| 训练脚本 | `logs/train.log` | 训练进度、模型保存 |
| 数据下载 | `logs/download.log` | 下载成功/失败 |
| 执行器 | `logs/executor.log` | 交易指令、T+1 检查 |

## 验证

```bash
# 检查日志大小
ls -la logs/

# 查看最近的错误
grep "ERROR" logs/aminqt.log | tail -20

# 检查风控触发
grep "回撤" logs/aminqt.log

# 检查交易记录
grep "买入\|卖出" logs/executor.log
```
