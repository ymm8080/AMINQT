# 110 — 运维限流与部署防护 (Operational Limits)

> **来源**: EWM Robot 项目 `090-operational-limits.mdc` (alwaysApply: true)
> **适配**: 删除了不适用的 Node-RED/Docker/SQLite WAL 内容，保留告警冷却、部署防护、资源监控

## 1. 内存缓存监控

全市场 CSV 加载到内存 `Dict[str, pd.DataFrame]` (ADR-003)，需要监控：

| 指标 | 阈值 | 动作 |
|------|------|------|
| 缓存内存占用 | > 500MB | 日志 WARNING，建议缩减股票池 |
| 缓存内存占用 | > 1GB | 日志 ERROR，选股可能变慢 |
| 缓存股票数 | < 5 | 日志 WARNING，股票池太小 |
| 选股响应时间 | > 5 秒 | 日志 ERROR，检查缓存是否失效 |

### 监控代码
```python
import sys

def log_cache_stats(cache: dict):
    total_size = sum(sys.getsizeof(df) for df in cache.values())
    logger.info("内存缓存: %d 只股票, %.1f MB", len(cache), total_size / 1024 / 1024)
    if total_size > 500 * 1024 * 1024:
        logger.warning("缓存内存占用 %.1f MB > 500MB", total_size / 1024 / 1024)
```

## 2. 日志轮转

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "logs/aminqt.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=3,               # 保留 3 个轮转文件
)
```

| 日志文件 | 最大大小 | 保留文件数 |
|----------|----------|-----------|
| `logs/aminqt.log` | 10MB | 3 |
| `logs/executor.log` | 10MB | 3 |
| `logs/train.log` | 10MB | 3 |

## 3. 部署防护

- **部署前检查 Git 工作区是否干净**:
  ```bash
  git status --porcelain
  # 有输出 → 拒绝部署，先提交
  ```

- **模型权重变更前必须备份**:
  ```bash
  cp app/models/trained/lstm_best.pth app/models/trained/backup/
  ```

- **配置变更前必须备份**:
  ```bash
  cp config/settings.py config/settings.py.bak
  ```

## 4. 告警冷却（与规则 100 配合）

- 同一告警原因 **30 分钟内最多输出 1 次到终端**
- 冷却期内日志仍记录（DEBUG 级别）
- 实现:
  ```python
  from datetime import datetime, timedelta

  _alert_history: dict[str, datetime] = {}
  ALERT_COOLDOWN_MIN = 30

  def should_alert(key: str) -> bool:
      now = datetime.now()
      last = _alert_history.get(key)
      if last and (now - last) < timedelta(minutes=ALERT_COOLDOWN_MIN):
          return False
      _alert_history[key] = now
      return True
  ```

## 5. 数据下载限流

| 指标 | 阈值 | 动作 |
|------|------|------|
| akshare 请求间隔 | ≥ 0.5 秒 | `time.sleep(0.5)` (铁律，防反爬) |
| 单只股票下载失败 | 连续 3 次 | 跳过该股票，日志 WARNING |
| 全量下载失败率 | > 20% | 日志 ERROR，检查网络/数据源 |
