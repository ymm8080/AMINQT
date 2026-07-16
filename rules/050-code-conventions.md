# 050 — 代码规范 (Code Conventions)

> **触发条件**: 编写或修改任何 Python 文件时

## 文件头

所有文件头部必须包含：
```python
# -*- coding: utf-8 -*-
```

## Docstring

所有函数必须包含 Google 风格 docstring：

```python
def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build windowed feature matrix X and future-return label y.

    Args:
        df: Canonicalized daily DataFrame (date, open, close, high,
            low, volume, amount).

    Returns:
        (X, y): X shape (N, WINDOW_DAYS, F>=25), y shape (N,).

    Raises:
        ValueError: If df has fewer than WINDOW_DAYS rows.
    """
```

## 日志

必须使用 `import logging`，**严禁 `print`**（模拟执行器除外）：

```python
import logging
logger = logging.getLogger(__name__)

# 关键步骤
logger.info("数据下载成功: %s (%d rows)", code, len(df))
logger.info("模型加载完成: %s", model_path)

# 报错
logger.error("选股失败: %s", exc)
logger.warning("账户回撤 %.2f%% > 限制", drawdown_pct)
```

## 错误捕获

所有关键代码块必须包含 `try-except`：

```python
try:
    df = adapter.fetch_daily(code, start, end)
except Exception as exc:
    logger.error("数据下载失败 %s: %s", code, exc)
    continue  # 跳过此股票，不中断全市场遍历
```

## 日期处理

- 使用 `datetime` 对象，严禁字符串直接比较
- 所有时间戳存储为 UTC+8 (Asia/Shanghai)
- 使用 `datetime.date` / `datetime.datetime` 类型

```python
# ✅ 正确
from datetime import date
if trade_date > date(2024, 12, 31):
    ...

# ❌ 错误
if "2024-12-31" > "2024-01-01":  # 字符串比较不可靠
    ...
```

## 路径处理

- 使用 `os.path.join()` 或 `pathlib.Path`，确保跨平台兼容
- 配置项统一在 `config/settings.py` 中定义

```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
```

## 反爬延时

akshare 数据下载必须包含延时：
```python
import time
time.sleep(0.5)  # 防止反爬
```

配置项：`DOWNLOAD_SLEEP_SEC = 0.5`

## Linting

- Python: `ruff` 零错误
- 提交前检查: `ruff check app/ scripts/ services/ tests/`
