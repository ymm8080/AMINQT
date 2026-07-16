# 020 — 数据格式映射规则 (Data Format Mapping)

> **触发条件**: 编写或修改 `data_loader.py`、数据适配器、任何读取 CSV 的代码时

## akshare 列名映射

akshare 下载的数据列名是**中文**，读取后必须**立即重命名**为英文：

| akshare 原始列名 | 重命名后 | 说明 |
|:---|:---|:---|
| `日期` | `date` | 交易日期 |
| `开盘` | `open` | 开盘价 |
| `收盘` | `close` | 收盘价 |
| `最高` | `high` | 最高价 |
| `最低` | `low` | 最低价 |
| `成交量` | `volume` | 成交量（股） |
| `成交额` | `amount` | 成交额（元） |

## 合规写法

```python
# ✅ 读取后立即重命名
COL_MAP = {
    '日期': 'date', '开盘': 'open', '收盘': 'close',
    '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount',
}
df = ak.stock_zh_a_hist(symbol, period='daily', adjust='qfq')
df = df.rename(columns=COL_MAP)
```

## 违规写法

```python
# ❌ 直接用中文列名
df['收盘'].rolling(5).mean()  # 后续所有代码都要用中文

# ❌ 不重命名，在 factor_engine 里映射
# 会导致 factor_engine 耦合数据源格式
```

## 数据适配器接口

```python
# data/adapters/base.py
class BaseAdapter(ABC):
    @abstractmethod
    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """返回标准列名: date, open, close, high, low, volume, amount"""
        ...

# data/adapters/akshare_adapter.py
class AkshareAdapter(BaseAdapter):
    def fetch_daily(self, symbol, start, end) -> pd.DataFrame:
        df = ak.stock_zh_a_hist(...)
        return df.rename(columns=COL_MAP)  # 适配器内部完成映射

# data/adapters/ifind_adapter.py
class IfindAdapter(BaseAdapter):
    def fetch_daily(self, symbol, start, end) -> pd.DataFrame:
        # iFinD 列名可能不同，适配器内部统一映射
        ...
```

## 性能规则

- **禁止逐个文件读取选股**: FastAPI 启动时全量加载 `Dict[str, pd.DataFrame]`
- 内存缓存: `DATA_CACHE: dict[str, pd.DataFrame]` 在 `app/main.py` startup 时填充
- 选股接口从内存读取，响应时间 < 5 秒
