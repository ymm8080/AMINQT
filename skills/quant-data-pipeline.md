# Skill: quant-data-pipeline (量化数据管道)

> **用途**: 编写或修改 `data_loader.py`、`download_data.py`、数据更新逻辑时的完整规范
> **触发**: 实现数据下载、CSV读写、列名重命名、增量更新、数据完整性校验时

## 列名映射（铁律）

akshare 下载的数据列名是**中文**，读取后必须**立即重命名**为英文，后续所有代码仅使用英文列名。

```python
COLUMN_MAP = {
    '日期': 'date',
    '开盘': 'open',
    '收盘': 'close',
    '最高': 'high',
    '最低': 'low',
    '成交量': 'volume',
    '成交额': 'amount',
    '振幅': 'amplitude',
    '涨跌幅': 'pct_change',
    '涨跌额': 'change',
    '换手率': 'turnover',
}
```

```python
def load_csv(path: str) -> pd.DataFrame:
    """读取CSV并重命名列名为英文标准列名。"""
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAP)
    assert 'date' in df.columns, f"列名映射失败: {path} 缺少 date 列"
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df
```

## 数据下载

```python
import akshare as ak
import time
import os
import logging

logger = logging.getLogger(__name__)

STOCK_LIST = ['000001', '000002', '600519', '000858', '600036']

def download_stock(code: str, start: str = '20180101', end: str = '20241231') -> str:
    """下载单只股票历史日K数据，保存至 data/raw/{code}.csv。
    
    Args:
        code: 股票代码 (如 '600519')
        start: 开始日期 YYYYMMDD
        end: 结束日期 YYYYMMDD
    
    Returns:
        保存的文件路径
    """
    try:
        df = ak.stock_zh_a_hist(symbol=code, period='daily',
                                start_date=start, end_date=end, adjust='qfq')
        if df.empty:
            logger.warning("下载 %s 返回空数据", code)
            return ""
        
        os.makedirs('data/raw', exist_ok=True)
        path = os.path.join('data/raw', f'{code}.csv')
        df.to_csv(path, index=False, encoding='utf-8-sig')
        logger.info("下载 %s: %d 行 → %s", code, len(df), path)
        return path
    except Exception as e:
        logger.error("下载 %s 失败: %s", code, e)
        return ""

def download_all(stock_list: list = None):
    """批量下载股票数据，含反爬延时。"""
    stock_list = stock_list or STOCK_LIST
    for code in stock_list:
        download_stock(code)
        time.sleep(0.5)  # 防止反爬
```

## 增量更新

```python
def incremental_update(code: str) -> str:
    """增量更新: 仅下载CSV中最后日期之后的新数据。"""
    path = os.path.join('data/raw', f'{code}.csv')
    
    if os.path.exists(path):
        df_old = load_csv(path)
        last_date = df_old['date'].max()
        start = (last_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
    else:
        df_old = None
        start = '20180101'
    
    end = datetime.now().strftime('%Y%m%d')
    if start >= end:
        logger.info("%s 数据已是最新 (%s)", code, last_date)
        return path
    
    df_new = ak.stock_zh_a_hist(symbol=code, period='daily',
                                start_date=start, end_date=end, adjust='qfq')
    if df_new.empty:
        logger.info("%s 无新数据", code)
        return path
    
    df_new = df_new.rename(columns=COLUMN_MAP)
    df_new['date'] = pd.to_datetime(df_new['date'])
    
    if df_old is not None:
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
    else:
        df_all = df_new.sort_values('date').reset_index(drop=True)
    
    df_all.to_csv(path, index=False, encoding='utf-8-sig')
    logger.info("增量更新 %s: +%d 行 (总计 %d)", code, len(df_new), len(df_all))
    return path
```

## 数据完整性校验

```python
def validate_data(df: pd.DataFrame, code: str = "") -> bool:
    """校验数据完整性，返回 True 表示通过。"""
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            logger.error("校验失败 %s: 缺少列 %s", code, col)
            return False
    
    if len(df) < 100:
        logger.error("校验失败 %s: 行数 %d < 100", code, len(df))
        return False
    
    if df['close'].isna().any():
        logger.error("校验失败 %s: close 有空值", code)
        return False
    
    # 检查日期连续性 (允许周末/节假日缺口)
    date_diff = df['date'].diff().dt.days
    abnormal_gaps = date_diff[date_diff > 10]  # 超过10天的缺口
    if len(abnormal_gaps) > 0:
        logger.warning("校验警告 %s: 发现 %d 处异常缺口 (>10天)", code, len(abnormal_gaps))
    
    # 检查价格合理性
    if (df['close'] <= 0).any():
        logger.error("校验失败 %s: 存在非正价格", code)
        return False
    
    logger.info("校验通过 %s: %d 行, 日期范围 %s ~ %s",
                code, len(df), df['date'].min().date(), df['date'].max().date())
    return True
```

## 内存缓存（全市场遍历性能优化）

```python
class DataCache:
    """启动时一次性加载所有CSV至内存，选股时从内存读取。"""
    
    _instance = None
    _cache: dict = {}  # {code: DataFrame}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_all(self, data_dir: str = 'data/raw'):
        """加载目录下所有CSV到内存。"""
        import glob
        files = glob.glob(os.path.join(data_dir, '*.csv'))
        for f in files:
            code = os.path.basename(f).replace('.csv', '')
            try:
                self._cache[code] = load_csv(f)
            except Exception as e:
                logger.error("加载 %s 失败: %s", f, e)
        logger.info("内存缓存加载完成: %d 只股票", len(self._cache))
    
    def get(self, code: str) -> pd.DataFrame:
        return self._cache.get(code)
    
    def get_all(self) -> dict:
        return self._cache.copy()
```

## 检查清单

- [ ] akshare 中文列名已重命名为英文标准列名
- [ ] `date` 列已转为 `datetime` 类型
- [ ] 数据按日期升序排列
- [ ] 下载间隔 `time.sleep(0.5)` 防止反爬
- [ ] 增量更新仅获取最新日期之后的数据
- [ ] 数据完整性校验通过 (列存在、行数≥100、无空值、价格>0)
- [ ] 全市场遍历使用内存缓存 (非逐个读取CSV)
- [ ] 所有路径使用 `os.path.join()` 跨平台兼容
- [ ] 所有函数包含 `try-except` 错误捕获
- [ ] 关键步骤使用 `logging` 记录日志
