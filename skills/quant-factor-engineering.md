# Skill: quant-factor-engineering (量化因子工程)

> **用途**: 编写或修改 `factor_engine.py` 时的完整检查清单
> **触发**: 实现技术指标计算、特征工程、滑动窗口逻辑时

## 必须包含的技术指标

### MACD
```python
df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
df['dif'] = df['ema12'] - df['ema26']
df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
df['bar'] = 2 * (df['dif'] - df['dea'])
```

### KDJ
```python
low_9 = df['low'].rolling(window=9).min()
high_9 = df['high'].rolling(window=9).max()
rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
df['k'] = rsv.ewm(com=2, adjust=False).mean()
df['d'] = df['k'].ewm(com=2, adjust=False).mean()
df['j'] = 3 * df['k'] - 2 * df['d']
```

### BOLL
```python
df['boll_mid'] = df['close'].rolling(window=20).mean()
df['boll_std'] = df['close'].rolling(window=20).std()
df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
```

### RSI
```python
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))
```

## 必须包含的衍生特征

```python
# 偏离度 (除零防护)
df['close_dif_dev'] = safe_divide(df['close'] - df['dif'], df['dif'])

# 乖离率
df['bias_ma5'] = df['close'] / df['close'].rolling(5).mean() - 1

# 5日线性斜率
from scipy.stats import linregress
def rolling_slope(series, window=5):
    def _slope(x):
        y = np.arange(len(x))
        return linregress(y, x).slope
    return series.rolling(window).apply(_slope, raw=True)

df['macd_slope'] = rolling_slope(df['dif'])
df['kdj_slope'] = rolling_slope(df['k'])
```

## 滑动窗口构建

```python
def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build (N, 20, F) feature matrix and (N,) future return labels."""
    feature_cols = [
        'open', 'close', 'high', 'low', 'volume', 'amount',
        'dif', 'dea', 'bar',                    # MACD
        'k', 'd', 'j',                          # KDJ
        'boll_upper', 'boll_mid', 'boll_lower', # BOLL
        'rsi',                                   # RSI
        'close_dif_dev', 'bias_ma5',            # 衍生
        'macd_slope', 'kdj_slope',              # 斜率
        # ... 更多特征 (≥ 25 个)
    ]

    X, y = [], []
    for i in range(WINDOW_DAYS, len(df) - HORIZON_DAYS):
        X.append(df[feature_cols].iloc[i-WINDOW_DAYS:i].values)
        # 标签: 未来5日收益率
        future_ret = (df['close'].iloc[i + HORIZON_DAYS] / df['close'].iloc[i]) - 1
        y.append(future_ret)

    X = np.array(X)  # (N, 20, F)
    y = np.array(y)  # (N,)

    # NaN 处理
    X = np.nan_to_num(X)

    return X, y
```

## 验证标准

```python
X, y = build_features(df)
assert X.ndim == 3, f"X 应为3维, got {X.ndim}"
assert X.shape[1] == WINDOW_DAYS == 20, f"时间窗口应为20, got {X.shape[1]}"
assert X.shape[2] >= 25, f"特征数应≥25, got {X.shape[2]}"
assert len(X) == len(y), "X 和 y 长度不一致"
assert not np.any(np.isnan(X)), "X 中有 NaN"
print(f"OK: X.shape={X.shape}, y.shape={y.shape}")
```

## 检查清单

- [ ] 所有指标使用 `rolling/expanding/ewm`（无未来函数）
- [ ] 除法使用 `safe_divide()`（除零防护）
- [ ] 特征矩阵执行 `np.nan_to_num(X)`
- [ ] X 形状 `(N, 20, ≥25)`
- [ ] y 为未来5日收益率
- [ ] 标签和特征严格分离
- [ ] 单元测试通过
