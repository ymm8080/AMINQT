# 010 — 未来函数防护规则 (Future Function Prevention)

> **触发条件**: 编写或修改 `factor_engine.py`、`intraday_learner.py`、任何技术指标计算代码时

## 核心原则

**计算第 t 天的特征/指标时，只能使用第 t 天及之前的数据。**

## 合规写法

```python
# ✅ rolling — 只看过去 N 天
df['ma5'] = df['close'].rolling(window=5).mean()

# ✅ expanding — 累积到当前
df['cum_return'] = df['close'].pct_change().expanding().sum()

# ✅ shift(positive) — 取过去数据
df['prev_close'] = df['close'].shift(1)  # 前一日收盘

# ✅ 计算第 t 天的 MACD，只用 t 及之前
df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
df['dif'] = df['ema12'] - df['ema26']
df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
df['bar'] = 2 * (df['dif'] - df['dea'])
```

## 违规写法

```python
# ❌ shift(-k) — 引入未来数据
df['future_close'] = df['close'].shift(-5)  # 未来5日收盘

# ❌ 用全量数据 fit 后再切片 — 信息泄漏
scaler = StandardScaler()
X_all = scaler.fit_transform(df[features])  # 用了全量统计量
X_train = X_all[:train_size]  # 泄漏了测试集信息

# ✅ 正确做法：只在训练集 fit
scaler = StandardScaler()
X_train = scaler.fit_transform(df[features][:train_size])
X_test = scaler.transform(df[features][train_size:])
```

## 验证方法

```python
# 验证：修改第 t 天的输入数据，第 t+k 天的指标不应变化
df_orig = df.copy()
df.loc[10, 'close'] = 9999  # 修改第10天收盘价
# 检查第 1-9 天的指标是否变化
assert (features[:10] == features_orig[:10]).all()
```

## 常见陷阱

| 陷阱 | 说明 | 防护 |
|------|------|------|
| 全量归一化 | 用全量数据 `fit_transform` 后切片 | 只在训练集 `fit`，测试集 `transform` |
| 未来标签泄漏 | `y = df['close'].shift(-5)` 本身没错，但不得进入特征 | 标签和特征严格分离 |
| 衍生指标未来 | `(close - DIF) / DIF` 中的 DIF 用了未来数据 | 确保 DIF 是 rolling/expanding 计算 |
| 回测泄漏 | 用全量数据选股后再回测 | 选股逻辑在回测循环内部逐日执行 |
