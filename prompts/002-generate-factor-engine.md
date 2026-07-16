---
prompt_id: 002
title: 生成因子引擎模块
model_tested: Claude Sonnet 4.5
date: 2026-07-16
version: 1.0
input: {df, window_days, horizon_days}
output: {factor_engine.py, X_matrix, y_labels}
known_limitations: "特征数须≥25；未来函数防护必须严格遵守"
---

# Prompt 002: 生成因子引擎模块

## 指令

生成 `app/core/factor_engine.py`，实现 `build_features(df)`：

1. **技术指标**（必须包含）：
   - MACD (DIF, DEA, BAR)
   - KDJ (K, D, J)
   - BOLL (上轨, 中轨, 下轨)
   - RSI

2. **衍生关系特征**（重点）：
   - `(close - DIF) / DIF` — 偏离度（使用 `safe_divide`，除零防护）
   - `close / MA5 - 1` — 乖离率
   - 过去5日指标的线性斜率

3. **滑动窗口**：
   - 输出 X 形状 `(样本数, 20天, ≥25特征)`
   - y 为未来5日收益率

4. **防护规则**（铁律一）：
   - 严禁使用未来函数（`shift(-k)` 禁止）
   - 只用 `rolling/expanding/ewm` 计算指标
   - `np.nan_to_num(X)` 输入模型前必须执行

## 验证标准

```python
X, y = build_features(df)
assert X.shape[1] == 20  # 时间窗口
assert X.shape[2] >= 25   # 特征数
assert not np.any(np.isnan(X))  # 无 NaN
```

## 合规检查

- [ ] 所有指标使用 `rolling/expanding/ewm`（无未来函数）
- [ ] 除法使用 `safe_divide()`（除零防护）
- [ ] `np.nan_to_num(X)` 执行
- [ ] X 形状 `(N, 20, ≥25)`
- [ ] y 为未来5日收益率
- [ ] 文件头 `# -*- coding: utf-8 -*-`
- [ ] 函数有 docstring（Google 风格）
- [ ] 使用 `logging`
- [ ] 参见 `skills/quant-factor-engineering.md` 完整规范
