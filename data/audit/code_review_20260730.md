# -*- coding: utf-8 -*-
# Code Review Log — 20260730
# Branch: dkpolishment
# Reviewer: CatPaw (AI Agent)

---

## 变更范围

| 文件 | 类型 | 变更量 |
|------|------|--------|
| `app/pipeline1/feature_engine_v35.py` | 修改 | +686/-188 行 |
| `app/pipeline1/train_runner.py` | 修改 | +123/-20 行 |
| `app/pipeline1/ic_screener.py` | 修改 | +50/-22 行 |
| `app/pipeline1/daily_pipeline.py` | 修改 | +19/-4 行 |
| `app/pipeline1/feature_registry.py` | 新增 | 520 行 |
| `app/pipeline1/feature_selector.py` | 新增 | 438 行 |
| `config/feature_selector.json` | 新增 | 33 行 |

变更核心: 引入三层特征选择体系 (FeatureRegistry -> FeatureSelector -> BruteForce/GateD) 替代原 ICScreener, 增加 dim 级门控、推理列裁剪、自动采纳新列。

---

## 🔴 CRITICAL

### [C1] 未来函数违规 — `shift(-1)` 被静态扫描命中

- **文件**: `app/pipeline1/feature_engine_v35.py`
- **行号**: 357
- **违规代码**:
  ```python
  df_sorted[label_col] = df_sorted.groupby("symbol")["close"].transform(
      lambda s: s.shift(-1) / s - 1
  )
  ```
- **违反铁律**: #1 (未来函数绝对禁止 — 严禁 `shift(-k)`)
- **leakage_audit 结果**: FAIL
  ```
  未来函数嫌疑: feature_engine_v35.py:357 [\.shift\(\s*-\d] lambda s: s.shift(-1) / s - 1
  ```
- **语义说明**: 此 `_fwd_ret_1d_ic` 是临时标签 (forward return), 用于 IC 预筛后立即 drop, 不作为模型特征输入。功能上合法 (与 `label_engine.py` 的标签前瞻同理)。
- **形式违规**: 项目惯例是标签前瞻用 numpy 切片 (见 `label_engine.py:34`, `ic_decay.py:20`), 而非 `shift(-k)`。审计工具的豁免仅覆盖 `label_engine.py`, 不覆盖 `feature_engine_v35.py`。
- **修复建议**: 改用 numpy 切片模式:
  ```python
  vals = s.values
  out = np.full(len(vals), np.nan)
  out[:-1] = vals[1:] / vals[:-1] - 1
  ```

---

## 🟠 HIGH

### [H1] BruteForce 特征双重生成 → 特征爆炸

- **文件**: `app/pipeline1/train_runner.py`
- **行号**: 104-116
- **问题**: `select_features()` 流程中 BruteForce 特征被生成两次:
  1. 第一次 (L104-113): 创建 `BruteForceGenerator`, 对原始 df 生成 `_brute_` 特征, `df.join(new_feats)` 增强面板
  2. 第二次 (L116): 调用 `selector.select(df, board)` -> `_run_bruteforce_dedup()` 内部再次 `BruteForceGenerator()._eligible(df)` + `generate(df)`
- **后果**: 第二次调用时 df 已包含 `_brute_` 列。`_eligible()` 的过滤条件是 `not c.startswith("dim")` and `not c.startswith("label_")` — `_brute_` 列不满足任一排除条件, 会被当作 raw 列再次生成二阶 BruteForce 特征 (`close_brute_pct5_brute_pct5` 等)。特征数指数级膨胀, 训练时间和内存暴增。
- **修复建议**: `select_features` 传入已生成的 generator 给 `selector.select(df, board, generator=gen)`, 或让 `_run_bruteforce_dedup` 排除已有 `_brute_` 列。

---

## 🟡 MEDIUM

### [M1] 缺少 `# -*- coding: utf-8 -*-` 文件头

- **文件**: `app/pipeline1/feature_registry.py`, `app/pipeline1/feature_selector.py`
- **违反**: AGENTS.md Post-Creation Checklist 要求文件头 `# -*- coding: utf-8 -*-`

### [M2] 7 个 ruff lint 错误

| 错误 | 文件 | 行号 | 说明 |
|------|------|------|------|
| E402 | `feature_engine_v35.py` | 20 | `from .cleaning_pipeline` 在 `logging` 之后 |
| E731 | `feature_engine_v35.py` | 132 | `_ok = lambda ...` 应改为 `def` |
| E731 | `feature_engine_v35.py` | 133 | `_ok_raw = lambda ...` 同上 |
| F401 | `feature_registry.py` | 23 | `import numpy as np` 未使用 |
| F841 | `feature_registry.py` | 238 | `factors_list` 赋值后未使用 |
| E401 | `feature_selector.py` | 7 | `import json, os, re, time, logging` 多 import 同行 |
| F401 | `feature_selector.py` | 9 | `from typing import Optional` 未使用 |

- **违反**: AGENTS.md 要求 "Zero linting errors on commit"

### [M3] 缺少 try-except 错误捕获

- **文件**: `app/pipeline1/feature_selector.py`
- **违反**: AGENTS.md 要求"所有代码必须包含 try-except"
- **缺失函数列表**:
  - `BruteForceGenerator.generate()`
  - `dedup_l2()`
  - `gate_d_ablation()`
  - `nan_filter()`
  - `FeatureSelector.select()` / `_run_bruteforce_dedup()` / `_run_gate_d()`
  - 版本管理方法 (`save_version`, `load_current`, `load_version`, `list_versions`, `get_status`, `rollback`)

### [M4] `fillna(0)` 违反铁律 #9

- **文件**: `app/pipeline1/feature_selector.py`
- **行号**: 208, 234
- **违规代码**:
  ```python
  full.fit(tr[avail].fillna(0), tr[label_col])
  m.fit(tr[top].fillna(0), tr[label_col])
  ```
- **说明**: 铁律 #9 要求 `np.nan_to_num(X)`。且 LightGBM 原生支持 NaN, 填 0 反而丢失缺失信号。

---

## 🔵 LOW

### [L1] `config/feature_selector.json` dim_gating 完全无效

config 中 `dim_gating` 的键名与实际 dim 方法名不匹配:

| config 键名 | 实际方法名 |
|---|---|
| `dim02_turnover` | `dim02_volatility` |
| `dim03_amplitude` | `dim03_fundamentals` |
| `dim04_volume_price` | `dim04_sector_effect` |
| `dim05_ma_trend` | `dim05_turnover_liquidity` |
| `dim06_momentum` | `dim06_valuation_size` |
| `dim07_volatility` | `dim07_limit_gene` |
| `dim08_fundamental` | `dim08_calendar_month` |

且此配置从未被代码读取 — dim 门控由 `FeatureRegistry.has_dim_group()` 驱动, 不读 config。死代码, 维护隐患。

### [L2] `_quick_ic_check` 性能问题

- **文件**: `app/pipeline1/feature_engine_v35.py`
- 每次调用都重新 `df.groupby("date")`。在 `_auto_adopt_new_columns` 中对每个 adoptable 列调用一次 (可能 50+ 列 x 500+ 交易日)。应预计算一次 groupby 并复用。

### [L3] registry 内部状态外部直接修改

- **文件**: `app/pipeline1/ic_screener.py`
- **行号**: 236
  ```python
  registry.get_all()[f]["trial_windows"] = trial_windows
  ```
- 绕过 registry mutation API, 直接修改内部 dict。应增加 `registry.increment_trial_window(f)` 方法。

### [L4] 函数内 import

- **文件**: `app/pipeline1/feature_engine_v35.py`
- **行号**: 452
  ```python
  from datetime import datetime  # noqa: F811
  ```
- 应在模块级导入。`# noqa: F811` 说明已知重复定义问题, 但正确解法是移到模块顶部。

### [L5] 模块级 `import lightgbm`

- **文件**: `app/pipeline1/feature_selector.py`
- **行号**: 15
- 顶层 `import lightgbm as lgb`。即使不使用 `gate_d` pipeline 也会在 import 时失败 (若 lightgbm 未安装)。应改为延迟导入。

### [L6] 推理路径不传 registry

- **文件**: `app/pipeline1/daily_pipeline.py`
- 推理时调用 `self.features.build(main_df, ..., inference_cols=main_cols)` 但不传 `registry`。推理时所有 dim 全部执行 (无 dim 门控), 只有 `_chgN`/`_xrank` 列被裁剪。推理优化不完整, 但功能正确。

---

## ✅ 测试结果

```
61 passed, 87 warnings in 18.29s
```

- 测试覆盖: `test_feature_registry.py` (15 tests) + `test_feature_selector.py` (46 tests)
- 87 个 numpy RuntimeWarning (测试数据太小导致 `std` 自由度 <= 0, 非生产问题)
- **未覆盖**: `feature_engine_v35.py` 的 `_auto_adopt_new_columns` / `_quick_ic_check` 路径无单元测试

---

## 汇总

| 严重级 | 数量 | 最严重项 |
|--------|------|----------|
| CRITICAL | 1 | `shift(-1)` 触发 leakage_audit FAIL |
| HIGH | 1 | BruteForce 双重生成导致特征爆炸 |
| MEDIUM | 4 | 7 个 lint 错误 + 缺 try-except |
| LOW | 6 | dim_gating 死配置 + 性能/代码异味 |

**建议优先修复顺序**: C1 -> H1 -> M2 (lint) -> M1 (header) -> M3 (try-except) -> M4 -> L1-L6
