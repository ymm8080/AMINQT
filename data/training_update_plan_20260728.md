# pctChg 训练/推理链路分析与改造计划

## 1. 现状: pctChg 已经流入模型 (且这是有问题的)

### 数据流追踪

```
pull_bs_extra_fields.py
  └─ 从 V3 panel raw close 计算 pctChg = (close/close.shift(1)-1)*100
     └─ 写入 data/bs_pctChg.parquet

merge_extra_fields_to_v3.py
  └─ left join pctChg → data/panel_full_enriched_v3.parquet (按 symbol+date)

panel_builder.py::assemble_panel()
  └─ 不产生 pctChg, 但缓存路径 panel_*.parquet 由调用方决定
  └─ train_runner.py / predict_runner.py / daily_pipeline.py 均直接加载
      data/panel_full_enriched_v3.parquet (含 pctChg 列)

feature_engine_v35.py::build()
  └─ dim01..dim30 全都不识别 pctChg (无专门处理)
  └─ _add_time_series_changes() → pctChg 是 float64 列, 不在 skip_exact 中
                                   → 自动生成 !!!
     ├─ pctChg_chg1, pctChg_chg3, pctChg_chg5, pctChg_chg10, pctChg_chg20
     └─ pctChg_pct_chg1, pctChg_pct_chg3, pctChg_pct_chg5, pctChg_pct_chg10, pctChg_pct_chg20

  └─ feature_columns() → pctChg 不在 id_cols 排除集, 故入选候选特征列
  └─ _add_cross_sectional_ranks() → 仅双创开启, pctChg 会被排名 (有用!)

train_runner.py::select_features()
  └─ ICScreener 逐因子评估 pctChg / pctChg_chgN / pctChg_pct_chgN
  └─ 有正向 IC → 保留; 无 → 丢弃 (但保留本身已有噪声)

predictor.py::predict()
  └─ bundle["feature_cols"] 包含上次训练筛选出的 pctChg_* 列
  └─ np.nan_to_num() 处理 NaN → 推理
```

### 问题分析

| 列 | 含义 | 问题严重度 |
|------|------|-----------|
| `pctChg` | 当日收益率 (raw close) ×100 | **高价值信号**, 但和 `ROC_1d` (close_hfq 计算) 近重复 |
| `pctChg_chg1` | 收益率的一阶差分 = 二阶导 | **噪声**: 日收益率本身 SNR 低, 再差分信噪比更差 |
| `pctChg_chg3/5/10/20` | 多周期收益率差分 | 部分含动量信息, 但 `ROC_Nd` 已提供更稳定的版本 |
| `pctChg_pct_chgN` | 收益率的百分比变化 | **高度噪声**: 对接近 0 的 pctChg 做除法 → 极端值 |

**结论**: pctChg 当前盲流过自动差分, 产生 10 个噪声衍生列. 这些列会:
- 膨胀特征空间 (候选列 +10, 每列需 IC 计算 → 耗时增加)
- 引入高噪声特征 (即使 IC 筛选可能丢弃, 但计算浪费 + L2 追踪需持久化)
- 与 dim01 的 ROC_Nd/hfq_c 特征冗余

---

## 2. 需要修改的文件清单

### 2.1 `feature_engine_v35.py` — 核心修改

#### (A) `_add_time_series_changes()` skip_exact 加入 `pctChg`

位置: 约第 120-125 行

```python
skip_exact = {"symbol", "date", "board", "industry", "month",
              "is_pre_holiday", "is_post_holiday", "list_days",
              "announce_date", "touched_limit_up", "is_virtual",
              "price_1455", "adv20", "limit_pct", "time",
              "PE_TTM", "score_rank", "rank_amount", "rank_ff_turnover",
              "liquidity_score",
              "pctChg"}  # ← 加入: raw return 不应被二次差分
```

目的: 阻止 `pctChg_chgN` / `pctChg_pct_chgN` 自动生成.

#### (B) `feature_columns()` id_cols 加入 `pctChg` (可选)

如果决定用 `ROC_1d` 代替 `pctChg` (两者高度相关但 close_hfq 更经济相关), 则把 `pctChg` 加入 id_cols:

```python
id_cols = {..., "pctChg"}
```

否则保留 `pctChg` 作为候选特征 (它和 `ROC_1d` 不完全等价: raw close vs hfq-close).

**建议**: 保留 `pctChg` 作为特征, 仅阻止其被差分. pctChg 反映实际交易收益率, ROC_1d 反映复权收益率, 两者差异含分红/送股信息, 且数据源不同提供了交叉验证.

#### (C) 新增 dim31: 从 pctChg 计算显式特征 (推荐)

在 dim30 之后新增一个维度, 专门从 `pctChg` 计算有意义的特征:

```python
def dim31_pctChg_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """从 pctChg(日收益率) 计算显式特征, 而非盲差分."""
    def per_stock(g: pd.DataFrame) -> pd.DataFrame:
        p = g["pctChg"]
        # 滚动均值/波动率 (原始收益率信号)
        g["pctChg_ma5"] = p.rolling(5).mean()
        g["pctChg_ma10"] = p.rolling(10).mean()
        g["pctChg_ma20"] = p.rolling(20).mean()
        g["pctChg_std5"] = p.rolling(5).std()
        g["pctChg_std10"] = p.rolling(10).std()
        g["pctChg_std20"] = p.rolling(20).std()
        # 连续上涨/下跌天数 (连胜/连败)
        g["pctChg_pos_streak"] = (p > 0).astype(int).groupby((p <= 0).cumsum()).cumsum()
        g["pctChg_neg_streak"] = (p < 0).astype(int).groupby((p >= 0).cumsum()).cumsum()
        # 极端收益率标记 (涨停/跌停日识别)
        g["pctChg_is_extreme"] = (p.abs() > 9.5).astype(float)  # 接近涨跌停
        return g
    return _apply_per_stock(df, per_stock)
```

然后在 `build()` 中调用:
```python
df = self.dim31_pctChg_features(df)
```

---

### 2.2 `panel_builder.py` — 无需修改

`pctChg` 已经通过 `merge_extra_fields_to_v3.py` 合并到 V3 面板. `panel_builder.py` 是数据装配器, 只关心自身产生的元数据列. 新增列自动透传 (panel 是 open schema, 不校验列白名单).

`assemble_panel()` 输出的面板经过 `enrich_panel()` 和 `enrich_alt_data()`, 然后 `to_parquet` 缓存. 但训练脚本直接加载 `data/panel_full_enriched_v3.parquet` 而非走 `assemble_panel()` — 这意味着 `pctChg` 的注入点在数据文件层面.

**确认**: 查看各调用方是否直接加载 V3 panel:

| 调用方 | 加载方式 | pctChg 可达? |
|--------|---------|-------------|
| `train_pipeline1.py` | 直接 `pd.read_parquet(V3_PANEL_PATH)` | 是 |
| `predict_pipeline1.py` | 同 | 是 |
| `daily_pipeline.py` | `panel_builder.assemble_panel()` → 缓存 | 仅当 V3 路径传入 |

---

### 2.3 `data_supply.py` — 无需修改

`REQUIRED_COLUMNS` 是 `fetch_daily()` 的产出校验, 不影响面板已有的列. `pctChg` 不在 `REQUIRED_COLUMNS` 中, 也不在 `backfill_ohlcv()` 的产出中 — 这说明 `pctChg` 确实由外部脚本注入.

---

### 2.4 `train_runner.py` — 无需修改

IC 筛选器会自动评估 `pctChg` (保留) 和 `pctChg_chgN`/`pctChg_pct_chgN` (加入 skip 后消失). 如果 2.1(B) 做了, 则 `pctChg` 本身也被排除; 否则它会通过筛选.

**建议**: 初始只做 2.1(A), 让 `pctChg` 保留为候选特征但不被差分. 观察 IC 表现后再决定是否加入 2.1(C) 的显式特征.

---

### 2.5 `predict_runner.py` — 无需修改

无特征列引用.

---

### 2.6 `daily_pipeline.py` — 无需修改

`DailySelectionPipeline.run()` 从 panel 构建特征 → 推理. 列名变动由 feature engine 版本控制.

---

### 2.7 Config 文件 — 无需修改

当前所有阈值/窗口参数在 `feature_engine_v35.py` 头部常量 (`MA_WINDOWS`, `BIAS_PERIODS` 等). pctChg 相关的特征周期同这些常量. 无需新增配置.

---

## 3. IC 筛选器影响分析

`screener.py::screen()` 接收 `feature_columns()` 输出作为 `candidates`, 逐因子计算:

- `daily_rank_ic_series`: 按日期横截面 Spearman rank IC
- `rolling_ic_dual`: 60 日滚动窗口
- `ic_stability`: ICIR
- `auc_score`: AUC
- `ic_t_stat_newey_west`: Newey-West HAC t 统计

所有 `pctChg_*` 新列都会参与全量评估. 在加入 skip 之前(即当前状态):

| 列 | 预计 IC | IC 筛选结果 |
|--------|---------|------------|
| `pctChg` | **中高正向** (收益率的自相关) | strong — 保留 |
| `pctChg_chg1` | 低 (<0.01) | dead — 丢弃 |
| `pctChg_chg3/5/10/20` | 低 | 大概率 dead |
| `pctChg_pct_chgN` | 极低/噪声 | dead |

加入 skip 后, 只有 `pctChg` 本身参与筛选.

---

## 4. 实施步骤

### Phase 1: 阻止噪声 (最小改动, 高优先级)

1. 在 `feature_engine_v35.py` `_add_time_series_changes()` 的 `skip_exact` 中加入 `"pctChg"`
2. 验证: 运行 `python scripts/train_pipeline1.py` 确认 `pctChg_chgN` 不在候选特征中
3. 验证: 运行 `python scripts/predict_pipeline1.py` 确认预测不报缺失列错误

### Phase 2: 显式特征 (可选, 价值驱动)

1. 实现 `dim31_pctChg_features()` (滚动均值/波动率/连胜连败)
2. 在 `build()` 的 dim30 后调用
3. 观察 IC 筛选后新特征的留存率

### Phase 3: 长期考虑

1. 如果 `pctChg` 与 `ROC_1d` 相关性 >0.99, 考虑只在 `id_cols` 排除 `pctChg` 并用 `ROC_1d`
2. 如果未来新增更多外部列 (如 `pcfNcfTTM`, `quickRatio`), 统一在 `_add_time_series_changes` 中按语义分类: raw return 类→skip, ratio 类→允许 _chgN, 价格类→已有 dim01 处理

---

## 5. 对照检查: 其他新增列是否受同样问题

从 `merge_extra_fields_to_v3.py` 的 `NEW_COLS`:

| 列名 | 语义 | 是否应被差分 | 建议 |
|------|------|-------------|------|
| `pctChg` | 日收益率 (百分比) | 否, 一阶导已是收益率 | 加入 skip_exact |
| `pcfNcfTTM` | 市现率 (TTM) | 是, ratio 特征 | 允许 _chgN |
| `quickRatio` | 速动比率 | 是, ratio 特征 | 允许 _chgN |
| `cashRatio` | 现金比率 | 是, ratio 特征 | 允许 _chgN |
| `assetToEquity` | 权益乘数 | 是, ratio 特征 | 允许 _chgN |
| `tangibleAssetToAsset` | 有形资产占比 | 是, ratio 特征 | 允许 _chgN |
| `ebitToInterest` | 利息保障倍数 | 是, ratio 特征 | 允许 _chgN |
| `CFOToNP` | 经营现金流/净利润 | 是, ratio 特征 | 允许 _chgN |
| `CFOToGr` | 经营现金流/营收 | 是, ratio 特征 | 允许 _chgN |

只有 `pctChg` 是特殊的 (它本身就是收益率, 已经是价格的一阶导).
