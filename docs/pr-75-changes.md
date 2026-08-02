# PR #75 变更总结 — run_daily_fast/run_pipeline1 KeyError 修复 + 2d 视界

> 分支 `fix/keyerror-infer` → `main`，20 文件，+196/-49。
> 详细函数说明见 `docs/keyerror-inference-fix.md`。

## 1. 根因

选择脚本 (`_abc_brute5k.py` 等) 训练时向面板注入 `_brute_` 后缀特征，
`FeatureEngineV35.build()` 推理端无法复现 → `predict()` 取 `latest[cols]`
整列缺失 → `KeyError` + 全零垃圾预测。

## 2. 核心修复（两层防 KeyError）

| 层 | 位置 | 改动 |
|----|------|------|
| 1 | `predict_runner.py` | 新增 `_is_inferable(path)`：加载模型包，`feature_cols` 含任一 `_brute_` 列 → 判定不可推理。`find_bundles()` 改为逐包扫描（最新→旧），取每个板块**第一个可推理**的包 |
| 2 | `predictor.py` | `predict()` 在 `latest[cols]` 索引前计算缺失列并补 `0.0` + 告警；`np.nan_to_num` 兜底 NaN/Inf。即使漏网也永不抛 KeyError |

实测：main → `main_2026W31_real_tr.pkl` (brute=0)，dual → `dual_current.pkl` (brute=0)。

## 3. 2d 视界扩展（T+1 制度下的中间执行视界）

`LABEL_HORIZONS` 从 `(1, 3, 5)` → `(1, 2, 3, 5)`。

- **`label_engine.py`**：新增全局 `LABEL_WEIGHTS = {1: 0.15, 2: 0.25, 3: 0.35, 5: 0.25}`
  （1d 权重最低 — T+1 买入当日不可卖最不可执行；3d 历史预测力最强）。
  修改此字典即全局生效。`split_sets()` 新增 `"2d"` 分片。
- **`checkpoint.py` / `dual_track_trainer.py`**：`MODEL_KINDS` 加 `"2d_reg"`；
  `dual_track_trainer` 标签映射加 `label_2d`，`feature_stability` 遍历加 2d。
- **`list_generator.py`**：`SCHEMA_VERSION` 1.2 → 1.3；`COMPOUND_W` 改从
  `LABEL_WEIGHTS` 派生；`SCHEMA_FIELDS` 加 `pred_ret_2d`。
- **`prediction_db.py`**：新增 `_migrate()` 为历史库补 `pred_ret_2d` /
  `actual_ret_2d` 列（`ALTER TABLE`，不改既有表）；插入/结局表含 2d。
- **`forecast_accuracy.py`**：`HORIZONS = (1, 2, 3, 5)`。
- **`feature_registry.py` / `ic_screener.py`**：IC 等级计算遍历含 2d。

## 4. OOS 验证改为跨视界加权 IC

- **`dual_track_trainer.validate_oos()`**：开关门判据从 `max(1d/3d/5d) IC`
  改为 `weighted_ic = Σ(LABEL_WEIGHTS[k] × IC_{k}d) / ΣW`（1d_cls 不参与，
  分类分不直接贡献收益率）。返回新增 `weighted_ic`。
- **`train_runner.py`**：日志同步改为打印 `weighted_ic`。

## 5. 综合排序分统一口径

`predictor.py` 的 `composite_score` 与 `list_generator.py` 的 `COMPOUND_W`
都改为由 `LABEL_WEIGHTS` 派生（除以总权重），避免两处权重漂移。

## 6. 测试修复（8 文件）

- 源链 **tushare-first 全部 mock**（消除对网络/真实数据的依赖）。
- 供应失败守卫改为 `append_today_to_panel`，防 1GB 面板 OOM。
- 补充 2d 视界用例：`test_train_predict_runner.TestHorizonWeights`、
  `list_to_panel`、`v38_dual_list` 等。

## 7. 验证

- `test_daily_pipeline` + `test_train_predict_runner`：21/21 passed
- 其余 6 个受影响测试文件：149/149 passed
- 合计 **170/170 green**（CI Test 复跑通过）
- Lint 通过（修复 `_daily_fetch.py` 双重 BOM；已整体回退该文件丢弃无关
  turnover 改动）

## 8. 文件清单（20）

```
app/pipeline1/checkpoint.py        app/pipeline1/dual_track_trainer.py
app/pipeline1/feature_registry.py  app/pipeline1/forecast_accuracy.py
app/pipeline1/ic_screener.py       app/pipeline1/label_engine.py
app/pipeline1/list_generator.py    app/pipeline1/predict_runner.py
app/pipeline1/prediction_db.py     app/pipeline1/predictor.py
app/pipeline1/train_runner.py      docs/keyerror-inference-fix.md
tests/test_daily_pipeline.py       tests/test_forecast_accuracy.py
tests/test_indicators.py           tests/test_pipeline1_list_to_panel.py
tests/test_pipeline1_v35.py        tests/test_pipeline1_v38.py
tests/test_pipeline1_v38_dual_list.py  tests/test_train_predict_runner.py
```
