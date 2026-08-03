# KeyError 推理修复 — 函数说明

> 提交 `e7c4442d` (backtesting) — run_daily_fast / run_pipeline1 KeyError 修复。
> 本文件总结该提交中为修复引入的函数/改动点。

## 根因

选择脚本 (`_abc_brute5k.py` 等) 训练时注入 `_brute_` 后缀特征，
`FeatureEngineV35.build()` 在推理端无法复现 → `predict()` 取 `latest[cols]`
时整列缺失 → `KeyError` + 全零垃圾预测。

## 修复层 1: `_is_inferable(path)` — `app/pipeline1/predict_runner.py:26`

模型包能否被推理端复现的判定器。

- 加载模型包 (`DualTrackTrainer.load`)；
- 取 `feature_cols`，若**任一列含 `_brute_`** → 不可推理 (False)；
- 加载失败 → 告警并返回 False (跳过)；
- 否则 True (全部特征推理端可复现)。

被 `find_bundles()` (`predict_runner.py:41`) 使用：扫描候选包 (字典序最大 → 最新)，
逐包调用 `_is_inferable`，取每个板块第一个可推理的包。若无可用包 → 告警。

## 修复层 2: `predict()` 缺列补 0 — `app/pipeline1/predictor.py:52`

`predict()` 在 `latest[cols]` 索引前做防御性清洗：

- 计算 `missing = [c for c in cols if c not in latest.columns]`；
- 缺失列补 `0.0` 并告警 (`特征缺失 N/M 补 0`);
- `np.nan_to_num(..., nan=0, posinf=0, neginf=0)` 兜底 NaN/Inf。

即使 `_is_inferable` 漏网 (或面板本身缺列)，推理也永不抛 KeyError。

## 验证

- `test_daily_pipeline` + `test_train_predict_runner`: 21/21 passed (12:44)
- 其余 6 个受影响测试文件: 149/149 passed
- 合计 **170/170 green**; `find_bundles` 实测:
  main → `main_2026W31_real_tr.pkl` (brute=0), dual → `dual_current.pkl` (brute=0)
