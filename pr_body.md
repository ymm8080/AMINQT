## 问题

训练预测质量低，模型坍塌（预测值零方差），根因：ES集/校准集数据不足导致早停坍塌+校准器退化。

## 修复内容

### 1. 模型坍塌根因修复 (`dual_track_trainer.py`)
- ES集日期 < 30 时跳过早停（防 `best_iteration=1` 坍塌）
- 校准集日期 < 30 时用 Platt Scaling（防 Isotonic 退化为阶跃函数）
- 段长按 `WINDOW_RATIOS` 比例动态计算（train:es:calib:test = 80:4:4:12）

### 2. 训练端不截断流动性 (`cleaning_pipeline.py`)
- `step2_liquidity` 新增 `apply_top_n` 参数
- 训练端 `run_train` 传 `apply_top_n=False`，不再截取前200，保留全谱股票
- 推理端 `run_inference` 行为不变

### 3. IC筛选放宽 (`ic_screener.py`)
- `ROLLING_MEAN_MIN`: 0.02 → 0.01
- `ROLLING_POS_RATIO_MIN`: 0.60 → 0.50
- `L2_NEG_PERIODS`: 3 → 6
- `nw_significant`: 1.96 → 1.28（90%置信）

### 4. 推理回退加固
- `predictor.py`: LambdaRank退化时回退 `pred_ret_1d` 横截面排名
- `list_generator.py`: 绝对阈值0命中时分位数回退取前20%
- `train_runner.py`: `select_features` 增加非数值列过滤

### 5. 数据深度
- `scripts/train_and_predict.py`: `YEARS` 1.5→3.0

## 验证
- 修复后预测方差恢复: 主板 pred_ret_1d std=0.0127, 创业板 std=0.0076
- 校准器切换为 Platt（小样本不再退化）
- 特征重要性分散: top5=33-58%（不再集中在1-2个特征）
