#!/usr/bin/env python3
"""从 predictions_3y.csv 筛选候选, 放宽阈值."""
import pandas as pd
import numpy as np

preds = pd.read_csv("predictions_3y.csv")
print(f"总预测: {len(preds)} 只")
print(f"prob_up 分布: min={preds['prob_up'].min():.4f} p25={preds['prob_up'].quantile(0.25):.4f} "
      f"median={preds['prob_up'].median():.4f} p75={preds['prob_up'].quantile(0.75):.4f} max={preds['prob_up'].max():.4f}")

# 放宽: prob_up > 0.40, pred_ret_3d > 0.5%, pred_ret_5d > 0.5%, pain_prob < 0.40
mask = (
    (preds["prob_up"] >= 0.40) &
    (preds["pred_ret_3d"] > 0.005) &
    (preds["pred_ret_5d"] > 0.005) &
    (preds["pain_prob"].fillna(1) < 0.40)
)
if "ATR_pct" in preds.columns:
    mask &= (preds["ATR_pct"].fillna(0.1) < 0.08)

filtered = preds[mask].copy()
# 综合评分: prob_up 权重 + 收益权重 - 风险权重
filtered["score"] = (
    filtered["prob_up"] * 0.3
    + filtered["pred_ret_3d"] * 20 * 0.3
    + filtered["pred_ret_5d"] * 10 * 0.2
    - filtered["pain_prob"].fillna(0.3) * 0.2
)
filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)

cols = ["symbol", "board", "industry", "pred_ret_1d", "pred_ret_3d",
        "pred_ret_5d", "prob_up", "pain_prob", "score"]
if "ATR_pct" in filtered.columns:
    cols.append("ATR_pct")
if "close" in filtered.columns:
    cols.append("close")

print(f"\n筛选后: {len(filtered)} 只")
print(f"条件: prob_up>=0.40, ret_3d>0.5%, ret_5d>0.5%, pain<0.40, ATR<8%")
print("=" * 120)
print(filtered[cols].head(30).to_string(index=False))

filtered.to_csv("filtered_candidates.csv", index=False)
print(f"\n已保存 filtered_candidates.csv")
