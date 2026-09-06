"""交付 CSV 百分比显示层 + 兼容两代文件的容忍读取 (2026-09-05 用户: 交付文件
预测值用百分比, 不要小数).

两条铁律:
  上游 DataFrame 保持数值 — fmt_pct_columns 只格式化副本供 to_csv/写 md,
  机器读者 (漂移监控/合并/影子混合) 不受写出层影响;
  旧文件 (数值) 与新文件 ("xx.xx%") 在盘上并存 (WORM 不改写) — parse_pct
  对两代都成立, 链上读者逐值调用即可无缝过渡.

格式约定: 0..1 比例列 ×100 两位小数加 "%"; already_pct 列 (如 Tushare
pctChg 本就是百分数值) 只加 "%" 不再 ×100; NaN → 空串. parse_pct 为其逆:
"78.16%" → 0.7816, 数值/数值串透传, 垃圾串 raise (fail-loud, 盘上文件
都是本工具链写的, 出现垃圾=上游坏了要炸出来).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# parallel_shortlist_*.csv: 预测幅度/概率/超额/实现收益/市场基线 (0..1 比例)
# 不格式化: score/norm_*/rank_blend/sel_20d 等复合分与内部秩, 列位语义不是比例
PCT_COLS_PARALLEL = (
    "pred_mag_3d",
    "pred_prob_3d",
    "pred_mag_5d",
    "pred_prob_5d",
    "pred_mag_10d",
    "pred_prob_10d",
    "pred_ret_3d",
    "pred_ret_5d",
    "pred_ret_10d",
    "pred_prob",
    "pred_excess_3d",
    "pred_excess_5d",
    "pred_excess_10d",
    "ret_10d",
    "ret_1d",
    "market_base_rate",
)

# legacy_stocklist_*.csv: 当日涨跌(day_change 为比例, 与 Tushare pctChg 不同)、
# 预测/概率/分位数/复合/痛苦概率/组合权重 (weight 0.0667 → "6.67%")
PCT_COLS_LEGACY = (
    "day_change",
    "pred_ret_3d",
    "pred_ret_5d",
    "pred_ret_10d",
    "prob_up",
    "prob_up_3d",
    "prob_up_5d",
    "prob_up_10d",
    "compound_ret",
    "compound_prob",
    "pred_q10",
    "pred_q50",
    "pred_q90",
    "pred_q50_3d",
    "pred_q50_5d",
    "uncertainty_width",
    "pain_prob",
    "weight",
    "ret_10d",
    "ret_1d",
    "market_base_rate",
)

_NA_STR = {"", "nan", "none", "nat", "null"}


def fmt_pct_columns(df: pd.DataFrame, cols, already_pct_cols=()) -> pd.DataFrame:
    """df 副本中 cols 列 → "xx.xx%" 文本 (already_pct_cols 只加 %); NaN → 空.

    纯显示层: 入参 df 不改, 未列出的列原样 (score 等复合分保持数值).
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        mult = 1.0 if c in set(already_pct_cols) else 100.0
        out[c] = out[c].map(
            lambda v, mult=mult: "" if pd.isna(v) else f"{v * mult:.2f}%"
        )
    return out


def parse_pct(v) -> float:
    """容忍读取: "78.16%" → 0.7816; float/int/数值串透传 (旧文件); NaN/空 → NaN.

    两代文件并存的过渡期读者统一走这里 — 列内逐值调用 (同一列不会混代, 但
    glob 拼接的多文件会新老混合).
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.lower() in _NA_STR:
        return np.nan
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)
