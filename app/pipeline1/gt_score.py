"""
E3 GT-Score 修正版 — 模型选择的唯一标尺 (PIPELINE1_V3.8 §2.7 E3, 检查清单 #69)
================================================================================
原提案乘性公式 (ic_mean×ic_t×ic_pos)/(1+|ic_worst|) 否决:
ic_t 与 ic_mean 数学同源 (t=mean×√n/std) 等于同一量乘两次;
乘性结构在任一项为负时方向混乱. 修正为加性+惩罚:

    gt_score = ic_mean
             + 0.5 × ic_pos_ratio × ic_mean        # 一致性奖励
             - 0.3 × |min(ic_worst_decile, 0)|      # 下行惩罚 (最差10%日子的IC)
             - 1.0 × max(daily_turnover - 0.45, 0)  # 换手惩罚 (超45%部分)

适用范围: 季度超参选择 / 因子剪枝回滚裁决 / 双模型融合健康度 / V3.7+ 一切 A/B
测试的裁决标准. 任何"预期提升"数字必须来自 GT-Score 回测, 禁止凭空填写.
"""

from __future__ import annotations

import numpy as np

TURNOVER_TARGET = 0.45  # 换手目标上限 (日均换手 30-45%, 超 45% 部分惩罚)


def gt_score(daily_ics, daily_turnover) -> float:
    """GT-Score 修正版 (加性+惩罚).

    Args:
        daily_ics: 日度 OOS Rank IC 序列 (净收益口径)
        daily_turnover: 日度换手序列 (标量亦接受)

    Returns:
        float: 越高越好; 可用于季度超参与 A/B 裁决.
    """
    ics = np.asarray(daily_ics, dtype=float)
    ic_mean = float(np.nanmean(ics))
    pos_ratio = float(np.nanmean(ics > 0))
    worst = float(np.nanpercentile(ics, 10))
    turnover_mean = float(np.nanmean(np.asarray(daily_turnover, dtype=float)))
    turnover_pen = max(turnover_mean - TURNOVER_TARGET, 0.0)
    return float(
        ic_mean
        + 0.5 * pos_ratio * ic_mean
        - 0.3 * abs(min(worst, 0.0))
        - 1.0 * turnover_pen
    )
