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
import pandas as pd

TURNOVER_TARGET = 0.45  # 换手目标上限 (日均换手 30-45%, 超 45% 部分惩罚)
D6_CONSEC_MONTHS = 3  # D.6: 攻击档连续 N 个月 GT-Score 低于稳定档 → 强制切回


def gt_score(daily_ics, daily_turnover) -> float:
    """GT-Score 修正版 (加性+惩罚).

    Args:
        daily_ics: 日度 OOS Rank IC 序列 (净收益口径)
        daily_turnover: 日度换手序列 (标量亦接受)

    Returns:
        float: 越高越好; 可用于季度超参与 A/B 裁决.
    """
    ics = np.asarray(daily_ics, dtype=float)
    if np.all(np.isnan(ics)):
        return 0.0
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


# ============================================================
# P21.1 双档 GT-Score 对比 (D.6 档位裁决, 月度报表)
# ============================================================
def monthly_gt_scores(dates, daily_ics, daily_turnover) -> dict[str, float]:
    """月度 GT-Score 报表: 同一打分函数按月分桶 (月份键 "YYYY-MM").

    Args:
        dates: 日度日期序列 ("YYYY-MM-DD" 或 Timestamp)
        daily_ics / daily_turnover: 与 gt_score 同口径的日度序列
    """
    df = pd.DataFrame(
        {"month": pd.to_datetime(dates).strftime("%Y-%m"), "ic": daily_ics,
         "to": daily_turnover}
    )
    return {m: gt_score(g["ic"], g["to"]) for m, g in df.groupby("month")}


def dual_profile_verdict(
    stable_monthly: dict[str, float],
    aggressive_monthly: dict[str, float],
    consecutive: int = D6_CONSEC_MONTHS,
) -> dict:
    """D.6 档位裁决: 用事实裁决档位, 不拍脑袋.

    同一批预测、两种执行方式, 逐月对比 GT-Score; 攻击档连续 N 个月
    低于稳定档 → 强制切回 stable 并书面归因. 只统计两档都有数据的月份;
    重叠月份不足 N 个月时不得裁决 (样本不足, 安全网#15).

    Returns:
        {'force_switch_to_stable': bool, 'trailing_below': int,
         'overlap_months': int, 'months': {month: {...}}}
    """
    months = sorted(set(stable_monthly) & set(aggressive_monthly))
    detail = {}
    trailing = 0
    for m in months:
        below = aggressive_monthly[m] < stable_monthly[m]
        detail[m] = {
            "stable": round(stable_monthly[m], 5),
            "aggressive": round(aggressive_monthly[m], 5),
            "aggressive_below": below,
        }
        trailing = trailing + 1 if below else 0  # 月份升序 → 末尾即最新连续段
    switch = len(months) >= consecutive and trailing >= consecutive
    return {
        "force_switch_to_stable": switch,
        "trailing_below": trailing,
        "overlap_months": len(months),
        "months": detail,
    }
