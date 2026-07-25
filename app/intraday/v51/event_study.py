"""
单规则事件研究 (V5.1 P20.0, 参数寻优关卡 0/1, 检查清单 #5)
==================================================================
每条规则独立做事件研究: 收集触发事件的后续净收益样本,
关卡 0: 事件数 ≥ 1000 (不足 → 规则不具备统计意义)
关卡 1: 显著性 t-stat > 3.0
铁律 #2: 规则必须有经济学解释 + 平原检测通过 (数据挖掘自欺).
"""

from __future__ import annotations

import numpy as np

from .safe_div import safe_divide

MIN_EVENTS = 1000  # 关卡 0: 事件数下限
T_STAT_MIN = 3.0  # 关卡 1: 显著性下限


def event_study(
    forward_returns: list[float] | np.ndarray,
    min_events: int = MIN_EVENTS,
    t_min: float = T_STAT_MIN,
) -> dict:
    """单规则事件研究 (双输入之一, B5 闸门事件研究均值的来源).

    Args:
        forward_returns: 每次触发后的净收益样本 (扣费后, E5 同口径)
    Returns:
        {'n', 'mean', 't_stat', 'pass_gate0', 'pass_gate1', 'pass'}
    """
    r = np.asarray(forward_returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 2:
        return {
            "n": n,
            "mean": 0.0,
            "t_stat": 0.0,
            "pass_gate0": False,
            "pass_gate1": False,
            "pass": False,
        }
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    se = safe_divide(std, float(np.sqrt(n)))
    t_stat = safe_divide(mean, se) if se > 0 else 0.0
    g0 = n >= min_events
    g1 = abs(t_stat) > t_min
    return {
        "n": n,
        "mean": round(mean, 6),
        "t_stat": round(t_stat, 3),
        "pass_gate0": g0,
        "pass_gate1": g1,
        "pass": g0 and g1,
    }
