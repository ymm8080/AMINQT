"""
参数寻优 (V5.1 §7, 三关审判 + 平原检测 + Walk-Forward, 检查清单 #13)
==========================================================================
评分函数: GT-Score (PIPELINE1 附录 E.3 同公式) — 日内规则与日线模型同一把尺.
寻优对象: B1/B2 正向参数、B3/B5/B6 否决阈值、S2 回撤带系数、S5a 窗口;
          S1 stop_price 不在寻优范围 (PIPELINE1 动态下发).
寻优频率: 季度, 与 PIPELINE1 超参窗口同步; 盘中绝对禁止热更新.

三关审判:
  关卡 1 显著性: t-stat > 3.0 (event_study)
  关卡 2 平原检测 (组合判据): 邻域 ≥70% 同向为正 且 邻域均值 ≥ 候选值 50%
  关卡 3 样本外衰减: 验证段超额 ≥ 训练段 50% (防 regime 依赖)
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

from .safe_div import safe_divide

logger = logging.getLogger(__name__)

T_STAT_MIN = 3.0
PLATEAU_SAME_DIR_MIN = 0.70  # 邻域 ≥70% 同向为正
PLATEAU_NEIGHBOR_MEAN_MIN = 0.50  # 邻域均值 ≥ 候选值 50%
OOS_DECAY_MIN = 0.50  # 验证段 ≥ 训练段 50%


# ============================================================
# 关卡 2: 平原检测 (拒绝孤峰)
# ============================================================
def plateau_check(neighbor_scores: list[float], candidate_score: float) -> dict:
    """平原检测 (组合判据): 候选参数的邻域表现必须同样为正且不太弱.

    孤峰 (只有候选点为正 / 邻域均值远低于候选) = 运气, 拒绝.
    """
    n = len(neighbor_scores)
    if n == 0:
        return {"pass": False, "reason": "无邻域样本"}
    positive = [s > 0 for s in neighbor_scores]
    same_dir = safe_divide(float(sum(positive)), float(n))
    neighbor_mean = float(np.mean(neighbor_scores))
    mean_ok = neighbor_mean >= 0.5 * candidate_score if candidate_score > 0 else False
    ok = same_dir >= PLATEAU_SAME_DIR_MIN and mean_ok
    if not ok:
        logger.error(
            "平原检测拒绝: 邻域同向 %.0f%% (<70%%) 或邻域均值 %.4f < 候选 %.4f×50%%",
            same_dir * 100,
            neighbor_mean,
            candidate_score,
        )
    return {
        "pass": ok,
        "same_dir_ratio": round(same_dir, 3),
        "neighbor_mean": round(neighbor_mean, 5),
    }


# ============================================================
# 关卡 3: 样本外衰减
# ============================================================
def oos_decay_check(train_score: float, oos_score: float) -> dict:
    """验证段超额 ≥ 训练段 50% (防 regime 依赖)."""
    if train_score <= 0:
        return {"pass": False, "decay": 0.0}
    keep = safe_divide(oos_score, train_score)
    return {"pass": keep >= OOS_DECAY_MIN, "decay": round(keep, 3)}


# ============================================================
# 三关审判 (组合裁决)
# ============================================================
def three_gate_verdict(
    t_stat: float,
    neighbor_scores: list[float],
    candidate_score: float,
    oos_score: float,
) -> dict:
    """三关全过 → 参数可进 candidate (四态流转起点)."""
    g1 = abs(t_stat) > T_STAT_MIN
    g2 = plateau_check(neighbor_scores, candidate_score)
    g3 = oos_decay_check(candidate_score, oos_score)
    return {
        "pass": g1 and g2["pass"] and g3["pass"],
        "gates": {"t_stat": g1, "plateau": g2, "oos_decay": g3},
    }


# ============================================================
# 网格寻优 + Walk-Forward
# ============================================================
def grid_search(param_grid: dict, evaluate_fn, top_k: int = 1) -> list[dict]:
    """网格寻优: evaluate_fn(params) → score (GT-Score). 返回 top_k.

    季度执行一次, 盘中绝对禁止热更新参数.
    """
    names = list(param_grid)
    combos = [
        dict(zip(names, v)) for v in itertools.product(*(param_grid[n] for n in names))
    ]
    scored = [(c, evaluate_fn(c)) for c in combos]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"params": c, "score": s} for c, s in scored[:top_k]]


def walk_forward(
    windows: list[tuple], evaluate_train_fn, evaluate_oos_fn
) -> list[dict]:
    """Walk-Forward: 每个窗口 训练段寻优 → 验证段复验 (GT-Score 评分).

    Args:
        windows: [(window_id, train_data, oos_data), ...]
        evaluate_train_fn(train_data) → {'params', 'score', 't_stat',
                                          'neighbor_scores'}
        evaluate_oos_fn(params, oos_data) → score
    Returns:
        每窗口 {'window_id', 'params', 'train_score', 'oos_score', 'verdict'}
    """
    results = []
    for wid, train, oos in windows:
        best = evaluate_train_fn(train)
        if best is None:
            continue
        oos_score = evaluate_oos_fn(best["params"], oos)
        verdict = three_gate_verdict(
            best["t_stat"], best["neighbor_scores"], best["score"], oos_score
        )
        results.append(
            {
                "window_id": wid,
                "params": best["params"],
                "train_score": best["score"],
                "oos_score": oos_score,
                "verdict": verdict,
            }
        )
        if not verdict["pass"]:
            logger.error("Walk-Forward 窗口 %s 三关未过: %s", wid, verdict["gates"])
    return results
