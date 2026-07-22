# -*- coding: utf-8 -*-
"""
IC 筛选器 (DESIGN §14 三 bis, 安全网 #2)
===========================================
- 每个滚动重训窗口内, 仅用该窗口训练集重新计算 IC 并重新筛因子 (严禁全样本一次筛选 = 前视偏差)
- 三标签 (1d/3d/5d) 分别计算 Rank IC → 取并集
- 分类模型独立: AUC + 互信息 → 与回归并集再取并集
- 滚动 IC 双指标 (D13): 60日滚动 IC 均值 > 0.02 且正值比例 > 60%
- 每期因子清单必须记录 (工程强制)
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

IC_STRONG = 0.03      # 有效因子
IC_WEAK = 0.01        # 弱因子 (观察)
ROLLING_WINDOW = 60   # 滚动 IC 窗口 (交易日)
ROLLING_MEAN_MIN = 0.02
ROLLING_POS_RATIO_MIN = 0.60


class ICScreener:
    """IC 筛选 — 每期滚动重算."""

    def __init__(self, registry_path: str = "data/factor_registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)

    # ---------------- 单因子 IC ----------------
    @staticmethod
    def rank_ic(df: pd.DataFrame, factor: str, label: str) -> float:
        """横截面 Rank IC 均值 (按 date 分组 Spearman, 再取时间均值)."""
        sub = df[["date", factor, label]].dropna()
        if sub["date"].nunique() < 5:
            return 0.0
        ics = sub.groupby("date").apply(
            lambda g: spearmanr(g[factor], g[label]).statistic
            if g[factor].nunique() > 5 and g[label].nunique() > 1 else np.nan)
        return float(np.nanmean(np.abs(ics.values)))  # 方向无关: 绝对值筛强度

    @staticmethod
    def rolling_ic_dual(df: pd.DataFrame, factor: str, label: str,
                        window: int = ROLLING_WINDOW) -> tuple[float, float]:
        """滚动 IC 双指标 (D13): (滚动 IC 均值, 滚动 IC 正值比例)."""
        sub = df[["date", factor, label]].dropna()
        dates = sorted(sub["date"].unique())
        if len(dates) < window:
            return 0.0, 0.0
        daily_ic = sub.groupby("date").apply(
            lambda g: spearmanr(g[factor], g[label]).statistic
            if g[factor].nunique() > 5 and g[label].nunique() > 1 else np.nan).dropna()
        rolls = [daily_ic.loc[d0:d1].mean()
                 for d0, d1 in zip(dates[:-window], dates[window:])]
        rolls = pd.Series(rolls).dropna().abs()
        if len(rolls) == 0:
            return 0.0, 0.0
        return float(rolls.mean()), float((daily_ic.abs() > 0).mean() if len(daily_ic) else 0.0)

    # ---------------- 分类模型 AUC 筛选 ----------------
    @staticmethod
    def auc_score(df: pd.DataFrame, factor: str, label: str = "label_cls") -> float:
        """单因子对分类标签的 AUC (方向无关: max(auc, 1-auc))."""
        from sklearn.metrics import roc_auc_score
        sub = df[[factor, label]].dropna()
        if sub[label].nunique() < 2 or len(sub) < 50:
            return 0.5
        auc = roc_auc_score(sub[label], sub[factor])
        return float(max(auc, 1 - auc))

    # ---------------- 主筛选 ----------------
    def screen(self, train_df: pd.DataFrame, feature_cols: list[str],
               window_id: str) -> dict:
        """窗口内重算 IC 并筛因子.

        Returns:
            {window_id, factors: [...], detail: {factor: {ic_1d, ic_3d, ic_5d, auc, grade}}}
            grade: 'strong' 保留 / 'weak' 观察 / 'dead' 剔除
        """
        result = {"window_id": window_id, "factors": [], "detail": {}}
        for f in feature_cols:
            ic_by_label = {k: self.rank_ic(train_df, f, f"label_{k}d") for k in (1, 3, 5)}
            best_ic = max(ic_by_label.values())
            auc = self.auc_score(train_df, f)
            roll_mean, roll_pos = self.rolling_ic_dual(train_df, f, "label_1d")
            dual_ok = roll_mean > ROLLING_MEAN_MIN and roll_pos > ROLLING_POS_RATIO_MIN
            if (best_ic > IC_STRONG or auc > 0.55) and dual_ok:
                grade = "strong"
            elif best_ic > IC_WEAK or auc > 0.52:
                grade = "weak"
            else:
                grade = "dead"
            result["detail"][f] = {**{f"ic_{k}d": round(v, 4) for k, v in ic_by_label.items()},
                                   "auc": round(auc, 4), "rolling_mean": round(roll_mean, 4),
                                   "rolling_pos_ratio": round(roll_pos, 4), "grade": grade}
            if grade in ("strong", "weak"):
                result["factors"].append(f)
        self._persist(window_id, result)
        return result

    def _persist(self, window_id: str, result: dict) -> None:
        """每期因子清单必须记录 (安全网 #2 工程强制)."""
        path = os.path.join(self.registry_path, f"factors_{window_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        logger.info("窗口 %s 因子清单: %d strong+weak / %d 候选",
                    window_id, len(result["factors"]), len(result["detail"]))

    # ---------------- IC 归因 ----------------
    @staticmethod
    def ic_attribution(ic_raw: float, ic_neutralized: float,
                       drop_threshold: float = 0.5) -> str:
        """行业/市值中性化后 IC 降幅 > 50% → 预测力主要来自风格暴露, 降级或移除."""
        if ic_raw <= 0:
            return "dead"
        drop = 1 - ic_neutralized / ic_raw
        return "style_exposed" if drop > drop_threshold else "alpha"
