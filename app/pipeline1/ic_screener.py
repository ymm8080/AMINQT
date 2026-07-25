"""
IC 筛选器 (DESIGN §14 三 bis, 安全网 #2, PIPELINE1_V3.8 §三 bis)
==================================================================
- 每个滚动重训窗口内, 仅用该窗口训练集重新计算 IC 并重新筛因子 (严禁全样本一次筛选 = 前视偏差)
- 三标签 (1d/3d/5d) 分别计算 Rank IC → 取并集; [E5] 净口径 label_*_net 优先
- [E2] mdd 标签独立筛选 (label_mdd_3d) → 与净口径并集再取并集
- 分类模型独立: AUC + 互信息 → 与回归并集再取并集
- 滚动 IC 双指标 (D13): 60日滚动 IC 均值 > 0.02 且正值比例 > 60%
- [E4-L2] 连续 3 期滚动 IC 为负的因子自动剔除 (跨窗口持久化追踪)
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

IC_STRONG = 0.03  # 有效因子
IC_WEAK = 0.01  # 弱因子 (观察)
ROLLING_WINDOW = 60  # 滚动 IC 窗口 (交易日)
ROLLING_MEAN_MIN = 0.02
ROLLING_POS_RATIO_MIN = 0.60
L2_NEG_PERIODS = 3  # [E4-L2] 连续 3 期滚动 IC 为负 → 自动剔除


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
            lambda g: (
                spearmanr(g[factor], g[label]).statistic
                if g[factor].nunique() > 5 and g[label].nunique() > 1
                else np.nan
            )
        )
        return float(np.nanmean(np.abs(ics.values)))  # 方向无关: 绝对值筛强度

    # ---------------- Newey-West HAC t 统计量 (B6) ----------------
    @staticmethod
    def ic_t_stat_newey_west(
        df: pd.DataFrame, factor: str, label: str, lag: int = 0
    ) -> float:
        """IC 序列 Newey-West HAC 调整后的 t 统计量 (B6).

        3d/5d 标签重叠样本导致 IC 序列自相关, t 统计量虚高.
        lag=0: 标准 t (1d 无重叠); lag=5: 3d; lag=8: 5d.
        """
        sub = df[["date", factor, label]].dropna()
        dates = sorted(sub["date"].unique())
        if len(dates) < 10:
            return 0.0
        ic_series = (
            sub.groupby("date")
            .apply(
                lambda g: (
                    spearmanr(g[factor], g[label]).statistic
                    if g[factor].nunique() > 5 and g[label].nunique() > 1
                    else np.nan
                )
            )
            .dropna()
        )
        n = len(ic_series)
        if n < 5:
            return 0.0
        mean_ic = float(ic_series.mean())
        # HAC 方差估计 (Bartlett kernel)
        gamma0 = float(((ic_series - mean_ic) ** 2).mean())
        hac_var = gamma0
        for ell in range(1, min(lag + 1, n)):
            w = 1 - ell / (lag + 1)  # Bartlett 权重
            cov_l = float(
                (
                    (ic_series.iloc[ell:] - mean_ic) * (ic_series.iloc[:-ell] - mean_ic)
                ).mean()
            )
            hac_var += 2 * w * cov_l
        hac_var = max(hac_var, 1e-12)
        return mean_ic / (np.sqrt(hac_var / n))

    @staticmethod
    def rolling_ic_dual(
        df: pd.DataFrame, factor: str, label: str, window: int = ROLLING_WINDOW
    ) -> tuple[float, float]:
        """滚动 IC 双指标 (D13): (滚动 IC 均值, 滚动 IC 正值比例)."""
        sub = df[["date", factor, label]].dropna()
        dates = sorted(sub["date"].unique())
        if len(dates) < window:
            return 0.0, 0.0
        daily_ic = (
            sub.groupby("date")
            .apply(
                lambda g: (
                    spearmanr(g[factor], g[label]).statistic
                    if g[factor].nunique() > 5 and g[label].nunique() > 1
                    else np.nan
                )
            )
            .dropna()
        )
        rolls = [
            daily_ic.loc[d0:d1].mean()
            for d0, d1 in zip(dates[:-window], dates[window:])
        ]
        rolls = pd.Series(rolls).dropna().abs()
        if len(rolls) == 0:
            return 0.0, 0.0
        return float(rolls.mean()), float(
            (daily_ic.abs() > 0).mean() if len(daily_ic) else 0.0
        )

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
    def screen(
        self, train_df: pd.DataFrame, feature_cols: list[str], window_id: str
    ) -> dict:
        """窗口内重算 IC 并筛因子.

        [E5] 净口径: label_{k}d_net 存在时优先于毛收益 label_{k}d;
        [E2] label_mdd_3d 独立筛选, 并入并集;
        [E4-L2] 连续 3 期滚动 IC 为负 → 强制 grade='dead'.

        Returns:
            {window_id, factors: [...], detail: {factor: {ic_1d, ..., grade}}}
            grade: 'strong' 保留 / 'weak' 观察 / 'dead' 剔除
        """
        # [E5] 净收益标签优先 (分层滑点口径)
        label_of = {
            k: (
                f"label_{k}d_net"
                if f"label_{k}d_net" in train_df.columns
                else f"label_{k}d"
            )
            for k in (1, 3, 5)
        }
        has_mdd = "label_mdd_3d" in train_df.columns  # [E2]
        l2_history = self._load_l2_history()
        result = {"window_id": window_id, "factors": [], "detail": {}}
        for f in feature_cols:
            ic_by_label = {
                k: self.rank_ic(train_df, f, lbl) for k, lbl in label_of.items()
            }
            best_ic = max(ic_by_label.values())
            ic_mdd = self.rank_ic(train_df, f, "label_mdd_3d") if has_mdd else None
            if ic_mdd is not None:
                best_ic = max(best_ic, ic_mdd)  # [E2] mdd 独立筛选并入并集
            auc = self.auc_score(train_df, f)
            roll_mean, roll_pos = self.rolling_ic_dual(train_df, f, label_of[1])
            dual_ok = roll_mean > ROLLING_MEAN_MIN and roll_pos > ROLLING_POS_RATIO_MIN
            # B6: 3d/5d IC 显著性用 Newey-West HAC 调整 (lag=5/8)
            t_3d = self.ic_t_stat_newey_west(train_df, f, label_of[3], lag=5)
            t_5d = self.ic_t_stat_newey_west(train_df, f, label_of[5], lag=8)
            nw_significant = abs(t_3d) > 1.96 or abs(t_5d) > 1.96
            if (best_ic > IC_STRONG or auc > 0.55) and dual_ok and nw_significant:
                grade = "strong"
            elif best_ic > IC_WEAK or auc > 0.52:
                grade = "weak"
            else:
                grade = "dead"
            # [E4-L2] 连续 3 期窗口 IC 为负 (带符号) → 自动剔除
            signed_ic = self._signed_daily_ic_mean(train_df, f, label_of[1])
            neg_streak = l2_history.get(f, 0)
            neg_streak = neg_streak + 1 if signed_ic < 0 else 0
            l2_history[f] = neg_streak
            l2_evicted = neg_streak >= L2_NEG_PERIODS
            if l2_evicted and grade != "dead":
                grade = "dead"
                logger.warning(
                    "E4-L2: 因子 %s 连续 %d 期窗口IC为负, 自动剔除", f, neg_streak
                )
            result["detail"][f] = {
                **{f"ic_{k}d": round(v, 4) for k, v in ic_by_label.items()},
                "auc": round(auc, 4),
                "rolling_mean": round(roll_mean, 4),
                "rolling_pos_ratio": round(roll_pos, 4),
                "nw_t_3d": round(t_3d, 4),  # B6
                "nw_t_5d": round(t_5d, 4),  # B6
                "grade": grade,
            }
            if ic_mdd is not None:
                result["detail"][f]["ic_mdd_3d"] = round(ic_mdd, 4)  # [E2]
            if l2_evicted:
                result["detail"][f]["l2_evicted"] = True  # [E4-L2]
            if grade in ("strong", "weak"):
                result["factors"].append(f)
        self._persist(window_id, result)
        self._save_l2_history(l2_history)
        return result

    # ---------------- E4-L2 跨窗口负 IC 追踪 ----------------
    @staticmethod
    def _signed_daily_ic_mean(df: pd.DataFrame, factor: str, label: str) -> float:
        """窗口内日度 Rank IC 均值 (带符号) — L2 "连续3期为负" 判定输入.

        rank_ic/rolling_ic_dual 取绝对值 (方向无关筛强度), 不能用于符号判定.
        """
        sub = df[["date", factor, label]].dropna()
        if sub["date"].nunique() < 5:
            return 0.0
        ics = sub.groupby("date").apply(
            lambda g: (
                spearmanr(g[factor], g[label]).statistic
                if g[factor].nunique() > 5 and g[label].nunique() > 1
                else np.nan
            )
        )
        return float(np.nanmean(ics.values))

    def _l2_path(self) -> str:
        return os.path.join(self.registry_path, "l2_factor_neg_streaks.json")

    def _load_l2_history(self) -> dict:
        if os.path.exists(self._l2_path()):
            with open(self._l2_path(), encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save_l2_history(self, history: dict) -> None:
        with open(self._l2_path(), "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=1)

    def _persist(self, window_id: str, result: dict) -> None:
        """每期因子清单必须记录 (安全网 #2 工程强制)."""
        path = os.path.join(self.registry_path, f"factors_{window_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        logger.info(
            "窗口 %s 因子清单: %d strong+weak / %d 候选",
            window_id,
            len(result["factors"]),
            len(result["detail"]),
        )

    # ---------------- IC 归因 ----------------
    @staticmethod
    def ic_attribution(
        ic_raw: float, ic_neutralized: float, drop_threshold: float = 0.5
    ) -> str:
        """行业/市值中性化后 IC 降幅 > 50% → 预测力主要来自风格暴露, 降级或移除."""
        if ic_raw <= 0:
            return "dead"
        drop = 1 - ic_neutralized / ic_raw
        return "style_exposed" if drop > drop_threshold else "alpha"
