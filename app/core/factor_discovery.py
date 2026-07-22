# -*- coding: utf-8 -*-
"""训练自动因子发现 (P11, ARCH §5.5).

每次训练后自动评估因子有效性, 输出 Top-K 因子报告:
LightGBM importance (30%) + SHAP (25%) + IC (25%) + ICIR (20%) 综合评分。
tech_ths_ctrl_ratio 强制保留在 Top-K (ARCH §5.13.8.A)。

环境降级策略 (重依赖全部惰性导入):
- lightgbm 缺失 → sklearn GradientBoostingRegressor 的
  ``feature_importances_`` 作为 lgbm_gain 兜底。
- shap 缺失 → sklearn ``permutation_importance`` 作为 shap_mean 兜底。
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

FORCE_TOP_FACTORS = ["tech_ths_ctrl_ratio"]  # 强制进入 Top-K (ARCH §5.13.8)

# 综合评分权重 (ARCH §5.5.1)
W_GAIN = 0.30
W_SHAP = 0.25
W_IC = 0.25
W_ICIR = 0.20

# ICIR 滚动窗口 (交易日, ARCH §5.5.2 Step 4)
_ICIR_WINDOW = 60


def _minmax_norm(s: pd.Series) -> pd.Series:
    """min-max 归一化到 [0,1]; 常数列 → 0."""
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


class FactorDiscovery:
    """因子发现器."""

    def __init__(self, config: dict = None) -> None:
        self.config = config or {}

    # ────────────────────────────────────────────────────────────────
    #  IC / ICIR
    # ────────────────────────────────────────────────────────────────

    def compute_ic(self, factor: pd.Series, y: pd.Series) -> float:
        """单因子 IC (Spearman 相关, pandas rank corr 实现).

        Args:
            factor: 因子值序列。
            y: 未来收益序列 (与 factor 对齐)。

        Returns:
            Spearman 相关系数; 有效样本 < 3 或零方差 → 0.0。
        """
        pair = pd.concat([factor, y], axis=1).dropna()
        if len(pair) < 3:
            return 0.0
        f_rank = pair.iloc[:, 0].rank()
        y_rank = pair.iloc[:, 1].rank()
        if f_rank.std() == 0 or y_rank.std() == 0:
            return 0.0
        ic = float(f_rank.corr(y_rank))
        return ic if np.isfinite(ic) else 0.0

    def compute_icir(self, factor: pd.Series, y: pd.Series) -> float:
        """单因子 ICIR (逐期 IC 序列的 均值/标准差).

        逐期划分: 索引为 DatetimeIndex 时按月分组; 否则按
        ``_ICIR_WINDOW`` 长度顺序切块。每期内需 ≥5 个有效样本。

        Args:
            factor: 因子值序列。
            y: 未来收益序列。

        Returns:
            IC_mean / IC_std; 期数 < 2 或 std=0 → 0.0。
        """
        pair = pd.concat([factor, y], axis=1).dropna()
        if len(pair) < 10:
            return 0.0

        ics: List[float] = []
        if isinstance(pair.index, pd.DatetimeIndex):
            groups = pair.groupby(pair.index.to_period("M"))
            for _, g in groups:
                if len(g) >= 5:
                    ics.append(self.compute_ic(g.iloc[:, 0], g.iloc[:, 1]))
        else:
            for start in range(0, len(pair), _ICIR_WINDOW):
                g = pair.iloc[start : start + _ICIR_WINDOW]
                if len(g) >= 5:
                    ics.append(self.compute_ic(g.iloc[:, 0], g.iloc[:, 1]))

        if len(ics) < 2:
            return 0.0
        arr = np.asarray(ics, dtype=float)
        std = float(arr.std(ddof=1))
        if std == 0 or not np.isfinite(std):
            return 0.0
        icir = float(arr.mean()) / std
        return icir if np.isfinite(icir) else 0.0

    # ────────────────────────────────────────────────────────────────
    #  特征重要性 (gain / shap, 重依赖惰性导入 + 兜底)
    # ────────────────────────────────────────────────────────────────

    def _fit_fallback_model(self, X: pd.DataFrame, y: pd.Series):
        """训练兜底 GBM (lightgbm 缺失时使用)."""
        params = dict(
            n_estimators=int(self.config.get("gbm_n_estimators", 80)),
            max_depth=int(self.config.get("gbm_max_depth", 3)),
            random_state=42,
        )
        model = GradientBoostingRegressor(**params)
        model.fit(X, y)
        return model

    def _compute_gain(self, X: pd.DataFrame, y: pd.Series, model=None) -> np.ndarray:
        """LightGBM gain 重要性; 缺失时 GBM feature_importances_ 兜底.

        优先级: 传入 model.feature_importances_ → 惰性 lightgbm 训练 →
        sklearn GBM 兜底。
        """
        if model is not None and hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
            if len(imp) == X.shape[1]:
                return imp
            logger.warning("model.feature_importances_ 维度不匹配, 改用兜底")

        try:
            import lightgbm as lgb  # noqa: WPS433 (刻意惰性导入)

            lgbm = lgb.LGBMRegressor(
                n_estimators=int(self.config.get("lgbm_n_estimators", 200)),
                random_state=42,
            )
            lgbm.fit(X, y)
            booster = lgbm.booster_
            return np.asarray(
                booster.feature_importance(importance_type="gain"), dtype=float
            )
        except ImportError:
            logger.warning(
                "lightgbm 未安装 — 用 sklearn GBM feature_importances_ 兜底 lgbm_gain"
            )
            gbm = self._fit_fallback_model(X, y)
            return np.asarray(gbm.feature_importances_, dtype=float)

    def _compute_shap(self, X: pd.DataFrame, y: pd.Series, model=None) -> np.ndarray:
        """SHAP mean(|shap|); 缺失时 permutation_importance 兜底.

        优先级: 惰性 shap.TreeExplainer(model) → sklearn
        permutation_importance (weight 25%)。
        """
        try:
            import shap  # noqa: WPS433 (刻意惰性导入)

            if model is None:
                model = self._fit_fallback_model(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            return np.abs(np.asarray(shap_values, dtype=float)).mean(axis=0)
        except ImportError:
            logger.warning(
                "shap 未安装 — 用 sklearn permutation_importance 兜底 shap_mean"
            )
            if model is None or not hasattr(model, "predict"):
                model = self._fit_fallback_model(X, y)
            perm = permutation_importance(
                model,
                X,
                y,
                n_repeats=int(self.config.get("perm_n_repeats", 5)),
                random_state=42,
            )
            return np.asarray(perm.importances_mean, dtype=float)

    # ────────────────────────────────────────────────────────────────
    #  综合评估
    # ────────────────────────────────────────────────────────────────

    def run(self, X: pd.DataFrame, y: pd.Series, model=None) -> pd.DataFrame:
        """执行因子发现.

        Args:
            X: 特征矩阵 (列为因子名)。
            y: 标签 (未来收益)。
            model: 已训练模型 (用于 SHAP/importance)。

        Returns:
            因子报告 DataFrame: [factor, lgbm_gain, shap_mean, ic, icir,
            composite_score, rank], 按 composite_score 降序。
        """
        if X.empty or len(y) == 0:
            raise ValueError("X / y 不能为空")
        X = X.copy()
        y = y.copy()
        # 对齐 & 去 NaN 标签
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        factor_names = list(X.columns)
        logger.info("因子发现: %d 因子 × %d 样本", len(factor_names), len(X))

        gain = self._compute_gain(X, y, model)
        shap_mean = self._compute_shap(X, y, model)

        ic_map: Dict[str, float] = {}
        icir_map: Dict[str, float] = {}
        for name in factor_names:
            ic_map[name] = self.compute_ic(X[name], y)
            icir_map[name] = self.compute_icir(X[name], y)

        report = pd.DataFrame(
            {
                "factor": factor_names,
                "lgbm_gain": gain,
                "shap_mean": shap_mean,
                "ic": [ic_map[n] for n in factor_names],
                "icir": [icir_map[n] for n in factor_names],
            }
        )

        report["composite_score"] = (
            W_GAIN * _minmax_norm(report["lgbm_gain"])
            + W_SHAP * _minmax_norm(report["shap_mean"])
            + W_IC * _minmax_norm(report["ic"])
            + W_ICIR * _minmax_norm(report["icir"])
        )
        report = report.sort_values("composite_score", ascending=False)
        report["rank"] = np.arange(1, len(report) + 1)
        report = report.reset_index(drop=True)
        logger.info(
            "因子发现完成: Top-1 = %s (score=%.4f)",
            report.iloc[0]["factor"],
            report.iloc[0]["composite_score"],
        )
        return report

    def get_top_factors(self, report: pd.DataFrame, top_k: int = 10) -> List[str]:
        """取 Top-K 因子 (FORCE_TOP_FACTORS 强制保留).

        Args:
            report: run() 输出的因子报告。
            top_k: 选取数量。

        Returns:
            因子名列表 (Top-K 按分降序, 强制因子补充在末尾)。
        """
        if report.empty:
            return list(FORCE_TOP_FACTORS)
        ordered = report.sort_values("composite_score", ascending=False)
        top = ordered["factor"].head(top_k).tolist()
        for forced in FORCE_TOP_FACTORS:
            if forced not in top and forced in set(report["factor"]):
                top.append(forced)
        return top
