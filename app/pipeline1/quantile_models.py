"""
E1 分位数回归五模型 + E2 痛苦预警模型 (PIPELINE1_V3.8 §四/§七, 检查清单 #65-#68)
================================================================================
[E1] quantiles = [0.10, 0.25, 0.50, 0.75, 0.90] 五分位 LightGBM (label_pm_{k}d_net 各分位, k=3/5):
  - 单调性后处理 (必须): 每样本强制 q10<=q25<=q50<=q75<=q90, 违例做保序回归投影
  - uncertainty_width = pred_q90 - pred_q10 (仓位阻尼输入)
  - 纪律: 分位数用于仓位阻尼与风险提示, 严禁用 pred_q10>0 做买入闸门
    (次日收益 q10 几乎必为负, 闸门=系统停摆)
  - 超参: 沿用回归超参, 不单独搜索 (避免调参维度爆炸)
[E2] 痛苦预警模型: label_pain (3日浮亏>5%) 分类 → pain_prob;
  pain_prob>30% 的候选即使预期收益为正也降仓50%或剔除 (阈值入季度优化窗口).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_COLS = ("pred_q10", "pred_q25", "pred_q50", "pred_q75", "pred_q90")
PAIN_DEMOTE_THRESHOLD = 0.30  # E2: pain_prob>30% → 降仓50%或剔除 (季度优化窗口)
# 分位头最小树保底 (2026-08-14): 与 dual_track_trainer.CLS_MIN_TREES 同机制 —
# es 窗过平 → 早停第 1 树 → 常数分位 (M20260812 main 3d q50 nunique=1, 闸门 q50>0 拦掉 100% 票)
QUANTILE_MIN_TREES = 30


class QuantileModelSet:
    """E1 分位数五模型 (同一特征空间, label_pm_{k}d_net 各分位, k=3/5)."""

    def __init__(self, base_params: dict | None = None):
        self.base_params = dict(base_params or {})
        self.base_params.setdefault("random_state", 42)
        self.models: dict[float, object] = {}
        self._iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

    # ---------------- 训练 ----------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        es_patience: int = 100,
    ) -> QuantileModelSet:
        """训练五分位模型 (objective='quantile', alpha=q). 早停 patience=100 (V3.8 §2.1)."""
        import lightgbm as lgb

        for q in QUANTILES:
            params = {
                **self.base_params,
                "objective": "quantile",
                "alpha": q,
            }
            params.pop("importance_type", None)
            model = lgb.LGBMRegressor(**params)
            callbacks = (
                [lgb.early_stopping(es_patience, verbose=False)]
                if eval_set is not None
                else None
            )
            model.fit(
                X,
                y,
                sample_weight=sample_weight,
                eval_set=[eval_set] if eval_set is not None else None,
                callbacks=callbacks,
            )
            if eval_set is not None:
                bi = getattr(model, "best_iteration_", None)
                if bi is not None and bi < QUANTILE_MIN_TREES:
                    logger.warning(
                        "E1 分位数 q=%.2f 早停仅 %d 树 (< %d), 重训 %d 树保底",
                        q,
                        bi,
                        QUANTILE_MIN_TREES,
                        QUANTILE_MIN_TREES,
                    )
                    params = {**params, "n_estimators": QUANTILE_MIN_TREES}
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X, y, sample_weight=sample_weight)
            self.models[q] = model
            logger.info("E1 分位数模型 q=%.2f 训练完成, 样本 %d", q, len(X))
        return self

    # ---------------- 推理 (含单调性后处理) ----------------
    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """返回 DataFrame[pred_q10..pred_q90, uncertainty_width], 行=样本.

        单调性后处理 (必须): 对每样本五分位序列做保序回归 (isotonic) 投影,
        强制 q10<=q25<=q50<=q75<=q90.
        """
        assert len(self.models) == len(QUANTILES), "QuantileModelSet 未 fit"
        qs = np.column_stack([self.models[q].predict(X) for q in QUANTILES])
        x = np.arange(len(QUANTILES))
        mono = np.apply_along_axis(lambda r: self._iso.fit_transform(x, r), 1, qs)
        out = pd.DataFrame(mono, columns=QUANTILE_COLS)
        out["uncertainty_width"] = out["pred_q90"] - out["pred_q10"]
        return out


class PainModel:
    """E2 痛苦预警模型: label_pain 分类 → pain_prob."""

    def __init__(self, base_params: dict | None = None):
        self.base_params = dict(base_params or {})
        self.base_params.setdefault("random_state", 42)
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        es_patience: int = 100,
    ) -> PainModel:
        import lightgbm as lgb

        params = {**self.base_params, "objective": "binary"}
        self.model = lgb.LGBMClassifier(**params)
        callbacks = (
            [lgb.early_stopping(es_patience, verbose=False)]
            if eval_set is not None
            else None
        )
        self.model.fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=[eval_set] if eval_set is not None else None,
            callbacks=callbacks,
        )
        logger.info("E2 痛苦预警模型训练完成, 样本 %d, 正例率 %.3f", len(X), np.mean(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """pain_prob ∈ [0,1] — 3日内浮亏超5%的预测概率."""
        assert self.model is not None, "PainModel 未 fit"
        return self.model.predict_proba(X)[:, 1]

    # ---------------- E2 降仓裁决 ----------------
    @staticmethod
    def pain_adjustment(
        pain_prob: float, threshold: float = PAIN_DEMOTE_THRESHOLD
    ) -> float:
        """pain_prob > 30% → 仓位乘 0.5 (降仓50%); 否则 1.0. (剔除交清单层裁决)"""
        return 0.5 if pain_prob > threshold else 1.0
