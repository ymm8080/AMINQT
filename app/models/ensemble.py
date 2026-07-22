# -*- coding: utf-8 -*-
"""多模型融合 (P10.5, ARCH §5.9.3).

融合策略: rank_mean / score_mean / weighted_score / stacking。
rank_mean / score_mean / weighted_score 无需训练; stacking 需先 fit_stacking。
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

ENSEMBLE_STRATEGIES = ["rank_mean", "score_mean", "weighted_score", "stacking"]


class ModelEnsemble:
    """多模型融合器.

    Attributes:
        strategy: 当前融合策略。
        weights: weighted_score 权重 {model_name: weight}。
        meta_learner_: stacking 元学习器 (fit_stacking 后可用)。
        member_names_: stacking 成员顺序 (保证 combine 列序一致)。
    """

    def __init__(self, strategy: str = "rank_mean",
                 weights: Dict[str, float] = None) -> None:
        """初始化融合策略.

        Args:
            strategy: ENSEMBLE_STRATEGIES 之一。
            weights: weighted_score 策略的模型权重 (自适应, 不写死)。

        Raises:
            ValueError: 未知融合策略。
        """
        if strategy not in ENSEMBLE_STRATEGIES:
            raise ValueError(
                f"未知融合策略: {strategy}. 可选: {ENSEMBLE_STRATEGIES}")
        self.strategy = strategy
        self.weights = dict(weights or {})
        self.meta_learner_ = None
        self.member_names_ = None

    @staticmethod
    def _validate(predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """校验并清洗预测字典.

        Args:
            predictions: {model_name: (N,) 得分}。

        Returns:
            清洗后的 {model_name: (N,)}。

        Raises:
            ValueError: 空字典或长度不一致。
        """
        if not predictions:
            raise ValueError("predictions 为空, 无法融合")
        cleaned = {
            name: np.nan_to_num(np.asarray(p, dtype=np.float64).ravel(),
                                nan=0.0, posinf=0.0, neginf=0.0)
            for name, p in predictions.items()
        }
        lengths = {len(p) for p in cleaned.values()}
        if len(lengths) != 1:
            raise ValueError(f"各模型预测长度不一致: {lengths}")
        return cleaned

    def combine(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """融合多模型预测.

        Args:
            predictions: {model_name: (N,) 得分}。

        Returns:
            (N,) 融合得分。

        Raises:
            RuntimeError: stacking 策略但未先 fit_stacking。
            ValueError: 输入非法或 stacking 成员缺失。
        """
        preds = self._validate(predictions)

        if self.strategy == "rank_mean":
            ranks = [
                pd.Series(p).rank(method="average").to_numpy()
                for p in preds.values()
            ]
            return np.mean(np.column_stack(ranks), axis=1)

        if self.strategy == "score_mean":
            return np.mean(np.column_stack(list(preds.values())), axis=1)

        if self.strategy == "weighted_score":
            names = list(preds)
            # 未配置权重的模型回退为等权; 权重归一化到 sum=1
            raw = np.array([self.weights.get(n, 1.0) for n in names],
                           dtype=np.float64)
            raw = np.clip(raw, 0.0, None)
            if raw.sum() <= 0:
                logger.warning("weighted_score 权重全为 0, 回退等权")
                raw = np.ones(len(names))
            w = raw / raw.sum()
            return np.column_stack([preds[n] for n in names]) @ w

        # stacking
        if self.meta_learner_ is None:
            raise RuntimeError(
                "stacking 策略需先调用 fit_stacking 训练元学习器")
        missing = [n for n in self.member_names_ if n not in preds]
        if missing:
            raise ValueError(f"stacking 缺少成员模型预测: {missing}")
        X_meta = np.column_stack([preds[n] for n in self.member_names_])
        return np.asarray(self.meta_learner_.predict(X_meta),
                          dtype=np.float64).ravel()

    def fit_stacking(self, predictions: Dict[str, np.ndarray],
                     y: np.ndarray) -> None:
        """stacking 策略: 训练元学习器 (sklearn LinearRegression).

        Args:
            predictions: {model_name: (N,) 各模型在验证集上的得分}。
            y: (N,) 真实标签。

        Raises:
            ValueError: y 长度与预测不一致。
        """
        preds = self._validate(predictions)
        y = np.nan_to_num(np.asarray(y, dtype=np.float64).ravel(),
                          nan=0.0, posinf=0.0, neginf=0.0)
        n = len(next(iter(preds.values())))
        if len(y) != n:
            raise ValueError(f"y 长度 {len(y)} 与预测长度 {n} 不一致")
        self.member_names_ = list(preds)
        X_meta = np.column_stack([preds[name] for name in self.member_names_])
        self.meta_learner_ = LinearRegression()
        self.meta_learner_.fit(X_meta, y)
        logger.info("stacking 元学习器已训练: members=%s, coef=%s",
                    self.member_names_,
                    np.round(self.meta_learner_.coef_, 4).tolist())
