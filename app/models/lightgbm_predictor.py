# -*- coding: utf-8 -*-
"""LightGBM 推理器 (P2, ARCH §9.3).

加载 LightGBM pkl 模型做基线推理 + 因子重要性输出 (供因子发现)。
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class LightGBMPredictor:
    """LightGBM 基线推理器."""

    def __init__(self, model_path: str = None) -> None:
        raise NotImplementedError("P2 待建")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """推理 (N, 特征数) → (N,) 得分."""
        raise NotImplementedError("P2 待建")

    def feature_importance(self) -> dict:
        """返回 {factor_name: gain} 供 FactorDiscovery."""
        raise NotImplementedError("P2 待建")
