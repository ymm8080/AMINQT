# -*- coding: utf-8 -*-
"""ONNX 推理器 (P2, ARCH §2/§9.3).

加载云端导出的 ONNX 模型做本地推理 (跨平台, 无需训练框架)。
按 Universe 路由对应模型文件。
"""

import logging

import numpy as np

from app.core.universe_manager import Universe

logger = logging.getLogger(__name__)


class ONNXPredictor:
    """ONNX 模型推理器."""

    def __init__(self, model_dir: str = "app/models/trained") -> None:
        raise NotImplementedError("P2 待建")

    def load(self, universe: Universe) -> None:
        """按 Universe 加载对应 ONNX 模型."""
        raise NotImplementedError("P2 待建")

    def predict(self, X: np.ndarray, universe: Universe) -> np.ndarray:
        """推理.

        Args:
            X: (N, 20, 85) 特征矩阵。
            universe: 路由模型。

        Returns:
            (N,) 得分 (上涨概率)。
        """
        raise NotImplementedError("P2 待建")

    def verify_against_snapshot(
        self, X: np.ndarray, snapshot_scores: np.ndarray, tol: float = 1e-4
    ) -> bool:
        """本地二次校验: 推理结果 vs 云端特征快照 (ARCH §2)."""
        raise NotImplementedError("P2 待建")
