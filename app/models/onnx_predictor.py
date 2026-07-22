# -*- coding: utf-8 -*-
"""ONNX 推理器 (P2, ARCH §2/§9.3).

加载云端导出的 ONNX 模型做本地推理 (跨平台, 无需训练框架)。
按 Universe 路由对应模型文件。
"""

from __future__ import annotations

import logging
import os

import numpy as np

from app.core.universe_manager import Universe

logger = logging.getLogger(__name__)


class ONNXPredictor:
    """ONNX 模型推理器.

    Args:
        model_dir: ONNX 模型目录, 文件名约定 {universe.value}.onnx
    """

    def __init__(self, model_dir: str = "app/models/trained") -> None:
        self.model_dir = model_dir
        self._sessions: dict = {}

    def load(self, universe: Universe) -> None:
        """按 Universe 加载对应 ONNX 模型."""
        import onnxruntime as ort

        path = os.path.join(self.model_dir, f"{universe.value}.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX 模型不存在: {path}")
        self._sessions[universe] = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        logger.info("ONNX 模型加载: %s", path)

    def predict(self, X: np.ndarray, universe: Universe) -> np.ndarray:
        """推理.

        Args:
            X: (N, 20, 85) 特征矩阵。
            universe: 路由模型。

        Returns:
            (N,) 得分 (上涨概率)。
        """
        if universe not in self._sessions:
            self.load(universe)
        sess = self._sessions[universe]
        X = np.nan_to_num(np.asarray(X, dtype=np.float32))     # NaN→0
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: X})[0]
        return np.asarray(out).reshape(-1)

    def verify_against_snapshot(
        self, X: np.ndarray, snapshot_scores: np.ndarray, tol: float = 1e-4,
        universe: Universe | None = None,
    ) -> bool:
        """本地二次校验: 推理结果 vs 云端特征快照 (ARCH §2)."""
        universe = universe or next(iter(self._sessions))
        local = self.predict(X, universe)
        return bool(np.allclose(local, snapshot_scores, atol=tol))
