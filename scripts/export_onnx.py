# -*- coding: utf-8 -*-
"""ONNX 导出脚本 (P2, ARCH §2).

训练好的 PyTorch 模型 → ONNX (跨平台本地推理) + 特征快照 (本地二次校验)。
"""

import logging

logger = logging.getLogger(__name__)


def export_onnx(
    model_path: str, out_path: str, input_shape: tuple = (1, 20, 85)
) -> None:
    """导出 ONNX.

    Args:
        model_path: 训练好的模型文件 (.pt)。
        out_path: 输出 .onnx 路径。
        input_shape: 输入形状 (batch, seq, features)。
    """
    raise NotImplementedError("P2 待建")


def export_feature_snapshot(X, scores, out_path: str) -> None:
    """导出特征快照 (本地二次校验对照)."""
    raise NotImplementedError("P2 待建")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit("用法: python scripts/export_onnx.py <model.pt> <out.onnx>")
