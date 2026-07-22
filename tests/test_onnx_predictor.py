# -*- coding: utf-8 -*-
"""P12: ONNXPredictor 测试 (加载 / 推理一致性 / NaN 处理 / Universe 路由)."""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from app.core.universe_manager import Universe  # noqa: E402
from app.models.onnx_predictor import ONNXPredictor  # noqa: E402


def _make_tiny_onnx(path: str) -> np.ndarray:
    """构造最小 ONNX 线性模型: y = X_flat @ w (input (N,20,2) → flatten 40 → (N,))."""
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.1, 40).astype(np.float32).reshape(40, 1)
    inp = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, ["N", 20, 2]
    )
    outp = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, ["N", 1]
    )
    shape_c = onnx.helper.make_tensor("shape", onnx.TensorProto.INT64, [2], [-1, 40])
    w_t = onnx.helper.make_tensor("w", onnx.TensorProto.FLOAT, [40, 1], w.flatten())
    reshape = onnx.helper.make_node("Reshape", ["input", "shape"], ["flat"])
    matmul = onnx.helper.make_node("MatMul", ["flat", "w"], ["output"])
    graph = onnx.helper.make_graph(
        [reshape, matmul], "tiny", [inp], [outp], [shape_c, w_t]
    )
    model = (
        onnx.helper.make_graph
        if False
        else onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 14)]
        )
    )
    model.ir_version = 8
    onnx.save(model, path)
    return w


@pytest.fixture()
def model_dir(tmp_path):
    w = _make_tiny_onnx(str(tmp_path / "main_board.onnx"))
    _make_tiny_onnx(str(tmp_path / "growth_boards.onnx"))
    return str(tmp_path), w


class TestONNXPredictor:
    def test_load_and_predict(self, model_dir):
        d, w = model_dir
        p = ONNXPredictor(model_dir=d)
        p.load(Universe.MAIN_BOARD)
        X = np.random.default_rng(1).normal(size=(3, 20, 2)).astype(np.float32)
        out = p.predict(X, Universe.MAIN_BOARD)
        assert out.shape == (3,)
        expected = X.reshape(3, 40) @ w
        np.testing.assert_allclose(out, expected.flatten(), atol=1e-5)

    def test_nan_handling(self, model_dir):
        d, _ = model_dir
        p = ONNXPredictor(model_dir=d)
        X = np.full((1, 20, 2), np.nan, dtype=np.float32)
        out = p.predict(X, Universe.MAIN_BOARD)  # NaN→0, 不崩溃
        assert np.isfinite(out).all()

    def test_universe_routing(self, model_dir):
        d, _ = model_dir
        p = ONNXPredictor(model_dir=d)
        X = np.ones((1, 20, 2), dtype=np.float32)
        p.predict(X, Universe.MAIN_BOARD)
        p.predict(X, Universe.GROWTH_BOARDS)
        assert set(p._sessions) == {Universe.MAIN_BOARD, Universe.GROWTH_BOARDS}

    def test_verify_against_snapshot(self, model_dir):
        d, w = model_dir
        p = ONNXPredictor(model_dir=d)
        X = np.random.default_rng(2).normal(size=(2, 20, 2)).astype(np.float32)
        snapshot = (X.reshape(2, 40) @ w).flatten()
        assert p.verify_against_snapshot(X, snapshot, universe=Universe.MAIN_BOARD)
        assert not p.verify_against_snapshot(
            X, snapshot + 1.0, universe=Universe.MAIN_BOARD
        )

    def test_missing_model_raises(self, tmp_path):
        p = ONNXPredictor(model_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            p.load(Universe.MAIN_BOARD)
