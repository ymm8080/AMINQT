# -*- coding: utf-8 -*-
"""模型库 (P10.5, ARCH §5.9.1 — 10 种模型).

LightGBM / XGBoost / CatBoost / RF / LSTM / GRU / Transformer / TCN / MLP / Ridge。
统一接口: fit / predict / save / load。
时序模型输入 (N, 20, 85); 非时序模型输入展平 (N, 1700)。

环境约束: torch / lightgbm / xgboost / catboost 不在基础环境, 相关导入全部
惰性化 (函数内 import), 缺失时抛出带安装提示的 RuntimeError。
模块级仅允许 numpy / pandas / sklearn / stdlib。
"""

import importlib.util
import logging
import pickle
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)

MODEL_NAMES = [
    "lightgbm", "xgboost", "catboost", "random_forest",
    "lstm", "gru", "transformer", "tcn", "mlp", "ridge",
]

SEQUENCE_MODELS = {"lstm", "gru", "transformer", "tcn"}  # 时序模型 (N,20,85)

# 重度第三方依赖 → (pip 包名, import 名)
_HEAVY_DEPS = {
    "lightgbm": ("lightgbm", "lightgbm"),
    "xgboost": ("xgboost", "xgboost"),
    "catboost": ("catboost", "catboost"),
    "lstm": ("torch", "torch"),
    "gru": ("torch", "torch"),
    "transformer": ("torch", "torch"),
    "tcn": ("torch", "torch"),
}

DEFAULT_PARAMS: Dict[str, dict] = {
    "lightgbm": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
                 "verbose": -1, "random_state": 42},
    "xgboost": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6,
                "random_state": 42},
    "catboost": {"iterations": 200, "learning_rate": 0.05, "depth": 6,
                 "verbose": False, "random_state": 42},
    "random_forest": {"n_estimators": 100, "n_jobs": -1, "random_state": 42},
    "lstm": {"hidden_size": 64, "num_layers": 2, "dropout": 0.2,
             "epochs": 10, "lr": 1e-3},
    "gru": {"hidden_size": 64, "num_layers": 2, "dropout": 0.2,
            "epochs": 10, "lr": 1e-3},
    "transformer": {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.1,
                    "epochs": 10, "lr": 1e-3},
    "tcn": {"channels": 64, "kernel_size": 3, "num_layers": 3, "dropout": 0.1,
            "epochs": 10, "lr": 1e-3},
    "mlp": {"hidden_layer_sizes": (128, 64), "max_iter": 300,
            "random_state": 42},
    "ridge": {"alpha": 1.0},
}


def dep_available(name: str) -> bool:
    """检查模型重度依赖是否已安装 (不触发 import).

    Args:
        name: MODEL_NAMES 之一。

    Returns:
        True 表示可构建; sklearn 系模型恒为 True。
    """
    if name not in _HEAVY_DEPS:
        return True
    _, import_name = _HEAVY_DEPS[name]
    return importlib.util.find_spec(import_name) is not None


def _require_dep(name: str) -> None:
    """缺失重度依赖时抛出带安装提示的 RuntimeError.

    Args:
        name: 模型名。

    Raises:
        RuntimeError: 依赖未安装。
    """
    if name in _HEAVY_DEPS and not dep_available(name):
        pip_name, _ = _HEAVY_DEPS[name]
        raise RuntimeError(
            f"模型 {name} 依赖 {pip_name}, 当前环境未安装。"
            f"请执行 `pip install {pip_name}` 后重试。"
        )


def _clean(y: np.ndarray) -> np.ndarray:
    """标签清洗: 转 float + 去 NaN/Inf.

    Args:
        y: 原始标签数组。

    Returns:
        清洗后的一维 float 数组。
    """
    return np.nan_to_num(np.asarray(y, dtype=np.float64).ravel(),
                         nan=0.0, posinf=0.0, neginf=0.0)


def _build_torch_net(name: str, input_size: int, params: dict):
    """惰性构建 torch 时序网络 (import torch 仅发生在此函数内).

    Args:
        name: lstm / gru / transformer / tcn。
        input_size: 每时间步特征数 F。
        params: 超参 (hidden_size / d_model / channels 等)。

    Returns:
        nn.Module, forward 输入 (batch, seq, F) → 输出 (batch,)。

    Raises:
        RuntimeError: torch 未安装。
    """
    _require_dep(name)
    import torch.nn as nn

    if name in ("lstm", "gru"):
        rnn_cls = nn.LSTM if name == "lstm" else nn.GRU

        class _RNNRegressor(nn.Module):
            """LSTM/GRU 回归器 (风格对齐 app/models/lstm_model.py)."""

            def __init__(self) -> None:
                super().__init__()
                self.rnn = rnn_cls(
                    input_size, params["hidden_size"], params["num_layers"],
                    batch_first=True, dropout=params["dropout"],
                )
                self.fc = nn.Linear(params["hidden_size"], 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        return _RNNRegressor()

    if name == "transformer":

        class _TransformerRegressor(nn.Module):
            """Transformer Encoder 回归器 (取最后时间步)."""

            def __init__(self) -> None:
                super().__init__()
                d_model = params["d_model"]
                self.proj = nn.Linear(input_size, d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=params["nhead"],
                    dim_feedforward=d_model * 4, dropout=params["dropout"],
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    layer, num_layers=params["num_layers"])
                self.fc = nn.Linear(d_model, 1)

            def forward(self, x):
                h = self.encoder(self.proj(x))
                return self.fc(h[:, -1, :]).squeeze(-1)

        return _TransformerRegressor()

    if name == "tcn":

        class _CausalConv1d(nn.Module):
            """因果卷积 (左侧 padding, 不看未来)."""

            def __init__(self, cin: int, cout: int, k: int,
                         dilation: int) -> None:
                super().__init__()
                self.pad = (k - 1) * dilation
                self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

            def forward(self, x):
                import torch.nn.functional as F

                return F.relu(self.conv(F.pad(x, (self.pad, 0))))

        class _TCNRegressor(nn.Module):
            """TCN 回归器 (膨胀因果卷积堆叠 + 最后时间步)."""

            def __init__(self) -> None:
                super().__init__()
                ch = params["channels"]
                self.convs = nn.ModuleList([
                    _CausalConv1d(input_size if i == 0 else ch, ch,
                                  params["kernel_size"], dilation=2 ** i)
                    for i in range(params["num_layers"])
                ])
                self.drop = nn.Dropout(params["dropout"])
                self.fc = nn.Linear(ch, 1)

            def forward(self, x):
                h = x.transpose(1, 2)  # (batch, F, seq)
                for conv in self.convs:
                    h = self.drop(conv(h))
                return self.fc(h[:, :, -1]).squeeze(-1)

        return _TCNRegressor()

    raise ValueError(f"未知时序模型: {name}")  # pragma: no cover


class ZooModel:
    """统一模型包装器: fit / predict / save / load.

    非时序模型内部持有 sklearn 风格估计器, 输入自动展平
    (N, 20, 85) → (N, 1700); 时序模型内部持有 torch nn.Module,
    保留 (N, 20, F) 时序结构。所有输入在进模型前 np.nan_to_num。
    """

    def __init__(self, name: str, estimator=None,
                 params: Optional[dict] = None,
                 input_size: Optional[int] = None) -> None:
        """初始化包装器.

        Args:
            name: MODEL_NAMES 之一 (或加载裸估计器时的 "custom")。
            estimator: 非时序模型的 sklearn 风格估计器 (时序模型为 None,
                fit 时按 input_size 惰性构建 torch 网络)。
            params: 超参字典。
            input_size: 时序模型每步特征数 (fit 时自动推断)。
        """
        self.name = name
        self.estimator = estimator
        self.params = params or {}
        self.input_size = input_size

    @property
    def is_sequence(self) -> bool:
        """是否时序模型 (保留 (N,20,F) 输入结构)."""
        return self.name in SEQUENCE_MODELS

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        """输入清洗 + 形状适配.

        Args:
            X: (N, 20, F) 或 (N, F') 特征。

        Returns:
            时序模型: (N, 20, F) float 数组; 非时序: (N, 20*F) 展平数组。
        """
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.is_sequence and X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, **kwargs) -> "ZooModel":
        """训练.

        Args:
            X: (N, 20, F) 或 (N, F') 特征。
            y: (N,) 标签。
            X_val: 验证特征 (保留给时序模型 early-stopping 扩展)。
            y_val: 验证标签 (同上)。
            **kwargs: 时序模型可覆盖 epochs / lr。

        Returns:
            self。

        Raises:
            RuntimeError: 时序模型但 torch 未安装, 或估计器缺失。
        """
        X = self._prepare(X)
        y = _clean(y)
        if self.is_sequence:
            self._fit_torch(X, y, **kwargs)
        else:
            if self.estimator is None:
                raise RuntimeError(
                    f"模型 {self.name} 缺少内部估计器, 无法 fit")
            self.estimator.fit(X, y)
        return self

    def _fit_torch(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """torch 时序模型训练循环 (全量 batch, 小 epoch).

        Args:
            X: (N, 20, F) 已清洗特征。
            y: (N,) 已清洗标签。
            **kwargs: epochs / lr 覆盖。
        """
        _require_dep(self.name)
        import torch

        self.input_size = X.shape[-1]
        if self.estimator is None:
            self.estimator = _build_torch_net(self.name, self.input_size,
                                              self.params)
        epochs = int(kwargs.get("epochs", self.params.get("epochs", 10)))
        lr = float(kwargs.get("lr", self.params.get("lr", 1e-3)))
        torch.manual_seed(42)
        opt = torch.optim.Adam(self.estimator.parameters(), lr=lr,
                               weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.estimator.train()
        for epoch in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.estimator(xt), yt)
            loss.backward()
            opt.step()
            logger.debug("%s epoch %d/%d loss=%.6f",
                         self.name, epoch + 1, epochs, loss.item())

    def predict(self, X: np.ndarray) -> np.ndarray:
        """推理.

        Args:
            X: (N, 20, F) 或 (N, F') 特征。

        Returns:
            (N,) 预测得分。

        Raises:
            RuntimeError: 模型尚未训练/加载。
        """
        if self.estimator is None:
            raise RuntimeError(
                f"模型 {self.name} 尚未训练或加载, 无法 predict")
        X = self._prepare(X)
        if self.is_sequence:
            _require_dep(self.name)
            import torch

            self.estimator.eval()
            with torch.no_grad():
                xt = torch.tensor(X, dtype=torch.float32)
                return self.estimator(xt).cpu().numpy().ravel()
        return np.asarray(self.estimator.predict(X),
                          dtype=np.float64).ravel()

    def save(self, path: str) -> None:
        """保存模型 (sklearn → pickle; torch → .pt).

        Args:
            path: 目标文件路径。

        Raises:
            RuntimeError: 模型尚未训练。
        """
        if self.estimator is None:
            raise RuntimeError(f"模型 {self.name} 尚未训练, 无法 save")
        if self.is_sequence:
            _require_dep(self.name)
            import torch

            torch.save({
                "format": "zoo_model",
                "name": self.name,
                "params": self.params,
                "input_size": self.input_size,
                "state_dict": self.estimator.state_dict(),
            }, path)
        else:
            with open(path, "wb") as f:
                pickle.dump({
                    "format": "zoo_model",
                    "name": self.name,
                    "params": self.params,
                    "estimator": self.estimator,
                }, f)
        logger.info("模型 %s 已保存 → %s", self.name, path)

    def load(self, path: str) -> "ZooModel":
        """从文件加载模型参数到当前实例.

        Args:
            path: save() 产出的文件路径。

        Returns:
            self。

        Raises:
            RuntimeError: 时序模型但 torch 未安装。
            ValueError: 文件格式不符。
        """
        if self.is_sequence or str(path).endswith(".pt"):
            _require_dep(self.name)
            import torch

            payload = torch.load(path, map_location="cpu",
                                 weights_only=False)
            if not isinstance(payload, dict) or "state_dict" not in payload:
                raise ValueError(f"{path} 不是 ZooModel torch 存档")
            self.params = payload.get("params", self.params)
            self.input_size = payload.get("input_size")
            self.estimator = _build_torch_net(self.name, self.input_size,
                                              self.params)
            self.estimator.load_state_dict(payload["state_dict"])
            self.estimator.eval()
        else:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and \
                    payload.get("format") == "zoo_model":
                self.name = payload["name"]
                self.params = payload.get("params", {})
                self.estimator = payload["estimator"]
            else:  # 裸 sklearn 估计器
                self.estimator = payload
        logger.info("模型 %s 已加载 ← %s", self.name, path)
        return self


def load_model(path: str) -> ZooModel:
    """从文件加载 ZooModel (自动识别 sklearn pickle / torch .pt).

    Args:
        path: save() 产出的文件路径。

    Returns:
        加载完成的 ZooModel。
    """
    if str(path).endswith(".pt"):
        _require_dep("lstm")  # 借时序名检查 torch 是否可用
        import torch

        name = torch.load(path, map_location="cpu",
                          weights_only=False).get("name", "lstm")
        return ZooModel(name).load(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and payload.get("format") == "zoo_model":
        model = ZooModel(payload["name"], estimator=payload["estimator"],
                         params=payload.get("params", {}))
    else:
        model = ZooModel("custom", estimator=payload)
    logger.info("ZooModel 已加载 ← %s (name=%s)", path, model.name)
    return model


def get_model(name: str, **params) -> ZooModel:
    """模型工厂.

    Args:
        name: MODEL_NAMES 之一。
        **params: 模型超参 (覆盖 DEFAULT_PARAMS)。

    Returns:
        统一接口模型实例 (fit/predict/save/load)。

    Raises:
        ValueError: 未知模型名。
        RuntimeError: 重度依赖 (lightgbm/xgboost/catboost/torch) 未安装。
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"未知模型: {name}. 可选: {MODEL_NAMES}")
    _require_dep(name)
    merged = dict(DEFAULT_PARAMS[name])
    merged.update(params)

    if name == "lightgbm":
        import lightgbm as lgb

        return ZooModel(name, lgb.LGBMRegressor(**merged), merged)
    if name == "xgboost":
        import xgboost as xgb

        return ZooModel(name, xgb.XGBRegressor(**merged), merged)
    if name == "catboost":
        import catboost as cb

        return ZooModel(name, cb.CatBoostRegressor(**merged), merged)
    if name == "random_forest":
        return ZooModel(name, RandomForestRegressor(**merged), merged)
    if name == "mlp":
        return ZooModel(name, MLPRegressor(**merged), merged)
    if name == "ridge":
        return ZooModel(name, Ridge(**merged), merged)
    # 时序模型: torch 网络在 fit 时按 input_size 惰性构建
    return ZooModel(name, estimator=None, params=merged)


def list_models() -> Dict[str, dict]:
    """返回 10 种模型元信息 {name: {type, input_shape, default_params}}."""
    meta = {
        "lightgbm": ("tree", "GBDT, 因子重要性+SHAP, 因子发现主力"),
        "xgboost": ("tree", "GBDT, 正则化强抗过拟合"),
        "catboost": ("tree", "GBDT, 类别特征友好"),
        "random_forest": ("tree", "Bagging, 稳定不易过拟合"),
        "lstm": ("sequence", "RNN 时序建模, 捕捉趋势拐点"),
        "gru": ("sequence", "LSTM 简化版, 训练更快"),
        "transformer": ("sequence", "自注意力捕捉长程依赖"),
        "tcn": ("sequence", "因果卷积, 并行训练快"),
        "mlp": ("dense", "最简深度学习, 快速验证下界"),
        "ridge": ("linear", "线性可解释, 极快基线"),
    }
    return {
        name: {
            "type": meta[name][0],
            "input_shape": "(N, 20, 85)" if name in SEQUENCE_MODELS
            else "(N, 1700) (由 (N,20,85) 展平)",
            "default_params": dict(DEFAULT_PARAMS[name]),
            "available": dep_available(name),
            "description": meta[name][1],
        }
        for name in MODEL_NAMES
    }
