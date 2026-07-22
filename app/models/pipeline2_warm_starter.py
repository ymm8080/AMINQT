# -*- coding: utf-8 -*-
"""Pipeline 2 Warm-Start (P7.5, ARCH §3.4, DESIGN_V1 §9 #3).

交易 Pipeline 模型不从头训练:
  ① 加载 Pipeline 1 日线模型参数 θ_p1 作为 Baseline (初始价格预测)
  ② 加近 3 个月 (~63 交易日) 五分钟数据微调重训练
  ③ OOS 验证通过 (IC > 0.03) 才替换; 失败保留旧模型 + 告警

特征: 85 维日线 + 25 维五分钟 = 110 维综合特征。
数据装配接受预计算特征帧 (日线因子帧 + 五分钟因子帧), 不触碰数据管线。
"""

import copy
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from app.models.model_selector import spearman_ic
from app.models.model_zoo import load_model

logger = logging.getLogger(__name__)

DAILY_FEATURE_DIM = 85
FIVEMIN_FEATURE_DIM = 25


class Pipeline2WarmStarter:
    """Pipeline 2 warm-start 训练器.

    Attributes:
        baseline_: 已加载的 Pipeline 1 模型 (θ_p1)。
        dataset_: build_5min_dataset 产出的 (X, y, index)。
    """

    def __init__(self, pipeline1_model_path: str,
                 retrain_window_days: int = 63) -> None:
        """初始化.

        Args:
            pipeline1_model_path: Pipeline 1 日线模型路径 (θ_p1)。
            retrain_window_days: 重训练数据窗口 (默认 63 交易日 ≈ 3 个月)。
        """
        self.pipeline1_model_path = str(pipeline1_model_path)
        self.retrain_window_days = int(retrain_window_days)
        self.baseline_ = None
        self.dataset_ = None

    def load_baseline(self) -> None:
        """加载 θ_p1 作为初始化参数 (迁移学习).

        支持 ZooModel pickle / torch .pt (经 model_zoo.load_model)。

        Raises:
            FileNotFoundError: 模型文件不存在。
        """
        if not os.path.exists(self.pipeline1_model_path):
            raise FileNotFoundError(
                f"Pipeline 1 模型不存在: {self.pipeline1_model_path}")
        self.baseline_ = load_model(self.pipeline1_model_path)
        logger.info("θ_p1 baseline 已加载 ← %s (name=%s)",
                    self.pipeline1_model_path, self.baseline_.name)

    def build_5min_dataset(self, symbols: list,
                           daily_features: Optional[pd.DataFrame] = None,
                           fivemin_features: Optional[pd.DataFrame] = None,
                           labels: Optional[np.ndarray] = None) -> tuple:
        """构建近 3 月五分钟训练集 (110 维 = 85 日线 + 25 五分钟).

        Args:
            symbols: 股票代码列表 (日志/校验用)。
            daily_features: 预计算日线因子帧, (N, 85), index 与五分钟帧对齐。
            fivemin_features: 预计算五分钟因子帧, (N, 25)。
            labels: (N,) 未来 5 根 K 线收益标签 (可选)。

        Returns:
            (X, y): X 为 (N, 110) nan_to_num 后的特征矩阵;
            y 为 (N,) 标签 (未提供时为 None)。

        Raises:
            RuntimeError: 预计算特征帧未提供 (数据管线不在本模块职责内)。
        """
        if daily_features is None or fivemin_features is None:
            raise RuntimeError(
                "build_5min_dataset 需要预计算特征帧: daily_features "
                "(N,85) 与 fivemin_features (N,25)。数据装配管线 "
                "(FactorEngine/IntradayFactorEngine) 不在本模块职责内。")
        if daily_features.shape[1] != DAILY_FEATURE_DIM:
            logger.warning("日线特征维度 %d ≠ %d", daily_features.shape[1],
                           DAILY_FEATURE_DIM)
        if fivemin_features.shape[1] != FIVEMIN_FEATURE_DIM:
            logger.warning("五分钟特征维度 %d ≠ %d",
                           fivemin_features.shape[1], FIVEMIN_FEATURE_DIM)
        combined = daily_features.join(fivemin_features, how="inner",
                                       lsuffix="_d", rsuffix="_m")
        X = np.nan_to_num(combined.to_numpy(dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        y = None
        if labels is not None:
            y = np.nan_to_num(
                pd.Series(np.asarray(labels, dtype=np.float64).ravel())
                .reindex(combined.index if len(labels) == len(combined.index)
                         else None).to_numpy()
                if len(labels) == len(combined.index)
                else np.asarray(labels, dtype=np.float64).ravel(),
                nan=0.0, posinf=0.0, neginf=0.0)
        self.dataset_ = (X, y, combined.index)
        logger.info("五分钟训练集构建完成: symbols=%d, X=%s, y=%s",
                    len(symbols), X.shape,
                    None if y is None else y.shape)
        return X, y

    def run(self, X_train=None, y_train=None, X_oos=None, y_oos=None,
            ic_threshold: float = 0.03,
            train_fraction: float = 0.8) -> dict:
        """完整 warm-start 流程.

        1. load_baseline() (若未加载)
        2. 数据: 显式传入, 否则用 build_5min_dataset 缓存按时间序 80/20 切分
        3. 以 θ_p1 为初始值 fine-tune (sklearn: partial_fit 或 refit;
           torch: 加载权重后小 lr 少 epoch, 由 ZooModel.fit 处理)
        4. OOS 验证: IC > 0.03 且不劣于 baseline 才替换; 否则保留 + 告警

        Args:
            X_train / y_train: 微调训练集 (可选)。
            X_oos / y_oos: OOS 验证集 (可选)。
            ic_threshold: OOS IC 门槛 (默认 0.03)。
            train_fraction: 使用缓存数据集时的训练段比例。

        Returns:
            {oos_pass, oos_ic, baseline_ic, replaced, baseline_source,
             warning, model}。

        Raises:
            ValueError: 无可用训练/验证数据。
        """
        if self.baseline_ is None:
            self.load_baseline()

        if X_train is None or X_oos is None:
            if self.dataset_ is None or self.dataset_[1] is None:
                raise ValueError(
                    "run 需要显式传入 X_train/y_train/X_oos/y_oos, "
                    "或先调用 build_5min_dataset(..., labels=...)")
            X_all, y_all, _ = self.dataset_
            cut = max(1, int(len(X_all) * train_fraction))
            X_train, y_train = X_all[:cut], y_all[:cut]
            X_oos, y_oos = X_all[cut:], y_all[cut:]
            if len(X_oos) == 0:
                raise ValueError("数据集过小, 切分后 OOS 段为空")

        baseline_ic = spearman_ic(self.baseline_.predict(X_oos), y_oos)

        candidate = copy.deepcopy(self.baseline_)
        estimator = getattr(candidate, "estimator", candidate)
        X_fit = np.nan_to_num(np.asarray(X_train, dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
        y_fit = np.nan_to_num(np.asarray(y_train, dtype=np.float64).ravel(),
                              nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(estimator, "partial_fit"):
            if X_fit.ndim == 3:
                X_fit = X_fit.reshape(X_fit.shape[0], -1)
            estimator.partial_fit(X_fit, y_fit)
            logger.info("warm-start fine-tune: partial_fit (%d 样本)",
                        len(y_fit))
        else:
            try:
                candidate.fit(X_fit, y_fit, epochs=10, lr=1e-5)
            except TypeError:
                candidate.fit(X_fit, y_fit)
            logger.info("warm-start fine-tune: refit (%d 样本)", len(y_fit))

        oos_ic = spearman_ic(candidate.predict(X_oos), y_oos)
        oos_pass = bool(oos_ic > ic_threshold)
        replaced = bool(oos_pass and oos_ic >= baseline_ic)
        warning = None
        if not replaced:
            warning = (f"warm-start 微调未达替换条件 (oos_pass={oos_pass}, "
                       f"oos_ic={oos_ic:.4f} vs baseline_ic="
                       f"{baseline_ic:.4f}, threshold={ic_threshold}), "
                       "保留 Pipeline 1 模型")
            logger.warning(warning)
        else:
            logger.info("warm-start 替换: baseline_ic=%.4f → oos_ic=%.4f",
                        baseline_ic, oos_ic)
        return {"oos_pass": oos_pass, "oos_ic": oos_ic,
                "baseline_ic": baseline_ic, "replaced": replaced,
                "baseline_source": self.pipeline1_model_path,
                "warning": warning,
                "model": candidate if replaced else self.baseline_}
