# -*- coding: utf-8 -*-
"""盘后自动增量训练 (P10.5, ARCH §5.9.4).

每日 16:00 微调: 增量数据 fine-tune → OOS 验证 → 优于旧模型才替换,
否则保留旧模型 + 告警 (回滚保护)。
Pipeline 2 模型走 Pipeline2WarmStarter (warm-start 微调)。

sklearn 模型微调策略: 有 partial_fit 用 partial_fit (纯增量);
否则在 旧数据+新数据 上暖重启 refit (deepcopy 候选, 不动旧模型)。
"""

import copy
import logging
from typing import Optional

import numpy as np

from app.models.model_selector import spearman_ic

logger = logging.getLogger(__name__)


class DailyTrainer:
    """盘后增量训练器.

    Attributes:
        last_oos_ic_: 最近一次 validate_oos 的 IC 值。
    """

    def __init__(self, config: dict = None) -> None:
        """初始化.

        Args:
            config: 配置, 支持 keys:
                ic_threshold (float): 替换门槛 (默认 0.03, ARCH §5.12.3.C);
                fine_tune_epochs (int): 微调 epoch (默认 10);
                fine_tune_lr (float): 微调学习率 (默认 1e-5);
                以及 run_daily 数据的默认键 (model/X_old/y_old/X_new/
                y_new/X_oos/y_oos)。
        """
        self.config = config or {}
        self.last_oos_ic_ = None

    def run_daily(self, model=None, X_old=None, y_old=None,
                  X_new=None, y_new=None, X_oos=None, y_oos=None,
                  ic_threshold: Optional[float] = None) -> dict:
        """每日训练流程: fine_tune → validate_oos → 替换 or 保留+告警.

        数据优先取参数, 缺省回落到 config 同名键。

        Args:
            model: 当前生效模型 (ZooModel 或 sklearn 估计器)。
            X_old / y_old: 历史训练数据 (用于暖重启 refit)。
            X_new / y_new: 当日增量数据。
            X_oos / y_oos: OOS 验证集。
            ic_threshold: 替换门槛, 默认取 config["ic_threshold"] 或 0.03。

        Returns:
            {replaced, oos_pass, old_ic, new_ic, warning, model} —
            model 字段为最终生效模型 (替换成功为新模型, 否则为旧模型)。

        Raises:
            ValueError: 必需数据缺失。
        """
        cfg = self.config
        model = model if model is not None else cfg.get("model")
        X_old = X_old if X_old is not None else cfg.get("X_old")
        y_old = y_old if y_old is not None else cfg.get("y_old")
        X_new = X_new if X_new is not None else cfg.get("X_new")
        y_new = y_new if y_new is not None else cfg.get("y_new")
        X_oos = X_oos if X_oos is not None else cfg.get("X_oos")
        y_oos = y_oos if y_oos is not None else cfg.get("y_oos")
        threshold = float(ic_threshold if ic_threshold is not None
                          else cfg.get("ic_threshold", 0.03))
        missing = [k for k, v in [("model", model), ("X_new", X_new),
                                  ("y_new", y_new), ("X_oos", X_oos),
                                  ("y_oos", y_oos)] if v is None]
        if missing:
            raise ValueError(f"run_daily 缺少必需数据: {missing}")

        old_ic = spearman_ic(model.predict(X_oos), y_oos)
        candidate = self.fine_tune(
            model, X_new, y_new,
            epochs=int(cfg.get("fine_tune_epochs", 10)),
            lr=float(cfg.get("fine_tune_lr", 1e-5)),
            X_old=X_old, y_old=y_old)
        new_ic = spearman_ic(candidate.predict(X_oos), y_oos)
        oos_pass = self.validate_oos(candidate, X_oos, y_oos,
                                     ic_threshold=threshold)

        replaced = bool(oos_pass and new_ic >= old_ic)
        warning = None
        if not replaced:
            warning = (f"新模型未通过替换门槛 (oos_pass={oos_pass}, "
                       f"new_ic={new_ic:.4f} vs old_ic={old_ic:.4f}, "
                       f"threshold={threshold}), 保留旧模型")
            logger.warning(warning)
        else:
            logger.info("模型替换: old_ic=%.4f → new_ic=%.4f", old_ic, new_ic)
        return {"replaced": replaced, "oos_pass": bool(oos_pass),
                "old_ic": old_ic, "new_ic": new_ic,
                "warning": warning,
                "model": candidate if replaced else model}

    def fine_tune(self, model, X_new, y_new, epochs: int = 10,
                  lr: float = 1e-5, X_old=None, y_old=None):
        """微调 (小学习率, 少 epoch).

        在 deepcopy 的候选模型上操作, 旧模型不被污染 (回滚保护)。
        sklearn: 无旧数据且估计器支持 partial_fit → 纯增量;
        否则合并 旧+新 数据暖重启 refit。时序 torch 模型透传 epochs/lr。

        Args:
            model: ZooModel 或 sklearn 估计器。
            X_new: 增量特征。
            y_new: 增量标签。
            epochs: 时序模型微调 epoch 数。
            lr: 时序模型微调学习率。
            X_old: 历史特征 (可选, 提供则合并 refit)。
            y_old: 历史标签 (可选)。

        Returns:
            微调后的候选模型 (新对象)。
        """
        candidate = copy.deepcopy(model)
        estimator = getattr(candidate, "estimator", candidate)

        if X_old is None and hasattr(estimator, "partial_fit"):
            X_inc = np.nan_to_num(np.asarray(X_new, dtype=np.float64),
                                  nan=0.0, posinf=0.0, neginf=0.0)
            if X_inc.ndim == 3:
                X_inc = X_inc.reshape(X_inc.shape[0], -1)
            y_inc = np.nan_to_num(np.asarray(y_new, dtype=np.float64).ravel(),
                                  nan=0.0, posinf=0.0, neginf=0.0)
            estimator.partial_fit(X_inc, y_inc)
            logger.info("fine_tune: partial_fit 增量 (%d 样本)", len(y_inc))
            return candidate

        if X_old is not None and y_old is not None:
            X_full = np.concatenate(
                [np.asarray(X_old, dtype=np.float64),
                 np.asarray(X_new, dtype=np.float64)], axis=0)
            y_full = np.concatenate(
                [np.asarray(y_old, dtype=np.float64).ravel(),
                 np.asarray(y_new, dtype=np.float64).ravel()])
        else:
            X_full = np.asarray(X_new, dtype=np.float64)
            y_full = np.asarray(y_new, dtype=np.float64).ravel()
        try:
            candidate.fit(X_full, y_full, epochs=epochs, lr=lr)
        except TypeError:
            candidate.fit(X_full, y_full)
        logger.info("fine_tune: 暖重启 refit (%d 样本)", len(y_full))
        return candidate

    def validate_oos(self, model, X_oos, y_oos,
                     ic_threshold: float = 0.03) -> bool:
        """OOS 验证: IC > 0.03 才允许替换 (ARCH §5.12.3.C).

        Args:
            model: 候选模型 (需有 predict)。
            X_oos: OOS 特征。
            y_oos: OOS 标签。
            ic_threshold: IC 门槛。

        Returns:
            True = 通过 (IC > ic_threshold)。
        """
        ic = spearman_ic(model.predict(X_oos), y_oos)
        self.last_oos_ic_ = ic
        passed = ic > ic_threshold
        logger.info("OOS 验证: IC=%.4f threshold=%.3f → %s",
                    ic, ic_threshold, "PASS" if passed else "FAIL")
        return passed
