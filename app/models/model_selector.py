# -*- coding: utf-8 -*-
"""模型选择器 (P10.5, ARCH §5.9.2).

基于回测结果自动选最优模型或融合组合; 支持用户手工指定。
对比报告: 10 单模型 + 融合组合搜索。依赖缺失的模型自动跳过 (warning)。
"""

import itertools
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from app.models.ensemble import ENSEMBLE_STRATEGIES, ModelEnsemble
from app.models.model_zoo import MODEL_NAMES, SEQUENCE_MODELS, get_model

logger = logging.getLogger(__name__)


def spearman_ic(pred: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩 IC (pandas rank corr), NaN 安全.

    Args:
        pred: (N,) 预测得分。
        y: (N,) 真实标签。

    Returns:
        IC 值; 计算失败/常数序列返回 0.0。
    """
    pred = np.nan_to_num(
        np.asarray(pred, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0
    )
    y = np.nan_to_num(
        np.asarray(y, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0
    )
    if len(pred) < 3 or np.ptp(pred) == 0 or np.ptp(y) == 0:
        return 0.0
    ic = pd.Series(pred).corr(pd.Series(y), method="spearman")
    return float(ic) if np.isfinite(ic) else 0.0


class ModelSelector:
    """模型选择器.

    Attributes:
        models_: train_all 训练成功的模型 {name: ZooModel}。
        val_predictions_: 验证集预测 {name: (N,)}。
        report_: 最近一次对比报告 DataFrame。
    """

    def __init__(self, config: dict = None) -> None:
        """初始化.

        Args:
            config: 配置, 支持 keys:
                model_params (dict): {model_name: 超参覆盖};
                ic_threshold (float): 及格 IC 线 (默认 0.03)。
        """
        self.config = config or {}
        self.models_ = {}
        self.val_predictions_ = {}
        self.report_ = None
        self._y_val = None

    def train_all(self, X_train, y_train, X_val, y_val) -> pd.DataFrame:
        """训练所有可用模型并生成对比报告.

        依赖缺失 (lightgbm/xgboost/catboost/torch) 的模型记为 skipped
        并 warning, 不中断流程。

        Args:
            X_train: (N, 20, F) 训练特征。
            y_train: (N,) 训练标签。
            X_val: (M, 20, F) 验证特征 (OOS)。
            y_val: (M,) 验证标签。

        Returns:
            对比 DataFrame: [model, family, status, oos_ic, rmse]。
        """
        params_map = self.config.get("model_params", {})
        self.models_ = {}
        self.val_predictions_ = {}
        self._y_val = np.asarray(y_val, dtype=np.float64).ravel()
        rows = []
        for name in MODEL_NAMES:
            family = "sequence" if name in SEQUENCE_MODELS else "flat"
            try:
                model = get_model(name, **params_map.get(name, {}))
            except RuntimeError as exc:
                logger.warning("跳过模型 %s: %s", name, exc)
                rows.append(
                    {
                        "model": name,
                        "family": family,
                        "status": f"skipped: {exc}",
                        "oos_ic": np.nan,
                        "rmse": np.nan,
                    }
                )
                continue
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                ic = spearman_ic(pred, self._y_val)
                rmse = float(np.sqrt(np.mean((pred - np.nan_to_num(self._y_val)) ** 2)))
                self.models_[name] = model
                self.val_predictions_[name] = pred
                rows.append(
                    {
                        "model": name,
                        "family": family,
                        "status": "trained",
                        "oos_ic": ic,
                        "rmse": rmse,
                    }
                )
                logger.info("模型 %s 训练完成: oos_ic=%.4f rmse=%.5f", name, ic, rmse)
            except Exception as exc:  # noqa: BLE001 — 单模型失败不阻断全表
                logger.warning("模型 %s 训练失败: %s", name, exc)
                rows.append(
                    {
                        "model": name,
                        "family": family,
                        "status": f"failed: {exc}",
                        "oos_ic": np.nan,
                        "rmse": np.nan,
                    }
                )
        self.report_ = pd.DataFrame(rows)
        return self.report_

    def _trained(self, report: pd.DataFrame) -> pd.DataFrame:
        """过滤训练成功的行.

        Args:
            report: train_all 报告。

        Returns:
            status == 'trained' 的子表。

        Raises:
            ValueError: 无可用模型。
        """
        trained = report[report["status"] == "trained"]
        if trained.empty:
            raise ValueError("报告中无训练成功的模型, 无法选择")
        return trained

    def select_best(
        self, report: pd.DataFrame, user_preference: Optional[str] = None
    ) -> dict:
        """选最优模型或融合组合.

        Args:
            report: train_all 产出的对比报告。
            user_preference: 用户指定模型名或融合策略 (优先级最高)。

        Returns:
            {name, type: 'single'|'ensemble', reason, metrics}。

        Raises:
            ValueError: user_preference 不在可用集合内。
        """
        trained = self._trained(report)

        if user_preference is not None:
            if user_preference in set(trained["model"]):
                row = trained[trained["model"] == user_preference].iloc[0]
                return {
                    "name": user_preference,
                    "type": "single",
                    "reason": f"用户指定模型 {user_preference}",
                    "metrics": row.to_dict(),
                }
            if user_preference in ENSEMBLE_STRATEGIES:
                candidates = [
                    c
                    for c in self.search_ensemble(report)
                    if c["strategy"] == user_preference
                ]
                if not candidates:
                    raise ValueError(
                        f"融合策略 {user_preference} 无可用组合 "
                        "(需 ≥2 个训练成功的模型)"
                    )
                best = candidates[0]
                return {
                    "name": "+".join(best["models"]),
                    "type": "ensemble",
                    "strategy": best["strategy"],
                    "reason": f"用户指定融合策略 {user_preference}",
                    "metrics": best,
                }
            raise ValueError(
                f"user_preference={user_preference} 不在可用模型 "
                f"{sorted(trained['model'])} 或融合策略 "
                f"{ENSEMBLE_STRATEGIES} 内"
            )

        best_single_row = trained.loc[trained["oos_ic"].idxmax()]
        best_single_ic = float(best_single_row["oos_ic"])
        result = {
            "name": best_single_row["model"],
            "type": "single",
            "reason": f"单模型 OOS IC 最高 ({best_single_ic:.4f})",
            "metrics": best_single_row.to_dict(),
        }

        ensembles = self.search_ensemble(report)
        if ensembles and ensembles[0]["oos_ic"] > best_single_ic:
            top = ensembles[0]
            result = {
                "name": "+".join(top["models"]),
                "type": "ensemble",
                "strategy": top["strategy"],
                "reason": f"融合组合 OOS IC ({top['oos_ic']:.4f}) "
                f"优于最优单模型 ({best_single_ic:.4f})",
                "metrics": top,
            }
        logger.info(
            "select_best → %s (%s): %s",
            result["name"],
            result["type"],
            result["reason"],
        )
        return result

    def search_ensemble(self, report: pd.DataFrame, top_k: int = 3) -> List[dict]:
        """Top-K 融合组合搜索 (rank_mean/score_mean/weighted_score/stacking).

        组合: Top-K 单模型的全部 ≥2 元子集 (K=3 → C(3,2)+C(3,3)=4 组)。
        stacking 用验证集前半拟合元学习器、后半评估, 避免同段过拟合;
        样本过少 (<10) 时退化为全段拟合+评估。

        Args:
            report: train_all 产出的对比报告。
            top_k: 参与组合搜索的单模型数。

        Returns:
            [{models, strategy, oos_ic, type}] 按 oos_ic 降序。
        """
        trained = self._trained(report).sort_values("oos_ic", ascending=False)
        top_names = [
            n for n in trained["model"].head(top_k) if n in self.val_predictions_
        ]
        if len(top_names) < 2 or self._y_val is None:
            logger.warning("可用模型 <2 或无验证标签, 融合搜索为空")
            return []

        y = self._y_val
        combos = [
            c
            for r in range(2, len(top_names) + 1)
            for c in itertools.combinations(top_names, r)
        ]
        results = []
        for combo in combos:
            for strategy in ENSEMBLE_STRATEGIES:
                ens = ModelEnsemble(strategy=strategy)
                preds = {n: self.val_predictions_[n] for n in combo}
                if strategy == "stacking":
                    if len(y) >= 10:
                        half = len(y) // 2
                        fit_preds = {n: p[:half] for n, p in preds.items()}
                        ens.fit_stacking(fit_preds, y[:half])
                        combined = ens.combine({n: p[half:] for n, p in preds.items()})
                        ic = spearman_ic(combined, y[half:])
                    else:
                        ens.fit_stacking(preds, y)
                        ic = spearman_ic(ens.combine(preds), y)
                else:
                    ic = spearman_ic(ens.combine(preds), y)
                results.append(
                    {
                        "models": list(combo),
                        "strategy": strategy,
                        "oos_ic": ic,
                        "type": "ensemble",
                    }
                )
        results.sort(key=lambda r: r["oos_ic"], reverse=True)
        logger.info(
            "融合搜索完成: %d 组合, 最优 %s/%s ic=%.4f",
            len(results),
            results[0]["models"],
            results[0]["strategy"],
            results[0]["oos_ic"],
        )
        return results
