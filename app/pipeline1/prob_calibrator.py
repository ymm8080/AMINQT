# -*- coding: utf-8 -*-
"""
概率校准器 (DESIGN §14.4, PIPELINE1_V3.5 §四)
=================================================
Platt Scaling (逻辑回归校准); 校准集与早停验证集物理隔离 (训练器切分已保证);
随月度重训滚动重校 (用最近窗口, 不用历史全集).
严禁直接使用 LightGBM 原始 predict_proba 输出.
验收: Brier <= 0.24; 可靠性曲线偏移 < 5%.
预警: 模型经常输出 70%+ = 泄漏或校准失败, 先查 bug.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)

BRIER_MAX = 0.24
RELIABILITY_TOL = 0.05
HIGH_PROB_WARN = 0.70


class ProbCalibrator:
    """Platt Scaling 概率校准 (V3.6 候选: Isotonic)."""

    def __init__(self, method: str = "platt"):
        assert method in ("platt", "isotonic")
        self.method = method
        self._lr: LogisticRegression | None = None
        self._iso: IsotonicRegression | None = None

    def fit(self, raw_prob: np.ndarray, y_calib: np.ndarray) -> "ProbCalibrator":
        """在校准集 (窗口第 701-710 天) 上拟合. raw_prob = 分类模型原始 predict_proba."""
        raw_prob = np.asarray(raw_prob, dtype=float)
        y_calib = np.asarray(y_calib, dtype=float)
        if self.method == "platt":
            self._lr = LogisticRegression()
            self._lr.fit(raw_prob.reshape(-1, 1), y_calib)
        else:
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(raw_prob, y_calib)
        return self

    def predict_proba(self, raw_prob: np.ndarray) -> np.ndarray:
        """校准后概率."""
        raw_prob = np.asarray(raw_prob, dtype=float)
        if self.method == "platt":
            assert self._lr is not None, "calibrator 未 fit"
            return self._lr.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
        assert self._iso is not None, "calibrator 未 fit"
        return self._iso.predict(raw_prob)

    # ---------------- 验收 ----------------
    def reliability_report(self, y_true: np.ndarray, prob: np.ndarray,
                           n_bins: int = 10) -> dict:
        """可靠性报告: Brier / LogLoss / 分桶偏差 / 高概率预警.

        偏移 = mean |桶内预测均值 - 桶内实际胜率|; 必须 < 5%.
        """
        y_true = np.asarray(y_true, dtype=float)
        prob = np.asarray(prob, dtype=float)
        bins = pd.cut(prob, bins=n_bins, labels=False, include_lowest=True)
        offsets, bucket_win_rates = [], {}
        for b in range(n_bins):
            mask = bins == b
            if mask.sum() < 5:
                continue
            offsets.append(abs(prob[mask].mean() - y_true[mask].mean()))
            bucket_win_rates[b] = {"pred": round(float(prob[mask].mean()), 4),
                                   "actual": round(float(y_true[mask].mean()), 4),
                                   "n": int(mask.sum())}
        report = {
            "brier": float(brier_score_loss(y_true, prob)),
            "logloss": float(log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7))),
            "reliability_offset": float(np.mean(offsets)) if offsets else 1.0,
            "buckets": bucket_win_rates,
            "pass": True,
        }
        report["pass"] = (report["brier"] <= BRIER_MAX
                          and report["reliability_offset"] <= RELIABILITY_TOL)
        if (prob > HIGH_PROB_WARN).mean() > 0.05:
            logger.warning("高概率预警: >5%% 样本 prob>%.0f%%, 疑似泄漏或校准失败, 先查 bug",
                           HIGH_PROB_WARN * 100)
            report["high_prob_warning"] = True
        return report
