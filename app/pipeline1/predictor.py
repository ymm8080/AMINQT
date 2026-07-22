# -*- coding: utf-8 -*-
"""
V3.5 推理预测器 (P14 推理端)
================================
加载板块模型包 (DualTrackTrainer.save) → 特征面板推理 →
pred_ret_1d/3d/5d + Platt 校准 prob_up → ListGenerator 候选输入.
维护 is_in_yesterday_list (Holding Bonus, 昨日清单回填).
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from .dual_track_trainer import DualTrackTrainer

logger = logging.getLogger(__name__)


class V35Predictor:
    """双板块推理: {'main': bundle_path, 'dual': bundle_path}."""

    def __init__(self, bundle_paths: dict[str, str]):
        self.bundles = {}
        for board, path in bundle_paths.items():
            if os.path.exists(path):
                self.bundles[board] = DualTrackTrainer.load(path)
            else:
                logger.error("模型包缺失: %s (%s)", board, path)

    def predict(self, features: pd.DataFrame, board: str) -> pd.DataFrame:
        """对单板块最新截面推理.

        Args:
            features: FeatureEngineV35.build() 输出 (取每 symbol 最新一行即可, 全历史也行)
            board: 'main' / 'dual'

        Returns:
            DataFrame: symbol, pred_ret_1d/3d/5d, prob_up (Platt 校准后)
        """
        bundle = self.bundles.get(board)
        if bundle is None:
            raise RuntimeError(f"板块 {board} 模型包未加载")
        cols = bundle["feature_cols"]
        latest = features.sort_values("date").groupby("symbol").tail(1).copy()
        X = latest[cols]
        models = bundle["models"]
        latest["pred_ret_1d"] = models["1d_reg"][0].predict(X)
        latest["pred_ret_3d"] = models["3d_reg"][0].predict(X)
        latest["pred_ret_5d"] = models["5d_reg"][0].predict(X)
        raw_prob = models["1d_cls"][0].predict_proba(X)[:, 1]
        # Platt 校准 (严禁原始 predict_proba)
        latest["prob_up"] = bundle["calibrator"].predict_proba(raw_prob)
        keep = [
            "symbol",
            "board",
            "industry",
            "pred_ret_1d",
            "pred_ret_3d",
            "pred_ret_5d",
            "prob_up",
        ]
        for opt in ("is_limit_up_close", "is_one_word_limit"):
            if opt in latest.columns:
                keep.append(opt)
        return latest[keep].reset_index(drop=True)

    @staticmethod
    def mark_yesterday_list(
        candidates: pd.DataFrame, yesterday_list: pd.DataFrame | None
    ) -> pd.DataFrame:
        """回填 is_in_yesterday_list (Holding Bonus 输入)."""
        candidates = candidates.copy()
        if yesterday_list is None or len(yesterday_list) == 0:
            candidates["is_in_yesterday_list"] = 0
        else:
            yesterday = set(yesterday_list["symbol"])
            candidates["is_in_yesterday_list"] = (
                candidates["symbol"].isin(yesterday).astype(int)
            )
        return candidates
