# -*- coding: utf-8 -*-
"""
V3.5 推理预测器 (P14 推理端)
================================
加载板块模型包 (DualTrackTrainer.save) → 特征面板推理 →
pred_ret_1d/3d/5d + Platt 校准 prob_up → ListGenerator 候选输入.
维护 is_in_yesterday_list (Holding Bonus, 昨日清单回填).

用户裁决 (2026-07-22): 日线预测只用本地 LightGBM 模型;
每日 DELTA 数据 (当日新 bar 追加到历史) 推理**明天的价格和概率** —
predict_tomorrow() 显式输出 pred_price_tomorrow + prob_up.
"""

from __future__ import annotations

import logging
import os

import numpy as np
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
        # np.nan_to_num: 特征面板可能含 NaN, 模型输入前必须清洗 (防 LightGBM 异常)
        X = np.nan_to_num(
            latest[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
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

    def predict_tomorrow(self, features: pd.DataFrame, board: str) -> pd.DataFrame:
        """每日 DELTA 推理: 输出**明天的价格和概率** (用户 2026-07-22 裁决).

        输入特征面板须已包含当日 (delta) bar — 由调用方将 delta 追加到历史后
        重算特征 (daily_pipeline.run 的 panel 路径已含当日).
        pred_price_tomorrow = close_T * (1 + pred_ret_1d)  [点估计, 非承诺价]

        Returns:
            predict() 全部列 + pred_price_tomorrow
        """
        try:
            out = self.predict(features, board)
            latest = features.sort_values("date").groupby("symbol").tail(1)
            out = out.merge(latest[["symbol", "close"]], on="symbol", how="left")
            out["pred_price_tomorrow"] = (
                out["close"] * (1 + out["pred_ret_1d"])
            ).round(3)
            return out.drop(columns=["close"])
        except Exception:
            logger.exception("predict_tomorrow 推理失败 (board=%s)", board)
            raise

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
