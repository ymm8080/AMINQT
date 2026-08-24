"""
V3.5 推理预测器 (P14 推理端)
================================
加载板块模型包 (DualTrackTrainer.save) → 特征面板推理 →
pred_ret_3d/5d/10d + Isotonic 校准 prob_up [E1] + 分布预测 pred_q10..q90
+ uncertainty_width [E1] + pain_prob [E2] → ListGenerator 候选输入.
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

from app.core.factor_engine import safe_divide

from .dual_track_trainer import DualTrackTrainer
from .label_engine import CLS_THRESHOLD, LABEL_WEIGHTS

logger = logging.getLogger(__name__)


def _cross_sectional_rank(values: np.ndarray) -> np.ndarray:
    """将一维数组转换为 [0, 1] 范围内的百分位排名 (用于回退 LambdaRank)."""
    from scipy.stats import rankdata

    ranks = rankdata(values)
    return (
        (ranks - 1) / (len(ranks) - 1)
        if len(ranks) > 1
        else np.zeros_like(values, dtype=float)
    )


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
            DataFrame: symbol, pred_ret_3d/5d/10d, prob_up (Platt 校准后)
        """
        bundle = self.bundles.get(board)
        if bundle is None:
            raise RuntimeError(f"板块 {board} 模型包未加载")
        cols = bundle["feature_cols"]
        latest = features.sort_values("date").groupby("symbol").tail(1).copy()
        # 推理端无法复现的特征 (训练注入 _brute_ / 面板缺列) 补 0 + 告警, 防 KeyError
        missing = [c for c in cols if c not in latest.columns]
        if missing:
            for c in missing:
                latest[c] = 0.0
            logger.warning(
                "[%s] 特征缺失 %d/%d 补 0: %s",
                board,
                len(missing),
                len(cols),
                missing[:5],
            )
        # np.nan_to_num: 特征面板可能含 NaN, 模型输入前必须清洗 (防 LightGBM 异常)
        X = np.nan_to_num(
            latest[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        models = bundle["models"]
        latest["pred_ret_3d"] = models["3d_reg"][0].predict(X)
        latest["pred_ret_5d"] = models["5d_reg"][0].predict(X)
        latest["pred_ret_10d"] = (
            models["10d_reg"][0].predict(X) if "10d_reg" in models else np.nan
        )
        # [08-24] prob_up 系列改回归残差派生概率: prob_up_kd = P(ret>CLS_THRESHOLD|pred_ret_kd)
        #         = 1 - F_e(CLS_THRESHOLD - pred_ret_kd), e = calib 段回归残差 (训练器已存
        #         bundle["reg_resid_{k}d"]). 继承 reg 头判别力 (08-24 原型: main 10d p_reg
        #         IC_ret +0.105 vs p_cls -0.136; cls 头 test 段 IC≈0/负, 不排序).
        #         事件口径 = 净收益>0.5% (reg 主标签), 较旧 Platt cls (毛收益>0.5%) 低 ~0.2pp.
        #         旧 bundle 无 reg_resid_* → 回退 Platt cls (原逻辑).
        calibrators = bundle.get("calibrators", {})

        def _platt_cls_prob(k: int, kind: str) -> np.ndarray:
            raw = models[kind][0].predict_proba(X)[:, 1]
            cal_k = calibrators.get(k) if calibrators else None
            return cal_k.predict_proba(raw) if cal_k is not None else raw

        def _reg_resid_prob(k: int) -> np.ndarray | None:
            resid = bundle.get(f"reg_resid_{k}d")
            if resid is None or len(resid) < 30:
                return None
            pred = latest[f"pred_ret_{k}d"].to_numpy(dtype=float)
            p = 1.0 - np.searchsorted(
                np.asarray(resid, dtype=float), CLS_THRESHOLD - pred
            ) / len(resid)
            return np.clip(p, 1e-6, 1.0 - 1e-6)

        for k in (3, 5, 10):
            kind = f"{k}d_cls"
            if kind not in models:
                continue
            p_cls = _platt_cls_prob(k, kind)
            p_reg = _reg_resid_prob(k)
            # 单个 pred 异常 (NaN) 时按列回退 p_cls, 不整列报废
            latest[f"prob_up_{k}d"] = (
                np.where(
                    np.isfinite(latest[f"pred_ret_{k}d"].to_numpy(dtype=float)),
                    p_reg,
                    p_cls,
                )
                if p_reg is not None
                else p_cls
            )
        # 3d 主概率别名 (旧 bundle 无 reg_resid_* 时 prob_up_3d 已是 Platt cls)
        if "prob_up_3d" in latest.columns:
            latest["prob_up"] = latest["prob_up_3d"]
        # 综合排序分: LABEL_WEIGHTS 加权 (修改字典即全局生效; 旧 bundle 缺 pred_ret_10d 时
        # 只对已预测的视界加权并按各自权重和归一, 保证新旧 bundle 得分口径一致)
        present_w = {
            k: w for k, w in LABEL_WEIGHTS.items() if f"pred_ret_{k}d" in latest.columns
        }
        total_w = sum(present_w.values())
        latest["composite_score"] = (
            sum(present_w[k] * latest[f"pred_ret_{k}d"].values for k in present_w)
            / total_w
        )
        keep = [
            "symbol",
            "board",
            "industry",
            "composite_score",
            "pred_ret_3d",
            "pred_ret_5d",
            "prob_up",
            "prob_up_3d",
            "prob_up_5d",
        ]
        # [10d] 新视界 (旧 bundle 无 10d 模型 → 补 NaN, 兼容回退)
        for col in ("pred_ret_10d", "prob_up_10d"):
            if col not in latest.columns:
                latest[col] = np.nan
            keep.append(col)
        # [E1] 分位数分布预测 (bundle 含 quantile_models 时)
        # 旧 bundle 1d 集向后兼容; 新 bundle 只含 quantile_models_3d/5d
        if "quantile_models" in bundle:
            dist = bundle["quantile_models"].predict(X)
            for col in dist.columns:
                latest[col] = dist[col].values
            keep += list(dist.columns)
        for horizon in (3, 5):
            qkey = f"quantile_models_{horizon}d"
            if qkey in bundle:
                dist = bundle[qkey].predict(X)
                for col in dist.columns:
                    latest[f"{col}_{horizon}d"] = dist[col].values
                keep += [f"{c}_{horizon}d" for c in dist.columns]
        # [E2] 痛苦预警 (bundle 含 pain_model 时)
        if "pain_model" in bundle:
            latest["pain_prob"] = bundle["pain_model"].predict_proba(X)
            keep.append("pain_prob")
        # [阶段四] LambdaRank 排序分 (bundle 含 rank_model 时)
        # 若 LambdaRank 退化 (常数输出) 或预测异常, 自动回退 pred_ret_3d 横截面排名
        if "rank_model" in bundle:
            try:
                raw_rank = bundle["rank_model"][0].predict(X)
                if np.std(raw_rank) < 1e-6:
                    logger.warning(
                        "[%s] LambdaRank 退化 (std=%.6f), 回退 pred_ret_3d 排名",
                        board,
                        np.std(raw_rank),
                    )
                    reg_pred = models["3d_reg"][0].predict(X)
                    raw_rank = _cross_sectional_rank(reg_pred)
            except Exception:
                logger.warning("[%s] LambdaRank 预测异常, 回退 pred_ret_3d 排名", board)
                reg_pred = models["3d_reg"][0].predict(X)
                raw_rank = _cross_sectional_rank(reg_pred)
            latest["rank_score"] = raw_rank
            keep.append("rank_score")
        # 当日涨跌幅 (看板 day_change 列): close/pre_close - 1, 除零防护
        if {"close", "pre_close"} <= set(latest.columns):
            latest["day_change"] = (
                safe_divide(latest["close"], latest["pre_close"].replace(0, np.nan)) - 1
            )
        # [E1] 分布版仓位权重输入; [E6] liquidity_cap 输入; [08-21] liquidity_score/date
        # 透传 — pool_blend_cut 切池依赖 (keep 丢列曾致切池静默失效)
        for opt in (
            "ATR_pct",
            "adv20",
            "is_limit_up_close",
            "is_one_word_limit",
            "day_change",
            "liquidity_score",
            "date",
        ):
            if opt in latest.columns:
                keep.append(opt)
        return latest[keep].reset_index(drop=True)

    def predict_tomorrow(self, features: pd.DataFrame, board: str) -> pd.DataFrame:
        """每日 DELTA 推理: 输出**明天的价格和概率** (用户 2026-07-22 裁决).

        输入特征面板须已包含当日 (delta) bar — 由调用方将 delta 追加到历史后
        重算特征 (daily_pipeline.run 的 panel 路径已含当日).
        pred_price_tomorrow = close_T * (1 + pred_ret_3d)  [点估计, 非承诺价]

        Returns:
            predict() 全部列 + pred_price_tomorrow
        """
        try:
            out = self.predict(features, board)
            latest = features.sort_values("date").groupby("symbol").tail(1)
            out = out.merge(latest[["symbol", "close"]], on="symbol", how="left")
            out["pred_price_tomorrow"] = (
                out["close"] * (1 + out["pred_ret_3d"])
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
