# -*- coding: utf-8 -*-
"""
双轨训练器 (DESIGN §14.4, PIPELINE1_V3.5 §二/§七)
=====================================================
LightGBM 双轨×8: (1d_reg Huber + 1d_cls binary + 3d_reg + 5d_reg) × (主板/双创).
- 720 日滚动窗口: 训练690 / 早停10 / 校准10 (与验证物理隔离!) / 测试10 (仅月度归因)
- 半衰期加权 250 天; Huber loss; early_stopping patience=50
- 超参纪律: 年度贝叶斯调优 (≤50 组), 月度仅固定超参重训
- 月度重训与清单生成解耦 (15:30 重训 / 16:00 旧模型出清单 / 18:00 前切换)
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_TOTAL = 720
TRAIN_DAYS = 690
ES_DAYS = 10  # 早停验证
CALIB_DAYS = 10  # 校准 (与早停物理隔离)
TEST_DAYS = 10  # 测试 (仅月度归因, 严禁反向调参)
HALF_LIFE = 250  # 半衰期加权 (天)
ES_PATIENCE = 50
OOS_IC_MIN = 0.03  # 新模型切换门槛

LGB_PARAMS_REG = dict(
    objective="huber",
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1,
)
LGB_PARAMS_CLS = dict(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1,
)
MODEL_KINDS = ("1d_reg", "1d_cls", "3d_reg", "5d_reg")


class DualTrackTrainer:
    """双轨训练 — 每个板块独立训练 4 个模型."""

    def __init__(self, model_dir: str = "models/pipeline1"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    # ---------------- 窗口切分 ----------------
    @staticmethod
    def split_window(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """720 日窗口四段切分: train(690) / es(10) / calib(10) / test(10).
        校准集与早停验证集物理隔离, 否则校准器学到被调参挑剩下的噪声."""
        dates = sorted(df["date"].unique())[-WINDOW_TOTAL:]
        seg = {
            "train": dates[:TRAIN_DAYS],
            "es": dates[TRAIN_DAYS : TRAIN_DAYS + ES_DAYS],
            "calib": dates[TRAIN_DAYS + ES_DAYS : TRAIN_DAYS + ES_DAYS + CALIB_DAYS],
            "test": dates[-TEST_DAYS:],
        }
        return {k: df[df["date"].isin(v)] for k, v in seg.items()}

    @staticmethod
    def time_weights(df: pd.DataFrame, half_life: int = HALF_LIFE) -> np.ndarray:
        """半衰期加权: 旧样本权重指数衰减, 让模型贴近近期市场结构."""
        dates = sorted(df["date"].unique())
        age = {d: len(dates) - 1 - i for i, d in enumerate(dates)}
        return np.array([0.5 ** (age[d] / half_life) for d in df["date"]])

    # ---------------- 单模型训练 ----------------
    def _train_one(
        self, kind: str, segs: dict[str, pd.DataFrame], feature_cols: list[str]
    ):
        import lightgbm as lgb

        label = {
            "1d_reg": "label_1d",
            "1d_cls": "label_cls",
            "3d_reg": "label_3d",
            "5d_reg": "label_5d",
        }[kind]
        train = segs["train"].dropna(subset=[label])  # per-model dropna (安全网 #7)
        es = segs["es"].dropna(subset=[label])
        X, y = train[feature_cols], train[label]
        X_es, y_es = es[feature_cols], es[label]
        w = self.time_weights(train)

        if kind.endswith("cls"):
            model = lgb.LGBMClassifier(**LGB_PARAMS_CLS)
        else:
            model = lgb.LGBMRegressor(**LGB_PARAMS_REG)
        model.fit(
            X,
            y,
            sample_weight=w,
            eval_set=[(X_es, y_es)],
            callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)],
        )
        return model, label

    # ---------------- 窗口训练 (单板块 4 模型) ----------------
    def train_window(
        self, df: pd.DataFrame, board: str, feature_cols: list[str]
    ) -> dict:
        """训练一个板块的 4 个模型. 返回 {kind: (model, label)} + 元数据."""
        segs = self.split_window(df)
        out = {"board": board, "feature_cols": feature_cols, "models": {}, "segs": segs}
        for kind in MODEL_KINDS:
            model, label = self._train_one(kind, segs, feature_cols)
            out["models"][kind] = (model, label)
            logger.info(
                "[%s] %s 训练完成, 样本 %d",
                board,
                kind,
                len(segs["train"].dropna(subset=[label])),
            )
        return out

    # ---------------- 校准器拟合 (随月度重训滚动重校) ----------------
    @staticmethod
    def fit_calibrator(trained: dict):
        """用校准集 (与早停物理隔离) 拟合 Platt 校准器 (安全网: 严禁原始 predict_proba)."""
        from .prob_calibrator import ProbCalibrator

        model, label = trained["models"]["1d_cls"]
        calib = trained["segs"]["calib"].dropna(subset=[label])
        cols = trained["feature_cols"]
        raw = model.predict_proba(calib[cols])[:, 1]
        calibrator = ProbCalibrator(method="platt").fit(raw, calib[label].values)
        trained["calibrator"] = calibrator
        return calibrator

    # ---------------- OOS 验证 + 切换 ----------------
    def validate_oos(self, trained: dict, ic_min: float = OOS_IC_MIN) -> dict:
        """测试段 Rank IC (仅月度归因段). IC >= 0.03 才允许切换新模型."""
        from .ic_screener import ICScreener

        test = trained["segs"]["test"]
        cols = trained["feature_cols"]
        ics = {}
        for kind, (model, label) in trained["models"].items():
            sub = test.dropna(subset=[label]).copy()
            if len(sub) < 30:
                ics[kind] = 0.0
                continue
            sub["_pred"] = model.predict(sub[cols])
            ics[kind] = ICScreener.rank_ic(
                sub.rename(columns={"_pred": "score"}), "score", label
            )
        return {"ics": ics, "pass": ics.get("1d_reg", 0.0) >= ic_min}

    def save(self, trained: dict, tag: str) -> str:
        """保存模型包 (含校准器; 若无则先拟合)."""
        if "calibrator" not in trained:
            self.fit_calibrator(trained)
        path = os.path.join(self.model_dir, f"{trained['board']}_{tag}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "board": trained["board"],
                    "feature_cols": trained["feature_cols"],
                    "models": trained["models"],
                    "calibrator": trained["calibrator"],
                },
                fh,
            )
        return path

    @staticmethod
    def load(path: str) -> dict:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ---------------- 特征相似度回退 ----------------
    @staticmethod
    def feature_similarity_check(trained: dict, threshold: float = 0.8) -> bool:
        """三回归模型 importance 排名 Spearman > 0.8 → 高度相似, 建议回退单模型+多输出."""
        from scipy.stats import spearmanr

        imps = []
        for kind in ("1d_reg", "3d_reg", "5d_reg"):
            model, _ = trained["models"][kind]
            imps.append(pd.Series(model.feature_importances_).rank())
        corrs = [
            spearmanr(imps[i], imps[j]).statistic
            for i in range(3)
            for j in range(i + 1, 3)
        ]
        return bool(np.nanmean(corrs) > threshold)

    # ---------------- 月度重训 (解耦) ----------------
    def monthly_retrain(
        self,
        panels: dict[str, pd.DataFrame],
        feature_cols_by_board: dict[str, list[str]],
        tag: str,
    ) -> dict:
        """panels: {'main': df, 'dual': df}. 15:30 启动, 与 16:00 清单生成并行.

        Returns:
            {board: {'path', 'oos': {...}, 'switched': bool}}
        """
        results = {}
        for board, df in panels.items():
            trained = self.train_window(df, board, feature_cols_by_board[board])
            oos = self.validate_oos(trained)
            path = self.save(trained, tag)
            # 切换决策: OOS 合格才切换, 否则保留旧模型 + 告警
            results[board] = {"path": path, "oos": oos, "switched": oos["pass"]}
            if not oos["pass"]:
                logger.warning(
                    "[%s] 新模型 OOS IC=%.4f < %.2f, 保留旧模型",
                    board,
                    oos["ics"].get("1d_reg", 0.0),
                    OOS_IC_MIN,
                )
        return results
