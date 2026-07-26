"""
双轨训练器 (DESIGN §14.4, PIPELINE1_V3.8 §二/§四/§七)
=====================================================
LightGBM 双轨×8: (1d_reg Huber + 1d_cls binary + 3d_reg + 5d_reg) × (主板/双创).
**日线预测只用本地 LightGBM 模型 (用户 2026-07-22 裁决), 无云端/ONNX/LSTM 依赖.**
- [V3.8] 750 日滚动窗口: 训练620 / 早停20 / 校准20 (与验证物理隔离!) / 测试90; patience=100
- [B11] OHLCV 回填达标 (<1250 交易日) 前首个训练窗口降为 540 日过渡, 达标后恢复 750 日
- [B10] 半衰期加权 250 天, 方向断言 weights[-1] > weights[0] (最新样本权重=1.0)
- [B9] PM 验收标签 label_pm_kd 存在时优先于研究口径 label_kd
- [E5] 净收益标签 label_*_net 存在时优先 (滑点分层口径, 训练/验收主标签)
- [E1] 分位数五模型 (q10/25/50/75/90, label_1d_net) + 保序单调性后处理
- [E2] 痛苦预警模型 (label_pain 分类 → pain_prob)
- [E1] 概率校准 Platt → Isotonic (月度滚动重校)
- Huber loss; early_stopping patience=100 (V3.8 §2.1)
- 超参纪律: 年度贝叶斯调优 (≤50 组), 每月仅重训; E1 分位数模型沿用回归超参不单独搜索
- **每周一次全局重训 (用户 2026-07-22 裁决: 周频, 非月频)**:
  每周第一个交易日 15:30 启动 (T-1 数据), 16:00 旧模型出清单, 18:00 前切换
- 重训与清单生成解耦 (重训绝不阻塞当日清单)
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_TOTAL = 750  # [V3.8] 620/20/20/90
WINDOW_TRANSITION = 540  # [B11] 回填达标前过渡窗口
B11_FULL_DEPTH = 1250  # [B11] 回填达标线 (≥5 年交易日): 达到才用 750 窗口
MIN_TRAIN_DAYS = 50  # 窗口 train 段下限, 不足即拒绝训练
TRAIN_DAYS = 620  # [V3.8] 训练 620 天
ES_DAYS = 20  # 早停验证
CALIB_DAYS = 20  # 校准 (与早停物理隔离)
TEST_DAYS = 90  # 测试 (仅归因, 严禁反向调参)
HALF_LIFE = 250  # 半衰期加权 (天)
ES_PATIENCE = 100  # [V3.8 §2.1] patience=100
OOS_IC_MIN = 0.03  # 新模型切换门槛

LGB_PARAMS_REG = {
    "objective": "huber",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "random_state": 42,
    "verbosity": -1,
}
LGB_PARAMS_CLS = {
    "objective": "binary",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "random_state": 42,
    "verbosity": -1,
}
MODEL_KINDS = ("1d_reg", "1d_cls", "3d_reg", "5d_reg")


def risk_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter non-tradable stocks (suspended/ST) before training.

    Limit-up/down is NOT filtered here — it's an execution-time constraint
    handled by app.core.risk_filter.apply_filters, not a training-time filter.
    Models must learn limit-up/down patterns.
    """
    mask = pd.Series(True, index=df.index)
    for col in ("is_suspended", "is_st"):
        if col in df.columns:
            mask &= ~df[col].astype(bool)
    return df[mask]


class DualTrackTrainer:
    """双轨训练 — 每个板块独立训练 4 个模型."""

    def __init__(self, model_dir: str = "models/pipeline1"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    # ---------------- 窗口切分 ----------------
    @staticmethod
    def split_window(
        df: pd.DataFrame, window_total: int = WINDOW_TOTAL
    ) -> dict[str, pd.DataFrame]:
        """滚动窗口四段切分 [V3.8 §2.1]: train / es / calib / test = 620/20/20/90 (750 窗口).

        校准集与早停验证集物理隔离, 否则校准器学到被调参挑剩下的噪声.
        [B11] window_total=540 为回填达标前过渡窗口 (es/calib/test 段长不变,
        仅压缩 train 段), 达标后恢复 750.
        """
        n_es_calib_test = ES_DAYS + CALIB_DAYS + TEST_DAYS
        train_days = window_total - n_es_calib_test
        dates = sorted(df["date"].unique())[-window_total:]
        seg = {
            "train": dates[:train_days],
            "es": dates[train_days : train_days + ES_DAYS],
            "calib": dates[train_days + ES_DAYS : train_days + ES_DAYS + CALIB_DAYS],
            "test": dates[-TEST_DAYS:],
        }
        return {k: df[df["date"].isin(v)] for k, v in seg.items()}

    @staticmethod
    def time_weights(df: pd.DataFrame, half_life: int = HALF_LIFE) -> np.ndarray:
        """半衰期加权: 旧样本权重指数衰减, 让模型贴近近期市场结构.

        [B10] 方向断言: 日期升序轴上 weights[-1] > weights[0]
        (最新样本权重=1.0), 写反即中止 — 训练前强制检查.
        """
        dates = sorted(df["date"].unique())
        age = {d: len(dates) - 1 - i for i, d in enumerate(dates)}
        if len(dates) > 1:
            w_first = 0.5 ** (age[dates[0]] / half_life)
            w_last = 0.5 ** (age[dates[-1]] / half_life)
            assert w_last > w_first, (
                "B10 半衰期权重方向错误: 最新样本权重必须最大 "
                f"(w_last={w_last:.4f} <= w_first={w_first:.4f})"
            )
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
        # [B9] PM 执行口径验收标签优先 (label_pm_kd), 缺失时回退研究口径 label_kd
        if kind != "1d_cls":
            pm_label = f"label_pm_{kind[0]}d"
            if pm_label in segs["train"].columns:
                label = pm_label
            # [E5] 净收益标签 (分层滑点) 优先于毛收益 — 训练/验收主标签口径 (D1)
            if f"{label}_net" in segs["train"].columns:
                label = f"{label}_net"
        train = segs["train"].dropna(subset=[label])  # per-model dropna (安全网 #7)
        es = segs["es"].dropna(subset=[label])
        train = risk_filter(train)
        es = risk_filter(es)
        X = np.nan_to_num(train[feature_cols].values, nan=0.0)
        y = train[label].values
        X_es = np.nan_to_num(es[feature_cols].values, nan=0.0)
        y_es = es[label].values
        w = self.time_weights(train)

        if kind.endswith("cls"):
            model = lgb.LGBMClassifier(**LGB_PARAMS_CLS)
        else:
            model = lgb.LGBMRegressor(**LGB_PARAMS_REG)
        try:
            model.fit(
                X,
                y,
                sample_weight=w,
                eval_set=[(X_es, y_es)],
                callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)],
            )
        except Exception as exc:
            logger.error("模型训练失败 [%s/%s]: %s", kind, label, exc)
            raise
        return model, label

    # ---------------- 窗口训练 (单板块 4 模型 + E1/E2) ----------------
    def train_window(
        self, df: pd.DataFrame, board: str, feature_cols: list[str]
    ) -> dict:
        """训练一个板块的 4 个模型 + E1 分位数五模型 + E2 痛苦预警 (标签齐备时).

        窗口深度自适应: ≥1250 交易日 (B11 达标) → 750; 否则过渡 min(540, 实际深度).
        3 年数据经步骤1 (list_days≥250) 过滤后约 490 日, 750 窗口会让 es/calib 落空.
        返回 {kind: (model, label)} + 元数据 + ['quantile_models'/'pain_model'].
        """
        depth = int(df.groupby("symbol")["date"].nunique().min()) if len(df) else 0
        window = WINDOW_TOTAL if depth >= B11_FULL_DEPTH else min(WINDOW_TRANSITION, depth)
        if window - (ES_DAYS + CALIB_DAYS + TEST_DAYS) < MIN_TRAIN_DAYS:
            raise RuntimeError(
                f"[{board}] 训练样本深度不足: {depth} 交易日 "
                f"(需 ≥ {ES_DAYS + CALIB_DAYS + TEST_DAYS + MIN_TRAIN_DAYS})"
            )
        if window != WINDOW_TOTAL:
            logger.info("[%s] B11 过渡窗口: 深度 %d 日 → 窗口 %d", board, depth, window)
        segs = self.split_window(df, window)
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
        self._train_extras(out)
        return out

    # ---------------- E1/E2 + LambdaRank: 分位数 + 痛苦预警 + 排序 ----------------
    def _train_extras(self, out: dict) -> None:
        """[E1] 分位数五模型 (label_1d_net) + [E2] 痛苦预警 (label_pain)
        + [阶段四] LambdaRank 排序模型 (lambdarank_truncation_level=25, 分位 gain).

        E1 沿用回归超参, 不单独搜索 (V3.8 §2.2, 避免调参维度爆炸).
        标签缺失时跳过 (向后兼容旧面板).
        """
        from .quantile_models import PainModel, QuantileModelSet

        segs, cols = out["segs"], out["feature_cols"]

        def _xy(seg_name: str, label: str):
            sub = risk_filter(segs[seg_name].dropna(subset=[label]))
            X = np.nan_to_num(sub[cols].values, nan=0.0)
            return sub, X, sub[label].values

        # E1: label 偏好 label_pm_1d_net → label_1d_net → label_1d (与 _train_one 同口径)
        q_label = next(
            (
                c
                for c in ("label_pm_1d_net", "label_1d_net", "label_1d")
                if c in segs["train"].columns
            ),
            None,
        )
        if q_label is not None:
            try:
                train, X, y = _xy("train", q_label)
                _, X_es, y_es = _xy("es", q_label)
                # E1 沿用回归超参 (objective 由 QuantileModelSet 按分位设置)
                params = {k: v for k, v in LGB_PARAMS_REG.items() if k != "objective"}
                qset = QuantileModelSet(params).fit(
                    X,
                    y,
                    sample_weight=self.time_weights(train),
                    eval_set=(X_es, y_es) if len(y_es) else None,
                    es_patience=ES_PATIENCE,
                )
                qset.label_ = q_label
                out["quantile_models"] = qset
                logger.info(
                    "[%s] E1 分位数五模型训练完成 (label=%s)", out["board"], q_label
                )
            except Exception as e:
                logger.warning("[%s] E1 分位数模型训练失败: %s", out["board"], e)
            # 阶段四: LambdaRank (标签=净收益截面分位 gain 0-4, group=date)
            try:
                out["rank_model"] = self._train_ranker(out, q_label)
                logger.info("[%s] LambdaRank 排序模型训练完成", out["board"])
            except Exception as e:
                logger.warning("[%s] LambdaRank 训练失败: %s", out["board"], e)

        if "label_pain" in segs["train"].columns:
            try:
                train, X, y = _xy("train", "label_pain")
                _, X_es, y_es = _xy("es", "label_pain")
                params = {k: v for k, v in LGB_PARAMS_CLS.items() if k != "objective"}
                pain = PainModel(params).fit(
                    X,
                    y,
                    sample_weight=self.time_weights(train),
                    eval_set=(X_es, y_es) if len(y_es) else None,
                    es_patience=ES_PATIENCE,
                )
                out["pain_model"] = pain
            except Exception as e:
                logger.warning("[%s] E2 痛苦预警模型训练失败: %s", out["board"], e)

    # ---------------- 阶段四: LambdaRank 排序模型 ----------------
    def _train_ranker(self, out: dict, label: str):
        """LambdaRank (lambdarank_truncation_level=25, V3.8 §2.2 搜索范围之一).

        标签: 净收益按 date 截面分位 → gain 0-4 (LGBMRanker 需非负整数 gain);
        group = 每个 date 的样本数 (排序以横截面为单位).
        """
        import lightgbm as lgb

        segs, cols = out["segs"], out["feature_cols"]

        def _prep(seg_name: str):
            sub = risk_filter(segs[seg_name].dropna(subset=[label])).sort_values("date")
            gains = (
                sub.groupby("date")[label]
                .rank(pct=True)
                .pipe(lambda s: (s * 5).clip(0, 4.999).astype(int))
            )
            group = sub.groupby("date").size().values
            X = np.nan_to_num(sub[cols].values, nan=0.0)
            return X, gains.values, group

        X, gains, group = _prep("train")
        X_es, gains_es, group_es = _prep("es")
        model = lgb.LGBMRanker(
            objective="lambdarank",
            lambdarank_truncation_level=25,
            n_estimators=LGB_PARAMS_REG["n_estimators"],
            learning_rate=LGB_PARAMS_REG["learning_rate"],
            random_state=42,
            verbosity=-1,
        )
        model.fit(
            X,
            gains,
            group=group,
            eval_set=[(X_es, gains_es)] if len(gains_es) else None,
            eval_group=[group_es] if len(gains_es) else None,
            callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)]
            if len(gains_es)
            else None,
        )
        return model, label

    # ---------------- 校准器拟合 (随月度重训滚动重校) ----------------
    @staticmethod
    def fit_calibrator(trained: dict):
        """用校准集 (与早停物理隔离) 拟合 Isotonic 校准器 (安全网: 严禁原始 predict_proba).

        [E1/V3.8] Platt Scaling → Isotonic Regression, 月度滚动重校.
        """
        from .prob_calibrator import ProbCalibrator

        model, label = trained["models"]["1d_cls"]
        calib = trained["segs"]["calib"].dropna(subset=[label])
        cols = trained["feature_cols"]
        raw = model.predict_proba(np.nan_to_num(calib[cols].values, nan=0.0))[:, 1]
        calibrator = ProbCalibrator(method="isotonic").fit(raw, calib[label].values)
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
            sub["_pred"] = model.predict(np.nan_to_num(sub[cols].values, nan=0.0))
            ics[kind] = ICScreener.rank_ic(
                sub.rename(columns={"_pred": "score"}), "score", label
            )
        return {"ics": ics, "pass": ics.get("1d_reg", 0.0) >= ic_min}

    def save(self, trained: dict, tag: str) -> str:
        """保存模型包 (含校准器; 若无则先拟合)."""
        if "calibrator" not in trained:
            self.fit_calibrator(trained)
        path = os.path.join(self.model_dir, f"{trained['board']}_{tag}.pkl")
        bundle = {
            "board": trained["board"],
            "feature_cols": trained["feature_cols"],
            "models": trained["models"],
            "calibrator": trained["calibrator"],
        }
        for extra in ("quantile_models", "pain_model", "rank_model"):  # E1/E2/排序
            if extra in trained:
                bundle[extra] = trained[extra]
        try:
            with open(path, "wb") as fh:
                pickle.dump(bundle, fh)
        except OSError as e:
            logger.error("模型保存失败 (%s): %s", path, e)
            raise
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

    # ---------------- 每周全局重训 (解耦) ----------------
    def weekly_retrain(
        self,
        panels: dict[str, pd.DataFrame],
        feature_cols_by_board: dict[str, list[str]],
        tag: str,
    ) -> dict:
        """每周一次全局重训 (用户 2026-07-22 裁决: 周频全局训练).

        panels: {'main': 主板750日面板, 'dual': 双创750日面板}.
        每周第一个交易日 15:30 启动, 与 16:00 清单生成并行.

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

    # 兼容别名 (V3.5 原文为月度, 用户 2026-07-22 裁决改为周频)
    monthly_retrain = weekly_retrain
