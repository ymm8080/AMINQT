"""
双轨训练器 (DESIGN §14.4, PIPELINE1_V3.8 §二/§四/§七)
=====================================================
LightGBM 双轨×6: (3d/5d/10d × reg + cls) × (主板/双创). 10d 视界 2026-08-07 加入 (排名键);
1d/2d 视界 2026-08-09 删除 (LABEL_WEIGHTS 权重 0, 复合排名分不引用, 1d 不可执行易误杀).
**日线预测只用本地 LightGBM 模型 (用户 2026-07-22 裁决), 无云端/ONNX/LSTM 依赖.**
- [V3.8] 750 日滚动窗口: 训练620 / 早停20 / 校准20 (与验证物理隔离!) / 测试90; patience=100
- [B11] OHLCV 回填达标 (<1250 交易日) 前首个训练窗口降为 540 日过渡, 达标后恢复 750 日
- [B10] 半衰期加权 250 天, 方向断言 weights[-1] > weights[0] (最新样本权重=1.0)
- [B9] PM 验收标签 label_pm_kd 存在时优先于研究口径 label_kd
- [E5] 净收益标签 label_*_net 存在时优先 (滑点分层口径, 训练/验收主标签)
- [E1] 分位数五模型 (q10/25/50/75/90, label_3d/5d_net) + 保序单调性后处理
- [E2] 痛苦预警模型 (label_pain 分类 → pain_prob)
- [E1] 概率校准 Platt → Isotonic (月度滚动重校)
- Huber loss; early_stopping patience=100 (V3.8 §2.1)
- 超参纪律: 年度贝叶斯调优 (≤50 组), 每月仅重训; E1 分位数模型沿用回归超参不单独搜索
- **每周一次全局重训 (用户 2026-07-22 裁决: 周频, 非月频)**:
  每周第一个交易日 15:30 启动 (T-1 数据), 16:00 旧模型出清单, 18:00 前切换
- 重训与清单生成解耦 (重训绝不阻塞当日清单)
"""

from __future__ import annotations

import gc
import logging
import os
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_checkpoint_module = None


def _get_checkpoint_cls():
    """Lazy-import TrainingCheckpoint to avoid circular imports."""
    global _checkpoint_module
    if _checkpoint_module is None:
        from .checkpoint import TrainingCheckpoint as _TC

        _checkpoint_module = _TC
    return _checkpoint_module


WINDOW_TOTAL = 770  # [V3.8] 设计目标窗口 (3年数据≈750交易日)
WINDOW_TRANSITION = 560  # [B11] 过渡窗口下限
B11_FULL_DEPTH = 1250  # [B11] 回填达标线 (≥5 年交易日): 达到才用 770 窗口
MIN_TRAIN_DAYS = 50  # 窗口 train 段下限, 不足即拒绝训练

# ---- 非训练段绝对下限 (统计需求推导, 不做固定比例) ----
# es:  早停验证需 ≥4000样本 → 200股/日×20日, IC_SE≈0.15/√4000≈0.002, 可检测ΔIC>0.005
# calib: isotonic分桶需 ≥10桶 → 200股/日×20日=4000样本, ~400/桶, 足够稳定
# test: OOS IC t-test → ICIR=IC_mean/IC_std×√n, IC≈0.10/0.15×√60=5.2, t>2稳健
_ES_FLOOR = 20
_CALIB_FLOOR = 20
_TEST_FLOOR = 60
_NON_TRAIN_FLOORS = _ES_FLOOR + _CALIB_FLOOR + _TEST_FLOOR  # =100


def _derive_seg_min_days(window_days: int) -> dict[str, int]:
    """从绝对统计下限 + 窗口总天数 推导各段最小日期数.

    窗口足够大 → train 段吸收余量, es/calib/test 保持绝对下限.
    窗口不足   → 等比例压缩非训练段, 保证 train 不低于 MIN_TRAIN_DAYS.
    """
    if window_days < _NON_TRAIN_FLOORS + MIN_TRAIN_DAYS:
        scale = max(window_days - MIN_TRAIN_DAYS, 1) / _NON_TRAIN_FLOORS
        return {
            "train": MIN_TRAIN_DAYS,
            "es": max(int(_ES_FLOOR * scale), 5),
            "calib": max(int(_CALIB_FLOOR * scale), 5),
            "test": max(int(_TEST_FLOOR * scale), 15),
        }
    return {
        "train": window_days - _NON_TRAIN_FLOORS,
        "es": _ES_FLOOR,
        "calib": _CALIB_FLOOR,
        "test": _TEST_FLOOR,
    }


# 全局段长下限: 用过渡窗口推导 (适用于所有短窗口场景)
SEG_MIN_DAYS = _derive_seg_min_days(WINDOW_TRANSITION)
MIN_ES_DATES = SEG_MIN_DAYS["es"]
# 兼容旧引用
TRAIN_DAYS = 620
ES_DAYS = SEG_MIN_DAYS["es"]
CALIB_DAYS = SEG_MIN_DAYS["calib"]
TEST_DAYS = SEG_MIN_DAYS["test"]  # 仅归因, 严禁反向调参
HALF_LIFE = 250  # 半衰期加权 (天)
ES_PATIENCE = 100  # [V3.8 §2.1] patience=100
# [2026-08-20] 分位头早停 patience 减半 (100→50): LightGBM 返回 best_iteration
# 树, patience 只控搜索停止点, 平坦尾段多搜的 50 轮纯浪费 (q10/25/50 实测早停
# 于 ~17-100 树 → 实际多训 100 轮无改进尾部; q75/q90 跑满 QUANTILE_MAX_TREES).
# 只作用 E1 分位数 (dual 板后段实测 13.7 分 → ~10 分, main 35.6 → ~27 分);
# 核心/rank/pain 头保持 ES_PATIENCE=100 (生产主头保守).
QUANTILE_ES_PATIENCE = 50
# [2026-08-12] 反锚 es 窗对短视界 cls 头可能过平 → best_iteration=1 常数 prob
# (main 3d_cls=1树, dual 5d_cls=3树). cls 头早停树数低于此值时重训固定轮数保底
# (num_leaves=15 强正则 + OOS test 窗兜底, 不会无界过拟合). 只作用 cls, reg 不参与.
CLS_MIN_TREES = 30
# [2026-08-17] reg 头同地板 (任务 #14): dual 3d/5d_reg es 早停 2 树 = 常数列
# (candidates_20260816 实测 pred_ret_3d std=0.0000/唯一值1), main 3d/5d 也退化
# (8/16 树, 唯一值 79/51 per 1835 只); 10d 头天然健康 (802+ 树) 不会触发.
REG_MIN_TREES = 30
OOS_IC_MIN = 0.01  # 新模型切换门槛 (signed mean IC, >0.01 即有效)

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


# 超参覆盖表 (2026-08-08 扫描定案): (board, kind) → num_leaves.
# 未列出 → 家族默认 (num_leaves=31, LightGBM 出厂). 只列有扫描证据的条目:
#   cls main 3d/5d/10d = 15: 跨视界一致赢 (IC/TOP-N 复验); 1d/2d 已删 (2026-08-09)
#   reg dual 3d/5d/10d = 15: 10d 复验双指标+扰动稳定, 5d IC 赢, 3d 平
#   pain (label_pain=3日浮亏单模型): 两板 15, 与 cls 表解耦
NUM_LEAVES_OVERRIDE: dict[tuple[str, str], int] = {
    ("main", "3d_cls"): 15,
    ("main", "5d_cls"): 15,
    ("main", "10d_cls"): 15,
    ("dual", "3d_reg"): 15,
    ("dual", "5d_reg"): 15,
    ("dual", "10d_reg"): 15,
    ("main", "pain"): 15,
    ("dual", "pain"): 15,
}

# 泛化超参覆盖表 (2026-08-17 补扫定案, WORM: legacy_dual_reg_sweep_20260817_034900):
#   dual 10d_reg = ms50_λ1: OOS 末60日 top10 超额 +2.48%→+3.41%, Rank IC 0.029→0.035,
#   3/3 子窗赢, 扰动 seed=43 不翻转. 起因 = dual 10d 排名信号弱 (日截面 corr 0.015),
#   reg 出厂默认 ms20/λ0 过拟合双创高波动噪声. 只落 dual 10d_reg 一个头, 勿扩散.
PARAMS_OVERRIDE: dict[tuple[str, str], dict] = {
    ("dual", "10d_reg"): {"min_child_samples": 50, "reg_lambda": 1.0},
}


def model_params(board: str, kind: str) -> dict:
    """按 (board, kind) 解析 LGBM 超参: 命中覆盖表则覆盖 num_leaves, 否则家族默认.

    kind 可为 "3d_cls"..."10d_cls"/"3d_reg"..."10d_reg"/"pain"/"cls"/"reg".
    """
    is_cls = kind == "pain" or kind.endswith("cls")
    base = LGB_PARAMS_CLS if is_cls else LGB_PARAMS_REG
    params = dict(base)
    nl = NUM_LEAVES_OVERRIDE.get((board, kind))
    if nl is not None:
        params["num_leaves"] = nl
    params.update(PARAMS_OVERRIDE.get((board, kind), {}))
    # [2026-08-11] 双创概率头: logloss 在 es 段早停到 1-2 树 → prob_up 全股几乎常数.
    # 改 AUC 早停 → ~20 树, prob 有真实截面区分度 (验证: dual 3d/5d/10d_cls
    # rankIC 0.041/0.060/0.035, 唯一值 45/228/242). main cls 信号强, AUC 早停反而
    # 塌缩 (51→2 树), 故只作用于 dual cls 三头.
    if board == "dual" and kind.endswith("cls"):
        params["metric"] = "auc"
    return params


MODEL_KINDS = (
    "3d_reg",
    "3d_cls",
    "5d_reg",
    "5d_cls",
    "10d_reg",
    "10d_cls",
)
EXTRA_KINDS = (
    "quantile_models_3d",
    "quantile_models_5d",
    "pain_model",
    "rank_model",
)


def risk_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter non-tradable stocks (suspended/ST) before training.

    Limit-up/down is NOT filtered here — it's an execution-time constraint
    handled by app.core.risk_filter.apply_filters, not a training-time filter.
    Models must learn limit-up/down patterns.
    """
    mask = pd.Series(True, index=df.index)
    for col in ("is_suspended",):
        if col in df.columns:
            mask &= ~df[col].astype(bool)
    return df[mask]


class DualTrackTrainer:
    """双轨训练 — 每个板块独立训练 6 个模型 (3 视界 × reg/cls)."""

    def __init__(self, model_dir: str = "models/pipeline1"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    # ---------------- 窗口切分 ----------------
    @staticmethod
    def split_window(
        df: pd.DataFrame, window_total: int = WINDOW_TOTAL
    ) -> dict[str, pd.DataFrame]:
        """四段切分 (反锚定): 训练段吸收 es, 各段全部靠向最近端.

        修复 2026-08-11: 原时序切分训练段取最老数据 (止于 ~5 个月前), 10d/双创头
        早停塌陷至 1-3 棵树 → 预测近常数. 现改为反锚定: 训练段止于 es 前
        (最近 ~80 交易日), es = 最近 20 日早停, calib = es (同窗口校准,
        物理隔离约定放宽为"校准+早停共享验证窗"), test = 最后 60 日 OOS 验收不变.
        窗口不足 (n <= test+es+MIN_TRAIN) 时回退原时序切分.
        """
        dates = sorted(df["date"].unique())[-window_total:]
        n = len(dates)
        # B11: 窗口不足时用过渡窗口推导段长 (train 段吸收余量, 可超出实际数据)
        seg_lens = _derive_seg_min_days(max(n, WINDOW_TRANSITION))
        es_d, _calib_d, test_d = seg_lens["es"], seg_lens["calib"], seg_lens["test"]
        if n > test_d + es_d + MIN_TRAIN_DAYS:
            seg_dates = {
                "train": dates[: -test_d - es_d],
                "es": dates[-test_d - es_d : -test_d],
                "calib": dates[-test_d - es_d : -test_d],  # = es, 共享验证窗校准
                "test": dates[-test_d:],
            }
        else:
            # 窗口过短: 原时序切分 (train 取前段, 保底长度)
            seg_lens["train"] = max(seg_lens["train"], MIN_TRAIN_DAYS)
            pos = 0
            seg_dates = {}
            for k in ("train", "es", "calib", "test"):
                seg_dates[k] = dates[pos : pos + seg_lens[k]]
                pos += seg_lens[k]
            seg_dates["test"] = dates[-seg_lens["test"] :]
        logger.info(
            "窗口切分(反锚): total=%d train=%d es=%d test=%d",
            n,
            len(seg_dates["train"]),
            len(seg_dates["es"]),
            len(seg_dates["test"]),
        )
        return {k: df[df["date"].isin(v)] for k, v in seg_dates.items()}

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

    # ---------------- 单模型训练 (低内存路径: float32 + gc) ----------------
    @staticmethod
    def _resolve_label(kind: str, columns) -> str | None:
        """解析 kind 的有效标签列 ([B9] PM / [E5] 净收益优先); 缺失返回 None.

        返回 None = 该视界标签在当前面板不可用 → 跳过训练 (向后兼容旧面板,
        与 _train_extras 的"标签缺失时跳过"同一惯例). 缺失只发生在旧面板 /
        测试夹具只构造部分视界; 生产面板由 label_engine 全量生成, 10 种齐备.
        """
        label = {
            "3d_reg": "label_3d",
            "3d_cls": "label_3d_cls",
            "5d_reg": "label_5d",
            "5d_cls": "label_5d_cls",
            "10d_reg": "label_10d",
            "10d_cls": "label_10d_cls",
        }[kind]
        # [B9] PM 执行口径验收标签优先 (label_pm_kd / label_pm_kd_cls), 缺失时回退研究口径
        if kind.endswith("cls"):
            pm_label = f"label_pm_{kind.split('d')[0]}d_cls"
        else:
            pm_label = f"label_pm_{kind.split('d')[0]}d"
        if pm_label in columns:
            label = pm_label
        # [E5] 净收益标签 (分层滑点) 优先于毛收益 — 训练/验收主标签口径 (D1)
        if f"{label}_net" in columns:
            label = f"{label}_net"
        return label if label in columns else None

    def _train_one(
        self,
        kind: str,
        segs: dict[str, pd.DataFrame],
        feature_cols: list[str],
        board: str,
    ):
        import gc

        import lightgbm as lgb

        label = self._resolve_label(kind, segs["train"].columns)
        if label is None:
            raise RuntimeError(
                f"[{kind}] 标签列缺失, 不应到达 _train_one (train_window 已守卫)"
            )
        train = segs["train"].dropna(subset=[label])  # per-model dropna (安全网 #7)
        es = segs["es"].dropna(subset=[label])
        train = risk_filter(train)
        es = risk_filter(es)

        # ── 内存: float32 下转 (混合 dtype 的 DataFrame.values 会 upcast 到 float64) ──
        cols_present = [c for c in feature_cols if c in train.columns]
        if cols_present:
            train[cols_present] = train[cols_present].astype("float32", copy=False)
        cols_es_present = [c for c in feature_cols if c in es.columns]
        if cols_es_present:
            es[cols_es_present] = es[cols_es_present].astype("float32", copy=False)

        X = np.nan_to_num(train[cols_present].values, nan=0.0)
        y = train[label].values
        X_es = np.nan_to_num(es[cols_es_present].values, nan=0.0)
        y_es = es[label].values
        w = self.time_weights(train)

        # 释放 DataFrame 引用 (X/y 已提取为 numpy 数组)
        del train, es
        gc.collect()

        if kind.endswith("cls"):
            model = lgb.LGBMClassifier(**model_params(board, kind))
        else:
            model = lgb.LGBMRegressor(**model_params(board, kind))
        # es_dates 从 segs 取 (train/es DataFrame 已释放, segs 仍持有 es 引用)
        es_dates = (
            segs["es"]["date"].nunique() if "date" in segs["es"].columns else len(y_es)
        )
        use_es = es_dates >= MIN_ES_DATES
        if not use_es:
            logger.warning(
                "[%s/%s] ES 集仅 %d 交易日 (< %d), 跳过早停",
                kind,
                label,
                es_dates,
                MIN_ES_DATES,
            )
        try:
            model.fit(
                X,
                y,
                sample_weight=w,
                eval_set=[(X_es, y_es)] if use_es else None,
                callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)]
                if use_es
                else None,
            )
            # [2026-08-12] cls 最小树保底: es 窗过平 → 早停 1 树 = 常数 prob.
            # [2026-08-17] reg 同地板 (REG_MIN_TREES): dual 3d/5d_reg 早停 2 树 =
            # 常数列 (std=0.0000), main 3d/5d 也退化 (8/16 树). 以固定轮数重训
            # (无早停), 保证预测有真实截面区分度.
            if use_es:
                floor = CLS_MIN_TREES if kind.endswith("cls") else REG_MIN_TREES
                bi = getattr(model, "best_iteration_", None)
                if bi is not None and bi < floor:
                    logger.warning(
                        "[%s/%s] 早停仅 %d 树 (< %d), 重训 %d 树保底",
                        board,
                        kind,
                        bi,
                        floor,
                        floor,
                    )
                    p = model_params(board, kind)
                    p["n_estimators"] = floor
                    make = (
                        lgb.LGBMClassifier
                        if kind.endswith("cls")
                        else lgb.LGBMRegressor
                    )
                    model = make(**p)
                    model.fit(X, y, sample_weight=w)
        except Exception as exc:
            logger.error("模型训练失败 [%s/%s]: %s", kind, label, exc)
            raise
        finally:
            # 训练完成后释放 numpy 数组 (模型已持有需要的内部状态)
            del X, y, X_es, y_es, w
            gc.collect()
        return model, label

    # ---------------- 窗口训练 (单板块 4 模型 + E1/E2, 支持断点续训) ----------------
    def train_window(
        self,
        df: pd.DataFrame,
        board: str,
        feature_cols: list[str],
        checkpoint=None,  # TrainingCheckpoint | None
    ) -> dict:
        """训练一个板块的 6 个模型 (3 视界 × reg/cls) + E1 分位数五模型(3d/5d) + E2 痛苦预警 (标签齐备时).

        固定使用 WINDOW_TOTAL 窗口 (B11 过渡逻辑已废弃).
        段长按比例动态分配 (split_window), 保证 es/calib/test 各 >= 最小值.

        若提供 checkpoint 且存在, 跳过已完成的模型种类 (断点续训).
        每完成一个模型种类即原子写入 checkpoint (crash-safe).
        """
        depth = int(df["date"].nunique()) if len(df) else 0
        window = WINDOW_TOTAL  # B11 过渡逻辑已废弃, 始终使用全窗口
        min_non_train = sum(v for k, v in SEG_MIN_DAYS.items() if k != "train")
        if depth < min_non_train + MIN_TRAIN_DAYS:
            raise RuntimeError(
                f"[{board}] 训练样本深度不足: {depth} 交易日 "
                f"(需 ≥ {ES_DAYS + CALIB_DAYS + TEST_DAYS + MIN_TRAIN_DAYS})"
            )
        segs = self.split_window(df, window)
        # 内存 (2026-08-13): split_window 已把数据拷贝进 segs, df (整块增强帧 ~3GB)
        # 不再被引用 → 立即释放, 否则训练全程驻留导致 main OOM
        del df
        gc.collect()

        # ── 内存: 只保留训练/校准/OOS 需要的列, 释放 OHLCV 原始列 ──
        keep_cols = set(feature_cols) | {
            "symbol",
            "date",
            "board",
            "is_suspended",
            "close_hfq",  # IC 衰减曲线 (ic_decay_curve) 下游需要
        }
        for seg_name, seg_df in segs.items():
            label_cols_in_seg = [c for c in seg_df.columns if c.startswith("label_")]
            keep_cols.update(label_cols_in_seg)
            drop_cols = [c for c in seg_df.columns if c not in keep_cols]
            if drop_cols:
                segs[seg_name] = seg_df.drop(columns=drop_cols)
                logger.debug(
                    "[%s] segs[%s] 释放 %d 列 (保留 %d)",
                    board,
                    seg_name,
                    len(drop_cols),
                    len(segs[seg_name].columns),
                )
        gc.collect()
        out = {
            "board": board,
            "feature_cols": feature_cols,
            "models": {},
            "segs": segs,
            "_window_total": window,
        }

        # ── 断点续训: 从 checkpoint 恢复已训练的模型 ──
        if checkpoint is not None and checkpoint.exists():
            partial = checkpoint.load_partial()
            if partial is not None and "models" in partial:
                out["models"] = partial["models"]
                # 恢复 extras
                for ek in EXTRA_KINDS:
                    if ek in partial:
                        out[ek] = partial[ek]
                logger.info(
                    "[%s] 从 checkpoint 恢复: models=%s extras=%s",
                    board,
                    list(out["models"].keys()),
                    [k for k in EXTRA_KINDS if k in out],
                )

        # ── 训练未完成的模型种类 ──
        for kind in MODEL_KINDS:
            if checkpoint is not None and kind in (checkpoint.completed_kinds or []):
                logger.info("[%s] %s — checkpoint 已完成, 跳过", board, kind)
                continue
            if self._resolve_label(kind, segs["train"].columns) is None:
                logger.info("[%s] %s — 标签列缺失, 跳过", board, kind)
                continue
            model, label = self._train_one(kind, segs, feature_cols, board)
            out["models"][kind] = (model, label)
            logger.info(
                "[%s] %s 训练完成, 样本 %d",
                board,
                kind,
                len(segs["train"].dropna(subset=[label])),
            )
            gc.collect()
            # 每完成一个模型种类立即写入 checkpoint
            if checkpoint is not None:
                self._save_checkpoint(out, checkpoint)

        # ── 训练未完成的 extras ──
        self._train_extras(out, checkpoint=checkpoint)
        return out

    @staticmethod
    def _save_checkpoint(out: dict, checkpoint) -> None:
        """Save incremental checkpoint (called after each model kind/extra)."""
        from .checkpoint import EXTRA_KINDS

        completed_kinds = list(out.get("models", {}).keys())
        completed_extras = [k for k in EXTRA_KINDS if k in out]
        checkpoint.save_progress(
            out, completed_kinds=completed_kinds, completed_extras=completed_extras
        )

    # ---------------- E1/E2 + LambdaRank: 分位数 + 痛苦预警 + 排序 (支持断点续训) ----------------
    def _train_extras(self, out: dict, checkpoint=None) -> None:
        """[E1] 分位数五模型 (3d/5d 净标签) + [E2] 痛苦预警 (label_pain)
        + [阶段四] LambdaRank 排序模型 (lambdarank_truncation_level=25, 分位 gain).

        E1 沿用回归超参, 不单独搜索 (V3.8 §2.2, 避免调参维度爆炸).
        标签缺失时跳过 (向后兼容旧面板).

        若提供 checkpoint, 跳过已完成的 extra 种类.
        每完成一个 extra 种类即写入 checkpoint.
        """
        from .quantile_models import PainModel, QuantileModelSet

        segs, cols = out["segs"], out["feature_cols"]

        def _xy(seg_name: str, label: str):
            sub = risk_filter(segs[seg_name].dropna(subset=[label]))
            # float32 下转: 避免混合 dtype 导致 .values upcast 到 float64
            cols_present = [c for c in cols if c in sub.columns]
            if cols_present:
                sub[cols_present] = sub[cols_present].astype("float32", copy=False)
            X = np.nan_to_num(sub[cols_present].values, nan=0.0)
            return sub, X, sub[label].values

        # 哪些 extras 已完成 (从 checkpoint)
        done_extras = (
            set(checkpoint.completed_extras or []) if checkpoint is not None else set()
        )

        # E1: 多视界分位数五模型 (3d/5d; 1d/2d 2026-08-09 删除)
        # label 偏好 label_pm_{k}d_net → label_{k}d_net → label_{k}d (与 _train_one 同口径)
        for horizon in (3, 5):
            q_label = next(
                (
                    c
                    for c in (
                        f"label_pm_{horizon}d_net",
                        f"label_{horizon}d_net",
                        f"label_{horizon}d",
                    )
                    if c in segs["train"].columns
                ),
                None,
            )
            if q_label is None:
                continue
            ranker_label = (
                q_label  # 记录最后解析到的标签 (循环末=5d), 供循环后单训 ranker
            )
            qkey = f"quantile_models_{horizon}d"
            if qkey not in done_extras:
                try:
                    train, X, y = _xy("train", q_label)
                    _, X_es, y_es = _xy("es", q_label)
                    # E1 沿用回归超参 (objective 由 QuantileModelSet 按分位设置)
                    params = {
                        k: v for k, v in LGB_PARAMS_REG.items() if k != "objective"
                    }
                    qset = QuantileModelSet(params).fit(
                        X,
                        y,
                        sample_weight=self.time_weights(train),
                        eval_set=(X_es, y_es) if len(y_es) else None,
                        es_patience=QUANTILE_ES_PATIENCE,
                    )
                    qset.label_ = q_label
                    out[qkey] = qset
                    logger.info(
                        "[%s] E1 分位数五模型训练完成 (label=%s, %dd)",
                        out["board"],
                        q_label,
                        horizon,
                    )
                except Exception as e:
                    logger.warning(
                        "[%s] E1 分位数模型 (%dd) 训练失败: %s",
                        out["board"],
                        horizon,
                        e,
                    )
                del train, X, y, X_es, y_es
                gc.collect()
                if checkpoint is not None:
                    self._save_checkpoint(out, checkpoint)
            else:
                logger.info(
                    "[%s] E1 分位数模型 (%dd) — checkpoint 已完成, 跳过",
                    out["board"],
                    horizon,
                )

        # 阶段四: LambdaRank (标签=净收益截面分位 gain 0-4, group=date)
        # [2026-08-20] 只训一次 (原在 horizon 循环内训两次: 3d 版被 5d 版覆盖, 纯浪费
        # main ~3 分/dual ~0.7 分每次重训). 标签 = 最后解析到的 q_label (生产=5d),
        # 与现状最终落盘 bundle 完全一致. done_extras 快照取自循环前, 判一次即可.
        if ranker_label is not None and "rank_model" not in done_extras:
            try:
                out["rank_model"] = self._train_ranker(out, ranker_label)
                logger.info("[%s] LambdaRank 排序模型训练完成", out["board"])
            except Exception as e:
                logger.warning("[%s] LambdaRank 训练失败: %s", out["board"], e)
            if checkpoint is not None:
                self._save_checkpoint(out, checkpoint)
        else:
            logger.info("[%s] LambdaRank — checkpoint 已完成, 跳过", out["board"])

        # E2 痛苦预警
        if "label_pain" in segs["train"].columns:
            if "pain_model" not in done_extras:
                try:
                    train, X, y = _xy("train", "label_pain")
                    _, X_es, y_es = _xy("es", "label_pain")
                    params = {
                        k: v
                        for k, v in model_params(out["board"], "pain").items()
                        if k != "objective"
                    }
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
                del train, X, y, X_es, y_es
                gc.collect()
                if checkpoint is not None:
                    self._save_checkpoint(out, checkpoint)
            else:
                logger.info("[%s] E2 痛苦预警 — checkpoint 已完成, 跳过", out["board"])

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
            # float32 下转
            cols_present = [c for c in cols if c in sub.columns]
            if cols_present:
                sub[cols_present] = sub[cols_present].astype("float32", copy=False)
            gains = (
                sub.groupby("date")[label]
                .rank(pct=True)
                .pipe(lambda s: (s * 5).clip(0, 4.999).astype(int))
            )
            group = sub.groupby("date").size().values
            X = np.nan_to_num(sub[cols_present].values, nan=0.0)
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
        """用校准集 (与早停物理隔离) 拟合校准器 (安全网: 严禁原始 predict_proba).

        [E1/V3.8] 恒用 Platt Scaling (08-06 定案: isotonic 阶跃会把生产概率带
        压成单平台 → prob_up 常数化, 见 _recalibrate_legacy_cls.py; 旧条件分支
        校准集 ≥20 日回退 isotonic 已在 08-17..08-23 每日清单复现平台坍缩).
        [多视界] 每个 cls 视界 (3/5/10d; 1d/2d 2026-08-09 删除) 一个 ProbCalibrator, 存
        trained["calibrators"] = {k: cal}; 向后兼容: trained["calibrator"] = 3d 别名.
        """
        from .label_engine import LABEL_HORIZONS
        from .prob_calibrator import ProbCalibrator

        calibrators = {}
        for k in LABEL_HORIZONS:
            kind = f"{k}d_cls"
            if kind not in trained["models"]:
                continue  # 该视界未训练 (标签列缺失) → 不注册校准器
            model, label = trained["models"][kind]
            calib = trained["segs"]["calib"].dropna(subset=[label])
            cols = trained["feature_cols"]
            # 校准集可能为空 (小窗口 + 多特征列 NaN), 此时跳过校准用原始 prob
            if len(calib) == 0:
                logger.warning(
                    "[%s] 校准集为空 (label=%s, %s), 跳过校准, 使用原始 predict_proba",
                    trained.get("board", "?"),
                    label,
                    kind,
                )
                calibrators[k] = None
                continue
            raw = model.predict_proba(np.nan_to_num(calib[cols].values, nan=0.0))[:, 1]
            calibrators[k] = ProbCalibrator(method="platt").fit(raw, calib[label].values)
        trained["calibrators"] = calibrators
        trained["calibrator"] = calibrators.get(3)  # 3d 别名 (主概率, 1d 已删)
        return trained["calibrator"]

    # ---------------- OOS 验证 + 切换 ----------------
    def validate_oos(self, trained: dict, ic_min: float = OOS_IC_MIN) -> dict:
        """测试段 Rank IC (仅月度归因段). IC >= 0.03 才允许切换新模型.

        切换判据 = 跨视界加权 IC (LABEL_WEIGHTS): 各回归模型 IC 按权重求和,
        1d 最不可执行 (T+1 买入当日不可卖) 权重最低, 3d 历史预测力最强.
        """
        from .ic_screener import ICScreener
        from .label_engine import LABEL_WEIGHTS

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
        # 跨视界加权 IC (回归模型; 分类分不直接贡献收益率)
        total_w = sum(LABEL_WEIGHTS.values())
        weighted_ic = (
            sum(LABEL_WEIGHTS[k] * ics.get(f"{k}d_reg", 0.0) for k in LABEL_WEIGHTS)
            / total_w
        )
        return {
            "ics": ics,
            "weighted_ic": weighted_ic,
            "pass": weighted_ic >= ic_min,
            "best_ic_key": max(ics, key=lambda k: ics.get(k, 0.0)),
        }

    def save(self, trained: dict, tag: str) -> str:
        """保存模型包 (含校准器; 若无则先拟合)."""
        if "calibrators" not in trained:
            self.fit_calibrator(trained)
        path = os.path.join(self.model_dir, f"{trained['board']}_{tag}.pkl")
        bundle = {
            "board": trained["board"],
            "feature_cols": trained["feature_cols"],
            "models": trained["models"],
            "calibrator": trained["calibrator"],
        }
        for (
            extra
        ) in (  # 多视界校准器 + E1/E2/排序 (quantile 3d/5d 必须落盘, gate3 用其中位数)
            "calibrators",
            "quantile_models_3d",
            "quantile_models_5d",
            "pain_model",
            "rank_model",
        ):
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
        from app.utils.safe_load import safe_pickle_load

        return safe_pickle_load(path)

    # ---------------- 特征相似度回退 ----------------
    @staticmethod
    def feature_similarity_check(trained: dict, threshold: float = 0.8) -> bool:
        """三回归模型 importance 排名 Spearman > 0.8 → 高度相似, 建议回退单模型+多输出."""
        from scipy.stats import spearmanr

        imps = []
        for kind in ("3d_reg", "5d_reg", "10d_reg"):
            model, _ = trained["models"][kind]
            imps.append(pd.Series(model.feature_importances_).rank())
        corrs = [
            spearmanr(imps[i], imps[j]).statistic
            for i in range(3)
            for j in range(i + 1, 3)
        ]
        return bool(np.nanmean(corrs) > threshold)

    # ---------------- 每周全局重训 (解耦, 支持断点续训) ----------------
    def weekly_retrain(
        self,
        panels: dict[str, pd.DataFrame],
        feature_cols_by_board: dict[str, list[str]],
        tag: str,
        resume: bool = False,
    ) -> dict:
        """每周一次全局重训 (用户 2026-07-22 裁决: 周频全局训练).

        panels: {'main': 主板750日面板, 'dual': 双创750日面板}.
        每周第一个交易日 15:30 启动, 与 16:00 清单生成并行.

        resume=True: 从 checkpoint 断点续训, 跳过已完成的模型种类.
        训练完成后自动清理 checkpoint 文件.

        Returns:
            {board: {'path', 'oos': {...}, 'switched': bool}}
        """
        TC = _get_checkpoint_cls()
        results = {}
        for board, df in panels.items():
            ck = TC(self.model_dir, board, tag) if resume else None

            if ck is not None and ck.exists():
                remaining = ck.remaining_kinds()
                logger.info(
                    "[%s] 断点续训: 已完成 %s, 剩余 %s",
                    board,
                    ck.completed_kinds,
                    remaining,
                )
                if not remaining and not ck.remaining_extras():
                    logger.info("[%s] 全部模型已完成, 仅执行 OOS 验证 + 归档", board)

            trained = self.train_window(
                df, board, feature_cols_by_board[board], checkpoint=ck
            )
            oos = self.validate_oos(trained)
            path = self.save(trained, tag)

            # 切换决策: OOS 合格才切换, 否则保留旧模型 + 告警
            results[board] = {"path": path, "oos": oos, "switched": oos["pass"]}
            if not oos["pass"]:
                logger.warning(
                    "[%s] 新模型 OOS weighted_IC=%.4f < %.2f, 保留旧模型",
                    board,
                    oos.get("weighted_ic", 0.0),
                    OOS_IC_MIN,
                )

            # 训练成功 → 清理 checkpoint (原子归档完成)
            if ck is not None:
                ck.clear()
                logger.info("[%s] checkpoint 已清理 (归档完成)", board)

        return results

    # 兼容别名 (V3.5 原文为月度, 用户 2026-07-22 裁决改为周频)
    monthly_retrain = weekly_retrain
