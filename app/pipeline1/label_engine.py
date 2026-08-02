"""
标签引擎 (DESIGN §14.1 安全网 #1/#7/#13, PIPELINE1_V3.8 §〇.1)
=================================================================
- 一律后复权价 (hfq); 早盘 pipeline 标签独立 (open(T+1) 基准)
- [B9] 晚盘验收标签 label_pm_kd = close(T+1+k)/price_1455(T+1)-1 (执行口径, 严格 T+1);
  旧 close(T)→close(T+k) 口径仅保留研究对照
- [E5] 滑点分层 (ADV20 三档 0.05%/0.10%/0.15%) + 净收益标签 label_*_net
  (主标签=可执行净收益, 沿用 D1: 毛收益 - COST - 2×分层滑点)
- [E2] 路径依赖标签 label_mdd_1d/3d/5d (期间最大浮亏) + label_pain (3日浮亏>5%)
- 横截面 0.1%/99.9% 缩尾 (B2); [B18] is_virtual=1 退市虚拟样本豁免缩尾
- 停牌污染置 NaN; 实盘训练遮蔽最近 6 天; 各模型各自 dropna (per-model), 不统一剔除
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

LABEL_HORIZONS = (1, 2, 3, 5)
# 评估/排序跨视界权重 (用户 2026-08-02 裁决: 预测 1/2/3/5d 并按权重评估).
# 1d 权重最低 — T+1 制度下买入后当日不可卖, 最不可执行; 3d 历史预测力最强.
# 修改此字典即全局生效 (validate_oos 加权 IC + predictor/list_generator 综合分).
LABEL_WEIGHTS = {1: 0.15, 2: 0.25, 3: 0.35, 5: 0.25}
CLS_THRESHOLD = 0.005  # +0.5% 覆盖双边成本 (佣金万2.5x2 + 印花税0.05% + 滑点0.05% ≈ 0.13%, 留安全垫)
COST = 0.0013  # round-trip 费用: 佣金万2.5双边 + 印花税0.05%卖出 (E5 净标签口径)
# E5 滑点分层 (按 ADV20): >5亿→0.05% / 1~5亿→0.10% / <1亿→0.15% (双边计入)
ADV20_TIER_HIGH = 5e8
ADV20_TIER_MID = 1e8
PAIN_THRESHOLD = -0.05  # E2: 3日内浮亏超5% → label_pain=1


def _label_reference(s: pd.Series, k: int) -> pd.Series:
    """Reference future values for label construction ONLY (never for features).

    Labels legitimately reference future prices - this is NOT look-ahead bias.
    This function is ONLY for label construction, never for feature computation.
    Uses numpy array slicing (no shift(-k)).
    """
    vals = s.values
    n = len(vals)
    result = np.full(n, np.nan, dtype=float)
    if n > k:
        result[: n - k] = vals[k:]
    return pd.Series(result, index=s.index)


def _safe_divide(numerator, denominator):
    """Safe division (zero-division guard): NaN where denominator is 0."""
    return numerator / denominator.replace(0, np.nan)


def slippage_tier(adv20: float) -> float:
    """E5 滑点分层: ADV20 > 5亿 → 0.05%; 1~5亿 → 0.10%; < 1亿 → 0.15% (单边)."""
    if pd.isna(adv20):
        return 0.0015  # ADV 未知按最差档 (保守)
    if adv20 > ADV20_TIER_HIGH:
        return 0.0005
    if adv20 > ADV20_TIER_MID:
        return 0.0010
    return 0.0015


def _future_window_min(vals: np.ndarray, start: int, window: int) -> np.ndarray:
    """min(vals[T+start .. T+start+window-1]) — 标签专用 (合法引用未来, 非特征).

    窗口内含 NaN → 结果 NaN (保守: 停牌/缺数据污染的路径标签不可用).
    """
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    if n >= start + window:
        w = sliding_window_view(vals[start:], window)
        m = np.min(w, axis=1)  # NaN 传播 (刻意, 不用 nanmin)
        out[: len(m)] = m
    return out


class LabelEngine:
    """标签定义. 铁律: 输入 df 必须已 sort_values(['symbol','date']) (安全网 #13)."""

    # ---------------- 主标签 ----------------
    @staticmethod
    def build_labels(df: pd.DataFrame, session: str = "PM") -> pd.DataFrame:
        """label_kd = close_hfq[T+k] / close_hfq[T] - 1, k=1/3/5 (groupby symbol!).

        session="PM": T 收盘 -> T+k 收盘 (研究对照口径)
        session="AM": open(T+1) -> close(T+k) (早盘买入 pipeline, T+1 制度最早 T+2 卖)

        [B9 执行口径] PM 验收标签: label_pm_kd = close_hfq(T+1+k) / price_1455(T+1) - 1
        (清单 T 日盘后生成, T 日成交=时间旅行; Rank IC/ICIR/胜率等验收指标一律按
         label_pm_kd 计算, 旧 label_kd 口径仅保留研究对照, 旧口径回测结果作废).
        [日K近似] 日线无 14:55 价, price_1455 列缺失时用 close(T+1) 代替
        (与 backtest_v35 日K近似口径一致).
        """
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        g = df.groupby("symbol")["close_hfq"]
        for k in LABEL_HORIZONS:
            future_close = g.transform(lambda s, kk=k: _label_reference(s, kk))
            df[f"label_{k}d"] = _safe_divide(future_close, df["close_hfq"]) - 1
        if session == "AM":
            g_open = df.groupby("symbol")["open"]
            for k in LABEL_HORIZONS:
                future_close = g.transform(lambda s, kk=k: _label_reference(s, kk))
                next_open = g_open.transform(lambda s: _label_reference(s, 1))
                df[f"label_{k}d"] = _safe_divide(future_close, next_open) - 1
        else:
            # B9: 晚盘执行口径标签 (验收权威)
            if "price_1455" in df.columns:
                exec_px = df.groupby("symbol")["price_1455"].transform(
                    lambda s: _label_reference(s, 1)
                )
            else:
                exec_px = g.transform(lambda s: _label_reference(s, 1))  # 日K近似
            for k in LABEL_HORIZONS:
                future_close = g.transform(lambda s, kk=k + 1: _label_reference(s, kk))
                df[f"label_pm_{k}d"] = _safe_divide(future_close, exec_px) - 1
            # [B9] PM 执行口径分类标签: label_pm_cls 基于 PM 验收标签, 非研究口径 label_1d
            df["label_pm_cls"] = (df["label_pm_1d"] > CLS_THRESHOLD).astype("float")
            df.loc[df["label_pm_1d"].isna(), "label_pm_cls"] = np.nan
        # 研究口径分类标签 (PM 存在时 label_pm_cls 优先; 仅作回退)
        df["label_cls"] = (df["label_1d"] > CLS_THRESHOLD).astype("float")
        df.loc[df["label_1d"].isna(), "label_cls"] = np.nan
        df = LabelEngine.add_net_labels(df)
        return df

    # ---------------- E5: 净收益标签 (分层滑点) ----------------
    @classmethod
    def add_net_labels(cls, df: pd.DataFrame) -> pd.DataFrame:
        """[E5] 主标签=可执行净收益 (沿用 D1): label_*_net = 毛收益 - COST - 2×分层滑点.

        滑点按 ADV20 三档 (slippage_tier); ADV20 缺失时由 amount 20 日均值现算
        (groupby symbol, 仅用历史 — adv20 是 t 日及以前的均值, 无未来函数).
        毛收益标签保留作研究对照; 训练/验收一律用 *_net 口径.
        """
        if "adv20" not in df.columns and "amount" in df.columns:
            df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
            df["adv20"] = (
                df.groupby("symbol")["amount"]
                .rolling(20, min_periods=20)
                .mean()
                .reset_index(level=0, drop=True)
            )
        slip = df["adv20"].map(slippage_tier) if "adv20" in df.columns else 0.0015
        cost_total = COST + 2 * slip
        for k in LABEL_HORIZONS:
            for prefix in (f"label_{k}d", f"label_pm_{k}d"):
                if prefix in df.columns:
                    df[f"{prefix}_net"] = df[prefix] - cost_total
        # [B9] PM 执行口径分类净标签 (label_pm_1d_net > 0 → 净收益为正)
        if "label_pm_1d_net" in df.columns:
            df["label_pm_cls_net"] = (df["label_pm_1d_net"] > 0).astype("float")
            df.loc[df["label_pm_1d_net"].isna(), "label_pm_cls_net"] = np.nan
        if "label_1d_net" in df.columns:
            df["label_cls_net"] = (df["label_1d_net"] > 0).astype("float")
            df.loc[df["label_1d_net"].isna(), "label_cls_net"] = np.nan
        return df

    # ---------------- E2: 路径依赖标签 (期间最大浮亏) ----------------
    @staticmethod
    def build_path_labels(df: pd.DataFrame) -> pd.DataFrame:
        """[E2] label_mdd_1d/3d/5d: 持有期内最大浮亏 (相对 T+1 收盘≈执行价).

        label_mdd_1d = min(low_hfq[T+1..T+2]) / close_hfq[T+1] - 1  (首个持有日内)
        label_mdd_3d = min(low_hfq[T+1..T+4]) / close_hfq[T+1] - 1
        label_mdd_5d = min(low_hfq[T+1..T+6]) / close_hfq[T+1] - 1
        label_pain   = 1 if label_mdd_3d < -5% else 0  (痛苦标签, 训练痛苦预警模型)

        实盘最伤人的不是期末小亏, 而是期间浮亏触发日内引擎被迫止损 ——
        止损后期末涨回来与你无关.
        """
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)  # 安全网 #13
        low_col = "low_hfq" if "low_hfq" in df.columns else "low"
        g_low = df.groupby("symbol")[low_col]
        g_close = df.groupby("symbol")["close_hfq"]
        exec_close = g_close.transform(lambda s: _label_reference(s, 1))
        for n, h in ((1, 2), (3, 4), (5, 6)):
            min_low = g_low.transform(
                lambda s, hh=h: pd.Series(
                    _future_window_min(s.values, 1, hh), index=s.index
                )
            )
            df[f"label_mdd_{n}d"] = _safe_divide(min_low, exec_close) - 1
        df["label_pain"] = (df["label_mdd_3d"] < PAIN_THRESHOLD).astype("float")
        df.loc[df["label_mdd_3d"].isna(), "label_pain"] = np.nan
        return df

    # ---------------- 缩尾 ----------------
    @staticmethod
    def winsorize_cross_section(
        df: pd.DataFrame, lower: float = 0.001, upper: float = 0.999
    ) -> pd.DataFrame:
        """横截面分位缩尾 (按 date 分组), B2: 0.1%/99.9% 仅防数据错误, 保留尾部真实收益.

        [B18] is_virtual=1 (D20 退市虚拟样本) 豁免: 不参与分位计算也不被 clip —
        否则 -50% 被剪至当日 0.1% 分位, "让模型学习归零"名存实亡.
        """
        label_cols = [f"label_{k}d" for k in LABEL_HORIZONS]
        label_cols += [
            f"label_pm_{k}d" for k in LABEL_HORIZONS if f"label_pm_{k}d" in df
        ]
        label_cols += [
            c for c in df.columns if c.endswith("_net") and "_cls" not in c
        ]  # E5 净标签 (排除二分类 *_cls_net)
        label_cols += [
            f"label_mdd_{k}d" for k in LABEL_HORIZONS if f"label_mdd_{k}d" in df
        ]
        if "is_virtual" in df.columns:
            real = df["is_virtual"] != 1
        else:
            real = pd.Series(True, index=df.index)
        for col in label_cols:
            quantiles = df[real].groupby("date")[col].quantile([lower, upper]).unstack()
            lo = df["date"].map(quantiles[lower])
            hi = df["date"].map(quantiles[upper])
            df.loc[real, col] = df.loc[real, col].clip(lo[real], hi[real])
        return df

    # ---------------- 停牌污染 ----------------
    @staticmethod
    def mask_suspension(df: pd.DataFrame) -> pd.DataFrame:
        """T 到 T+N 区间内存在停牌 → label_Nd 置 NaN (脏标签: 复牌价差非真实持有收益).

        Note: ST/suspended stock filtering is handled by risk_filter in
        dual_track_trainer before training, not here. This method only masks
        labels contaminated by suspension gaps.
        """
        for n in LABEL_HORIZONS:
            rolling_sum = (
                df.groupby("symbol")["is_suspended"]
                .rolling(n + 1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            # Label masking: check if suspension exists in [T, T+n] window
            vals = rolling_sum.values
            length = len(vals)
            suspended_vals = np.zeros(length, dtype=bool)
            if length > n:
                suspended_vals[: length - n] = vals[n:] > 0
            suspended = pd.Series(suspended_vals, index=rolling_sum.index)
            df[f"label_{n}d"] = df[f"label_{n}d"].where(~suspended, np.nan)
            if f"label_{n}d_net" in df.columns:  # E5 净标签同步遮蔽
                df[f"label_{n}d_net"] = df[f"label_{n}d_net"].where(~suspended, np.nan)
            # E2: mdd 路径窗口 [T+1, T+n+1] ⊂ [T, T+n+1]
            if f"label_mdd_{n}d" in df.columns:
                rolling_mdd = (
                    df.groupby("symbol")["is_suspended"]
                    .rolling(n + 2)
                    .sum()
                    .reset_index(level=0, drop=True)
                )
                vals_m = rolling_mdd.values
                susp_m = np.zeros(len(vals_m), dtype=bool)
                if len(vals_m) > n + 1:
                    susp_m[: len(vals_m) - n - 1] = vals_m[n + 1 :] > 0
                susp_m = pd.Series(susp_m, index=rolling_mdd.index)
                df[f"label_mdd_{n}d"] = df[f"label_mdd_{n}d"].where(~susp_m, np.nan)
        if "label_mdd_3d" in df.columns:
            df["label_pain"] = (df["label_mdd_3d"] < PAIN_THRESHOLD).astype("float")
            df.loc[df["label_mdd_3d"].isna(), "label_pain"] = np.nan
        df["label_cls"] = df["label_cls"].where(df["label_1d"].notna(), np.nan)
        if "label_pm_cls" in df.columns:
            df["label_pm_cls"] = df["label_pm_cls"].where(
                df["label_pm_1d"].notna(), np.nan
            )
        if "label_cls_net" in df.columns:
            df["label_cls_net"] = df["label_cls_net"].where(
                df["label_1d_net"].notna(), np.nan
            )
        if "label_pm_cls_net" in df.columns:
            df["label_pm_cls_net"] = df["label_pm_cls_net"].where(
                df["label_pm_1d_net"].notna(), np.nan
            )
        return df

    # ---------------- 实盘标签遮蔽 ----------------
    @staticmethod
    def mask_recent_days(df: pd.DataFrame, days: int = 6) -> pd.DataFrame:
        """实盘训练剔除最近 N 天 (V3.8: 6 天 — label_5d 需 T+6 收盘价, 标签未生成)."""
        df["date"].max() - pd.Timedelta(days=days * 2)  # 自然日宽松上界
        recent_dates = sorted(df["date"].unique())[-days:]
        mask = df["date"].isin(recent_dates)
        cols = [f"label_{k}d" for k in LABEL_HORIZONS] + ["label_cls", "label_pm_cls"]
        cols += [
            f"label_pm_{k}d" for k in LABEL_HORIZONS if f"label_pm_{k}d" in df.columns
        ]
        cols += [c for c in df.columns if c.endswith("_net")]
        cols += [c for c in df.columns if c.startswith("label_mdd_")] + ["label_pain"]
        for col in [c for c in cols if c in df.columns]:
            df.loc[mask, col] = np.nan
        return df

    # ---------------- per-model dropna ----------------
    @staticmethod
    def per_model_dropna(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """各模型各自丢弃缺失标签 (不统一剔除最后 5 天).

        Returns:
            {'1d': df_1d, '3d': df_3d, '5d': df_5d, 'cls': df_cls,
             'pain': df_pain (E2, 若 label_pain 存在)}
        """
        cls_label = "label_pm_cls" if "label_pm_cls" in df.columns else "label_cls"
        out = {
            "1d": df.dropna(subset=["label_1d"]),
            "2d": df.dropna(subset=["label_2d"]),
            "3d": df.dropna(subset=["label_3d"]),
            "5d": df.dropna(subset=["label_5d"]),
            "cls": df.dropna(subset=[cls_label]),
        }
        if "label_pain" in df.columns:  # E2 痛苦预警模型
            out["pain"] = df.dropna(subset=["label_pain"])
        return out
