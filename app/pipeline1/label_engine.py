"""
标签引擎 (DESIGN §14.1 安全网 #1/#7/#13, PIPELINE1_V3.5 §〇.1)
=================================================================
- 一律后复权价 (hfq); 早盘 pipeline 标签独立 (open(T+1) 基准)
- [B9] 晚盘验收标签 label_pm_kd = close(T+1+k)/price_1455(T+1)-1 (执行口径, 严格 T+1);
  旧 close(T)→close(T+k) 口径仅保留研究对照
- 横截面 0.1%/99.9% 缩尾 (B2); [B18] is_virtual=1 退市虚拟样本豁免缩尾
- 停牌污染置 NaN; 实盘训练遮蔽最近 5 天; 各模型各自 dropna (per-model), 不统一剔除
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_HORIZONS = (1, 3, 5)
CLS_THRESHOLD = 0.005  # +0.5% 覆盖双边成本 (佣金万2.5x2 + 印花税0.05% + 滑点0.05% ≈ 0.13%, 留安全垫)


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
        df["label_cls"] = (df["label_1d"] > CLS_THRESHOLD).astype("float")
        df.loc[df["label_1d"].isna(), "label_cls"] = np.nan
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
        """T 到 T+N 区间内存在停牌 → label_Nd 置 NaN (脏标签: 复牌价差非真实持有收益)."""
        # risk_filter: exclude ST stocks before label masking
        # (suspended stocks retained — needed for rolling suspension detection)
        if "is_st" in df.columns:
            df = df[~df["is_st"].astype(bool)].copy()
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
        df["label_cls"] = df["label_cls"].where(df["label_1d"].notna(), np.nan)
        return df

    # ---------------- 实盘标签遮蔽 ----------------
    @staticmethod
    def mask_recent_days(df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
        """实盘训练剔除最近 N 天 (label_5d 需要 T+5 收盘价, 最近 5 天标签未生成)."""
        df["date"].max() - pd.Timedelta(days=days * 2)  # 自然日宽松上界
        recent_dates = sorted(df["date"].unique())[-days:]
        mask = df["date"].isin(recent_dates)
        for col in [f"label_{k}d" for k in LABEL_HORIZONS] + ["label_cls"]:
            df.loc[mask, col] = np.nan
        return df

    # ---------------- per-model dropna ----------------
    @staticmethod
    def per_model_dropna(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """各模型各自丢弃缺失标签 (不统一剔除最后 5 天).

        Returns:
            {'1d': df_1d, '3d': df_3d, '5d': df_5d, 'cls': df_cls}
        """
        return {
            "1d": df.dropna(subset=["label_1d"]),
            "3d": df.dropna(subset=["label_3d"]),
            "5d": df.dropna(subset=["label_5d"]),
            "cls": df.dropna(subset=["label_cls"]),
        }
