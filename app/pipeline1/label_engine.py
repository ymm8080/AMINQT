# -*- coding: utf-8 -*-
"""
标签引擎 (DESIGN §14.1 安全网 #1/#7/#13, PIPELINE1_V3.5 §〇.1)
=================================================================
- 一律后复权价 (hfq); 早盘 pipeline 标签独立 (open(T+1) 基准)
- 横截面 1%/99% 缩尾; 停牌污染置 NaN; 实盘训练遮蔽最近 5 天
- 各模型各自 dropna (per-model), 不统一剔除
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_HORIZONS = (1, 3, 5)
CLS_THRESHOLD = 0.005  # +0.5% 覆盖双边成本 (佣金万2.5x2 + 印花税0.05% + 滑点0.05% ≈ 0.13%, 留安全垫)


class LabelEngine:
    """标签定义. 铁律: 输入 df 必须已 sort_values(['symbol','date']) (安全网 #13)."""

    # ---------------- 主标签 ----------------
    @staticmethod
    def build_labels(df: pd.DataFrame, session: str = "PM") -> pd.DataFrame:
        """label_kd = close_hfq[T+k] / close_hfq[T] - 1, k=1/3/5 (groupby symbol!).

        session="PM": T 收盘 -> T+k 收盘 (晚盘买入 pipeline)
        session="AM": open(T+1) -> close(T+k) (早盘买入 pipeline, T+1 制度最早 T+2 卖)
        """
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        g = df.groupby("symbol")["close_hfq"]
        for k in LABEL_HORIZONS:
            df[f"label_{k}d"] = g.shift(-k) / df["close_hfq"] - 1
        if session == "AM":
            g_open = df.groupby("symbol")["open"]
            for k in LABEL_HORIZONS:
                df[f"label_{k}d"] = g.shift(-k) / g_open.shift(-1) - 1
        df["label_cls"] = (df["label_1d"] > CLS_THRESHOLD).astype("float")
        df.loc[df["label_1d"].isna(), "label_cls"] = np.nan
        return df

    # ---------------- 缩尾 ----------------
    @staticmethod
    def winsorize_cross_section(df: pd.DataFrame,
                                lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
        """横截面分位缩尾 (按 date 分组), 防止极值主导损失."""
        for col in [f"label_{k}d" for k in LABEL_HORIZONS]:
            df[col] = df.groupby("date")[col].transform(
                lambda x: x.clip(x.quantile(lower), x.quantile(upper)))
        return df

    # ---------------- 停牌污染 ----------------
    @staticmethod
    def mask_suspension(df: pd.DataFrame) -> pd.DataFrame:
        """T 到 T+N 区间内存在停牌 → label_Nd 置 NaN (脏标签: 复牌价差非真实持有收益)."""
        for n in LABEL_HORIZONS:
            suspended = (df.groupby("symbol")["is_suspended"]
                         .rolling(n + 1).sum().shift(-n)
                         .reset_index(level=0, drop=True) > 0)
            df[f"label_{n}d"] = df[f"label_{n}d"].where(~suspended, np.nan)
        df["label_cls"] = df["label_cls"].where(df["label_1d"].notna(), np.nan)
        return df

    # ---------------- 实盘标签遮蔽 ----------------
    @staticmethod
    def mask_recent_days(df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
        """实盘训练剔除最近 N 天 (label_5d 需要 T+5 收盘价, 最近 5 天标签未生成)."""
        cutoff = df["date"].max() - pd.Timedelta(days=days * 2)  # 自然日宽松上界
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
