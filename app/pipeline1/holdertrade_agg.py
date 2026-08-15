"""holdertrade_agg.py — stk_holdertrade 原始事件 → 日频面板聚合列 (dim31_holdertrade 上游).

语义与 panel_builder.agg_map (sh_net_change_sign=sum(sh_net_sign),
sh_change_amt_total=sum(sh_change_amt), sh_evt_start_date=min, sh_evt_end_date=max)
及 _backfill_holder_ratio (G/P/C signed ratio 拆分) 完全一致. 单一权威实现:
_daily_fetch 每日增量与一次性回填脚本共用, 防止两处漂移.

输入: DataSupplyChain.fetch_holdertrade 输出帧
      (symbol, date=announce_date, sh_change_vol, sh_change_amt, sh_change_ratio,
       sh_holder_type, sh_holder_name, sh_change_type, announce_date,
       evt_start_date, evt_end_date, sh_net_sign)
输出: (symbol, date) 日聚合帧, 10 面板列.
      sh_net_sign 与 sh_net_change_sign 同为当日 sum(sign) (多记录日 |sum|>1;
      历史面板该列曾是单记录值, 无特征消费, 展示口径统一从 sum).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLDER_PANEL_COLS = [
    "sh_net_change_sign",
    "sh_change_amt_total",
    "sh_change_vol",
    "sh_net_sign",
    "sh_evt_start_date",
    "sh_evt_end_date",
    "sh_net_ratio",
    "sh_g_ratio",
    "sh_p_ratio",
    "sh_c_ratio",
]


def agg_holdertrade_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """按 (symbol, ann_date) 聚合 → 10 面板列 (向量化, 无 per-stock 循环)."""
    r = raw.copy()
    for c in ("sh_net_sign", "sh_change_vol", "sh_change_amt"):
        if c not in r.columns:
            r[c] = 0
    if "sh_change_ratio" not in r.columns:
        r["sh_change_ratio"] = np.nan
    r["sh_net_sign"] = pd.to_numeric(r["sh_net_sign"], errors="coerce").fillna(0.0)
    r["change_ratio"] = pd.to_numeric(r["sh_change_ratio"], errors="coerce")
    r["signed_ratio"] = r["sh_net_sign"] * r["change_ratio"]
    r["holder_type"] = (
        r["sh_holder_type"].fillna("").astype(str).str.upper()
        if "sh_holder_type" in r.columns
        else ""
    )
    r["sr_g"] = np.where(r["holder_type"] == "G", r["signed_ratio"], 0.0)
    r["sr_p"] = np.where(r["holder_type"] == "P", r["signed_ratio"], 0.0)
    r["sr_c"] = np.where(r["holder_type"] == "C", r["signed_ratio"], 0.0)
    return (
        r.groupby(["symbol", "date"], as_index=False)
        .agg(
            sh_net_change_sign=("sh_net_sign", "sum"),
            sh_change_amt_total=("sh_change_amt", "sum"),
            sh_change_vol=("sh_change_vol", "sum"),
            sh_net_sign=("sh_net_sign", "sum"),
            sh_evt_start_date=("evt_start_date", "min"),
            sh_evt_end_date=("evt_end_date", "max"),
            sh_net_ratio=("signed_ratio", "sum"),
            sh_g_ratio=("sr_g", "sum"),
            sh_p_ratio=("sr_p", "sum"),
            sh_c_ratio=("sr_c", "sum"),
        )
        .sort_values("date")
    )
