"""legacy 预测输出级时间平滑 (2026-08-06) — 对齐 parallel _shortlist_t5_t10.ema_smooth.

用户问题: 同一只股票相邻交易日预测(预期涨幅/达到概率)剧烈变化.
Layer 2 (输出级 EMA): 每股 forecast 列 = 近 K 个可用交易日 raw 预测的衰减加权均值
(w_k = α·(1-α)^k, 归一化, gap-robust), 不动 score/rank/weight. 历史底稿 WORM 落盘
legacy_preds_raw_<date>__<module>.csv → STOCK_LIST_DIR (模块标签见 module-tag 约定).
首日/无同模块历史 = no-op. 必须在 lister.emit 之前应用, 使 E7 准入 + d3 排名用稳定值.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from config.settings import (
    LEGACY_SMOOTH_ALPHA as ALPHA,
)
from config.settings import (
    LEGACY_SMOOTH_ENABLED as ENABLED,
)
from config.settings import (
    LEGACY_SMOOTH_K as K,
)
from config.settings import (
    STOCK_LIST_DIR,
)

RAW_PREFIX = "legacy_preds_raw"

# 平滑对象: 模型直接输出的预测列 (复合/排名/风控列不动)
FORECAST_PREFIXES = ("pred_ret_", "prob_up", "pred_q50")


def _module_suffix(module: str) -> str:
    return f"__{module}" if module != "na" else ""


def _forecast_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FORECAST_PREFIXES)]


def persist_raw_preds(df: pd.DataFrame, trade_date: str, module: str) -> str:
    """WORM: 每股当日 raw 预测 (平滑前) → legacy_preds_raw_<date>__<module>.csv."""
    if not ENABLED:
        return ""
    cols = _forecast_cols(df)
    if "symbol" not in cols:
        cols = ["symbol"] + cols
    d = df[cols].copy()
    d = d.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)
    os.makedirs(str(STOCK_LIST_DIR), exist_ok=True)
    fp = STOCK_LIST_DIR / f"{RAW_PREFIX}_{trade_date}{_module_suffix(module)}.csv"
    d.to_csv(fp, index=False)
    return str(fp)


def load_raw_history(trade_date: str, module: str) -> pd.DataFrame:
    """读 <选股日> 之前同模块 raw 预测 → 长表 (symbol, hist_date, forecast 列)."""
    suffix = _module_suffix(module)
    frames = []
    for fp in STOCK_LIST_DIR.glob(f"{RAW_PREFIX}_*.csv"):
        m = re.match(rf"{RAW_PREFIX}_(\d{{8}})(.*)\.csv$", fp.name)
        if not m:
            continue
        if suffix and m.group(2) != suffix:
            continue
        if not suffix and m.group(2):
            continue  # module=na 时不混入带模块标记的历史
        if m.group(1) >= trade_date:
            continue  # 不含今日 (今日 raw 由调用方直接给)
        d = pd.read_csv(fp, dtype={"symbol": str})
        d["hist_date"] = m.group(1)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["symbol", "hist_date"], keep="last"
    )


def smooth_preds(df: pd.DataFrame, trade_date: str, module: str) -> pd.DataFrame:
    """EMA 平滑 forecast 列 (每股独立); 无历史/未启用 → 原样返回."""
    if not ENABLED:
        return df
    hist = load_raw_history(trade_date, module)
    if hist.empty:
        return df
    weights = np.array([ALPHA * (1 - ALPHA) ** k for k in range(K)])
    weights /= weights.sum()
    out = df.copy()
    cols = _forecast_cols(df)
    for sym in out["symbol"].unique():
        h = hist[hist["symbol"] == sym].sort_values("hist_date", ascending=False)
        if h.empty:
            continue
        src = h.head(K - 1)  # 最多 K-1 个旧日 (今日占 k=0)
        mask = out["symbol"] == sym
        for col in cols:
            today_v = out.loc[mask, col]
            if today_v.empty or not np.isfinite(float(today_v.iloc[0])):
                continue
            pairs = [
                (w, v)
                for w, v in zip(
                    weights,
                    [float(today_v.iloc[0])] + [float(x) for x in src[col]],
                    strict=False,
                )
                if np.isfinite(v)
            ]
            if not pairs:
                continue
            ww = np.array([p[0] for p in pairs])
            ww /= ww.sum()
            out.loc[mask, col] = float(np.dot([p[1] for p in pairs], ww))
    return out
