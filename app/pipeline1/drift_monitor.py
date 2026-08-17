"""legacy 幅度漂移监控核心逻辑 (2026-08-17).

08-17 诊断: pred_ret_10d 系统高估 (main 均值 +4.03% vs 实现 +1.10%, dual +6.59% vs
+1.32%), 偏差随时间扩大 → 生产 "pred>0" 闸 100% 空转. 修漂移 = 重训, 监控 = 每日
全池预测均值 vs T+10 净实现均值的偏差, 滚动窗超阈值 → 提醒提前重训.

本模块纯逻辑 (可单测); CLI 在 scripts/_monitor_legacy_drift.py.
数据口径与 _diag_legacy_prob_head_replay 完全一致: buy=决策日后第 buy_lag 个交易日
收盘, sell=第 sell_lag 个交易日收盘, ps/pb-1-COST; 停牌每股 ffill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BOARD_MAP = {"main": "main", "GEM": "dual", "STAR": "dual"}


def board_of(legacy_board: str) -> str:
    """legacy 板块命名 (main/GEM/STAR) → 监控板块 (main/dual)."""
    return BOARD_MAP.get(str(legacy_board), str(legacy_board))


def compute_realized(
    panel: pd.DataFrame,
    decision_dates: pd.Series,
    buy_lag: int = 1,
    sell_lag: int = 11,
    cost: float = 0.0020,
) -> pd.DataFrame:
    """panel(symbol,date,close_hfq) → (date, symbol, realized_net), 仅已实现行.

    决策日 D (可为非交易日, 如周六生产跑批) → i = D 前最近交易日索引;
    buy=cal[i+buy_lag], sell=cal[i+sell_lag]; 缺日期/非正价 → 行丢弃.
    """
    cal = np.sort(pd.to_datetime(panel["date"]).dt.normalize().unique().to_numpy())
    pivot = (
        panel.assign(dt=pd.to_datetime(panel["date"]).dt.normalize())
        .pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
        .sort_index()
    )
    pivot = pivot.reindex(columns=pd.to_datetime(cal)).ffill(axis=1)
    out: list[pd.DataFrame] = []
    for d in pd.DatetimeIndex(pd.to_datetime(decision_dates)).normalize().unique():
        d = np.datetime64(d)
        i = int(np.searchsorted(cal, d, side="right")) - 1
        if i < 0 or i + sell_lag >= len(cal):
            continue
        buy_dt, sell_dt = pd.Timestamp(cal[i + buy_lag]), pd.Timestamp(cal[i + sell_lag])
        pb, ps = pivot[buy_dt], pivot[sell_dt]
        sub = (
            (ps / pb - 1.0 - cost).where(pb > 0).rename("realized_net").reset_index()
            .dropna(subset=["realized_net"])
        )
        if len(sub):
            sub["date"] = pd.Timestamp(d)
            out.append(sub[["date", "symbol", "realized_net"]])
    if not out:
        return pd.DataFrame(columns=["date", "symbol", "realized_net"])
    return pd.concat(out, ignore_index=True)


def daily_bias(preds: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    """逐日逐板块偏差: 全池 pred 均值 − 净实现均值 (等日权重滚动用逐日值).

    preds: (date, symbol, board, pred_ret_10d); realized: (date, symbol, realized_net).
    """
    m = preds.merge(realized, on=["date", "symbol"], how="inner")
    m = m.dropna(subset=["pred_ret_10d", "realized_net"])
    if m.empty:
        return pd.DataFrame(columns=["date", "board", "n", "pred_mean", "real_mean", "bias"])
    g = m.groupby(["date", "board"])
    out = g.agg(
        pred_mean=("pred_ret_10d", "mean"),
        real_mean=("realized_net", "mean"),
        n=("realized_net", "size"),
    ).reset_index()
    out["bias"] = out["pred_mean"] - out["real_mean"]
    return out.sort_values(["board", "date"]).reset_index(drop=True)


def rolling_bias(
    daily: pd.DataFrame, window_days: int = 42, min_matured_days: int = 20
) -> pd.DataFrame:
    """逐板块滚动偏差: 尾 window_days 日均值 (等日权重); 成熟日 < min 则 None."""
    rows = []
    for board, g in daily.groupby("board"):
        g = g.sort_values("date")
        b = g["bias"].rolling(window_days, min_periods=min_matured_days).mean()
        last = b.iloc[-1] if len(b) else np.nan
        rows.append(
            {
                "board": board,
                "n_days": int(len(g)),
                "latest_date": g["date"].iloc[-1] if len(g) else None,
                "latest_bias": float(g["bias"].iloc[-1]) if len(g) else None,
                "bias": float(last) if np.isfinite(last) else None,
            }
        )
    return pd.DataFrame(rows)


def check_drift(rolling: pd.DataFrame, thresholds: dict) -> list[dict]:
    """滚动偏差 > 阈值 → 告警条目 (仅对已知板块)."""
    alerts = []
    for _, r in rolling.iterrows():
        th = thresholds.get(r["board"])
        if th is None or r["bias"] is None:
            continue
        if r["bias"] > th:
            alerts.append(
                {
                    "board": r["board"],
                    "bias": r["bias"],
                    "threshold": th,
                    "n_days": r["n_days"],
                }
            )
    return alerts
