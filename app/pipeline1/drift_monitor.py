"""legacy 幅度漂移监控核心逻辑 (2026-08-17).

08-17 诊断: pred_ret_10d 系统高估 (main 均值 +4.03% vs 实现 +1.10%, dual +6.59% vs
+1.32%), 偏差随时间扩大 → 生产 "pred>0" 闸 100% 空转. 修漂移 = 重训, 监控 = 每日
全池预测均值 vs T+10 净实现均值的偏差, 滚动窗超阈值 → 提醒提前重训.

本模块纯逻辑 (可单测); CLI 在 scripts/_monitor_legacy_drift.py.
数据口径与 _diag_legacy_prob_head_replay 完全一致: buy=决策日后第 buy_lag 个交易日
收盘, sell=第 sell_lag 个交易日收盘, ps/pb-1-COST; 停牌每股 ffill.
"""

from __future__ import annotations

from pathlib import Path

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
        buy_dt, sell_dt = (
            pd.Timestamp(cal[i + buy_lag]),
            pd.Timestamp(cal[i + sell_lag]),
        )
        pb, ps = pivot[buy_dt], pivot[sell_dt]
        sub = (
            (ps / pb - 1.0 - cost)
            .where(pb > 0)
            .rename("realized_net")
            .reset_index()
            .dropna(subset=["realized_net"])
        )
        if len(sub):
            sub["date"] = pd.Timestamp(d)
            out.append(sub[["date", "symbol", "realized_net"]])
    if not out:
        return pd.DataFrame(columns=["date", "symbol", "realized_net"])
    return pd.concat(out, ignore_index=True)


def compute_realized_mfe(
    panel: pd.DataFrame,
    decision_dates: pd.Series,
    horizon: int = 10,
    cost: float = 0.0030,
) -> pd.DataFrame:
    """panel(symbol,date,high_hfq,close_hfq) → (date,symbol,realized_mfe), 仅已成熟行.

    口径同 backtest.add_mfe_labels: T+1 收盘买, max(high_hfq[T+2..T+1+horizon])
    / close_hfq[T+1] - 1 - cost. 决策日锚定/停牌 ffill 规则与 compute_realized 一致.
    """
    cal = np.sort(pd.to_datetime(panel["date"]).dt.normalize().unique().to_numpy())
    dt = panel.assign(dt=pd.to_datetime(panel["date"]).dt.normalize())
    close_p = (
        dt.pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
        .sort_index()
        .reindex(columns=pd.to_datetime(cal))
        .ffill(axis=1)
    )
    high_p = (
        dt.pivot_table(index="symbol", columns="dt", values="high_hfq", aggfunc="last")
        .sort_index()
        .reindex(columns=pd.to_datetime(cal))
    )
    out: list[pd.DataFrame] = []
    for d in pd.DatetimeIndex(pd.to_datetime(decision_dates)).normalize().unique():
        d = np.datetime64(d)
        i = int(np.searchsorted(cal, d, side="right")) - 1
        if i < 0 or i + horizon + 1 >= len(cal):
            continue
        pb = close_p[pd.Timestamp(cal[i + 1])]
        peak = high_p[pd.to_datetime(cal[i + 2 : i + horizon + 2])].max(axis=1)
        sub = (
            (peak / pb - 1.0 - cost)
            .where(pb > 0)
            .rename("realized_mfe")
            .reset_index()
            .dropna(subset=["realized_mfe"])
        )
        if len(sub):
            sub["date"] = pd.Timestamp(d)
            out.append(sub[["date", "symbol", "realized_mfe"]])
    if not out:
        return pd.DataFrame(columns=["date", "symbol", "realized_mfe"])
    return pd.concat(out, ignore_index=True)


def accumulate_parallel_picks(run_root: Path) -> pd.DataFrame:
    """扫描 BACKTEST_RESULT_DIR 下 last_*_days_picks_dual.csv → (date,symbol,board,pred_ret_10d).

    每个 run 目录含近 15 交易日的滚窗快照 (每日重生成, WORM), 同一决策日会出现在多个
    run 中; 按 run ts 升序拼接后 (date,symbol) keep-first = 决策日当天生成的那份
    (as-predicted 语义, 与回测口径一致). pred_ret_10d 取 mfe_10d (校准预期幅度).
    """
    parts = []
    for f in sorted(Path(run_root).glob("*/last_*_days_picks_dual.csv")):
        parts.append(pd.read_csv(f, dtype={"symbol": str}))
    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "board", "pred_ret_10d"])
    df = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["date", "symbol"], keep="first"
    )
    return pd.DataFrame(
        {
            "date": df["date"],
            "symbol": df["symbol"],
            "board": "dual",
            "pred_ret_10d": df["mfe_10d"],
        }
    )


def daily_bias(preds: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    """逐日逐板块偏差: 全池 pred 均值 − 净实现均值 (等日权重滚动用逐日值).

    preds: (date, symbol, board, pred_ret_10d); realized: (date, symbol, realized_net).
    """
    m = preds.merge(realized, on=["date", "symbol"], how="inner")
    m = m.dropna(subset=["pred_ret_10d", "realized_net"])
    if m.empty:
        return pd.DataFrame(
            columns=["date", "board", "n", "pred_mean", "real_mean", "bias"]
        )
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


def bin_calibration(
    prob: pd.Series, event: pd.Series, n_bins: int = 5
) -> tuple[pd.DataFrame, float]:
    """(prob, event) → 分位桶校准表 + ECE (expected calibration error).

    p_reg 集中 [0.25, 0.55] → 等宽桶失真, 用 qcut 分位桶保证每桶有样本.
    ECE = Σ bin权重 × |realized_rate − mean_prob|. 样本 < 2×n_bins 或事件单值
    → 返回空表 + nan (积累期/信号退化, 上层当 None 处理).
    """
    df = pd.DataFrame({"prob": prob, "event": event}).dropna()
    if len(df) < 2 * n_bins or df["event"].nunique() < 2:
        return pd.DataFrame(), np.nan
    df["bin"] = pd.qcut(df["prob"], q=n_bins, duplicates="drop")
    g = (
        df.groupby("bin", observed=True)
        .agg(
            n=("event", "size"), mean_prob=("prob", "mean"), realized=("event", "mean")
        )
        .query("n > 0")
    )
    if g.empty:
        return pd.DataFrame(), np.nan
    g["gap"] = (g["realized"] - g["mean_prob"]).abs()
    ece = float((g["gap"] * g["n"]).sum() / g["n"].sum())
    return g, ece


def rolling_calibration(
    preds: pd.DataFrame,
    realized: pd.DataFrame,
    cost: float = 0.0020,
    thr: float = 0.005,
    n_bins: int = 5,
    window_days: int = 42,
    min_matured_days: int = 20,
) -> pd.DataFrame:
    """逐板块滚动窗校准: 尾 window_days 成熟日全池 (prob_up_10d, gross>thr 事件) 池化 → ECE.

    prob 建模事件是 gross (label_10d = close[T+11]/open[T+1]−1 > 0.5%), 不是 net;
    用同一 realized_net (=gross_cc−cost) 还原 gross 事件: gross_cc = realized_net + cost
    > thr. close 基 vs open 基的隔夜缺口是常数偏移, 由回放参照吸收, 监控看的是变化.
    preds: (date, symbol, board, prob_up_10d); realized: (date, symbol, realized_net).
    """
    m = preds.merge(realized, on=["date", "symbol"], how="inner")
    m = m.dropna(subset=["prob_up_10d", "realized_net"])
    cols = ["board", "n_days", "n_rows", "ece", "bins"]
    if m.empty:
        return pd.DataFrame(columns=cols)
    m["event"] = (m["realized_net"] + cost > thr).astype("int8")
    rows: list[dict] = []
    for board, g in m.groupby("board"):
        dates = np.sort(pd.to_datetime(g["date"].unique()))
        base = {
            "board": board,
            "n_days": int(len(dates)),
            "n_rows": 0,
            "ece": None,
            "bins": [],
        }
        if len(dates) < min_matured_days:
            rows.append(base)
            continue
        tail = g[g["date"].isin(pd.to_datetime(dates[-window_days:]))]
        gtab, ece = bin_calibration(tail["prob_up_10d"], tail["event"], n_bins)
        if gtab.empty:
            rows.append(base)
            continue
        bins = (
            gtab.reset_index()[["bin", "n", "mean_prob", "realized", "gap"]]
            .assign(bin=lambda t: t["bin"].astype(str))
            .to_dict("records")
        )
        rows.append({**base, "n_rows": int(len(tail)), "ece": ece, "bins": bins})
    return pd.DataFrame(rows)


def rolling_calibration_mfe(
    preds: pd.DataFrame,
    realized: pd.DataFrame,
    thr: float = 0.06,
    n_bins: int = 5,
    window_days: int = 42,
    min_matured_days: int = 20,
) -> pd.DataFrame:
    """parallel 版滚动校准: 尾 window_days 成熟日 (pred_prob_10d, MFE>thr 事件) 池化 → ECE.

    pred_prob_10d 建模事件 = net MFE > ABS_TARGET["10d"] (=0.06), realized 为
    compute_realized_mfe 的 net 值 → event = realized_mfe > thr, 无 cost 还原步.
    preds: (date, symbol, board, pred_prob_10d); realized: (date, symbol, realized_mfe).
    """
    m = preds.merge(realized, on=["date", "symbol"], how="inner")
    m = m.dropna(subset=["pred_prob_10d", "realized_mfe"])
    cols = ["board", "n_days", "n_rows", "ece", "bins"]
    if m.empty:
        return pd.DataFrame(columns=cols)
    m["event"] = (m["realized_mfe"] > thr).astype("int8")
    rows: list[dict] = []
    for board, g in m.groupby("board"):
        dates = np.sort(pd.to_datetime(g["date"].unique()))
        base = {
            "board": board,
            "n_days": int(len(dates)),
            "n_rows": 0,
            "ece": None,
            "bins": [],
        }
        if len(dates) < min_matured_days:
            rows.append(base)
            continue
        tail = g[g["date"].isin(pd.to_datetime(dates[-window_days:]))]
        gtab, ece = bin_calibration(tail["pred_prob_10d"], tail["event"], n_bins)
        if gtab.empty:
            rows.append(base)
            continue
        bins = (
            gtab.reset_index()[["bin", "n", "mean_prob", "realized", "gap"]]
            .assign(bin=lambda t: t["bin"].astype(str))
            .to_dict("records")
        )
        rows.append({**base, "n_rows": int(len(tail)), "ece": ece, "bins": bins})
    return pd.DataFrame(rows)


def check_calibration(rolling: pd.DataFrame, thresholds: dict) -> list[dict]:
    """滚动窗 ECE > 阈值 → 告警条目 (仅对已知板块)."""
    alerts = []
    for _, r in rolling.iterrows():
        th = thresholds.get(r["board"])
        if th is None or r["ece"] is None:
            continue
        if r["ece"] > th:
            alerts.append(
                {
                    "board": r["board"],
                    "ece": r["ece"],
                    "threshold": th,
                    "n_days": r["n_days"],
                }
            )
    return alerts
