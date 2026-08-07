"""_diag_lhb_event_pool.py — LHB 事件池研究 (用户 2026-08-04 方法论).

EVENT 是事件性的: 不是把单只股票的 LHB 与非 LHB 时段混在一起训练,
而是把同类事件聚合 — 不同股票的 LHB 事件, 取事件前1个月(20交易日) 到
事件后1个月, 所有事件样本放在一起找 feature 的作用.

本脚本对全市场×3年所有 LHB 事件 (lhb_net_buy!=0):
1. 事件窗口收益轨迹: 全部事件聚合到相对日 [-20,+20], 平均相对收益曲线
2. 事件特征分层: 按 净买/净卖符号 + 金额三分位, 事件后 T+2/3/5/10/20 平均收益 & 胜率
3. 事件池内 feature 作用: 事件日 lhb_net_buy 金额(归一化) 跨事件样本对 T+2/3/5 的 rank IC

输出落盘 data/_diag_lhb_event_pool_<ts>.log (WORM).
"""

import gc
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH

logging.disable(logging.CRITICAL)
np.seterr(all="ignore")

W = 20  # 事件前/后交易日窗口 (约1个月)
HORIZONS = (2, 3, 5, 10, 20)


def _load() -> pd.DataFrame:
    read_cols = [
        "date",
        "symbol",
        "is_suspended",
        "close_hfq",
        "lhb_net_buy",
        "lhb_buy_amt",
        "lhb_sell_amt",
        "lhb_inst_buy",
        "lhb_inst_sell",
    ]
    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["ridx"] = df.groupby("symbol").cumcount()
    return df


def _f(v):
    return f"{v:+.4f}" if v == v else "   nan"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    out = []
    df = _load()
    latest = df["date"].max()
    cutoff = latest - pd.DateOffset(years=3)
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    df = df[~df["is_suspended"].astype(bool)].reset_index(drop=True)
    out.append(f"--- 全市场×3年 | rows={len(df):,} stocks={df['symbol'].nunique()} ---")

    ev_mask = df["lhb_net_buy"].fillna(0) != 0
    out.append(f"LHB 事件数 = {int(ev_mask.sum()):,}")

    # 构建事件窗口: 每事件取 ridx-W .. ridx+W
    events = df.loc[ev_mask, ["symbol", "date", "ridx"]].copy()
    events["evt_id"] = np.arange(len(events))
    off = pd.DataFrame({"off": np.arange(-W, W + 1)})
    ev_long = events.merge(off, how="cross")  # 事件数 × 41
    ev_long["trg"] = ev_long["ridx"] + ev_long["off"]
    win = ev_long.merge(
        df[["symbol", "ridx", "close_hfq"]],
        left_on=["symbol", "trg"],
        right_on=["symbol", "ridx"],
        how="left",
        suffixes=("", "_x"),
    )
    # 事件日(off=0)收盘价作为基准
    base = win.loc[win["off"] == 0, ["evt_id", "close_hfq"]].rename(
        columns={"close_hfq": "base"}
    )
    win = win.merge(base, on="evt_id", how="left")
    win["rel"] = win["close_hfq"] / win["base"] - 1.0

    # 1. 事件窗口平均收益轨迹 (分段累计)
    out.append("\n  1) 事件窗口平均相对收益 (相对事件日收盘, 全事件平均):")
    segs = [
        (-20, -16),
        (-15, -11),
        (-10, -6),
        (-5, -1),
        (0, 0),
        (1, 5),
        (6, 10),
        (11, 15),
        (16, 20),
    ]
    out.append(f"{'交易日段':<14}{'平均rel':>10}{'n':>8}")
    for a, b in segs:
        m = win["off"].between(a, b) & win["rel"].notna()
        out.append(
            f"[{a:+d},{b:+d}]{'':<8}{_f(win.loc[m, 'rel'].mean()):>10}"
            f"{int(m.sum()):>8,}"
        )

    # 事件日 rel(=0) 及事件后单日点
    out.append("\n  2) 事件后分点位平均收益 & 胜率 (相对事件日收盘):")
    out.append(f"{'分组':<16}{'T+2':>10}{'T+3':>10}{'T+5':>10}{'T+10':>10}{'T+20':>10}")
    # 每个事件在 horizon 点的 rel
    piv = win.pivot_table(index="evt_id", columns="off", values="rel")

    def _row(tag, idx, piv=piv):
        parts = [f"{tag:<16}"]
        for h in HORIZONS:
            s = piv.loc[idx, h].dropna()
            if len(s) == 0:
                parts.append(f"{'   nan':>10}")
            else:
                parts.append(f"{_f(s.mean()):>10}")
        out.append("  ".join(parts))

    # 基准 + 净买/净卖 (sign 顺序 = evt_id 0..N-1)
    _row("全部事件", piv.index)
    sign = df.loc[ev_mask, "lhb_net_buy"].to_numpy()
    for tag, m in (("净买>0", sign > 0), ("净卖<0", sign < 0)):
        _row(tag, piv.index[m])

    # 金额三分位 (lhb_net_buy abs)
    amt = np.abs(sign)
    q = np.quantile(amt, [1 / 3, 2 / 3])
    for tag, m in (
        ("净买额小1/3", amt <= q[0]),
        ("净买额中1/3", (amt > q[0]) & (amt <= q[1])),
        ("净买额大1/3", amt > q[1]),
    ):
        _row(tag, piv.index[m])

    # 3. 事件池内 feature 对事件后收益的 rank IC (跨事件样本)
    out.append("\n  3) 事件池内 feature 对事件后收益的 rank IC (跨事件):")
    feat = pd.DataFrame({"lhb_net_buy": df.loc[ev_mask, "lhb_net_buy"].values})
    feat["abs_net"] = feat["lhb_net_buy"].abs()
    feat["inst_ratio"] = df.loc[ev_mask, "lhb_inst_buy"].values / (
        df.loc[ev_mask, "lhb_buy_amt"].values + 1e-9
    )
    out.append(f"{'feature':<14}{'T+2':>9}{'T+3':>9}{'T+5':>9}{'T+10':>9}{'T+20':>9}")
    for c in feat.columns:
        row = [f"{c:<14}"]
        for h in HORIZONS:
            y = piv[h] if h in piv else pd.Series(dtype=float)
            m = feat[c].notna().values & y.notna().values
            if m.sum() < 30:
                row.append(f"{'   nan':>9}")
                continue
            r = feat.loc[m, c].astype(float).rank()
            ry = y[m].astype(float).rank()
            v = r.corr(ry)
            row.append(f"{_f(v):>9}")
        out.append("  ".join(row))

    text = "\n".join(out)
    print(text)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_diag_lhb_event_pool_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}")

    del df, win, ev_long, piv
    gc.collect()


if __name__ == "__main__":
    main()
