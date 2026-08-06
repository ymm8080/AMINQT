# -*- coding: utf-8 -*-
"""_diag_event_pool.py — 通用事件池时间对齐研究 (BT 大宗 / HOLDER 增减持).

方法论 (用户 2026-08-04, MEMORIZE IT): EVENT 是事件性的 — 把同类事件聚合,
取每个事件 前1个月(20交易日) → 后1个月, 全部事件样本放在一起找 feature 的作用.
不做个股 TS 混训 / 不做日截面常规模.

对每组事件:
1. 事件窗口平均相对收益轨迹 [-20,+20] (相对事件日收盘, 全事件平均)
2. 事件特征分层: 按方向符号 + 规模分位, 事件后 T+2/3/5/10/20 平均收益
3. 事件池内跨事件 rank IC: 事件特征对 T+2/3/5/10/20 收益

事件定义 (与上游一致):
- BT: 4 个 bt_* 原始列任一非空 (bt_v3_train_eval.py 同定义)
- HOLDER: sh_net_ratio 非空 (KIMI 比例列已按 (symbol, 公告日) 聚合)

用法: python scripts/_diag_event_pool.py
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

W = 20
HORIZONS = (2, 3, 5, 10, 20)

BT_COLS = ["bt_count", "bt_disc_raw", "bt_inst_absorb", "bt_amt_ratio_float_mv"]
# holder 事件 = 稀疏 ratio 列 (0.43%, 未被 dim29 ffill); sh_change_amt_total 被 ffill 污染 51.9% 不能用
HOLDER_COLS = ["sh_net_ratio", "sh_g_ratio", "sh_c_ratio"]


def _load() -> pd.DataFrame:
    read_cols = list(
        dict.fromkeys(
            ["date", "symbol", "is_suspended", "close_hfq"] + BT_COLS + HOLDER_COLS
        )
    )
    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["ridx"] = df.groupby("symbol").cumcount()
    return df


def _f(v):
    return f"{v:+.4f}" if v == v else "   nan"


def build_window(df, ev_mask, W=W):
    events = df.loc[ev_mask, ["symbol", "date", "ridx"]].copy()
    events["evt_id"] = np.arange(len(events))
    off = pd.DataFrame({"off": np.arange(-W, W + 1)})
    ev_long = events.merge(off, how="cross")
    ev_long["trg"] = ev_long["ridx"] + ev_long["off"]
    win = ev_long.merge(
        df[["symbol", "ridx", "close_hfq"]],
        left_on=["symbol", "trg"],
        right_on=["symbol", "ridx"],
        how="left",
        suffixes=("", "_x"),
    )
    base = win.loc[win["off"] == 0, ["evt_id", "close_hfq"]].rename(
        columns={"close_hfq": "base"}
    )
    win = win.merge(base, on="evt_id", how="left")
    win["rel"] = win["close_hfq"] / win["base"] - 1.0
    piv = win.pivot_table(index="evt_id", columns="off", values="rel")
    return win, piv


def report_event(out, title, df, ev_mask, feat_cols, f0_col):
    out.append("=" * 78)
    out.append(f"  {title}")
    out.append("=" * 78)
    n_ev = int(ev_mask.sum())
    out.append(f"事件数 = {n_ev:,} (事件行覆盖率 {n_ev / len(df):.2%})")
    if n_ev < 200:
        out.append("样本过少, 跳过")
        out.append("")
        return

    win, piv = build_window(df, ev_mask)

    # 1. 事件窗口轨迹
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
    out.append(f"{'交易日段':<14}{'平均rel':>10}{'n':>9}")
    for a, b in segs:
        m = win["off"].between(a, b) & win["rel"].notna()
        out.append(
            f"[{a:+d},{b:+d}]{'':<8}{_f(win.loc[m, 'rel'].mean()):>10}{int(m.sum()):>9,}"
        )

    # 2. 事件特征分层
    out.append("\n  2) 事件后分点位平均收益 (相对事件日收盘):")
    out.append(f"{'分组':<16}{'T+2':>10}{'T+3':>10}{'T+5':>10}{'T+10':>10}{'T+20':>10}")
    ev_feat = df.loc[ev_mask, feat_cols].reset_index(drop=True)

    def _row(tag, idx):
        parts = [f"{tag:<16}"]
        for h in HORIZONS:
            s = piv.loc[idx, h].dropna()
            parts.append(f"{_f(s.mean()):>10}" if len(s) else f"{'   nan':>10}")
        out.append("  ".join(parts))

    _row("全部事件", piv.index)
    f0 = ev_feat[f0_col].astype(float).to_numpy()
    _row(f"{f0_col}>0", piv.index[f0 > 0])
    _row(f"{f0_col}==0", piv.index[f0 == 0])
    _row(f"{f0_col}<0", piv.index[f0 < 0])
    # 规模分位 (f0 绝对值, 排除 0)
    am = np.abs(f0)
    nz = f0 != 0
    q = (
        np.nanquantile(am[nz], [1 / 3, 2 / 3])
        if nz.sum()
        else np.array([np.nan, np.nan])
    )
    _row("|f0| 小1/3", piv.index[am <= q[0]])
    _row("|f0| 中1/3", piv.index[(am > q[0]) & (am <= q[1])])
    _row("|f0| 大1/3", piv.index[am > q[1]])

    # 3. 事件池内 rank IC
    out.append("\n  3) 事件池内特征对事件后收益的 rank IC (跨事件样本):")
    out.append(f"{'feature':<20}{'T+2':>9}{'T+3':>9}{'T+5':>9}{'T+10':>9}{'T+20':>9}")
    for c in feat_cols:
        x = ev_feat[c].astype(float)
        row = [f"{c:<20}"]
        for h in HORIZONS:
            y = piv[h] if h in piv else pd.Series(dtype=float)
            m = x.notna().to_numpy() & y.notna().to_numpy()
            if m.sum() < 30:
                row.append(f"{'   nan':>9}")
                continue
            r = x[m].rank()
            ry = y[m].rank()
            row.append(f"{_f(r.corr(ry)):>9}")
        out.append("  ".join(row))
    out.append("")

    del win, piv, ev_feat
    gc.collect()


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

    bt_mask = df[BT_COLS].notna().any(axis=1)
    report_event(out, "BT 大宗交易 事件池", df, bt_mask, BT_COLS, "bt_disc_raw")

    ho_mask = df[HOLDER_COLS].notna().any(axis=1)
    report_event(
        out, "HOLDER 大股东增减持 事件池", df, ho_mask, HOLDER_COLS, "sh_net_ratio"
    )

    text = "\n".join(out)
    print(text)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_diag_event_pool_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}")

    del df
    gc.collect()


if __name__ == "__main__":
    main()
