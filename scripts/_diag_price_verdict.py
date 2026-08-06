# -*- coding: utf-8 -*-
"""_diag_price_verdict.py — 价格列判定矛盾定位 (2026-08-04).

现象: 全量跑 (69列, _classify_freq_full.py) close_hfq = TS·周 负 -0.043 (反转);
类比列跑 (_classify_freq_analog.py) close/open_hfq 等 = TS·月 正 +0.06x (动量).
两个脚本方法不同: 全量跑直接读面板+LabelEngine.build_labels; 类比跑走完整训练管线.

本脚本用与全量跑完全相同的口径 (面板列 + LabelEngine.build_labels + mask + 3y),
对 close / close_hfq / open / open_hfq 并排算 6格 → 看价格族判定是否自洽.
输出: data/_diag_price_verdict_<ts>.log (WORM).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.label_engine import LabelEngine
from scripts._diag_column_feed import LABELS, MASK_RECENT_DAYS, WEIGHTS
from scripts._classify_freq_full import (
    MIN_CROSS,
    MIN_OBS,
    WINDOWS,
    _f,
    _wtsic,
    group_spearman,
)


def classify(work, g_sym, g_date, lab_sym, lab_date, col):
    g_grp = work.groupby("symbol")
    wins = {}
    for w in WINDOWS.values():
        wins[f"{col}_p{w}"] = (work[col] / g_grp[col].shift(w) - 1.0).astype("float64")
    tsic, xic = {}, {}
    for f, wc in wins.items():
        wr_sym = wc.groupby(g_sym.values).rank()
        wr_date = wc.groupby(g_date.values).rank()
        tsic[f] = {l: group_spearman(wr_sym, lab_sym[l], g_sym, MIN_OBS) for l in LABELS}
        xic[f] = {l: group_spearman(wr_date, lab_date[l], g_date, MIN_CROSS) for l in LABELS}
    ts = {w: _wtsic(tsic[f"{col}_p{w}"]) for w in (1, 5, 20)}
    xs = {w: _wtsic(xic[f"{col}_p{w}"]) for w in (1, 5, 20)}
    cells = {
        "TS日": ts[1], "TS周": ts[5], "TS月": ts[20],
        "XS日": xs[1], "XS周": xs[5], "XS月": xs[20],
    }
    return cells


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    need = ["date", "symbol", "is_suspended", "close_hfq", "close", "open", "open_hfq"]
    cols = ["date", "symbol", "is_suspended"]
    for c in ["close_hfq", "close", "open", "open_hfq"]:
        if c in pd.read_parquet(PANEL_V3_PATH, columns=[c]).columns:
            cols.append(c)
    df = pd.read_parquet(PANEL_V3_PATH, columns=cols)
    df = LabelEngine.build_labels(df, session="PM")
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    cutoff = df["date"].max() - pd.DateOffset(years=3)
    work = df[df["date"] >= cutoff].reset_index(drop=True)
    g_sym, g_date = work["symbol"], work["date"]
    lab_sym = {l: work.groupby("symbol")[l].rank() for l in LABELS}
    lab_date = {l: work.groupby("date")[l].rank() for l in LABELS}

    lines = []
    lines.append("=" * 78)
    lines.append("  价格列判定矛盾定位 — 全量跑口径 (面板列 + LabelEngine + mask + 3y)")
    lines.append(f"  rows={len(work):,} stocks={work['symbol'].nunique():,} "
                 f"{work['date'].min():%Y-%m-%d} ~ {work['date'].max():%Y-%m-%d}")
    lines.append("=" * 78)
    lines.append(f"{'col':<12}{'TS日':>8}{'TS周':>8}{'TS月':>8}"
                 f"{'XS日':>8}{'XS周':>8}{'XS月':>8}  ← 判定")
    lines.append("-" * 78)

    # close 与 close_hfq 的 20日变化相关性 (证明二者近乎同一条序列)
    if "close" in work.columns and "close_hfq" in work.columns:
        gg = work.groupby("symbol")
        p_close = work["close"] / gg["close"].shift(20) - 1.0
        p_hfq = work["close_hfq"] / gg["close_hfq"].shift(20) - 1.0
        r = p_close.corr(p_hfq)
        lines.append(f"[相关性] close_p20 vs close_hfq_p20 相关系数 = {r:.4f}")
        lines.append("-" * 78)

    for c in ["close_hfq", "close", "open", "open_hfq"]:
        if c not in work.columns:
            lines.append(f"{c:<12}  (列缺失)")
            continue
        cells = classify(work, g_sym, g_date, lab_sym, lab_date, c)
        best = max(cells, key=lambda k: abs(cells[k]))
        lines.append(
            f"{c:<12}{_f(cells['TS日']):>8}{_f(cells['TS周']):>8}{_f(cells['TS月']):>8}"
            f"{_f(cells['XS日']):>8}{_f(cells['XS周']):>8}{_f(cells['XS月']):>8}"
            f"  ← {best} ({abs(cells[best]):.4f})"
        )

    text = "\n".join(lines)
    print(text, flush=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_diag_price_verdict_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}", flush=True)


if __name__ == "__main__":
    main()
