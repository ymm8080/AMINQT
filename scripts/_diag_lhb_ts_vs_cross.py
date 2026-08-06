# -*- coding: utf-8 -*-
"""_diag_lhb_ts_vs_cross.py — LHB 龙虎榜: 个股 time-series IC vs 截面 IC vs 事件研究.

方法论 (用户 2026-08-04): 个股类特征用 per-stock TSIC, 截面类用横截面,
LHB 归属需数据测试判定. 本脚本对 lhb_* 列三口径对照:
1. 横截面 IC  (daily_rank_ic_multi) — 事件日 vs 全市场截面排名
2. per-stock TSIC (per_stock_ts_ic) — 该股自身 lhb 序列 vs 自身涨幅序列
3. 事件研究: lhb_net_buy != 0 的事件日, 平均 T+2/3/5 涨幅 + 胜率 vs 全样本基准,
   按净买入符号分组; 并减当日全市场横截面中位(超额).

HS300×1年 + 全市场×3年 两窗口. 输出落盘 data/_diag_lhb_ts_vs_cross_<ts>.log (WORM).

用法: python scripts/_diag_lhb_ts_vs_cross.py
"""
import gc
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.label_engine import LabelEngine
from scripts._diag_column_feed import (
    LABELS,
    MASK_RECENT_DAYS,
    daily_rank_ic_multi,
    weighted_ic,
)
from scripts._diag_chip_weekly import per_stock_ts_ic

logging.disable(logging.CRITICAL)

LHB_COLS = [
    "lhb_net_buy",
    "lhb_buy_amt",
    "lhb_sell_amt",
    "lhb_top_buy",
    "lhb_top_sell",
    "lhb_retail_buy",
    "lhb_retail_sell",
    "lhb_inst_buy",
    "lhb_inst_sell",
]
EVENT_COL = "lhb_net_buy"


def _load(hs300_only: bool) -> pd.DataFrame:
    read_cols = list(
        dict.fromkeys(["date", "symbol", "is_suspended", "close_hfq"] + LHB_COLS)
    )
    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)
    if hs300_only:
        import akshare as ak

        cons = ak.index_stock_cons(symbol="000300")
        col = "品种代码" if "品种代码" in cons.columns else "成分券代码"
        syms = [str(s).strip().split(".")[0] for s in cons[col]]
        df = df[df["symbol"].isin(syms)].reset_index(drop=True)
    df = LabelEngine.build_labels(df, session="PM")
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    return df


def _f(v):
    return f"{v:+.4f}" if v == v else "   nan"


def _event_study(tr, out):
    out.append(
        f"  事件研究 (事件={EVENT_COL}!=0): 平均 T+2/3/5 涨幅 & 胜率 vs 全样本基准"
    )
    base = {}
    for lab in LABELS:
        s = tr[lab].dropna()
        base[lab] = (s.mean(), (s > 0).mean())
    out.append(
        f"    基准(全样本): "
        f"T2 {_f(base['label_pm_2d_net'][0])}(胜率 {base['label_pm_2d_net'][1]:.1%})  "
        f"T3 {_f(base['label_pm_3d_net'][0])}(胜率 {base['label_pm_3d_net'][1]:.1%})  "
        f"T5 {_f(base['label_pm_5d_net'][0])}(胜率 {base['label_pm_5d_net'][1]:.1%})"
    )
    ev = tr[tr[EVENT_COL].fillna(0) != 0]
    # 当日市场横截面中位(超额基准)
    med = tr.groupby("date")[list(LABELS)].transform("median")
    for tag, sub in (("  全部事件", ev), ("  净买>0", ev[ev[EVENT_COL] > 0]),
                     ("  净卖<0", ev[ev[EVENT_COL] < 0])):
        if len(sub) < 10:
            out.append(f"    {tag}: n={len(sub)} 样本过少")
            continue
        parts = [f"    {tag}: n={len(sub)}"]
        for lab in LABELS:
            s = sub[lab].dropna()
            if len(s) < 10:
                continue
            ex = (sub[lab] - med.loc[sub.index, lab]).dropna()
            hit = (s > 0).mean()
            parts.append(
                f"{lab.replace('label_pm_', '').replace('_net', '')} "
                f"avg {_f(s.mean())} 超额 {_f(ex.mean())} 胜率 {hit:.1%}"
            )
        out.append("  ".join(parts))
    out.append("")


def _report(tr, label, out):
    out.append(f"--- {label} | rows={len(tr):,} stocks={tr['symbol'].nunique()} "
               f"事件日={int((tr[EVENT_COL].fillna(0) != 0).sum()):,} ---")
    # 1. 横截面 IC
    xic = daily_rank_ic_multi(tr, LHB_COLS, LABELS)
    out.append("  [截面IC]  col            wIC      IC2d     IC3d     IC5d")
    for c in LHB_COLS:
        per = {lab: xic[lab][c] for lab in LABELS}
        out.append(
            f"    {c:<18}{_f(weighted_ic(per)):>8}{_f(per['label_pm_2d_net']):>9}"
            f"{_f(per['label_pm_3d_net']):>9}{_f(per['label_pm_5d_net']):>9}"
        )
    # 2. per-stock TSIC
    work = tr.dropna(subset=LABELS).copy()
    feats = {c: work[c].astype(float) for c in LHB_COLS}
    tsic = per_stock_ts_ic(work, feats, LABELS, min_obs=10)
    out.append("  [个股TSIC]  col            wTSIC    TS2d     TS3d     TS5d")
    for c in LHB_COLS:
        per = {lab: tsic[lab][c] for lab in LABELS}
        out.append(
            f"    {c:<18}{_f(weighted_ic(per)):>8}{_f(per['label_pm_2d_net']):>9}"
            f"{_f(per['label_pm_3d_net']):>9}{_f(per['label_pm_5d_net']):>9}"
        )
    _event_study(tr, out)
    del work, xic, tsic, feats
    gc.collect()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    out = []
    full = _load(hs300_only=False)
    latest = full["date"].max()
    cutoff3 = latest - pd.DateOffset(years=3)
    _report(full[full["date"] >= cutoff3].reset_index(drop=True), "全市场×3年", out)
    del full
    gc.collect()

    text = "\n".join(out)
    print(text)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_diag_lhb_ts_vs_cross_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}")


if __name__ == "__main__":
    main()
