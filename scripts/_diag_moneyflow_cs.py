# -*- coding: utf-8 -*-
"""_diag_moneyflow_cs.py — 主力持仓比例代理 截面诊断 (逐日横截面 Rank IC + 分位单调性).

背景 (2026-08-06): 用户质疑个股时序 wTSIC -0.066 是否掩盖了截面判别力.
实际用法是把特征按日截面排序打分 → 真正的验收口径是逐日横截面 Rank IC
(每日: 全市场股票特征值秩 vs 前瞻收益秩, Spearman), 再加分位单调性.

复用 _diag_moneyflow_hold.py 的数据管线 (panel + Tushare moneyflow 3y 缓存 + 7 变体特征).

WORM → data/_diag_moneyflow_cs_<ts>.json + .log

用法: python scripts/_diag_moneyflow_cs.py [--days 750] [--universe all]
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.pipeline1.label_engine import LabelEngine
from scripts._diag_column_feed import MASK_RECENT_DAYS
from scripts._diag_moneyflow_hold import (
    backfill_moneyflow,
    build_features,
    load_hs300,
    load_panel_window,
)
from scripts._measure_topn import HORIZONS
from scripts._reclassify_all_features import add_label_pm_10d_net

logging.disable(logging.CRITICAL)

MIN_N = 30  # 当日截面最小样本


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def daily_cs_ic(work: pd.DataFrame, feat: str, label: str) -> dict:
    """逐日横截面 Spearman Rank IC → 加权均值 / IR / 同号占比."""
    ics, ns = [], []
    for d, sub in work.groupby("date", sort=True):
        x, y = sub[feat].values, sub[label].values
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < MIN_N:
            continue
        r = spearmanr(x[m], y[m]).statistic
        ics.append(r)
        ns.append(int(m.sum()))
    if not ics:
        return {"mean": None, "ir": None, "n_days": 0, "sign_consist": None}
    ics, ns = np.array(ics), np.array(ns)
    wmean = float(np.average(ics, weights=ns))
    sd = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
    return {
        "mean": round(wmean, 4),
        "ir": round(wmean / sd, 3) if sd > 0 else None,
        "n_days": int(len(ics)),
        "sign_consist": round(float((ics * np.sign(wmean) > 0).mean()), 3)
        if wmean != 0
        else None,
    }


def quintile_spread(work: pd.DataFrame, feat: str, label: str) -> dict:
    """每日按特征秩等分 5 组 → 各组前瞻收益 → 跨日加权 Q1..Q5 + Q5-Q1 价差."""
    df = work[["date", feat, label]].dropna(subset=[label]).copy()
    df = df.dropna(subset=[feat])
    if not len(df):
        return {"q": {}, "spread": None, "n": 0}
    df["pct"] = df.groupby("date")[feat].rank(pct=True)
    df["q"] = (df["pct"] * 5).clip(0, 4).astype(int)
    grp = df.groupby(["date", "q"])[label].agg(["mean", "count"]).reset_index()
    rows = []
    for q in range(5):
        sub = grp[grp["q"] == q]
        rows.append(float(np.average(sub["mean"], weights=sub["count"])))
    return {
        "q": {str(i + 1): round(rows[i], 4) for i in range(5)},
        "spread": round(rows[4] - rows[0], 4),
        "n": int(len(df)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=750)
    ap.add_argument("--universe", choices=["hs300", "all"], default="all")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    ts = _ts()
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        f"_diag_moneyflow_cs_{ts}.log",
    )
    json_path = log_path.replace(".log", ".json")
    out = []
    print(f"[start] universe={args.universe} × 最近 {args.days} 交易日, ts={ts}")

    hs300 = set(load_hs300()) if args.universe == "hs300" else None
    work, trade_dates = load_panel_window(args.days, hs300)
    print(
        f"[panel] {len(work)} 行 / {work['symbol'].nunique()} 只 / "
        f"{trade_dates[0].date()} ~ {trade_dates[-1].date()}"
    )
    out.append(
        f"[panel] {len(work)} 行 / {work['symbol'].nunique()} 只 / "
        f"{trade_dates[0].date()} ~ {trade_dates[-1].date()}"
    )

    mf = backfill_moneyflow(trade_dates, refresh=args.refresh)
    work = build_features(work, mf)
    del mf
    gc.collect()

    work = LabelEngine.build_labels(work, session="PM")
    work = LabelEngine.mask_suspension(work)
    work = LabelEngine.mask_recent_days(work, days=MASK_RECENT_DAYS)
    work = add_label_pm_10d_net(work)
    print(f"[labels] 生产行集 {len(work)} 行")
    out.append(f"[labels] 生产行集 {len(work)} 行")

    feats = {
        "main_hold_ratio": "主力持仓代理(全窗累计/流通市值)",
        "main_hold_ratio_60d": "主力持仓代理(60日滚动/流通市值)",
        "hold_chg_5d": "主力持仓周变化(60d代理5日差)",
        "hold_chg_20d": "主力持仓月变化(60d代理20日差)",
        "hold_chg_5d_x_sgnbias5": "持仓周变化×sgn(bias5)",
        "hold_chg_20d_x_sgnbias20": "持仓月变化×sgn(bias20)",
        "hold_60d_x_sgnbias20": "持仓60d×sgn(bias20)",
    }
    res = {
        "ts": ts,
        "universe": args.universe,
        "days": args.days,
        "n_rows": int(len(work)),
        "n_symbols": int(work["symbol"].nunique()),
        "horizons": [int(k) for k in HORIZONS],
    }

    # 1) 逐日横截面 Rank IC
    out.append("")
    out.append("=== 逐日横截面 Rank IC (Spearman, 加权均值 / IR / 同号占比) ===")
    out.append(
        f"{'feature':<28}{'T+5d wIC':>12}{'IR':>8}{'consist':>9}"
        f"{'T+10d wIC':>12}{'IR':>8}{'consist':>9}"
    )
    out.append("-" * 80)
    for name, label in feats.items():
        row = {}
        for k in HORIZONS:
            lab = f"label_pm_{k}d_net"
            row[k] = daily_cs_ic(work, name, lab)
        res[name] = {"label": label, "csic": row}
        out.append(
            f"{name:<28}"
            f"{_f(row[5]['mean']):>12}{_f(row[5]['ir']):>8}{_f(row[5]['sign_consist']):>9}"
            f"{_f(row[10]['mean']):>12}{_f(row[10]['ir']):>8}{_f(row[10]['sign_consist']):>9}"
        )

    # 2) 分位单调性: 每日等分 5 组, Q1..Q5 前瞻收益, Q5-Q1 价差
    out.append("")
    out.append("=== 分位单调性 (每日按特征秩等分 5 组, 跨日加权前瞻净收益) ===")
    for name in ("main_hold_ratio", "main_hold_ratio_60d"):
        for k in HORIZONS:
            lab = f"label_pm_{k}d_net"
            qs = quintile_spread(work, name, lab)
            res.setdefault(name, {}).setdefault("quintile", {})[k] = qs
            qs_txt = " | ".join(f"Q{i}: {v:+.4f}" for i, v in qs["q"].items())
            out.append(
                f"[{name}] T+{k}d  {qs_txt}  | Q5-Q1={qs['spread']:+.4f}  n={qs['n']}"
            )

    print("\n".join(out))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[WORM] {log_path}")
    print(f"[WORM] {json_path}")


def _f(v):
    return f"{v:+.4f}" if v is not None and v == v else "   nan"


if __name__ == "__main__":
    main()
