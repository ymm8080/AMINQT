# -*- coding: utf-8 -*-
"""Evaluate ALL columns with IC + IR (Information Ratio = IC_mean / IC_std) per feature."""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.daily_rank_ic import daily_rank_ic_series, mean_rank_ic
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.cleaning_pipeline import board_of

panel = pd.read_parquet("data/panel_full_enriched_v3.parquet")
if "board" not in panel.columns:
    panel["board"] = panel["symbol"].map(board_of)
max_d = panel["date"].max()
cutoff = max_d - pd.Timedelta(days=400)
recent = panel[panel["date"] >= cutoff]

SKIP = {
    "symbol",
    "date",
    "board",
    "industry",
    "name",
    "tradestatus",
    "time",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "close_hfq",
    "is_suspended",
    "announce_date",
    "limit_pct",
    "PE_TTM",
    "touched_limit_up",
    "score_rank",
    "rank_amount",
    "rank_ff_turnover",
    "liquidity_score",
    "churn_suspect",
    "is_virtual",
    "price_1455",
    "adv20",
    "schema_version",
    "market_state",
    "free_float_turnover_rate",
}

rng = np.random.RandomState(42)

for bv, bn in [("main", "main"), ("GEM", "dual")]:
    mask = (
        recent["board"] == bv if bn == "main" else recent["board"].isin(["GEM", "STAR"])
    )
    syms = recent[mask]["symbol"].unique()
    pick = rng.choice(syms, min(200, len(syms)), replace=False)
    df = recent[recent["symbol"].isin(pick)].copy()
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)  # 屏蔽近端未成熟标签, 防 IC 泄漏
    lc = "label_1d_net" if "label_1d_net" in df.columns else "label_1d"
    raw_cols = [
        c
        for c in df.columns
        if c not in SKIP and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]

    results = []
    for col in raw_cols:
        valid = df[[col, lc, "date"]].dropna()
        if len(valid) < 50:
            continue
        try:
            ic = mean_rank_ic(valid, col, lc)  # signed mean IC
            ics = daily_rank_ic_series(valid, col, lc)
            ic_std = float(np.nanstd(ics.values)) if len(ics) >= 5 else 0
            ir = ic / ic_std if ic_std > 0 else 0  # Information Ratio = IC / IC_std
            pos = float((ics > 0).mean())
            nan = float(df[col].isna().mean())
            results.append(
                {
                    "factor": col,
                    "IC": round(ic, 4),
                    "IR": round(ir, 2),
                    "IC_std": round(ic_std, 4),
                    "pos": round(pos, 3),
                    "nan": round(nan, 3),
                }
            )
        except Exception:
            pass

    # Sort by signed IC descending
    ranked = sorted(results, key=lambda x: x["IC"], reverse=True)

    print(f"\n{'=' * 100}")
    print(
        f"  {bn.upper()} — ALL {len(ranked)} FEATURES: IC + IR (Information Ratio) per feature"
    )
    print(f"{'=' * 100}")
    print(
        f"  {'Rank':<4s} {'Column':<30s} {'IC':>8s} {'IR':>8s} {'IC_std':>8s} {'Pos%':>7s} {'NaN%':>7s}"
    )
    print(f"  {'-' * 80}")
    for i, t in enumerate(ranked, 1):
        if i > 50:
            break
        print(
            f"  {i:<4d} {t['factor']:<30s} {t['IC']:>+8.4f} {t['IR']:>+8.2f} {t['IC_std']:>8.4f} {t['pos']:>7.1%} {t['nan']:>7.1%}"
        )

    # Bottom by IC
    bottom = sorted(results, key=lambda x: x["IC"])
    print("\n  --- Bottom 20 (strongest negative IC) ---")
    for i, t in enumerate(bottom[:20], 1):
        print(
            f"  {i:<4d} {t['factor']:<30s} {t['IC']:>+8.4f} {t['IR']:>+8.2f} {t['IC_std']:>8.4f} {t['pos']:>7.1%} {t['nan']:>7.1%}"
        )

    # Summary
    n_pos = sum(1 for r in ranked if r["IC"] > 0.01)
    n_neg = sum(1 for r in ranked if r["IC"] < -0.01)
    n_flat = len(ranked) - n_pos - n_neg
    print(f"\n  IC > +0.01: {n_pos}  |  IC < -0.01: {n_neg}  |  |IC| <= 0.01: {n_flat}")
