# -*- coding: utf-8 -*-
"""
Full-column IC/ICIR evaluation — every column in enriched panel.
Usage: python _icir_dim_eval.py
"""
from __future__ import annotations
import sys, os, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.getcwd())
from app.pipeline1.label_engine import LabelEngine
from app.utils.daily_rank_ic import daily_rank_ic_series

# -- IC / ICIR utils --
def rank_ic(df, x, y):
    """日度 Rank IC 均值 (带符号). 委托公共模块保证口径一致."""
    ics = daily_rank_ic_series(df, x, y)
    return float(ics.mean()) if len(ics) else 0.0

def icir(df, x, y):
    """ICIR = |mean| / std (稳定性). 委托公共模块."""
    ics = daily_rank_ic_series(df, x, y)
    if len(ics) < 10: return 0.0
    m, s = float(ics.mean()), float(ics.std())
    return abs(m) / s if s > 0 else 0.0

# -- Skip columns --
SKIP = {
    "symbol", "date", "announce_date",
}
SKIP_PREFIX = ("label_",)  # skip label cols themselves

# ── Load + label ──
DATA = "data/panel_full_enriched_v4_20260729.parquet"
print("=" * 120)
print("FULL-COLUMN IC/ICIR EVALUATION")
print("Data:", DATA)
print("=" * 120)

df = pd.read_parquet(DATA)
print(f"Loaded: {len(df):,} rows, {len(df.columns)} cols, date {df['date'].min().date()} -> {df['date'].max().date()}")

# ==== IC-safe filtering: 剔除 ST/停牌/涨跌停/次新股 ====
def _ic_tradable_mask(df):
    mask = pd.Series(True, index=df.index)
    for col in ("is_st", "is_suspended"):
        if col in df.columns:
            mask &= ~df[col].astype(bool)
    if "limit_pct" in df.columns and "pre_close" in df.columns:
        limit_up = df["pre_close"] * (1 + df["limit_pct"] / 100)
        at_limit = df["close"] >= limit_up * 0.995
        mask &= ~at_limit
        limit_down = df["pre_close"] * (1 - df["limit_pct"] / 100)
        at_limit_down = df["close"] <= limit_down * 1.005
        mask &= ~at_limit_down
    if "list_days" in df.columns:
        mask &= df["list_days"] >= 60
    return mask

ic_mask = _ic_tradable_mask(df)
removed = (~ic_mask).sum()
print(f"IC-sample filter: removed {removed:,} / {len(df):,} rows ({removed/len(df)*100:.1f}%)")
df = df[ic_mask].copy()

df = LabelEngine.build_labels(df, session="PM")
df = LabelEngine.mask_recent_days(df, days=6)  # 屏蔽近端未成熟标签, 防 IC 泄漏

MAIN = "label_pm_1d_net"
AUX  = "label_pm_5d_net"

# ── Candidate columns ──
all_cols = [c for c in df.columns if c not in SKIP and not c.startswith(SKIP_PREFIX)]
print(f"Candidate columns to evaluate: {len(all_cols)}")

# ── Evaluate all ──
results = []
for i, f in enumerate(all_cols):
    try:
        ic1 = rank_ic(df, f, MAIN)
        ir1 = icir(df, f, MAIN)
        ic5 = rank_ic(df, f, AUX)
        ir5 = icir(df, f, AUX)
        sub = df[["date", f, MAIN]].dropna()
        nd = sub["date"].nunique()
    except Exception as e:
        print(f"  SKIP {f}: {e}")
        results.append((f, 0.0, 0.0, 0.0, 0.0, 0))
        continue
    results.append((f, ic1, ir1, ic5, ir5, nd))
    if (i + 1) % 20 == 0:
        print(f"  ... {i+1}/{len(all_cols)} done")

print(f"\nEvaluation complete: {len(all_cols)} columns\n")

# ── Sort by |IC_1d| desc ──
results.sort(key=lambda x: abs(x[1]), reverse=True)

def grade(v):
    av = abs(v)
    if av >= 0.03: return "***"
    if av >= 0.02: return "**"
    if av >= 0.01: return "*"
    return "."

# ── Full table ──
hdr = f"  {'Rank':<5s} {'Field':<30s} {'IC_1d':>8s} {'ICIR_1d':>8s} {'IC_5d':>8s} {'ICIR_5d':>8s} {'nDays':>7s}  {'Verdict'}"
print(hdr)
print(f"  {'─'*5} {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}  {'─'*20}")

strong_cnt = weak_cnt = dead_cnt = 0
strong_list = []
weak_list = []
dead_list = []

for rank_pos, (f, ic1, ir1, ic5, ir5, nd) in enumerate(results, 1):
    g1 = grade(ic1)
    if abs(ic1) >= 0.02 and ir1 >= 0.3:
        verdict = "STRONG"
        strong_cnt += 1; strong_list.append(f)
    elif abs(ic1) >= 0.01:
        verdict = "WEAK"
        weak_cnt += 1; weak_list.append(f)
    else:
        verdict = "DEAD"
        dead_cnt += 1; dead_list.append(f)
    print(f"  {rank_pos:<5d} {f:<30s} {ic1:>7.4f}{g1} {ir1:>7.2f} {ic5:>7.4f}{grade(ic5)} {ir5:>7.2f} {nd:>7d}  {verdict}")

# ── Summary ──
print(f"\n{'='*120}")
print(f"SUMMARY")
print(f"{'='*120}")
print(f"  STRONG (|IC|>=0.02 & ICIR>=0.3): {strong_cnt} cols")
for f in strong_list:
    print(f"    + {f}")
print(f"\n  WEAK   (|IC|>=0.01):             {weak_cnt} cols")
for f in weak_list:
    print(f"    ~ {f}")
print(f"\n  DEAD   (|IC|<0.01):              {dead_cnt} cols")
for f in dead_list:
    print(f"    - {f}")

print(f"\n{'='*120}")
print("Gate (P19.0): |IC|>=0.03 & ICIR>=0.3 & high-vol-bucket IC>=0.02")
print("Recommend: STRONG -> plumb into feature_engine, WEAK -> optional, DEAD -> skip")
print("=" * 120)
