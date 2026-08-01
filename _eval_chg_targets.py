# -*- coding: utf-8 -*-
"""Targeted test: does adding _chgN to OHLCV and CYQ improve IC?

Runs directly on panel columns (no full feature engine) — fast, targeted.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Config ──
OHLCV_COLS = [
    'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq',
    'open', 'high', 'low', 'close',
    'volume', 'amount', 'turnover_rate',
]
CYQ_COLS = [
    'chip_concentration', 'conc_90', 'winner_ratio',
    'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct',
    'weight_avg', 'winner_rate', 'his_low', 'his_high',
]
WINDOWS = (1, 3, 5, 10, 20)
LABEL_COLS = ['label_1d', 'label_3d', 'label_5d']

def compute_ic(df, factor_col, label_col):
    """Mean |Rank IC| with t-stat."""
    sub = df[['date', factor_col, label_col]].dropna()
    if len(sub) < 100:
        return {'mean_abs_ic': 0.0, 'mean_ic': 0.0, 'n_dates': 0, 'ic_std': 0.0}
    daily_ics = []
    for _, g in sub.groupby('date'):
        if len(g) < 30:
            continue
        try:
            ic = spearmanr(g[factor_col], g[label_col]).statistic
            if not np.isnan(ic):
                daily_ics.append(ic)
        except (ValueError, TypeError):
            pass
    if len(daily_ics) < 10:
        return {'mean_abs_ic': 0.0, 'mean_ic': 0.0, 'n_dates': 0, 'ic_std': 0.0}
    ics = np.array(daily_ics)
    return {
        'mean_abs_ic': float(np.abs(ics).mean()),
        'mean_ic': float(ics.mean()),
        'n_dates': len(daily_ics),
        'ic_std': float(ics.std()),
        't_stat': float(ics.mean() / (ics.std() / np.sqrt(len(ics)))) if ics.std() > 0 else 0,
    }

# ── 1. Load ──
print("Loading V3 panel...", flush=True)
t0 = time.time()
df = pd.read_parquet('data/panel_full_enriched_v3.parquet')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date'])
print(f"  {len(df):,} rows, {df['symbol'].nunique()} stocks, {df['date'].nunique()} dates ({time.time()-t0:.1f}s)", flush=True)

# ── 2. Build labels ──
print("\nBuilding labels (PM session)...", flush=True)
t1 = time.time()
from app.pipeline1.label_engine import LabelEngine
df = LabelEngine.build_labels(df, session="PM")
df = LabelEngine.mask_recent_days(df, days=6)  # 屏蔽近端未成熟标签, 防 IC 泄漏
print(f"  Done in {time.time()-t1:.1f}s", flush=True)

# ── 3. Find available columns ──
ohlcv_avail = [c for c in OHLCV_COLS if c in df.columns]
cyq_avail = [c for c in CYQ_COLS if c in df.columns]
print(f"\nAvailable OHLCV: {len(ohlcv_avail)}/{len(OHLCV_COLS)}")
print(f"Available CYQ:    {len(cyq_avail)}/{len(CYQ_COLS)}")

# ── 4. Compute _chgN for OHLCV and CYQ groups ──
print("\nComputing _chgN / _pct_chgN for OHLCV + CYQ only...", flush=True)
t1 = time.time()
added_ohlcv = []
added_cyq = []

for col in ohlcv_avail:
    grp = df.groupby('symbol')[col]
    for w in WINDOWS:
        abs_col = f"{col}_chg{w}"
        pct_col = f"{col}_pct_chg{w}"
        df[abs_col] = grp.diff(w)
        df[pct_col] = grp.pct_change(w, fill_method=None)
        if df[abs_col].notna().sum() > 100:
            added_ohlcv.append(abs_col)
            added_ohlcv.append(pct_col)

for col in cyq_avail:
    grp = df.groupby('symbol')[col]
    for w in WINDOWS:
        abs_col = f"{col}_chg{w}"
        pct_col = f"{col}_pct_chg{w}"
        df[abs_col] = grp.diff(w)
        df[pct_col] = grp.pct_change(w, fill_method=None)
        if df[abs_col].notna().sum() > 100:
            added_cyq.append(abs_col)
            added_cyq.append(pct_col)

print(f"  Added {len(added_ohlcv)} OHLCV-derived columns")
print(f"  Added {len(added_cyq)} CYQ-derived columns")
print(f"  Done in {time.time()-t1:.1f}s", flush=True)

# ── 5. IC Evaluation ──
print("\n" + "=" * 90)
print("IC COMPARISON: Baseline (raw) vs OHLCV _chg vs CYQ _chg")
print("=" * 90)

for label in LABEL_COLS:
    print(f"\n--- {label} ---")

    # Baseline: raw columns
    raw_ics = []
    for col in ohlcv_avail + cyq_avail:
        r = compute_ic(df, col, label)
        if r['n_dates'] > 10:
            raw_ics.append(r['mean_abs_ic'])

    # OHLCV-derived
    ohlcv_ics = []
    for col in added_ohlcv:
        r = compute_ic(df, col, label)
        if r['n_dates'] > 10:
            ohlcv_ics.append(r['mean_abs_ic'])

    # CYQ-derived
    cyq_ics = []
    for col in added_cyq:
        r = compute_ic(df, col, label)
        if r['n_dates'] > 10:
            cyq_ics.append(r['mean_abs_ic'])

    print(f"  {'Group':<20} {'N features':>10} {'Mean|IC|':>10} {'Max|IC|':>10} {'t>0 ratio':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, ics in [('RAW (levels)', raw_ics), ('OHLCV _chg/pct', ohlcv_ics), ('CYQ _chg/pct', cyq_ics)]:
        if ics:
            arr = np.array(ics)
            positive = np.mean(arr > 0.001)
            print(f"  {name:<20} {len(arr):>10} {arr.mean():>10.4f} {arr.max():>10.4f} {positive:>10.1%}")
        else:
            print(f"  {name:<20} {0:>10} {'--':>10} {'--':>10} {'--':>10}")

    # Top-5 OHLCV-derived
    print(f"\n  🔥 Top-5 OHLCV-derived for {label}:")
    ohlcv_results = [(col, compute_ic(df, col, label)['mean_abs_ic'])
                     for col in added_ohlcv]
    ohlcv_results.sort(key=lambda x: -x[1])
    for col, ic_val in ohlcv_results[:5]:
        print(f"    {col:<40} {ic_val:.4f}")

    # Top-5 CYQ-derived
    print(f"\n  🔥 Top-5 CYQ-derived for {label}:")
    cyq_results = [(col, compute_ic(df, col, label)['mean_abs_ic'])
                   for col in added_cyq]
    cyq_results.sort(key=lambda x: -x[1])
    for col, ic_val in cyq_results[:5]:
        print(f"    {col:<40} {ic_val:.4f}")

# ── 6. Summary: per-window comparison ──
print("\n" + "=" * 90)
print("PER-WINDOW BREAKDOWN (mean|IC| across all labels)")
print("=" * 90)
print(f"  {'Window':>8} {'OHLCV |IC|':>12} {'OHLCV N':>8} {'CYQ |IC|':>12} {'CYQ N':>8}")
print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8}")
for w in WINDOWS:
    ohlcv_w = [c for c in added_ohlcv if f'_chg{w}' in c or f'_pct_chg{w}' in c]
    cyq_w = [c for c in added_cyq if f'_chg{w}' in c or f'_pct_chg{w}' in c]
    ohlcv_ic_w = []
    cyq_ic_w = []
    for c in ohlcv_w:
        for lbl in LABEL_COLS:
            r = compute_ic(df, c, lbl)
            if r['n_dates'] > 10:
                ohlcv_ic_w.append(r['mean_abs_ic'])
    for c in cyq_w:
        for lbl in LABEL_COLS:
            r = compute_ic(df, c, lbl)
            if r['n_dates'] > 10:
                cyq_ic_w.append(r['mean_abs_ic'])
    o_mean = np.mean(ohlcv_ic_w) if ohlcv_ic_w else 0
    c_mean = np.mean(cyq_ic_w) if cyq_ic_w else 0
    print(f"  chg{w:>3}      {o_mean:>12.4f} {len(ohlcv_ic_w):>8} {c_mean:>12.4f} {len(cyq_ic_w):>8}")

print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
