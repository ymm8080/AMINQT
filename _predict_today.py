# -*- coding: utf-8 -*-
"""Quick prediction for 2026-07-29 using V3 panel data."""
import logging, sys, os, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger("predict")

import pandas as pd
import numpy as np
from app.pipeline1.predict_runner import find_bundles, run_prediction

bundle_paths = {
    'main': 'models/pipeline1/main_current.pkl',
    'dual': 'models/pipeline1/dual_current.pkl',
}
print("=== BUNDLES ===")
for b, p in bundle_paths.items():
    print(f"  {b}: {p} exists={os.path.exists(p)}")

t0 = time.time()
print("\n=== LOADING V3 PANEL ===")
panel = pd.read_parquet('data/panel_full_enriched_v3.parquet')
print(f"Loaded: {len(panel)} rows, {panel['symbol'].nunique()} stocks, {len(panel.columns)} cols in {time.time()-t0:.1f}s")
print(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")

today = panel[panel['date'] == pd.to_datetime('2026-07-29')]
print(f"2026-07-29: {len(today)} rows, {today['symbol'].nunique()} stocks")

# Trim to last 200 trading days for feature computation speed
all_dates = sorted(panel['date'].unique())
print(f"Total unique dates: {len(all_dates)}")
recent_dates = all_dates[-220:]  # ~1 year of daily data for features
panel_recent = panel[panel['date'].isin(recent_dates)].copy()
print(f"Trimmed panel: {len(panel_recent)} rows, {panel_recent['date'].nunique()} dates")

t1 = time.time()
print("\n=== RUNNING PREDICTION ===")
result = run_prediction(
    panel=panel_recent,
    trade_date='20260729',
    bundle_paths=bundle_paths,
    list_dir='data/lists/_fresh',
)
elapsed = time.time() - t1
print(f"Prediction done in {elapsed:.1f}s")

print(f"\n=== RESULT ===")
print(f"Mode: {result.get('mode')}, Empty: {result.get('empty')}, Cap: {result.get('cap_position', 'N/A')}")

if not result.get('empty') and len(result.get('list', pd.DataFrame())):
    lst = result['list']
    print(f"\n=== TOP 25 PICKS ({len(lst)} total) ===")
    show_cols = ['symbol', 'board', 'composite_score', 'pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up']
    avail = [c for c in show_cols if c in lst.columns]
    top = lst.nlargest(min(25, len(lst)), 'composite_score')
    print(top[avail].to_string())

    if 'board' in lst.columns:
        print(f"\n=== BOARD DISTRIBUTION ===\n{lst['board'].value_counts().to_string()}")
    if 'industry' in lst.columns:
        print(f"\n=== TOP 10 INDUSTRIES ===\n{lst['industry'].value_counts().head(10).to_string()}")
    if 'prob_up' in lst.columns:
        print(f"\n=== PROB_UP ===\n{lst['prob_up'].describe().to_string()}")

    # Rank score stats
    if 'rank_score' in lst.columns:
        print(f"\n=== RANK_SCORE ===\n{lst['rank_score'].describe().to_string()}")

    # day_change stats
    if 'day_change' in lst.columns:
        print(f"\n=== TODAY DAY_CHANGE (already realized) ===\n{lst['day_change'].describe().to_string()}")
else:
    print("EMPTY LIST GENERATED")
    print(f"Valve: {result.get('valve_state')}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
