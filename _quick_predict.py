# -*- coding: utf-8 -*-
"""Quick prediction: load model, compute features on recent data, predict."""
import sys, os, time, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

t_total = time.time()

# ── 1. Load panel ──
t0 = time.time()
print("Loading panel...", flush=True)
panel = pd.read_parquet('data/panel_full_enriched_v3.parquet')
print(f"  {len(panel):,} rows, {panel['symbol'].nunique():,} stocks, {panel['date'].nunique()} dates in {time.time()-t0:.1f}s", flush=True)

# ── 2. Load models ──
print("\nLoading models...", flush=True)
from app.pipeline1.dual_track_trainer import DualTrackTrainer

bundles = {}
for board in ('main', 'dual'):
    path = f'models/pipeline1/{board}_current.pkl'
    b = DualTrackTrainer.load(path)
    bundles[board] = b
    print(f"  {board}: {len(b['feature_cols'])} feature cols, models={list(b['models'].keys())}", flush=True)
    if 'quantile_models' in b:
        print(f"    quantile_models present")
    if 'calibrator' in b:
        print(f"    calibrator present")
    if 'rank_model' in b:
        print(f"    rank_model present")

# ── 3. Trim to recent dates (last 220 trading days ≈ 1 year) ──
all_dates = sorted(panel['date'].unique())
recent = all_dates[-250:]
panel_trim = panel[panel['date'].isin(recent)].copy()
print(f"\nTrimmed to last {panel_trim['date'].nunique()} dates, {len(panel_trim):,} rows", flush=True)

# ── 4. Run cleaning (inference mode) ──
print("\n=== CLEANING (step 0→4) ===", flush=True)
from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner = CleaningPipeline()
t1 = time.time()
main_df, dual_df, valve_state = cleaner.run_inference(panel_trim)
print(f"  main: {len(main_df):,} rows, dual: {len(dual_df):,} rows, valve={valve_state} in {time.time()-t1:.1f}s", flush=True)

if valve_state == "empty":
    print("  VALVE EMPTY - no stocks pass liquidity filter", flush=True)
    sys.exit(1)

# ── 5. Build features ──
print("\n=== FEATURE BUILD ===", flush=True)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
engine = FeatureEngineV35()

t1 = time.time()
frames = {}
for board, df in [('main', main_df), ('dual', dual_df)]:
    if len(df) == 0:
        print(f"  {board}: no data, skipping", flush=True)
        continue
    feat_cols = bundles[board]['feature_cols']
    feat = engine.build(df, None, inference_cols=feat_cols)
    frames[board] = feat
    print(f"  {board}: feature {feat.shape} built in {time.time()-t1:.1f}s", flush=True)
    t1 = time.time()

# ── 6. Predict ──
print("\n=== PREDICTION ===", flush=True)
from app.pipeline1.predictor import V35Predictor

predictor = V35Predictor({
    b: f'models/pipeline1/{b}_current.pkl' for b in bundles
})

all_candidates = []
for board, feat in frames.items():
    df = frames[board]
    surv_main = main_df if board == 'main' else dual_df
    latest_date = surv_main['date'].max()
    latest_symbols = set(surv_main[surv_main['date'] == latest_date]['symbol'])
    today_feat = df[df['symbol'].isin(latest_symbols)]
    print(f"  {board}: predicting on {len(today_feat)} stocks (latest date: {latest_date.date()})", flush=True)
    if len(today_feat) == 0:
        continue
    preds = predictor.predict(today_feat, board)
    print(f"    {len(preds)} predictions", flush=True)
    all_candidates.append(preds)

if not all_candidates:
    print("NO CANDIDATES", flush=True)
    sys.exit(1)

candidates = pd.concat(all_candidates, ignore_index=True)
print(f"\nTotal candidates: {len(candidates)}", flush=True)

# ── 7. Generate list ──
print("\n=== LIST GENERATION ===", flush=True)
from app.pipeline1.list_generator import ListGenerator

lister = ListGenerator()
result = lister.emit(candidates, env=None, market_state='range')

print(f"\n{'='*60}")
print(f"  RESULT: mode={result.get('mode')}, empty={result.get('empty')}", flush=True)
print(f"  Cap position: {result.get('cap_position', 'N/A')}", flush=True)
print(f"{'='*60}\n")

if result.get('empty') or len(result.get('list', pd.DataFrame())) == 0:
    print("EMPTY LIST - no stocks selected", flush=True)
    sys.exit(0)

lst = result['list']
show = ['symbol', 'board', 'composite_score', 'pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up']
avail = [c for c in show if c in lst.columns]

print(f"\n=== TOP 25 PICKS (out of {len(lst)}) ===")
top = lst.nlargest(min(25, len(lst)), 'composite_score')
print(top[avail].to_string(), flush=True)

print(f"\n=== BOARD DISTRIBUTION ===")
if 'board' in lst.columns:
    print(lst['board'].value_counts().to_string(), flush=True)

if 'industry' in lst.columns:
    print(f"\n=== TOP 10 INDUSTRIES ===")
    print(lst['industry'].value_counts().head(10).to_string(), flush=True)

print(f"\n=== PROB_UP STATS ===")
if 'prob_up' in lst.columns:
    print(lst['prob_up'].describe().to_string(), flush=True)

print(f"\nTotal time: {time.time()-t_total:.1f}s", flush=True)

# Save
lst.to_parquet('data/lists/_list_20260729_fresh.parquet', index=False)
print("Saved to data/lists/_list_20260729_fresh.parquet", flush=True)
