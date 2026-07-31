# -*- coding: utf-8 -*-
"""Quick prediction V2: use current-code-compatible models (no xrank, no relative_limit_strength)."""
import sys, os, time, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

t_total = time.time()

# ── 1. Load panel ──
print("Loading V3 panel...", flush=True)
panel = pd.read_parquet('data/panel_full_enriched_v3.parquet')
print(f"  {len(panel):,} rows, {panel['symbol'].nunique():,} stocks, {panel['date'].nunique()} dates", flush=True)

# ── 2. Load models ──
print("\nLoading models...", flush=True)
from app.pipeline1.dual_track_trainer import DualTrackTrainer

MODEL_PATHS = {
    'main': 'models/pipeline1/main_2026W31_fix.pkl',
    'dual': 'models/pipeline1/dual_2026W31_fix.pkl',
}
bundles = {}
for board, path in MODEL_PATHS.items():
    b = DualTrackTrainer.load(path)
    bundles[board] = b
    xrank_n = sum(1 for c in b['feature_cols'] if c.endswith('_xrank'))
    print(f"  {board}: {len(b['feature_cols'])} cols, {xrank_n} xrank, calibrator={'calibrator' in b}", flush=True)

# ── 3. Trim to recent dates ──
all_dates = sorted(panel['date'].unique())
recent = all_dates[-250:]
panel_trim = panel[panel['date'].isin(recent)].copy()
print(f"\nTrimmed to {panel_trim['date'].nunique()} dates, {len(panel_trim):,} rows", flush=True)

# ── 4. Cleaning ──
print("\n=== CLEANING ===", flush=True)
from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner = CleaningPipeline()
t1 = time.time()
main_df, dual_df, valve_state = cleaner.run_inference(panel_trim)
print(f"  main: {len(main_df):,} rows, dual: {len(dual_df):,} rows, valve={valve_state} ({time.time()-t1:.1f}s)", flush=True)

if valve_state == "empty":
    print("VALVE EMPTY", flush=True)
    sys.exit(1)

# ── 5. Build features ──
print("\n=== FEATURE BUILD ===", flush=True)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
engine = FeatureEngineV35()

frames = {}
for board, df in [('main', main_df), ('dual', dual_df)]:
    if len(df) == 0:
        continue
    t1 = time.time()
    feat_cols = bundles[board]['feature_cols']
    use_xrank = board == 'dual'  # dual gets cross_sectional_rank per train_runner
    feat = engine.build(df, None, cross_sectional_rank=use_xrank, inference_cols=feat_cols)
    frames[board] = feat
    print(f"  {board}: {feat.shape} (xrank={use_xrank}) in {time.time()-t1:.1f}s", flush=True)

# ── 6. Predict ──
print("\n=== PREDICTION ===", flush=True)
from app.pipeline1.predictor import V35Predictor
predictor = V35Predictor(MODEL_PATHS)

all_candidates = []
for board, feat in frames.items():
    surv = main_df if board == 'main' else dual_df
    latest_date = surv['date'].max()
    latest_symbols = set(surv[surv['date'] == latest_date]['symbol'])
    today_feat = feat[feat['symbol'].isin(latest_symbols)]
    print(f"  {board}: {len(today_feat)} stocks on {latest_date.date()}", flush=True)
    if len(today_feat) == 0:
        continue
    preds = predictor.predict(today_feat, board)
    all_candidates.append(preds)

candidates = pd.concat(all_candidates, ignore_index=True)
print(f"\nTotal candidates: {len(candidates)}", flush=True)

# ── 7. List generation ──
print("\n=== LIST GENERATION ===", flush=True)
from app.pipeline1.list_generator import ListGenerator
lister = ListGenerator()
result = lister.emit(candidates, env=None, market_state='range')

print(f"\n{'='*60}")
print(f"  RESULT: mode={result.get('mode')}, empty={result.get('empty')}")
print(f"  Cap position: {result.get('cap_position', 'N/A')}")
print(f"{'='*60}")

if result.get('empty') or len(result.get('list', pd.DataFrame())) == 0:
    print("EMPTY LIST", flush=True)
    sys.exit(0)

lst = result['list']
show = ['symbol', 'board', 'composite_score', 'pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up']
avail = [c for c in show if c in lst.columns]

print(f"\n=== TOP 25 PICKS (out of {len(lst)}) ===")
top = lst.nlargest(min(25, len(lst)), 'composite_score')
print(top[avail].to_string())

if 'board' in lst.columns:
    print(f"\n=== BOARD DISTRIBUTION ===\n{lst['board'].value_counts().to_string()}")

if 'industry' in lst.columns:
    print(f"\n=== TOP 10 INDUSTRIES ===\n{lst['industry'].value_counts().head(10).to_string()}")

if 'prob_up' in lst.columns:
    print(f"\n=== PROB_UP STATS ===\n{lst['prob_up'].describe().to_string()}")

print(f"\nTotal: {time.time()-t_total:.1f}s", flush=True)

# Save
lst.to_parquet('data/lists/_list_20260729_fresh_v2.parquet', index=False)
print("Saved to data/lists/_list_20260729_fresh_v2.parquet")
