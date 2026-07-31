# -*- coding: utf-8 -*-
"""Step 1: Build and cache features to avoid timeout."""
import sys,os,time,warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

t0=time.time()
print('Loading V3...', flush=True)
panel=pd.read_parquet('data/panel_full_enriched_v3.parquet')
dates=sorted(panel['date'].unique())[-250:]
panel=panel[panel['date'].isin(dates)].copy()
print(f'{len(panel)} rows, {panel["date"].nunique()} dates ({time.time()-t0:.1f}s)', flush=True)

print('Cleaning...', flush=True)
from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner=CleaningPipeline()
t1=time.time()
main_df,dual_df,valve=cleaner.run_inference(panel)
print(f'main={len(main_df)} dual={len(dual_df)} valve={valve} ({time.time()-t1:.1f}s)', flush=True)

from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.dual_track_trainer import DualTrackTrainer
engine=FeatureEngineV35()
MODEL_PATHS={'main':'models/pipeline1/main_2026W31_fix.pkl','dual':'models/pipeline1/dual_2026W31_fix.pkl'}
bundles={b:DualTrackTrainer.load(p) for b,p in MODEL_PATHS.items()}

for board,df in [('main',main_df),('dual',dual_df)]:
    t1=time.time()
    cols=bundles[board]['feature_cols']
    print(f'{board}: building features ({len(cols)} cols)...', flush=True)
    feat=engine.build(df,None,cross_sectional_rank=False,inference_cols=cols)
    path=f'data/_feat_{board}_20260729.parquet'
    feat.to_parquet(path,index=False)
    print(f'{board}: {feat.shape} saved to {path} ({time.time()-t1:.1f}s)', flush=True)

print(f'\nDONE in {time.time()-t0:.0f}s', flush=True)
