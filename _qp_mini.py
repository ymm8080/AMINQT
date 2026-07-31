# -*- coding: utf-8 -*-
"""Minimal memory prediction — 60 days, only stocks present on latest date."""
import sys,os,time,warnings,gc
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
t_total=time.time()

print('Loading V3 (lightweight)...', flush=True)
panel=pd.read_parquet('data/panel_full_enriched_v3.parquet')
dates=sorted(panel['date'].unique())[-90:]  # 90 trading days ≈ 4 months
panel=panel[panel['date'].isin(dates)].copy()
# Only keep stocks present on latest date
latest_date=panel['date'].max()
active_syms=set(panel[panel['date']==latest_date]['symbol'])
panel=panel[panel['symbol'].isin(active_syms)].copy()
del active_syms; gc.collect()
print(f'{len(panel):,} rows, {panel["symbol"].nunique():,} stocks, {panel["date"].nunique()} dates ({time.time()-t_total:.1f}s)', flush=True)

# Models
MODEL_PATHS={'main':'models/pipeline1/main_2026W31_fix.pkl','dual':'models/pipeline1/dual_2026W31_fix.pkl'}
from app.pipeline1.dual_track_trainer import DualTrackTrainer
bundles={b:DualTrackTrainer.load(p) for b,p in MODEL_PATHS.items()}
for b,info in bundles.items():
    print(f'{b}: {len(info["feature_cols"])} cols', flush=True)

print('Cleaning...', flush=True)
from app.pipeline1.cleaning_pipeline import CleaningPipeline
t1=time.time()
main_df,dual_df,valve=CleaningPipeline().run_inference(panel)
gc.collect()
print(f'main={len(main_df):,} dual={len(dual_df):,} valve={valve} ({time.time()-t1:.1f}s)', flush=True)
if valve=='empty': print('VALVE EMPTY'); sys.exit(1)

# Free panel memory
del panel; gc.collect()

print('Features...', flush=True)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
engine=FeatureEngineV35()
frames={}
for board,df in [('main',main_df),('dual',dual_df)]:
    if len(df)==0: continue
    t1=time.time()
    cols=bundles[board]['feature_cols']
    feat=engine.build(df,None,cross_sectional_rank=False,inference_cols=cols)
    frames[board]=feat
    gc.collect()
    print(f'  {board}: {feat.shape} ({time.time()-t1:.1f}s)', flush=True)

print('Predicting...', flush=True)
from app.pipeline1.predictor import V35Predictor
predictor=V35Predictor(MODEL_PATHS)
candidates=[]
for board,feat in frames.items():
    surv=main_df if board=='main' else dual_df
    latest=surv['date'].max()
    syms=set(surv[surv['date']==latest]['symbol'])
    tf=feat[feat['symbol'].isin(syms)]
    print(f'  {board}: {len(tf)} stocks on {latest.date()}', flush=True)
    if len(tf):
        preds=predictor.predict(tf,board)
        candidates.append(preds)
cand=pd.concat(candidates,ignore_index=True)
print(f'{len(cand)} candidates', flush=True)

print('List generation...', flush=True)
from app.pipeline1.list_generator import ListGenerator
result=ListGenerator().emit(cand,env=None,market_state='range')
print(f'mode={result.get("mode")} empty={result.get("empty")} cap={result.get("cap_position","N/A")}', flush=True)

if not result.get('empty') and len(result.get('list',pd.DataFrame())):
    lst=result['list']
    show=['symbol','board','composite_score','pred_ret_1d','pred_ret_3d','pred_ret_5d','prob_up']
    avail=[c for c in show if c in lst.columns]
    top=lst.nlargest(min(25,len(lst)),'composite_score')
    print(f'\n{"="*70}')
    print(f'  TOP PICKS ({len(lst)} total) — 2026-07-29')
    print(f'{"="*70}')
    print(top[avail].to_string())
    if 'board' in lst.columns:
        print(f'\nBoard distribution:\n{lst["board"].value_counts().to_string()}')
    if 'industry' in lst.columns:
        print(f'\nTop industries:\n{lst["industry"].value_counts().head(10).to_string()}')
    if 'prob_up' in lst.columns:
        print(f'\nProb_up stats:\n{lst["prob_up"].describe().to_string()}')
    lst.to_parquet('data/lists/_list_20260729_final.parquet',index=False)
    print('\nSaved!')
else:
    print('EMPTY LIST')

print(f'\nTotal time: {time.time()-t_total:.0f}s')
