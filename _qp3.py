# -*- coding: utf-8 -*-
import sys,os,time,warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
t_total=time.time()

print('Loading V3...', flush=True)
panel=pd.read_parquet('data/panel_full_enriched_v3.parquet')
dates=sorted(panel['date'].unique())[-250:]
panel=panel[panel['date'].isin(dates)].copy()
print(f'{len(panel)} rows, {panel["date"].nunique()} dates', flush=True)

print('Cleaning...', flush=True)
from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner=CleaningPipeline()
t1=time.time()
main_df,dual_df,valve=cleaner.run_inference(panel)
print(f'main={len(main_df)} dual={len(dual_df)} valve={valve} ({time.time()-t1:.1f}s)', flush=True)

print('Features main...', flush=True)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
engine=FeatureEngineV35()
MODEL_PATHS={'main':'models/pipeline1/main_2026W31_fix.pkl','dual':'models/pipeline1/dual_2026W31_fix.pkl'}
from app.pipeline1.dual_track_trainer import DualTrackTrainer
bundles={b:DualTrackTrainer.load(p) for b,p in MODEL_PATHS.items()}

t1=time.time()
feat_main=engine.build(main_df,None,cross_sectional_rank=False,inference_cols=bundles['main']['feature_cols'])
print(f'main features: {feat_main.shape} ({time.time()-t1:.1f}s)', flush=True)

t1=time.time()
feat_dual=engine.build(dual_df,None,cross_sectional_rank=False,inference_cols=bundles['dual']['feature_cols'])
print(f'dual features: {feat_dual.shape} ({time.time()-t1:.1f}s)', flush=True)

print('Predicting...', flush=True)
from app.pipeline1.predictor import V35Predictor
predictor=V35Predictor(MODEL_PATHS)
candidates=[]
for board,feat,surv in [('main',feat_main,main_df),('dual',feat_dual,dual_df)]:
    latest=surv['date'].max()
    syms=set(surv[surv['date']==latest]['symbol'])
    tf=feat[feat['symbol'].isin(syms)]
    print(f'{board}: {len(tf)} stocks on {latest.date()}', flush=True)
    if len(tf):
        preds=predictor.predict(tf,board)
        candidates.append(preds)
cand=pd.concat(candidates,ignore_index=True)
print(f'{len(cand)} candidates', flush=True)

print('List gen...', flush=True)
from app.pipeline1.list_generator import ListGenerator
result=ListGenerator().emit(cand,env=None,market_state='range')
print(f'mode={result.get("mode")} empty={result.get("empty")} cap={result.get("cap_position","N/A")}', flush=True)

if not result.get('empty') and len(result.get('list',pd.DataFrame())):
    lst=result['list']
    show=['symbol','board','composite_score','pred_ret_1d','pred_ret_3d','pred_ret_5d','prob_up']
    avail=[c for c in show if c in lst.columns]
    top=lst.nlargest(min(25,len(lst)),'composite_score')
    print(f'\n=== TOP PICKS ({len(lst)} total) ===')
    print(top[avail].to_string())
    if 'board' in lst.columns:
        print(f'\nBoard:\n{lst["board"].value_counts().to_string()}')
    if 'prob_up' in lst.columns:
        print(f'\nProb_up:\n{lst["prob_up"].describe().to_string()}')
    lst.to_parquet('data/lists/_list_20260729_v2.parquet',index=False)
    print('Saved!')
else:
    print('EMPTY')
print(f'Total: {time.time()-t_total:.0f}s')
