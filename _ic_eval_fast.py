# -*- coding: utf-8 -*-
"""Fast IC eval: does _chgN on OHLCV/CYQ improve IC? Re-uses label+panel path."""
import sys,os,warnings; warnings.filterwarnings('ignore')
import time, numpy as np, pandas as pd
from scipy.stats import spearmanr
t0=time.time()

OHLCV_COLS=['open_hfq','high_hfq','low_hfq','close_hfq','open','high','low','close','volume','amount','turnover_rate']
CYQ_COLS=['chip_concentration','conc_90','winner_ratio','cost_5pct','cost_15pct','cost_50pct','cost_85pct','cost_95pct','weight_avg','winner_rate','his_low','his_high']
W=(1,3,5,10,20)
L=['label_1d','label_3d','label_5d']

def ic(df,col,label):
    sub=df[['date',col,label]].dropna()
    if len(sub)<100: return None
    ics=[]
    for _,g in sub.groupby('date'):
        if len(g)<30: continue
        try:
            v=spearmanr(g[col],g[label]).statistic
            if not np.isnan(v): ics.append(v)
        except: pass
    if len(ics)<10: return None
    a=np.array(ics)
    return {'mean_abs':float(abs(a).mean()),'mean':float(a.mean()),'n':len(a),'std':float(a.std())}

print('Load...',flush=True)
df=pd.read_parquet('data/panel_full_enriched_v3.parquet')
df['date']=pd.to_datetime(df['date']); df=df.sort_values(['symbol','date'])

print('Labels...',flush=True)
from app.pipeline1.label_engine import LabelEngine
df=LabelEngine.build_labels(df,session='PM')
df=LabelEngine.mask_recent_days(df,days=6)  # 屏蔽近端未成熟标签, 防 IC 泄漏

ohlcv=[c for c in OHLCV_COLS if c in df.columns]
cyq=[c for c in CYQ_COLS if c in df.columns]
print(f'OHLCV avail: {len(ohlcv)}/{len(OHLCV_COLS)}  CYQ avail: {len(cyq)}/{len(CYQ_COLS)}',flush=True)

print('Computing _chg...',flush=True)
added_o=[]; added_c=[]
for col in ohlcv:
    grp=df.groupby('symbol')[col]
    for w in W:
        a=f'{col}_chg{w}'; p=f'{col}_pct_chg{w}'
        df[a]=grp.diff(w); df[p]=grp.pct_change(w,fill_method=None)
        if df[a].notna().sum()>100: added_o+=[a,p]
for col in cyq:
    grp=df.groupby('symbol')[col]
    for w in W:
        a=f'{col}_chg{w}'; p=f'{col}_pct_chg{w}'
        df[a]=grp.diff(w); df[p]=grp.pct_change(w,fill_method=None)
        if df[a].notna().sum()>100: added_c+=[a,p]
print(f'OHLCV-derived: {len(added_o)}  CYQ-derived: {len(added_c)} ({time.time()-t0:.1f}s)',flush=True)

print('\n'+'='*95)
print('IC COMPARISON: baseline (raw levels) vs OHLCV-derived vs CYQ-derived')
print('='*95)
for label in L:
    raw_ic=[ic(df,c,label)['mean_abs'] for c in ohlcv+cyq if ic(df,c,label)]
    o_ic=[ic(df,c,label)['mean_abs'] for c in added_o if ic(df,c,label)]
    c_ic=[ic(df,c,label)['mean_abs'] for c in added_c if ic(df,c,label)]
    print(f'\n--- {label} ---')
    for name,vals in [('RAW levels',raw_ic),('OHLCV _chg/pct',o_ic),('CYQ _chg/pct',c_ic)]:
        if vals:
            a=np.array(vals); pos=np.mean(a>0.001)
            print(f'  {name:<20} N={len(vals):>3}  mean|IC|={a.mean():.4f}  max={a.max():.4f}  pos>{0.001:.3f}={pos:.0%}')
        else: print(f'  {name:<20} (no valid IC)')

    # Top-5
    o_top=sorted([(c,ic(df,c,label)['mean_abs']) for c in added_o if ic(df,c,label)],key=lambda x:-x[1])
    c_top=sorted([(c,ic(df,c,label)['mean_abs']) for c in added_c if ic(df,c,label)],key=lambda x:-x[1])
    print(f'  Top OHLCV: {", ".join(f"{c}({v:.4f})" for c,v in o_top[:5])}')
    print(f'  Top CYQ:   {", ".join(f"{c}({v:.4f})" for c,v in c_top[:5])}')

# Per-window
print('\n'+'='*95)
print('PER-WINDOW (mean|IC| across all labels)')
print(f'  {"Window":>8} {"OHLCV mean|IC|":>15} {"OHLCV N":>8} {"CYQ mean|IC|":>15} {"CYQ N":>8}')
for w in W:
    ow=[c for c in added_o if f'_chg{w}' in c or f'_pct_chg{w}' in c]
    cw=[c for c in added_c if f'_chg{w}' in c or f'_pct_chg{w}' in c]
    o_ics=[]; c_ics=[]
    for c in ow:
        for lbl in L:
            r=ic(df,c,lbl)
            if r: o_ics.append(r['mean_abs'])
    for c in cw:
        for lbl in L:
            r=ic(df,c,lbl)
            if r: c_ics.append(r['mean_abs'])
    print(f'  chg{w:>3}      {np.mean(o_ics):>15.4f} {len(o_ics):>8}  {np.mean(c_ics):>15.4f} {len(c_ics):>8}')

print(f'\nTotal: {time.time()-t0:.0f}s')
