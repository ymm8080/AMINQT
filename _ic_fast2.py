# -*- coding: utf-8 -*-
"""Lean IC eval: sampled dates, no Top-N print, minimal memory."""
import time,numpy as np,pandas as pd
from scipy.stats import spearmanr

OHLCV=['open_hfq','high_hfq','low_hfq','close_hfq','open','high','low','close','volume','amount','turnover_rate']
CYQ=['chip_concentration','conc_90','winner_ratio','cost_5pct','cost_15pct','cost_50pct','cost_85pct','cost_95pct','weight_avg','winner_rate','his_low','his_high']
W=(1,3,5,10,20); L=['label_1d','label_3d','label_5d']

t0=time.time()
df=pd.read_parquet('data/panel_full_enriched_v3.parquet')
df['date']=pd.to_datetime(df['date']); df=df.sort_values(['symbol','date'])

# Sample dates (1 in 5) to speed up IC + reduce memory
dates=sorted(df['date'].unique())
sample_dates=dates[::5]
df=df[df['date'].isin(sample_dates)].copy()
print(f'Sampled {len(sample_dates)}/{len(dates)} dates, {len(df):,} rows ({time.time()-t0:.1f}s)',flush=True)

# Labels
from app.pipeline1.label_engine import LabelEngine
df=LabelEngine.build_labels(df,session='PM')
df=LabelEngine.mask_recent_days(df,days=6)  # 屏蔽近端未成熟标签, 防 IC 泄漏
print(f'Labels done ({time.time()-t0:.1f}s)',flush=True)

ohlcv=[c for c in OHLCV if c in df.columns]
cyq=[c for c in CYQ if c in df.columns]

# Compute _chg on sampled data
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

# Drop raw cols to save memory (only keep derived + labels)
keep=set(added_o)|set(added_c)|set(ohlcv)|set(cyq)|{'date','symbol'}|set(L)
df=df[list(keep)].copy()

# IC eval
def ic_bulk(df,cols,label):
    """Bulk IC: one pass per column, returns list of mean_abs_ic."""
    results=[]
    sub=df[['date',label]].dropna()
    for col in cols:
        v=df[col]
        valid=sub.index.intersection(v.dropna().index)
        if len(valid)<100: continue
        d=pd.DataFrame({'date':df.loc[valid,'date'],'x':v.loc[valid],'y':df.loc[valid,label]})
        ics=[]
        for _,g in d.groupby('date'):
            if len(g)<30: continue
            try:
                s=spearmanr(g['x'],g['y']).statistic
                if not np.isnan(s): ics.append(s)
            except: pass
        if len(ics)>=10:
            a=np.array(ics)
            results.append(abs(a).mean())
    return results

print(f'\n{"="*85}')
print(f'IC COMPARISON  (sampled 1/{5} dates, {len(sample_dates)} dates)')
print(f'{"="*85}')
raw_ics_all={}; o_ics_all={}; c_ics_all={}
for label in L:
    t1=time.time()
    raw=ic_bulk(df,ohlcv+cyq,label)
    o_chg=ic_bulk(df,added_o,label)
    c_chg=ic_bulk(df,added_c,label)
    print(f'\n--- {label} ({time.time()-t1:.1f}s) ---')
    for name,vals in [('RAW levels',raw),('OHLCV _chg/pct',o_chg),('CYQ _chg/pct',c_chg)]:
        if vals:
            a=np.array(vals); pos=np.mean(np.array(vals)>0.001)
            print(f'  {name:<20} N={len(vals):>3}  mean|IC|={a.mean():.4f}  max={a.max():.4f}  pos>{0.001:.3f}={pos:.0%}')
        else: print(f'  {name:<20} (no valid IC)')

print(f'\n{"="*85}')
print('DECISION: Should _chgN/_pct_chgN be added to OHLCV and CYQ?')
print(f'{"="*85}')
for lbl in L:
    r=np.mean(raw_ics_all.get(lbl,[])) if raw_ics_all.get(lbl) else 0
    o=np.mean(o_ics_all.get(lbl,[])) if o_ics_all.get(lbl) else 0
    c=np.mean(c_ics_all.get(lbl,[])) if c_ics_all.get(lbl) else 0
    best=max(r,o,c)
    winner='RAW' if r==best else ('OHLCV-chg' if o==best else 'CYQ-chg')
    print(f'  {lbl}: RAW={r:.4f}  OHLCV-chg={o:.4f}  CYQ-chg={c:.4f}  →  BEST={winner}')

print(f'\nTotal: {time.time()-t0:.0f}s')
