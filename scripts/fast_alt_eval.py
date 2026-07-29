# -*- coding: utf-8 -*-
"""FAST alt data eval — skip per-stock APIs, use bulk + cached."""
import sys, os, time, logging, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==== 1. Load panel ====
df = pd.read_parquet('data/panel_full_enriched.parquet')
rng = np.random.RandomState(42)
syms = rng.choice(df['symbol'].unique(), 500, replace=False)
df = df[df['symbol'].isin(syms)].sort_values(['symbol','date']).reset_index(drop=True)
logger.info('Panel: %d stocks, %d rows', df['symbol'].nunique(), len(df))

from app.pipeline1.data_supply import DataSupplyChain
from app.pipeline1.cleaning_pipeline import board_of, get_limit_pct
supply = DataSupplyChain()
pro = supply._tushare_pro()

# Industry from Tushare
basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
basic['symbol'] = basic['ts_code'].str.replace('.SZ','').str.replace('.SH','')
ind_map = dict(zip(basic['symbol'], basic['industry'].fillna('综合')))
df['industry'] = df['symbol'].map(ind_map).fillna('综合')
df['board'] = df['symbol'].map(board_of)
logger.info('Industry: %d unique', df['industry'].nunique())

# ==== 2. FAST data fetching (bulk only, no per-stock) ====

# 2a. fina_indicator: use daily_basic (bulk PE/PB) + derive what we can
# Already in panel from original build — skip. dim22 will use existing PE_TTM if available.

# 2b. holdernumber from CACHE (already fetched 5500 rows)
hn_path = 'data/supply_cache/alt_data/holdernumber/all_20240102_20260727.parquet'
if os.path.exists(hn_path):
    hn = pd.read_parquet(hn_path)
    hn = hn.dropna(subset=['holder_count'])  # only rows with actual data
    hn = hn.sort_values('announce_date')
    df = df.sort_values('date')
    df = pd.merge_asof(df, hn[['symbol','announce_date','holder_count']],
                        left_on='date', right_on='announce_date', by='symbol', direction='backward')
    logger.info('holdernumber: %d rows from cache, cov=%.1f%%', len(hn), (1-df['holder_count'].isna().mean())*100)

# 2c. margin: bulk per-date, sample 12 dates
t0 = time.time()
mg_frames = []
sample_dates = sorted(df['date'].dropna().unique())[::50]  # every 50th trading day
for d in sample_dates:
    dt = d.strftime('%Y%m%d')
    try:
        raw = pro.margin_detail(trade_date=dt)
        if raw is not None and len(raw) > 0:
            raw['symbol'] = raw['ts_code'].str.replace('.SZ','').str.replace('.SH','')
            raw['date'] = pd.to_datetime(d)
            for c in ['rzye','rqye','rzmre','rqmcl']:
                if c in raw.columns: raw[c] = pd.to_numeric(raw[c], errors='coerce')
            mg_frames.append(raw[['symbol','date','rzye','rqye','rzmre','rqmcl']])
    except: pass
    time.sleep(0.2)
if mg_frames:
    mg = pd.concat(mg_frames, ignore_index=True).rename(
        columns={'rzye':'margin_balance','rqye':'short_balance','rzmre':'margin_buy_amt','rqmcl':'short_sell_vol'})
    df = df.merge(mg, on=['symbol','date'], how='left')
logger.info('margin: %d days, cov=%.1f%%, %.1fs', len(mg_frames),
            (1-df['margin_balance'].isna().mean())*100 if 'margin_balance' in df.columns else 0, time.time()-t0)

# 2d. moneyflow: bulk per-date
t0 = time.time()
mf_frames = []
sample_dates = sorted(df['date'].dropna().unique())[::50]  # every 50th trading day
for d in sample_dates:
    dt = d.strftime('%Y%m%d')
    try:
        raw = pro.moneyflow(trade_date=dt)
        if raw is not None and len(raw) > 0:
            raw['symbol'] = raw['ts_code'].str.replace('.SZ','').str.replace('.SH','')
            raw['date'] = pd.to_datetime(d)
            if 'net_mf_amount' in raw.columns: raw['net_mf_amount'] = pd.to_numeric(raw['net_mf_amount'], errors='coerce')
            mf_frames.append(raw[['symbol','date','net_mf_amount']])
    except: pass
    time.sleep(0.2)
if mf_frames:
    mf = pd.concat(mf_frames, ignore_index=True)
    mf = mf.rename(columns={'net_mf_amount':'main_money_flow'})
    df = df.merge(mf, on=['symbol','date'], how='left')
    # Don't overwrite existing main_money_flow from akshare if already present
logger.info('moneyflow: %d days, cov=%.1f%%, %.1fs', len(mf_frames),
            (1-df['main_money_flow'].isna().mean())*100 if 'main_money_flow' in df.columns else 0, time.time()-t0)

# Coverage
print('\n=== Data Coverage ===')
for c in ['roe','holder_count','margin_balance','main_money_flow']:
    pct = (1-df[c].isna().mean())*100 if c in df.columns else 0
    print(f'  {c:<25s}: {pct:.1f}%')

# ==== 3. Build features ====
df['is_st'] = False; df['is_suspended'] = False
df['list_days'] = df.groupby('symbol').cumcount() + 1
df['limit_pct'] = [get_limit_pct(b, d) for b, d in zip(df['board'], df['date'])]

from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine
t0 = time.time()
fe = FeatureEngineV35()
df = fe.build(df)
df = LabelEngine.build_path_labels(df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=6)
logger.info('Features: %d cols in %.1fs', len(df.columns), time.time()-t0)

# ==== 4. IC eval ====
label_1d = 'label_1d_net' if 'label_1d_net' in df.columns else 'label_1d'
label_3d = 'label_3d_net' if 'label_3d_net' in df.columns else 'label_3d'

def rank_ic(df, factor, label):
    sub = df[[factor,'date',label]].dropna()
    if len(sub) < 500: return 0.0
    ics = [g[[factor,label]].corr(method='spearman').iloc[0,1]
           for _,g in sub.groupby('date') if len(g)>=10]
    ics = [i for i in ics if not np.isnan(i)]
    return float(np.mean(np.abs(ics))) if ics else 0.0

dims_features = {
    'dim28_sector': ['sw_ret_1d','sw_ret_5d','sw_ret_20d','sw_vol_20d','sw_relative_strength','sw_rotation_position','sw_momentum_accel','sw_turnover_anomaly'],
    'dim27_indflow': ['ind_flow_composite'],
    'dim23_shareholder': ['holder_count_log','holder_count_qoq','holder_count_yoy','holder_qoq_accel','avg_shares_log','avg_shares_qoq','avg_shares_yoy','holder_concentration_zscore'],
    'dim24_margin': ['margin_balance_chg_1d','margin_balance_chg_5d','short_balance_ratio','margin_buy_ratio','margin_balance_ma20_dev','margin_balance_yoy','margin_pressure_score'],
    'dim22_finPIT': ['roe_zscore','roa_zscore','margin_composite','growth_composite','quality_score','efficiency_score','roe_stability','margin_trend'],
    'dim25_northbound': ['north_net_buy_5d','north_net_buy_20d','north_net_buy_streak','north_buy_ratio','north_sh_sz_divergence','north_momentum_5d','north_flow_zscore'],
    'dim26_lhb': ['lhb_inst_net_buy_5d','lhb_inst_net_buy_20d','lhb_inst_count_5d','lhb_inst_buy_ratio','lhb_abnormal_score'],
}

print()
print('=' * 100)
print(f"{'Dim':<20s} {'Factor':<30s} {'IC_1d':<8s} {'IC_3d':<8s} {'NaN%':<7s} {'Signal'}")
print('-' * 100)

results = {}
for dim, feats in dims_features.items():
    best_ic, best_f, best_nan = 0, '', 100
    for f in feats:
        if f not in df.columns: continue
        nan_r = df[f].isna().mean()
        if nan_r > 0.95: continue
        ic1 = rank_ic(df, f, label_1d)
        ic3 = rank_ic(df, f, label_3d)
        best = max(ic1, ic3)
        if best > 0.001:
            sig = 'STRONG' if best>=0.03 else ('OK' if best>=0.02 else ('weak' if best>=0.01 else '-'))
            print(f'{dim:<20s} {f:<30s} {ic1:<8.4f} {ic3:<8.4f} {nan_r*100:<7.1f} {sig}')
        if best > best_ic:
            best_ic, best_f, best_nan = best, f, nan_r
    verdict = 'INCLUDE' if best_ic>=0.02 else ('WATCH' if best_ic>=0.01 else 'SKIP')
    results[dim] = {'best_factor': best_f, 'best_ic': round(best_ic,5), 'nan_pct': round(best_nan*100,1), 'verdict': verdict}

print('-' * 100)
print()
print('=== VERDICT ===')
for dim, r in results.items():
    print(f"  {dim:<20s}: {r['best_factor']:<30s} IC={r['best_ic']:.5f}  NaN={r['nan_pct']:.1f}%  --> {r['verdict']}")

os.makedirs('data/factor_registry', exist_ok=True)
out = {'timestamp': pd.Timestamp.now().isoformat(), 'results': results}
with open(f'data/factor_registry/alt_eval_fast_{pd.Timestamp.now():%Y%m%d_%H%M}.json', 'w') as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)
