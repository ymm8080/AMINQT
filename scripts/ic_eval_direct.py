# -*- coding: utf-8 -*-
"""Direct IC eval — dim21-28 daily + time-series features. No API calls."""
import sys, os, time, logging, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==== Load panel + merge cached alt data ====
df = pd.read_parquet('data/panel_full_enriched.parquet')
df = df.sort_values(['symbol','date']).reset_index(drop=True)
logger.info('Panel: %d stocks, %d rows', df['symbol'].nunique(), len(df))

# Industry + board
from app.pipeline1.data_supply import DataSupplyChain
from app.pipeline1.cleaning_pipeline import board_of, get_limit_pct
supply = DataSupplyChain()
pro = supply._tushare_pro()
basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
basic['symbol'] = basic['ts_code'].str.replace('.SZ','').str.replace('.SH','')
ind_map = dict(zip(basic['symbol'], basic['industry'].fillna('综合')))
df['industry'] = df['symbol'].map(ind_map).fillna('综合')
df['board'] = df['symbol'].map(board_of)

# Merge cached margin
mg_cache = 'data/supply_cache/alt_data/margin_panel.parquet'
if os.path.exists(mg_cache):
    mg = pd.read_parquet(mg_cache)
    df = df.merge(mg, on=['symbol','date'], how='left')
    logger.info('Margin merged: cov=%.1f%%', (1-df['margin_balance'].isna().mean())*100)

# Merge cached holdernumber
hn_cache = 'data/supply_cache/alt_data/holdernumber/all_20240102_20260727.parquet'
if os.path.exists(hn_cache):
    hn = pd.read_parquet(hn_cache).dropna(subset=['holder_count']).sort_values('announce_date')
    df = df.sort_values('date')
    df = pd.merge_asof(df, hn[['symbol','announce_date','holder_count']],
                        left_on='date', right_on='announce_date', by='symbol', direction='backward')
    logger.info('Holdernumber merged: cov=%.1f%%', (1-df['holder_count'].isna().mean())*100)

# ==== Build features ====
df['is_st'] = False; df['is_suspended'] = False
df['list_days'] = df.groupby('symbol').cumcount() + 1
df['limit_pct'] = [get_limit_pct(b, d) for b, d in zip(df['board'], df['date'])]

from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine
t0 = time.time()
fe = FeatureEngineV35()
df = fe.build(df)  # includes dim21-28 + _add_time_series_changes(_chg1/3/5/10/20)
df = LabelEngine.build_path_labels(df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=6)
logger.info('Features: %d cols in %.1fs', len(df.columns), time.time()-t0)

# ==== IC eval ====
label_1d = 'label_1d_net' if 'label_1d_net' in df.columns else 'label_1d'
label_3d = 'label_3d_net' if 'label_3d_net' in df.columns else 'label_3d'
label_5d = 'label_5d_net' if 'label_5d_net' in df.columns else 'label_5d'

def rank_ic(df, factor, label):
    sub = df[[factor,'date',label]].dropna()
    if len(sub) < 500: return 0.0
    ics = [g[[factor,label]].corr(method='spearman').iloc[0,1]
           for _,g in sub.groupby('date') if len(g)>=10]
    ics = [i for i in ics if not np.isnan(i)]
    return float(np.mean(np.abs(ics))) if ics else 0.0

# dim21-28 feature prefixes
dims = {
    'dim21_cyq': ['conc_90','benefit_part','cost_bias','cost_spread','chip_skew','conc_trend_20d','benefit_trend_5d'],
    'dim22_finPIT': ['roe_qoq','roa_qoq','margin_chg','growth_accel','profit_accel','debt_leveraging','efficiency_chg','ocf_stability','roe_trend_4q','margin_trend_4q','quality_momentum'],
    'dim23_shareholder': ['holder_count_qoq','holder_count_yoy','holder_qoq_accel','avg_shares_qoq','avg_shares_yoy','holder_concentration_zscore'],
    'dim24_margin': ['margin_balance_chg_1d','margin_balance_chg_5d','short_balance_ratio','margin_buy_ratio','margin_balance_ma20_dev','margin_pressure_score'],
    'dim25_northbound': ['north_net_buy_5d','north_net_buy_20d','north_net_buy_streak','north_buy_ratio','north_momentum_5d'],
    'dim26_lhb': ['lhb_inst_net_buy_5d','lhb_inst_net_buy_20d','lhb_inst_count_5d','lhb_abnormal_score'],
    'dim27_indflow': ['ind_margin_chg_5d','ind_margin_accel','ind_holder_trend_20d','ind_north_chg_5d','ind_capital_flow'],
    'dim28_sector': ['sw_ret_5d','sw_ret_20d','sw_vol_20d','sw_relative_strength','sw_rotation_position','sw_momentum_accel','sw_turnover_anomaly'],
}

# Also check _chg1/_chg3/_chg5 auto-generated for top existing factors
chg_check = ['MACD','RSI','bias_60','ATR_pct','BB_width','chip_concentration','conc_90','benefit_part','amihud_illiquidity']
dims['_time_series_chg'] = [f'{c}_chg{w}' for c in chg_check for w in [1,3,5]]

print()
print('=' * 100)
print(f"{'Dim':<20s} {'Factor':<35s} {'IC_1d':<8s} {'IC_3d':<8s} {'IC_5d':<8s} {'NaN%':<7s} {'Signal'}")
print('-' * 100)

results = {}
for dim, feats in dims.items():
    best_ic1, best_ic3, best_ic5 = 0, 0, 0
    best_f, best_nan = '', 100
    shown = 0
    for f in feats:
        if f not in df.columns: continue
        nan_r = df[f].isna().mean()
        if nan_r > 0.95: continue
        ic1 = rank_ic(df, f, label_1d)
        ic3 = rank_ic(df, f, label_3d)
        ic5 = rank_ic(df, f, label_5d)
        best = max(ic1, ic3, ic5)
        if best > 0.005:
            sig = 'STRONG' if best>=0.03 else ('OK' if best>=0.02 else ('weak' if best>=0.01 else '-'))
            print(f'{dim:<20s} {f:<35s} {ic1:<8.4f} {ic3:<8.4f} {ic5:<8.4f} {nan_r*100:<7.1f} {sig}')
            shown += 1
        if best > best_ic1:
            best_ic1, best_ic3, best_ic5 = ic1, ic3, ic5
            best_f, best_nan = f, nan_r
    verdict = 'INCLUDE' if best_ic1>=0.02 else ('WATCH' if best_ic1>=0.01 else 'SKIP')
    results[dim] = {'best': best_f, 'ic1': round(best_ic1,5), 'ic3': round(best_ic3,5), 'ic5': round(best_ic5,5), 'nan': round(best_nan*100,1), 'shown': shown, 'verdict': verdict}

print('-' * 100)
print()
print('=== VERDICT ===')
for dim, r in results.items():
    status = '>> INCLUDE <<' if r['verdict']=='INCLUDE' else ''
    print(f"  {dim:<22s}: {r['best']:<35s} IC1={r['ic1']:.5f} IC3={r['ic3']:.5f} IC5={r['ic5']:.5f} NaN={r['nan']:.1f}% shown={r['shown']} -> {r['verdict']} {status}")

os.makedirs('data/factor_registry', exist_ok=True)
out_path = f'data/factor_registry/ic_direct_{pd.Timestamp.now():%Y%m%d_%H%M}.json'
json.dump({'timestamp': pd.Timestamp.now().isoformat(), 'results': results}, open(out_path, 'w'), ensure_ascii=False, indent=2)
logger.info('Saved: %s', out_path)
