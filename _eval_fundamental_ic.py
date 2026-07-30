import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# Load needed cols only
COLS = ['symbol','date','close','industry'] + \
    ['roe','roa','gross_margin','net_margin','asset_turnover',
     'debt_ratio','eps_yoy','rev_yoy','profit_yoy']

import pyarrow.parquet as pq
schema = pq.read_schema('data/panel_full_enriched_v4_20260729.parquet')
avail = [c for c in COLS if c in schema.names]
panel = pd.read_parquet('data/panel_full_enriched_v4_20260729.parquet', columns=avail)

# Last 18 months, sample 200 stocks
cutoff = panel['date'].max() - pd.Timedelta(days=540)
panel = panel[panel['date'] >= cutoff]
stocks = np.random.choice(panel['symbol'].unique(), size=min(200, panel['symbol'].nunique()), replace=False)
panel = panel[panel['symbol'].isin(stocks)].sort_values(['symbol','date'])

# Forward return
panel['fwd_1d'] = panel.groupby('symbol')['close'].transform(lambda x: x.shift(-1)/x - 1)
panel = panel.dropna(subset=['fwd_1d'])

# Identify REPORT DATES: dates where fundamental values CHANGE for a stock
fina_cols = [c for c in ['roe','roa','gross_margin','net_margin','asset_turnover','debt_ratio'] if c in panel.columns]
for c in fina_cols:
    panel[f'{c}_changed'] = panel.groupby('symbol')[c].transform(lambda x: x.diff().abs() > 1e-8)
# A date is a "report date" for a stock if ANY fundamental column changed
panel['is_report'] = False
for c in fina_cols:
    panel['is_report'] = panel['is_report'] | panel[f'{c}_changed']

report_dates = set(panel[panel['is_report']]['date'].unique())
print(f'Total dates: {panel.date.nunique()}, Report dates: {len(report_dates)}')
print(f'Report date fraction: {len(report_dates)/panel.date.nunique():.1%}')

def compute_ic(df, col, label, date_filter=None, industry=None):
    """Compute daily Rank IC. Optionally filter dates or restrict to one industry."""
    sub = df[['symbol','date',col,label,'industry']].dropna()
    if len(sub) < 100: return 0,0,0
    if date_filter is not None: sub = sub[sub['date'].isin(date_filter)]
    if industry: sub = sub[sub['industry'] == industry]
    ics = []
    for d, g in sub.groupby('date'):
        if len(g) < 10: continue
        ic,_ = spearmanr(g[col], g[label])
        if not np.isnan(ic): ics.append(ic)
    a = np.array(ics)
    if len(a) < 10: return 0,0,0
    return round(a.mean(),5), round(a.mean()/a.std() if a.std()>0 else 0,4), len(a)

def sector_ic(df, col, label):
    """Average IC computed within each industry separately, then averaged."""
    if 'industry' not in df.columns: return 0,0,0
    sub = df[['symbol','date',col,label,'industry']].dropna()
    if len(sub) < 100: return 0,0,0
    all_ics = []
    for ind in sub['industry'].dropna().unique()[:10]:  # top 10 industries
        ind_df = sub[sub['industry'] == ind]
        if ind_df['symbol'].nunique() < 5: continue
        for d, g in ind_df.groupby('date'):
            if len(g) < 5: continue
            ic,_ = spearmanr(g[col], g[label])
            if not np.isnan(ic): all_ics.append(ic)
    a = np.array(all_ics)
    if len(a) < 20: return 0,0,0
    return round(a.mean(),5), round(a.mean()/a.std() if a.std()>0 else 0,4), len(a)

# Evaluate
print(f'\n{"="*110}')
print(f'CROSS-SECTION EVALUATION: Raw Fundamentals')
print(f'{"Col":<20} {"All-dates IC":>12} {"ICIR":>8} | {"Report-only IC":>12} {"ICIR":>8} | {"Sector-relative IC":>14} {"ICIR":>8}')
print('-'*110)

for col in fina_cols:
    # All dates
    a_ic, a_ir, a_n = compute_ic(panel, col, 'fwd_1d')
    # Report dates only
    r_ic, r_ir, r_n = compute_ic(panel, col, 'fwd_1d', date_filter=report_dates)
    # Sector-relative
    s_ic, s_ir, s_n = sector_ic(panel, col, 'fwd_1d')

    a_str = f'{a_ic:.5f} ({a_ir:.3f})' if a_n>0 else 'N/A'
    r_str = f'{r_ic:.5f} ({r_ir:.3f})' if r_n>0 else 'N/A'
    s_str = f'{s_ic:.5f} ({s_ir:.3f})' if s_n>0 else 'N/A'

    # Flag if report-only or sector ICIR improves >20%
    flag = ''
    if r_n>0 and a_n>0 and r_ir > a_ir * 1.2:
        flag += ' [REPORT SPIKE]'
    if s_n>0 and a_n>0 and s_ir > a_ir * 1.2:
        flag += ' [SECTOR BOOST]'

    print(f'{col:<20} {a_ic:>8.5f} ({a_ir:>6.3f}) | {r_ic:>8.5f} ({r_ir:>6.3f}) | {s_ic:>10.5f} ({s_ir:>6.3f}){flag}')

# Also evaluate growth cols
print(f'\n{"="*110}')
print(f'GROWTH COLUMNS (eps_yoy, rev_yoy, profit_yoy)')
print(f'{"Col":<20} {"All-dates IC":>12} {"ICIR":>8} | {"Report-only IC":>12} {"ICIR":>8} | {"Sector-relative IC":>14} {"ICIR":>8}')
print('-'*110)

growth_cols = [c for c in ['eps_yoy','rev_yoy','profit_yoy'] if c in panel.columns]
for col in growth_cols:
    a_ic, a_ir, a_n = compute_ic(panel, col, 'fwd_1d')
    r_ic, r_ir, r_n = compute_ic(panel, col, 'fwd_1d', date_filter=report_dates)
    s_ic, s_ir, s_n = sector_ic(panel, col, 'fwd_1d')

    flag = ''
    if r_n>0 and a_n>0 and r_ir > a_ir * 1.2:
        flag += ' [REPORT SPIKE]'
    if s_n>0 and a_n>0 and s_ir > a_ir * 1.2:
        flag += ' [SECTOR BOOST]'

    print(f'{col:<20} {a_ic:>8.5f} ({a_ir:>6.3f}) | {r_ic:>8.5f} ({r_ir:>6.3f}) | {s_ic:>10.5f} ({s_ir:>6.3f}){flag}')

print(f'\nCONCLUSION: If report-only or sector-relative ICIR > all-dates ICIR,')
print(f'the cross-sectional information is REAL and raw columns should be kept.')
