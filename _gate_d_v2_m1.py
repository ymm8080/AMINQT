import pandas as pd, numpy as np, json, tempfile, shutil, time, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
import lightgbm as lgb
from collections import defaultdict

print('Gate D v2 M1: MAIN Feature Importance Ranking')
t0 = time.time()

# ── 1. Load CSI300 constituents ──
with open('D:/AMINQT/AMINQT CODES/data/csi300_constituents.json') as f:
    csi300 = set(json.load(f))
print(f'CSI300 constituents: {len(csi300)}')

# ── 2. Load panel data, filter last 12 months, intersect CSI300 ──
panel = pd.read_parquet('D:/AMINQT/AMINQT CODES/data/panel_full_enriched_v4_20260729.parquet')
print(f'Raw panel: {len(panel):,} rows, {panel["symbol"].nunique()} stocks')
cutoff = panel['date'].max() - pd.Timedelta(days=365)
panel = panel[panel['date'] >= cutoff]
in_panel = csi300 & set(panel['symbol'].unique())
panel = panel[panel['symbol'].isin(in_panel)].copy()
print(f'Filtered (CSI300, 12mo): {len(panel):,} rows, {panel["symbol"].nunique()} stocks')
print(f'Date range: {panel["date"].min()} ~ {panel["date"].max()}')

# ── 3. Clean ──
from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner = CleaningPipeline()
main_df, _ = cleaner.run_train(panel)
print(f'After clean: {len(main_df):,} rows, {main_df["symbol"].nunique()} stocks')

# ── 4. Seed registry + feature engineering ──
reg_dir = tempfile.mkdtemp()
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame, select_features
from app.pipeline1.ic_screener import ICScreener

registry = FeatureRegistry(path=os.path.join(reg_dir, 'feature_registry.json'))
sample = main_df.groupby('symbol', group_keys=False).apply(lambda g: g.head(min(30, len(g)))).reset_index(drop=True)
registry._seed(sample)

fe = FeatureEngineV35()
screener = ICScreener(registry_path=reg_dir)
df = prepare_board_frame(main_df, fe, cross_sectional_rank=False, registry=registry)
feat_cols = select_features(df, 'main', 'imp', screener, registry=registry)
print(f'Features selected: {len(feat_cols)}')

# ── 5. Train/val split ──
label = 'label_pm_1d_net'
if label not in df.columns:
    label = 'label_1d_net'
dates = sorted(df['date'].unique())
split = int(len(dates) * 0.75)
train_df = df[df['date'].isin(dates[:split])].dropna(subset=[label])
test_df  = df[df['date'].isin(dates[split:])].dropna(subset=[label])
print(f'Train: {len(train_df):,} rows ({dates[0]} ~ {dates[split-1]})')
print(f'Test:  {len(test_df):,} rows ({dates[split]} ~ {dates[-1]})')

X_train = train_df[feat_cols].fillna(0)
y_train = train_df[label]

# ── 6. Train LightGBM ──
model = lgb.LGBMRegressor(
    n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1
)
model.fit(X_train, y_train)
print(f'Model trained (took {time.time()-t0:.0f}s)')

# ── 7. Gain importance ──
imp = pd.DataFrame({'feature': feat_cols, 'gain': model.booster_.feature_importance(importance_type='gain')})
imp = imp.sort_values('gain', ascending=False).reset_index(drop=True)
imp['gain_pct'] = imp['gain'] / imp['gain'].sum()
imp['cum_gain'] = imp['gain_pct'].cumsum()

# Attach dim_group via registry
def get_dim(f):
    meta = registry.get_meta(f)
    return meta.get('dim_group', 'unknown') if meta else 'unknown'
imp['dim'] = imp['feature'].apply(get_dim)

# ── 8. Report ──
print('\n' + '='*100)
print('TOP 30 FEATURES BY GAIN IMPORTANCE')
print('='*100)
print(f'{"#":>3}  {"Feature":<45} {"Dim Group":<30} {"Gain%":>7}  {"Cum%":>7}')
print('-'*100)
for i, (_, r) in enumerate(imp.head(30).iterrows()):
    print(f'{i+1:>3}. {r["feature"]:<45} {r["dim"]:<30} {r["gain_pct"]:>6.2%}  {r["cum_gain"]:>6.2%}')

print('\n' + '='*60)
print('CUMULATIVE GAIN THRESHOLDS')
print('='*60)
for thresh in [0.50, 0.80, 0.90, 0.95]:
    n = (imp['cum_gain'] < thresh).sum() + 1
    pct_covered = imp.head(n)['gain_pct'].sum()
    print(f'  {thresh:.0%} cum gain: {n:>4} features  (actual {pct_covered:.2%})')

# ── 9. Per-dim importance ──
dim_gain = defaultdict(float)
dim_count = defaultdict(int)
dim_list = []
for _, r in imp.iterrows():
    dim_gain[r['dim']] += r['gain_pct']
    dim_count[r['dim']] += 1
    dim_list.append(r['dim'])

print('\n' + '='*80)
print('PER-DIM-GROUP IMPORTANCE (top 15)')
print('='*80)
print(f'{"Dim Group":<35} {"Gain%":>8}  {"#Feat":>6}  {"Avg Gain%":>9}')
print('-'*80)
for dg, g in sorted(dim_gain.items(), key=lambda x: -x[1])[:15]:
    cnt = dim_count[dg]
    print(f'{dg:<35} {g:>7.2%}  {cnt:>6}  {g/cnt:>8.2%}')

# Total features used
print(f'\nTotal features used: {len(imp)}')
print(f'Total dimensions represented: {len(dim_gain)}')
print(f'\nScript completed in {time.time()-t0:.0f}s')

# ── 10. Cleanup ──
shutil.rmtree(reg_dir)
