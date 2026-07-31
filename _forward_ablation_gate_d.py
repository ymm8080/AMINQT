import pandas as pd, numpy as np, json, tempfile, shutil, time, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
import lightgbm as lgb
from scipy.stats import spearmanr

print('MAIN board CSI300 Forward Ablation (12mo)')
t0 = time.time()
with open('data/csi300_constituents.json') as f:
    csi300 = set(json.load(f))

panel = pd.read_parquet('data/panel_full_enriched_v4_20260729.parquet')
cutoff = panel['date'].max() - pd.Timedelta(days=365)
panel = panel[panel['date'] >= cutoff]
in_panel = csi300 & set(panel['symbol'].unique())
panel = panel[panel['symbol'].isin(in_panel)].copy()
print(f'Panel: {len(panel):,} rows, {panel.symbol.nunique()} stocks ({time.time()-t0:.1f}s)')

t0 = time.time()
from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner = CleaningPipeline()
main_df, _ = cleaner.run_train(panel)
print(f'Main: {len(main_df):,} rows, {main_df.symbol.nunique()} stocks ({time.time()-t0:.1f}s)')

t0 = time.time()
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
feat_cols = select_features(df, 'main', 'ablation', screener, registry=registry)
print(f'Built: {len(df.columns)} cols, {len(feat_cols)} features ({time.time()-t0:.1f}s)')

label = 'label_pm_1d_net'
if label not in df.columns: label = 'label_1d_net'

dates = sorted(df['date'].unique())
split = int(len(dates) * 0.75)
train_df = df[df['date'].isin(dates[:split])].dropna(subset=[label])
test_df = df[df['date'].isin(dates[split:])].dropna(subset=[label])
print(f'Train: {len(train_df):,} rows, {len(dates[:split])}d | Test: {len(test_df):,} rows, {len(dates[split:])}d')

X_train = train_df[feat_cols].fillna(0)
X_test = test_df[feat_cols].fillna(0)
y_train = train_df[label]
y_test = test_df[label]

print(f'Training...')
model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1)
model.fit(X_train, y_train)

imp = pd.DataFrame({'feature': feat_cols, 'gain': model.booster_.feature_importance(importance_type='gain')})
imp = imp.sort_values('gain', ascending=False)
imp['gain_pct'] = imp['gain'] / imp['gain'].sum()
imp['cum_gain'] = imp['gain_pct'].cumsum()

print(f'\nTop 10:')
for _, r in imp.head(10).iterrows():
    print(f'  {r["feature"]:<45} {r["gain_pct"]:.2%}')
for thresh in [0.50, 0.80, 0.90, 0.95]:
    n = (imp['cum_gain'] < thresh).sum() + 1
    print(f'  {thresh:.0%} gain: {n} features')

def oos_ic(preds, df_t, lab):
    df_e = df_t.copy(); df_e['pred'] = preds
    ics = []
    for d, g in df_e.groupby('date'):
        if len(g) < 10: continue
        ic, _ = spearmanr(g['pred'], g[lab])
        if not np.isnan(ic): ics.append(ic)
    a = np.array(ics)
    return round(a.mean(),5), round(a.mean()/a.std() if a.std()>0 else 0,4)

preds_all = model.predict(X_test)
ic_a, ir_a = oos_ic(preds_all, test_df, label)
print(f'\nAll {len(feat_cols)} feats: OOS IC={ic_a:.5f} ICIR={ir_a:.4f}')

print(f'Ablation:')
results = []
for n in [10, 20, 30, 50, 75, 100, 150, 200, len(feat_cols)]:
    top = imp.head(min(n, len(imp)))['feature'].tolist()
    m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X_train[top], y_train)
    ic, ir = oos_ic(m.predict(X_test[top]), test_df, label)
    results.append({'n': n, 'ic': ic, 'icir': ir})
    print(f'  n={n:>3}: IC={ic:+.5f} ICIR={ir:+.4f}')

df_r = pd.DataFrame(results)
best = df_r.loc[df_r['icir'].idxmax()]
sat = df_r[df_r['icir'] >= best['icir'] * 0.95].iloc[0]
print(f'\nMAIN: Best n={best["n"]:.0f} ICIR={best["icir"]:.4f} | Saturation n={sat["n"]:.0f} ({sat["n"]/len(feat_cols):.0%})')
shutil.rmtree(reg_dir)
