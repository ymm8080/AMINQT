import pandas as pd, numpy as np, json, tempfile, shutil, time, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
import lightgbm as lgb
from scipy.stats import spearmanr

print('Gate D v2 M2: DUAL Forward Ablation')

panel = pd.read_parquet('data/panel_full_enriched_v4_20260729.parquet')
cutoff = panel['date'].max() - pd.Timedelta(days=365)
panel = panel[panel['date'] >= cutoff]
dual = panel[panel['board'].isin(['GEM','STAR'])].copy()
if dual.symbol.nunique() > 300:
    stocks = np.random.choice(dual['symbol'].unique(), size=300, replace=False)
    dual = dual[dual['symbol'].isin(stocks)]
print(f'Dual: {len(dual):,} rows, {dual.symbol.nunique()} stocks')

from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner = CleaningPipeline()
main_df, dual_df = cleaner.run_train(dual)
board_df = dual_df if len(dual_df) > len(main_df) else main_df
bname = 'dual' if len(dual_df) > len(main_df) else 'main'
print(f'{bname}: {len(board_df):,} rows, {board_df.symbol.nunique()} stocks')

reg_dir = tempfile.mkdtemp()
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame, select_features
from app.pipeline1.ic_screener import ICScreener

registry = FeatureRegistry(path=os.path.join(reg_dir, 'feature_registry.json'))
sample = board_df.groupby('symbol', group_keys=False).apply(lambda g: g.head(min(30, len(g)))).reset_index(drop=True)
registry._seed(sample)

fe = FeatureEngineV35()
screener = ICScreener(registry_path=reg_dir)
df = prepare_board_frame(board_df, fe, cross_sectional_rank=True, registry=registry)
feat_cols = select_features(df, bname, 'ablation', screener, registry=registry)
print(f'Features: {len(feat_cols)}')

label = 'label_pm_1d_net'
if label not in df.columns: label = 'label_1d_net'
dates = sorted(df['date'].unique())
split = int(len(dates) * 0.75)
train_df = df[df['date'].isin(dates[:split])].dropna(subset=[label])
test_df = df[df['date'].isin(dates[split:])].dropna(subset=[label])
print(f'Train: {len(train_df):,} rows, {len(dates[:split])}d | Test: {len(test_df):,} rows, {len(dates[split:])}d')

X_train = train_df[feat_cols].fillna(0)
y_train = train_df[label]
X_test = test_df[feat_cols].fillna(0)
y_test = test_df[label]

def oos_ic(preds, df_t, lab):
    df_e = df_t.copy(); df_e['pred'] = preds
    ics = []
    for d, g in df_e.groupby('date'):
        if len(g) < 10: continue
        ic, _ = spearmanr(g['pred'], g[lab])
        if not np.isnan(ic): ics.append(ic)
    a = np.array(ics)
    return round(float(a.mean()),5), round(float(a.mean()/a.std() if a.std()>0 else 0),4)

model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1)
model.fit(X_train, y_train)

imp = pd.DataFrame({'feature': feat_cols, 'gain': model.booster_.feature_importance(importance_type='gain')})
imp = imp.sort_values('gain', ascending=False)

preds_all = model.predict(X_test)
ic_all, icir_all = oos_ic(preds_all, test_df, label)
print(f'All {len(feat_cols)} feats: IC={ic_all:+.5f} ICIR={icir_all:.4f}')

print('\nForward ablation:')
results = []
for n in [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, len(feat_cols)]:
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
print(f'\nDUAL BOARD:')
print(f'  Best: n={best["n"]:.0f}, ICIR={best["icir"]:.4f}')
print(f'  Saturation (95%): n={sat["n"]:.0f} ({sat["n"]/len(feat_cols):.0%} of features)')
print(f'  All features: ICIR={icir_all:.4f}')

shutil.rmtree(reg_dir)
