"""Scan VarThresh thresholds with chronological signed IC backtest."""
import pandas as pd, numpy as np, warnings, sys, os, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_selector import BruteForceGenerator, dedup_l2

t0 = time.time()
print('Loading + generating...', flush=True)
panel = pd.read_parquet('data/panel_full_enriched_v3.parquet')
dates = sorted(panel['date'].unique())[-125:]
panel = panel[panel['date'].isin(dates)]

main_syms = sorted(panel[~panel['board'].isin(['GEM', 'STAR'])]['symbol'].unique())
np.random.seed(42)
syms_300 = sorted(np.random.choice(main_syms, size=min(300, len(main_syms)), replace=False))
panel = panel[panel['symbol'].isin(syms_300)]

cleaner = CleaningPipeline(); m, d = cleaner.run_train(panel)
df = m; df = LabelEngine.build_labels(df); df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=6); df = df.dropna(subset=['label_1d_net'])
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

gen = BruteForceGenerator(); raw_cols = gen._eligible(df)
FAMILIES = ['pct_change', 'rolling_mean', 'rolling_std', 'rolling_max', 'diff', 'momentum', 'EMA']
feat_dfs = []; base_cols = set(df.columns)
for fam in FAMILIES:
    new = gen.generate_family(df, fam, raw_cols=raw_cols, dtype='float32')
    bf = [c for c in new.columns if '_brute_' in c and c not in base_cols]
    if bf: feat_dfs.append(new[bf])
    del new
keep_base = [c for c in base_cols if c in df.columns]
all_feats = pd.concat([df[keep_base]] + feat_dfs, axis=1)
label = all_feats['label_1d_net'].values; del feat_dfs
numeric_cols = [c for c in all_feats.columns if all_feats[c].dtype in ('float64', 'float32', 'int64', 'int32')]
print('{} cols, {} rows ({}s)'.format(len(numeric_cols), len(all_feats), int(time.time()-t0)))

# Chronological split
dates_arr = all_feats['date'].values
uniq_dates = sorted(all_feats['date'].unique())
cut_date = uniq_dates[int(len(uniq_dates) * 0.75)]
train_mask = dates_arr < cut_date
test_mask = dates_arr >= cut_date
print('Train: {} rows, Test: {} rows, cutoff={}'.format(train_mask.sum(), test_mask.sum(), str(cut_date)[:10]))

# DedupL2
sample = all_feats.sample(min(5000, len(all_feats)), random_state=42)
survived_dedup = dedup_l2(numeric_cols, sample[numeric_cols])

# Compute MinMax variance on TRAIN only
variances, names = [], []
for c in survived_dedup:
    train_vals = all_feats.loc[train_mask, c].values.astype(np.float64)
    valid = train_vals[~np.isnan(train_vals)]
    if len(valid) < 10: continue
    uniq, cnts = np.unique(valid, return_counts=True)
    if cnts.max() / len(valid) > 0.95: continue
    mn, mx = valid.min(), valid.max()
    if mx == mn: continue
    normed = (valid - mn) / (mx - mn)
    v = np.var(normed, ddof=0)
    if not np.isnan(v):
        variances.append(v); names.append(c)

variances = np.array(variances)
num_features = all_feats[numeric_cols].fillna(0).values.astype(np.float32)
num_idx = {c: i for i, c in enumerate(numeric_cols)}

print('After Mode+Invalid: {} features'.format(len(variances)))
print('Variance: min={:.6f} p50={:.6f} max={:.4f}'.format(variances.min(), np.median(variances), variances.max()))

header = '{:<10} {:<6} {:>9} {:>9} {:>9} {:>7}'.format('Thresh', 'Feats', 'Mean IC', 'IC_std', 'IR', 'IC>0%')
print('\n' + header)
print('-' * 55)

for th in [0.001, 0.0015, 0.002, 0.0022, 0.0025, 0.0028, 0.003]:
    keep = [c for i, c in enumerate(names) if variances[i] >= th and c in num_idx]
    if len(keep) < 20: continue
    idx = [num_idx[c] for c in keep]
    X_tr = num_features[train_mask][:, idx]
    y_tr = label[train_mask].astype(np.float32)
    X_te = num_features[test_mask][:, idx]
    y_te = label[test_mask].astype(np.float32)

    gbm = LGBMRegressor(n_estimators=150, max_depth=6, num_leaves=31, learning_rate=0.05,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.6,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1)
    gbm.fit(X_tr, y_tr)
    pred = gbm.predict(X_te)

    td = pd.DataFrame({'d': dates_arr[test_mask], 'p': pred, 'l': y_te})
    ics = {}
    for d, g in td.groupby('d'):
        if len(g) >= 10:
            ic, _ = spearmanr(g['p'], g['l'])
            ics[d] = ic
    ic_s = pd.Series(ics).dropna()
    mean_ic = ic_s.mean()
    ic_std = ic_s.std()
    ir = mean_ic / ic_std if ic_std > 0 else 0
    pos = (ic_s > 0).mean() * 100

    print('{:<10} {:<6} {:>+.4f}     {:>+.4f}    {:>+.3f} {:>6.1f}%'.format(th, len(keep), mean_ic, ic_std, ir, pos))

print('\nDone ({}s)'.format(int(time.time()-t0)))
