"""Fix calibrator and re-predict with 1d_cls probability."""
import pickle, json, pandas as pd, numpy as np, pyarrow.parquet as pq, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LogisticRegression

FPATH = 'data/factor_registry/features_main_20260730T195247.parquet'
MODEL_PATH = 'models/pipeline1/main_current.pkl'

with open(MODEL_PATH, 'rb') as f:
    b = pickle.load(f)
feats = [c for c in b['feature_cols'] if c in pq.read_schema(FPATH).names]
reg = b['models']['1d_reg'][0]
cls = b['models']['1d_cls'][0]

df = pd.read_parquet(FPATH, columns=['symbol', 'date', 'label_1d_net'] + feats)
df = df.dropna(subset=['label_1d_net'])
dates = sorted(df['date'].unique())
n = len(dates)
calib = df[df['date'].isin(set(dates[int(n * 0.85):int(n * 0.90)]))]
today = df[df['date'] == df['date'].max()]
today = today[today['symbol'].str.match(r'^(60[0-3]|00[0-2]|601|603|605)')]

# Re-fit calibrator on 1d_cls output
X_ca = calib[feats].fillna(0).values.astype(np.float32)
cls_raw_ca = cls.predict_proba(X_ca)[:, 1]
cal = LogisticRegression(penalty=None, solver='lbfgs')
cal.fit(cls_raw_ca.reshape(-1, 1), (calib['label_1d_net'].values > 0).astype(int))
print(f'Platt calibrator re-fit on {len(calib):,} rows')

# Predict today
X_t = today[feats].fillna(0).values.astype(np.float32)
pred_1d = reg.predict(X_t)
prob_up = cal.predict_proba(cls.predict_proba(X_t)[:, 1].reshape(-1, 1))[:, 1]

r = pd.DataFrame({
    'symbol': today['symbol'].values,
    'pred_1d': pred_1d,
    'prob_up': prob_up,
    'adj': pred_1d * prob_up,
}).sort_values('adj', ascending=False)

os.makedirs('data/lists', exist_ok=True)
r.to_parquet('data/lists/list_20260730.parquet', index=False)

# Update model bundle with fixed calibrator
b['calibrator'] = cal
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(b, f)

print(f'\n{"Rank":<5} {"Symbol":<10} {"Pred_1d":>10} {"Prob":>7}')
print(f'{"-"*38}')
for i, (_, row) in enumerate(r.head(10).iterrows(), 1):
    print(f'{i:<5} {row.symbol:<10} {row.pred_1d:>+.6f} {row.prob_up:>6.1%}')
print(f'\nProb range: [{r.prob_up.min():.1%}, {r.prob_up.max():.1%}]')
print(f'{len(r)} candidates. Model updated.')
