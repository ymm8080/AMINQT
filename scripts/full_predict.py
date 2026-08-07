"""Full prediction: 1d/3d/5d reg + 1d_cls + q10/q90 + pain. Uses existing 1d_reg/1d_cls, trains rest."""

import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression

FPATH = "data/factor_registry/features_main_20260730T195247.parquet"
MODEL_PATH = "models/pipeline1/main_current.pkl"

with open(MODEL_PATH, "rb") as f:
    b = pickle.load(f)
feats = [c for c in b["feature_cols"] if c in pq.read_schema(FPATH).names]
reg_1d = b["models"]["1d_reg"][0]
cls_1d = b["models"]["1d_cls"][0]
print(f"Loaded: {len(feats)}feats 1d_reg+1d_cls")

# Load all labels
label_cols = ["symbol", "date", "label_1d_net"]
for lbl in ["label_3d_net", "label_5d_net"]:
    if lbl in pq.read_schema(FPATH).names:
        label_cols.append(lbl)
df = pd.read_parquet(FPATH, columns=label_cols + feats).dropna(subset=["label_1d_net"])
# Drop rows with NaN in 3d/5d labels (masked days)
for lbl in ["label_3d_net", "label_5d_net"]:
    if lbl in df.columns:
        df = df.dropna(subset=[lbl])
dates = sorted(df["date"].unique())
n = len(dates)
train = df[df["date"].isin(set(dates[: int(n * 0.85)]))]
calib = df[df["date"].isin(set(dates[int(n * 0.85) : int(n * 0.90)]))]
today = df[df["date"] == df["date"].max()]
today = today[today["symbol"].str.match(r"^(60[0-3]|00[0-2]|601|603|605)")]
print(f"train={len(train):,} calib={len(calib):,} today={len(today)}")

X_tr = train[feats].fillna(0).values.astype(np.float32)
y1 = train["label_1d_net"].values.astype(np.float32)
y3 = train["label_3d_net"].values.astype(np.float32) if "label_3d_net" in train else y1
y5 = train["label_5d_net"].values.astype(np.float32) if "label_5d_net" in train else y1
X_t = today[feats].fillna(0).values.astype(np.float32)

CR = dict(
    n_estimators=200,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

# --- Train 3d_reg ---
t0 = time.time()
reg_3d = LGBMRegressor(**CR).fit(X_tr, y3)
print(f"3d_reg: {time.time() - t0:.0f}s")

# --- Train 5d_reg ---
t0 = time.time()
reg_5d = LGBMRegressor(**CR).fit(X_tr, y5)
print(f"5d_reg: {time.time() - t0:.0f}s")

# --- Train q10 (10th percentile) ---
t0 = time.time()
q10 = LGBMRegressor(
    objective="quantile",
    alpha=0.10,
    **{k: v for k, v in CR.items() if k != "reg_alpha"},
).fit(X_tr, y1)
print(f"q10: {time.time() - t0:.0f}s")

# --- Train q90 (90th percentile) ---
t0 = time.time()
q90 = LGBMRegressor(
    objective="quantile",
    alpha=0.90,
    **{k: v for k, v in CR.items() if k != "reg_alpha"},
).fit(X_tr, y1)
print(f"q90: {time.time() - t0:.0f}s")

# --- Train pain (P(loss > 2%)) ---
t0 = time.time()
pain = LGBMClassifier(
    n_estimators=150,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
).fit(X_tr, (y1 < -0.02).astype(int))
print(f"pain: {time.time() - t0:.0f}s")

# --- Re-fit Platt calibrator ---
X_ca = calib[feats].fillna(0).values.astype(np.float32)
cls_raw_ca = cls_1d.predict_proba(X_ca)[:, 1]
cal = LogisticRegression(penalty=None, solver="lbfgs")
cal.fit(cls_raw_ca.reshape(-1, 1), (calib["label_1d_net"].values > 0).astype(int))
print("Platt calibrator re-fit")

# --- Predict ALL ---
print("Predicting...")
P = {}
P["pred_1d"] = reg_1d.predict(X_t)
P["prob_up"] = cal.predict_proba(cls_1d.predict_proba(X_t)[:, 1].reshape(-1, 1))[:, 1]
P["pred_3d"] = reg_3d.predict(X_t)
P["pred_5d"] = reg_5d.predict(X_t)
P["q10"] = q10.predict(X_t)
P["q90"] = q90.predict(X_t)
P["pain"] = pain.predict_proba(X_t)[:, 1]
P["adj"] = P["pred_1d"] * P["prob_up"] * (1 - P["pain"])

r = pd.DataFrame(
    {
        "symbol": today["symbol"].values,
        "pred_1d": P["pred_1d"],
        "prob_up": P["prob_up"],
        "pred_3d": P["pred_3d"],
        "pred_5d": P["pred_5d"],
        "q10": P["q10"],
        "q90": P["q90"],
        "pain": P["pain"],
        "adj_score": P["adj"],
    }
).sort_values("adj_score", ascending=False)

os.makedirs("data/lists", exist_ok=True)
r.to_parquet("data/lists/list_20260730.parquet", index=False)

# Save updated bundle
b["models"]["3d_reg"] = (reg_3d, {})
b["models"]["5d_reg"] = (reg_5d, {})
b["quantile_models"] = {"q10": q10, "q90": q90}
b["pain_model"] = pain
b["calibrator"] = cal
with open(MODEL_PATH, "wb") as f:
    pickle.dump(b, f)

# --- REPORT ---
print(f"\n{'=' * 90}")
print("  FULL PREDICTION REPORT — Top 10 Risk-Adjusted")
print(f"{'=' * 90}")
print(
    f"{'Rk':<3} {'Symbol':<10} {'1d':>8} {'3d':>8} {'5d':>8} {'Prob':>6} {'Pain':>6} {'Q10':>8} {'Q90':>8}"
)
print(f"{'-' * 85}")
for i, (_, row) in enumerate(r.head(10).iterrows(), 1):
    print(
        f"{i:<3} {row.symbol:<10} {row.pred_1d:>+.4f} {row.pred_3d:>+.4f} {row.pred_5d:>+.4f} "
        f"{row.prob_up:>5.1%} {row.pain:>5.1%} {row.q10:>+.4f} {row.q90:>+.4f}"
    )

print(f"\nPred_1d: [{r.pred_1d.min():+.4f}, {r.pred_1d.max():+.4f}]")
print(f"Pred_3d: [{r.pred_3d.min():+.4f}, {r.pred_3d.max():+.4f}]")
print(f"Pred_5d: [{r.pred_5d.min():+.4f}, {r.pred_5d.max():+.4f}]")
print(f"Prob_up: [{r.prob_up.min():.1%}, {r.prob_up.max():.1%}]")
print(f"Pain:    [{r.pain.min():.1%}, {r.pain.max():.1%}]")
print(f"Q10-Q90: [{r.q10.min():+.4f}, {r.q90.max():+.4f}]")
print(f"\n{len(r)} candidates. Model saved: {MODEL_PATH}")
