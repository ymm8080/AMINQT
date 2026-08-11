"""诊断 legacy 10d_reg 头为何早停到 3 棵树 (2026-08-10) — 精确复刻生产.

直接调用 dual_track_trainer.split_window + time_weights, 复刻 risk_filter (is_suspended),
只换 10d_reg 单头, 数树. numpy 布尔掩码取数, 规避 pandas block consolidation OOM.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
import pickle

from app.pipeline1.dual_track_trainer import DualTrackTrainer

FRAME = r"data/_diag_stage_main_3y.parquet"
BUNDLE = r"models/pipeline1/main_20260810.pkl"

pf = pq.ParquetFile(FRAME)
schema = set(pf.schema.names)
bundle = pickle.load(open(BUNDLE, "rb"))
full_cols = [c for c in bundle["feature_cols"] if c in schema]
need = ["date", "symbol"] + full_cols + ["label_pm_10d_net", "is_suspended"]
df = pf.read(columns=need).to_pandas()
for c in full_cols:
    df[c] = df[c].astype("float32")
print(f"frame rows={len(df)} cols={len(need)}", flush=True)

# split_window 只在日期上切分, 无需整帧物化段
dates = sorted(df["date"].unique())[-770:]
n = len(dates)
seg_lens = {"train": 615, "es": 20, "calib": 20, "test": 60}
# 用 _derive_seg_min_days 复刻真实段长
from app.pipeline1.dual_track_trainer import _derive_seg_min_days
seg_lens = _derive_seg_min_days(max(n, 560))
seg_lens["train"] = max(seg_lens["train"], 50)
pos = 0
seg_dates = {}
for k in ("train", "es", "calib", "test"):
    seg_dates[k] = dates[pos:pos + seg_lens[k]]
    pos += seg_lens[k]
seg_dates["test"] = dates[-seg_lens["test"]:]
print(f"total={n} seg_lens={seg_lens}", flush=True)
for k in seg_dates:
    print(f"  {k}: {seg_dates[k][0]:%Y-%m-%d}..{seg_dates[k][-1]:%Y-%m-%d} ({len(seg_dates[k])})", flush=True)


def seg_array(k, xcols):
    """numpy 布尔掩码取段 (规避 pandas block consolidation OOM)."""
    d = df["date"].values
    seg_nd = np.array(seg_dates[k]).astype("datetime64[ns]")
    in_seg = np.isin(d, seg_nd)
    keep = in_seg & ~df["is_suspended"].to_numpy(dtype=bool) & df["label_pm_10d_net"].notna().to_numpy()
    idx = np.flatnonzero(keep)
    arr = df.iloc[idx].loc[:, xcols].to_numpy(dtype=np.float32)
    y = df.iloc[idx]["label_pm_10d_net"].to_numpy(dtype=float)
    return arr, y


X, y = seg_array("train", full_cols)
X_es, y_es = seg_array("es", full_cols)
print(f"X={X.shape} X_es={X_es.shape}", flush=True)
d = df["date"].values
seg_nd_train = np.array(seg_dates["train"]).astype("datetime64[ns]")
train_keep = (np.isin(d, seg_nd_train)
              & ~df["is_suspended"].to_numpy(dtype=bool)
              & df["label_pm_10d_net"].notna().to_numpy())
train_dates_sub = sorted(set(seg_dates["train"]))
w_map = {dp: 0.5 ** ((len(train_dates_sub) - 1 - i) / 250) for i, dp in enumerate(train_dates_sub)}
w = np.array([w_map[dp] for dp in d[train_keep]])
print(f"w shape={w.shape} w[0]={w[0]:.4f} w[-1]={w[-1]:.4f}", flush=True)

m = lgb.LGBMRegressor(
    objective="huber", n_estimators=1000, learning_rate=0.05,
    random_state=42, verbosity=-1, num_leaves=31,
)
m.fit(X, y, sample_weight=w,
      eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"\n10d_reg EXACT-PROD: trees={m.booster_.num_trees()} best_iteration={m.best_iteration_}", flush=True)
p = m.predict(X[:2000])
print(f"pred sample: std={np.std(p):.6f} min={p.min():.6f} max={p.max():.6f}", flush=True)

del m
gc.collect()
m2 = lgb.LGBMRegressor(
    objective="huber", n_estimators=1000, learning_rate=0.05,
    random_state=42, verbosity=-1, num_leaves=31,
)
m2.fit(X, y, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"10d_reg EXACT-PROD noweight: trees={m2.booster_.num_trees()} best_iteration={m2.best_iteration_}", flush=True)
