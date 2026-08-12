"""检查生产 bundle (main_20260810) 的 3d/5d/10d reg 模型在 OOS 测试段是否也坍缩.

若生产 3d/5d IC 正常 → sweep 训练路径有 bug; 若同为 ~0 → 系统性现象.
不重训, 只 load bundle + 对生产过滤测试行 predict.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
import gc
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.utils.daily_rank_ic import mean_rank_ic

CACHE = r"data/_sweep_uni/main.parquet"
BUNDLE = r"models/pipeline1/main_20260810.pkl"

with open(BUNDLE, "rb") as f:
    bundle = pickle.load(f)
print("bundle keys:", list(bundle.keys()))
models = bundle["models"]
print("model kinds:", list(models.keys()) if isinstance(models, dict) else type(models))
feature_cols = bundle["feature_cols"]
gc.collect()

schema = pq.ParquetFile(CACHE).schema.names
labels = [f"label_pm_{h}d_net" for h in (3, 5, 10)]
need = [c for c in ["date", "amount"] + feature_cols + labels if c in schema]
# 只读末 60 交易日切片 (pyarrow 过滤), 避免读全表 2.2GiB 与 sweep 抢内存.
dcol = pq.read_table(CACHE, columns=["date"]).to_pandas()["date"]
dates = sorted(dcol.unique())
test_dates = dates[-60:]
del dcol
gc.collect()
test = pq.read_table(
    CACHE, columns=need, filters=[("date", "in", test_dates)]
).to_pandas()
# 生产过滤行 (5e7, 0.2)
test = test[test["amount"] >= 5e7]
r = test.groupby("date")["amount"].rank(pct=True)
test = test[r > 0.2]
print(f"test rows={len(test):,} days={test['date'].nunique()}", flush=True)

for kind in ("3d_reg", "5d_reg", "10d_reg"):
    h = kind.split("d")[0]
    label = f"label_pm_{h}d_net"
    if kind not in models:
        print(f"[{kind}] not in bundle, skip", flush=True)
        continue
    m = models[kind]
    if isinstance(m, tuple):
        m = m[0]  # 生产 bundle: (LGBMRegressor, kind_label)
    sub = test.dropna(subset=[label]).copy()
    if len(sub) < 30:
        print(f"[{kind}] too few test rows", flush=True)
        continue
    cols_p = [c for c in feature_cols if c in sub.columns]
    pred = m.predict(np.nan_to_num(sub[cols_p].values, nan=0.0))
    per_day_uniq = (
        pd.DataFrame({"date": sub["date"].values, "_p": pred})
        .groupby("date")["_p"]
        .nunique()
    )
    df = sub.rename(columns={"_p": "score"})
    df["score"] = pred
    ic = mean_rank_ic(df, "score", label)
    ntree = m.booster_.num_trees() if hasattr(m, "booster_") else "?"
    print(
        f"[{kind}] trees={ntree} IC={ic:.4f} pred std={pred.std():.4f} "
        f"per_day_uniq min={per_day_uniq.min()} median={per_day_uniq.median():.0f}",
        flush=True,
    )
