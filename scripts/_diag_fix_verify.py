"""验证修法: 扫描 train_reg 关掉早停, 固定 N 棵树 → 3d/5d/10d 预测是否健康.

正确不相交切分 (sweep 口径): train=dates[:-80], es=dates[-80:-60], test=dates[-60:].
combo 2 (3e7, 0.2). 打印 trees / pred 每日常数性 / 测试 IC.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline1.dual_track_trainer import (
    LGB_PARAMS_REG,
    NUM_LEAVES_OVERRIDE,
    DualTrackTrainer,
    risk_filter,
)
from app.utils.daily_rank_ic import mean_rank_ic
from scripts._sweep_liquidity_filter import ES_DAYS, TEST_DAYS, apply_filter

CACHE = r"data/_sweep_uni/main.parquet"
MA, BP = 3e7, 0.2
N_TREES = 200  # 固定树数, 无早停

t = DualTrackTrainer.load(r"models/pipeline1/main_20260810.pkl")
cols = t["feature_cols"]
del t
gc.collect()

schema = pq.ParquetFile(CACHE).schema.names
labels = [f"label_pm_{h}d_net" for h in (3, 5, 10)]
need = [c for c in ["date", "amount"] + cols + labels if c in schema]
cache = pd.read_parquet(CACHE, columns=need)
dates = sorted(cache["date"].unique())
test_dates = dates[-TEST_DAYS:]
es_dates = dates[-TEST_DAYS - ES_DAYS : -TEST_DAYS]
train_dates = dates[: -TEST_DAYS - ES_DAYS]
print(f"train={len(train_dates)} es={len(es_dates)} test={len(test_dates)}", flush=True)

import lightgbm as lgb

train_all = apply_filter(cache.loc[cache["date"].isin(train_dates), need], MA, BP)
es_all = apply_filter(cache.loc[cache["date"].isin(es_dates), need], MA, BP)
test_rows = apply_filter(cache[cache["date"].isin(test_dates)], 5e7, 0.2)

for kind in ("3d_reg", "5d_reg", "10d_reg"):
    h = kind.split("d")[0]
    label = f"label_pm_{h}d_net"
    train = train_all.dropna(subset=[label])
    es = es_all.dropna(subset=[label])
    train = risk_filter(train)
    es = risk_filter(es)
    cols_p = [c for c in cols if c in train.columns]
    X = np.nan_to_num(train[cols_p].values, nan=0.0, copy=False)
    y = train[label].values
    w = DualTrackTrainer.time_weights(train)
    params = dict(LGB_PARAMS_REG)
    nl = NUM_LEAVES_OVERRIDE.get(("main", kind))
    if nl is not None:
        params["num_leaves"] = nl
    params["n_estimators"] = N_TREES
    m = lgb.LGBMRegressor(**params)
    m.fit(X, y, sample_weight=w)
    ntree = m.booster_.num_trees()
    sub = test_rows.dropna(subset=[label]).copy()
    pred = m.predict(np.nan_to_num(sub[cols_p].values, nan=0.0))
    uniq = (
        pd.DataFrame({"date": sub["date"].values, "_p": pred})
        .groupby("date")["_p"]
        .nunique()
    )
    df = sub.copy()
    df["_pred"] = pred
    ic = mean_rank_ic(df.rename(columns={"_pred": "score"}), "score", label)
    print(
        f"[{kind}] trees={ntree} IC={ic:.4f} pred_std={pred.std():.4f} "
        f"perday_uniq min={uniq.min()} med={uniq.median():.0f}",
        flush=True,
    )
    del train, es, m
    gc.collect()
