"""诊断: sweep 里 main 3d/5d 模型为何 IC 恰为 0.0 (预测每日常数? 树太少?).

复刻 combo 2 (3e7, 0.2) 的 train/es/test 切片与 train_reg, 训练 3d/5d/10d,
打印 num_trees + 测试集预测统计 (每日常数 → IC NaN → 0.0).
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline1.dual_track_trainer import DualTrackTrainer
from scripts._sweep_liquidity_filter import (
    ES_DAYS,
    TEST_DAYS,
    apply_filter,
    train_reg,
)

BOARD = "main"
MA, BP = 3e7, 0.2
CACHE = r"data/_sweep_uni/main.parquet"

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
print(
    f"train_dates={len(train_dates)} es_dates={len(es_dates)} test_dates={len(test_dates)}",
    flush=True,
)

for kind in ("3d_reg", "5d_reg", "10d_reg"):
    h = kind.split("d")[0]
    label = f"label_pm_{h}d_net"
    train = apply_filter(cache.loc[cache["date"].isin(train_dates), need], MA, BP)
    es = apply_filter(cache.loc[cache["date"].isin(es_dates), need], MA, BP)
    model = train_reg(BOARD, kind, train, es, cols)
    ntree = model.booster_.num_trees()
    best = getattr(model, "best_iteration_", None)
    test = apply_filter(cache.loc[cache["date"].isin(test_dates), need], 5e7, 0.2)
    sub = test.dropna(subset=[label]).copy()
    pred = model.predict(np.nan_to_num(sub[cols].values, nan=0.0))
    per_day_uniq = (
        pd.DataFrame({"date": sub["date"].values, "_p": pred})
        .groupby("date")["_p"]
        .nunique()
    )
    print(
        f"[{kind}] trees={ntree} best={best} pred: min={pred.min():.4f} max={pred.max():.4f} "
        f"std={pred.std():.4f} nunique={np.unique(pred).size} "
        f"per_day_uniq min={per_day_uniq.min()} median={per_day_uniq.median():.0f}",
        flush=True,
    )
    # label 健康度
    print(
        f"    label {label}: nonnull={sub[label].notna().sum():,} "
        f"nunique={sub[label].nunique()} std={sub[label].std():.4f}",
        flush=True,
    )
    del train, es, model
    gc.collect()
