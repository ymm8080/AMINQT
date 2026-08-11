# -*- coding: utf-8 -*-
"""定位 sweep 3d/5d 坍缩根因: es 块位置是否导致早停到 3 棵树.

对照实验 (main, combo 2 = 3e7_0.2):
  A. 我的 sweep: es = dates[-80:-60]  (20日, 紧贴测试前)
  B. 生产位置:   es = dates[-100:-80] (与 split_window es 同位置)
各训练 5d_reg, 打印 trees / best_iteration / 测试预测每日常数性.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts._sweep_liquidity_filter import TEST_DAYS, apply_filter, train_reg
from app.pipeline1.dual_track_trainer import DualTrackTrainer

CACHE = r"data/_sweep_uni/main.parquet"
MA, BP = 3e7, 0.2

t = DualTrackTrainer.load(r"models/pipeline1/main_20260810.pkl")
cols = t["feature_cols"]
del t
gc.collect()

schema = pq.ParquetFile(CACHE).schema.names
labels = [f"label_pm_{h}d_net" for h in (3, 5, 10)]
need = [c for c in ["date", "amount"] + cols + labels if c in schema]
cache = pd.read_parquet(CACHE, columns=need)
dates = sorted(cache["date"].unique())
print(f"total dates={len(dates)} last={dates[-1]}", flush=True)
test_dates = dates[-TEST_DAYS:]
train_all = cache[cache["date"].isin(dates[: -TEST_DAYS])]
test_rows = apply_filter(cache[cache["date"].isin(test_dates)], 5e7, 0.2)

def run(name, es_dates):
    train = apply_filter(train_all.copy(), MA, BP)
    es = apply_filter(cache[cache["date"].isin(es_dates)], MA, BP)
    m = train_reg("main", "5d_reg", train, es, cols)
    ntree = m.booster_.num_trees()
    sub = test_rows.dropna(subset=["label_pm_5d_net"]).copy()
    pred = m.predict(np.nan_to_num(sub[cols].values, nan=0.0))
    uniq = (
        pd.DataFrame({"date": sub["date"].values, "_p": pred})
        .groupby("date")["_p"]
        .nunique()
    )
    print(
        f"[{name}] es {es_dates[0].date()}..{es_dates[-1].date()} n_es={len(es):,} "
        f"trees={ntree} best={getattr(m, 'best_iteration_', None)} "
        f"pred_std={pred.std():.4f} perday_uniq min={uniq.min()} med={uniq.median():.0f}",
        flush=True,
    )
    del train, es, m
    gc.collect()

run("A_sweep_es", dates[-TEST_DAYS - 20 : -TEST_DAYS])   # [-80:-60]
run("B_prod_es", dates[-TEST_DAYS - 40 : -TEST_DAYS - 20])  # [-100:-80]
