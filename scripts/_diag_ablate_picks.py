"""诊断: 3y vs 1y 裁剪在 OOS 同日期上, 狙击池 score 为何不同 (消融 picks 非逐位一致根因)."""

import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from app.pipeline_parallel.config import SYSTEMS
from app.pipeline_parallel.scoring import pool_score, select_topn
from scripts._ablate_train_window_quality import load_window

t0 = time.time()
work3 = load_window(726)
gc.collect()
print(f"3y loaded {time.time() - t0:.0f}s", flush=True)
work1 = load_window(242)
gc.collect()
print(f"1y loaded {time.time() - t0:.0f}s", flush=True)

pool = SYSTEMS["sniper"].pool
board = "main"
OOS_DAYS = 126


def topk(work, lab):
    sub = work[work["board"] == board]
    dates = np.sort(sub["date"].unique())
    cutoff = dates[-OOS_DAYS]
    bm = sub["date"].values >= cutoff
    s = sub[bm]
    score = pool_score(s, pool)
    s = s.copy()
    s["score"] = score.values
    top = select_topn(s, score, SYSTEMS["sniper"].top_n)
    return s, top


s3, t3 = topk(work3, "3y")
s1, t1 = topk(work1, "1y")
print(f"s3 rows={len(s3)} s1 rows={len(s1)} top3={len(t3)} top1={len(t1)}", flush=True)

# 按 date 对齐两组 top-5 选股, 找首个差异日
t3d = {d: set(g["symbol"]) for d, g in t3.groupby("date")}
t1d = {d: set(g["symbol"]) for d, g in t1.groupby("date")}
first = None
for d in sorted(t3d):
    if t3d[d] != t1d.get(d):
        first = d
        break
print(f"首个差异日: {first}", flush=True)
if first is not None:
    miss3 = t1d[first] - t3d[first]  # 在 1y 被选但 3y 没选
    miss1 = t3d[first] - t1d[first]  # 在 3y 被选但 1y 没选
    print(f"  3y 有 1y 无: {sorted(miss1)}", flush=True)
    print(f"  1y 有 3y 无: {sorted(miss3)}", flush=True)
    # 该日全横截面的 score 排序对比
    r3 = s3[s3["date"] == first].copy().sort_values("score", ascending=False)
    r1 = s1[s1["date"] == first].copy().sort_values("score", ascending=False)
    r3 = r3.reset_index(drop=True)
    r1 = r1.reset_index(drop=True)
    n = min(15, len(r3))
    print(f"  --- {first} 日 score 前 {n} (3y vs 1y) ---", flush=True)
    m = r3.merge(
        r1[["symbol", "score"]], on="symbol", how="outer", suffixes=("_3y", "_1y")
    ).sort_values("score_3y", ascending=False)
    print(m.head(n).to_string(), flush=True)
    # 边界符号的池特征值
    probe = sorted(set(miss3) | set(miss1))
    if probe:
        cols = ["symbol"] + list(pool)
        a3 = r3[r3["symbol"].isin(probe)][cols]
        a1 = r1[r1["symbol"].isin(probe)][cols]
        print("  --- 边界符号池特征值 (3y) ---", flush=True)
        print(a3.to_string(), flush=True)
        print("  --- 边界符号池特征值 (1y) ---", flush=True)
        print(a1.to_string(), flush=True)

# 全局: score 列级最大偏差
for c in pool:
    if c in s3.columns and c in s1.columns:
        j = pd.merge(
            s3[["symbol", "date", c]],
            s1[["symbol", "date", c]],
            on=["symbol", "date"],
            suffixes=("_3y", "_1y"),
        )
        d = (j[f"{c}_3y"] - j[f"{c}_1y"]).abs()
        if d.max() > 1e-12:
            print(
                f"  列 {c}: 最大偏差 {d.max():.3e} at "
                f"{j.loc[d.idxmax(), 'symbol']} {j.loc[d.idxmax(), 'date']}",
                flush=True,
            )

print(f"done {time.time() - t0:.0f}s", flush=True)
