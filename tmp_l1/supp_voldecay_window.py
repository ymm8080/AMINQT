# -*- coding: utf-8 -*-
"""补充: vol_decay_ratio 限断板后20日内的口径 (主脚本为不限期expanding口径)"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

OUT = r"D:\AMINQT\AMINQT CODES\tmp_l1"
feats = pd.read_parquet(f"{OUT}\\l1_features_sample.parquet")
dates = np.sort(feats["date"].unique())
ev = feats[feats["date"] >= dates[-500]].copy()

MIN_CS = 30


def ts_ic(d, x, y="r5"):
    return d.groupby("date").apply(
        lambda g: g[x].corr(g[y], method="spearman") if len(g) >= MIN_CS else np.nan
    )


def summarize(ic):
    ic = ic.dropna()
    t = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))
    return round(ic.mean(), 4), round(t, 2), len(ic)


# 限20日内: days_since_board_break (已封顶20) 非NaT 即在窗口内
sub = ev[ev["days_since_board_break"].notna()].copy()
print("capped-subset rows:", len(sub), "cs size:", len(sub) // 500)

for col in ["vol_decay_ratio", "days_since_board_break", "pullback_depth"]:
    ic_full, t_full, n = summarize(ts_ic(sub, col))
    half = sub["date"].unique()
    h = sorted(half)
    ic1, t1, _ = summarize(ts_ic(sub[sub["date"].isin(set(h[: len(h) // 2]))], col))
    ic2, t2, _ = summarize(ts_ic(sub[sub["date"].isin(set(h[len(h) // 2 :]))], col))
    print(f"{col}: capped20 IC={ic_full} t={t_full} n={n} | h1 {ic1}(t{t1}) h2 {ic2}(t{t2})")
