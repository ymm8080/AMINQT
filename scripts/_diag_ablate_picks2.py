# -*- coding: utf-8 -*-
"""探针: 603488 在 3y vs 1y 帧内, 2026-02-03 前后 5 日窗口的 ret/_vol_pct/pv_corr_5 输入对比."""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from scripts._ablate_train_window_quality import load_window

t0 = time.time()
SYM = "603488"
work3 = load_window(726)
work1 = load_window(242)
print(f"loaded {time.time()-t0:.0f}s", flush=True)


def ret_of(df):
    return df.groupby("symbol", sort=False)["close_hfq"].pct_change()


def vol_of(df):
    return df.groupby("symbol", sort=False)["volume"].pct_change()


for wname, work in (("3y", work3), ("1y", work1)):
    s = work[work["symbol"] == SYM].sort_values("date")
    s = s.assign(ret=ret_of(s), _vol_pct=vol_of(s))
    tail = s.tail(12)[
        ["date", "close_hfq", "volume", "is_suspended", "ret", "_vol_pct",
         "vol_ratio", "pv_corr_5", "ma5", "rps_60"]
    ]
    print(f"\n===== {SYM} {wname} 尾部 12 行 =====", flush=True)
    print(tail.to_string(), flush=True)

# 两帧在 2026-02-03 前 5 日 (含该日) 的输入逐位对比
d0 = pd.Timestamp("2026-02-03")
for wname, work in (("3y", work3), ("1y", work1)):
    s = work[(work["symbol"] == SYM) & (work["date"] <= d0)].sort_values("date")
    s = s.assign(ret=ret_of(s), _vol_pct=vol_of(s))
    win = s.tail(6)[["date", "close_hfq", "volume", "ret", "_vol_pct", "pv_corr_5"]]
    print(f"\n--- {SYM} {wname} <=2026-02-03 末 6 行 ---", flush=True)
    print(win.to_string(), flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
