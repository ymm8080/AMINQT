# -*- coding: utf-8 -*-
"""_cyq_recompute_v3_0909 引擎冒烟: 合成边界数据对照 python canonical."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
sys.path.insert(0, r"D:\AMINQT\AMINQT CODES\tmp_t")

from _cyq_recompute_v3_0909 import CYQ14, compute_stock_vec  # noqa: E402
from app.pipeline1.cyq_calculator import _compute_cyq_for_stock  # noqa: E402


def synth(sym, n, seed):
    rng = np.random.default_rng(seed)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.03, n)).round(2)
    open_ = close * (1 + rng.normal(0, 0.008, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.012, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.012, n)))
    # 边界: 一字板段 + NaN 换手段
    for i in range(20, 40):
        open_[i] = high[i] = low[i] = close[i]
    hsl = rng.uniform(0.5, 15, n)
    hsl[50:70] = np.nan
    return pd.DataFrame({
        "symbol": sym, "date": pd.bdate_range("2024-01-01", periods=n),
        "open": open_, "close": close, "high": high.round(2), "low": low.round(2),
        "turnover_rate": hsl,
    })


fails = 0
for k, (sym, n, seed) in enumerate([("000001", 300, 1), ("300911", 250, 2),
                                    ("688001", 400, 3)]):
    g = synth(sym, n, seed)
    mine = compute_stock_vec(g)
    canon = _compute_cyq_for_stock(
        g[["date", "open", "close", "high", "low", "turnover_rate"]])
    m = mine.merge(canon, on="date", suffixes=("_v", "_p"), how="inner")
    assert len(m) == n - 60, f"{sym}: 行数 {len(m)} != {n - 60}"
    for col in CYQ14:
        d = (m[f"{col}_v"] - m[f"{col}_p"]).abs().max()
        a = 1e-10 if col in ("winner_ratio", "pct_70_con", "pct_90_con") else 1e-9
        flag = "OK " if d <= a else "FAIL"
        if d > a:
            fails += 1
        print(f"{flag} {sym}.{col}: maxdiff={d:.3e}")
print("SMOKE", "PASS" if fails == 0 else f"FAIL({fails})")
