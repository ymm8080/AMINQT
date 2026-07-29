#!/usr/bin/env python3
"""基准测试: 本地 ChipDistribution 计算速度."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from app.indicators.chip_distribution import ChipDistribution

V3_PATH = "data/panel_full_enriched_v3.parquet"

# 加载 v3, 取 5 只股票测试
print("加载 v3...")
v3 = pd.read_parquet(V3_PATH)

test_symbols = ["000001", "000002", "600519", "002594", "000063"]
print(f"测试 {len(test_symbols)} 只股票...")

times = []
for sym in test_symbols:
    df = v3[v3["symbol"] == sym].sort_values("date").reset_index(drop=True)
    if len(df) == 0:
        print(f"  {sym}: 无数据")
        continue

    # 用 free_share 或 float_share 作为流通股本, 否则用 volume 估算
    if "float_share" in df.columns and df["float_share"].notna().any():
        float_shares = df["float_share"].dropna().iloc[0] * 1e4  # 万股 -> 股
    elif "free_share" in df.columns and df["free_share"].notna().any():
        float_shares = df["free_share"].dropna().iloc[0] * 1e4
    else:
        # 用成交量反推: float_shares ≈ volume / turn
        turn_col = "turn" if "turn" in df.columns else None
        if turn_col and df[turn_col].notna().any():
            valid = df[df[turn_col] > 0]
            if len(valid):
                float_shares = float(
                    valid["volume"].iloc[0] / (valid[turn_col].iloc[0] / 100)
                )
            else:
                float_shares = 1e8
        else:
            float_shares = 1e8

    t0 = time.time()
    chip = ChipDistribution(n_bins=400)
    result = chip.build(df.copy(), float_shares=float_shares)
    elapsed = time.time() - t0
    times.append(elapsed)

    print(
        f"  {sym}: {len(df)} 行, float_shares={float_shares / 1e8:.2f}亿股, 耗时 {elapsed:.2f}s"
    )

avg_time = np.mean(times)
total_est = avg_time * 3244
print(f"\n平均: {avg_time:.2f}s/股")
print(f"预计 3244 股: {total_est:.0f}s = {total_est / 60:.1f} 分钟")
