#!/usr/bin/env python3
"""对比本地 ChipDistribution 计算结果 vs akshare stock_cyq_em 精确数据."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import akshare as ak  # noqa: E402
from app.indicators.chip_distribution import ChipDistribution  # noqa: E402

V3_PATH = "data/panel_full_enriched_v3.parquet"

# 测试股票
test_codes = ["000001", "600519", "002594"]

print("=" * 80)
print("对比: 本地 ChipDistribution vs akshare stock_cyq_em")
print("=" * 80)

# 加载 v3
v3 = pd.read_parquet(V3_PATH)

for code in test_codes:
    print(f"\n{'=' * 80}")
    print(f"股票: {code}")
    print(f"{'=' * 80}")

    # 1. akshare 数据
    try:
        ak_df = ak.stock_cyq_em(symbol=code, adjust="qfq")
        ak_df.columns = [
            "date",
            "winner_ratio_ak",
            "avg_cost_ak",
            "pct_90_low_ak",
            "pct_90_high_ak",
            "pct_90_con_ak",
            "pct_70_low_ak",
            "pct_70_high_ak",
            "pct_70_con_ak",
        ]
        ak_df["date"] = pd.to_datetime(ak_df["date"])
        ak_df["winner_ratio_ak"] = ak_df["winner_ratio_ak"] * 100  # 转百分比
        print(
            f"akshare: {len(ak_df)} 行, {ak_df['date'].min().date()} ~ {ak_df['date'].max().date()}"
        )
    except Exception as e:
        print(f"akshare 失败: {e}")
        continue

    # 2. 本地计算
    df = v3[v3["symbol"] == code].sort_values("date").reset_index(drop=True)
    if len(df) == 0:
        print(f"v3 中无 {code}")
        continue

    # 获取流通股本
    if "float_share" in df.columns and df["float_share"].notna().any():
        float_shares = df["float_share"].dropna().iloc[0] * 1e4
    elif "free_share" in df.columns and df["free_share"].notna().any():
        float_shares = df["free_share"].dropna().iloc[0] * 1e4
    else:
        valid = df[df["turn"] > 0] if "turn" in df.columns else pd.DataFrame()
        if len(valid):
            float_shares = float(
                valid["volume"].iloc[0] / (valid["turn"].iloc[0] / 100)
            )
        else:
            float_shares = 1e8

    chip = ChipDistribution(n_bins=400)

    # 逐日计算, 提取筹码分布指标
    results = []
    for _, r in df.iterrows():
        a01 = (r["close"] + r["open"] + r["low"] + r["high"]) / 4
        t = min(r["volume"] / float_shares, 1.0)
        if chip.dist is None or chip.dist.sum() == 0:
            chip.grid = np.linspace(r["low"] * 0.9, r["high"] * 1.1, chip.n_bins)
            chip.dist = np.zeros(chip.n_bins)
            chip.dist = chip._triangle(r["low"], r["high"], a01)
        else:
            chip.dist *= 1 - t
            chip.dist += t * chip._triangle(r["low"], r["high"], a01)

        # 计算指标
        dist_sum = chip.dist.sum()
        if dist_sum < 1e-12:
            continue

        # 累积分布
        cum = np.cumsum(chip.dist) / dist_sum

        # winner_ratio (获利比例) = WINNER(close)
        benefit = chip.winner(r["close"]) * 100

        # avg_cost (平均成本) = sum(grid * dist) / sum(dist)
        avg_cost = float((chip.grid * chip.dist).sum() / dist_sum)

        # pct_70: 15% ~ 85% 分位
        idx_15 = np.searchsorted(cum, 0.15)
        idx_85 = np.searchsorted(cum, 0.85)
        pct_70_low = float(chip.grid[min(idx_15, len(chip.grid) - 1)])
        pct_70_high = float(chip.grid[min(idx_85, len(chip.grid) - 1)])
        pct_70_con = (pct_70_high - pct_70_low) / avg_cost if avg_cost > 0 else 0

        # pct_90: 5% ~ 95% 分位
        idx_05 = np.searchsorted(cum, 0.05)
        idx_95 = np.searchsorted(cum, 0.95)
        pct_90_low = float(chip.grid[min(idx_05, len(chip.grid) - 1)])
        pct_90_high = float(chip.grid[min(idx_95, len(chip.grid) - 1)])
        pct_90_con = (pct_90_high - pct_90_low) / avg_cost if avg_cost > 0 else 0

        results.append(
            {
                "date": r["date"],
                "winner_ratio_calc": benefit,
                "avg_cost_calc": avg_cost,
                "pct_70_low_calc": pct_70_low,
                "pct_70_high_calc": pct_70_high,
                "pct_70_con_calc": pct_70_con,
                "pct_90_low_calc": pct_90_low,
                "pct_90_high_calc": pct_90_high,
                "pct_90_con_calc": pct_90_con,
            }
        )

    calc_df = pd.DataFrame(results)
    print(
        f"本地计算: {len(calc_df)} 行, {calc_df['date'].min().date()} ~ {calc_df['date'].max().date()}"
    )

    # 3. 合并对比 (只比 akshare 有数据的日期)
    merged = ak_df.merge(calc_df, on="date", how="inner")
    print(f"重叠日期: {len(merged)} 行")

    if len(merged) == 0:
        print("无重叠日期!")
        continue

    # 4. 误差分析
    metrics = [
        ("winner_ratio", "winner_ratio_ak", "winner_ratio_calc"),
        ("avg_cost", "avg_cost_ak", "avg_cost_calc"),
        ("pct_70_low", "pct_70_low_ak", "pct_70_low_calc"),
        ("pct_70_high", "pct_70_high_ak", "pct_70_high_calc"),
        ("pct_70_con", "pct_70_con_ak", "pct_70_con_calc"),
        ("pct_90_low", "pct_90_low_ak", "pct_90_low_calc"),
        ("pct_90_high", "pct_90_high_ak", "pct_90_high_calc"),
        ("pct_90_con", "pct_90_con_ak", "pct_90_con_calc"),
    ]

    print(
        f"\n{'指标':<16} {'akshare均值':>12} {'计算均值':>12} {'MAE':>10} {'MAPE':>8} {'相关系数':>8}"
    )
    print("-" * 70)
    for name, ak_col, calc_col in metrics:
        ak_vals = merged[ak_col].astype(float)
        calc_vals = merged[calc_col].astype(float)
        mae = (ak_vals - calc_vals).abs().mean()
        mape = (
            (ak_vals - calc_vals).abs() / ak_vals.abs().replace(0, np.nan)
        ).mean() * 100
        corr = ak_vals.corr(calc_vals) if len(ak_vals) > 1 else 0
        print(
            f"{name:<16} {ak_vals.mean():>12.4f} {calc_vals.mean():>12.4f} {mae:>10.4f} {mape:>7.1f}% {corr:>8.4f}"
        )

    # 5. 打印最近 5 天对比
    print("\n最近 5 天对比:")
    cols = [
        "date",
        "winner_ratio_ak",
        "winner_ratio_calc",
        "avg_cost_ak",
        "avg_cost_calc",
        "pct_70_con_ak",
        "pct_70_con_calc",
    ]
    print(merged[cols].tail(5).to_string(index=False))
