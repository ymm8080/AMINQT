"""Analyze chip_concentration, profit_ratio, free_float_turnover_rate vs their potential duplicates."""

import os

import pyarrow.parquet as pq

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")

cols = [
    "chip_concentration",
    "profit_ratio",
    "free_float_turnover_rate",
    "pct_70_con_x",
    "pct_90_con_x",
    "pct_70_con_y",
    "pct_90_con_y",
    "winner_ratio_x",
    "winner_ratio_y",
    "turnover_rate",
]
cols = [c for c in cols if c in pq.read_schema(PANEL).names]

print(f"Reading columns: {cols}")
df = pq.read_table(PANEL, columns=cols).to_pandas()
print(f"Rows: {len(df):,}")

print("\n" + "=" * 80)
print("1. chip_concentration vs pct_90_con_x (are they duplicates?)")
print("=" * 80)
if "chip_concentration" in df.columns and "pct_90_con_x" in df.columns:
    both = df[["chip_concentration", "pct_90_con_x"]].dropna()
    print(
        f"  chip_concentration: NaN={df['chip_concentration'].isna().mean() * 100:.2f}%, range=[{df['chip_concentration'].min():.6f}, {df['chip_concentration'].max():.6f}]"
    )
    print(
        f"  pct_90_con_x:       NaN={df['pct_90_con_x'].isna().mean() * 100:.2f}%, range=[{df['pct_90_con_x'].min():.6f}, {df['pct_90_con_x'].max():.6f}]"
    )
    print(f"  Both non-null: {len(both):,}")
    if len(both) > 0:
        corr = both["chip_concentration"].corr(both["pct_90_con_x"])
        diff = (both["chip_concentration"] - both["pct_90_con_x"]).abs()
        print(f"  Correlation: {corr:.6f}")
        print(f"  Mean diff: {diff.mean():.6f}, Max diff: {diff.max():.6f}")
        print(f"  Exact match (diff<0.001): {(diff < 0.001).mean() * 100:.2f}%")

print("\n" + "=" * 80)
print("2. profit_ratio vs winner_ratio_x (are they duplicates?)")
print("=" * 80)
if "profit_ratio" in df.columns and "winner_ratio_x" in df.columns:
    both = df[["profit_ratio", "winner_ratio_x"]].dropna()
    print(
        f"  profit_ratio:    NaN={df['profit_ratio'].isna().mean() * 100:.2f}%, range=[{df['profit_ratio'].min():.6f}, {df['profit_ratio'].max():.6f}]"
    )
    print(
        f"  winner_ratio_x:  NaN={df['winner_ratio_x'].isna().mean() * 100:.2f}%, range=[{df['winner_ratio_x'].min():.6f}, {df['winner_ratio_x'].max():.6f}]"
    )
    print(f"  Both non-null: {len(both):,}")
    if len(both) > 0:
        corr = both["profit_ratio"].corr(both["winner_ratio_x"])
        diff = (both["profit_ratio"] - both["winner_ratio_x"]).abs()
        print(f"  Correlation: {corr:.6f}")
        print(f"  Mean diff: {diff.mean():.6f}, Max diff: {diff.max():.6f}")
        print(f"  Exact match (diff<0.001): {(diff < 0.001).mean() * 100:.2f}%")
        # Check if profit_ratio = winner_ratio * 100
        diff100 = (both["profit_ratio"] - both["winner_ratio_x"] * 100).abs()
        print(
            f"  profit_ratio vs winner_ratio*100: mean_diff={diff100.mean():.6f}, max_diff={diff100.max():.6f}"
        )
        print(f"  Match ratio*100 (diff<0.1): {(diff100 < 0.1).mean() * 100:.2f}%")

print("\n" + "=" * 80)
print("3. free_float_turnover_rate vs turnover_rate (are they duplicates?)")
print("=" * 80)
if "free_float_turnover_rate" in df.columns and "turnover_rate" in df.columns:
    both = df[["free_float_turnover_rate", "turnover_rate"]].dropna()
    print(
        f"  free_float_turnover_rate: NaN={df['free_float_turnover_rate'].isna().mean() * 100:.2f}%"
    )
    print(
        f"  turnover_rate:           NaN={df['turnover_rate'].isna().mean() * 100:.2f}%"
    )
    print(f"  Both non-null: {len(both):,}")
    if len(both) > 0:
        corr = both["free_float_turnover_rate"].corr(both["turnover_rate"])
        diff = (both["free_float_turnover_rate"] - both["turnover_rate"]).abs()
        print(f"  Correlation: {corr:.6f}")
        print(f"  Mean diff: {diff.mean():.6f}, Max diff: {diff.max():.6f}")
        print(f"  Exact match (diff<0.001): {(diff < 0.001).mean() * 100:.2f}%")
