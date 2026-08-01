#!/usr/bin/env python3
"""Verify fina_indicator coverage in v3 after fill, by year and by stock."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

v3 = pd.read_parquet(
    "data/panel_full_enriched_v3.parquet",
    columns=[
        "symbol",
        "date",
        "roe",
        "eps_yoy",
        "gross_margin",
        "net_margin",
        "profit_yoy",
        "rev_yoy",
        "debt_ratio",
        "roa",
        "current_ratio",
        "asset_turnover",
    ],
)
v3["year"] = v3["date"].dt.year

print("=" * 80)
print("FINA_INDICATOR Coverage in V3 (post-fill)")
print("=" * 80)
print(
    f"V3: {len(v3)} rows, {v3['symbol'].nunique()} stocks, {v3['date'].min()} ~ {v3['date'].max()}"
)
print(
    f"Columns checked: {[c for c in v3.columns if c not in ('symbol', 'date', 'year')]}"
)

# Per-year NaN rates
print("\n=== NaN rate by year ===")
fina_cols = [c for c in v3.columns if c not in ("symbol", "date", "year")]
for yr in sorted(v3["year"].unique()):
    sub = v3[v3["year"] == yr]
    n = len(sub)
    parts = []
    for c in fina_cols:
        na = sub[c].isna().mean() * 100
        parts.append(f"{c}={na:.1f}%")
    print(f"  {yr}: {n:>8,} rows | {' | '.join(parts[:4])}")
    print(f"  {' ':>14s} | {' | '.join(parts[4:])}")

# Per-stock roe coverage in 2023
print("\n=== 2023 per-stock roe coverage ===")
v3_2023 = v3[v3["year"] == 2023]
stock_cov = v3_2023.groupby("symbol")["roe"].apply(lambda x: x.notna().mean() * 100)
zero_cov = stock_cov[stock_cov == 0]
print(f"Stocks with 0% roe coverage in 2023: {len(zero_cov)}")
if len(zero_cov) > 0:
    print(f"  {sorted(zero_cov.index.tolist())}")

# Check the 12 previously-zero stocks
prev_zero = [
    "000620",
    "000692",
    "000796",
    "002021",
    "002157",
    "002482",
    "002564",
    "300010",
    "600589",
    "600671",
    "603030",
    "688520",
]
print("\n=== Previously zero-coverage stocks (2023 roe) ===")
for s in prev_zero:
    sub = v3_2023[v3_2023["symbol"] == s]
    if len(sub) > 0:
        roe_na = sub["roe"].isna().mean() * 100
        print(f"  {s}: {len(sub)} rows, roe NaN={roe_na:.1f}%")
    else:
        print(f"  {s}: NO 2023 rows in v3")

# Overall summary
print("\n=== Summary ===")
for c in fina_cols:
    na = v3[c].isna().mean() * 100
    print(f"  {c:20s}: {na:.1f}% NaN (overall)")
