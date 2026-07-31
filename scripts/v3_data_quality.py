#!/usr/bin/env python3
"""V3 Panel Data Quality Check after CYQ derivation."""

import numpy as np
import pandas as pd

df = pd.read_parquet("data/panel_full_enriched_v3.parquet")
n = len(df)
print(f"V3 Panel: {n:,} rows, {len(df.columns)} cols")
print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
print(f"Symbols: {df['symbol'].nunique():,}")
print()

# ── 1. CYQ fill rates ──
print("=" * 70)
print("1. CYQ Column Fill Rates")
print("=" * 70)
cyq = [
    "benefit_part",
    "avg_cost",
    "weight_avg",
    "cost_5pct",
    "cost_15pct",
    "cost_50pct",
    "cost_85pct",
    "cost_95pct",
    "pct_70_low",
    "pct_70_high",
    "pct_70_con",
    "pct_90_low",
    "pct_90_high",
    "pct_90_con",
]
for c in cyq:
    if c in df.columns:
        nn = df[c].notna().sum()
        print(f"  {c:<15s}  NaN={df[c].isna().mean() * 100:5.2f}%  non_null={nn:,}")
print()

# ── 2. OHLCV integrity ──
print("=" * 70)
print("2. OHLCV Integrity")
print("=" * 70)
checks = [
    ("high >= low", df["high"] >= df["low"]),
    ("high >= open", df["high"] >= df["open"]),
    ("high >= close", df["high"] >= df["close"]),
    ("low <= open", df["low"] <= df["open"]),
    ("low <= close", df["low"] <= df["close"]),
    ("volume >= 0", df["volume"] >= 0),
]
for name, mask in checks:
    violations = (~mask & mask.notna()).sum()
    status = "PASS" if violations == 0 else f"FAIL ({violations:,})"
    print(f"  {name:<20s}  {status}")
print()

# ── 3. Duplicate (symbol, date) ──
print("=" * 70)
print("3. Duplicate (symbol, date) Pairs")
print("=" * 70)
dups = df.duplicated(subset=["symbol", "date"]).sum()
print(f"  Duplicates: {dups}")
print()

# ── 4. Infinity values ──
print("=" * 70)
print("4. Infinity Values")
print("=" * 70)
num_cols = df.select_dtypes(include=[np.number]).columns
inf_count = 0
for c in num_cols:
    cinf = np.isinf(df[c]).sum()
    if cinf > 0:
        print(f"  {c}: {cinf} inf")
        inf_count += cinf
if inf_count == 0:
    print("  PASS — no infinity values")
print()

# ── 5. CYQ value sanity ──
print("=" * 70)
print("5. CYQ Value Sanity Checks")
print("=" * 70)

# 5a. cost_5pct <= cost_15pct <= cost_50pct <= cost_85pct <= cost_95pct
valid = df.dropna(
    subset=["cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct"]
)
n_valid = len(valid)
mono = (
    (valid["cost_5pct"] <= valid["cost_15pct"])
    & (valid["cost_15pct"] <= valid["cost_50pct"])
    & (valid["cost_50pct"] <= valid["cost_85pct"])
    & (valid["cost_85pct"] <= valid["cost_95pct"])
)
violations = (~mono).sum()
print(
    f"  cost_5 ≤ 15 ≤ 50 ≤ 85 ≤ 95:  {'PASS' if violations == 0 else f'FAIL ({violations:,}/{n_valid:,})'}"
)

# 5b. benefit_part in [0, 1] (or [0, 100] if Tushare winner_rate stored directly)
bp_valid = df["benefit_part"].dropna()
bp_oob_01 = ((bp_valid < 0) | (bp_valid > 1)).sum()
bp_oob_100 = ((bp_valid < 0) | (bp_valid > 100)).sum()
bp_min, bp_max = bp_valid.min(), bp_valid.max()
if bp_oob_01 == 0:
    print(f"  benefit_part in [0,1]:       PASS  (range: {bp_min:.4f} ~ {bp_max:.4f})")
elif bp_oob_100 == 0:
    print(f"  benefit_part in [0,100]:    PASS  (range: {bp_min:.2f} ~ {bp_max:.2f})")
    print(
        "    ⚠ benefit_part is [0,100] not [0,1] — fetch_cyq_remaining stored winner_rate directly"
    )
else:
    print(f"  benefit_part range:          FAIL  (range: {bp_min:.2f} ~ {bp_max:.2f})")

# 5c. pct_70_con in [0, 1]
p70 = df["pct_70_con"].dropna()
p70_oob = ((p70 < 0) | (p70 > 1)).sum()
print(
    f"  pct_70_con in [0,1]:         {'PASS' if p70_oob == 0 else f'FAIL ({p70_oob:,})'}"
)

# 5d. pct_90_con in [0, 1]
p90 = df["pct_90_con"].dropna()
p90_oob = ((p90 < 0) | (p90 > 1)).sum()
print(
    f"  pct_90_con in [0,1]:         {'PASS' if p90_oob == 0 else f'FAIL ({p90_oob:,})'}"
)

# 5e. pct_70_low <= pct_70_high
valid70 = df.dropna(subset=["pct_70_low", "pct_70_high"])
v70 = (valid70["pct_70_low"] <= valid70["pct_70_high"]).sum()
v70_fail = len(valid70) - v70
print(
    f"  pct_70_low ≤ pct_70_high:    {'PASS' if v70_fail == 0 else f'FAIL ({v70_fail:,})'}"
)

# 5f. pct_90_low <= pct_90_high
valid90 = df.dropna(subset=["pct_90_low", "pct_90_high"])
v90 = (valid90["pct_90_low"] <= valid90["pct_90_high"]).sum()
v90_fail = len(valid90) - v90
print(
    f"  pct_90_low ≤ pct_90_high:    {'PASS' if v90_fail == 0 else f'FAIL ({v90_fail:,})'}"
)

# 5g. Derived columns match source
print()
print("  Derived column consistency (should be 100% exact):")
checks2 = [
    ("pct_70_low == cost_15pct", "pct_70_low", "cost_15pct"),
    ("pct_70_high == cost_85pct", "pct_70_high", "cost_85pct"),
    ("pct_90_low == cost_5pct", "pct_90_low", "cost_5pct"),
    ("pct_90_high == cost_95pct", "pct_90_high", "cost_95pct"),
]
for label, a, b in checks2:
    both = df.dropna(subset=[a, b])
    match = (both[a] == both[b]).mean() * 100
    max_diff = (both[a] - both[b]).abs().max()
    print(f"    {label:<30s}  match={match:.2f}%  max_diff={max_diff:.8f}")

# 5h. Concentration formula verification (on filled rows only)
filled = df["pct_70_con"].notna() & df["cost_85pct"].notna() & df["cost_15pct"].notna()
denom70 = (df.loc[filled, "pct_70_high"] + df.loc[filled, "pct_70_low"]).replace(
    0, np.nan
)
calc70 = (df.loc[filled, "pct_70_high"] - df.loc[filled, "pct_70_low"]) / denom70
match70 = (df.loc[filled, "pct_70_con"] == calc70).mean() * 100
print(
    f"    pct_70_con == formula       match={match70:.2f}%  (on {filled.sum():,} rows)"
)

filled90 = df["pct_90_con"].notna() & df["cost_95pct"].notna() & df["cost_5pct"].notna()
denom90 = (df.loc[filled90, "pct_90_high"] + df.loc[filled90, "pct_90_low"]).replace(
    0, np.nan
)
calc90 = (df.loc[filled90, "pct_90_high"] - df.loc[filled90, "pct_90_low"]) / denom90
match90 = (df.loc[filled90, "pct_90_con"] == calc90).mean() * 100
print(
    f"    pct_90_con == formula       match={match90:.2f}%  (on {filled90.sum():,} rows)"
)
print()

# ── 6. NaN summary by category ──
print("=" * 70)
print("6. NaN Summary by Category")
print("=" * 70)
categories = {
    "OHLCV": ["open", "high", "low", "close", "volume"],
    "HFQ": ["open_hfq", "high_hfq", "low_hfq", "close_hfq"],
    "CYQ (all 14)": cyq,
    "Bias": [c for c in df.columns if c.startswith("bias_")],
    "Valuation": ["pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv"],
}
for cat, cols in categories.items():
    present = [c for c in cols if c in df.columns]
    if present:
        avg_nan = df[present].isna().mean().mean() * 100
        worst = max(df[present].isna().mean() * 100)
        worst_col = present[df[present].isna().mean().argmax()]
        print(
            f"  {cat:<20s}  avg_nan={avg_nan:5.2f}%  worst={worst:.2f}% ({worst_col})"
        )
print()

# ── 7. Sample data ──
print("=" * 70)
print("7. Sample: 000001 last 3 rows CYQ")
print("=" * 70)
sample = df[df["symbol"] == "000001"].sort_values("date").tail(3)
show = [
    "date",
    "close",
    "benefit_part",
    "cost_5pct",
    "cost_15pct",
    "cost_50pct",
    "cost_85pct",
    "cost_95pct",
    "pct_70_con",
    "pct_90_con",
    "weight_avg",
]
print(sample[show].to_string(index=False))
