"""Quick check of V3 panel structure."""

import pandas as pd
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

panel_path = "data/panel_full_enriched_v3.parquet"
df = pd.read_parquet(
    panel_path,
    columns=[
        "symbol",
        "date",
        "close",
        "pre_close",
        "pctChg",
        "margin_buy_amt",
        "short_sell_vol",
        "north_net_buy_sh",
        "north_net_buy_sz",
        "holder_count",
        "lhb_net_buy",
        "sh_net_change_sign",
    ],
)

print(
    f"Panel: {len(df)} rows, {df['symbol'].nunique()} symbols, {df['date'].nunique()} dates"
)
print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")

# Before coverage
print("\n=== BEFORE Coverage ===")
for c in [
    "pre_close",
    "pctChg",
    "margin_buy_amt",
    "short_sell_vol",
    "north_net_buy_sh",
    "north_net_buy_sz",
    "holder_count",
    "lhb_net_buy",
    "sh_net_change_sign",
]:
    pct = (1 - df[c].isna().mean()) * 100 if c in df.columns else 0
    print(f"  {c:<25s}: {pct:.1f}%")

# Verify pre_close formula
valid = df.dropna(subset=["close", "pctChg", "pre_close"])
if len(valid) > 0:
    computed = valid["close"] / (1 + valid["pctChg"] / 100)
    diff = (computed - valid["pre_close"]).abs()
    print(
        f"\npre_close formula check: {len(valid)} rows, max diff={diff.max():.6f}, mean diff={diff.mean():.8f}"
    )

# All trading dates
all_dates = sorted(df["date"].dropna().unique())
print(f"\nTotal trading dates: {len(all_dates)}")
print(f"First 3: {all_dates[:3]}")
print(f"Last 3: {all_dates[-3:]}")

# Save date list for the fetch script
import json

date_strs = [d.strftime("%Y%m%d") for d in all_dates]
with open("data/supply_cache/all_trading_dates.json", "w") as f:
    json.dump(date_strs, f)
print(
    f"\nSaved {len(date_strs)} trading dates to data/supply_cache/all_trading_dates.json"
)
