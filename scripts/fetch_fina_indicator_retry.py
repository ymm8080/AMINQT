"""Retry missing fina_indicator stocks with incremental checkpointing."""

import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd  # noqa: E402

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402

out_dir = "data/supply_cache/alt_data/fina_indicator"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/all_20230701_20260727.parquet"

fields = (
    "ts_code,ann_date,end_date,"
    "roe,roa,gross_margin,netprofit_margin,"
    "dt_eps_yoy,or_yoy,netprofit_yoy,"
    "debt_to_assets,current_ratio,assets_turn,"
    "ocfps,revenue_ps,bps,eps,dt_eps,roe_yoy,q_roe,q_ocf_to_sales"
)

# Load existing
if os.path.exists(out_path):
    existing = pd.read_parquet(out_path)
    done_ts = set(existing["ts_code"].unique())
    print(f"Existing: {len(existing)} rows, {len(done_ts)} stocks", flush=True)
else:
    existing = None
    done_ts = set()

# Get full list + find missing
panel = pd.read_parquet("data/panel_full_enriched.parquet", columns=["symbol"])
panel_symbols = set(panel["symbol"].unique())
supply = DataSupplyChain()
pro = supply._tushare_pro()
stocks = pro.stock_basic()
stocks["symbol"] = stocks["ts_code"].str.split(".").str[0]
ak = stocks[stocks["symbol"].isin(panel_symbols)].copy()
ak = ak.drop_duplicates("ts_code")
all_ts = set(ak["ts_code"].tolist())
missing = sorted(all_ts - done_ts)
total = len(missing)
print(f"Missing: {total} stocks", flush=True)
if not missing:
    print("All stocks fetched!", flush=True)
    sys.exit(0)

# Fetch sequentially, checkpoint every 200
start = time.time()
batch = []
checkpoint = done_ts.copy() if existing is not None else set()
errors = []

for i, ts in enumerate(missing):
    try:
        raw = pro.fina_indicator(
            ts_code=ts, start_date="20230701", end_date="20260727", fields=fields
        )
        if len(raw) > 0:
            raw["symbol"] = ts.split(".")[0]
            batch.append(raw)
            checkpoint.add(ts)
    except Exception as e:
        errors.append((ts, str(e)))

    # Checkpoint every 100 or at end
    if (i + 1) % 100 == 0 or i == total - 1:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (total - i - 1) / rate if rate > 0 else 0
        ok_r = sum(len(r) for r in batch)
        print(
            f"[{i + 1}/{total}] {len(batch)} OK, {ok_r} rows, "
            f"{len(errors)} err | {elapsed:.0f}s ETA {eta:.0f}s",
            flush=True,
        )

        # Save checkpoint
        if batch:
            new_df = pd.concat(batch, ignore_index=True)
            new_df = new_df.drop_duplicates(["ts_code", "ann_date", "end_date"])
            final = new_df
            if os.path.exists(out_path):
                old = pd.read_parquet(out_path)
                final = pd.concat([old, new_df], ignore_index=True)
                final = final.drop_duplicates(["ts_code", "ann_date", "end_date"])
            final = final.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
            final.to_parquet(out_path, index=False)
            batch = []  # clear batch for next checkpoint
            print(f"  -> saved {len(final)} rows", flush=True)

elapsed = time.time() - start
print(f"\nDone in {elapsed:.0f}s | {len(errors)} errors", flush=True)
if errors:
    for ts, e in errors[:10]:
        print(f"  {ts}: {e}", flush=True)
