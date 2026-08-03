# -*- coding: utf-8 -*-
"""Backfill 2023-01 block-trade data into the V3 panel (4 bt_ cols) + raw cache.

Background: the raw cache `block_trade_full.parquet` and the panel bt_ columns
both start at 2023-02-01 — the backtest window's first month (2023-01-03 →
2023-01-31) had no block-trade events, so the dim33 EWMA features
(bt_act/disc/inst_abs/mv_ratio_ewma, half-life h=10) are dead there. Tushare
`block_trade` DOES have Jan 2023 data (verified: 01-05=298 / 01-16=267 / 01-31=137
rows), so the gap is a fetch-start choice, not a data limit.

This is a ONE-TIME historical backfill — deliberately NOT in `_daily_fetch.py`
(which only fetches the current day). It replicates the exact L1+L2 aggregation
from `_daily_fetch.py` §6.5 (faithful restore, no formula drift):
  - L1 noise filter (only excludes from aggregation): buyer==seller / same-broker
    with disc<-10% / vol<10万股 with |disc|<1%
  - L2: bt_count=有效单数, bt_disc_raw=max(0,-disc), bt_inst_absorb=any_inst*
    total_amt/daily_amt, bt_amt_ratio_float_mv=total_amt/circ_mv
Units: vol=万股, amount=万元, daily_amt=面板amount/10 (千元→万元), circ_mv=万元, close=元.

WORM: backs up the raw cache and the panel before writing. Panel writes must be
sequential — never run concurrently with any other panel-writing script.
"""

import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

import tushare as ts

load_dotenv()

from config.settings import PANEL_V3_PATH

PANEL = Path(PANEL_V3_PATH)
ROOT = Path(__file__).resolve().parent.parent
BT_CACHE = (
    ROOT
    / "data"
    / "supply_cache"
    / "alt_data"
    / "block_trade"
    / "block_trade_full.parquet"
)
BT_COLS = ["bt_count", "bt_disc_raw", "bt_inst_absorb", "bt_amt_ratio_float_mv"]
CACHE_KEEP = [
    "symbol",
    "date",
    "ts_code",
    "trade_date",
    "price",
    "vol",
    "amount",
    "buyer",
    "seller",
]
INST_KW = (
    "机构专用",
    "QFII",
    "合格境外",
    "社保",
    "养老",
    "资产管理",
    "资管",
    "保险",
    "信托",
)
START_D, END_D = "20230103", "20230131"
SLEEP_S = 0.4


def _broker(seat):
    s = str(seat)
    idx = s.find("证券")
    return s[idx : idx + 2] if idx >= 0 else ""


def _agg_day(bt_day: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Replicate `_daily_fetch.py` §6.5 for one day. Returns overlay df or empty."""
    _bt = bt_day.copy()
    snap2 = snap.copy()
    snap2["daily_amt"] = snap2["amount"] / 10.0  # 千元 → 万元
    _bt = _bt.merge(
        snap2[["symbol", "close", "daily_amt", "circ_mv"]], on=["symbol"], how="left"
    )
    _bt = _bt[_bt["close"].notna()].copy()
    if not len(_bt):
        return pd.DataFrame(columns=["symbol", "date"] + BT_COLS)

    same_broker = (_bt["buyer"].map(_broker) == _bt["seller"].map(_broker)) & (
        _bt["buyer"] != _bt["seller"]
    )
    _bt["discount"] = (_bt["price"] - _bt["close"]) / _bt["close"].replace(0, np.nan)
    _bt["is_noise"] = (
        (_bt["buyer"] == _bt["seller"])
        | (same_broker & (_bt["discount"] < -0.1))
        | ((_bt["vol"] < 10) & (_bt["discount"].abs() < 0.01))
    )
    _bt["is_inst_buyer"] = _bt["buyer"].map(lambda s: any(k in str(s) for k in INST_KW))
    v = _bt[~_bt["is_noise"]].copy()
    if not len(v):
        return pd.DataFrame(columns=["symbol", "date"] + BT_COLS)

    grp = v.groupby("symbol")
    total_amt = grp["amount"].sum()
    wavg = (v["price"] * v["vol"]).groupby(v["symbol"]).sum() / grp[
        "vol"
    ].sum().replace(0, np.nan)
    close = grp["close"].first()
    daily_amt = grp["daily_amt"].first()
    circ_mv = grp["circ_mv"].first()
    disc = (wavg - close) / close.replace(0, np.nan)
    any_inst = v.groupby("symbol")["is_inst_buyer"].max()
    out = pd.DataFrame(
        {
            "bt_count": grp.size(),
            "bt_disc_raw": (-disc).clip(lower=0),
            "bt_inst_absorb": any_inst * total_amt / daily_amt.replace(0, np.nan),
            "bt_amt_ratio_float_mv": total_amt / circ_mv.replace(0, np.nan),
        }
    ).reset_index()
    out["date"] = pd.Timestamp(_bt["trade_date"].iloc[0])
    return out[["symbol", "date"] + BT_COLS]


def main():
    token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
    if not token:
        print("FATAL: No Tushare token found.")
        return
    pro = ts.pro_api(token)

    if not PANEL.exists():
        print(f"FATAL: panel not found: {PANEL}")
        return

    # ── 1. Trading days in Jan 2023 = panel's dates in window ──
    pdate = pq.read_table(PANEL, columns=["date"]).to_pandas()
    pdate["date"] = pd.to_datetime(pdate["date"])
    days = sorted(
        pdate.loc[
            (pdate["date"] >= pd.Timestamp("2023-01-01"))
            & (pdate["date"] <= pd.Timestamp("2023-01-31")),
            "date",
        ]
        .dt.strftime("%Y%m%d")
        .unique()
    )
    print(f"Jan 2023 trading days in panel: {len(days)}  ({days[0]}..{days[-1]})")

    # ── 2. Fetch each day + aggregate (snap from panel OHLCV) ──
    panel_snap = pq.read_table(
        PANEL, columns=["symbol", "date", "close", "amount", "circ_mv"]
    ).to_pandas()
    panel_snap["date"] = pd.to_datetime(panel_snap["date"])
    panel_snap = panel_snap[
        (panel_snap["date"] >= pd.Timestamp("2023-01-01"))
        & (panel_snap["date"] <= pd.Timestamp("2023-01-31"))
    ]

    all_raw, overlays = [], []
    for dd in days:
        try:
            bt = pro.block_trade(trade_date=dd)
        except Exception as e:
            print(f"  {dd}: FETCH ERROR {type(e).__name__}: {e}")
            continue
        if not len(bt):
            continue
        bt["symbol"] = bt["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
        bt["date"] = pd.Timestamp(dd)
        bt = bt[[c for c in CACHE_KEEP if c in bt.columns]]
        all_raw.append(bt)
        snap = panel_snap[panel_snap["date"] == pd.Timestamp(dd)]
        ov = _agg_day(bt, snap)
        if len(ov):
            overlays.append(ov)
            print(f"  {dd}: raw={len(bt):>4}  agg(symbols)={len(ov):>4}")
        else:
            print(
                f"  {dd}: raw={len(bt):>4}  agg(symbols)=0 (all noise / no panel match)"
            )
        time.sleep(SLEEP_S)

    if not all_raw:
        print("FATAL: no data fetched for Jan 2023.")
        return

    # ── 3. Merge raw rows into cache (WORM backup, dedup keep=last, atomic) ──
    raw = pd.concat(all_raw, ignore_index=True)
    BT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(BT_CACHE) if BT_CACHE.exists() else pd.DataFrame()
    print(f"\nRaw cache: {len(old):,} rows -> +{len(raw)} Jan rows")
    if BT_CACHE.exists():
        bak = BT_CACHE.with_name(
            f"block_trade_full_prebt2023_01_{time.strftime('%Y%m%d_%H%M%S')}.parquet"
        )
        old.to_parquet(bak, index=False)
        print(f"Cache WORM backup: {bak}")
    merged = pd.concat([old, raw], ignore_index=True)
    merged = merged.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    merged.to_parquet(BT_CACHE, index=False)
    print(
        f"Raw cache now: {len(merged):,} rows | "
        f"range {merged['trade_date'].min()}..{merged['trade_date'].max()}"
    )

    # ── 4. WORM-backup panel, then overlay the 4 bt_ cols for Jan dates ──
    pfree = os.statvfs(PANEL.anchor or "/") if hasattr(os, "statvfs") else None
    if pfree:
        gb = pfree.f_bavail * pfree.f_frsize / 1e9
        print(f"\nPanel drive free: {gb:.1f} GB")
        if gb < 3:
            print("FATAL: <3GB free on panel drive — aborting before write.")
            return
    bak = PANEL.with_name(
        f"panel_full_enriched_v3_prebt2023_01_{time.strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    shutil.copy2(PANEL, bak)
    print(f"Panel WORM backup: {bak}")

    overlay = pd.concat(overlays, ignore_index=True)
    overlay["date"] = pd.to_datetime(overlay["date"])

    pf = pq.ParquetFile(PANEL)
    schema = pf.schema_arrow
    pf.close()
    full = pq.read_table(PANEL).to_pandas()
    print(f"Read full panel: {len(full):,} rows, {len(full.columns)} cols")

    full = full.merge(overlay, on=["symbol", "date"], how="left", suffixes=("", "_ov"))
    for c in BT_COLS:
        ovc = f"{c}_ov"
        full[c] = full[c].where(full[ovc].isna(), full[ovc])
    full = full.drop(columns=[f"{c}_ov" for c in BT_COLS])
    full = full[list(schema.names)]  # align order to schema
    print("Overlaid Jan bt_ values into panel.")

    tmp = PANEL.with_name(PANEL.name + ".tmp")
    table = pa.Table.from_pandas(full, schema=schema, preserve_index=False)
    pq.write_table(table, tmp, compression="snappy")
    del full, table
    os.replace(tmp, PANEL)
    print("Panel replaced (atomic).")

    # ── 5. Verify ──
    v = pq.read_table(PANEL, columns=["date"] + BT_COLS).to_pandas()
    v["date"] = pd.to_datetime(v["date"])
    jan = v["date"].between(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-31"))
    print("\nVerification (Jan 2023 bt_ coverage):")
    for c in BT_COLS:
        nn = v.loc[jan, c].notna().sum()
        print(f"  {c:24s} Jan non-null={nn:>6,}")
    total = v.loc[jan & v["bt_count"].notna()].groupby("date").size().sum()
    print(f"  Jan event-days total: {int(total):,}")
    print(f"  Panel bt_ non-null total (all dates): {v['bt_count'].notna().sum():,}")


if __name__ == "__main__":
    main()
