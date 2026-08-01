#!/usr/bin/env python3
"""Fetch remaining CYQ stocks from Tushare, then merge into V3 safely.

Step 1: Fetch 428 remaining stocks -> append to cyq_full.parquet
Step 2: Merge cyq_full.parquet into V3 -> write to .tmp -> rename (atomic)
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_env_path = ROOT / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import pandas as pd  # noqa: E402
import tushare as ts  # noqa: E402
import logging  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("cyq_resume")

V3_PATH = ROOT / "data" / "panel_full_enriched_v3.parquet"
CYQ_CACHE = (
    ROOT / "data" / "supply_cache" / "alt_data" / "cyq_tushare" / "cyq_full.parquet"
)
START_DATE = "20230101"
END_DATE = "20260728"
THROTTLE = 0.35


def fetch_one(pro, ts_code):
    for attempt in range(2):
        try:
            raw = pro.cyq_perf(
                ts_code=ts_code, start_date=START_DATE, end_date=END_DATE
            )
            if raw is not None and len(raw):
                return pd.DataFrame(
                    {
                        "symbol": ts_code.split(".")[0],
                        "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d"),
                        "benefit_part": pd.to_numeric(
                            raw["winner_rate"], errors="coerce"
                        ),
                        "avg_cost": pd.to_numeric(raw["weight_avg"], errors="coerce"),
                        "cost_5pct": pd.to_numeric(raw["cost_5pct"], errors="coerce"),
                        "cost_15pct": pd.to_numeric(raw["cost_15pct"], errors="coerce"),
                        "cost_50pct": pd.to_numeric(raw["cost_50pct"], errors="coerce"),
                        "cost_85pct": pd.to_numeric(raw["cost_85pct"], errors="coerce"),
                        "cost_95pct": pd.to_numeric(raw["cost_95pct"], errors="coerce"),
                    }
                )
            return None
        except Exception:
            if attempt == 0:
                time.sleep(3)
            else:
                raise


def main():
    t0 = time.time()

    # -- Load V3 stock list --
    logger.info("Loading V3...")
    v3_syms = pd.read_parquet(V3_PATH, columns=["symbol"])
    v3_symbols = sorted(v3_syms["symbol"].unique().tolist())
    logger.info("V3: %d stocks", len(v3_symbols))

    # -- Load existing CYQ cache --
    cached = pd.read_parquet(CYQ_CACHE)
    done_symbols = set(cached["symbol"].unique())
    logger.info("CYQ cache: %d stocks done", len(done_symbols))

    remaining = [s for s in v3_symbols if s not in done_symbols]
    logger.info("Remaining: %d stocks", len(remaining))

    if remaining:
        token = os.environ.get("TUSHARE_TOKEN")
        ts.set_token(token)
        pro = ts.pro_api()

        frames = [cached]
        fail_count = 0
        for i, sym in enumerate(remaining):
            ts_code = f"{sym}.{'SZ' if sym.startswith(('0', '3', '1')) else 'SH'}"
            try:
                df = fetch_one(pro, ts_code)
                if df is not None and len(df):
                    frames.append(df)
            except Exception as e:
                fail_count += 1
                logger.debug("  Skip %s: %s", sym, str(e)[:80])

            if THROTTLE and i < len(remaining) - 1:
                time.sleep(THROTTLE)

            if (i + 1) % 50 == 0:
                pd.concat(frames, ignore_index=True).drop_duplicates(
                    subset=["symbol", "date"]
                ).to_parquet(CYQ_CACHE, index=False)
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(remaining) - i - 1) / rate
                logger.info(
                    "  Progress: %d/%d (%.0f%%) | fail %d | ETA %.0fs",
                    i + 1,
                    len(remaining),
                    (i + 1) / len(remaining) * 100,
                    fail_count,
                    eta,
                )

        cyq_all = pd.concat(frames, ignore_index=True)
        cyq_all = cyq_all.drop_duplicates(subset=["symbol", "date"])
        cyq_all = cyq_all.sort_values(["symbol", "date"]).reset_index(drop=True)
        cyq_all.to_parquet(CYQ_CACHE, index=False)
        logger.info(
            "CYQ cache updated: %d rows, %d stocks, %d failed, %.1f min",
            len(cyq_all),
            cyq_all["symbol"].nunique(),
            fail_count,
            (time.time() - t0) / 60,
        )
    else:
        logger.info("All stocks done! Proceeding to merge.")

    # -- Step 2: Merge into V3 (safe write) --
    logger.info("=" * 50)
    logger.info("Merging CYQ into V3 (safe write: temp + rename)")
    logger.info("=" * 50)

    cyq = pd.read_parquet(CYQ_CACHE)
    logger.info(
        "CYQ: %d rows, %d stocks, %s ~ %s",
        len(cyq),
        cyq["symbol"].nunique(),
        cyq["date"].min(),
        cyq["date"].max(),
    )

    v3 = pd.read_parquet(V3_PATH)
    logger.info("V3: %d rows, %d cols", len(v3), len(v3.columns))

    cyq_cols = [
        "benefit_part",
        "avg_cost",
        "cost_5pct",
        "cost_15pct",
        "cost_50pct",
        "cost_85pct",
        "cost_95pct",
    ]
    old = [c for c in cyq_cols if c in v3.columns]
    if old:
        v3 = v3.drop(columns=old)
        logger.info("Dropped %d old CYQ columns", len(old))

    data_cols = [c for c in cyq.columns if c not in ("symbol", "date")]
    v3 = v3.merge(
        cyq[["symbol", "date"] + data_cols], on=["symbol", "date"], how="left"
    )
    logger.info("Merged %d columns", len(data_cols))

    for c in data_cols:
        nn = v3[c].notna().sum()
        logger.info("  %s: %d/%d (%.1f%%)", c, nn, len(v3), nn / len(v3) * 100)

    tmp_path = str(V3_PATH) + ".tmp"
    logger.info("Writing to temp: %s", tmp_path)
    v3.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, str(V3_PATH))
    logger.info("V3 saved: %d rows, %d cols", len(v3), len(v3.columns))
    logger.info("Total time: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
