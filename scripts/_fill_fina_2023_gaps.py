"""Surgical fill of 4 fina fields gapped in 2023 from the 20260802 backfill cache.

Root cause: fetch_fina_2023.py FIELDS omitted roe_dt/ar_turn/inv_turn/ocf_to_or,
so V3 has ~3% 2023 coverage on roe_deducted/ar_turnover/inventory_turnover/ocf_to_or.
Today's backfill_20260802_full/ cache (3244 per-stock files) carries all 4 fields.

Only these 4 columns are touched, only NaN cells are filled (fillna), via
merge_asof backward on announce_date (PIT: never use report_period directly).
Existing 2024-26 values are preserved untouched.
"""

import logging
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fill_fina_gaps")

V3_PATH = os.getenv("FILL_V3_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
CACHE_DIR = "data/supply_cache/alt_data/fina_indicator/backfill_20260802_full"

# cache raw field -> v3 column (only the 4 gapped in 2023)
FILL_MAP = {
    "roe_dt": "roe_deducted",
    "ar_turn": "ar_turnover",
    "inv_turn": "inventory_turnover",
    "ocf_to_or": "ocf_to_or",
}


def main() -> None:
    t0 = time.time()
    if not os.path.isdir(CACHE_DIR):
        logger.error("缓存目录不存在: %s", CACHE_DIR)
        sys.exit(1)

    logger.info("加载 v3...")
    v3 = pd.read_parquet(V3_PATH)
    logger.info(
        "v3: %d 行 %d 列, %s ~ %s",
        len(v3),
        len(v3.columns),
        v3["date"].min(),
        v3["date"].max(),
    )

    logger.info("加载缓存...")
    frames = []
    for f in sorted(os.listdir(CACHE_DIR)):
        if not f.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(CACHE_DIR, f))
        if len(df):
            frames.append(df)
    fina = pd.concat(frames, ignore_index=True)
    logger.info("缓存: %d 行, %d 股", len(fina), fina["symbol"].nunique())

    fina = fina.rename(columns=FILL_MAP)
    fina["announce_date"] = pd.to_datetime(
        fina["ann_date"], format="%Y%m%d", errors="coerce"
    )
    fina = fina.dropna(subset=["announce_date"])
    keep = ["symbol", "announce_date"] + list(FILL_MAP.values())
    fina = fina[keep].drop_duplicates(subset=["symbol", "announce_date"], keep="last")

    # panel date -> datetime for merge_asof (handles int64 or already-datetime)
    v3["_date"] = pd.to_datetime(v3["date"], errors="coerce")
    fina["_date"] = fina["announce_date"]
    fina = fina.sort_values("_date").reset_index(drop=True)

    # before overwrite: WORM dated backup
    backup = V3_PATH.replace(
        ".parquet",
        "_prefina2023_{}.parquet".format(pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")),
    )
    shutil.copy2(V3_PATH, backup)
    logger.info("备份: %s", backup)

    for _raw, col in FILL_MAP.items():
        if col not in v3.columns:
            logger.info("  v3 无列 %s, 跳过", col)
            continue
        cand_col = f"_cand_{col}"
        sub = (
            fina[["symbol", "_date", col]]
            .dropna(subset=[col])
            .rename(columns={col: cand_col})
        )
        if not len(sub):
            logger.info("  %s 缓存无数据, 跳过", col)
            continue
        sub = sub.sort_values("_date")
        cand = pd.merge_asof(
            v3.sort_values("_date"),
            sub,
            on="_date",
            by="symbol",
            direction="backward",
        )[cand_col]
        n_fill = int((v3[col].isna() & cand.notna()).sum())
        before = float(v3[col].notna().mean() * 100)
        v3[col] = v3[col].fillna(cand)
        after = float(v3[col].notna().mean() * 100)
        logger.info(
            "  %s: fill %d 行, 覆盖率 %.2f%% -> %.2f%%",
            col,
            n_fill,
            before,
            after,
        )

    v3 = v3.drop(columns=["_date"])
    v3 = v3.drop(columns=[c for c in v3.columns if c.startswith("_cand_")])

    logger.info("保存 v3...")
    v3.to_parquet(V3_PATH, index=False)
    logger.info(
        "完成: %d 行 %d 列, %.1f 秒", len(v3), len(v3.columns), time.time() - t0
    )


if __name__ == "__main__":
    main()
