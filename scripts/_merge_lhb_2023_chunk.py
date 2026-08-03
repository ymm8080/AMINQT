# -*- coding: utf-8 -*-
"""Merge the LHB 2023-06 chunk (cached, unmerged) into the V3 panel.

Panel lhb_net_buy/buy_amt/sell_amt cover 2024-01-02 ~ latest but have ZERO
2023 rows. The cache lhb/lhb_20230612_20230616.parquet (359 rows, 5 days)
carries that week. Surgical: symbol+date merge, coalesce only — 2024+
values untouched.
"""

import logging
import os
import shutil
import sys
import time
from datetime import date as _date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("merge_lhb_2023")

V3_PATH = os.getenv("FILL_V3_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
LHB_CACHE = "data/supply_cache/alt_data/lhb/lhb_20230612_20230616.parquet"
LHB_COLS = ["lhb_net_buy", "lhb_buy_amt", "lhb_sell_amt"]


def _find_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        v = df[c]
        if v.dropna().apply(lambda x: isinstance(x, _date)).all():
            return c
    raise ValueError("no datetime.date column found in lhb cache")


def main() -> None:
    t0 = time.time()
    logger.info("加载 v3: %s", V3_PATH)
    v3 = pd.read_parquet(V3_PATH)
    logger.info("v3: %d 行 %d 列", len(v3), len(v3.columns))

    lhb = pd.read_parquet(LHB_CACHE)
    dcol = _find_date_col(lhb)
    lhb["date"] = pd.to_datetime(lhb[dcol])
    keep = ["symbol", "date"] + [c for c in LHB_COLS if c in lhb.columns]
    lhb = lhb[keep].drop_duplicates(subset=["symbol", "date"], keep="last")
    lhb["date"] = lhb["date"].dt.tz_localize(None)
    logger.info(
        "lhb chunk: %d 行, %s ~ %s", len(lhb), lhb["date"].min(), lhb["date"].max()
    )

    missing = [c for c in LHB_COLS if c not in v3.columns]
    if missing:
        logger.warning("v3 缺列: %s — 新增", missing)

    backup = V3_PATH.replace(
        ".parquet",
        "_prelhb2023_{}.parquet".format(pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")),
    )
    shutil.copy2(V3_PATH, backup)
    logger.info("备份: %s", backup)

    # merge candidate values with suffix, then coalesce into existing cols
    merged = v3.merge(lhb, on=["symbol", "date"], how="left", suffixes=("", "_lhb"))
    for c in LHB_COLS:
        cand = merged[f"{c}_lhb"] if f"{c}_lhb" in merged.columns else None
        if cand is None:
            logger.info("  %s: 缓存无此列, 跳过", c)
            continue
        n_fill = (
            int((v3[c].isna() & cand.notna()).sum())
            if c in v3.columns
            else int(cand.notna().sum())
        )
        if c not in v3.columns:
            v3[c] = cand
        else:
            v3[c] = v3[c].fillna(cand)
        after = float(v3[c].notna().mean() * 100)
        logger.info("  %s: fill %d 行, 覆盖率 %.2f%%", c, n_fill, after)
    merged = None  # free memory

    logger.info("保存 v3...")
    v3.to_parquet(V3_PATH, index=False)
    logger.info(
        "完成: %d 行 %d 列, %.1f 秒", len(v3), len(v3.columns), time.time() - t0
    )


if __name__ == "__main__":
    main()
