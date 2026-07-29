#!/usr/bin/env python3
"""填充 fina_indicator: 季度数据, 用 announce_date forward-fill 到交易日."""

import sys
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
import logging  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fina_fill")

V3_PATH = "data/panel_full_enriched_v3.parquet"

# Tushare raw → normalized column mapping (per data_supply.py col_rename)
RAW_TO_NORM = {
    "ann_date": "announce_date",
    "end_date": "report_period",
    "ts_code": "_ts_code",
    "roe_dt": "roe_deducted",
    "np_margin": "net_margin",
    "netprofit_margin": "net_margin",
    "dt_eps_yoy": "eps_yoy",
    "or_yoy": "rev_yoy",
    "netprofit_yoy": "profit_yoy",
    "cf_sales": "op_cf_ratio",
    "debt_to_assets": "debt_ratio",
    "assets_turn": "asset_turnover",
    "ar_turn": "ar_turnover",
    "inv_turn": "inventory_turnover",
    # these already match, no rename needed:
    # roe, roa, gross_margin, current_ratio, ocf_to_or
}


def normalize_fina(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Tushare column names to v3 panel convention."""
    # Detect raw format: has 'ann_date' but not 'announce_date'
    if "ann_date" in df.columns and "announce_date" not in df.columns:
        df = df.rename(columns=RAW_TO_NORM)
        logger.info("  raw Tushare format → normalized (%d cols)", len(df.columns))
    else:
        # Already-normalized files may still have raw column names
        # that weren't caught by the ann_date check (e.g. per-stock files
        # with announce_date but also ts_code/end_date)
        rename_safe = {
            k: v
            for k, v in RAW_TO_NORM.items()
            if k in df.columns and v not in df.columns
        }
        if rename_safe:
            df = df.rename(columns=rename_safe)

    # Ensure symbol exists
    if "symbol" not in df.columns and "_ts_code" in df.columns:
        df["symbol"] = df["_ts_code"].str.replace(".SZ", "").str.replace(".SH", "")

    # Convert announce_date to datetime (may be YYYYMMDD string or ISO date)
    if "announce_date" in df.columns:
        df["announce_date"] = pd.to_datetime(
            df["announce_date"], format="mixed", errors="coerce"
        )

    # Convert report_period similarly
    if "report_period" in df.columns:
        df["report_period"] = pd.to_datetime(
            df["report_period"], format="mixed", errors="coerce"
        )

    # Ensure numeric columns (skip metadata + leaked raw cols)
    skip_numeric = {
        "symbol",
        "_ts_code",
        "ts_code",
        "announce_date",
        "report_period",
        "end_date",
        "date",
    }
    for c in df.columns:
        if c not in skip_numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main():
    t0 = time.time()
    logger.info("加载 v3...")
    v3 = pd.read_parquet(V3_PATH)
    logger.info("v3: %d 行 %d 列", len(v3), len(v3.columns))

    # 加载 fina_indicator 缓存
    fina_dir = "data/supply_cache/alt_data/fina_indicator"
    if not os.path.isdir(fina_dir):
        logger.error("fina_indicator 缓存不存在")
        return

    files = sorted(f for f in os.listdir(fina_dir) if f.endswith(".parquet"))
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(os.path.join(fina_dir, f))
            if len(df) == 0:
                continue
            df = normalize_fina(df)
            if "symbol" in df.columns:
                frames.append(df)
        except Exception as e:
            logger.debug("  %s 跳过: %s", f, e)
    if not frames:
        logger.error("无 fina_indicator 数据")
        return

    fina = pd.concat(frames, ignore_index=True)
    # 合并: 同一 symbol + announce_date, 取各列首个非 NaN 值
    # (per-stock 文件有 roe/gross_margin 等, all_ 文件有 eps_yoy/net_margin 等)
    fina = fina.sort_values(["symbol", "announce_date"])
    # all_ 文件排在前面 (字母序), 其列优先; per-stock 文件补缺
    fina = fina.groupby(["symbol", "announce_date"], sort=False).first().reset_index()
    logger.info(
        "fina_indicator: %d 行 %d 列, %d 股",
        len(fina),
        len(fina.columns),
        fina["symbol"].nunique(),
    )

    # 用 announce_date 作为 date, 然后 forward-fill
    fina["date"] = pd.to_datetime(fina["announce_date"])
    fina = fina.dropna(subset=["date"])
    fina = fina.sort_values(["symbol", "date"])

    # 只保留需要的列
    skip = {
        "symbol",
        "_ts_code",
        "ts_code",
        "announce_date",
        "report_period",
        "end_date",
        "date",
    }
    data_cols = [c for c in fina.columns if c not in skip]
    logger.info("数据列: %s", data_cols)

    # 删除 v3 中已有的同名列 (避免冲突)
    old = [c for c in data_cols if c in v3.columns]
    if old:
        v3 = v3.drop(columns=old)
        logger.info("删除 %d 列旧数据", len(old))

    # merge_asof: 对每只股票, 找到 <= 交易日的最近一条财报
    fina_clean = fina[["symbol", "date"] + data_cols].drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    # merge_asof 要求 on 列全局排序
    v3_sorted = v3.sort_values("date").reset_index(drop=True)
    fina_sorted = fina_clean.sort_values("date").reset_index(drop=True)
    v3 = pd.merge_asof(
        v3_sorted,
        fina_sorted,
        on="date",
        by="symbol",
        direction="backward",
    )
    logger.info("merge_asof 完成")

    # 保存
    logger.info("保存 v3...")
    v3.to_parquet(V3_PATH, index=False)
    logger.info(
        "完成: %d 行 %d 列, 耗时 %.1f 秒", len(v3), len(v3.columns), time.time() - t0
    )


if __name__ == "__main__":
    main()
