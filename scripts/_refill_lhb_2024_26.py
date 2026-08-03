# -*- coding: utf-8 -*-
"""Refill real LHB (龙虎榜) 2024-01-02 ~ 2026-07-27 into the PARQUET V3 panel.

Root cause of the all-zero rows: scripts/fetch_alt_bulk.py searched AKShare
columns 净买入额/买入金额/卖出金额 but the real columns are
龙虎榜净买额/龙虎榜买入额/龙虎榜卖出额 — mismatch -> scalar 0.0 written.

Eastmoney's datacenter-web endpoint intermittently throttles this IP for
minutes at a time (result:null), so this refill sources LHB from Tushare's
top_list interface instead — same underlying 龙虎榜 data, reliable per-day API.

This script:
  1. Fetches real LHB per trade date via pro.top_list.
  2. Caches the normalized result at the path fetch_tushare_bulk.py expects
     (data/supply_cache/alt_data/lhb/all_20240102_20260727.parquet).
  3. WORM-backups V3, then overwrites only currently-all-zero lhb cells
     with real values (unmatched -> NaN). 2026-07-29~31 real rows untouched.
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
import tushare as ts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("refill_lhb")

V3_PATH = os.getenv("FILL_V3_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
LHB_CACHE = "data/supply_cache/alt_data/lhb/all_20240102_20260727.parquet"
LHB_COLS = ["lhb_net_buy", "lhb_buy_amt", "lhb_sell_amt"]
START, END = "20240102", "20260727"
_STAGING = "data/supply_cache/alt_data/lhb/_months"


def _fetch_day(pro, ymd: str) -> pd.DataFrame:
    """Fetch top_list for one trade date, normalize to [symbol, date, lhb_*].

    Retries any transient error (read timeout, connection drop, rate-limit) up
    to 5x with 10-40s backoff; both Eastmoney and Tushare have been flaky.
    """
    last = None
    for attempt in range(1, 6):
        try:
            raw = pro.top_list(trade_date=ymd)
            break
        except Exception as exc:
            last = exc
            wait = 10 * attempt
            logger.warning("  %s 重试 %d/5: %s (%ds 后重试)", ymd, attempt, str(exc)[:120], wait)
            time.sleep(wait)
    else:
        raise last
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["symbol", "date"] + LHB_COLS)
    out = pd.DataFrame()
    out["symbol"] = raw["ts_code"].astype(str).str.slice(0, 6)
    out["date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d", errors="coerce")
    out["lhb_net_buy"] = pd.to_numeric(raw["net_amount"], errors="coerce")
    out["lhb_buy_amt"] = pd.to_numeric(raw["l_buy"], errors="coerce")
    out["lhb_sell_amt"] = pd.to_numeric(raw["l_sell"], errors="coerce")
    return out.dropna(subset=["symbol", "date"])


def fetch_real() -> pd.DataFrame:
    """Fetch per trade date via Tushare, cache incrementally, then concat."""
    if os.path.exists(LHB_CACHE):
        df = pd.read_parquet(LHB_CACHE)
        logger.info("缓存已存在: %s (%d 行)", LHB_CACHE, len(df))
        return df
    os.makedirs(_STAGING, exist_ok=True)
    pro = ts.pro_api(timeout=60)
    cal = pro.trade_cal(exchange="SSE", start_date=START, end_date=END)
    trading = sorted(cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).tolist())
    logger.info("交易日: %d 天 (%s ~ %s)", len(trading), trading[0], trading[-1])
    frames = []
    for i, ymd in enumerate(trading, 1):
        stage = os.path.join(_STAGING, f"{ymd}.parquet")
        if os.path.exists(stage):
            day = pd.read_parquet(stage)
        else:
            day = _fetch_day(pro, ymd)
            day.to_parquet(stage, index=False)
            time.sleep(0.4)
        if len(day):
            frames.append(day)
        if i % 100 == 0:
            logger.info("  %d/%d 天 (%s)", i, len(trading), ymd)
    if not frames:
        logger.error("无任何数据")
        return pd.DataFrame(columns=["symbol", "date"] + LHB_COLS)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    df.to_parquet(LHB_CACHE, index=False)
    logger.info("真实 LHB 已缓存: %s (%d 行)", LHB_CACHE, len(df))
    return df


def main() -> None:
    t0 = time.time()
    logger.info("STEP 1: fetch real LHB %s~%s", START, END)
    lhb = fetch_real()
    if len(lhb) == 0:
        logger.error("无真实 LHB 数据, 中止")
        sys.exit(1)
    lhb["date"] = pd.to_datetime(lhb["date"])

    logger.info("STEP 2: 加载 v3: %s", V3_PATH)
    v3 = pd.read_parquet(V3_PATH)
    v3["date"] = pd.to_datetime(v3["date"])
    logger.info("v3: %d 行 %d 列", len(v3), len(v3.columns))

    backup = V3_PATH.replace(".parquet", "_prelhb2024_{}.parquet".format(
        pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")))
    shutil.copy2(V3_PATH, backup)
    logger.info("备份: %s", backup)

    # 剔除 Tushare 数据异常: 单只股票单日龙虎榜买入/卖出 > 1000 亿为伪值
    n_art = int(((lhb[["lhb_buy_amt", "lhb_sell_amt"]].fillna(0).abs() > 1e11).any(axis=1)).sum())
    if n_art:
        logger.info("剔除不合理超大额 LHB (%d 行): %s", n_art, "buy/sell > 1e11")
        lhb = lhb[~((lhb[["lhb_buy_amt", "lhb_sell_amt"]].fillna(0).abs() > 1e11).any(axis=1))]

    zero_mask = v3[LHB_COLS].eq(0).all(axis=1)
    n_zero = int(zero_mask.sum())
    logger.info("待修复全零行: %d", n_zero)
    if n_zero == 0:
        logger.info("无全零行, 无需修复")
        return

    tmp = v3.loc[zero_mask, ["symbol", "date"]].merge(
        lhb, on=["symbol", "date"], how="left", suffixes=("", "_lhb")
    )

    # 匹配率过低说明缓存与面板错位, 禁止静默把全零行写成 NaN
    key = "lhb_net_buy_lhb" if "lhb_net_buy_lhb" in tmp.columns else "lhb_net_buy"
    matched = int(tmp[key].notna().sum())
    if matched < 0.5 * n_zero:
        logger.error("LHB 匹配率过低: %d/%d, 中止不改写面板", matched, n_zero)
        sys.exit(1)
    logger.info("匹配到真实 LHB: %d/%d 全零行", matched, n_zero)
    for c in LHB_COLS:
        cand = tmp[f"{c}_lhb"] if f"{c}_lhb" in tmp.columns else tmp[c]
        v3.loc[zero_mask, c] = cand.values
        n_real = int(v3.loc[zero_mask, c].ne(0).sum())
        logger.info("  %s: %d 行为真实值 (其余置 NaN)", c, n_real)

    v3.to_parquet(V3_PATH, index=False)
    logger.info("完成: %d 行 %d 列, %.1f 秒", len(v3), len(v3.columns), time.time() - t0)


if __name__ == "__main__":
    if "--fetch-only" in sys.argv:
        lhb = fetch_real()
        logger.info("fetch-only 完成: %d 行", len(lhb))
        sys.exit(0)
    main()
