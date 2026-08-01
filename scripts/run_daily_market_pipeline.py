#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline 1: 每日 16:00 收盘后行情数据拉取.

交易日 15:00 收盘后, Tushare 数据约 15:30~16:00 完成结算并可用.
本脚本在 16:00 定时触发, 拉取当日全市场行情数据:

  - OHLCV 日线 (pro.daily)
  - 每日估值 (pro.daily_basic: PE/PB/换手/市值)
  - 涨跌停价 (pro.stk_limit)
  - 融资融券 (pro.margin_detail)
  - 龙虎榜 (pro.top_list / pro.top_inst)
  - 申万行业指数 (pro.sw_daily)
  - 筹码分布 (pro.cyq_perf)

公告类数据 (fina_indicator / holdertrade / holdernumber / anns_d)
由 08:00 Pipeline 2 单独拉取, 此处不涉及.

各源独立失败不阻断, 结果写入 data/supply_cache/ (parquet, WORM).
日志写入 data/daily_market_log_YYYYMMDD.md.

用法:
    python scripts/run_daily_market_pipeline.py
    python scripts/run_daily_market_pipeline.py --date 20260729
    python scripts/run_daily_market_pipeline.py --refresh
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.pipeline1.data_supply import DataSupplyChain, _with_timeout  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daily_market_pipeline")

# 拉取的数据源列表 — v3 面板所有列, 除公告类 (fina_indicator/holdertrade/holdernumber/anns_d)
MARKET_SOURCES = [
    "ohlcv",
    "daily_basic",
    "stk_limit",
    "margin",
    # northbound 已移除 — 港交所 2024-08-20 停止日度披露
    "lhb",
    "sector_index",
    "cyq_tushare",
]

SOURCE_LABELS = {
    "ohlcv": "OHLCV 日线",
    "daily_basic": "每日估值 (PE/PB/换手/市值)",
    "stk_limit": "涨跌停价格",
    "margin": "融资融券",
    "lhb": "龙虎榜",
    "sector_index": "申万行业指数",
    "cyq_tushare": "筹码分布 (Tushare cyq_perf)",
}


def is_trade_date(trade_date: str) -> bool:
    """简单判断: 周末非交易日. 交易日历精确判断需 Tushare trade_cal."""
    dt = datetime.strptime(trade_date, "%Y%m%d")
    return dt.weekday() < 5  # 0=Mon ... 4=Fri


def _fetch_cyq_daily(supply: DataSupplyChain, trade_date: str, refresh: bool):
    """拉取当日全市场筹码分布 (cyq_perf).

    Tushare cyq_perf 支持按 trade_date 全量拉取, 一次返回所有股票.
    缓存到 data/supply_cache/alt_data/cyq_tushare/<trade_date>.parquet.
    """
    import pandas as pd

    path = supply._alt_cache_path("cyq_tushare", trade_date)
    if not refresh and os.path.exists(path):
        return pd.read_parquet(path)

    pro = supply._tushare_pro()
    if pro is None:
        raise RuntimeError("Tushare 不可用 (TUSHARE_TOKEN 未配置)")

    raw = pro.cyq_perf(trade_date=trade_date)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "symbol": raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", ""),
            "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d", errors="coerce"),
            "benefit_part": pd.to_numeric(
                raw.get("benefit_part", None), errors="coerce"
            ),
            "avg_cost": pd.to_numeric(raw.get("avg_cost", None), errors="coerce"),
            "pct_70_low": pd.to_numeric(raw.get("pct_70_low", None), errors="coerce"),
            "pct_70_high": pd.to_numeric(raw.get("pct_70_high", None), errors="coerce"),
            "pct_70_con": pd.to_numeric(raw.get("pct_70_con", None), errors="coerce"),
            "pct_90_low": pd.to_numeric(raw.get("pct_90_low", None), errors="coerce"),
            "pct_90_high": pd.to_numeric(raw.get("pct_90_high", None), errors="coerce"),
            "pct_90_con": pd.to_numeric(raw.get("pct_90_con", None), errors="coerce"),
            "cost_5pct": pd.to_numeric(raw.get("cost_5pct", None), errors="coerce"),
            "cost_15pct": pd.to_numeric(raw.get("cost_15pct", None), errors="coerce"),
            "cost_50pct": pd.to_numeric(raw.get("cost_50pct", None), errors="coerce"),
            "cost_85pct": pd.to_numeric(raw.get("cost_85pct", None), errors="coerce"),
            "cost_95pct": pd.to_numeric(raw.get("cost_95pct", None), errors="coerce"),
            "weight_avg": pd.to_numeric(raw.get("weight_avg", None), errors="coerce"),
        }
    )
    out.to_parquet(path, index=False)
    logger.info("cyq_perf: %d stocks", len(out))
    return out


def _fetch_sector_index_tushare(
    supply: DataSupplyChain, trade_date: str, refresh: bool
):
    """拉取当日申万行业指数 (Tushare sw_daily).

    一次 API 调用返回全部申万行业指数当日数据.
    缓存到 data/supply_cache/alt_data/sector_index/<trade_date>.parquet.
    """
    import pandas as pd

    path = supply._alt_cache_path("sector_index", trade_date)
    if not refresh and os.path.exists(path):
        return pd.read_parquet(path)

    pro = supply._tushare_pro()
    if pro is None:
        raise RuntimeError("Tushare 不可用 (TUSHARE_TOKEN 未配置)")

    raw = pro.sw_daily(trade_date=trade_date)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "index_code": raw["ts_code"].str.replace(".SI", ""),
            "index_name": raw.get("name", ""),
            "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d", errors="coerce"),
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("vol", None), errors="coerce"),
            "amount": pd.to_numeric(raw.get("amount", None), errors="coerce"),
            "ret_pct": pd.to_numeric(raw.get("pct_change", None), errors="coerce")
            / 100.0,
        }
    )
    out.to_parquet(path, index=False)
    logger.info("sw_daily: %d indices", len(out))
    return out


def fetch_market_data(trade_date: str, refresh: bool = False) -> dict:
    """拉取当日全市场行情数据.

    Args:
        trade_date: 'YYYYMMDD'
        refresh: 强制刷新缓存

    Returns:
        {source: {"rows": int, "status": "ok"|"fail"|"skip", "msg": str}}
    """
    supply = DataSupplyChain()
    results: dict[str, dict] = {}

    for src in MARKET_SOURCES:
        t0 = time.time()
        try:
            if src == "ohlcv":
                df = supply._tushare_fetch_daily(trade_date)
            elif src == "daily_basic":
                df = supply.fetch_daily_basic(trade_date=trade_date, refresh=refresh)
            elif src == "stk_limit":
                df = supply.fetch_stk_limit(trade_date=trade_date, refresh=refresh)
            elif src == "margin":
                df = supply.fetch_margin(trade_date=trade_date, refresh=refresh)
            elif src == "lhb":
                df = supply.fetch_lhb(trade_date=trade_date, refresh=refresh)
            elif src == "sector_index":
                df = _fetch_sector_index_tushare(supply, trade_date, refresh)
            elif src == "cyq_tushare":
                df = _with_timeout(
                    lambda: _fetch_cyq_daily(supply, trade_date, refresh),
                    timeout=120,
                )
            else:
                continue

            rows = len(df) if df is not None else 0
            elapsed = time.time() - t0
            results[src] = {
                "rows": rows,
                "status": "ok" if rows > 0 else "empty",
                "msg": f"{rows} rows in {elapsed:.1f}s",
            }
            logger.info(
                "  %s (%s): %d rows (%.1fs)",
                src,
                SOURCE_LABELS.get(src, ""),
                rows,
                elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            results[src] = {
                "rows": 0,
                "status": "fail",
                "msg": f"{exc} ({elapsed:.1f}s)",
            }
            logger.warning(
                "  %s (%s): FAIL %s (%.1fs)",
                src,
                SOURCE_LABELS.get(src, ""),
                exc,
                elapsed,
            )

    return results


def write_log(trade_date: str, results: dict, elapsed_total: float) -> None:
    """写日志到 data/daily_market_log_YYYYMMDD.md (WORM: 不覆盖)."""
    log_path = ROOT / "data" / f"daily_market_log_{trade_date}.md"
    lines = [
        f"# Daily Market Pipeline - {trade_date}",
        "",
        f"- **Trigger**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total elapsed**: {elapsed_total:.1f}s",
        "",
        "| Source | Description | Rows | Status | Note |",
        "|--------|-------------|------|--------|------|",
    ]
    for src in MARKET_SOURCES:
        r = results.get(src, {})
        label = SOURCE_LABELS.get(src, src)
        lines.append(
            f"| {src} | {label} | {r.get('rows', 0)} | "
            f"{r.get('status', 'N/A')} | {r.get('msg', '')} |"
        )
    lines.append("")

    # WORM: 已存在则追加, 不覆盖
    "a" if log_path.exists() else "w"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Log written: %s", log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline 1: 16:00 daily market data")
    parser.add_argument("--date", help="Trade date YYYYMMDD (default: today)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cache")
    args = parser.parse_args()

    trade_date = args.date or datetime.now().strftime("%Y%m%d")

    if not is_trade_date(trade_date):
        logger.info("%s is not a trade date (weekend), skipping.", trade_date)
        return 0

    logger.info("=" * 60)
    logger.info("Pipeline 1: Daily Market Data")
    logger.info("Trade date: %s | Refresh: %s", trade_date, args.refresh)
    logger.info("=" * 60)

    t0 = time.time()
    results = fetch_market_data(trade_date, refresh=args.refresh)
    elapsed_total = time.time() - t0

    # Summary
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    fail = sum(1 for r in results.values() if r["status"] == "fail")
    empty = sum(1 for r in results.values() if r["status"] == "empty")
    logger.info("")
    logger.info(
        "Summary: %d ok, %d empty, %d fail (total %.1fs)",
        ok,
        empty,
        fail,
        elapsed_total,
    )

    write_log(trade_date, results, elapsed_total)

    return 1 if fail == len(MARKET_SOURCES) else 0


if __name__ == "__main__":
    raise SystemExit(main())
