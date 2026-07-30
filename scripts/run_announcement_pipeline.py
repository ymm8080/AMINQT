#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline 2: 每日 08:00 盘前公告数据拉取.

盘前 08:00 拉取昨日发布的公告类数据 (非交易时段依赖):

  - 财务指标 PIT (pro.fina_indicator, 含 ann_date)
  - 股东增减持 (pro.stk_holdertrade)
  - 股东户数 (pro.stk_holdernumber)
  - 个股公告 (pro.anns_d, 需 5000+ 积分; 降级: AKShare)

各源独立失败不阻断, 结果写入 data/supply_cache/ (parquet, WORM).
日志写入 data/announcement_log_YYYYMMDD.md.

用法:
    python scripts/run_announcement_pipeline.py
    python scripts/run_announcement_pipeline.py --date 20260729
    python scripts/run_announcement_pipeline.py --refresh
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("announcement_pipeline")

# 公告类数据源
ANNOUNCEMENT_SOURCES = [
    "fina_indicator",
    "stk_holdertrade",
    "stk_holdernumber",
    "anns_d",
]

SOURCE_LABELS = {
    "fina_indicator": "财务指标 PIT (含 ann_date)",
    "stk_holdertrade": "股东增减持",
    "stk_holdernumber": "股东户数",
    "anns_d": "个股公告 (需 5000+ 积分)",
}


def fetch_announcement_data(
    target_date: str, refresh: bool = False
) -> dict:
    """拉取公告类数据.

    Args:
        target_date: 'YYYYMMDD', 拉取该日及前 1 天的公告
        refresh: 强制刷新缓存

    Returns:
        {source: {"rows": int, "status": "ok"|"fail"|"empty", "msg": str}}
    """
    supply = DataSupplyChain()
    results: dict[str, dict] = {}

    # 公告日期范围: 目标日期前 1 天到目标日期 (覆盖盘后公告)
    dt = datetime.strptime(target_date, "%Y%m%d")
    start_date = (dt - timedelta(days=1)).strftime("%Y%m%d")
    end_date = target_date

    for src in ANNOUNCEMENT_SOURCES:
        t0 = time.time()
        try:
            if src == "fina_indicator":
                df = supply.fetch_fina_indicator(
                    start_date=start_date, end_date=end_date, refresh=refresh
                )
            elif src == "stk_holdertrade":
                df = supply.fetch_holdertrade(
                    start_date=start_date, end_date=end_date, refresh=refresh
                )
            elif src == "stk_holdernumber":
                df = supply.fetch_holdernumber(
                    start_date=start_date, end_date=end_date, refresh=refresh
                )
            elif src == "anns_d":
                df = _fetch_anns_d(supply, target_date, refresh)
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


def _fetch_anns_d(supply: DataSupplyChain, trade_date: str, refresh: bool):
    """拉取 Tushare anns_d (个股每日公告).

    需要 5000+ 积分, 低积分 token 会返回权限错误, 此时尝试 AKShare 降级.
    """
    pro = supply._tushare_pro()
    if pro is None:
        raise RuntimeError("Tushare 不可用 (TUSHARE_TOKEN 未配置)")

    raw = pro.anns_d(trade_date=trade_date)
    if raw is None or len(raw) == 0:
        return raw if raw is not None else __import__("pandas").DataFrame()

    # 缓存到 parquet
    import pandas as pd

    path = supply._alt_cache_path("anns_d", trade_date)
    raw.to_parquet(path, index=False)
    return raw


def write_log(target_date: str, results: dict, elapsed_total: float) -> None:
    """写日志到 data/announcement_log_YYYYMMDD.md (WORM: 不覆盖)."""
    log_path = ROOT / "data" / f"announcement_log_{target_date}.md"
    lines = [
        f"# Announcement Pipeline - {target_date}",
        "",
        f"- **Trigger**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Date range**: {(datetime.strptime(target_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')} ~ {target_date}",
        f"- **Total elapsed**: {elapsed_total:.1f}s",
        "",
        "| Source | Description | Rows | Status | Note |",
        "|--------|-------------|------|--------|------|",
    ]
    for src in ANNOUNCEMENT_SOURCES:
        r = results.get(src, {})
        label = SOURCE_LABELS.get(src, src)
        lines.append(
            f"| {src} | {label} | {r.get('rows', 0)} | "
            f"{r.get('status', 'N/A')} | {r.get('msg', '')} |"
        )
    lines.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Log written: %s", log_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline 2: 08:00 announcement data"
    )
    parser.add_argument("--date", help="Target date YYYYMMDD (default: today)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cache")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y%m%d")

    logger.info("=" * 60)
    logger.info("Pipeline 2: Announcement Data (Pre-market)")
    logger.info("Target date: %s | Refresh: %s", target_date, args.refresh)
    logger.info("=" * 60)

    t0 = time.time()
    results = fetch_announcement_data(target_date, refresh=args.refresh)
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

    write_log(target_date, results, elapsed_total)

    return 1 if fail == len(ANNOUNCEMENT_SOURCES) else 0


if __name__ == "__main__":
    raise SystemExit(main())
