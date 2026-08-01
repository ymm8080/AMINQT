#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline 2: 每日 22:40 公告数据拉取 + V3 面板更新.

22:40 拉取当日及前日发布的公告类数据:
  - 财务指标 PIT (pro.fina_indicator, 含 ann_date)
  - 股东增减持 (pro.stk_holdertrade) → 聚合后更新 V3 面板今日行
  - 股东户数 (pro.stk_holdernumber)
  - 个股公告 (pro.anns_d, 需 5000+ 积分; 降级: AKShare)

各源独立失败不阻断, 结果写入 data/supply_cache/ (parquet, WORM).
holdertrade 额外更新 V3 面板 (panel_full_enriched_v3.parquet) 的今日行.
日志写入 data/announcement_log_YYYYMMDD.md.

执行顺序: 必须在 Pipeline 1 (_daily_fetch.py) 之后运行.

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

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402
from config.settings import data_others_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("announcement_pipeline")

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")

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
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """拉取公告类数据.

    Args:
        target_date: 'YYYYMMDD'
        refresh: 强制刷新缓存

    Returns:
        (results, dataframes)
    """
    supply = DataSupplyChain()
    results: dict[str, dict] = {}
    dataframes: dict[str, pd.DataFrame] = {}

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
            dataframes[src] = df if df is not None else pd.DataFrame()
            logger.info(
                "  %s (%s): %d rows (%.1fs)",
                src, SOURCE_LABELS.get(src, ""), rows, elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            results[src] = {
                "rows": 0,
                "status": "fail",
                "msg": f"{exc} ({elapsed:.1f}s)",
            }
            dataframes[src] = pd.DataFrame()
            logger.warning(
                "  %s (%s): FAIL %s (%.1fs)",
                src, SOURCE_LABELS.get(src, ""), exc, elapsed,
            )

    return results, dataframes


def _fetch_anns_d(supply: DataSupplyChain, trade_date: str, refresh: bool):
    """拉取 Tushare anns_d (个股每日公告)."""
    pro = supply._tushare_pro()
    if pro is None:
        raise RuntimeError("Tushare 不可用 (TUSHARE_TOKEN 未配置)")

    raw = pro.anns_d(trade_date=trade_date)
    if raw is None or len(raw) == 0:
        return raw if raw is not None else pd.DataFrame()

    path = supply._alt_cache_path("anns_d", trade_date)
    raw.to_parquet(path, index=False)
    return raw


def update_v3_panel_holdertrade(
    trade_date: str,
    holdertrade_df: pd.DataFrame,
) -> dict:
    """更新 V3 面板今日行: 用 holdertrade 公告数据覆盖 forward-fill 旧值.

    Args:
        trade_date: 'YYYYMMDD'
        holdertrade_df: fetch_holdertrade 返回的 DataFrame

    Returns:
        {"updated": int, "status": "ok"|"skip"|"fail", "msg": str}
    """
    if not os.path.exists(PANEL):
        msg = f"panel not found: {PANEL}"
        logger.warning("V3 panel update skipped: %s", msg)
        return {"updated": 0, "status": "skip", "msg": msg}

    today_ts = pd.Timestamp(trade_date)

    try:
        schema = pq.read_schema(PANEL)
    except Exception as exc:
        msg = f"failed to read panel schema: {exc}"
        logger.error("V3 panel update: %s", msg)
        return {"updated": 0, "status": "fail", "msg": msg}

    panel_cols = schema.names

    try:
        today_rows = pq.read_table(
            PANEL, filters=[("date", "=", today_ts)]
        ).to_pandas()
    except Exception as exc:
        msg = f"failed to read today's rows: {exc}"
        logger.error("V3 panel update: %s", msg)
        return {"updated": 0, "status": "fail", "msg": msg}

    if not len(today_rows):
        msg = "no today rows in panel (Pipeline 1 _daily_fetch.py not run yet?)"
        logger.warning("V3 panel update skipped: %s", msg)
        return {"updated": 0, "status": "skip", "msg": msg}

    logger.info("V3 panel today: %d stocks, %d cols", len(today_rows), len(panel_cols))

    if not len(holdertrade_df):
        msg = "no holdertrade data fetched"
        logger.info("V3 panel update skipped: %s", msg)
        return {"updated": 0, "status": "skip", "msg": msg}

    if "announce_date" not in holdertrade_df.columns:
        msg = "holdertrade data missing announce_date column"
        logger.warning("V3 panel update skipped: %s", msg)
        return {"updated": 0, "status": "skip", "msg": msg}

    # 聚合 by (symbol, announce_date) — 同 panel_builder.py 逻辑
    daily_net = (
        holdertrade_df.groupby(["symbol", "announce_date"])
        .agg(
            sh_net_change_sign=("sh_net_sign", "sum"),
            sh_change_amt_total=("sh_change_amt", "sum"),
        )
        .reset_index()
    )
    daily_net = daily_net.rename(columns={"announce_date": "date"})
    daily_net["date"] = pd.to_datetime(daily_net["date"])
    daily_today = daily_net[daily_net["date"] == today_ts].copy()

    if not len(daily_today):
        msg = "no holdertrade announced today (forward-filled values remain)"
        logger.info("V3 panel update: %s", msg)
        return {"updated": 0, "status": "skip", "msg": msg}

    logger.info(
        "Holdertrade today: %d stocks with announcements",
        daily_today["symbol"].nunique(),
    )

    update_map = daily_today.set_index("symbol")
    updated_count = 0

    for col in ["sh_net_change_sign", "sh_change_amt_total"]:
        if col in panel_cols and col in update_map.columns:
            mask = today_rows["symbol"].isin(update_map.index)
            today_rows.loc[mask, col] = today_rows.loc[mask, "symbol"].map(
                update_map[col]
            )
            updated_count = max(updated_count, int(mask.sum()))

    # 单条记录列 (当日最新一条)
    ht_today = holdertrade_df[
        pd.to_datetime(holdertrade_df["announce_date"]) == today_ts
    ].copy()
    if len(ht_today):
        latest_per_stock = (
            ht_today.sort_values("announce_date").groupby("symbol").last()
        )
        for col in ["sh_change_amt", "sh_change_vol", "sh_net_sign"]:
            if col in panel_cols and col in latest_per_stock.columns:
                mask = today_rows["symbol"].isin(latest_per_stock.index)
                today_rows.loc[mask, col] = today_rows.loc[mask, "symbol"].map(
                    latest_per_stock[col]
                )

    logger.info("V3 panel: updated %d stocks with holdertrade data", updated_count)

    # 类型对齐
    for field in schema:
        if field.name in today_rows.columns:
            try:
                today_rows[field.name] = today_rows[field.name].astype(
                    field.type.to_pandas_dtype()
                )
            except Exception:
                pass

    # 重写面板 (移除旧今日行, 写入更新后今日行)
    try:
        today_table = pa.Table.from_pandas(
            today_rows, schema=schema, preserve_index=False
        )

        pf = pq.ParquetFile(PANEL)
        tmp_path = PANEL + ".tmp"
        writer = pq.ParquetWriter(tmp_path, schema=schema)

        for rg_idx in range(pf.metadata.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            rg_df = rg.to_pandas()
            rg_df = rg_df[rg_df["date"] != today_ts]
            if len(rg_df):
                writer.write_table(
                    pa.Table.from_pandas(rg_df, schema=schema, preserve_index=False)
                )

        writer.write_table(today_table)
        writer.close()
        pf.close()

        if os.path.exists(PANEL):
            os.remove(PANEL)
        os.rename(tmp_path, PANEL)

        msg = f"updated {updated_count} stocks, panel rewritten OK"
        logger.info("V3 panel update: %s", msg)
        return {"updated": updated_count, "status": "ok", "msg": msg}

    except Exception as exc:
        msg = f"panel rewrite failed: {exc}"
        logger.error("V3 panel update: %s", msg)
        tmp_path = PANEL + ".tmp"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return {"updated": 0, "status": "fail", "msg": msg}


def write_log(
    target_date: str,
    results: dict,
    v3_result: dict | None,
    elapsed_total: float,
) -> None:
    """写日志到 data/announcement_log_YYYYMMDD.md."""
    log_path = Path(data_others_path("data")) / f"announcement_log_{target_date}.md"
    lines = [
        f"# Announcement Pipeline - {target_date}",
        "",
        f"- **Trigger**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Date range**: {(datetime.strptime(target_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')} ~ {target_date}",
        f"- **Total elapsed**: {elapsed_total:.1f}s",
        "",
        "## Fetch Results",
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
    lines.append("## V3 Panel Update (holdertrade)")
    lines.append("")
    if v3_result:
        lines.append(f"- **Status**: {v3_result.get('status', 'N/A')}")
        lines.append(f"- **Updated**: {v3_result.get('updated', 0)} stocks")
        lines.append(f"- **Note**: {v3_result.get('msg', '')}")
    else:
        lines.append("- **Status**: not attempted")

    lines.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Log written: %s", log_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline 2: 22:40 announcement data + V3 panel update"
    )
    parser.add_argument("--date", help="Target date YYYYMMDD (default: today)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cache")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y%m%d")

    logger.info("=" * 60)
    logger.info("Pipeline 2: Announcement Data + V3 Panel Update")
    logger.info("Target date: %s | Refresh: %s", target_date, args.refresh)
    logger.info("Panel: %s", PANEL)
    logger.info("=" * 60)

    t0 = time.time()

    results, dataframes = fetch_announcement_data(
        target_date, refresh=args.refresh
    )

    holdertrade_df = dataframes.get("stk_holdertrade", pd.DataFrame())
    v3_result = update_v3_panel_holdertrade(target_date, holdertrade_df)

    elapsed_total = time.time() - t0

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    fail = sum(1 for r in results.values() if r["status"] == "fail")
    empty = sum(1 for r in results.values() if r["status"] == "empty")
    logger.info("")
    logger.info(
        "Summary: %d ok, %d empty, %d fail | V3: %s (%d stocks) (total %.1fs)",
        ok, empty, fail,
        v3_result.get("status", "N/A"),
        v3_result.get("updated", 0),
        elapsed_total,
    )

    write_log(target_date, results, v3_result, elapsed_total)

    return 1 if fail == len(ANNOUNCEMENT_SOURCES) else 0


if __name__ == "__main__":
    raise SystemExit(main())
