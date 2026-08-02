#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3 面板数据获取管道 — 填充 panel_full_enriched_v3.parquet 所有列.

流程:
  1. 加载现有 v3 面板
  2. 对每个数据源, 拉取指定日期范围的新数据, merge 回面板
  3. 生成覆盖率报告
  4. 保存面板 (WORM: 先备份旧文件, 再写新文件)

用法:
    # 全量拉取 (默认最近 3 年)
    python scripts/data_fetch_pipeline.py

    # 指定日期范围
    python scripts/data_fetch_pipeline.py --start 20240101 --end 20260728

    # 仅拉取部分数据源
    python scripts/data_fetch_pipeline.py --sources daily_basic stk_limit

    # 强制刷新缓存
    python scripts/data_fetch_pipeline.py --refresh

    # 只做覆盖率报告, 不拉取
    python scripts/data_fetch_pipeline.py --report-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402
from app.pipeline1.panel_builder import enrich_alt_data  # noqa: E402
from config.settings import PANEL_V3_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data_fetch_pipeline")

V3_PATH = PANEL_V3_PATH

# 数据源分组 (用于 UI 显示)
SOURCE_GROUPS = {
    "daily_basic": "每日估值 (PE/PB/市值/换手)",
    "stk_limit": "涨跌停价格",
    "margin": "融资融券",
    "northbound": "北向资金",
    "lhb": "龙虎榜",
    "fina_indicator": "财务指标 PIT",
    "holdernumber": "股东户数",
    "holdertrade": "股东增减持",
    "sector_index": "申万行业指数",
    "cyq_tushare": "筹码分布 (Tushare)",
}

ALL_SOURCES = list(SOURCE_GROUPS.keys())

# 各数据源在 v3 中的 Marker 列 (用于判断是否已有数据)
SOURCE_MARKERS = {
    "northbound": "north_net_buy_sh",
    "margin": "margin_balance",
    "fina_indicator": "roe",
    "lhb": "lhb_net_buy",
    "holdernumber": "holder_count",
    "holdertrade": "sh_net_change_sign",
    "sector_index": "sw_ret_1d",
    "daily_basic": "pe_ttm",
    "stk_limit": "up_limit_raw",
    "cyq_tushare": "winner_ratio",
}

# 各数据源在 v3 中的完整列前缀 (用于清理旧数据再 merge)
SOURCE_COL_PREFIXES = {
    "northbound": ["north_"],
    "margin": ["margin_", "short_"],
    "fina_indicator": [
        "roe",
        "roe_deducted",
        "roa",
        "gross_margin",
        "net_margin",
        "eps_yoy",
        "rev_yoy",
        "profit_yoy",
        "op_cf_ratio",
        "debt_ratio",
        "current_ratio",
        "asset_turnover",
        "ar_turnover",
        "inventory_turnover",
        "ocf_to_or",
        "announce_date",
    ],
    "lhb": ["lhb_"],
    "holdernumber": ["holder_count", "avg_shares_per_holder"],
    "holdertrade": ["sh_"],
    "sector_index": ["sw_"],
    "daily_basic": [
        "turnover_rate_f",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_mv",
        "circ_mv",
        "total_share",
        "float_share",
        "free_share",
    ],
    "stk_limit": ["up_limit_raw", "down_limit_raw"],
    "cyq_tushare": [
        "winner_ratio",
        "avg_cost",
        "pct_70_low",
        "pct_70_high",
        "pct_70_con",
        "pct_90_low",
        "pct_90_high",
        "pct_90_con",
        "cost_5pct",
        "cost_15pct",
        "cost_50pct",
        "cost_85pct",
        "cost_95pct",
        "weight_avg",
    ],
}


def load_v3() -> pd.DataFrame:
    """加载现有 v3 面板."""
    logger.info("加载 v3 面板: %s", V3_PATH)
    panel = pd.read_parquet(V3_PATH)
    logger.info(
        "  形状: %s, %d 只股票, 日期: %s ~ %s",
        panel.shape,
        panel["symbol"].nunique(),
        panel["date"].min(),
        panel["date"].max(),
    )
    return panel


def save_v3(panel: pd.DataFrame) -> None:
    """保存 v3 面板 (WORM: 先备份旧文件)."""
    backup_path = V3_PATH.with_suffix(".backup.parquet")
    if V3_PATH.exists():
        logger.info("备份旧面板: %s", backup_path)
        V3_PATH.rename(backup_path)

    panel.to_parquet(V3_PATH, index=False)
    size_mb = V3_PATH.stat().st_size / 1024 / 1024
    logger.info(
        "保存完成: %s (%.1f MB, %d 行, %d 列)",
        V3_PATH,
        size_mb,
        len(panel),
        len(panel.columns),
    )


def drop_stale_source_columns(panel: pd.DataFrame, source: str) -> pd.DataFrame:
    """删除数据源对应的旧列, 准备重新 merge."""
    prefixes = SOURCE_COL_PREFIXES.get(source, [])
    if not prefixes:
        return panel

    cols_to_drop = []
    for col in panel.columns:
        if col in ("symbol", "date"):
            continue
        for prefix in prefixes:
            if col == prefix or col.startswith(prefix):
                cols_to_drop.append(col)
                break

    if cols_to_drop:
        logger.info("  删除 %d 列旧数据: %s", len(cols_to_drop), cols_to_drop[:5])
        panel = panel.drop(columns=cols_to_drop)
    return panel


def compute_coverage(panel: pd.DataFrame) -> dict:
    """计算各数据源在 v3 面板中的覆盖率."""
    report = {}
    for source, marker in SOURCE_MARKERS.items():
        if marker in panel.columns:
            non_na = panel[marker].notna().sum()
            total = len(panel)
            pct = non_na / total * 100
            report[source] = {
                "marker": marker,
                "non_na": non_na,
                "total": total,
                "coverage_pct": round(pct, 1),
                "has_data": pct > 0,
            }
        else:
            report[source] = {
                "marker": marker,
                "non_na": 0,
                "total": len(panel),
                "coverage_pct": 0.0,
                "has_data": False,
                "missing": True,
            }
    return report


def print_coverage_report(report: dict) -> None:
    """打印覆盖率报告."""
    logger.info("=" * 60)
    logger.info("覆盖率报告")
    logger.info("=" * 60)
    for source, info in report.items():
        label = SOURCE_GROUPS.get(source, source)
        if info.get("missing"):
            logger.info("  %-20s: 列缺失", label)
        else:
            logger.info(
                "  %-20s: %5.1f%% (%s / %s 行)",
                label,
                info["coverage_pct"],
                info["non_na"],
                info["total"],
            )


def fetch_source(
    panel: pd.DataFrame,
    supply: DataSupplyChain,
    source: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """拉取单个数据源并 merge 回面板."""
    logger.info("--- 拉取: %s (%s) ---", source, SOURCE_GROUPS.get(source, ""))
    panel = drop_stale_source_columns(panel, source)

    # 使用 panel_builder 的 enrich_alt_data 进行 merge
    panel = enrich_alt_data(
        panel,
        supply,
        sources=[source],
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )

    # 统计新列
    marker = SOURCE_MARKERS.get(source)
    if marker and marker in panel.columns:
        non_na = panel[marker].notna().sum()
        pct = non_na / len(panel) * 100
        logger.info("  %s 覆盖率: %.1f%% (%s/%s)", source, pct, non_na, len(panel))
    else:
        logger.warning("  %s: marker 列 %s 未找到", source, marker)

    return panel


def validate_ohlcv(panel: pd.DataFrame) -> list[str]:
    """OHLCV 数据完整性校验."""
    issues = []
    for col in ["open", "high", "low", "close", "volume"]:
        na_pct = panel[col].isna().mean() * 100
        if na_pct > 5:
            issues.append(f"{col}: {na_pct:.1f}% NaN")
    return issues


def run_pipeline(
    start_date: str | None = None,
    end_date: str | None = None,
    sources: list[str] | None = None,
    refresh: bool = False,
    report_only: bool = False,
    backup: bool = True,
) -> dict:
    """运行数据获取管道.

    Args:
        start_date: 起始日期 'YYYYMMDD' (默认从面板最早日期)
        end_date: 截止日期 'YYYYMMDD' (默认面板最晚日期)
        sources: 数据源列表 (默认全部)
        refresh: 强制刷新缓存
        report_only: 仅报告, 不拉取
        backup: 保存前备份旧文件

    Returns:
        包含 pipeline 结果的 dict
    """
    t0 = time.time()
    panel = load_v3()
    issues = validate_ohlcv(panel)
    if issues:
        logger.warning("OHLCV 数据异常: %s", issues)

    if start_date is None:
        start_date = panel["date"].min().strftime("%Y%m%d")
    if end_date is None:
        end_date = panel["date"].max().strftime("%Y%m%d")
    if sources is None:
        sources = ALL_SOURCES

    logger.info("日期范围: %s ~ %s", start_date, end_date)
    logger.info("数据源: %s", sources)
    logger.info("刷新缓存: %s", refresh)

    # 报告覆盖率 (拉取前)
    before_report = compute_coverage(panel)
    print_coverage_report(before_report)

    if report_only:
        return {
            "status": "report_only",
            "panel_shape": panel.shape,
            "symbols": panel["symbol"].nunique(),
            "date_range": (panel["date"].min(), panel["date"].max()),
            "coverage": before_report,
            "elapsed": time.time() - t0,
        }

    # 拉取数据
    supply = DataSupplyChain()
    for src in sources:
        try:
            panel = fetch_source(
                panel,
                supply,
                src,
                start_date=start_date,
                end_date=end_date,
                refresh=refresh,
            )
        except Exception as exc:
            logger.warning("数据源 %s 拉取失败 (不阻断): %s", src, exc)

    # 最终覆盖率报告
    after_report = compute_coverage(panel)
    logger.info("")
    logger.info("=" * 60)
    logger.info("拉取后覆盖率报告")
    logger.info("=" * 60)
    print_coverage_report(after_report)

    # 统计提升
    logger.info("")
    logger.info("覆盖率变化:")
    for src in sources:
        b = before_report.get(src, {})
        a = after_report.get(src, {})
        bp = b.get("coverage_pct", 0)
        ap = a.get("coverage_pct", 0)
        if ap > bp:
            logger.info("  %s: %.1f%% → %.1f%% (+%.1f%%)", src, bp, ap, ap - bp)

    # 保存
    if backup:
        save_v3(panel)

    elapsed = time.time() - t0
    logger.info("管道完成: %.1f 秒", elapsed)

    return {
        "status": "success",
        "panel_shape": panel.shape,
        "symbols": panel["symbol"].nunique(),
        "date_range": (panel["date"].min(), panel["date"].max()),
        "coverage": after_report,
        "coverage_before": before_report,
        "elapsed": elapsed,
        "sources_fetched": sources,
    }


def main():
    parser = argparse.ArgumentParser(description="v3 面板数据获取管道")
    parser.add_argument("--start", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", help="截止日期 YYYYMMDD")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_SOURCES,
        help="数据源列表 (默认全部)",
    )
    parser.add_argument("--refresh", action="store_true", help="强制刷新缓存")
    parser.add_argument("--report-only", action="store_true", help="仅报告, 不拉取")
    parser.add_argument("--no-backup", action="store_true", help="不备份旧文件")
    args = parser.parse_args()

    result = run_pipeline(
        start_date=args.start,
        end_date=args.end,
        sources=args.sources,
        refresh=args.refresh,
        report_only=args.report_only,
        backup=not args.no_backup,
    )

    print(f"\n管道状态: {result['status']}")
    print(
        f"面板: {result['panel_shape'][0]} 行, {result['panel_shape'][1]} 列, {result['symbols']} 只股票"
    )
    print(f"耗时: {result['elapsed']:.1f} 秒")


if __name__ == "__main__":
    main()
