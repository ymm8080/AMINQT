#!/usr/bin/env python3
"""批量拉取全A筹码分布数据 (Tushare cyq_perf).

Usage:
    python scripts/fetch_cyq_perf.py                          # 全部1042只
    python scripts/fetch_cyq_perf.py --symbols 000001,600519  # 指定股票
    python scripts/fetch_cyq_perf.py --resume                 # 断点续传
    python scripts/fetch_cyq_perf.py --dry-run                # 只显示范围不拉取
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()  # noqa: E402

from app.pipeline1.data_supply import DataSupplyChain, DataSupplyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = "data/cyq_perf_panel.parquet"
PROGRESS = "data/cyq_perf_progress.txt"  # 已完成的 symbol, 断点续传用
START_DATE = "20240101"
END_DATE = "20260727"


def load_symbols_from_panel(panel_path: str = "data/panel_3y.parquet") -> list[str]:
    panel = pd.read_parquet(panel_path, columns=["symbol"])
    return sorted(panel["symbol"].unique().tolist())


def load_done() -> set[str]:
    if not os.path.exists(PROGRESS):
        return set()
    with open(PROGRESS, "r") as f:
        return set(line.strip() for line in f if line.strip())


def save_done(symbol: str) -> None:
    with open(PROGRESS, "a") as f:
        f.write(f"{symbol}\n")


def main():
    parser = argparse.ArgumentParser(description="批量拉取 Tushare cyq_perf 筹码分布")
    parser.add_argument("--symbols", type=str, default=None,
                        help="逗号分隔的股票代码 (默认: 从 panel_3y.parquet 读取全量)")
    parser.add_argument("--start", type=str, default=START_DATE, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, default=END_DATE, help="截止日期 YYYYMMDD")
    parser.add_argument("--resume", action="store_true", help="断点续传 (跳过已完成的)")
    parser.add_argument("--dry-run", action="store_true", help="只显示范围不拉取")
    parser.add_argument("--throttle", type=float, default=0.35,
                        help="每股间隔秒数 (免费 token 限流 ~200次/分钟, 默认0.35s)")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = load_symbols_from_panel()

    done = load_done() if args.resume else set()
    todo = [s for s in symbols if s not in done]

    logger.info("=" * 60)
    logger.info("cyq_perf 批量拉取")
    logger.info(f"  股票总数: {len(symbols)}")
    logger.info(f"  已完成:   {len(done)}")
    logger.info(f"  待拉取:   {len(todo)}")
    logger.info(f"  时间范围: {args.start} ~ {args.end}")
    logger.info(f"  输出:     {OUTPUT}")
    logger.info(f"  限流:     {args.throttle}s/股 (预估 {len(todo) * args.throttle / 60:.0f} 分钟)")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("dry-run: 前10只: %s", todo[:10])
        return

    if not todo:
        logger.info("全部完成, 无需拉取")
        return

    chain = DataSupplyChain()
    frames = []

    # 加载已有数据 (续传模式)
    if args.resume and os.path.exists(OUTPUT):
        existing = pd.read_parquet(OUTPUT)
        frames.append(existing)
        logger.info(f"已加载现有 {len(existing)} 行")

    failed = []
    t0 = time.time()
    for i, sym in enumerate(todo):
        ts_code = f"{sym}.{'SZ' if sym.startswith(('0', '3', '1')) else 'SH'}"
        try:
            df = chain.fetch_chip_distribution(ts_code, args.start, args.end)
            if len(df):
                frames.append(df)
                save_done(sym)
            else:
                logger.warning("[%d/%d] %s: 无数据", i + 1, len(todo), sym)
        except DataSupplyError as exc:
            logger.warning("[%d/%d] %s: %s", i + 1, len(todo), sym, exc)
            failed.append(sym)

        if args.throttle and i < len(todo) - 1:
            time.sleep(args.throttle)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            eta = (len(todo) - i - 1) / rate
            logger.info(
                "进度: %d/%d (%.1f%%) | 速率 %.1f/min | ETA %.0fmin | 失败 %d",
                i + 1, len(todo), (i + 1) / len(todo) * 100,
                rate, eta, len(failed),
            )
            # 每50只存一次中间结果
            if frames:
                panel = pd.concat(frames, ignore_index=True)
                panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
                panel.to_parquet(OUTPUT, index=False)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("拉取完成: %d 成功 / %d 失败 | 耗时 %.0f 分钟",
                 len(todo) - len(failed), len(failed), elapsed / 60)

    if frames:
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
        panel.to_parquet(OUTPUT, index=False)
        logger.info(
            "已保存 %s: %d 行, %d 只股票, %s ~ %s",
            OUTPUT, len(panel), panel["symbol"].nunique(),
            panel["date"].min().strftime("%Y-%m-%d"),
            panel["date"].max().strftime("%Y-%m-%d"),
        )

    if failed:
        logger.warning("失败列表 (%d): %s", len(failed), ", ".join(failed[:20]))
        with open("data/cyq_perf_failed.txt", "w") as f:
            f.write("\n".join(failed))


if __name__ == "__main__":
    main()
