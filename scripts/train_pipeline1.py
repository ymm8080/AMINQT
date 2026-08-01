# -*- coding: utf-8 -*-
"""Pipeline-1 训练入口 (周频重训)
=====================================================
用法:
    python scripts/train_pipeline1.py --symbols 600519 601318 300750
    python scripts/train_pipeline1.py --symbols-file config/universe.txt
    python scripts/train_pipeline1.py --symbols-file config/universe.txt --tag 2026W30

数据: 最近 3 年 akshare 日线 (用户 2026-07-26 裁决), 逐股缓存于
data/supply_cache, 面板缓存于 data/processed/panel_<end>_3y.parquet (WORM).
产物: models/pipeline1/{main,dual}_<tag>.pkl + OOS IC 报告 (stdout).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402
from app.pipeline1.panel_builder import assemble_panel, load_or_fetch_meta  # noqa: E402
from app.pipeline1.train_runner import run_training  # noqa: E402
from config.settings import data_others_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("train_pipeline1")


def load_symbols(args: argparse.Namespace) -> list[str]:
    """universe 来源: --symbols 直传 > --symbols-file (每行一个代码) > watchlist."""
    if args.symbols:
        return [s.strip() for s in args.symbols if s.strip()]
    if args.symbols_file and Path(args.symbols_file).exists():
        lines = Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
        return [s.strip() for s in lines if s.strip() and not s.startswith("#")]
    import json

    watchlist = Path(data_others_path("data/watchlist.json"))
    if watchlist.exists():
        symbols = json.loads(watchlist.read_text(encoding="utf-8"))
        if symbols:
            logger.info("universe 取自 data/watchlist.json (%d 只)", len(symbols))
            return [str(s) for s in symbols]
    raise SystemExit(
        "未指定 universe: 用 --symbols / --symbols-file, 或填充 data/watchlist.json"
    )


def main() -> dict:
    parser = argparse.ArgumentParser(description="Pipeline-1 周频训练")
    parser.add_argument("--symbols", nargs="*", default=None, help="股票代码列表")
    parser.add_argument(
        "--symbols-file", default=None, help="universe 文件 (每行一代码)"
    )
    parser.add_argument("--years", type=int, default=3, help="回看年数 (默认 3)")
    parser.add_argument("--end", default=None, help="数据截止 YYYY-MM-DD (默认今天)")
    parser.add_argument(
        "--tag", default=None, help="模型标签 (默认 ISO 周, 如 2026W30)"
    )
    parser.add_argument("--model-dir", default="models/pipeline1")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存强制重拉")
    parser.add_argument("--no-ic-screen", action="store_true", help="跳过 IC 筛选")
    args = parser.parse_args()

    symbols = load_symbols(args)
    tag = args.tag or time.strftime("%GW%V")  # 周频标签 (ISO 年+周)
    logger.info("universe=%d 只, years=%d, tag=%s", len(symbols), args.years, tag)

    supply = DataSupplyChain()
    try:
        industry_map, name_map = load_or_fetch_meta(refresh=args.refresh)
    except Exception as exc:
        logger.warning("元数据拉取失败 (%s), 行业/名称用默认值", exc)
        industry_map = name_map = None
    panel = assemble_panel(
        supply,
        symbols,
        end=args.end,
        years=args.years,
        refresh=args.refresh,
        industry_map=industry_map,
        name_map=name_map,
    )
    results = run_training(
        panel, tag, model_dir=args.model_dir, use_ic_screen=not args.no_ic_screen
    )
    for board, res in results.items():
        logger.info(
            "[%s] path=%s OOS=%s switched=%s",
            board,
            res["path"],
            res["oos"],
            res["switched"],
        )
    return results


if __name__ == "__main__":
    main()
