#!/usr/bin/env python3
"""从 Tushare cyq_perf 批量拉取筹码分布数据并填充 v3."""
import sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv
load_dotenv()
import tushare as ts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cyq")

V3_PATH = "data/panel_full_enriched_v3.parquet"
CACHE_PATH = "data/supply_cache/alt_data/cyq_tushare/cyq_full.parquet"

def main():
    t0 = time.time()
    token = os.getenv("TUSHARE_TOKEN")
    ts.set_token(token)
    pro = ts.pro_api()

    logger.info("=" * 60)
    logger.info("批量拉取 cyq_perf 筹码分布数据")
    logger.info("=" * 60)

    # 1. 加载 v3, 获取股票列表和日期范围
    logger.info("加载 v3...")
    v3 = pd.read_parquet(V3_PATH)
    symbols = sorted(v3["symbol"].unique().tolist())
    start_date = v3["date"].min().strftime("%Y%m%d")
    end_date = v3["date"].max().strftime("%Y%m%d")
    logger.info("v3: %d 行, %d 股, %s ~ %s", len(v3), len(symbols), start_date, end_date)

    # 2. 检查缓存 (支持断点续传)
    cached = pd.DataFrame()
    done_symbols = set()
    if os.path.exists(CACHE_PATH):
        cached = pd.read_parquet(CACHE_PATH)
        done_symbols = set(cached["symbol"].unique())
        logger.info("缓存: %d 行, %d 股已完成", len(cached), len(done_symbols))

    # 3. 批量拉取
    remaining = [s for s in symbols if s not in done_symbols]
    logger.info("待拉取: %d 股", len(remaining))

    frames = [cached] if len(cached) else []
    for i, sym in enumerate(remaining):
        ts_code = f"{sym}.{'SZ' if sym.startswith(('0', '3', '1')) else 'SH'}"
        success = False
        for attempt in range(3):
            try:
                raw = pro.cyq_perf(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if raw is not None and len(raw):
                    out = pd.DataFrame({
                        "symbol": sym,
                        "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d"),
                        "benefit_part": pd.to_numeric(raw["winner_rate"], errors="coerce"),
                        "avg_cost": pd.to_numeric(raw["weight_avg"], errors="coerce"),
                        "cost_5pct": pd.to_numeric(raw["cost_5pct"], errors="coerce"),
                        "cost_15pct": pd.to_numeric(raw["cost_15pct"], errors="coerce"),
                        "cost_50pct": pd.to_numeric(raw["cost_50pct"], errors="coerce"),
                        "cost_85pct": pd.to_numeric(raw["cost_85pct"], errors="coerce"),
                        "cost_95pct": pd.to_numeric(raw["cost_95pct"], errors="coerce"),
                    })
                    frames.append(out)
                success = True
                break
            except Exception as e:
                if "200" in str(e) and ("cyq_perf" in str(e) or "access" in str(e).lower()):
                    logger.info("  %s 限流, 等待10s重试 (%d/3)", sym, attempt + 1)
                    time.sleep(10)
                else:
                    logger.warning("  跳过 %s: %s", sym, str(e)[:60])
                    break
        if not success:
            logger.warning("  %s 重试3次仍失败, 跳过", sym)

        # 每 50 股保存一次 (断点续传)
        if (i + 1) % 50 == 0:
            logger.info("  进度: %d/%d (%.0f%%)", i + 1, len(remaining), (i + 1) / len(remaining) * 100)
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(CACHE_PATH, index=False)

    # 4. 合并保存缓存
    cyq = pd.concat(frames, ignore_index=True)
    cyq = cyq.drop_duplicates(subset=["symbol", "date"])
    cyq = cyq.sort_values(["symbol", "date"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cyq.to_parquet(CACHE_PATH, index=False)
    logger.info("拉取完成: %d 行 %d 列, %d 股, 耗时 %.1f 分钟",
                len(cyq), len(cyq.columns), cyq["symbol"].nunique(),
                (time.time() - t0) / 60)

    # 5. 删除 v3 中的旧 cyq 列
    cyq_cols = ["benefit_part", "avg_cost", "cost_5pct", "cost_15pct",
                "cost_50pct", "cost_85pct", "cost_95pct"]
    old = [c for c in cyq_cols if c in v3.columns]
    if old:
        v3 = v3.drop(columns=old)
        logger.info("删除 %d 列旧 cyq 数据", len(old))

    # 6. Merge 到 v3
    data_cols = [c for c in cyq.columns if c not in ("symbol", "date")]
    v3 = v3.merge(cyq[["symbol", "date"] + data_cols], on=["symbol", "date"], how="left")
    logger.info("merge 完成: %d 列", len(data_cols))

    # 7. 保存 v3
    logger.info("保存 v3...")
    v3.to_parquet(V3_PATH, index=False)
    logger.info("完成: %s (%d 行 %d 列)", V3_PATH, len(v3), len(v3.columns))
    logger.info("总耗时: %.1f 分钟", (time.time() - t0) / 60)

if __name__ == "__main__":
    main()