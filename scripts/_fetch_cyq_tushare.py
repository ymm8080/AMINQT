#!/usr/bin/env python3
"""从 Tushare cyq_perf 批量拉取筹码分布数据并填充 v3.
优化: 短超时(15s) + 单次重试 + 跳过失败 + 断点续传.
"""

import sys
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import logging  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
import tushare as ts  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("cyq")

V3_PATH = "data/panel_full_enriched_v3.parquet"
CACHE_PATH = "data/supply_cache/alt_data/cyq_tushare/cyq_full.parquet"


def fetch_one(pro, ts_code, start_date, end_date):
    """单次拉取, 15s 超时, 失败重试1次."""
    for attempt in range(2):
        try:
            raw = pro.cyq_perf(
                ts_code=ts_code, start_date=start_date, end_date=end_date, timeout=15
            )
            if raw is not None and len(raw):
                return raw
            return None
        except Exception:
            if attempt == 0:
                time.sleep(3)  # 等3s重试一次
            else:
                raise


def main():
    t0 = time.time()
    token = os.getenv("TUSHARE_TOKEN")
    ts.set_token(token)
    pro = ts.pro_api()

    logger.info("=" * 60)
    logger.info("批量拉取 cyq_perf (优化版: 15s超时, 1次重试)")
    logger.info("=" * 60)

    # 1. 加载 v3
    logger.info("加载 v3...")
    v3 = pd.read_parquet(V3_PATH)
    symbols = sorted(v3["symbol"].unique().tolist())
    start_date = v3["date"].min().strftime("%Y%m%d")
    end_date = v3["date"].max().strftime("%Y%m%d")
    logger.info(
        "v3: %d 行, %d 股, %s ~ %s", len(v3), len(symbols), start_date, end_date
    )

    # 2. 断点续传
    cached = pd.DataFrame()
    done_symbols = set()
    if os.path.exists(CACHE_PATH):
        cached = pd.read_parquet(CACHE_PATH)
        done_symbols = set(cached["symbol"].unique())
        logger.info("缓存: %d 行, %d 股已完成", len(cached), len(done_symbols))

    remaining = [s for s in symbols if s not in done_symbols]
    logger.info("待拉取: %d 股", len(remaining))
    if not remaining:
        logger.info("全部已完成, 直接 merge")
    else:
        frames = [cached] if len(cached) else []
        fail_count = 0
        for i, sym in enumerate(remaining):
            ts_code = f"{sym}.{'SZ' if sym.startswith(('0', '3', '1')) else 'SH'}"
            try:
                raw = fetch_one(pro, ts_code, start_date, end_date)
                if raw is not None and len(raw):
                    out = pd.DataFrame(
                        {
                            "symbol": sym,
                            "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d"),
                            "winner_ratio": pd.to_numeric(
                                raw["winner_rate"], errors="coerce"
                            ),
                            "avg_cost": pd.to_numeric(
                                raw["weight_avg"], errors="coerce"
                            ),
                            "cost_5pct": pd.to_numeric(
                                raw["cost_5pct"], errors="coerce"
                            ),
                            "cost_15pct": pd.to_numeric(
                                raw["cost_15pct"], errors="coerce"
                            ),
                            "cost_50pct": pd.to_numeric(
                                raw["cost_50pct"], errors="coerce"
                            ),
                            "cost_85pct": pd.to_numeric(
                                raw["cost_85pct"], errors="coerce"
                            ),
                            "cost_95pct": pd.to_numeric(
                                raw["cost_95pct"], errors="coerce"
                            ),
                        }
                    )
                    frames.append(out)
            except Exception as e:
                fail_count += 1
                logger.debug("  跳过 %s: %s", sym, str(e)[:50])

            # 每 100 股保存 + 报告
            if (i + 1) % 100 == 0:
                os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
                pd.concat(frames, ignore_index=True).to_parquet(CACHE_PATH, index=False)
                logger.info(
                    "  进度: %d/%d (%.0f%%), 失败%d",
                    i + 1,
                    len(remaining),
                    (i + 1) / len(remaining) * 100,
                    fail_count,
                )

        # 保存最终缓存
        cyq = pd.concat(frames, ignore_index=True)
        cyq = cyq.drop_duplicates(subset=["symbol", "date"])
        cyq = cyq.sort_values(["symbol", "date"]).reset_index(drop=True)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        cyq.to_parquet(CACHE_PATH, index=False)
        logger.info(
            "拉取完成: %d 行, %d 股, 失败%d, 耗时 %.1f 分钟",
            len(cyq),
            cyq["symbol"].nunique(),
            fail_count,
            (time.time() - t0) / 60,
        )

    # 3. Merge 到 v3
    logger.info("加载 cyq 缓存...")
    cyq = pd.read_parquet(CACHE_PATH)
    logger.info("cyq: %d 行, %d 股", len(cyq), cyq["symbol"].nunique())

    cyq_cols = [
        "winner_ratio",
        "avg_cost",
        "cost_5pct",
        "cost_15pct",
        "cost_50pct",
        "cost_85pct",
        "cost_95pct",
    ]
    old = [c for c in cyq_cols if c in v3.columns]
    if old:
        v3 = v3.drop(columns=old)
        logger.info("删除 %d 列旧 cyq 数据", len(old))

    data_cols = [c for c in cyq.columns if c not in ("symbol", "date")]
    v3 = v3.merge(
        cyq[["symbol", "date"] + data_cols], on=["symbol", "date"], how="left"
    )
    logger.info("merge 完成: %d 列", len(data_cols))

    logger.info("保存 v3...")
    v3.to_parquet(V3_PATH, index=False)
    logger.info("完成: %s (%d 行 %d 列)", V3_PATH, len(v3), len(v3.columns))
    logger.info("总耗时: %.1f 分钟", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
