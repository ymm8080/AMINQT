# -*- coding: utf-8 -*-
"""从 Tushare 拉取 hk_hold (沪深港通持股明细), 逐股填充 V3 面板.

V3 = data/panel_full_enriched_v3.parquet (2.7M rows, 105 cols, 3244 stocks).
日期范围: 2023-01-03 ~ 2026-07-28 (863 交易日).

hk_hold 返回每股每日的北向持股:
  vol   — 持股数量 (股)
  ratio — 持股比例 (占总股本 %)

新增 2 列:
  north_hold_vol  — 北向持股量 (股)
  north_hold_pct  — 北向持股比例 (%)

仅沪深港通标的 (~1500只/日) 有值, 非通标的为 NaN (结构性缺失, 正常).

数据源: Tushare pro.hk_hold(trade_date=YYYYMMDD)
  - 每日返回 ~1500 行 (沪股通 + 深股通标的)
  - 5000 积分: 500 次/分钟 → sleep 0.15s (~400 次/min, 安全)
  - 863 交易日 × 1 次/日 = 863 次 API 调用, 预计 ~4 分钟

断点续传: 每拉完一个日期就追加保存到 parquet, 中断后重跑自动跳过已拉日期.

用法:
  python fill_hk_hold_v3.py
"""
import os
import sys
import logging
import time
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PANEL_PATH = "data/panel_full_enriched_v3.parquet"
CACHE_PATH = "data/supply_cache/alt_data/hk_hold_all.parquet"
HK_HOLD_COLS = ["north_hold_vol", "north_hold_pct"]


def main():
    # ── 1. 加载 V3 面板日期列表 ──
    if not os.path.exists(PANEL_PATH):
        logger.error("面板文件不存在: %s", PANEL_PATH)
        sys.exit(1)

    panel_dates = pd.read_parquet(PANEL_PATH, columns=["date"])["date"].unique()
    panel_dates = sorted(pd.to_datetime(panel_dates))
    start_date = panel_dates[0].strftime("%Y%m%d")
    end_date = panel_dates[-1].strftime("%Y%m%d")
    logger.info(
        "V3 面板: %d 交易日 (%s ~ %s), %d stocks",
        len(panel_dates), start_date, end_date,
        pd.read_parquet(PANEL_PATH, columns=["symbol"])["symbol"].nunique(),
    )

    # ── 2. 初始化 Tushare ──
    from app.pipeline1.data_supply import DataSupplyChain

    supply = DataSupplyChain()
    pro = supply._tushare_pro()
    if pro is None:
        logger.error("TUSHARE_TOKEN 未配置")
        sys.exit(1)
    logger.info("Tushare pro_api 初始化成功")

    # ── 3. 断点续传: 加载已缓存的日期 ──
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    done_dates = set()
    existing_frames = []
    if os.path.exists(CACHE_PATH):
        old = pd.read_parquet(CACHE_PATH)
        existing_frames.append(old)
        done_dates = set(old["date"].dt.strftime("%Y%m%d").unique())
        logger.info("断点续传: 已有 %d 交易日数据, 跳过", len(done_dates))

    # ── 4. 逐日拉取 hk_hold ──
    todo_dates = [
        d for d in panel_dates
        if d.strftime("%Y%m%d") not in done_dates
    ]
    logger.info("待拉取: %d 交易日", len(todo_dates))

    new_frames = []
    n_ok = 0
    n_empty = 0
    n_err = 0

    for i, d in enumerate(todo_dates):
        dt = d.strftime("%Y%m%d")
        try:
            raw = pro.hk_hold(trade_date=dt)
            if raw is None or len(raw) == 0:
                n_empty += 1
                continue

            # 标准化列名
            raw = raw.copy()
            raw["symbol"] = raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
            raw["date"] = pd.to_datetime(dt, format="%Y%m%d")
            for c in ("vol", "ratio"):
                if c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")

            keep = ["symbol", "date", "vol", "ratio"]
            keep = [c for c in keep if c in raw.columns]
            new_frames.append(raw[keep])
            n_ok += 1

        except Exception as e:
            n_err += 1
            if n_err <= 5:
                logger.warning("hk_hold %s 失败: %s", dt, e)

        # 进度 + 限速
        if (i + 1) % 50 == 0:
            logger.info(
                "进度: %d/%d (%.0f%%)  ok=%d  empty=%d  err=%d",
                i + 1, len(todo_dates), (i + 1) / len(todo_dates) * 100,
                n_ok, n_empty, n_err,
            )
            # 增量保存 (每 50 天写一次)
            if new_frames:
                batch = pd.concat(new_frames, ignore_index=True)
                all_so_far = existing_frames + [batch]
                combined = pd.concat(all_so_far, ignore_index=True)
                combined = combined.drop_duplicates(subset=["symbol", "date"])
                combined.to_parquet(CACHE_PATH, index=False)
                logger.info("  增量保存: %d 行 (累计 %d 交易日)",
                            len(combined), combined["date"].nunique())

        time.sleep(0.15)  # ~400 calls/min, 安全在 500/min 限制内

    # ── 5. 最终保存 ──
    if new_frames:
        batch = pd.concat(new_frames, ignore_index=True)
        all_so_far = existing_frames + [batch]
        combined = pd.concat(all_so_far, ignore_index=True)
        combined = combined.drop_duplicates(subset=["symbol", "date"])
        combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
        combined.to_parquet(CACHE_PATH, index=False)
    elif existing_frames:
        combined = pd.concat(existing_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["symbol", "date"])
    else:
        logger.error("无数据拉取成功")
        sys.exit(1)

    logger.info(
        "hk_hold 拉取完成: %d 交易日, %d 行, ok=%d, empty=%d, err=%d",
        combined["date"].nunique(), len(combined), n_ok, n_empty, n_err,
    )

    # ── 6. 重命名列并合并回 V3 ──
    logger.info("\n=== 合并 hk_hold 到 V3 面板 ===")

    hk = combined.rename(columns={"vol": "north_hold_vol", "ratio": "north_hold_pct"})
    hk = hk[["symbol", "date", "north_hold_vol", "north_hold_pct"]]

    # 用 pyarrow 高效合并 (不全量加载到 pandas)
    full_table = pq.read_table(PANEL_PATH)

    # 转为 pandas 做 merge (2.7M rows, 可接受)
    logger.info("加载完整面板到 pandas (%d rows)...", full_table.num_rows)
    panel = full_table.to_pandas()

    # 先移除旧列 (如果存在)
    for c in HK_HOLD_COLS:
        if c in panel.columns:
            panel = panel.drop(columns=[c])

    # merge
    panel = panel.merge(hk, on=["symbol", "date"], how="left")

    # 统计
    for c in HK_HOLD_COLS:
        nn = panel[c].notna().sum()
        logger.info("  %-20s  %8d non-null  (%5.2f%%)", c, nn, nn / len(panel) * 100)

    # WORM: 保存到新文件 (不覆盖原 V3)
    today = datetime.now().strftime("%Y%m%d")
    out_path = f"data/panel_full_enriched_v3_hkhold_{today}.parquet"
    panel.to_parquet(out_path, index=False)
    logger.info("已保存: %s (%d rows, %d cols)", out_path, len(panel), panel.shape[1])

    # ── 7. 汇总 ──
    logger.info("\n" + "=" * 60)
    logger.info("=== hk_hold 全量填充汇总 ===")
    logger.info("=" * 60)
    logger.info("面板: %s ~ %s (%d 交易日, %d 股票)",
                panel["date"].min().date(), panel["date"].max().date(),
                panel["date"].nunique(), panel["symbol"].nunique())
    logger.info("hk_hold 数据: %d 交易日, %d 行 (per-stock)",
                hk["date"].nunique(), len(hk))
    logger.info("新增列: north_hold_vol (持股量), north_hold_pct (持股比例)")
    logger.info("注意: 仅沪深港通标的 (~1500只/日) 有值, 非通标的为 NaN (结构性缺失)")
    logger.info("如需覆盖 V3: Copy-Item %s %s -Force", out_path, PANEL_PATH)


if __name__ == "__main__":
    main()
