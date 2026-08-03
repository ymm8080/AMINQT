# -*- coding: utf-8 -*-
"""一次性脚本: 给 V3 面板回填龙虎榜席位分类金额 (KIMI LHB v2.0 上游数据).

按 panel 的每个交易日拉取 Tushare top_inst (机构席位 + 全席位买卖明细),
按 lhb_seats.classify_seat 静态分类聚合出每 (symbol, date) 的:
  lhb_inst_buy/sell   机构专用席位
  lhb_top_buy/sell    顶级游资席位 (静态清单)
  lhb_quant_buy/sell  量化席位 (华鑫上海/华宝东大名路)
  lhb_retail_buy/sell 散户席位 (东财拉萨系)

WORM: 先备份 panel_full_enriched_v3_preseats_v2_<ts>.parquet 再写回.
用法:
  python scripts/_backfill_lhb_seats_v2.py            # 全量
  python scripts/_backfill_lhb_seats_v2.py --dry-run  # 只拉取聚合, 不写回
  python scripts/_backfill_lhb_seats_v2.py --dates 20240102,20240103
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("backfill_lhb_seats_v2")

sys.path.insert(0, ".")

from app.pipeline1.lhb_seats import classify_seat  # noqa: E402

PANEL_PATH = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
SEAT_COLS = [
    "lhb_inst_buy",
    "lhb_inst_sell",
    "lhb_top_buy",
    "lhb_top_sell",
    "lhb_quant_buy",
    "lhb_quant_sell",
    "lhb_retail_buy",
    "lhb_retail_sell",
]
SLEEP_SEC = 0.4
MAX_RETRY = 3


def _fetch_date(pro, d: str) -> pd.DataFrame:
    """拉取单日 top_inst 并聚合为 (symbol, cls, buy, sell). 失败抛异常."""
    raw = pro.top_inst(trade_date=d)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["symbol", "cls", "buy", "sell"])
    raw = raw.copy()
    raw["symbol"] = (
        raw["ts_code"]
        .str.replace(".SZ", "", regex=False)
        .str.replace(".SH", "", regex=False)
    )
    buy_col = next((c for c in ("buy", "buy_amount") if c in raw.columns), None)
    sell_col = next((c for c in ("sell", "sell_amount") if c in raw.columns), None)
    if buy_col is None or sell_col is None:
        return pd.DataFrame(columns=["symbol", "cls", "buy", "sell"])
    raw["buy"] = pd.to_numeric(raw[buy_col], errors="coerce").fillna(0.0)
    raw["sell"] = pd.to_numeric(raw[sell_col], errors="coerce").fillna(0.0)
    # 同一席位同日可同时出现在买卖两侧 (side=0/1) — 每 (symbol, exalter) 去重取 max
    seat = (
        raw.groupby(["symbol", "exalter"])
        .agg(buy=("buy", "max"), sell=("sell", "max"))
        .reset_index()
    )
    seat["cls"] = seat["exalter"].map(classify_seat)
    return (
        seat.groupby(["symbol", "cls"])
        .agg(buy=("buy", "sum"), sell=("sell", "sum"))
        .reset_index()
    )


def _wideify(daily: pd.DataFrame) -> pd.DataFrame:
    """把长表 (symbol, date, cls, buy, sell) 转宽 (每类两列, 按 symbol×date).

    席位金额必须落在具体股票上 (spec §2.1 机构动能是单股买卖盘之比),
    不能按 date 聚合 — 否则同一日期所有股票共享同一席位值, 特征失效.
    """
    rows = []
    for (d, sym), g in daily.groupby(["date", "symbol"]):
        rec = {"date": d, "symbol": sym}
        for cls in ("inst", "top", "quant"):
            cg = g[g["cls"] == cls]
            if len(cg):
                rec[f"lhb_{cls}_buy"] = float(cg["buy"].sum())
                rec[f"lhb_{cls}_sell"] = float(cg["sell"].sum())
        # spec §2.4: retail = 东财拉萨系 + 其他非 inst/top/quant 席位 ("非聪明钱"整体)
        cg = g[g["cls"].isin(["retail", "other"])]
        if len(cg):
            rec["lhb_retail_buy"] = float(cg["buy"].sum())
            rec["lhb_retail_sell"] = float(cg["sell"].sum())
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=["date", "symbol"] + SEAT_COLS)
    out = pd.DataFrame(rows)
    for c in SEAT_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[["date", "symbol"] + SEAT_COLS]


def backfill(dates: list[str], dry_run: bool) -> pd.DataFrame:
    import tushare as ts

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN 未设置 (.env)")
    pro = ts.pro_api(token)

    daily_frames = []
    failed: list[str] = []
    t0 = time.time()
    for i, d in enumerate(dates):
        ok = False
        for attempt in range(MAX_RETRY):
            try:
                agg = _fetch_date(pro, d)
                ok = True
                break
            except Exception as exc:
                if attempt < MAX_RETRY - 1:
                    time.sleep(SLEEP_SEC * (attempt + 1) * 2)
                else:
                    logger.error("  日期 %s 失败 (3 次重试): %s", d, str(exc)[:120])
                    failed.append(d)
        if ok and len(agg):
            agg["date"] = pd.to_datetime(d, format="%Y%m%d")
            daily_frames.append(agg)
        if (i + 1) % 50 == 0:
            logger.info("  进度 %d/%d, %.0fs", i + 1, len(dates), time.time() - t0)
        time.sleep(SLEEP_SEC)

    logger.info("拉取完成: %d 个交易日, 失败 %d", len(daily_frames), len(failed))
    if failed:
        logger.warning("失败日期: %s", ",".join(failed))

    if not daily_frames:
        raise RuntimeError("无任何 top_inst 数据, 中止")

    long = pd.concat(daily_frames, ignore_index=True)
    wide = _wideify(long)
    logger.info(
        "席位聚合: %d 行, 各列非空数: %s",
        len(wide),
        {c: int(wide[c].notna().sum()) for c in SEAT_COLS},
    )
    return wide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dates", type=str, default=None, help="逗号分隔 YYYYMMDD")
    args = parser.parse_args()

    panel = pd.read_parquet(PANEL_PATH)
    logger.info(
        "面板: %d 行, %d symbols, %d 列, 日期 %s ~ %s",
        panel.shape[0],
        panel["symbol"].nunique(),
        panel.shape[1],
        panel["date"].min().strftime("%Y%m%d"),
        panel["date"].max().strftime("%Y%m%d"),
    )

    if args.dates:
        dates = args.dates.split(",")
    else:
        dates = sorted(panel["date"].dt.strftime("%Y%m%d").unique().tolist())
    logger.info("回填 %d 个交易日 ...", len(dates))

    wide = backfill(dates, args.dry_run)

    if args.dry_run:
        logger.info("dry-run 结束, 不写回面板")
        return

    # 幂等: 已有席位列先删
    existing = [c for c in SEAT_COLS if c in panel.columns]
    if existing:
        logger.info("面板已有席位列 %s, 删除后重新回填", existing)
        panel = panel.drop(columns=existing)

    # WORM 备份
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{os.path.splitext(PANEL_PATH)[0]}_preseats_v2_{ts_str}.parquet"
    logger.info("备份: %s", backup)
    panel.to_parquet(backup, index=False)

    # 写回 8 列 (按 symbol×date 对齐, 未上榜股票席位为 NaN)
    panel = panel.merge(wide, on=["symbol", "date"], how="left")
    logger.info("写回: %s (列数 %d)", PANEL_PATH, len(panel.columns))
    panel.to_parquet(PANEL_PATH, index=False)

    # 验证
    for c in SEAT_COLS:
        cov = panel[c].notna().mean()
        logger.info("  %s 覆盖率: %.1f%%", c, 100 * cov)


if __name__ == "__main__":
    main()
