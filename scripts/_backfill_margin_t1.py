#!/usr/bin/env python3
"""_backfill_margin_t1.py — 回填 V3 面板 margin 4 列冻结 (2026-08-14).

根因: Tushare margin_detail 自 ~07-25 起 T+1 发布 (当日行 fetch 恒空), _daily_fetch
的 ffill 链停在最后真实值 07-24 → 全市场 margin_balance/short_balance/margin_buy_amt/
short_sell_vol 冻结 3 周 (600519 等 12 日同值). Tushare 侧 07-27..08-13 各交易日均有
真实数据, 08-14 未发布 (T+1).

做法: 逐交易日拉取 margin_detail → 覆盖写回面板对应日行 (WORM 备份先);
08-14 行按 T+1 语义取 08-13 真实值 (今日行 = 已知最新).

用法:
    python scripts/_backfill_margin_t1.py            # 执行
    python scripts/_backfill_margin_t1.py --dry-run  # 只打印计划不写盘
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tushare as ts
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv()

from config.settings import PANEL_V3_PATH  # noqa: E402

MARGIN_COLS = ["margin_balance", "short_balance", "margin_buy_amt", "short_sell_vol"]
SRC_MAP = {"rzye": "margin_balance", "rqye": "short_balance",
           "rzmre": "margin_buy_amt", "rqyl": "short_sell_vol"}
START = dt.date(2026, 7, 25)   # 冻结起点 (07-24 是面板最后真实值)
END = dt.date(2026, 8, 13)     # 最新已发布日 (08-14 T+1 未发布)
TODAY = dt.date(2026, 8, 14)   # 今日行按 T+1 语义回填昨日真实值


def fetch_margin(pro, d: dt.date) -> pd.DataFrame:
    """单日 margin_detail → (symbol, date, 4 面板列) 帧; 非交易日返回空."""
    df = None
    for attempt in range(4):
        try:
            df = pro.margin_detail(trade_date=d.strftime("%Y%m%d"))
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    if not len(df):
        return pd.DataFrame()
    out = df.rename(columns={"ts_code": "symbol", "trade_date": "date"})
    out = out.rename(columns=SRC_MAP)[["symbol", "date", *MARGIN_COLS]].copy()
    out["symbol"] = out["symbol"].astype(str).str[:6]  # 600519.SH → 600519
    out["date"] = pd.Timestamp(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不写盘")
    args = ap.parse_args()

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("FATAL: TUSHARE_TOKEN 未设置"); return 1
    pro = ts.pro_api(token)

    # 1. 拉取 07-27..08-13 真实 margin
    frames, n_days = [], 0
    d = START + dt.timedelta(days=1)
    while d <= END:
        f = fetch_margin(pro, d)
        time.sleep(1.0)  # Tushare 限流 1 req/s
        if len(f):
            frames.append(f); n_days += 1
            print(f"  {d}: {len(f)} 行")
        d += dt.timedelta(days=1)
    if not frames:
        print("FATAL: 无任何 margin 数据可取"); return 1
    corr = pd.concat(frames, ignore_index=True)
    print(f"已取 {n_days} 个交易日, {len(corr)} 行 (symbol×date 唯一: "
          f"{corr.drop_duplicates(['symbol','date']).shape[0]})")

    # 2. 今日行 = 最新已发布日 (08-13) 真实值 (T+1 语义)
    last = corr[corr["date"] == pd.Timestamp(END)].copy()
    last["date"] = pd.Timestamp(TODAY)
    corr = pd.concat([corr, last], ignore_index=True)
    target_dates = sorted(corr["date"].dt.date.unique())
    print(f"回填目标日期: {target_dates} (+{TODAY} 取 {END} 值)")

    if args.dry_run:
        print("[dry-run] 不写盘"); return 0

    # 3. WORM 备份 + 流式重写面板 (row-group 覆盖, 同 _daily_fetch 追加路径)
    bak = PANEL_V3_PATH.with_name(
        f"panel_full_enriched_v3_pre_margin_backfill_{TODAY:%Y%m%d}.parquet")
    print(f"备份 → {bak}")
    shutil.copy2(PANEL_V3_PATH, bak)

    pf = pq.ParquetFile(PANEL_V3_PATH)
    tmp = PANEL_V3_PATH.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(tmp, pf.schema_arrow)
    n_patched = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        g = pf.read_row_group(rg_idx).to_pandas()
        m = g[["symbol", "date"]].merge(corr, on=["symbol", "date"], how="left")
        for c in MARGIN_COLS:
            v = m[c].to_numpy()
            if v.size:
                g[c] = np.where(pd.isna(v), g[c], v)
        n_patched += int(m[c].notna().sum())
        writer.write_table(pa.Table.from_pandas(g, schema=pf.schema_arrow,
                                                preserve_index=False))
    writer.close(); pf.close()
    os.remove(PANEL_V3_PATH)
    os.rename(tmp, PANEL_V3_PATH)
    print(f"完成: 覆盖 {n_patched} 行")

    # 4. 验证
    p = pd.read_parquet(PANEL_V3_PATH, columns=["date", "symbol", *MARGIN_COLS])
    d14 = p[p["date"] == pd.Timestamp(TODAY)]
    for c in MARGIN_COLS:
        print(f"  {TODAY} {c}: {d14[c].notna().sum()}/3220 non-nan")
    s = p[(p["symbol"] == "600519") & (p["date"].isin(
        [pd.Timestamp(TODAY), pd.Timestamp(dt.date(2026, 8, 13))]))]
    print(s.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
