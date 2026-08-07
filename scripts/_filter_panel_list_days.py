"""Remove V3 panel rows whose TRUE trading-days-since-IPO < 150.

Semantics (chosen over the codebase sentinel for a retroactive backfill):
  - 真实交易日历 = SSE 开市日 (2022-01-01 .. 2026-07-31, 覆盖所有相关上市日与行)
  - list_days(row) = real_cal.searchsorted(row_date, "right")
                      - real_cal.searchsorted(list_date, "left")
  - 剔除 list_days < INGEST_MIN_LIST_DAYS (=150) 的行

为什么不用 _gate_base_panel 的哨兵口径: 哨兵把"上市日早于面板起点"的股票
一律视为非次新, 导致 2022 年上市股票在 2023 年初 (真实上市天数 <150) 的行
被误保留. 本次回溯清理正是为修正该缺口 — 结果: 9,050 行 / 123 只 (全部 2023 行,
2022 年上市). 2023+ 上市的新股行已被 daily fetch 入库门剔除, 无残留.

WORM: --apply 时先备份 *_prelistfilter_<ts>.parquet 再写回同名路径.
内存有界: 掩码在 symbol/date 两列上算, 过滤走 pyarrow row-group 级.

Usage:
    python scripts/_filter_panel_list_days.py          # dry-run (report only)
    python scripts/_filter_panel_list_days.py --apply  # WORM backup + write back
"""

import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tushare as ts
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import INGEST_MIN_LIST_DAYS  # noqa: E402

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
CAL_CACHE = r"D:\AMINQT\PARQUET\_tmp_real_cal_2022_2026.parquet"
APPLY = "--apply" in sys.argv


def fetch_stock_basic():
    for attempt in range(1, 4):
        try:
            df = pro.stock_basic(fields="ts_code,symbol,name,list_date")
            if len(df):
                return df.set_index("symbol")
            print(f"  stock_basic empty (attempt {attempt}/3)")
        except Exception as e:
            print(f"  stock_basic attempt {attempt}/3 FAILED ({e})")
        time.sleep(2 * attempt)
    return None


def get_real_cal():
    if os.path.exists(CAL_CACHE):
        return pd.DatetimeIndex(sorted(pd.read_parquet(CAL_CACHE)["cal"]))
    cal = pro.trade_cal(exchange="SSE", start_date="20220101", end_date="20260731")
    real_cal = pd.DatetimeIndex(
        sorted(pd.to_datetime(cal[cal["is_open"] == 1]["cal_date"], format="%Y%m%d"))
    )
    pd.DataFrame({"cal": real_cal}).to_parquet(CAL_CACHE, index=False)
    return real_cal


def main():
    print(f"Panel: {PANEL}")
    print(
        f"INGEST_MIN_LIST_DAYS = {INGEST_MIN_LIST_DAYS} (trading days, true-since-IPO)"
    )

    # ── 1. 读 symbol/date 两列 (轻量) 算掩码 ──
    sd = pq.read_table(PANEL, columns=["symbol", "date"]).to_pandas()
    n0 = len(sd)

    stock_info = fetch_stock_basic()
    if stock_info is None or len(stock_info) == 0:
        sys.exit("FATAL: stock_basic unavailable — aborting")

    real_cal = get_real_cal()
    print(
        f"  real calendar {real_cal.min().date()}..{real_cal.max().date()}: {len(real_cal)} open days"
    )

    list_date = pd.to_datetime(
        sd["symbol"].map(stock_info["list_date"]), format="%Y%m%d", errors="coerce"
    )
    n_missing = int(list_date.isna().sum())
    row_dates = pd.DatetimeIndex(sd["date"])
    list_days = real_cal.searchsorted(row_dates, side="right") - real_cal.searchsorted(
        list_date, side="left"
    )
    drop_young = np.asarray(list_days < INGEST_MIN_LIST_DAYS)
    keep_mask = ~drop_young

    print(f"  missing list_date rows (counted 0 days, dropped): {n_missing}")
    print(
        f"  young rows (true <{INGEST_MIN_LIST_DAYS}td): {int(drop_young.sum()):,}  "
        f"({drop_young.sum() / n0:.3%})"
    )
    print(f"  kept: {int(keep_mask.sum()):,} / {n0:,}")
    if drop_young.any():
        ydf = sd.loc[drop_young, ["symbol", "date"]]
        ydf["list_date"] = ydf["symbol"].map(stock_info["list_date"])
        print(
            f"  affected symbols: {ydf['symbol'].nunique()}  (list years: "
            f"{sorted(set(str(x)[:4] for x in ydf['list_date']))})"
        )
        print("\n被剔股票样例 (symbol, list_date, 行数, 最早/最晚日期):")
        print(
            ydf.assign(d=ydf["date"])
            .groupby(["symbol", "list_date"])["d"]
            .agg(["count", "min", "max"])
            .to_string()
        )

    if not APPLY:
        print("\nDRY-RUN: 未写盘. 加 --apply 执行 (WORM 备份 + 写回).")
        return

    # ── 2. 备份 (WORM 日期后缀) ──
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = PANEL.replace(".parquet", f"_prelistfilter_{ts_str}.parquet")
    print(f"\nBackup -> {backup}")
    shutil.copy2(PANEL, backup)

    # ── 3. 逐 row-group 过滤 + 写回 (内存有界) ──
    tmp = PANEL + ".prelistfilter_tmp.parquet"
    pf = pq.ParquetFile(PANEL)
    with pq.ParquetWriter(tmp, pf.schema_arrow) as writer:
        offset = 0
        kept_total = 0
        for i in range(pf.num_row_groups):
            rg = pf.metadata.row_group(i)
            start, end = offset, offset + rg.num_rows
            rg_mask = keep_mask[start:end]
            offset = end
            if not rg_mask.any():
                print(f"  rg{i}: {rg.num_rows} rows -> 0 (all dropped)")
                continue
            tbl = pf.read_row_group(i).filter(pa.array(rg_mask))
            writer.write_table(tbl)
            kept_total += tbl.num_rows
            print(f"  rg{i}: {rg.num_rows} rows -> {tbl.num_rows} kept")
    assert offset == n0, f"offset {offset} != n0 {n0}"

    # ── 4. 原子替换 ──
    print(f"\nWriting panel ({kept_total} rows)...")
    os.replace(tmp, PANEL)
    pf2 = pq.ParquetFile(PANEL)
    print(f"Done. rows={pf2.metadata.num_rows:,}  cols={len(pf2.schema.names)}")
    print(f"Backup preserved at: {backup}")


if __name__ == "__main__":
    _token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
    if not _token:
        sys.exit("FATAL: No Tushare token")
    pro = ts.pro_api(_token)
    main()
