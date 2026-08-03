# -*- coding: utf-8 -*-
"""一次性: V3 基座面板入库门 (ST + 上市 <150 交易日) + 物理删除 is_st/list_days 两列.

在 ingest gate 上线前, 基座面板仍含历史 ST/次新行 (旧 cleaning 阶段才剔).
本脚本对基座面板执行与 _daily_fetch 相同的入库扫描语义:
  - 每行按 stock_basic.list_date 在该行日期的面板交易日历上计上市交易日数
  - 剔除 ST/*ST 行 与 上市 < INGEST_MIN_LIST_DAYS 交易日 的行
  - 物理删除 is_st / list_days 两列 (V3 不再存储)
  - is_suspended 保留

WORM: 先备份 原面板 → *_gate_base_<ts>.parquet, 再写回同名路径.

内存注意: 2.7M×132 行面板的 pandas 布尔掩码拷贝会分配 ~2.4GB 连续内存,
本机 16GB RAM 下易 OOM. 故过滤/删列/写回全部走 pyarrow Table 层 (不落 pandas).
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
from app.core.universe_manager import name_is_st  # noqa: E402

load_dotenv()

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")

_token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
if not _token:
    sys.exit("FATAL: No Tushare token")
pro = ts.pro_api(_token)


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


def main():
    print(f"Panel: {PANEL}")
    print(f"INGEST_MIN_LIST_DAYS = {INGEST_MIN_LIST_DAYS} (trading days)")

    # ── 1. 备份 (WORM 日期后缀) ──
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = PANEL.replace(".parquet", f"_gate_base_{ts_str}.parquet")
    print(f"Backup -> {backup}")
    shutil.copy2(PANEL, backup)

    # ── 2. 读 symbol/date (仅两列, 轻量) 计算交易日历 ──
    print("\nReading symbol/date...")
    sd = pq.read_table(PANEL, columns=["symbol", "date"]).to_pandas()
    n0 = len(sd)
    trade_cal = pd.DatetimeIndex(sorted(sd["date"].unique()))
    print(
        f"  rows={n0}  trade calendar: {len(trade_cal)} days  "
        f"{trade_cal[0].date()} .. {trade_cal[-1].date()}"
    )

    stock_info = fetch_stock_basic()
    if stock_info is None or len(stock_info) == 0:
        sys.exit(
            "FATAL: stock_basic unavailable — aborting (refusing to gate on stale cache)"
        )

    # ── 3. 每行上市交易日数 (掩码在 symbol/date 上算, 不放大) ──
    names = sd["symbol"].map(stock_info["name"]).fillna("")
    is_st = names.map(name_is_st)
    list_date = pd.to_datetime(
        sd["symbol"].map(stock_info["list_date"]), format="%Y%m%d", errors="coerce"
    )
    row_dates = pd.DatetimeIndex(sd["date"])
    left = trade_cal.searchsorted(list_date, side="left")
    right = trade_cal.searchsorted(row_dates, side="right")
    list_days = right - left
    # 上市日早于面板起点 → 非次新 (searchsorted 会给 left=0, 误判为上市首日), 置大哨兵值
    pre_panel = (list_date < trade_cal[0]).to_numpy()
    list_days = np.where(pre_panel, 10**9, list_days)
    print(f"  pre-panel-listed rows (sentinel): {int(pre_panel.sum())}")

    drop_st = is_st.to_numpy()
    drop_young = np.asarray(list_days < INGEST_MIN_LIST_DAYS)
    keep_mask = ~drop_st & ~drop_young
    print(f"  ST rows: {int(drop_st.sum())}  ({drop_st.sum() / n0:.1%})")
    print(
        f"  young rows (<{INGEST_MIN_LIST_DAYS}td): {int(drop_young.sum())}  "
        f"({drop_young.sum() / n0:.1%})"
    )
    print(f"  kept: {int(keep_mask.sum())} / {n0}")

    # ── 4. 逐 row-group 过滤 + 删列 (内存有界, 避免 2.7M×132 整表拷贝) ──
    print("\nFiltering row-group by row-group...")
    tmp = PANEL + ".gating_tmp.parquet"
    pf = pq.ParquetFile(PANEL)
    assert "is_suspended" in pf.schema_arrow.names
    new_schema = pa.schema(
        [f for f in pf.schema_arrow if f.name not in ("is_st", "list_days")]
    )
    assert "is_st" not in new_schema.names and "list_days" not in new_schema.names

    offset = 0
    kept_total = 0
    with pq.ParquetWriter(tmp, new_schema) as writer:
        for i in range(pf.num_row_groups):
            rg = pf.metadata.row_group(i)
            start, end = offset, offset + rg.num_rows
            rg_mask = keep_mask[start:end]
            offset = end
            if not rg_mask.any():
                print(f"  rg{i}: {rg.num_rows} rows -> 0 (all dropped)")
                continue
            tbl = pf.read_row_group(i)
            tbl = tbl.filter(pa.array(rg_mask))
            tbl = tbl.drop(["is_st", "list_days"])
            writer.write_table(tbl)
            kept_total += tbl.num_rows
            print(f"  rg{i}: {rg.num_rows} rows -> {tbl.num_rows} kept")
    assert offset == n0, f"offset {offset} != n0 {n0}"

    # ── 5. 原子替换 ──
    print(f"\nWriting panel ({kept_total} rows)...")
    os.replace(tmp, PANEL)
    print(f"Done. rows={kept_total}  cols={len(new_schema.names)}")
    print(f"Backup preserved at: {backup}")


if __name__ == "__main__":
    main()
