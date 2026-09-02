# -*- coding: utf-8 -*-
"""一次性: V3 面板尾部 sw 三列 (sw_ret_1d/sw_index_close/sw_index_vol) 存量回填.

背景 (2026-09-02 审计): 08-27 与 09-01 两日 Tushare index_daily 全市场拉取失败,
_daily_fetch 旧逻辑静默 continue → 面板这两日 3 列全 NaN. 防再犯已由
app/pipeline1/sw_sector_fetch.py (重试+完整性 fail-fast) 落地; 本脚本用
data/processed/sw_daily_history.parquet (07-31 冻结已修复, 现新鲜至 09-02)
把这两日的存量 NaN 补上.

口径与 _daily_fetch sw 段完全一致:
  sw_ret_1d = pct_change/100, sw_index_close = close, sw_index_vol = vol/1e6 (round 2);
  行业名→代码别名 = {v: k for k, v in SW_INDEX_CODES.items()} + 电气设备→801730.
只填能映射到 L1 指数行情的行业行; 映射不到 (UNKNOWN 等) 保持 NaN, 与健康日分布一致.

内存: 沿用 tmp_t/repair_fina_panel_20260902.py 的逐行组流式重写 (峰值 ~1.5GB),
前置闸 RAM>=6GB + 无重训/预测冲突, tmp + os.replace 原子替换, 非目标行逐行组断言.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from app.pipeline1.data_supply import SW_INDEX_CODES  # noqa: E402
from scripts._run_guard import find_conflicts  # noqa: E402

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
SW_HIST = os.path.join(_REPO, "data", "processed", "sw_daily_history.parquet")
TARGET_DATES = [pd.Timestamp("2026-08-27"), pd.Timestamp("2026-09-01")]
SW_COLS = ["sw_ret_1d", "sw_index_close", "sw_index_vol"]
MIN_RAM_GB = 6.0

_t0 = time.time()
_peak_rss = psutil.Process().memory_info().rss / 2**30


def _note(msg):
    global _peak_rss
    _peak_rss = max(_peak_rss, psutil.Process().memory_info().rss / 2**30)
    print(f"[{time.time() - _t0:6.1f}s rss={_rss_gb():.2f}GB] {msg}", flush=True)


def _rss_gb():
    return psutil.Process().memory_info().rss / 2**30


def build_sw_map():
    """行业名 → {三列值} (按目标日期), 口径与 _daily_fetch sw 段一致."""
    hist = pq.read_table(SW_HIST).to_pandas()
    hist = hist[hist["level"] == "L1"]
    hist["trade_date"] = pd.to_datetime(hist["trade_date"])
    ind2code = {v: k for k, v in SW_INDEX_CODES.items()}
    ind2code["电气设备"] = "801730"
    code2ind = {}
    for ind, code in ind2code.items():
        code2ind.setdefault(str(code).split(".")[0], ind)
    maps = {}
    for d in TARGET_DATES:
        day = hist[hist["trade_date"] == d]
        m = {}
        for _, r in day.iterrows():
            ind = code2ind.get(str(r["ts_code"]).split(".")[0])
            if ind is None:
                continue
            m[ind] = {
                "sw_ret_1d": float(r["pct_change"]) / 100.0,
                "sw_index_close": float(r["close"]),
                "sw_index_vol": round(float(r["vol"]) / 1e6, 2),
            }
        maps[d] = m
        _note(f"{d.date()}: L1 行情 {len(day)} 条 → 行业映射 {len(m)} 个")
    return maps


def main():
    ram_ok = _avail_gb() >= MIN_RAM_GB
    conflicts = find_conflicts()
    if not ram_ok or conflicts:
        print(f"SKIP: 前置闸未过 (ram_ok={ram_ok}, conflicts={conflicts})")
        return 1

    maps = build_sw_map()

    pf = pq.ParquetFile(PANEL)
    schema = pf.schema_arrow
    orig_rows = pf.metadata.num_rows
    orig_names = list(schema.names)
    n_rg = pf.metadata.num_row_groups
    date_type = schema.field("date").type
    date_set = pa.scalar(TARGET_DATES[0].to_pydatetime(), type=date_type)

    tmp_path = PANEL + ".swbackfill_tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    writer = pq.ParquetWriter(tmp_path, schema)
    total_filled, total_kept_nan = 0, 0
    try:
        for rg_i in range(n_rg):
            rg = pf.read_row_group(rg_i)
            in_lo = pc.greater_equal(rg["date"], date_set)
            in_hi = pc.less(rg["date"], pa.scalar(
                (TARGET_DATES[-1] + pd.Timedelta(days=1)).to_pydatetime(),
                type=date_type))
            mask_np = pc.and_(in_lo, in_hi).to_numpy(zero_copy_only=False)
            idx = np.flatnonzero(mask_np)
            if len(idx):
                small_full = rg.select(["date", "industry"]).take(
                    pa.array(idx)).to_pandas()
                small_full["date"] = pd.to_datetime(small_full["date"])
                mask_arrow = pa.array(mask_np)
                unmask = ~mask_np
                for col in SW_COLS:
                    new_vals = np.array([
                        maps.get(d, {}).get(ind, {}).get(col, np.nan)
                        for d, ind in zip(small_full["date"], small_full["industry"])
                    ], dtype="float64")
                    old_np = rg.column(col).to_numpy(zero_copy_only=False)
                    old_masked = old_np[idx]
                    # 映射不到的格 (NaN) 保留面板原值 (健康日这些行本就 NaN)
                    final_masked = np.where(np.isnan(new_vals), old_masked, new_vals)
                    kept_nan = int(np.isnan(final_masked).sum())
                    filled = int((~np.isnan(final_masked)).sum())
                    new_col = pc.replace_with_mask(
                        rg.column(col), mask_arrow, pa.array(final_masked))
                    new_np = new_col.to_numpy(zero_copy_only=False)
                    unmask = ~mask_np
                    assert np.array_equal(
                        old_np[unmask], new_np[unmask], equal_nan=True
                    ), f"非目标行被改动: rg{rg_i} {col}"
                    rg = rg.set_column(schema.get_field_index(col), col, new_col)
                    total_filled += filled
                    total_kept_nan += kept_nan
                    del old_np, new_np, old_masked, final_masked, new_col
                total_masked = len(idx)
                del small_full
            else:
                total_masked = 0
            assert rg.num_rows == pf.metadata.row_group(rg_i).num_rows
            assert list(rg.schema.names) == orig_names
            writer.write_table(rg)
            del rg, mask_np
            gc.collect()
            _note(f"row group {rg_i + 1}/{n_rg} written (masked {total_masked})")
    finally:
        pf.close()  # Windows: 句柄未释放 os.replace 目标文件会 WinError 5
        writer.close()

    pf2 = pq.ParquetFile(tmp_path)
    assert pf2.metadata.num_rows == orig_rows, "行数变化"
    assert list(pf2.schema_arrow.names) == orig_names, "schema 列名序列变化"
    pf2.close()
    os.replace(tmp_path, PANEL)
    _note("os.replace done — panel updated in place")

    chk = pq.read_table(
        PANEL, columns=["date", "sw_ret_1d"],
        filters=[("date", ">=", TARGET_DATES[0].to_datetime64())],
    ).to_pandas()
    chk["date"] = pd.to_datetime(chk["date"])
    for d in TARGET_DATES:
        sub = chk[chk["date"] == d]
        _note(f"复核 {d.date()}: sw_ret_1d nonnull={sub['sw_ret_1d'].notna().sum()}")
    _note(f"summary: filled={total_filled}, kept_nan={total_kept_nan}")
    print(f"DONE peak process rss={_peak_rss:.2f}GB")
    return 0


def _avail_gb():
    return psutil.virtual_memory().available / 2**30


if __name__ == "__main__":
    sys.exit(main())
