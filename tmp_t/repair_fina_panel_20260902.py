"""一次性: V3 面板尾部 fina 列存量修复 (2026-09-02 事故).

Bug: _daily_fetch.py 第 6 节只做 ffill (面板历史停在 Q1 值), 公告管线每日写入的
data/supply_cache/alt_data/fina_indicator/ 快照无人接进面板 → Q2 财报季
(2026-08-14→09-01) 4950 只股票只有 15 只 roe 变化.

修复方式: PIT 逐公告日回放 (app.pipeline1.fina_cache_merge.replay_fina_asof) —
修复窗内每股只有自身公告日之后的行才拿到新值, 不一律覆盖最新值 (否则 08-28
公告的股票 08-10 行就有 Q2 值, 违反 look-ahead 铁律). 修复起点 = 缓存最新一期
(end_date=20260630) 的最早 announce_date 往前留余量, 下限 2026-07-01.

内存约束 (关键, 本机 15.8GB RAM / 面板 396 万行 × 120 列):
  严禁整表转 pandas, 严禁整表常驻. 逐行组 (row group) 流式重写: 每次只读一个
  行组 (最大 ~105 万行 ≈ 1GB arrow) → 掩码行替换 fina 列 (pc.replace_with_mask
  短 replacements, 非目标行位图原样保留) → 写 tmp → 释放 → 下一组; 结束后
  os.replace 原子替换. 行组边界与 schema 精确保留 (_daily_fetch 靠逐行组流式
  读面板, 单组峰值必须维持原量级). 相比"读全表 (~4GB) 再改"峰值 4GB → ~1.5GB.
  非目标行的值逐行组断言一致.

前置闸 (任一不过即跳过物理修复, 只保留代码修复): psutil available >= 6GB 且
scripts._run_guard.find_conflicts() 为空 (无重训/预测重活在跑).
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

from app.pipeline1.fina_cache_merge import (  # noqa: E402
    load_fina_cache,
    replay_fina_asof,
)
from scripts._run_guard import find_conflicts  # noqa: E402

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
CACHE_DIR = os.path.join(
    _REPO, "data", "supply_cache", "alt_data", "fina_indicator"
)
REPAIR_FLOOR = pd.Timestamp("2026-07-01")  # 修复起点下限 (面板 7 月起 fina 已陈旧)
ANNOUNCE_MARGIN = pd.Timedelta(days=7)     # 最早公告日往前余量
MIN_RAM_GB = 6.0                           # 前置闸: 可用内存下限 (任务规定)
DATE_TYPE = pa.timestamp("ns")


def _avail_gb():
    return psutil.virtual_memory().available / 2**30


def _rss_gb():
    return psutil.Process().memory_info().rss / 2**30


_t0 = time.time()
_peak_rss = _rss_gb()


def _note(msg):
    global _peak_rss
    _peak_rss = max(_peak_rss, _rss_gb())
    print(f"[{time.time() - _t0:6.1f}s rss={_rss_gb():.2f}GB avail={_avail_gb():.2f}GB] {msg}",
          flush=True)


def main():
    # ── 前置闸 ──────────────────────────────────────────────────────────────
    vm_ok = _avail_gb() >= MIN_RAM_GB
    conflicts = find_conflicts()
    if not vm_ok or conflicts:
        print(f"SKIP: 前置闸未过 (ram_ok={vm_ok}, conflicts={conflicts}) — "
              f"物理修复不执行, 代码修复 (修复1+2) 已交付, 本修复待执行")
        return 1

    # ── 1. 缓存 + 修复起点 ─────────────────────────────────────────────────
    cache = load_fina_cache(CACHE_DIR)
    if not len(cache):
        print("SKIP: fina 缓存为空 — 无新真相可回放, 物理修复不执行")
        return 1
    latest_period = cache["report_period"].max()
    earliest_ann = cache.loc[
        cache["report_period"] == latest_period, "announce_date"
    ].min()
    repair_start = max(REPAIR_FLOOR, earliest_ann - ANNOUNCE_MARGIN)
    _note(f"cache rows={len(cache)} symbols={cache['symbol'].nunique()} "
          f"latest_period={latest_period.date()} "
          f"earliest_announce={earliest_ann.date()}")
    _note(f"repair window: date >= {repair_start.date()}")

    # ── 2. 逐行组流式重写 ──────────────────────────────────────────────────
    pf = pq.ParquetFile(PANEL)
    schema = pf.schema_arrow
    orig_rows = pf.metadata.num_rows
    orig_names = list(schema.names)
    n_rg = pf.metadata.num_row_groups
    date_type = schema.field("date").type
    date_scalar = pa.scalar(repair_start.to_datetime64(), type=date_type)

    tmp_path = PANEL + ".repair_tmp"
    if os.path.exists(tmp_path):  # 上次硬中断残留 → 清掉再写 (防占盘)
        os.remove(tmp_path)
    writer = pq.ParquetWriter(tmp_path, schema)
    total_masked, total_changed = 0, {}
    try:
        for rg_i in range(n_rg):
            rg = pf.read_row_group(rg_i)
            mask_np = pc.greater_equal(rg["date"], date_scalar).to_numpy(
                zero_copy_only=False
            )
            idx = np.flatnonzero(mask_np)
            if len(idx):
                # PIT 回放仅针对本行组掩码行 (小帧转 pandas, <10 万行)
                small = rg.select(["symbol", "date"]).take(pa.array(idx)).to_pandas()
                replayed = replay_fina_asof(small, cache)
                cols = [c for c in replayed.columns if c in schema.names]
                mask_arrow = pa.array(mask_np)
                unmask = ~mask_np
                for col in cols:
                    old_np = rg.column(col).to_numpy(zero_copy_only=False)
                    new_masked = replayed[col].to_numpy(dtype="float64")
                    old_masked = old_np[idx]
                    # 缓存无公告/缺值的格 (NaN) 保留面板原值, 不写 NaN 覆盖
                    final_masked = np.where(
                        np.isnan(new_masked), old_masked, new_masked
                    )
                    changed = ~(
                        (final_masked == old_masked)
                        | (np.isnan(final_masked) & np.isnan(old_masked))
                    )
                    new_col = pc.replace_with_mask(
                        rg.column(col), mask_arrow, pa.array(final_masked)
                    )
                    # 断言: 本行组非目标行的值逐列一致 (位图由 replace_with_mask
                    # 原样保留, 仍验证)
                    new_np = new_col.to_numpy(zero_copy_only=False)
                    assert np.array_equal(
                        old_np[unmask], new_np[unmask], equal_nan=True
                    ), f"非目标行被改动: rg{rg_i} {col}"
                    rg = rg.set_column(schema.get_field_index(col), col, new_col)
                    changed_syms = small["symbol"].to_numpy()[changed]
                    _acc = total_changed.setdefault(
                        col, [0, set()]
                    )
                    _acc[0] += int(changed.sum())
                    _acc[1].update(pd.unique(changed_syms).tolist())
                    del old_np, new_np, old_masked, final_masked, new_col
                total_masked += len(idx)
                del small
            assert rg.num_rows == pf.metadata.row_group(rg_i).num_rows
            assert list(rg.schema.names) == orig_names
            writer.write_table(rg)  # 保留原行组边界
            del rg, mask_np
            gc.collect()
            _note(f"row group {rg_i + 1}/{n_rg} written (masked {len(idx)})")
    finally:
        pf.close()  # 必须先关: Windows 上句柄未释放时 os.replace 目标文件 → WinError 5
        writer.close()
        # 异常路径: tmp 留存待查 (真实面板未被 os.replace 触碰), 下次运行预清理

    # ── 3. 表级断言 + 原子替换 ─────────────────────────────────────────────
    pf2 = pq.ParquetFile(tmp_path)
    assert pf2.metadata.num_rows == orig_rows, "行数变化"
    assert list(pf2.schema_arrow.names) == orig_names, "schema 列名序列变化"
    pf2.close()
    os.replace(tmp_path, PANEL)  # 原子替换
    _note("os.replace done — panel updated in place")

    # ── 4. 修复后抽查 ──────────────────────────────────────────────────────
    chk = pq.read_table(
        PANEL, columns=["symbol", "date", "roe"],
        filters=[("date", ">=", repair_start.to_datetime64())],
    ).to_pandas()
    _s = cache[cache["symbol"] == "002044"].sort_values("announce_date")
    if len(_s):
        _ann = _s["announce_date"].iloc[-1]
        _new_roe = _s["roe"].iloc[-1]
        _hit = chk[(chk["symbol"] == "002044") & (chk["date"] > _ann)]
        _note(f"抽查 002044: 公告 {_ann.date()} 后 {_hit['roe'].notna().sum()} 行 "
              f"roe 应全为 {_new_roe}: "
              f"{bool((_hit['roe'].dropna() == _new_roe).all())}")
    _note(f"repair summary: window >= {repair_start.date()}, "
          f"masked rows={total_masked}, "
          f"total changed rows={sum(v[0] for v in total_changed.values())}")
    for col in sorted(total_changed):
        n_rows, syms = total_changed[col]
        print(f"    {col}: {n_rows} rows / {len(syms)} symbols changed")
    print(f"DONE peak process rss={_peak_rss:.2f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
