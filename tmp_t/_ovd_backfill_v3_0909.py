# -*- coding: utf-8 -*-
"""ovd_wdist/ovd_15p 面板全量回补 (2026-09-09, WRITE-PIN 生产化收尾).

背景: tmp_t/_pin_ab_l1ovd_0909.py 三臂 A/B 终判 main pin += ovd两, 但生产面板
无 ovd 列 — feature_selector._run_gate_d 对 pin 缺列只 warning 静默剔除,
不回补则今夜重训退化为 B1 等效白丢增益.

口径: 与 A/B compute_ovd_board 逐字节同核 (FACTOR=150, RANGE_DAYS=120 滚动窗,
三角形权重, 窗内倒序累积衰减, turnover NaN→0, 前导 119 行 NaN 预热);
full_true150/cyq_ext 同源. feature_columns 黑名单外放行 (v35:3903).

重写模式: _volume_repair_v3_0909.py 先例 — 全表 arrow 读入, WORM 备份,
tmp 原压缩写 + os.replace 原子换. 幂等: 已有 ovd 列则替换列值.
终态: OVD_BACKFILL_DONE status=ok|aborted + tmp_t/_ovd_backfill_result.json
"""
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = r"D:\AMINQT\AMINQT CODES"
sys.path.insert(0, ROOT)
PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
RESULT = os.path.join(ROOT, "tmp_t", "_ovd_backfill_result.json")
NEWCOLS = ["ovd_wdist", "ovd_15p"]
FACTOR = 150
RANGE_DAYS = 120
AR = np.arange(FACTOR)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_ovd_all(dfl):
    """全历史逐股 ovd (A/B compute_ovd_board 同核, 无日期过滤)."""
    rows = []
    tc = time.time()
    n_done = 0
    syms = dfl["symbol"].nunique()
    for sym, g in dfl.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        g = g.dropna(subset=["open", "high", "low", "close"])
        n = len(g)
        if n < RANGE_DAYS:
            continue
        o, hi, lo, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        # turnover_rate NaN 按 0 处理 (与 cyq_ext 89b52584 修复口径一致)
        hsl = np.clip(np.nan_to_num(g["turnover_rate"].to_numpy(float)) / 100.0, 0.0, 1.0)
        wdist = np.full(n, np.nan)
        p15 = np.full(n, np.nan)
        for T in range(RANGE_DAYS - 1, n):
            s0 = T - RANGE_DAYS + 1
            hh, ll, oo, cc = hi[s0:T + 1], lo[s0:T + 1], o[s0:T + 1], c[s0:T + 1]
            hw = hsl[s0:T + 1]
            cT = c[T]
            if not np.isfinite(cT) or cT <= 0:
                continue
            maxp, minp = float(hh.max()), float(ll.min())
            acc = max(0.01, (maxp - minp) / (FACTOR - 1))
            avg = (oo + hh + ll + cc) / 4.0
            grid = minp + acc * AR
            gr = grid[None, :]

            # 三角形权重 (生产算法向量化副本)
            flat = hh <= ll
            rngv = np.maximum(hh - ll, 1e-12)
            left = gr <= avg[:, None]
            num = np.where(left, gr - ll[:, None], hh[:, None] - gr)
            den = np.where(left, (avg - ll)[:, None], (hh - avg)[:, None])
            den = np.where(np.abs(den) < 1e-12, 1.0, den)
            w = num / den
            mask = (gr >= ll[:, None]) & (gr <= hh[:, None])
            density = np.where(flat, float(FACTOR - 1), 2.0 / rngv)
            w = np.where(mask, w, 0.0) * (density * hw)[:, None]
            if flat.any():
                w[flat] = 0.0
                ab = np.floor((avg - minp) / acc).astype(int)
                fi = np.nonzero(flat & (ab >= 0) & (ab < FACTOR))[0]
                w[fi, ab[fi]] += (FACTOR - 1) * hw[fi] / 2.0

            # 窗内倒序累积衰减 (与生产顺序衰减逐日等价, 且无全局下溢)
            D = np.empty(len(hw))
            D[:-1] = np.cumprod((1.0 - hw)[::-1])[-2::-1]
            D[-1] = 1.0
            xdata = np.maximum((w * D[:, None]).sum(axis=0), 0.0)
            tot = xdata.sum()
            if tot <= 1e-12:
                continue
            d = (grid - cT) / cT
            abv = d > 0.0
            mab = float(xdata[abv].sum())
            wdist[T] = float((xdata[abv] * d[abv]).sum() / mab) if mab > 1e-12 else 0.0
            p15[T] = float(xdata[d >= 0.15 - 1e-9].sum() / tot)
        rows.append(pd.DataFrame({"symbol": sym, "date": g["date"],
                                  "ovd_wdist": wdist, "ovd_15p": p15}))
        n_done += 1
        if n_done % 100 == 0:
            el = time.time() - tc
            eta = el / n_done * (syms - n_done) / 60.0
            log(f"ovd {n_done}/{syms} {el:.0f}s ({el / n_done:.2f}s/stock) ETA {eta:.0f}min")
    out = pd.concat(rows, ignore_index=True)
    log(f"ovd compute done: stocks={out['symbol'].nunique()} rows={len(out):,} "
        f"({time.time() - tc:.0f}s)")
    return out


def main():
    from scripts._run_guard import find_conflicts
    # 排队: 只等前驱 ohlc 修复 (predecessor-only, 勿用全量哨兵 — 会与后继互等死锁)
    PRED = ("_ohlc_repair_v3_0909.py",)
    conflicts = find_conflicts(sentinels=PRED)
    attempt = 0
    while conflicts and attempt < 60:
        sens = [c.get("sentinel", "?") for c in conflicts]
        log(f"run-guard 冲突, 等10min重试 ({attempt + 1}/60): {sens}")
        time.sleep(600)
        attempt += 1
        conflicts = find_conflicts(sentinels=PRED)
    if conflicts:
        log("OVD_BACKFILL_DONE status=aborted reason=guard_conflict")
        return 2

    t0 = time.time()
    log(f"read panel: {PANEL}")
    pf = pq.ParquetFile(PANEL)
    comp = pf.metadata.row_group(0).column(0).compression
    tbl = pq.read_table(PANEL)
    log(f"panel rows={tbl.num_rows:,} cols={tbl.num_columns} comp={comp}")

    pdf = tbl.select(["symbol", "date", "open", "high", "low", "close",
                      "turnover_rate"]).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])

    ovd = compute_ovd_all(pdf)

    # 行对齐: (symbol,date) 一对一 merge 回原行序, 键重复即大声失败
    merged = pdf[["symbol", "date"]].merge(
        ovd, on=["symbol", "date"], how="left", validate="one_to_one"
    )
    assert len(merged) == tbl.num_rows, "merge 行数漂移"

    cov = {
        "rows_total": int(tbl.num_rows),
        "rows_ovd_nonnan": int(merged["ovd_wdist"].notna().sum()),
        "coverage_pct": round(float(merged["ovd_wdist"].notna().mean() * 100), 2),
        "stocks_with_ovd": int(ovd["symbol"].nunique()),
        "wdist_mean": round(float(merged["ovd_wdist"].mean(skipna=True)), 5),
        "p15_mean": round(float(merged["ovd_15p"].mean(skipna=True)), 5),
    }
    log(f"coverage: {cov}")

    # 幂等: 已有 ovd 列先删再补
    for c in NEWCOLS:
        if c in tbl.column_names:
            tbl = tbl.drop([c])
            log(f"drop pre-existing col {c} (idempotent rerun)")

    tbl2 = tbl.append_column("ovd_wdist", pa.array(
        merged["ovd_wdist"].to_numpy(dtype="float64"), type=pa.float64()))
    tbl2 = tbl2.append_column("ovd_15p", pa.array(
        merged["ovd_15p"].to_numpy(dtype="float64"), type=pa.float64()))
    del tbl, pdf, ovd, merged

    bak = PANEL + ".bak_0909_ovd"
    if not os.path.exists(bak):
        shutil.copy2(PANEL, bak)
        log(f"WORM backup -> {bak}")
    tmp = PANEL + ".tmp_ovd_0909"
    pq.write_table(tbl2, tmp, compression=comp)
    os.replace(tmp, PANEL)
    del tbl2

    ok = pq.ParquetFile(PANEL)
    has = [c for c in NEWCOLS if c in ok.schema_arrow.names]
    assert len(has) == 2, f"回读缺列: {has}"
    log(f"verify: rows={ok.num_rows:,} cols含ovd ✓ backup={os.path.getsize(bak):,}B")

    cov.update({
        "status": "ok",
        "elapsed_s": round(time.time() - t0, 1),
        "compression": comp,
        "finished": datetime.now().isoformat(timespec="seconds"),
    })
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(cov, f, ensure_ascii=False, indent=1)
    log(f"OVD_BACKFILL_DONE status=ok elapsed={cov['elapsed_s']}s -> {RESULT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        log("OVD_BACKFILL_DONE status=aborted")
        sys.exit(1)
