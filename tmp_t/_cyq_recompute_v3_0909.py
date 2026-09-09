# -*- coding: utf-8 -*-
"""cyq_panel.parquet 全量重算 + 面板 cyq 列历史覆盖 (2026-09-09, 勿挂账当天执行).

背景: NaN 换手 bug (min(1.0,nan)=1.0 把缺换手当全换手) 已双修
(cyq_ext 89b52584 + cyq_calculator 566f7d72), 但 data/cyq_panel.parquet
3.69M 行历史值仍为污染值 (355/500 抽样股受染), 且夜链增量只补新日期不刷历史.

瓶颈: 生产 compute_cyq_panel 纯 python 循环全量需 15-25h → 本脚本向量化
(xdata 核与 tmp_chip/_full_true150 / _pin_ab_l1ovd_0909 同源已验证),
14 列导出全向量化, 运行时 ~1h. 唯一浮点差源: tot 用 cumsum[-1] (顺序累加,
与 python sum 同比特) → 分位线索引精确一致; weight_avg 点积 ulp 差 ≤1e-9.

验证闸: 20 抽样股 (seed42) 对照修复后 cyq_calculator._compute_cyq_for_stock
逐行逐列, 任何列超差即 abort 不落盘.

动作: ① 缓存 14 列全量重算 (WORM .bak_0909_cyqfix, 原子替换)
      ② 面板在场 cyq 列 (winner_ratio/pct_90_high/pct_90_con/cost_50pct/
         cost_95pct/weight_avg) 按 (symbol,date) 覆盖 (WORM 备份, 原子替换)
终态: CYQ_RECOMPUTE_DONE status=ok|aborted + tmp_t/_cyq_recompute_result.json
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
CACHE = os.path.join(ROOT, "data", "cyq_panel.parquet")
RESULT = os.path.join(ROOT, "tmp_t", "_cyq_recompute_result.json")
FACTOR = 150
RANGE_DAYS = 120
AR = np.arange(FACTOR)
CYQ14 = [
    "winner_ratio", "avg_cost",
    "pct_70_low", "pct_70_high", "pct_70_con",
    "pct_90_low", "pct_90_high", "pct_90_con",
    "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct",
    "weight_avg",
]
PCTS = {  # cost_at_percentile 参数 → 列名
    0.05: "cost_5pct", 0.15: "cost_15pct", 0.50: "cost_50pct",
    0.85: "cost_85pct", 0.95: "cost_95pct",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _xdata_core(o, hi, lo, c, hsl, T):
    """窗内 120 日 → 当日 xdata 分布 (与生产逐日衰减等价的向量化副本)."""
    s0 = max(0, T - RANGE_DAYS + 1)
    hh, ll, oo, cc = hi[s0:T + 1], lo[s0:T + 1], o[s0:T + 1], c[s0:T + 1]
    hw = hsl[s0:T + 1]
    cT = c[T]
    maxp, minp = float(hh.max()), float(ll.min())
    acc = max(0.01, (maxp - minp) / (FACTOR - 1))
    avg = (oo + hh + ll + cc) / 4.0
    grid = minp + acc * AR
    gr = grid[None, :]

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

    D = np.empty(len(hw))
    D[:-1] = np.cumprod((1.0 - hw)[::-1])[-2::-1]
    D[-1] = 1.0
    xdata = np.maximum((w * D[:, None]).sum(axis=0), 0.0)
    return xdata, grid, cT, acc


def _cost_pct(cum, tot, pct, grid):
    """与 python `cumsum>=target` 循环逐位一致 (tot=cum[-1] 顺序累加同比特)."""
    target = tot * pct
    idx = int(np.searchsorted(cum, target, side="left"))
    if idx > FACTOR - 1:
        idx = FACTOR - 1
    return float(grid[idx])


def _safe_div(a, b, default):
    return a / b if b != 0 else default


def _export_day(xdata, grid, cT):
    cum = np.cumsum(xdata)
    tot = float(cum[-1]) if len(cum) else 0.0
    if tot > 0:
        below = float(xdata[cT >= grid].sum())
        winner_ratio = below / tot
        weight_avg = float((grid * xdata).sum() / tot)
    else:
        winner_ratio = 0.5
        weight_avg = float(cT)
    row = {"winner_ratio": winner_ratio, "weight_avg": weight_avg}
    for pct, col in PCTS.items():
        row[col] = _cost_pct(cum, tot, pct, grid)
    row["avg_cost"] = _cost_pct(cum, tot, 0.50, grid)
    for tag, plo, phi in (("70", 0.15, 0.85), ("90", 0.05, 0.95)):
        lo_v = _cost_pct(cum, tot, plo, grid)
        hi_v = _cost_pct(cum, tot, phi, grid)
        row[f"pct_{tag}_low"] = lo_v
        row[f"pct_{tag}_high"] = hi_v
        row[f"pct_{tag}_con"] = _safe_div(hi_v - lo_v, hi_v + lo_v, 0.0)
    return row


def compute_stock_vec(g):
    """单股全历史 14 列 (T 从 RANGE_DAYS//2 起, 部分窗, 同生产预热语义)."""
    n = len(g)
    out = {c: np.full(n, np.nan) for c in CYQ14}
    o, hi, lo, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    hsl = np.clip(np.nan_to_num(g["turnover_rate"].to_numpy(float)) / 100.0, 0.0, 1.0)
    for T in range(RANGE_DAYS // 2, n):
        if not np.isfinite(c[T]) or c[T] <= 0:
            continue
        xdata, grid, cT, _ = _xdata_core(o, hi, lo, c, hsl, T)
        row = _export_day(xdata, grid, cT)
        for k, v in row.items():
            out[k][T] = v
    res = pd.DataFrame({"date": g["date"].to_numpy(), **out})
    return res


def verify(dfl, n_sample=20):
    """抽样对照修复后的 python canonical, 超差即 abort."""
    from app.pipeline1.cyq_calculator import _compute_cyq_for_stock

    rng = np.random.default_rng(42)
    syms = dfl["symbol"].unique()
    cand = []
    for s in syms:
        g = dfl[dfl["symbol"] == s]
        if len(g) >= 250:
            cand.append(s)
    picks = list(rng.choice(cand, size=min(n_sample, len(cand)), replace=False))
    tol = {"winner_ratio": 1e-10, "weight_avg": 1e-9, "pct_70_con": 1e-9,
           "pct_90_con": 1e-9}
    worst = {}
    for s in picks:
        g = dfl[dfl["symbol"] == s].sort_values("date").reset_index(drop=True)
        g = g.dropna(subset=["open", "high", "low", "close"])
        mine = compute_stock_vec(g)
        canon = _compute_cyq_for_stock(
            g[["date", "open", "close", "high", "low", "turnover_rate"]]
        )
        m = mine.merge(canon, on="date", suffixes=("_v", "_p"), how="inner")
        assert len(m) >= 200, f"{s}: 对齐行数异常 {len(m)}"
        for col in CYQ14:
            d = (m[f"{col}_v"] - m[f"{col}_p"]).abs().max()
            a = tol.get(col, 1e-9)
            worst[col] = max(worst.get(col, 0.0), float(d) if d == d else 0.0)
            if d > a:
                log(f"VERIFY FAIL {s}.{col}: maxdiff={d:.3e} > {a:.0e}")
                return False, worst
    log(f"verify PASS {len(picks)}股 worst per col: "
        + ", ".join(f"{k}={v:.1e}" for k, v in sorted(worst.items(), key=lambda x: -x[1])[:6]))
    return True, worst


def main():
    from scripts._run_guard import find_conflicts
    # 排队: 只等前驱 (ohlc 修复 → ovd 回补), 勿用全量哨兵 (互等死锁)
    PRED = ("_ohlc_repair_v3_0909.py", "_ovd_backfill_v3_0909.py")
    conflicts = find_conflicts(sentinels=PRED)
    attempt = 0
    while conflicts and attempt < 60:
        sens = [c.get("sentinel", "?") for c in conflicts]
        log(f"run-guard 冲突, 等10min重试 ({attempt + 1}/60): {sens}")
        time.sleep(600)
        attempt += 1
        conflicts = find_conflicts(sentinels=PRED)
    if conflicts:
        log("CYQ_RECOMPUTE_DONE status=aborted reason=guard_conflict")
        return 2

    t0 = time.time()
    need = ["symbol", "date", "open", "high", "low", "close", "turnover_rate"]
    log("read panel (7 cols)")
    dfl = pq.read_table(PANEL, columns=need).to_pandas()
    dfl["date"] = pd.to_datetime(dfl["date"])
    log(f"panel rows={len(dfl):,} stocks={dfl['symbol'].nunique()}")

    ok, worst = verify(dfl)
    if not ok:
        log("CYQ_RECOMPUTE_DONE status=aborted reason=verify_fail")
        return 3

    log("full vectorized recompute ...")
    tc = time.time()
    parts = []
    n_done = 0
    syms = dfl["symbol"].nunique()
    for sym, g in dfl.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        g = g.dropna(subset=["open", "high", "low", "close"])
        if len(g) < RANGE_DAYS // 2:
            continue
        res = compute_stock_vec(g)
        res.insert(0, "symbol", sym)
        parts.append(res)
        n_done += 1
        if n_done % 100 == 0:
            el = time.time() - tc
            log(f"recompute {n_done}/{syms} {el:.0f}s ({el / n_done:.2f}s/stock) "
                f"ETA {el / n_done * (syms - n_done) / 60:.0f}min")
    newcache = pd.concat(parts, ignore_index=True)
    newcache = newcache[["symbol", "date"] + CYQ14]
    log(f"recompute done: rows={len(newcache):,} stocks={newcache['symbol'].nunique()} "
        f"({time.time() - tc:.0f}s)")
    del dfl, parts

    # ① 缓存重写 (WORM + 原子)
    if os.path.exists(CACHE):
        bak = CACHE + ".bak_0909_cyqfix"
        if not os.path.exists(bak):
            shutil.copy2(CACHE, bak)
            log(f"WORM cache backup -> {bak}")
        old = pd.read_parquet(CACHE, columns=["symbol", "date"])
        log(f"old cache rows={len(old):,} (新 {len(newcache):,}, "
            f"差 {len(newcache) - len(old):+,})")
    tmp = CACHE + ".tmp_cyqfix"
    newcache["date"] = newcache["date"].astype("datetime64[ns]")
    newcache.to_parquet(tmp, index=False)
    os.replace(tmp, CACHE)
    log(f"cache rewritten: {CACHE}")

    # ② 面板在场 cyq 列覆盖
    tbl = pq.read_table(PANEL)
    overlay = [c for c in CYQ14 if c in tbl.column_names]
    log(f"panel overlay cols: {overlay}")
    comp = pq.ParquetFile(PANEL).metadata.row_group(0).column(0).compression
    pdf = tbl.select(["symbol", "date"] + overlay).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    ren = {c: c + "_new" for c in overlay}
    merged = pdf[["symbol", "date"]].merge(
        newcache.rename(columns=ren), on=["symbol", "date"],
        how="left", validate="one_to_one",
    )
    assert len(merged) == tbl.num_rows, "overlay merge 行数漂移"
    stats = {}
    for c in overlay:
        vals = merged[c + "_new"].to_numpy(dtype="float64")
        base = pdf[c].to_numpy(dtype="float64")
        mask = ~np.isnan(vals)
        stats[c] = round(float(mask.mean() * 100), 2)
        base[mask] = vals[mask]
        tbl = tbl.drop([c]).append_column(c, pa.array(base, type=pa.float64()))
    del pdf, merged
    log(f"overlay coverage pct: {stats}")

    pbak = PANEL + ".bak_0909_cyqfix"
    if not os.path.exists(pbak):
        shutil.copy2(PANEL, pbak)
        log(f"WORM panel backup -> {pbak}")
    ptmp = PANEL + ".tmp_cyqfix"
    pq.write_table(tbl, ptmp, compression=comp)
    os.replace(ptmp, PANEL)
    del tbl

    chk = pq.ParquetFile(PANEL)
    log(f"panel verify: rows={chk.num_rows:,} cols={chk.num_columns} "
        f"cyq在={[c for c in CYQ14 if c in chk.schema_arrow.names]}")

    out = {
        "status": "ok",
        "cache_rows": int(len(newcache)),
        "cache_stocks": int(newcache["symbol"].nunique()),
        "panel_overlay": stats,
        "verify_worst": {k: f"{v:.2e}" for k, v in worst.items()},
        "elapsed_s": round(time.time() - t0, 1),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"CYQ_RECOMPUTE_DONE status=ok elapsed={out['elapsed_s']}s -> {RESULT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        log("CYQ_RECOMPUTE_DONE status=aborted")
        sys.exit(1)
