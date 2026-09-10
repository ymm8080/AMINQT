# -*- coding: utf-8 -*-
"""09-09 用户: PARALLEL TOP10 预测质量很差 — 轻量实测近期交付 TOP10 已实现质量.

只读: STOCK LIST 目录近期 parallel_shortlist_*.csv + _diag_stage_{board}_3y.parquet
少量列 (symbol/date/label_pm_{3,5,10}d_net). 无训练无回测, 重训期间可跑.
口径: 每日每板块 cut=T-10 (交付桶) 已实现净收益均值 vs 同日板内全池均值 (α),
只统计已成熟日 (label 非 NaN 自动排除).
"""
import re
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from config.settings import DATA_DIR, STOCK_LIST_DIR

PM = {"3d": "label_pm_3d_net", "5d": "label_pm_5d_net", "10d": "label_pm_10d_net"}
NDAYS = 10

# ── 近期交付清单 ──────────────────────────────────────────────
frames = []
for fp in sorted(STOCK_LIST_DIR.glob("parallel_shortlist_*.csv")):
    m = re.match(r"parallel_shortlist_(\d{8})__M.*\.csv$", fp.name)
    if not m:
        continue
    d = pd.read_csv(fp, dtype={"symbol": str})
    if "cut" not in d.columns or "board" not in d.columns:
        continue
    d = d[d["cut"] == "T-10"].copy()
    if d.empty:
        continue
    d["sym"] = d["symbol"].astype(str).str.zfill(6)
    d["ld"] = m.group(1)
    frames.append(d[["ld", "board", "sym"]])
lists = pd.concat(frames, ignore_index=True)
dates = sorted(lists["ld"].unique())[-NDAYS:]
lists = lists[lists["ld"].isin(dates)]
print(f"[lists] 近 {len(dates)} 个交付日: {dates[0]}..{dates[-1]} "
      f"(每日每板块 T-10 = 交付桶)", flush=True)

# ── 面板已实现净收益 ──────────────────────────────────────────
panels = {}
for board in ("main", "dual"):
    t = pq.read_table(
        str(DATA_DIR / f"_diag_stage_{board}_3y.parquet"),
        columns=["symbol", "date"] + list(PM.values()),
    ).to_pandas()
    t["symbol"] = t["symbol"].astype(str).str.zfill(6)
    t["ld"] = pd.to_datetime(t["date"]).dt.strftime("%Y%m%d")
    panels[board] = t

# ── 逐日 α ────────────────────────────────────────────────────
rows = []
for ld in dates:
    day = lists[lists["ld"] == ld]
    for board in ("main", "dual"):
        sel = day[day["board"] == board]["sym"].unique()
        if not len(sel):
            continue
        pan = panels[board]
        pl = pan[pan["ld"] == ld]
        hit = pl[pl["symbol"].isin(sel)]
        if hit.empty:
            continue
        rec = {"ld": ld, "board": board, "n": len(hit)}
        for h, col in PM.items():
            if col not in pl.columns:
                continue
            top = hit[col].dropna()
            pool = pl[col].dropna()
            if len(top) and len(pool):
                rec[f"top_{h}"] = top.mean()
                rec[f"pool_{h}"] = pool.mean()
                rec[f"wr_{h}"] = (top > 0).mean()
        rows.append(rec)
df = pd.DataFrame(rows)
if df.empty:
    print("[error] 无可评估清单日", flush=True)
    raise SystemExit(1)

print("\n── 逐日明细 (top=交付桶均值, pool=同日板内全池, α=top-pool) ──", flush=True)
for h in ("3d", "5d", "10d"):
    sub = df.dropna(subset=[f"top_{h}"])
    if sub.empty:
        continue
    sub = sub.assign(**{f"al_{h}": sub[f"top_{h}"] - sub[f"pool_{h}"]})
    a = sub[f"al_{h}"]
    print(f"\n[T+{h}] 成熟 {len(sub)} 板块日:", flush=True)
    for _, r in sub.sort_values("ld").iterrows():
        print(f"  {r['ld']} {r['board']:<5} n={int(r['n']):>2} "
              f"top {r[f'top_{h}']:+.2%} (wr {r[f'wr_{h}']:.0%}) "
              f"vs pool {r[f'pool_{h}']:+.2%} → α {r[f'al_{h}']:+.2%}", flush=True)
    print(f"  ── 均值: top {sub[f'top_{h}'].mean():+.2%} | pool "
          f"{sub[f'pool_{h}'].mean():+.2%} | α {a.mean():+.2%} "
          f"| α>0 占比 {(a > 0).mean():.0%} | top>0 占比 "
          f"{(sub[f'wr_{h}'] > 0).mean():.0%}", flush=True)

# 分层: 板内 rank 1-10 / 11-20? (交付桶只 T-10) → 改看带 co_occur/共现票 vs 单系统
print("\n── 共现(双系统)票 vs 单系统票 (3d, 近窗成熟日) ──", flush=True)
sel_rows = []
for fp in sorted(STOCK_LIST_DIR.glob("parallel_shortlist_*.csv")):
    m = re.match(r"parallel_shortlist_(\d{8})__M.*\.csv$", fp.name)
    if not m or m.group(1) not in dates:
        continue
    d = pd.read_csv(fp, dtype={"symbol": str})
    if "cut" not in d.columns:
        continue
    d = d[d["cut"] == "T-10"].copy()
    d["sym"] = d["symbol"].astype(str).str.zfill(6)
    d["ld"] = m.group(1)
    sel_rows.append(d[["ld", "board", "sym", "co_occur"]])
sels = pd.concat(sel_rows, ignore_index=True)
for board in ("main", "dual"):
    pan = panels[board]
    for lbl, mask in (("共现", sels["co_occur"].astype(bool)), ("单系统", ~sels["co_occur"].astype(bool))):
        sub = sels[(sels["board"] == board) & mask]
        if sub.empty:
            continue
        rets = []
        for ld, g in sub.groupby("ld"):
            pl = pan[pan["ld"] == ld]
            hit = pl[pl["symbol"].isin(g["sym"])][PM["3d"]].dropna()
            if len(hit):
                pool = pl[PM["3d"]].dropna().mean()
                rets.append((hit.mean() - pool, hit.mean()))
        if rets:
            al = pd.Series([r[0] for r in rets])
            tp = pd.Series([r[1] for r in rets])
            print(f"  {board:<5} {lbl}: {len(rets)} 日 | 3d top {tp.mean():+.2%} "
                  f"| α {al.mean():+.2%} | α>0 {(al > 0).mean():.0%}", flush=True)
print("\nDONE", flush=True)
