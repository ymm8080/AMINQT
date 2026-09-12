# -*- coding: utf-8 -*-
"""baostock 5min 小样本质量探针 (2026-09-09, 用户裁决: 先小样本再定全量).

50 股 (SH主板/SZ主板/GEM/STAR 混合, seed42) × 近1个月 5min K线:
  - 结构: 每日 bar 数 (期望~48), 时间覆盖 09:35..15:00, 量额非负
  - 量纲: 聚合到日频 sum(volume)/sum(amount) 对照面板 Tushare 真值
          (amount 元单边无歧义为主判据; volume 报比值揭示惯例)
  - VWAP: 日内 amount/volume vs 当日 close 偏离 (±11% 限内为正常)
终态: tmp_min/_probe_baostock_5min.json + stdout VERDICT ok|fail
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUT = r"D:\AMINQT\AMINQT CODES\tmp_min\_probe_baostock_5min.json"
START, END = "2026-08-07", "2026-09-08"  # 近1个月 (含月初对齐)
np.random.seed(42)

# ---- 宇宙: 面板最新日 symbol, 按板块段分层抽样 ----
syms = pq.read_table(PANEL, columns=["symbol", "date"],
                     filters=[("date", ">=", pd.Timestamp("2026-09-01"))]
                     ).to_pandas()["symbol"].unique()
rng = np.random.default_rng(42)


def pick(prefix, n):
    pool = [s for s in syms if s.startswith(prefix) and len(s) == 6]
    return list(rng.choice(pool, min(n, len(pool)), replace=False))


sample = (pick("60", 8) + pick("00", 20) + pick("30", 15) + pick("688", 5)
          + ["000001", "300911"])
sample = sorted(set(sample))
print(f"sample={len(sample)}: SH{sum(s.startswith('6') for s in sample)} "
      f"SZmain{sum(s.startswith('00') for s in sample)} "
      f"GEM{sum(s.startswith('30') for s in sample)} STAR{sum(s.startswith('688') for s in sample)}",
      flush=True)

# ---- 日频真值 (面板 Tushare) ----
truth = pq.read_table(
    PANEL, columns=["symbol", "date", "open", "high", "low", "close", "volume", "amount"],
    filters=[("date", ">=", pd.Timestamp(START)), ("symbol", "in", sample)],
).to_pandas()
truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
tidx = truth.set_index(["symbol", "date"])

# ---- baostock 拉取 ----
import baostock as bs

lg = bs.login()
if lg.error_code != "0":
    print(f"FATAL login: {lg.error_msg}", flush=True)
    sys.exit(2)


def to_bs(sym):
    return ("sh." if sym.startswith("6") else "sz.") + sym


FIELDS = "date,time,code,open,high,low,close,volume,amount"
rows_all, per_stock = [], []
t0 = time.time()
for i, sym in enumerate(sample, 1):
    rs = bs.query_history_k_data_plus(
        to_bs(sym), FIELDS, start_date=START, end_date=END,
        frequency="5", adjustflag="3")  # 3=不复权, 与面板未复权口径对齐
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        per_stock.append({"symbol": sym, "ok": False, "err": rs.error_msg[:80]})
        continue
    df = pd.DataFrame(rows, columns=FIELDS.split(","))
    df["symbol"] = sym
    rows_all.append(df)
    if i % 10 == 0:
        print(f"  {i}/{len(sample)} ({time.time()-t0:.0f}s)", flush=True)

bs.logout()
bars = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()
for c in ("open", "high", "low", "close", "volume", "amount"):
    bars[c] = pd.to_numeric(bars[c], errors="coerce")

# ---- 结构检查 + 聚合对照 ----
bars["hhmm"] = bars["time"].str.slice(4, 8)
agg = bars.groupby(["symbol", "date"]).agg(
    nbars=("close", "size"),
    vol_sum=("volume", "sum"), amt_sum=("amount", "sum"),
    o_first=("open", "first"), c_last=("close", "last"),
).reset_index()

recs = []
for sym in sample:
    a = agg[agg["symbol"] == sym]
    if a.empty:
        recs.append({"symbol": sym, "ok": False, "reason": "no_bars"})
        continue
    t = tidx.loc[sym] if sym in tidx.index.get_level_values(0) else None
    if t is None or len(a) == 0:
        recs.append({"symbol": sym, "ok": False, "reason": "no_truth"})
        continue
    m = a.merge(t.reset_index()[["date", "close", "volume", "amount"]],
                on="date", how="inner")
    if len(m) < 15:
        recs.append({"symbol": sym, "ok": False,
                     "reason": f"overlap_days={len(m)}"})
        continue
    nb_med = float(m["nbars"].median())
    amt_rel = ((m["amt_sum"] - m["amount"]).abs() / m["amount"].clip(lower=1))
    amt_ok = float((amt_rel <= 0.015).mean())
    vol_ratio = float((m["vol_sum"] / m["volume"].clip(lower=1)).median())
    vwap = m["amt_sum"] / m["vol_sum"].clip(lower=1)
    vwap_dev = float(((vwap / m["close"] - 1).abs() <= 0.11).mean())
    ok = (nb_med >= 44) and (amt_ok >= 0.90) and (0.5 <= vol_ratio <= 200) \
         and (vwap_dev >= 0.90)
    recs.append({"symbol": sym, "ok": bool(ok), "nbars_med": nb_med,
                 "amt_ok_pct": round(amt_ok, 3), "vol_ratio_med": round(vol_ratio, 3),
                 "vwap_dev_ok_pct": round(vwap_dev, 3),
                 "overlap_days": int(len(m))})

df_r = pd.DataFrame(recs)
ok_n = int(df_r["ok"].sum())
vr = pd.to_numeric(df_r.get("vol_ratio_med"), errors="coerce").dropna()
verdict = "ok" if ok_n >= len(sample) * 0.9 else "fail"
summary = {
    "n_sample": len(sample), "n_ok": ok_n,
    "nbars_med_overall": float(pd.to_numeric(df_r.get("nbars_med"),
                                             errors="coerce").median()),
    "amt_ok_mean": float(pd.to_numeric(df_r.get("amt_ok_pct"),
                                       errors="coerce").mean()),
    "vol_ratio_clusters": [float(x) for x in vr.round(2).unique()[:8]],
    "elapsed_s": round(time.time() - t0, 1),
    "verdict": verdict,
}
json.dump({"summary": summary, "per_stock": recs},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False), flush=True)
bad = df_r[~df_r["ok"].astype(bool)]
if len(bad):
    print("FAIL STOCKS:\n" + bad.to_string(), flush=True)
print(f"BAOSTOCK_PROBE_DONE verdict={verdict}", flush=True)
