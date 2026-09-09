# -*- coding: utf-8 -*-
"""
日内结构指纹 phase-2: 交付池删查线回放 (2026-09-09)
候选: f_close_low_ma5 = mean5(close/low-1), phase-1 残差IC -0.064 独立于r5/r10.
协议: 对齐 F批 _envrecheck_vpkill_0908 — 池=历史交付短名单(legacy+parallel),
  特征用交付日D已完成bar算(T+1起FWD10), 组内当日分位删顶组, 报 净变化/被删组/误杀大赢(FWD>=15%).
参照系: R2=删amt_agree10顶20% 净+0.62pp 零误杀大赢 (已接线PR#145).
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
SLDIR = r"D:/AMINQT/DAILY OPERATION/STOCK LIST"

# ---------- 面板特征 + 前瞻收益 ----------
p = pq.read_table(PANEL, columns=["symbol", "date", "high", "low", "close", "amount", "volume"]).to_pandas()
HAS_AT = "amt_agree10" in pq.read_schema(PANEL).names
if HAS_AT:
    p = p.merge(
        pq.read_table(PANEL, columns=["symbol", "date", "amt_agree10"]).to_pandas(),
        on=["symbol", "date"], how="left")
print(f"[amt_agree10 in panel] {HAS_AT}")
p["symbol"] = p["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
p = p.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
cl = p["close"] / p["low"] - 1.0
p["f_cl5"] = cl.groupby(p["symbol"]).transform(lambda s: s.rolling(5).mean())
p["f_vw5"] = ((p["close"] / (p["amount"] / p["volume"]) - 1.0)
              .groupby(p["symbol"]).transform(lambda s: s.rolling(5).mean()))
c = p.groupby("symbol")["close"]
p["fwd10"] = c.shift(-11) / c.shift(-1) - 1.0
feats = p[["symbol", "date", "f_cl5", "f_vw5", "fwd10"] + (["amt_agree10"] if HAS_AT else [])]

# ---------- 交付清单 ----------
rows = []
for f in glob.glob(os.path.join(SLDIR, "legacy_stocklist_*.csv")):
    m = re.search(r"(\d{8})", os.path.basename(f))
    t = pd.read_csv(f, usecols=["symbol"], dtype={"symbol": str})
    t["date"] = pd.to_datetime(m.group(1))
    t["module"] = "legacy"
    rows.append(t)
for f in glob.glob(os.path.join(SLDIR, "parallel_shortlist_*.csv")):
    m = re.search(r"(\d{8})", os.path.basename(f))
    t = pd.read_csv(f, dtype=str)
    if len(t) == 0:
        continue
    dcol = t["date"].iloc[0] if "date" in t.columns else m.group(1)
    t = t[["symbol"]].dropna()
    t["date"] = pd.to_datetime(dcol)
    t["module"] = "parallel"
    rows.append(t)
dl = pd.concat(rows, ignore_index=True)
dl["symbol"] = dl["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
dl = dl.drop_duplicates(["module", "date", "symbol"])
print(f"delivery rows={len(dl):,}  days={dl['date'].nunique()}  {dl['date'].min().date()}..{dl['date'].max().date()}")

df = dl.merge(feats, on=["symbol", "date"], how="left")
df = df[df["fwd10"].notna()]
print(f"evaluable rows={len(df):,}  days={df['date'].nunique()}  {df['date'].min().date()}..{df['date'].max().date()}")

# ---------- 删查回放 (池化=逐日一致口径) ----------
def replay(df, feat, q, pre_del=None):
    per_day = []
    for (d, mo), sub0 in df.groupby(["date", "module"]):
        sub = sub0
        if pre_del is not None and HAS_AT:
            m = sub[pre_del].notna()
            if m.sum() >= 5:
                sub = sub[sub[pre_del] < sub.loc[m, pre_del].quantile(0.8)]
        if len(sub) < 5 or sub[feat].isna().all():
            continue
        thr = sub[feat].quantile(q)
        mdel = sub[feat] >= thr
        if mdel.sum() == 0:
            continue
        pool, kept, dele = sub["fwd10"], sub.loc[~mdel, "fwd10"], sub.loc[mdel, "fwd10"]
        per_day.append({"date": d, "module": mo, "n": len(sub), "ndel": int(mdel.sum()),
                        "pool": pool.mean(), "kept": kept.mean(), "dele": dele.mean(),
                        "gain": kept.mean() - pool.mean(),
                        "pool_win": (pool > 0).mean(), "kept_win": (kept > 0).mean(),
                        "pool_big": int((pool >= 0.15).sum()), "del_big": int((dele >= 0.15).sum())})
    return pd.DataFrame(per_day)

def pooled(r):
    out = {}
    for mo, rr in [("legacy", r[r["module"] == "legacy"]), ("parallel", r[r["module"] == "parallel"]),
                   ("ALL", r)]:
        if len(rr) == 0:
            continue
        w = rr["n"] - rr["ndel"]
        out[mo] = {"days": len(rr),
                   "delpct": rr["ndel"].sum() / rr["n"].sum(),
                   "pool": (rr["pool"] * rr["n"]).sum() / rr["n"].sum(),
                   "kept": (rr["kept"] * w).sum() / w.sum(),
                   "kept_win": (rr["kept_win"] * w).sum() / w.sum(),
                   "dele": (rr["dele"] * rr["ndel"]).sum() / rr["ndel"].sum(),
                   "del_big": int(rr["del_big"].sum())}
        out[mo]["gain"] = out[mo]["kept"] - out[mo]["pool"]
    return out

print(f"\n{'arm':<30}{'module':<10}{'days':>5}{'删%':>6}{'池均':>9}{'留均':>9}{'净变化':>9}{'留胜率':>8}{'删组均':>9}{'误杀大赢':>9}")
ARMS = [("f_cl5", 0.8, "f_cl5 顶20%", None), ("f_cl5", 0.9, "f_cl5 顶10%", None),
        ("f_cl5", 0.7, "f_cl5 顶30%", None), ("f_vw5", 0.8, "f_vw5 顶20%", None)]
if HAS_AT:
    ARMS += [("f_cl5", 0.8, "f_cl5 顶20%|先删at10顶20%", "amt_agree10")]
for feat, q, name, pre in ARMS:
    r = replay(df, feat, q, pre_del=pre)
    g = pooled(r)
    for mo in ["legacy", "parallel", "ALL"]:
        if mo not in g:
            continue
        x = g[mo]
        print(f"{name:<30}{mo:<10}{x['days']:>5}{x['delpct']:>6.0%}"
              f"{x['pool']:>+9.4f}{x['kept']:>+9.4f}{x['gain']:>+9.4f}{x['kept_win']:>8.1%}"
              f"{x['dele']:>+9.4f}{x['del_big']:>9}")
print("\n参照: R2=删amt_agree10顶20% 净+0.62pp/10日 零误杀大赢; FWD未扣成本")
