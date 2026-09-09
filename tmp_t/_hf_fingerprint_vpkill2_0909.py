# -*- coding: utf-8 -*-
"""
f_cl5 删查线 终判回放 (2026-09-09 第二轮, 自主拍板版)
三池: A=交付清单(refreshed) B=legacy预测宽池 C=parallel预测宽池
预注册判据 (跑前定死, 防事后拟合):
  接线  := A池 ALL净增益>=+0.5pp 且 legacy/parallel 同号 且 A池误杀大赢率<=1% 且 B/C池同号且>=+0.3pp
  否则  := 不接线, 等档案再涨, 复跑本脚本
协议: 对齐 _hf_fingerprint_vpkill_0909 (T+1起FWD10, 组内当日分位删顶组, 池化逐日一致口径)
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
SLDIR = r"D:/AMINQT/DAILY OPERATION/STOCK LIST"

# ---------- 面板特征 ----------
p = pq.read_table(PANEL, columns=["symbol", "date", "high", "low", "close"]).to_pandas()
p["symbol"] = p["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
p = p.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
cl = p["close"] / p["low"] - 1.0
p["f_cl5"] = cl.groupby(p["symbol"]).transform(lambda s: s.rolling(5).mean())
c = p.groupby("symbol")["close"]
p["fwd10"] = c.shift(-11) / c.shift(-1) - 1.0
feats = p[["symbol", "date", "f_cl5", "fwd10"]]

# ---------- 池 A: 交付清单 ----------
rows = []
for f in glob.glob(os.path.join(SLDIR, "legacy_stocklist_*.csv")):
    m = re.search(r"(\d{8})", os.path.basename(f))
    t = pd.read_csv(f, usecols=["symbol"], dtype={"symbol": str})
    t["date"] = pd.to_datetime(m.group(1)); t["module"] = "legacy"; rows.append(t)
for f in glob.glob(os.path.join(SLDIR, "parallel_shortlist_*.csv")):
    m = re.search(r"(\d{8})", os.path.basename(f))
    t = pd.read_csv(f, dtype=str)
    if len(t) == 0:
        continue
    dcol = t["date"].iloc[0] if "date" in t.columns else m.group(1)
    t = t[["symbol"]].dropna()
    t["date"] = pd.to_datetime(dcol); t["module"] = "parallel"; rows.append(t)
dl = pd.concat(rows, ignore_index=True)
dl["symbol"] = dl["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
dl = dl.drop_duplicates(["module", "date", "symbol"])
poolA = dl

# ---------- 池 B/C: 预测宽池 (同数据日期取最新批次) ----------
def load_preds(pattern, module):
    picked = {}
    for f in glob.glob(os.path.join(SLDIR, pattern)):
        b = os.path.basename(f)
        md = re.search(r"__D(\d{8})", b)
        m1 = re.search(r"__(\d{8})", b)
        if not (md or m1):
            continue
        ddate = md.group(1) if md else m1.group(1)
        batch = re.search(r"_(\d{8})", b)
        batch = batch.group(1) if batch else "0"
        key = (ddate, batch, b)
        if ddate not in picked or key > picked[ddate]:
            picked[ddate] = key
    out = []
    for ddate, (_d, _b, b) in picked.items():
        try:
            t = pd.read_csv(os.path.join(SLDIR, b), usecols=["symbol"], dtype={"symbol": str})
        except Exception:
            continue
        if len(t) < 30:  # 诊断样本(几行)不算池
            continue
        t["date"] = pd.to_datetime(ddate); t["module"] = module
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "date", "module"])

poolB = load_preds("legacy_preds_raw_*.csv", "legacy")
poolC = load_preds("parallel_preds_raw_*.csv", "parallel")

# ---------- 回放 ----------
def replay(pool, feat="f_cl5", q=0.8):
    df = pool.merge(feats, on=["symbol", "date"], how="left")
    df = df[df["fwd10"].notna()]
    per_day = []
    for (d, mo), sub in df.groupby(["date", "module"]):
        if len(sub) < 5 or sub[feat].isna().all():
            continue
        thr = sub[feat].quantile(q)
        mdel = sub[feat] >= thr
        if mdel.sum() == 0:
            continue
        poolv, kept, dele = sub["fwd10"], sub.loc[~mdel, "fwd10"], sub.loc[mdel, "fwd10"]
        per_day.append({"date": d, "module": mo, "n": len(sub), "ndel": int(mdel.sum()),
                        "pool": poolv.mean(), "kept": kept.mean(), "dele": dele.mean(),
                        "del_big": int((dele >= 0.15).sum())})
    r = pd.DataFrame(per_day)
    if not len(r):
        return None, df
    out = {}
    for mo, rr in [("legacy", r[r["module"] == "legacy"]), ("parallel", r[r["module"] == "parallel"]), ("ALL", r)]:
        if len(rr) == 0:
            continue
        w = rr["n"] - rr["ndel"]
        out[mo] = {"days": len(rr), "delpct": rr["ndel"].sum() / rr["n"].sum(),
                   "pool": (rr["pool"] * rr["n"]).sum() / rr["n"].sum(),
                   "kept": (rr["kept"] * w).sum() / w.sum(),
                   "del_big": int(rr["del_big"].sum()), "nrows": int(rr["n"].sum())}
        out[mo]["gain"] = out[mo]["kept"] - out[mo]["pool"]
    return out, df

print(f"{'pool':<10}{'module':<10}{'days':>5}{'删%':>6}{'池均':>9}{'留均':>9}{'净变化':>9}{'误杀大赢':>9}{'rows':>8}")
results = {}
for tag, pool in [("A交付", poolA), ("B预测legacy", poolB), ("C预测par", poolC)]:
    g, df = replay(pool)
    results[tag] = g
    if g is None:
        print(f"{tag}: 无可评日"); continue
    for mo in ["legacy", "parallel", "ALL"]:
        if mo not in g: continue
        x = g[mo]
        print(f"{tag:<10}{mo:<10}{x['days']:>5}{x['delpct']:>6.0%}{x['pool']:>+9.4f}"
              f"{x['kept']:>+9.4f}{x['gain']:>+9.4f}{x['del_big']:>9}{x['nrows']:>8}")

# ---------- 终判 (预注册判据) ----------
print("\n== 终判 (判据跑前定死) ==")
a = results.get("A交付") or {}
b = results.get("B预测legacy") or {}
cp = results.get("C预测par") or {}
okA = a.get("ALL", {}).get("gain", -9) >= 0.005 and a.get("legacy", {}).get("gain", -9) > 0 and a.get("parallel", {}).get("gain", -9) > 0
misA = a.get("ALL", {}).get("del_big", 99) / max(a.get("ALL", {}).get("nrows", 1), 1)
okBC = b.get("ALL", {}).get("gain", -9) >= 0.003 and cp.get("ALL", {}).get("gain", -9) >= 0.003
verdict = okA and misA <= 0.01 and okBC
print(f"A交付 ALL净增益>=0.5pp且双模块同号: {okA} (gain={a.get('ALL', {}).get('gain', float('nan')):+.4f})")
print(f"A池误杀大赢率<=1%: {misA <= 0.01} ({misA:.2%})")
print(f"B/C宽池同号且>=+0.3pp: {okBC} (B={b.get('ALL', {}).get('gain', float('nan')):+.4f} C={cp.get('ALL', {}).get('gain', float('nan')):+.4f})")
print(f"\nVERDICT: {'接线' if verdict else '不接线-继续等档案'}")
