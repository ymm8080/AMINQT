# -*- coding: utf-8 -*-
"""
close/low 窗口长度扫描 (2026-09-09): ma5 vs ma10 vs ma20 叠加价值
协议: 对齐 _hf_fingerprint_infotest_0909 (日截面Spearman vs T+1起FWD10, 全窗+125d, 半窗稳)
增量判据: 长窗对短窗截面回归后的**残差IC** — 只有残差IC显著才是"叠加有用",
单纯IC高只说明是同一信号的更平滑版 (嵌套窗口天然高相关).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"

df = pq.read_table(PANEL, columns=["symbol", "date", "high", "low", "close"]).to_pandas()
df["symbol"] = df["symbol"].astype(str)
df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

hi, lo, cl = df["high"], df["low"], df["close"]
bad = (hi < lo) | (hi < np.maximum(cl, cl) - 1e-9) | (lo > cl + 1e-9)
print(f"[validate] bad rows={bad.sum():,} -> excluded")
df = df[~bad.values].reset_index(drop=True)

base = df["close"] / df["low"] - 1.0
g = df.groupby("symbol", sort=False)
for w in (5, 10, 20):
    df[f"cl{w}"] = g[base.name] if False else None  # placeholder
# 直接 transform
for w in (5, 10, 20):
    df[f"cl{w}"] = base.groupby(df["symbol"]).transform(lambda s: s.rolling(w, min_periods=w).mean())
# 短窗变动方向 (ma5-vs-ma20 类"窗口斜率")
df["cl5_d_cl20"] = df["cl5"] - df["cl20"]

c = df.groupby("symbol")["close"]
df["fwd10"] = c.shift(-11) / c.shift(-1) - 1.0

FEATS = ["cl5", "cl10", "cl20", "cl5_d_cl20"]

# 长窗增量: 残差IC (cl10 对 cl5; cl20 对 [cl5,cl10]; slope 对 [cl5,cl20])
RESID = {
    "cl10__resid_vs_cl5": ["cl10", ["cl5"]],
    "cl20__resid_vs_cl5": ["cl20", ["cl5"]],
    "cl20__resid_vs_cl5_10": ["cl20", ["cl5", "cl10"]],
    "cl5_d_cl20__resid": ["cl5_d_cl20", ["cl5", "cl20"]],
}

records = []
for dt, d in df.groupby("date", sort=True):
    m0 = d["fwd10"].notna()
    if m0.sum() < 50:
        continue
    rec = {"date": dt}
    for f in FEATS:
        m = m0 & d[f].notna()
        if m.sum() < 50:
            continue
        rec[f + "__ic"] = np.corrcoef(d.loc[m, f].rank(), d.loc[m, "fwd10"].rank())[0, 1]
    for tag, (target, regs) in RESID.items():
        m = m0 & d[target].notna()
        for r in regs:
            m &= d[r].notna()
        if m.sum() < 50:
            continue
        sub = d.loc[m]
        X = np.column_stack([sub[r].values for r in regs] + [np.ones(m.sum())])
        beta, *_ = np.linalg.lstsq(X, sub[target].values, rcond=None)
        resid = sub[target].values - X @ beta
        rec[tag + "__ic"] = np.corrcoef(pd.Series(resid).rank(), sub["fwd10"].rank())[0, 1]
    # 嵌套窗秩相关
    mm = m0 & d["cl5"].notna() & d["cl10"].notna() & d["cl20"].notna()
    if mm.sum() >= 50:
        s = d.loc[mm]
        rec["corr5_10"] = np.corrcoef(s["cl5"].rank(), s["cl10"].rank())[0, 1]
        rec["corr5_20"] = np.corrcoef(s["cl5"].rank(), s["cl20"].rank())[0, 1]
    records.append(rec)

ic = pd.DataFrame(records).set_index("date")
print(f"days={len(ic)}")

def summ(tag, mask):
    sub = ic[mask]
    h = len(sub) // 2
    keys = [f + "__ic" for f in FEATS] + [t + "__ic" for t in RESID]
    print(f"\n== {tag} days={len(sub)} ==")
    print(f"{'量':<26}{'mean':>8}{'t':>7}{'h1':>8}{'h2':>8}")
    for k in keys:
        s = sub[k].dropna()
        if len(s) < 20:
            continue
        t_ = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if s.std(ddof=1) > 0 else np.nan
        print(f"{k:<26}{s.mean():>+8.4f}{t_:>7.1f}{s.iloc[:h].mean():>+8.4f}{s.iloc[h:].mean():>+8.4f}")
    for k in ["corr5_10", "corr5_20"]:
        s = sub[k].dropna()
        if len(s):
            print(f"{k:<26}{s.mean():>8.3f}")

d125 = ic.index >= (ic.index.max() - pd.Timedelta(days=182))
summ("全窗", pd.Series(True, index=ic.index))
summ("近125交易日", pd.Series(d125, index=ic.index))
print("\n判读: 残差IC |IC|>=0.02 t>=2 双半窗同号 = 长窗叠加有用; 否则只是同一信号平滑版")
