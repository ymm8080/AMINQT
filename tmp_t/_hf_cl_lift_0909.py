# -*- coding: utf-8 -*-
"""
新窗口因子对现有模型预测的增量 (2026-09-09, 轻量叠层代理, 免重训)
问: 在生产模型预测值 (pred_ret_10d) 之上, 叠 cl5/cl20 还能加多少预测力?
法: 预测档案(legacy/parallel) 合并 f_cl5/f_cl20/fwd10, 日截面:
  1) 基线: Spearman(pred, fwd10)
  2) 增量A: 正交叠层 rank(pred) 标准化 + w*rank(cl) 标准化, w 扫 {0.2,0.35,0.5}
  3) 增量B: fwd10 对 [pred] 截面回归的残差 与 cl 的 IC (纯残差口径)
  4) 新因子与 pred 的相关 (是否已被模型隐含)
口径: T+1 起 FWD10, 与删查线/信息测试一致. 代理结论非重训结论.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
SLDIR = r"D:/AMINQT/DAILY OPERATION/STOCK LIST"

p = pq.read_table(PANEL, columns=["symbol", "date", "low", "close"]).to_pandas()
p["symbol"] = p["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
p = p.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
base = p["close"] / p["low"] - 1.0
for w, nm in [(5, "f_cl5"), (20, "f_cl20")]:
    p[nm] = base.groupby(p["symbol"]).transform(lambda s: s.rolling(w, min_periods=w).mean())
c = p.groupby("symbol")["close"]
p["fwd10"] = c.shift(-11) / c.shift(-1) - 1.0
feats = p[["symbol", "date", "f_cl5", "f_cl20", "fwd10"]]

def load_preds(pattern, module, pred_col):
    picked = {}
    for f in glob.glob(os.path.join(SLDIR, pattern)):
        b = os.path.basename(f)
        md = re.search(r"__D(\d{8})", b)
        m1 = re.search(r"__(\d{8})", b)
        if not (md or m1):
            continue
        ddate = md.group(1) if md else m1.group(1)
        bt = re.search(r"_(\d{8})", b)
        bt = bt.group(1) if bt else "0"
        key = (ddate, bt, b)
        if ddate not in picked or key > picked[ddate]:
            picked[ddate] = key
    out = []
    for ddate, (_d, _b, b) in picked.items():
        try:
            t = pd.read_csv(os.path.join(SLDIR, b), dtype={"symbol": str})
        except Exception:
            continue
        if len(t) < 30 or pred_col not in t.columns:
            continue
        t = t[["symbol", pred_col]].dropna()
        t = t.rename(columns={pred_col: "pred"})
        t["date"] = pd.to_datetime(ddate)
        t["module"] = module
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

pools = [x for x in [load_preds("legacy_preds_raw_*.csv", "legacy", "pred_ret_10d"),
                     load_preds("parallel_preds_raw_*.csv", "parallel", "pred_mag_10d")] if len(x)]
dl = pd.concat(pools, ignore_index=True)
df = dl.merge(feats, on=["symbol", "date"], how="left")
df = df[df["fwd10"].notna() & df["f_cl5"].notna()]
print(f"rows={len(df):,} days={df['date'].nunique()} evaluable {df['date'].min().date()}..{df['date'].max().date()}")

def zr(s):
    r = s.rank()
    return (r - r.mean()) / (r.std() + 1e-12)

rows = []
for (d, mo), sub in df.groupby(["date", "module"]):
    if len(sub) < 50:
        continue
    y = sub["fwd10"]
    zp = zr(sub["pred"])
    z5, z20 = zr(sub["f_cl5"]), zr(sub["f_cl20"])
    rec = {"date": d, "module": mo, "n": len(sub),
           "ic_pred": np.corrcoef(zp, y.rank())[0, 1],
           "corr_pred_cl5": np.corrcoef(zp, z5)[0, 1],
           "corr_pred_cl20": np.corrcoef(zp, z20)[0, 1]}
    # 残差口径: pred 残差 (fwd 对 zp 截面回归) 与 cl 的 IC
    X = np.column_stack([zp.values, np.ones(len(sub))])
    beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    resid = y.values - X @ beta
    rec["resid_ic_cl5"] = np.corrcoef(z5.values, pd.Series(resid).rank())[0, 1]
    rec["resid_ic_cl20"] = np.corrcoef(z20.values, pd.Series(resid).rank())[0, 1]
    rec["resid_ic_cl20_vs5"] = np.nan
    # cl20 对 [zp,z5] 正交后残差 IC
    X2 = np.column_stack([zp.values, z5.values, np.ones(len(sub))])
    b2, *_ = np.linalg.lstsq(X2, sub["f_cl20"].values, rcond=None)
    r2v = sub["f_cl20"].values - X2 @ b2
    rec["resid_ic_cl20_vs5"] = np.corrcoef(pd.Series(r2v).rank(), pd.Series(resid).rank())[0, 1]
    # 叠层组合: pred + w*cl5 (负向因子取负号)
    for w in (0.2, 0.35, 0.5):
        combo = zp - w * z5
        rec[f"ic_combo5_w{w}"] = np.corrcoef(combo, y.rank())[0, 1]
        combo2 = zp - 0.35 * z5 - w * z20
        rec[f"ic_combo520_w{w}"] = np.corrcoef(combo2, y.rank())[0, 1]
    rows.append(rec)

r = pd.DataFrame(rows)
print(f"\ndays={len(r)}")

def rep(tag, sub):
    h = len(sub) // 2
    cols = ["ic_pred", "corr_pred_cl5", "corr_pred_cl20", "resid_ic_cl5", "resid_ic_cl20",
            "resid_ic_cl20_vs5", "ic_combo5_w0.2", "ic_combo5_w0.35", "ic_combo5_w0.5",
            "ic_combo520_w0.2", "ic_combo520_w0.35", "ic_combo520_w0.5"]
    print(f"\n== {tag} days={len(sub)} ==")
    print(f"{'量':<22}{'mean':>8}{'t':>6}{'h1':>8}{'h2':>8}{'vs基线':>9}")
    base_ic = sub["ic_pred"].mean()
    for c in cols:
        s = sub[c].dropna()
        if len(s) < 4:
            continue
        t_ = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if s.std(ddof=1) > 0 else np.nan
        dl_ = f"{s.mean()-base_ic:+.4f}" if c.startswith("ic_combo") else ""
        print(f"{c:<22}{s.mean():>+8.4f}{t_:>6.1f}{s.iloc[:h].mean():>+8.4f}{s.iloc[h:].mean():>+8.4f}{dl_:>9}")

for mo in ["legacy", "parallel"]:
    rep(mo, r[r["module"] == mo])
rep("ALL", r)
print("\n解读: resid_ic_cl5/cl20 = 生产预测残差上新因子的纯增量IC; ic_combo* = 线性叠后总IC与基线差")
