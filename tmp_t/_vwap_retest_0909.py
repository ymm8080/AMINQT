# -*- coding: utf-8 -*-
"""
vwap 族信息量复测 (2026-09-09, OHLC qfq 污染修复后).
原测 (tmp_t/_hf_fingerprint_infotest_0909.py) 因 close=qfq / volume=vwap 近似 全部作废.
协议对齐原脚本: 个股 TS IC (日截面 Spearman vs T+1→T+11 10日 c2c), 全窗+125d, h1/h2 双半窗,
顶20%/底20% 双尾超额, 动量(r5,r10)残差 IC.
单位归一: hand(手)惯例 symbol volume=手 → 真vwap=(amount/volume)/100 (÷K, 非×K!),
K=median r>50 判 hand. 注意: 本脚本初版曾误用 ×K → hand票 f≈-0.99 恒定钉死截面底部,
导致 ma5≡ma20(corr 0.9998) 与伪残差 +0.034 t=+6.8; 修正后真增量=0, 判词见记忆
hf-fingerprint-verdict-0909.
前置: 须在 _ohlc_repair_v3_0909.py 写回完成后运行 (面板 close=裸价).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
REGIME = r"data/others/_regime_label_daily_20260908.parquet"

cols = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
df = pq.read_table(PANEL, columns=cols).to_pandas()
df["symbol"] = df["symbol"].astype(str)
df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
print(f"panel rows={len(df):,} {df['date'].min()}..{df['date'].max()}")

hi, lo, op, cl = df["high"], df["low"], df["open"], df["close"]
bad = (hi < lo) | (hi < np.maximum(op, cl) - 1e-9) | (lo > np.minimum(op, cl) + 1e-9) | (df["volume"] < 0)
print(f"[validate] bad OHLCV rows={bad.sum():,} ({bad.mean():.5%}) -> excluded")
df = df[~bad.values]

# ---------- 单位自检 + hand 归一 ----------
r = (df["amount"] / df["volume"] / df["close"]).replace([np.inf, -np.inf], np.nan)
sym_med = r.groupby(df["symbol"]).median()
hand = set(sym_med[sym_med > 50].index)
K = df["symbol"].map(lambda s: 100.0 if s in hand else 1.0)
print(f"[units] gu={len(sym_med) - len(hand)} hand={len(hand)}; gu r中位={sym_med[~sym_med.index.isin(hand)].median():.3f} "
      f"hand r中位={(sym_med[sym_med.index.isin(hand)].median() if hand else float('nan')):.1f}")

# ---------- 特征 (仅当日已完成 bar, 无前视) ----------
vwap = df["amount"] / df["volume"] / K  # 手惯例: 真vwap=(amount/volume)/100, 是除非乘
df["f_vwap_dev"] = df["close"] / vwap - 1.0
g = df.groupby("symbol", sort=False)
df["f_vwap_dev_ma5"] = g["f_vwap_dev"].transform(lambda s: s.rolling(5).mean())
df["f_vwap_dev_ma20"] = g["f_vwap_dev"].transform(lambda s: s.rolling(20).mean())
df["f_vwap_dev_chg5"] = g["f_vwap_dev"].transform(lambda s: s - s.shift(5))
FEATS = ["f_vwap_dev", "f_vwap_dev_ma5", "f_vwap_dev_ma20", "f_vwap_dev_chg5"]

# 健全性: 修复后 vwap 应有自然散布 (≠close)
spread = (df["f_vwap_dev"].abs().median())
print(f"[sanity] |f_vwap_dev| 中位={spread:.4f} (>0.001 即非恒等; 修复前≈0)")
assert spread > 0.001, "vwap 仍恒等于 close — 修复未生效?"

c = df.groupby("symbol")["close"]
df["fwd10"] = c.shift(-11) / c.shift(-1) - 1.0
g2 = df.groupby("symbol")["close"]
df["r5"] = g2.transform(lambda s: s / s.shift(5) - 1.0)
df["r10"] = g2.transform(lambda s: s / s.shift(10) - 1.0)

lab = pq.read_table(REGIME).to_pandas()
if "date" not in lab.columns:
    lab = lab.reset_index()
lab["date"] = lab["date"].astype(df["date"].dtype)
df = df.merge(lab[["date", "trend20", "share_gt_ma20", "nh_nl"]], on="date", how="left")
ENV_DIMS = {"trend20": lambda v: v > 0, "share_gt_ma20": lambda v: v > 0.5, "nh_nl": lambda v: v > 0}

records = []
for dt, d in df.groupby("date", sort=True):
    m0 = d["fwd10"].notna()
    if m0.sum() < 50:
        continue
    base_mean = d.loc[m0, "fwd10"].mean()
    rec = {"date": dt, "n": int(m0.sum()), "base": base_mean}
    for f in FEATS:
        m = m0 & d[f].notna()
        if m.sum() < 50:
            rec[f + "__ic"] = np.nan
            continue
        x, y = d.loc[m, f], d.loc[m, "fwd10"]
        rec[f + "__ic"] = np.corrcoef(x.rank(), y.rank())[0, 1]
        q80, q20 = x.quantile(0.8), x.quantile(0.2)
        rec[f + "__top"] = y[x >= q80].mean() - base_mean
        rec[f + "__bot"] = y[x <= q20].mean() - base_mean
    mm = m0 & d["f_vwap_dev_ma5"].notna() & d["r5"].notna() & d["r10"].notna()
    if mm.sum() >= 50:
        sub = d.loc[mm]
        X = np.column_stack([sub["r5"].values, sub["r10"].values, np.ones(int(mm.sum()))])
        beta, *_ = np.linalg.lstsq(X, sub["f_vwap_dev_ma5"].values, rcond=None)
        resid = sub["f_vwap_dev_ma5"].values - X @ beta
        rec["f_vwap_dev_ma5_resid__ic"] = np.corrcoef(
            pd.Series(resid).rank(), sub["fwd10"].rank())[0, 1]
        rec["corr_vw5_r5"] = np.corrcoef(sub["f_vwap_dev_ma5"].rank(), sub["r5"].rank())[0, 1]
    records.append(rec)
ic = pd.DataFrame(records).set_index("date")
print(f"ic days={len(ic)}  n/day median={int(ic['n'].median())}")

lab2 = lab.set_index("date").reindex(ic.index)
ALL = {"f_vwap_dev", "f_vwap_dev_ma5", "f_vwap_dev_ma20", "f_vwap_dev_chg5"}
RESID = {"f_vwap_dev_ma5"}


def summarize(tag, mask):
    sub = ic[mask]
    if len(sub) < 20:
        return
    h = len(sub) // 2
    print(f"\n== {tag}  days={len(sub)} ==", flush=True)
    print(f"{'feat':<20}{'IC':>8}{'t':>7}{'IC>0%':>7}{'h1':>8}{'h2':>8}{'顶20%超':>9}{'底20%超':>9}{'顶胜日%':>8}")
    for f in sorted(ALL | RESID):
        s = sub[f + "__ic"].dropna() if f in RESID else sub[f + "__ic"].dropna()
        if len(s) < 20:
            continue
        tstat = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if s.std(ddof=1) > 0 else np.nan
        line = f"{f:<20}{s.mean():>+8.4f}{tstat:>7.1f}{(s>0).mean():>7.0%}{s.iloc[:h].mean():>+8.4f}{s.iloc[h:].mean():>+8.4f}"
        if f in ALL:
            top, bot = sub[f + "__top"].mean(), sub[f + "__bot"].mean()
            line += f"{top:>+9.4f}{bot:>+9.4f}{(sub[f+'__top']>0).mean():>8.0%}"
        print(line, flush=True)


summarize("全窗", pd.Series(True, index=ic.index))
summarize("近125d", ic.index >= ic.index[-125])
for dim, fn in ENV_DIMS.items():
    if dim in lab2:
        summarize(f"{dim}=强", fn(lab2[dim]).fillna(False).reindex(ic.index).fillna(False))
        summarize(f"{dim}=弱", ~fn(lab2[dim]).fillna(True).reindex(ic.index).fillna(True))

if "corr_vw5_r5" in ic:
    print(f"\ncorr(f_vwap_dev_ma5, r5) 全窗中位={ic['corr_vw5_r5'].median():.3f} (低=信息独立于动量)")
print("VWAP RETEST DONE")
