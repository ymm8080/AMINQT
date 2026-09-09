# -*- coding: utf-8 -*-
"""
日内结构指纹 phase-1 信息量测试 (2026-09-09)
来源: REFERENCE/Design All/高频特征与模型.docx L2 — VWAP偏离/收盘位置/回撤修复
数据: 日频 OHLCV 即可构造, 零新数据 (真尾盘30分钟需分钟数据, 本地零存量, 不在本测试范围)
协议: 对齐 tmp_t/_factor_infotest_0908.py —
  个股 TS IC (日截面 Spearman, vs T+1收盘起10日 c2c), 全窗 + 125d 双窗, h1/h2 半窗,
  顶20%/底20% 双尾超额 (未扣成本, 参照 09-08 口径), trend20/share_gt_ma20/nh_nl 三环境维度.
判读: |IC|>=0.02 且 t>=2 且双半窗同号为有信号; 稳定负IC = 删查线候选 (参照量价族R2先例).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
REGIME = r"data/others/_regime_label_daily_20260908.parquet"

# ---------- 加载 ----------
cols = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
df = pq.read_table(PANEL, columns=cols).to_pandas()
df["symbol"] = df["symbol"].astype(str)
df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
print(f"panel rows={len(df):,} {df['date'].min()}..{df['date'].max()}")

# ---------- OHLCV 校验 (铁律: 异常不静默丢弃, 报数后剔除) ----------
hi, lo, op, cl = df["high"], df["low"], df["open"], df["close"]
bad = (hi < lo) | (hi < np.maximum(op, cl) - 1e-9) | (lo > np.minimum(op, cl) + 1e-9) | (df["volume"] < 0)
print(f"[validate] bad OHLCV rows={bad.sum():,} ({bad.mean():.5%}) -> excluded")
df = df[~bad.values]

# ---------- volume 单位自检 (本面板实测 volume=股, vwap=amount/volume) ----------
r = (df["amount"] / df["volume"] / df["close"]).replace([np.inf, -np.inf], np.nan)
med = float(np.nanmedian(r))
print(f"[units] amount/volume/close median={med:.2f} (≈1 -> volume=股)")
assert 0.8 < med < 1.25, "volume 单位异常, 停下检查"

# ---------- 特征 (全部只用当日已完成 bar, 无前视) ----------
vwap = df["amount"] / df["volume"]
df["f_vwap_dev"] = df["close"] / vwap - 1.0                       # 收盘 vs 日内均价 (尾盘强度代理)
rng = df["high"] - df["low"]
df["f_clv"] = (df["close"] - df["low"]) / rng.where(rng > 1e-6)   # 收盘在当日区间位置
df["f_close_low"] = df["close"] / df["low"] - 1.0                 # 距当日最低修复深度

g = df.groupby("symbol", sort=False)
for base in ["f_vwap_dev", "f_clv", "f_close_low"]:
    df[base + "_ma5"] = g[base].transform(lambda s: s.rolling(5).mean())
    df[base + "_chg5"] = g[base].transform(lambda s: s - s.shift(5))
FEATS = [c for c in df.columns if c.startswith("f_") and not c.startswith("fwd")]

# ---------- 前瞻收益: T+1 收盘 -> T+11 收盘 (10日 c2c, 严格 t+1 起) ----------
c = df.groupby("symbol")["close"]
df["fwd10"] = c.shift(-11) / c.shift(-1) - 1.0

# ---------- 环境标签 ----------
lab = pq.read_table(REGIME).to_pandas()
if "date" not in lab.columns:
    lab = lab.reset_index()
lab["date"] = lab["date"].astype(df["date"].dtype)
df = df.merge(lab[["date", "trend20", "share_gt_ma20", "nh_nl"]], on="date", how="left")
ENV_DIMS = {"trend20": lambda v: v > 0, "share_gt_ma20": lambda v: v > 0.5, "nh_nl": lambda v: v > 0}

# ---------- 动量重叠检查: f_close_low_ma5 是否只是 r5/r10 反转的影子 ----------
g2 = df.groupby("symbol")["close"]
df["r5"] = g2.transform(lambda s: s / s.shift(5) - 1.0)
df["r10"] = g2.transform(lambda s: s / s.shift(10) - 1.0)

# ---------- 日截面 IC + 双尾超额 ----------
records = []
for dt, d in df.groupby("date", sort=True):
    m0 = d["fwd10"].notna()
    if m0.sum() < 50:
        continue
    base_mean = d.loc[m0, "fwd10"].mean()
    rec = {"date": dt, "n": int(m0.sum()), "base": base_mean}
    for f in FEATS:
        m = m0 & d[f].notna()
        x, y = d.loc[m, f], d.loc[m, "fwd10"]
        if m.sum() < 50:
            rec[f + "__ic"] = np.nan
            continue
        rec[f + "__ic"] = np.corrcoef(x.rank(), y.rank())[0, 1]
        q80, q20 = x.quantile(0.8), x.quantile(0.2)
        top = y[x >= q80]
        bot = y[x <= q20]
        rec[f + "__top"] = top.mean() - base_mean
        rec[f + "__bot"] = bot.mean() - base_mean
    # 动量对照 + 残差IC: close_low_ma5 对 (r5,r10) 截面回归后的残差 IC
    mm = m0 & d["f_close_low_ma5"].notna() & d["r5"].notna() & d["r10"].notna()
    if mm.sum() >= 50:
        sub = d.loc[mm]
        rec["ref_r5__ic"] = np.corrcoef(sub["r5"].rank(), sub["fwd10"].rank())[0, 1]
        X = np.column_stack([sub["r5"].values, sub["r10"].values, np.ones(mm.sum())])
        beta, *_ = np.linalg.lstsq(X, sub["f_close_low_ma5"].values, rcond=None)
        resid = sub["f_close_low_ma5"].values - X @ beta
        rec["f_close_low_ma5_resid__ic"] = np.corrcoef(
            pd.Series(resid).rank(), sub["fwd10"].rank())[0, 1]
        rec["corr_clv5_r5"] = np.corrcoef(sub["f_close_low_ma5"].rank(), sub["r5"].rank())[0, 1]
    records.append(rec)
ic = pd.DataFrame(records).set_index("date")
print(f"ic days={len(ic)}  n/day median={int(ic['n'].median())}")

lab2 = lab.set_index("date").reindex(ic.index)

def summarize(tag, mask):
    sub = ic[mask]
    if len(sub) < 20:
        return
    h = len(sub) // 2
    print(f"\n== {tag}  days={len(sub)} ==")
    print(f"{'feat':<18}{'IC':>8}{'t':>7}{'IC>0%':>7}{'h1':>8}{'h2':>8}{'顶20%超':>9}{'底20%超':>9}{'顶胜日%':>8}")
    for f in FEATS:
        s = sub[f + "__ic"].dropna()
        if len(s) < 20:
            continue
        tstat = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if s.std(ddof=1) > 0 else np.nan
        top = sub[f + "__top"].mean()
        bot = sub[f + "__bot"].mean()
        topwin = (sub[f + "__top"] > 0).mean()
        print(f"{f:<18}{s.mean():>+8.4f}{tstat:>7.1f}{(s>0).mean():>7.0%}"
              f"{s.iloc[:h].mean():>+8.4f}{s.iloc[h:].mean():>+8.4f}"
              f"{top:>+9.4f}{bot:>+9.4f}{topwin:>8.0%}")

d125 = ic.index >= (ic.index.max() - pd.Timedelta(days=182))
summarize("全窗 (2024-06..)", pd.Series(True, index=ic.index))
summarize("近125交易日", pd.Series(d125, index=ic.index))

# 动量重叠对照表
for tag, mask in [("全窗", pd.Series(True, index=ic.index)), ("近125d", pd.Series(d125, index=ic.index))]:
    sub = ic[mask]
    h = len(sub) // 2
    print(f"\n== 动量重叠 [{tag}] ==")
    for k in ["ref_r5__ic", "f_close_low_ma5_resid__ic", "corr_clv5_r5"]:
        s = sub[k].dropna()
        if len(s) > 20:
            print(f"{k:<28} mean={s.mean():+.4f}  h1={s.iloc[:h].mean():+.4f}  h2={s.iloc[h:].mean():+.4f}  (r5自身IC与残差IC对照, corr=close_low_ma5与r5秩相关)")

for dim, fn in ENV_DIMS.items():
    strong = fn(lab2[dim].fillna(0)).values
    summarize(f"环境[{dim}=强] 全窗", pd.Series(strong, index=ic.index))
    summarize(f"环境[{dim}=弱] 全窗", pd.Series(~strong, index=ic.index))

print("\nDONE (超额未扣成本, 全成本约0.35%/回; 判读: |IC|>=0.02 t>=2 双半窗同号=有信号; 稳定负=删查线候选)")
