# -*- coding: utf-8 -*-
"""
fade_score 对密度线生产模型特征空间的残差IC (2026-09-09 用户: "为什么又没按市场情况分别测算").
生产 207 特征中 157 列训练时由特征引擎动态生成 (面板无现成列), 逐一复刻不现实;
对照集 = 面板已有的 50 列模型特征 + 族覆盖网格 (覆盖 fade 四原料的全部 horizon/变形:
  动量 r{1..250}×8 + bias6档 + 波动 std{5,20,60} + 换手 ma{5,20,60}+rank + 位置
  close_vs_high/pct90/range_pos + 量比×2 + 振幅×2 + 隔夜).
判据: 某行情带 |残差IC|>=0.02 t>=2 双半同号 => 该带 fade 携带模型未见信息 (→pin A/B);
否则模型已见原件 (它有 ROC/bias/ATR/turn_rank/close_vs_high 全家).
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
FEATS = json.load(open("tmp_t/_neg200_features.json", encoding="utf-8"))

schema = pq.read_schema(PANEL)
base = ["symbol", "date", "open", "high", "low", "close", "pre_close",
        "turnover_rate", "volume"]
HAVE = [c for c in FEATS if c in schema.names and c not in base]
print(f"[cov] 模型特征面板直取 {len(HAVE)}/{len(FEATS)}; 其余用族网格代理")

tb = pq.read_table(PANEL, columns=base + HAVE)
df = tb.to_pandas()
del tb
df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)
for c in base[2:]:
    df[c] = df[c].astype(np.float64)
if HAVE:
    df[HAVE] = df[HAVE].astype(np.float32)
df["symbol"] = df["symbol"].astype(str)
df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

hi, lo, cl = df["high"], df["low"], df["close"]
bad = (hi < lo) | (hi < cl - 1e-9) | (lo > cl + 1e-9)
df = df[~bad.values].reset_index(drop=True)

g = df.groupby("symbol", sort=False)
ret = df["close"] / df["pre_close"] - 1.0

# ---- 族覆盖网格 (fade 原料空间全集代理) ----
for n in [1, 3, 5, 10, 20, 60, 120, 250]:
    df[f"g_r{n}"] = g["close"].pct_change(n)
for n in [5, 20, 60]:
    df[f"g_std{n}"] = ret.groupby(df["symbol"]).transform(lambda s: s.rolling(n).std())
for n in [5, 20, 60]:
    df[f"g_turn{n}"] = g["turnover_rate"].transform(lambda s: s.rolling(n).mean())
df["g_turn_rank20"] = df["g_turn20"].groupby(df["date"]).rank(pct=True)
for n in [5, 10, 20, 60, 120, 250]:
    ma = g["close"].transform(lambda s: s.rolling(n).mean())
    df[f"g_bias{n}"] = df["close"] / ma - 1.0
mx250 = g["high"].transform(lambda s: s.rolling(250, min_periods=60).max())
mn250 = g["low"].transform(lambda s: s.rolling(250, min_periods=60).min())
df["g_cvhigh250"] = df["close"] / mx250 - 1.0
df["g_pctl250"] = (df["close"] - mn250) / (mx250 - mn250)
for n in [5, 20]:
    df[f"g_vratio{n}"] = df["volume"] / g["volume"].transform(
        lambda s: s.rolling(n).mean())
    df[f"g_amp{n}"] = ((df["high"] - df["low"]) / df["pre_close"]).groupby(
        df["symbol"]).transform(lambda s: s.rolling(n).mean())
df["g_overnight"] = df["open"] / df["pre_close"] - 1.0

GRID = [c for c in df.columns if c.startswith("g_")]
df[GRID] = df[GRID].replace([np.inf, -np.inf], np.nan)
print(f"[grid] 族网格 {len(GRID)} 列 + 模型直取 {len(HAVE)} 列")

# fade 原料 + 目标
df["r20"] = df["g_r20"]
df["vol20"] = df["g_std20"]
df["turn20"] = df["g_turn20"]
df["pos250"] = df["g_cvhigh250"]
df["fwd10"] = g["close"].shift(-11) / g["close"].shift(-1) - 1.0
RAW = ["r20", "vol20", "turn20", "pos250"]

CTRL = GRID + HAVE
XALL = df[CTRL].astype(np.float32)
KEEP = ["date", "fwd10"] + RAW
df = df[KEEP]

records = []
for dt, idx in df.groupby("date", sort=True).indices.items():
    d = df.iloc[idx]
    m0 = d["fwd10"].notna()
    for f in RAW:
        m0 &= d[f].notna()
    if m0.sum() < 600:
        continue
    X = XALL.iloc[idx[m0]]
    X = X.fillna(X.median()).dropna(axis=1, how="any")
    if X.shape[1] < 40 or len(X) < X.shape[1] + 100:
        continue
    s = d.loc[m0.values, ["fwd10"] + RAW]
    ranks = pd.DataFrame({f: s[f].rank(pct=True) for f in RAW})
    fade = ranks.mean(axis=1)
    vals = s["fwd10"].rank()
    rec = {"date": dt, "n": len(s), "k": X.shape[1]}
    rec["fade__ic"] = np.corrcoef(fade.rank(), vals)[0, 1]
    M = np.column_stack([X.values, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(M, fade.values, rcond=None)
    r = fade.values - M @ beta
    rec["fade__resid_vs_model"] = np.corrcoef(pd.Series(r).rank(), vals)[0, 1]
    beta2, *_ = np.linalg.lstsq(M, vals.values.astype(float), rcond=None)
    r2 = vals.values.astype(float) - M @ beta2
    rec["fwd_resid_vs_model"] = np.corrcoef(fade.rank(), pd.Series(r2).rank())[0, 1]
    records.append(rec)

ic = pd.DataFrame(records).set_index("date")
print(f"days={len(ic)} n/day median={ic['n'].median():.0f} k median={ic['k'].median():.0f}")

idx2 = pd.read_parquet(r"data/processed/sh_index_daily_20260909.parquet")
idx2 = idx2.sort_values("trade_date").reset_index(drop=True)
c = idx2["close"].astype(float)
idx2["above_ma20"] = c / c.rolling(20).mean() - 1.0
idx2["dt"] = pd.to_datetime(idx2["trade_date"], format="%Y%m%d")
reg = pd.Series(
    pd.cut(idx2["above_ma20"], [-np.inf, -0.02, 0.0, 0.02, np.inf],
           labels=["深跌破MA20(<-2%)", "线下浅(-2~0)", "线上(0~+2%)", "线上强(>+2%)"]).values,
    index=idx2["dt"].values,
)
ic = ic.join(reg.rename("regime"), how="inner")
ic["weak"] = np.where(ic["regime"].isin(["深跌破MA20(<-2%)", "线下浅(-2~0)"]), "弱市", "强市")
print(f"[regime] 切到 {len(ic)} 日")


def summ(tag, mask):
    sub = ic[mask]
    h = len(sub) // 2
    print(f"\n== {tag} days={len(sub)} ==")
    for k in ["fade__ic", "fade__resid_vs_model", "fwd_resid_vs_model"]:
        s_ = sub[k].dropna()
        if len(s_) < 20:
            continue
        t_ = s_.mean() / (s_.std(ddof=1) / np.sqrt(len(s_))) if s_.std(ddof=1) > 0 else np.nan
        print(f"{k:<24}{s_.mean():>+9.4f}{t_:>7.1f}{s_.iloc[:h].mean():>+9.4f}"
              f"{s_.iloc[h:].mean():>+9.4f}")


summ("全窗", pd.Series(True, index=ic.index))
for lab in ["深跌破MA20(<-2%)", "线下浅(-2~0)", "线上(0~+2%)", "线上强(>+2%)", "弱市", "强市"]:
    m = (ic["weak"] == lab) if lab in ("弱市", "强市") else (ic["regime"] == lab)
    if m.sum() >= 40:
        summ(f"行情={lab}", m)
    else:
        print(f"\n== 行情={lab} days={m.sum()} <40 跳过 ==")

print("\n判读: fade__resid_vs_model 某带 |IC|>=0.02 t>=2 双半同号 = 该带模型未见信息 (→pin A/B);")
print("      fwd_resid_vs_model = 反向口径 (模型残差收益里 fade 还剩多少), 互为印证")
