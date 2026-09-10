# -*- coding: utf-8 -*-
"""
fade_score 对密度线目标的残差IC (2026-09-09 用户: "FADE SCORE 对密度目标的残差IC").
密度线 = legacy 概率头 prob_up_10d 的 TOP20 带 → 目标口径 FWD10 = C[t+11]/C[t+1]-1
(与 _hf_cl_windows_0909 / _hf_fingerprint_infotest_0909 协议一致).
协议: 日截面 Spearman; 全窗 + 近125交易日; 双半窗同号;
增量判据: fade_score 对四原料 (r20/vol20/turn20/pos250, 原料rank与原料原值两套)
截面回归后的残差 IC — |残差IC|>=0.02 t>=2 双半同号 = 叠加有用 (→pin A/B),
否则只是原料的重组 (模型已看到原件).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"

df = pq.read_table(
    PANEL, columns=["symbol", "date", "high", "low", "close", "pre_close", "turnover_rate"]
).to_pandas()
df["symbol"] = df["symbol"].astype(str)
df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

hi, lo, cl = df["high"], df["low"], df["close"]
bad = (hi < lo) | (hi < cl - 1e-9) | (lo > cl + 1e-9)
print(f"[validate] bad rows={bad.sum():,} -> excluded")
df = df[~bad.values].reset_index(drop=True)

g = df.groupby("symbol", sort=False)
ret = df["close"] / df["pre_close"] - 1.0
df["r20"] = g["close"].pct_change(20)
df["vol20"] = ret.groupby(df["symbol"]).transform(lambda s: s.rolling(20).std())
df["turn20"] = g["turnover_rate"].transform(lambda s: s.rolling(20).mean())
df["pos250"] = (
    df["close"] / g["high"].transform(lambda s: s.rolling(250, min_periods=60).max()) - 1.0
)
df["fwd10"] = g["close"].shift(-11) / g["close"].shift(-1) - 1.0

RAW = ["r20", "vol20", "turn20", "pos250"]
records = []
for dt, d in df.groupby("date", sort=True):
    m0 = d["fwd10"].notna()
    for f in RAW:
        m0 &= d[f].notna()
    if m0.sum() < 50:
        continue
    s = d.loc[m0]
    # fade_score = 全市场日截面 pct-rank 复合 (与 _fade_gate.compute_fade_profile 同口径)
    ranks = pd.DataFrame({f: s[f].rank(pct=True) for f in RAW})
    fade = ranks.mean(axis=1)
    rec = {"date": dt, "n": len(s)}
    vals = s["fwd10"].rank()
    rec["fade__ic"] = np.corrcoef(fade.rank(), vals)[0, 1]
    for f in RAW:
        rec[f + "__ic"] = np.corrcoef(ranks[f], vals)[0, 1]

    def resid_ic(target, X_df):
        X = np.column_stack([X_df[c].values for c in X_df.columns] + [np.ones(len(X_df))])
        beta, *_ = np.linalg.lstsq(X, target.values, rcond=None)
        r = target.values - X @ beta
        return np.corrcoef(pd.Series(r).rank(), vals)[0, 1]

    # 残差1: fade 对四原料 rank (组合相对自身零件的加权增量)
    rec["fade__resid_vs_ranks"] = resid_ic(fade, ranks)
    # 残差2: fade 对四原料原值 (更强判据: 连原料的非线性rank信息都扣掉)
    rec["fade__resid_vs_raw"] = resid_ic(fade, s[RAW])
    records.append(rec)

ic = pd.DataFrame(records).set_index("date")
print(f"days={len(ic)} n/day median={ic['n'].median():.0f}")

KEYS = [f + "__ic" for f in RAW] + [
    "fade__ic", "fade__resid_vs_ranks", "fade__resid_vs_raw",
]


def summ(tag, mask):
    sub = ic[mask]
    h = len(sub) // 2
    print(f"\n== {tag} days={len(sub)} ==")
    print(f"{'量':<24}{'mean':>9}{'t':>7}{'h1':>9}{'h2':>9}")
    for k in KEYS:
        s_ = sub[k].dropna()
        if len(s_) < 20:
            continue
        t_ = s_.mean() / (s_.std(ddof=1) / np.sqrt(len(s_))) if s_.std(ddof=1) > 0 else np.nan
        print(f"{k:<24}{s_.mean():>+9.4f}{t_:>7.1f}{s_.iloc[:h].mean():>+9.4f}"
              f"{s_.iloc[h:].mean():>+9.4f}")


d125 = ic.index >= (ic.index.max() - pd.Timedelta(days=182))
summ("全窗", pd.Series(True, index=ic.index))
summ("近125交易日", pd.Series(d125, index=ic.index))

# ---- 行情切片 (2026-09-09 用户: "为什么又没按市场情况分别测算") ----
# 状态=上证当日收盘距MA20 (特征日 t 可知, 无未来信息); 与 _fade_market_forecast 同四档
idx = pd.read_parquet(r"data/processed/sh_index_daily_20260909.parquet")
idx = idx.sort_values("trade_date").reset_index(drop=True)
c = idx["close"].astype(float)
ma20 = c.rolling(20).mean()
idx["dt"] = pd.to_datetime(idx["trade_date"], format="%Y%m%d")
# ic.index 的 date=特征日 (特征用该日收盘, fwd10 从 t+1 起) → 当日收盘距MA20 盘后可知, 不 shift
idx["above_ma20"] = c / ma20 - 1.0
reg = pd.Series(
    pd.cut(idx["above_ma20"], [-np.inf, -0.02, 0.0, 0.02, np.inf],
           labels=["深跌破MA20(<-2%)", "线下浅(-2~0)", "线上(0~+2%)", "线上强(>+2%)"]).values,
    index=idx["dt"].values,
)
ic2 = ic.join(reg.rename("regime"), how="inner")
ic2["weak"] = np.where(ic2["regime"].isin(["深跌破MA20(<-2%)", "线下浅(-2~0)"]), "弱市(线下)", "强市(线上)")
print(f"\n[regime] 切到 {len(ic2)}/{len(ic)} 日")
for lab, m in [("深跌破MA20(<-2%)", ic2["regime"] == "深跌破MA20(<-2%)"),
               ("线下浅(-2~0)", ic2["regime"] == "线下浅(-2~0)"),
               ("线上(0~+2%)", ic2["regime"] == "线上(0~+2%)"),
               ("线上强(>+2%)", ic2["regime"] == "线上强(>+2%)"),
               ("弱市(线下)合并", ic2["weak"] == "弱市(线下)"),
               ("强市(线上)合并", ic2["weak"] == "强市(线上)")]:
    if m.sum() >= 40:
        summ(f"行情={lab}", m)
    else:
        print(f"\n== 行情={lab} days={m.sum()} <40 跳过 ==")

print("\n判读: fade__resid_vs_* |IC|>=0.02 t>=2 双半同号 = 密度模型叠加有用 (→pin A/B);")
print("      否则 fade_score = 四原料重组, 模型已看到原件, 无注入价值")
