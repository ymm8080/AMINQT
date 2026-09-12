# -*- coding: utf-8 -*-
"""筹码/对手盘维度缺口原型验证 (只读, 只写 tmp_chip/).

缺口 A: 上方套牢盘密度分档 — 现价上方每5%价格区间的换手衰减质量占比
        (cyq_ext 的 mass_above_* 从未入面板, 分档口径完全缺失)
缺口 B: 获利盘导数 + 出清完成判定 (winner_ratio 低水平 x 不再减少)
        (registry 440 中无任何 winner_ratio 导数; 只有 wr_sm5 平滑)

口径: 换手率衰减单桶近似 (typical price 落桶, 120日窗, 与 CYQ 算法同衰减律).
先验证近似保真度: proxy 获利盘 vs 面板 winner_ratio 的逐股 spearman.

样本: np.random.seed(42) 抽 100 只, 最近 250 交易日 (120 日预热 + 5 日前向).
评估: r5 = close(t+5)/close(t)-1 的逐股 TS IC / t / 残差IC / 前后半窗.
达标线: |t|>3 且双半窗同号.
"""

import numpy as np
import pandas as pd
from scipy import stats

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
CYQ = r"D:/AMINQT/AMINQT CODES/data/cyq_panel.parquet"
OUT_DIR = r"D:/AMINQT/AMINQT CODES/tmp_chip"
DATE_MIN = "2024-06-01"
N_STOCKS = 100
WARM = 120
EVAL_N = 250
BAND = 0.05
NPB = 50  # bincount 桶数, band 平移 +10

np.random.seed(42)

# ---------- 1. 选样 ----------
import datetime as _dt
DMIN = pd.Timestamp(_dt.date.fromisoformat(DATE_MIN))
light = pd.read_parquet(
    PANEL, columns=["symbol", "date"], filters=[("date", ">=", DMIN)]
)
cnt = light.groupby("symbol").size()
eligible = cnt[cnt >= WARM + EVAL_N + 30].index.tolist()
rng = np.random.default_rng(42)
syms = sorted(rng.choice(eligible, size=min(N_STOCKS, len(eligible)), replace=False))
print(f"eligible={len(eligible)} sampled={len(syms)}")

need_cols = [
    "symbol", "date", "open", "high", "low", "close", "turnover_rate",
    "winner_ratio", "pct_90_con", "cost_50pct", "cost_95pct",
    "peak_roc_5d", "chip_entropy", "resistance_dist", "support_dist",
]
df = pd.read_parquet(
    PANEL, columns=need_cols,
    filters=[("symbol", "in", syms), ("date", ">=", DMIN)],
).reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

# Kimi 口径 90% 宽度对照需要 cost_5pct (面板无, 从 calculator cyq 面板 join)
cyq5 = pd.read_parquet(
    CYQ, columns=["symbol", "date", "cost_5pct"],
    filters=[("symbol", "in", syms), ("date", ">=", DMIN)],
).reset_index(drop=True)
cyq5["date"] = pd.to_datetime(cyq5["date"])
df = df.merge(cyq5, on=["symbol", "date"], how="left")

# ---------- 2. 缺口特征计算 ----------
rows = []
fidelity = []  # proxy winner vs panel winner 逐股 spearman
for sym, g in df.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    n = len(g)
    if n < WARM + EVAL_N + 30:
        continue
    o, hi, lo, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    h = np.clip(np.nan_to_num(g["turnover_rate"].to_numpy(float)) / 100.0, 0.0, 1.0)
    p = (o + hi + lo + c) / 4.0
    U = np.maximum(np.cumprod(1.0 - h), 1e-12)

    start = n - EVAL_N - 5
    rec = {k: np.full(EVAL_N, np.nan) for k in (
        "ovd_0_5", "ovd_5_15", "ovd_15p", "ovd_total", "ovd_wdist", "wr_proxy")}

    for i, T in enumerate(range(start, start + EVAL_N)):
        j0 = max(0, T - WARM + 1)
        sl = slice(j0, T + 1)
        w = h[sl] * U[T] / U[sl]
        band = np.floor((p[sl] - c[T]) / (BAND * c[T])).astype(int)
        band = np.clip(band, -10, NPB - 11)
        m = np.bincount(band + 10, weights=w, minlength=NPB)
        tot = m.sum()
        if tot <= 1e-12:
            continue
        rec["ovd_0_5"][i] = m[10] / tot
        rec["ovd_5_15"][i] = (m[11] + m[12]) / tot
        rec["ovd_15p"][i] = m[13:].sum() / tot
        rec["ovd_total"][i] = m[10:].sum() / tot
        ov = m[10:]
        wd = float(np.sum(ov * (np.arange(len(ov)) + 0.5) * BAND)) / max(ov.sum(), 1e-12)
        rec["ovd_wdist"][i] = wd
        rec["wr_proxy"][i] = m[:10].sum() / tot

    out = pd.DataFrame(rec)
    out["date"] = g["date"].iloc[start:start + EVAL_N].to_numpy()
    for k in ("close", "winner_ratio", "pct_90_con", "cost_50pct", "cost_95pct",
              "cost_5pct", "peak_roc_5d", "chip_entropy", "resistance_dist",
              "support_dist"):
        out[k] = g[k].iloc[start:start + EVAL_N].to_numpy()
    out["symbol"] = sym

    # 近似保真度
    ok = out["wr_proxy"].notna() & out["winner_ratio"].notna()
    if ok.sum() > 50:
        fidelity.append(stats.spearmanr(out.loc[ok, "wr_proxy"],
                                        out.loc[ok, "winner_ratio"]).statistic)
    rows.append(out)

d = pd.concat(rows, ignore_index=True)
print(f"proto panel: {len(d)} rows, {d['symbol'].nunique()} stocks")
print(f"approx fidelity (proxy wr vs panel wr, per-stock spearman): "
      f"mean={np.mean(fidelity):.3f} min={np.min(fidelity):.3f}")

# ---------- 3. 目标 / 基线 / 缺口特征 ----------
d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
grp = d.groupby("symbol", sort=False)
d["r5"] = grp["close"].shift(-5) / d["close"] - 1.0
d["ret5_bwd"] = d["close"] / grp["close"].shift(5) - 1.0
d["cost_bias"] = (d["close"] - d["cost_50pct"]) / d["cost_50pct"].replace(0, np.nan)
d["wr_chg5"] = grp["winner_ratio"].diff(5)
d["wr_sm5"] = grp["winner_ratio"].transform(lambda s: s.rolling(5).mean())
d["wr_chg20"] = grp["winner_ratio"].diff(20)
# 出清完成: 获利盘<30% 且 5/20 日不再减少
d["wr_clear_evt"] = ((d["winner_ratio"] < 0.30) & (d["wr_chg5"] >= 0)).astype(float)
d["wr_clear_evt20"] = ((d["winner_ratio"] < 0.30) & (d["wr_chg20"] >= 0)).astype(float)
# 连续版: 低获利盘深度 x 企稳幅度
d["wr_low_flat"] = np.clip(0.30 - d["winner_ratio"], 0, None) * np.clip(d["wr_chg5"], 0, None)
# Kimi 90% 宽度口径 (对照)
d["kimi_width"] = (d["cost_95pct"] - d["cost_5pct"]) / d["cost_50pct"].replace(0, np.nan)

GAP_FEATS = ["ovd_0_5", "ovd_5_15", "ovd_15p", "ovd_total", "ovd_wdist",
             "wr_chg5", "wr_chg20", "wr_clear_evt", "wr_clear_evt20", "wr_low_flat"]
print("evt rates: clear5={:.1%} clear20={:.1%} wr<0.3={:.1%}".format(
    d["wr_clear_evt"].mean(), d["wr_clear_evt20"].mean(),
    (d["winner_ratio"] < 0.30).mean()))
BASE_FEATS = ["winner_ratio", "wr_sm5", "pct_90_con", "cost_bias",
              "resistance_dist", "peak_roc_5d", "kimi_width"]
CONTROLS = ["winner_ratio", "pct_90_con", "cost_bias", "peak_roc_5d",
            "chip_entropy", "resistance_dist", "ret5_bwd"]

d = d.dropna(subset=["r5"]).reset_index(drop=True)


def stock_ic(dd, f, y):
    ok = dd[f].notna() & dd[y].notna()
    if ok.sum() < 40 or dd.loc[ok, f].nunique() < 2:
        return np.nan
    return stats.spearmanr(dd.loc[ok, f], dd.loc[ok, y]).statistic


def resid_ic(dd, f):
    cols = list(dict.fromkeys(CONTROLS + [f, "r5"]))
    dd2 = dd[cols].dropna()
    if len(dd2) < 60 or dd2[f].nunique() < 2:
        return np.nan
    R = dd2.rank(pct=True).to_numpy()
    y, Xc, xf = R[:, -1], R[:, :-1], R[:, -2]
    X = np.column_stack([np.ones(len(Xc)), Xc])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return stats.spearmanr(xf, resid).statistic


res = []
for f in GAP_FEATS + BASE_FEATS:
    ics, rics, h1, h2 = [], [], [], []
    for sym, dd in d.groupby("symbol"):
        dd = dd.sort_values("date")
        ics.append(stock_ic(dd, f, "r5"))
        rics.append(resid_ic(dd, f))
        m = len(dd)
        h1.append(stock_ic(dd.iloc[: m // 2], f, "r5"))
        h2.append(stock_ic(dd.iloc[m // 2:], f, "r5"))
    ics, rics = np.array(ics, float), np.array(rics, float)
    h1, h2 = np.array(h1, float), np.array(h2, float)
    t = np.nanmean(ics) / (np.nanstd(ics, ddof=1) / np.sqrt(np.isfinite(ics).sum()))
    rt = np.nanmean(rics) / (np.nanstd(rics, ddof=1) / np.sqrt(np.isfinite(rics).sum()))
    same = (np.nanmean(h1) * np.nanmean(h2) > 0)
    res.append({
        "feat": f, "ts_ic": round(np.nanmean(ics), 4), "t": round(t, 2),
        "resid_ic": round(np.nanmean(rics), 4), "resid_t": round(rt, 2),
        "h1_ic": round(np.nanmean(h1), 4), "h2_ic": round(np.nanmean(h2), 4),
        "half_same_sign": bool(same),
        "pass": bool(abs(t) > 3 and same and abs(np.nanmean(ics)) > 0.01),
    })

res = pd.DataFrame(res)
res["type"] = ["GAP"] * len(GAP_FEATS) + ["BASE(ref)"] * len(BASE_FEATS)
res.to_csv(f"{OUT_DIR}/proto_results.csv", index=False, encoding="utf-8-sig")
d.to_parquet(f"{OUT_DIR}/proto_features.parquet", index=False)

# Kimi 口径与现行 pct_90_con 等价性
ok = d["kimi_width"].notna() & d["pct_90_con"].notna()
eq = [stats.spearmanr(dd["kimi_width"], dd["pct_90_con"]).statistic
      for _, dd in d[ok].groupby("symbol") if len(dd) > 50]
print(f"\nkimi_width vs pct_90_con per-stock spearman: mean={np.mean(eq):.3f}")
print("\n===== RESULTS =====")
print(res.to_string(index=False))
print("\nGAP pass:", res[res["type"] == "GAP"]["pass"].tolist())
