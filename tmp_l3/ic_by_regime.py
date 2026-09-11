# -*- coding: utf-8 -*-
"""Factor x regime interaction: cross-sectional IC of stock factors with fwd-5d
returns, conditioned on market sentiment regime terciles (High/Mid/Low by
time-series quantile). No leakage: factors use data <= t, fwd5 = t+1..t+5,
regime uses t close info.
"""
import numpy as np
import pandas as pd
from scipy import stats
import pyarrow.parquet as pq

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
REG = r"D:\AMINQT\AMINQT CODES\tmp_l3\regime_daily.csv"
N_SAMPLE, WINDOW = 500, 500

reg = pd.read_csv(REG, parse_dates=["date"])
all_dates = np.sort(reg["date"].unique())
eval_dates = all_dates[-WINDOW:]
t0 = eval_dates[0]

# ---- sample 500 eligible symbols (>=400 obs in window), seed 42
meta = pq.read_table(PANEL, columns=["symbol", "date"]).to_pandas()
meta["date"] = pd.to_datetime(meta["date"])
cnt = meta[meta["date"] >= t0].groupby("symbol")["date"].count()
elig = cnt[cnt >= 400].index.to_numpy()
rng = np.random.default_rng(42)
syms = np.sort(rng.choice(elig, size=N_SAMPLE, replace=False))
del meta, cnt
print(f"eligible={len(elig)} sampled={len(syms)} window={str(t0)[:10]}..{str(eval_dates[-1])[:10]}")

# ---- read only sampled rows, needed cols
tb = pq.read_table(PANEL, columns=["symbol", "date", "close", "high", "low", "pre_close"],
                   filters=[("date", ">=", pd.Timestamp(t0)),
                            ("symbol", "in", syms.tolist())])
df = tb.to_pandas()
df["symbol"] = df["symbol"].astype("category")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

pc = df["pre_close"].to_numpy("float64")
ok = np.isfinite(pc) & (pc > 0)
ret = np.where(ok, df["close"].to_numpy("float64") / np.where(ok, pc, 1) - 1, np.nan)
amp = np.where(ok, (df["high"].to_numpy("float64") - df["low"].to_numpy("float64")) / np.where(ok, pc, 1), np.nan)
c2l = np.where(ok, df["close"].to_numpy("float64") / df["low"].to_numpy("float64") - 1, np.nan)
df["ret"], df["amp"], df["c2l"] = ret, amp, c2l

g = df.groupby("symbol", observed=True)
r5 = (1 + df["ret"]).groupby(df["symbol"], observed=True).rolling(5).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
df["r5"] = r5
df["rev5"] = -r5
df["amp5"] = g["amp"].rolling(5).mean().reset_index(level=0, drop=True)
df["c2l5"] = g["c2l"].rolling(5).mean().reset_index(level=0, drop=True)
# fwd5 = product of next 5 daily rets (per symbol, NaN if window crosses group end)
gs = df.groupby("symbol", observed=True)["ret"]
cum = pd.Series(1.0, index=df.index)
for k in range(1, 6):
    cum = cum * (1 + gs.shift(-k))
df["fwd5"] = cum - 1
FACTORS = ["r5", "rev5", "amp5", "c2l5"]

# ---- daily cross-sectional spearman IC
ic = {}
for f in FACTORS:
    sub = df.pivot_table(index="date", columns="symbol", values=f, observed=True)
    fwd = df.pivot_table(index="date", columns="symbol", values="fwd5", observed=True)
    fwd = fwd.reindex(index=sub.index, columns=sub.columns)
    rows = {}
    for d in sub.index:
        x, y = sub.loc[d].to_numpy(), fwd.loc[d].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100:
            continue
        rho, _ = stats.spearmanr(x[m], y[m])
        rows[d] = rho
    ic[f] = pd.Series(rows)
ic = pd.DataFrame(ic)
ic = ic.loc[ic.index.isin(eval_dates)]

# ---- regime terciles on eval window (time-series quantiles of each regime col)
REGCOLS = ["limit_up_count", "limit_up_count_ma5", "blow_rate", "max_board_height", "lu_premium_1d"]
rw = reg[reg["date"].isin(eval_dates)].set_index("date")
q = rw[REGCOLS].quantile([1/3, 2/3])
terc = {}
for c in REGCOLS:
    t_ = pd.Series("Mid", index=rw.index)
    t_[rw[c] < q.loc[1/3, c]] = "Low"
    t_[rw[c] > q.loc[2/3, c]] = "High"
    terc[c] = t_

def ts(x):  # t-stat of mean != 0
    x = x.dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan

out = []
for c in REGCOLS:
    for f in FACTORS:
        s = terc[c].to_frame("b").join(ic[[f]])
        row = {"regime_col": c, "factor": f}
        for b in ["Low", "Mid", "High"]:
            v = s.loc[s["b"] == b, f]
            row[f"ic_{b}"] = round(v.mean(), 4)
            row[f"t_{b}"] = round(ts(v), 2)
            row[f"n_{b}"] = len(v.dropna())
        lo, hi = s.loc[s["b"] == "Low", f].dropna(), s.loc[s["b"] == "High", f].dropna()
        row["dHiLo"] = round(hi.mean() - lo.mean(), 4)
        row["t_dHiLo"] = round((hi.mean() - lo.mean()) / np.sqrt(hi.var(ddof=1)/len(hi) + lo.var(ddof=1)/len(lo)), 2)
        out.append(row)
res = pd.DataFrame(out)
res.to_csv(r"D:\AMINQT\AMINQT CODES\tmp_l3\ic_by_regime.csv", index=False)
print("\n=== Factor IC x regime tercile (mean IC / t / n) ===")
print(res.to_string(index=False))

# ---- secondary: regime col vs market fwd-5d (time-series corr)
fwd_mkt5 = (1 + rw["mkt_ret"]).rolling(5).apply(np.prod, raw=True).shift(-5) - 1
ts_rows = []
for c in REGCOLS + ["touched_count"]:
    m = np.isfinite(rw[c]) & np.isfinite(fwd_mkt5)
    pr = stats.pearsonr(rw[c][m], fwd_mkt5[m])
    sp = stats.spearmanr(rw[c][m], fwd_mkt5[m])
    ts_rows.append({"regime_col": c, "n": int(m.sum()),
                    "pearson": round(pr[0], 4), "p": round(pr[1], 4),
                    "spearman": round(sp[0], 4), "p_sp": round(sp[1], 4)})
tsdf = pd.DataFrame(ts_rows)
tsdf.to_csv(r"D:\AMINQT\AMINQT CODES\tmp_l3\regime_ts_corr.csv", index=False)
print("\n=== Regime vs market fwd5 (time-series) ===")
print(tsdf.to_string(index=False))

# full-window factor IC baseline
print("\n=== Factor IC full window ===")
for f in FACTORS:
    print(f, round(ic[f].mean(), 4), "t=", round(ts(ic[f]), 2))
