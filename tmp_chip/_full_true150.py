# -*- coding: utf-8 -*-
"""真 150 档 CYQ 分布下的筹码缺口特征全量验证 (只读面板, 只写 tmp_chip/_full*).

背景: proto 单桶近似下 ovd_wdist TS IC +0.124 / ovd_15p +0.102.
本脚本从 150 档 xdata 精确重算 (与 app/pipeline1/cyq_calculator.py 生产算法
完全同源: 120日窗 + 换手衰减 + 三角形分布 + 一字板特判), 5% 价格带聚合.

样本: np.random.seed(42) 抽 500 只 × 最近 500 交易日 (120 预热 + 5 前向).
目标: r5 = close_hfq(t+5)/close_hfq(t)-1.
评估: 逐股 TS IC / t / 残差IC(两套控制集) / 前后半窗.
达标线: |t|>3 且残差 |t|>3 且双半窗同号.
"""

import time

import numpy as np
import pandas as pd
from scipy import stats

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUT_DIR = r"D:/AMINQT/AMINQT CODES/tmp_chip"

FACTOR = 150          # cyq_calculator.FACTOR
RANGE_DAYS = 120      # cyq_calculator.RANGE_DAYS
WARM = RANGE_DAYS
EVAL_N = 500
N_STOCKS = 500
BAND = 0.05
NB_UP = 30            # 上方带数 (band 0..29, 与原型 clip 一致)

np.random.seed(42)

NEED = WARM + EVAL_N + 5          # 625
SLACK = 10                        # 允许的停牌缺行

# ---------- 1. 选样 (与原型同法: DMIN 起行数够) ----------
dts = pd.read_parquet(PANEL, columns=["date"])["date"]
all_dates = np.sort(dts.unique())
DMIN = pd.Timestamp(all_dates[-(NEED + SLACK + 30)])
print(f"panel dates: {all_dates[0]} .. {all_dates[-1]}, DMIN={DMIN.date()}")

light = pd.read_parquet(PANEL, columns=["symbol", "date"], filters=[("date", ">=", DMIN)])
cnt = light.groupby("symbol").size()
last_ok = light.groupby("symbol")["date"].max() >= pd.Timestamp(all_dates[-8])
eligible = cnt[(cnt >= NEED - SLACK) & last_ok].index.tolist()
rng = np.random.default_rng(42)
syms = sorted(rng.choice(eligible, size=min(N_STOCKS, len(eligible)), replace=False))
print(f"eligible={len(eligible)} sampled={len(syms)}")

need_cols = [
    "symbol", "date", "open", "high", "low", "close", "close_hfq", "turnover_rate",
    "winner_ratio", "pct_90_con", "cost_50pct", "peak_roc_5d", "chip_entropy",
    "resistance_dist",
]
df = pd.read_parquet(
    PANEL, columns=need_cols,
    filters=[("symbol", "in", syms), ("date", ">=", DMIN)],
).reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

AR = np.arange(FACTOR)

# ---------- 2. 逐股: 真 150 档分布 + 缺口特征 ----------
rows = []
fid = []   # wr_true vs panel winner_ratio 逐股 spearman
t0 = time.time()
done = 0
for sym, g in df.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    g = g.dropna(subset=["open", "high", "low", "close", "turnover_rate"])
    n = len(g)
    if n < NEED - SLACK:
        continue
    o, hi, lo, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    hsl = np.clip(np.nan_to_num(g["turnover_rate"].to_numpy(float)) / 100.0, 0.0, 1.0)

    start = n - EVAL_N - 5
    rec = {k: np.full(EVAL_N, np.nan) for k in (
        "ovd_wdist", "ovd_wdist_ex", "ovd_15p", "ovd_total", "wr_true")}

    for i, T in enumerate(range(start, start + EVAL_N)):
        s0 = max(0, T - RANGE_DAYS + 1)
        hh, ll, oo, cc = hi[s0:T + 1], lo[s0:T + 1], o[s0:T + 1], c[s0:T + 1]
        hw = hsl[s0:T + 1]
        cT = c[T]
        if not np.isfinite(cT) or cT <= 0:
            continue
        maxp, minp = float(hh.max()), float(ll.min())
        acc = max(0.01, (maxp - minp) / (FACTOR - 1))
        avg = (oo + hh + ll + cc) / 4.0
        grid = minp + acc * AR
        gr = grid[None, :]

        # 三角形权重 (生产算法向量化副本)
        flat = hh <= ll
        rngv = np.maximum(hh - ll, 1e-12)
        left = gr <= avg[:, None]
        num = np.where(left, gr - ll[:, None], hh[:, None] - gr)
        den = np.where(left, (avg - ll)[:, None], (hh - avg)[:, None])
        den = np.where(np.abs(den) < 1e-12, 1.0, den)
        w = num / den
        mask = (gr >= ll[:, None]) & (gr <= hh[:, None])
        density = np.where(flat, float(FACTOR - 1), 2.0 / rngv)
        w = np.where(mask, w, 0.0) * (density * hw)[:, None]
        if flat.any():
            w[flat] = 0.0
            ab = np.floor((avg - minp) / acc).astype(int)
            fi = np.nonzero(flat & (ab >= 0) & (ab < FACTOR))[0]
            w[fi, ab[fi]] += (FACTOR - 1) * hw[fi] / 2.0

        # 窗内倒序累积衰减 (与生产顺序衰减逐日等价, 且无全局下溢)
        D = np.empty(len(hw))
        D[:-1] = np.cumprod((1.0 - hw)[::-1])[-2::-1]
        D[-1] = 1.0
        xdata = np.maximum((w * D[:, None]).sum(axis=0), 0.0)
        tot = xdata.sum()
        if tot <= 1e-12:
            continue

        d = (grid - cT) / cT
        abv = d > 0.0                      # 现价上方档 (grid > close, 生产口径)
        mab = float(xdata[abv].sum())
        if mab > 1e-12:
            band = np.clip(np.floor(d[abv] / BAND).astype(int), 0, NB_UP - 1)
            mb = np.bincount(band, weights=xdata[abv], minlength=NB_UP)
            rec["ovd_wdist"][i] = float(
                (mb * (np.arange(NB_UP) + 0.5) * BAND).sum() / max(mb.sum(), 1e-12))
            rec["ovd_wdist_ex"][i] = float((xdata[abv] * d[abv]).sum() / mab)
        rec["ovd_15p"][i] = float(xdata[d >= 0.15 - 1e-9].sum() / tot)
        rec["ovd_total"][i] = mab / tot
        rec["wr_true"][i] = float(xdata[d <= 0.0].sum() / tot)

    out = pd.DataFrame(rec)
    out["date"] = g["date"].iloc[start:start + EVAL_N].to_numpy()
    for k in ("close", "close_hfq", "winner_ratio", "pct_90_con", "cost_50pct",
              "peak_roc_5d", "chip_entropy", "resistance_dist"):
        out[k] = g[k].iloc[start:start + EVAL_N].to_numpy()
    out["symbol"] = sym
    ok = out["wr_true"].notna() & out["winner_ratio"].notna()
    if ok.sum() > 50:
        fid.append(stats.spearmanr(out.loc[ok, "wr_true"],
                                   out.loc[ok, "winner_ratio"]).statistic)
    rows.append(out)
    done += 1
    if done % 50 == 0:
        el = time.time() - t0
        print(f"[{done}/{len(syms)}] {el:.0f}s elapsed, {el / done:.2f}s/stock",
              flush=True)

d = pd.concat(rows, ignore_index=True)
print(f"\nfull panel: {len(d)} rows, {d['symbol'].nunique()} stocks, "
      f"{time.time() - t0:.0f}s")
print(f"fidelity (wr_true vs panel winner_ratio, per-stock spearman): "
      f"mean={np.mean(fid):.4f} min={np.min(fid):.4f} n={len(fid)}")

# ---------- 3. 目标 / 控制变量 ----------
d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
grp = d.groupby("symbol", sort=False)
d["r5"] = grp["close_hfq"].shift(-5) / d["close_hfq"] - 1.0
d["ret5_bwd"] = d["close_hfq"] / grp["close_hfq"].shift(5) - 1.0
d["cost_bias"] = (d["close"] - d["cost_50pct"]) / d["cost_50pct"].replace(0, np.nan)

GAP_FEATS = ["ovd_wdist", "ovd_wdist_ex", "ovd_15p", "ovd_total"]
CONTROLS_TASK = ["winner_ratio", "cost_bias", "ret5_bwd"]                    # 任务口径
CONTROLS_PROTO = ["winner_ratio", "pct_90_con", "cost_bias", "peak_roc_5d",  # 原型口径
                  "chip_entropy", "resistance_dist", "ret5_bwd"]
d = d.dropna(subset=["r5"]).reset_index(drop=True)


def stock_ic(dd, f, y):
    ok = dd[f].notna() & dd[y].notna()
    if ok.sum() < 40 or dd.loc[ok, f].nunique() < 2:
        return np.nan
    return stats.spearmanr(dd.loc[ok, f], dd.loc[ok, y]).statistic


def resid_ic(dd, f, controls):
    cols = list(dict.fromkeys(controls + [f, "r5"]))
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
for f in GAP_FEATS:
    ics, r3, r7, h1, h2 = [], [], [], [], []
    for _, dd in d.groupby("symbol"):
        dd = dd.sort_values("date")
        ics.append(stock_ic(dd, f, "r5"))
        r3.append(resid_ic(dd, f, CONTROLS_TASK))
        r7.append(resid_ic(dd, f, CONTROLS_PROTO))
        m = len(dd)
        h1.append(stock_ic(dd.iloc[: m // 2], f, "r5"))
        h2.append(stock_ic(dd.iloc[m // 2:], f, "r5"))
    ics, r3, r7 = map(lambda a: np.array(a, float), (ics, r3, r7))
    h1, h2 = np.array(h1, float), np.array(h2, float)

    def _t(a):
        return np.nanmean(a) / (np.nanstd(a, ddof=1) / np.sqrt(np.isfinite(a).sum()))

    same = np.nanmean(h1) * np.nanmean(h2) > 0
    res.append({
        "feat": f, "ts_ic": round(np.nanmean(ics), 4), "t": round(_t(ics), 2),
        "resid3_ic": round(np.nanmean(r3), 4), "resid3_t": round(_t(r3), 2),
        "resid7_ic": round(np.nanmean(r7), 4), "resid7_t": round(_t(r7), 2),
        "h1_ic": round(np.nanmean(h1), 4), "h2_ic": round(np.nanmean(h2), 4),
        "half_same_sign": bool(same),
        "pass": bool(abs(_t(ics)) > 3 and abs(_t(r3)) > 3 and same),
    })

res = pd.DataFrame(res)
res.to_csv(f"{OUT_DIR}/_full_results.csv", index=False, encoding="utf-8-sig")
d.to_parquet(f"{OUT_DIR}/_full_features.parquet", index=False)

print("\n===== TRUE-150 RESULTS (500 stocks x 500 days) =====")
print(res.to_string(index=False))
print("\nPROXY ref: ovd_wdist IC +0.124 t=8.9 resid +0.0102 t=6.5 | "
      "ovd_15p IC +0.102 t=6.9 resid +0.0081 t=4.9")
print("PASS:", res["pass"].tolist())
