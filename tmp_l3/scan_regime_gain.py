# -*- coding: utf-8 -*-
"""全场景 regime 信息增益扫描 (tmp_l3 / _scan 前缀, 不覆盖既有文件).

轴1 场景 x 轴2 因子族 x 轴3 horizon:
  A) regime 直连: 场景内按 regime 列高/低三分位分日, 个股前向收益的日均值差 (spread + Welch t)
  B) 因子 IC 调制: 场景内逐日截面 Spearman IC (rank-Pearson), regime 高/低日 IC 差 (ΔIC + Welch Δt)
  C) top20 by |Δt| + 双半窗稳定性 + 多检验校正 (Δt/√N_eff 缩水 + Bonferroni z 阈值)

无泄漏: 因子/场景/regime 全用 <=t 信息; 前向收益 close_hfq(t+k)/close_hfq(t)-1, k=1/3/5/10.
样本: eligible(>=400 obs) 抽 500 只 seed42 (default_rng, 与 tmp_l3 前作同口径), 最近500交易日+40预热.
"""
import json
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
REG = r"D:\AMINQT\AMINQT CODES\tmp_l3\regime_daily.csv"
OUT = r"D:\AMINQT\AMINQT CODES\tmp_l3"
N_SAMPLE, EVAL_DAYS, WARMUP = 500, 500, 40
MIN_CS = 10          # 场景内逐日截面最少股票数 (Spearman IC)
MIN_DAY_DIRECT = 5   # 直连测试逐日最少股票数
REGCOLS = ["limit_up_count", "limit_up_count_ma5", "blow_rate",
           "max_board_height", "lu_premium_1d"]
FACTORS = ["mom5", "mom10", "mom20", "rev5", "std5", "std20", "amp5",
           "to_level", "to_trend", "amihud20", "vp_div", "gap", "c2l5",
           "winner_ratio", "cost_bias", "up_days"]
HZ = [1, 3, 5, 10]

# ---------- 1. 窗口与抽样 ----------
reg = pd.read_csv(REG, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
reg_dates = reg["date"].to_numpy()
eval_dates = reg_dates[-EVAL_DAYS:]
t_eval0 = pd.Timestamp(eval_dates[0])

meta = pq.read_table(PANEL, columns=["symbol", "date"]).to_pandas()
meta["date"] = pd.to_datetime(meta["date"])
cnt = meta[meta["date"] >= t_eval0].groupby("symbol")["date"].count()
elig = cnt[cnt >= 400].index.to_numpy()
rng = np.random.default_rng(42)
syms = np.sort(rng.choice(elig, size=N_SAMPLE, replace=False))
del meta, cnt
print(f"[sample] eligible={len(elig)} sampled={len(syms)} "
      f"eval={str(t_eval0)[:10]}..{str(eval_dates[-1])[:10]}")

# ---------- 2. 读面板 (预过滤: 列+日期+抽样股) ----------
cols = ["symbol", "date", "open", "high", "low", "close", "pre_close", "close_hfq",
        "volume", "amount", "turnover_rate", "winner_ratio", "cost_bias"]
load0 = t_eval0 - pd.Timedelta(days=WARMUP * 3)  # 日历日放大保证 >=40 交易日
tb = pq.read_table(PANEL, columns=cols,
                   filters=[("date", ">=", load0), ("symbol", "in", syms.tolist())])
df = tb.to_pandas()
del tb
df["date"] = pd.to_datetime(df["date"])
# 截到 eval0 前至少 WARMUP 个交易日
all_d = np.sort(df["date"].unique())
warm_dates = all_d[all_d < t_eval0]
if len(warm_dates) > WARMUP:
    keep0 = warm_dates[-WARMUP]
else:
    keep0 = warm_dates[0] if len(warm_dates) else t_eval0
df = df[df["date"] >= keep0].sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
print(f"[load] rows={len(df)} window={str(df['date'].min())[:10]}..{str(df['date'].max())[:10]} "
      f"to_NaN={df['turnover_rate'].isna().mean():.3f}")

# ---------- 3. 逐股特征 (组内全向量化) ----------
def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date")
    c, pc, low, hi, op = g["close"], g["pre_close"], g["low"], g["high"], g["open"]
    hfq, to, amt = g["close_hfq"], g["turnover_rate"], g["amount"]
    sym = str(g["symbol"].iloc[0])
    out = pd.DataFrame(index=g.index)
    thr = 0.198 if sym.startswith(("30", "68")) else 0.098
    lim = ((c / pc - 1) >= thr).fillna(False)
    brk = lim.shift(1, fill_value=False) & ~lim
    grp = brk.cumsum()
    pos = brk.groupby(grp).cumcount()
    dsb = pos.where(grp > 0)
    out["dsb"] = dsb.where(dsb <= 20)                       # 断板后0-20日
    out["lim_prev"] = lim.shift(1, fill_value=False)         # 涨停次日
    streak = lim.groupby((~lim).cumsum()).cumcount() + 1     # 当前连板数(含今)
    out["streak2"] = streak.where(lim) >= 2                   # 连板>=2中
    ret = hfq.pct_change()
    out["ret"] = ret
    up = (ret > 0).fillna(False)
    out["up_days"] = up.groupby((~up).cumsum()).cumcount()    # 连涨天数(含今)
    for k in (5, 10, 20):
        out[f"mom{k}"] = hfq / hfq.shift(k) - 1
    out["rev5"] = -out["mom5"]
    out["std5"] = ret.rolling(5, min_periods=5).std()
    out["std20"] = ret.rolling(20, min_periods=20).std()
    amp = ((hi - low) / pc).replace([np.inf, -np.inf], np.nan)
    out["amp5"] = amp.rolling(5, min_periods=5).mean()
    tsrc = to if to.notna().mean() > 0.95 else g["volume"]    # 换手缺失回退 volume
    out["to_level"] = tsrc.rolling(20, min_periods=20).mean()
    out["to_trend"] = (tsrc.rolling(5, min_periods=5).mean()
                       / tsrc.rolling(20, min_periods=20).mean() - 1)
    illq = (ret.abs() / amt).replace([np.inf, -np.inf], np.nan)
    out["amihud20"] = illq.rolling(20, min_periods=20).mean() * 1e9
    vchg = (g["volume"].rolling(5, min_periods=5).mean()
            / g["volume"].rolling(20, min_periods=20).mean()).replace([np.inf, -np.inf], np.nan)
    out["vp_div"] = (np.sign(out["mom5"]) / vchg.clip(0.2, 5)).replace([np.inf, -np.inf], np.nan)
    out["gap"] = (op / pc - 1).replace([np.inf, -np.inf], np.nan)
    c2l = (c / low - 1).replace([np.inf, -np.inf], np.nan)
    out["c2l5"] = c2l.rolling(5, min_periods=5).mean()
    out["amount20"] = amt.rolling(20, min_periods=20).mean()
    out["winner_ratio"] = g["winner_ratio"].values
    out["cost_bias"] = g["cost_bias"].values
    # 潜伏压缩: quiet_drift 20d OLS 斜率/价格 x100 + vol 压缩
    tser = pd.Series(np.arange(len(g), dtype=float), index=g.index)
    slope = hfq.rolling(20, min_periods=20).cov(tser) / tser.rolling(20, min_periods=20).var()
    out["quiet_drift"] = (slope / hfq).replace([np.inf, -np.inf], np.nan) * 100
    out["vol_comp"] = (out["std5"] / out["std20"]).replace([np.inf, -np.inf], np.nan)
    # 前向收益 (复权)
    for k in HZ:
        out[f"fwd{k}"] = hfq.shift(-k) / hfq - 1
    out["date"] = g["date"].values
    out["symbol"] = sym
    return out

feats = df.groupby("symbol", group_keys=False, observed=True).apply(_per_stock).reset_index(drop=True)
ev = feats[feats["date"] >= t_eval0].copy()
del feats, df
ev = ev[ev["date"].isin(set(pd.to_datetime(eval_dates)))]  # 对齐 regime 日历
print(f"[eval] rows={len(ev)} stocks={ev['symbol'].nunique()} days={ev['date'].nunique()}")

# ---------- 4. 场景旗标 (eval 内逐日截面分位) ----------
r_std20 = ev.groupby("date")["std20"].rank(pct=True)
r_to = ev.groupby("date")["to_level"].rank(pct=True)
r_amt = ev.groupby("date")["amount20"].rank(pct=True)
r_qd = ev.groupby("date")["quiet_drift"].rank(pct=True)
SCEN = {
    "all": np.ones(len(ev), dtype=bool),
    "brk0_20": ev["dsb"].notna().to_numpy(),
    "quiet_comp": ((r_qd <= 1 / 3) & (ev["vol_comp"] < 1)).to_numpy(),
    "next_lu": ev["lim_prev"].to_numpy(),
    "streak2": ev["streak2"].to_numpy(),
    "hivol": (r_std20 >= 2 / 3).to_numpy(),
    "hito": (r_to >= 2 / 3).to_numpy(),
    "illiq": (r_amt <= 1 / 3).to_numpy(),
    "bigup": (ev["mom5"] > 0.10).to_numpy(),
    "bigdn": (ev["mom5"] < -0.10).to_numpy(),
}
for name, m in SCEN.items():
    sub = ev.loc[np.asarray(m, dtype=bool)]
    print(f"[scen] {name:11s} rows={len(sub):7d} days={sub['date'].nunique():4d} "
          f"mean_cs={len(sub)/max(sub['date'].nunique(),1):6.1f}")

# ---------- 5. regime 三分位 (eval 窗时序分位) ----------
rw = reg[reg["date"].isin(pd.to_datetime(eval_dates))].set_index("date")
terc = {}
for c in REGCOLS:
    q1, q2 = rw[c].quantile([1 / 3, 2 / 3])
    t_ = pd.Series("Mid", index=rw.index)
    t_[rw[c] < q1] = "Low"
    t_[rw[c] > q2] = "High"
    terc[c] = t_
ev_dates_ts = pd.DatetimeIndex(pd.to_datetime(eval_dates))
half = len(ev_dates_ts) // 2
is_h1 = pd.Series(ev_dates_ts < ev_dates_ts[half], index=ev_dates_ts)

# ---------- 6. IC 引擎: 场景内逐日截面 Spearman (rank-Pearson, bincount 向量化) ----------
def daily_rank(dates_code: np.ndarray, x: np.ndarray, n_dates: int) -> np.ndarray:
    return pd.Series(x).groupby(dates_code).rank().to_numpy()

def masked_pearson_by_day(dc: np.ndarray, u: np.ndarray, v: np.ndarray,
                          n_dates: int, min_cs: int) -> np.ndarray:
    m = np.isfinite(u) & np.isfinite(v)
    u0 = np.where(m, np.nan_to_num(u), 0.0)
    v0 = np.where(m, np.nan_to_num(v), 0.0)
    n = np.bincount(dc, weights=m.astype(float), minlength=n_dates)
    su = np.bincount(dc, weights=u0, minlength=n_dates)
    sv = np.bincount(dc, weights=v0, minlength=n_dates)
    suu = np.bincount(dc, weights=u0 * u0, minlength=n_dates)
    svv = np.bincount(dc, weights=v0 * v0, minlength=n_dates)
    suv = np.bincount(dc, weights=u0 * v0, minlength=n_dates)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu, mv = su / n, sv / n
        cov = suv / n - mu * mv
        vu = suu / n - mu * mu
        vv = svv / n - mv * mv
        ic = cov / np.sqrt(vu * vv)
    ic[n < min_cs] = np.nan
    return ic

date_index = pd.DatetimeIndex(sorted(ev["date"].unique()))
dc_map = {d: i for i, d in enumerate(date_index)}
ic_store = {}   # (scen, factor, h) -> Series(date)
fwd_mean_store = {}  # (scen, h) -> Series(date)  直连用日均值
scen_ns = {}
for sname, mask in SCEN.items():
    sub = ev.loc[np.asarray(mask, dtype=bool)].copy()
    scen_ns[sname] = (len(sub), sub["date"].nunique())
    if len(sub) < 200:
        for f in FACTORS:
            for h in HZ:
                ic_store[(sname, f, h)] = pd.Series(np.nan, index=date_index)
        for h in HZ:
            fwd_mean_store[(sname, h)] = pd.Series(np.nan, index=date_index)
        continue
    dc = sub["date"].map(dc_map).to_numpy()
    ranks = {f: daily_rank(dc, sub[f].to_numpy(float), len(date_index)) for f in FACTORS}
    for h in HZ:
        v = daily_rank(dc, sub[f"fwd{h}"].to_numpy(float), len(date_index))
        for f in FACTORS:
            ic = masked_pearson_by_day(dc, ranks[f], v, len(date_index), MIN_CS)
            ic_store[(sname, f, h)] = pd.Series(ic, index=date_index)
        # 直连: 日均前向收益
        dm = sub.groupby("date")[f"fwd{h}"].mean()
        cntd = sub.groupby("date")[f"fwd{h}"].count()
        s = dm.where(cntd >= MIN_DAY_DIRECT)
        fwd_mean_store[(sname, h)] = s.reindex(date_index)
print("[ic] engine done")

def welch(a: pd.Series, b: pd.Series):
    a, b = a.dropna(), b.dropna()
    if len(a) < 15 or len(b) < 15:
        return np.nan, len(a), len(b)
    t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return t, len(a), len(b)

# ---------- 7A. regime 直连 ----------
direct_rows = []
for sname in SCEN:
    for h in HZ:
        s_full = fwd_mean_store[(sname, h)]
        for rc in REGCOLS:
            tb_ = terc[rc].reindex(s_full.index)
            hi, lo = s_full[tb_ == "High"], s_full[tb_ == "Low"]
            t, n_hi, n_lo = welch(hi, lo)
            h1m = is_h1.reindex(hi.index, fill_value=False)
            h2m = ~is_h1.reindex(hi.index, fill_value=True)
            sp1 = hi[h1m].mean() - lo[is_h1.reindex(lo.index, fill_value=False)].mean()
            sp2 = hi[h2m].mean() - lo[~is_h1.reindex(lo.index, fill_value=True)].mean()
            direct_rows.append({
                "scenario": sname, "regime": rc, "h": h,
                "spread_HiLo": round(hi.mean() - lo.mean(), 5) if len(hi) else np.nan,
                "t": round(t, 2) if np.isfinite(t) else np.nan,
                "n_hi": n_hi, "n_lo": n_lo,
                "sp_h1": round(sp1, 5), "sp_h2": round(sp2, 5),
                "stable": (np.sign(sp1) == np.sign(sp2)) if np.isfinite(sp1) and np.isfinite(sp2) else False,
            })
direct = pd.DataFrame(direct_rows)
direct.to_csv(f"{OUT}\\scan_direct.csv", index=False, encoding="utf-8-sig")

# ---------- 7B. IC 调制 ----------
mod_rows = []
for sname in SCEN:
    for f in FACTORS:
        for h in HZ:
            s_full = ic_store[(sname, f, h)]
            base_lo = base_hi = np.nan
            for rc in REGCOLS:
                tb_ = terc[rc].reindex(s_full.index)
                hi, lo = s_full[tb_ == "High"], s_full[tb_ == "Low"]
                t, n_hi, n_lo = welch(hi, lo)
                hi1, lo1 = hi[is_h1.reindex(hi.index, fill_value=False)], lo[is_h1.reindex(lo.index, fill_value=False)]
                hi2, lo2 = hi[~is_h1.reindex(hi.index, fill_value=True)], lo[~is_h1.reindex(lo.index, fill_value=True)]
                d1 = hi1.mean() - lo1.mean() if len(hi1) > 5 and len(lo1) > 5 else np.nan
                d2 = hi2.mean() - lo2.mean() if len(hi2) > 5 and len(lo2) > 5 else np.nan
                mod_rows.append({
                    "scenario": sname, "factor": f, "h": h, "regime": rc,
                    "ic_lo": round(lo.mean(), 4) if len(lo) else np.nan,
                    "ic_hi": round(hi.mean(), 4) if len(hi) else np.nan,
                    "dIC": round(hi.mean() - lo.mean(), 4) if len(hi) and len(lo) else np.nan,
                    "dt": round(t, 2) if np.isfinite(t) else np.nan,
                    "n_hi": n_hi, "n_lo": n_lo,
                    "dIC_h1": round(d1, 4) if np.isfinite(d1) else np.nan,
                    "dIC_h2": round(d2, 4) if np.isfinite(d2) else np.nan,
                    "stable": (np.sign(d1) == np.sign(d2)) if np.isfinite(d1) and np.isfinite(d2) else False,
                })
mod = pd.DataFrame(mod_rows)
mod.to_csv(f"{OUT}\\scan_ic_mod.csv", index=False, encoding="utf-8-sig")
print(f"[out] scan_direct.csv rows={len(direct)}, scan_ic_mod.csv rows={len(mod)}")

# ---------- 8. 多检验校正 ----------
# 各轴有效检验数: equicorrelation 公式 M_eff = k / (1 + (k-1) * rho_bar)
def meff(k, rho):
    return max(1.0, k / (1 + (k - 1) * rho))

# regime 轴: 5 列 eval 期两两 |spearman|
regm = rw[REGCOLS].corr(method="spearman").abs().to_numpy()
tri = np.triu_indices(len(REGCOLS), 1)
rho_reg = regm[tri].mean()
# 因子轴: 'all' 场景 h=5 各因子日 IC 序列两两 |corr|
icmat = pd.DataFrame({f: ic_store[("all", f, 5)] for f in FACTORS})
rho_fact = icmat.corr(method="spearman").abs().to_numpy()[np.triu_indices(len(FACTORS), 1)].mean()
# horizon 轴: fwd1/3/5/10 日均序列两两 |corr|
hmat = pd.DataFrame({h: fwd_mean_store[("all", h)] for h in HZ})
rho_hz = hmat.corr(method="spearman").abs().to_numpy()[np.triu_indices(len(HZ), 1)].mean()
# 场景轴: 两两 Jaccard 重叠
keys = [k for k in SCEN if k != "all"]
jac = []
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        ma, mb = np.asarray(SCEN[a], bool), np.asarray(SCEN[b], bool)
        jac.append((ma & mb).sum() / max((ma | mb).sum(), 1))
rho_scen = float(np.mean(jac))
m_scen = meff(len(keys), rho_scen)
m_reg = meff(len(REGCOLS), rho_reg)
m_fact = meff(len(FACTORS), rho_fact)
m_hz = meff(len(HZ), rho_hz)
n_eff = m_scen * m_reg * m_fact * m_hz
n_nominal_mod = len(keys) * len(REGCOLS) * len(FACTORS) * len(HZ)
n_nominal_dir = len(keys) * len(REGCOLS) * len(HZ)
bonf_z = stats.norm.ppf(1 - 0.05 / (2 * n_eff))
print(f"[mtest] rho_scen={rho_scen:.2f} rho_reg={rho_reg:.2f} rho_fact={rho_fact:.2f} "
      f"rho_hz={rho_hz:.2f} -> M_eff: scen={m_scen:.1f} reg={m_reg:.1f} fact={m_fact:.1f} hz={m_hz:.1f}")
print(f"[mtest] n_eff={n_eff:.0f} (nominal mod={n_nominal_mod}, dir={n_nominal_dir}) "
      f"bonf_z(0.05, N_eff)={bonf_z:.2f}")

mod["dt_adj_sqrtN"] = (mod["dt"] / np.sqrt(n_eff)).round(3)
mod["pass_bonf_z"] = mod["dt"].abs() > bonf_z
mod.to_csv(f"{OUT}\\scan_ic_mod.csv", index=False, encoding="utf-8-sig")

top = mod.reindex(mod["dt"].abs().sort_values(ascending=False).index).head(20).copy()
top["ns_rows"] = top["scenario"].map(lambda s: scen_ns[s][0])
top["n_days"] = top["scenario"].map(lambda s: scen_ns[s][1])
cols_keep = ["scenario", "regime", "factor", "h", "ic_lo", "ic_hi", "dIC", "dt",
             "n_hi", "n_lo", "ns_rows", "dIC_h1", "dIC_h2", "stable",
             "dt_adj_sqrtN", "pass_bonf_z"]
top[cols_keep].to_csv(f"{OUT}\\scan_top20.csv", index=False, encoding="utf-8-sig")

summary = {
    "seed": 42, "n_stocks": int(len(syms)), "eval_window": [str(t_eval0)[:10], str(eval_dates[-1])[:10]],
    "min_cs_ic": MIN_CS, "n_ic_tests": int(n_nominal_mod), "n_direct_tests": int(n_nominal_dir),
    "rho_scen": round(rho_scen, 3), "rho_regime": round(rho_reg, 3),
    "rho_factor": round(rho_fact, 3), "rho_horizon": round(rho_hz, 3),
    "n_eff": round(n_eff, 1), "bonf_z_0.05": round(float(bonf_z), 2),
    "scen_ns": {k: [int(v[0]), int(v[1])] for k, v in scen_ns.items()},
    "ic_engine": "rank-Pearson == Spearman, min_cs=10, welch t on Hi/Lo day IC series",
}
with open(f"{OUT}\\scan_summary.json", "w", encoding="utf-8") as fo:
    json.dump(summary, fo, ensure_ascii=False, indent=2)

print("\n=== TOP20 by |dt| (IC modulation) ===")
print(top[cols_keep].to_string(index=False))
print("\n=== DIRECT |t|>2 ===")
dd = direct[direct["t"].abs() > 2].sort_values("t", key=abs, ascending=False)
print(dd.to_string(index=False) if len(dd) else "(none)")
print("\n=== survive sqrt-N_eff shrink (|dt_adj|>2) ===")
sv = mod[mod["dt_adj_sqrtN"].abs() > 2]
print(sv[cols_keep[:8]].to_string(index=False) if len(sv) else "(none)")
