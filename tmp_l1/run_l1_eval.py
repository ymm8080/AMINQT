# -*- coding: utf-8 -*-
"""L1 日频状态机特征实验: 断板后状态序列 + 潜伏压缩状态
样本: np.random.seed(42) 抽 400 只 + 最近 ~500 交易日 (前 40 日仅作 rolling 预热)
评估: r5 = close_hfq(t+5)/close_hfq(t)-1 前向标签; TS IC + 简单 t + 残差IC(中性化 mom5 + close_vs_low_ma5) + 前后半窗
注: 涨停判定 = close/pre_close-1 >= 阈值 (30/68 前缀 19.8%, 其余 9.8%) — 近似口径,
    未处理 ST 5%/北交所 30%/分档 rounding/上市首日无涨跌幅。
"""
import json
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning)

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
OUT_DIR = r"D:\AMINQT\AMINQT CODES\tmp_l1"
WARMUP = 40          # 特征预热天数 (不计入评估)
EVAL_DAYS = 500      # 评估窗口
N_STOCKS = 400
MIN_CS = 30          # 截面最少股票数

# ---------- 1. 抽样 (seed 42) ----------
key = pq.read_table(PANEL, columns=["symbol", "date"])
kdf = key.to_pandas()
dates_all = np.sort(kdf["date"].unique())
dates_win = dates_all[-(EVAL_DAYS + WARMUP):]  # 540 日含预热
cutoff = dates_win[0]
syms_all = sorted(kdf["symbol"].unique())
rng = np.random.RandomState(42)
syms = list(rng.choice(syms_all, size=min(N_STOCKS, len(syms_all)), replace=False))
del key, kdf
print(f"[sample] dates={len(dates_win)} ({dates_win[0]} .. {dates_win[-1]}), stocks={len(syms)}")

# ---------- 2. 读面板 (pyarrow 预过滤列 + 日期) ----------
cols = ["symbol", "date", "close", "pre_close", "low", "volume",
        "turnover_rate", "close_hfq"]
tbl = pq.read_table(PANEL, columns=cols, filters=[("date", ">=", cutoff)])
df = tbl.to_pandas()
del tbl
df = df[df["symbol"].isin(syms)].sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"[load] rows={len(df)}, to_rate NaN ratio={df['turnover_rate'].isna().mean():.3f}")

# ---------- 3. 特征构造 (groupby(symbol) 一次性 apply, 组内全向量化) ----------
def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, pc, low = g["close"], g["pre_close"], g["low"]
    hfq = g["close_hfq"]
    to = g["turnover_rate"]
    out = pd.DataFrame(index=g.index)

    # --- 涨停/断板 (近似口径, 见文件头) ---
    sym = str(df.at[g.index[0], "symbol"])  # include_groups=False 需回查
    thr = 0.198 if sym.startswith(("30", "68")) else 0.098
    lim = ((c / pc - 1) >= thr).fillna(False)          # NaN 视为未涨停
    brk = lim.shift(1, fill_value=False) & ~lim        # 断板日: 昨涨停今日未涨停
    grp = brk.cumsum()                                 # 断板 epoch id
    pos = brk.groupby(grp).cumcount()                  # 距断板天数, 断板日=0

    # 1) days_since_board_break: 距最近断板天数, >20 归 NaN
    dsb = pos.where(grp > 0)
    out["days_since_board_break"] = dsb.where(dsb <= 20)

    # 2) pullback_depth: 现价/断板日收盘-1 (限 20 日内)
    c_at_brk = c.where(brk).ffill()
    out["pullback_depth"] = (c / c_at_brk - 1).where(out["days_since_board_break"].notna())

    # 3) vol_decay_ratio: 当日换手 / (断板日..昨日 换手均值), 需 >=2 个观测
    exp_mean = to.groupby(grp).cumsum() / (pos + 1)    # epoch 内 expanding 均值
    denom = exp_mean.shift(1)
    out["vol_decay_ratio"] = (to / denom).replace([np.inf, -np.inf], np.nan) \
        .where(dsb.notna() & (pos >= 2))

    # --- 潜伏压缩状态 ---
    ret = hfq.pct_change()                             # 日收益用复权价

    # 4) flat_days: |10日累计涨幅|<3% 的连续天数 (横盘计数)
    cond = (hfq / hfq.shift(10) - 1).abs() < 0.03
    cond = cond.fillna(False)
    rid = (cond != cond.shift(1)).cumsum()
    out["flat_days"] = pd.Series(
        np.where(cond, cond.groupby(rid).cumcount() + 1, 0), index=g.index
    )

    # 5) vol_compression: 5日收益std / 20日收益std
    out["vol_compression"] = (
        ret.rolling(5, min_periods=5).std() / ret.rolling(20, min_periods=20).std()
    ).replace([np.inf, -np.inf], np.nan)

    # 6) turnover_trend: 5日均换手/20日均换手-1 (换手缺失时退回 volume)
    tsrc = to if to.notna().mean() > 0.95 else g["volume"]
    m5 = tsrc.rolling(5, min_periods=5).mean()
    m20 = tsrc.rolling(20, min_periods=20).mean()
    out["turnover_trend"] = (m5 / m20 - 1).replace([np.inf, -np.inf], np.nan)

    # 7) quiet_drift: 20日 close_hfq 对时间 OLS 斜率/价格 (真回归, 非代理) x100
    tser = pd.Series(np.arange(len(g)), index=g.index, dtype=float)
    slope = hfq.rolling(20, min_periods=20).cov(tser) / tser.rolling(20, min_periods=20).var()
    out["quiet_drift"] = (slope / hfq).replace([np.inf, -np.inf], np.nan) * 100

    # --- 基准特征 (与 v35 引擎同口径, 原始价) ---
    cl_low = (c / low - 1).replace([np.inf, -np.inf], np.nan)
    out["close_vs_low_ma5"] = cl_low.rolling(5, min_periods=5).mean() * 100
    out["mom5"] = c / c.shift(5) - 1

    # --- 标签: r5 前向 5 日收益 (复权) ---
    out["r5"] = hfq.shift(-5) / hfq - 1
    out["date"] = g["date"].values
    out["symbol"] = sym
    return out

feats = df.groupby("symbol", group_keys=False).apply(_per_stock, include_groups=False)
feats = feats.reset_index(drop=True)

feat_cols = ["days_since_board_break", "pullback_depth", "vol_decay_ratio",
             "flat_days", "vol_compression", "turnover_trend", "quiet_drift"]

# ---------- 4. 评估 (仅最近 500 交易日) ----------
eval_df = feats[feats["date"] >= dates_win[WARMUP]].copy()
dates_ev = np.sort(eval_df["date"].unique())
half = len(dates_ev) // 2
d1, d2 = set(dates_ev[:half]), set(dates_ev[half:])
print(f"[eval] dates={len(dates_ev)}, rows={len(eval_df)}")


def ts_ic(d: pd.DataFrame, x: str, y: str = "r5") -> pd.Series:
    def _s(g):
        if len(g) < MIN_CS:
            return np.nan
        return g[x].corr(g[y], method="spearman")
    return d.groupby("date").apply(_s)


def resid_ts_ic(d: pd.DataFrame, x: str) -> pd.Series:
    """逐日截面 OLS 中性化 [mom5, close_vs_low_ma5] 后残差与 r5 的 Spearman IC"""
    neuts = ["mom5", "close_vs_low_ma5"]

    def _s(g):
        g = g[[x, "r5"] + [c for c in neuts if c != x]].dropna()
        if len(g) < MIN_CS + 3:
            return np.nan
        X = np.column_stack([np.ones(len(g))] + [g[c].values for c in neuts])
        beta, *_ = np.linalg.lstsq(X, g[x].values, rcond=None)
        resid = g[x].values - X @ beta
        rs = pd.Series(resid).rank()
        ry = g["r5"].rank()
        return np.corrcoef(rs, ry)[0, 1]
    return d.groupby("date").apply(_s)


def summarize(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if len(ic) < 20:
        return {"n": len(ic), "ic": np.nan, "t": np.nan}
    t = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))
    return {"n": int(len(ic)), "ic": round(ic.mean(), 4), "t": round(t, 2)}


rows = []
for col in feat_cols + ["close_vs_low_ma5"]:  # 基准特征一并复算验证口径
    full = summarize(ts_ic(eval_df, col))
    h1 = summarize(ts_ic(eval_df[eval_df["date"].isin(d1)], col))
    h2 = summarize(ts_ic(eval_df[eval_df["date"].isin(d2)], col))
    res = summarize(resid_ts_ic(eval_df, col))
    cov = eval_df[col].notna().mean()
    rows.append({
        "feature": col, "coverage": round(cov, 3),
        "ic": full["ic"], "t": full["t"],
        "ic_h1": h1["ic"], "t_h1": h1["t"], "ic_h2": h2["ic"], "t_h2": h2["t"],
        "resid_ic": res["ic"], "resid_t": res["t"],
    })
    print(rows[-1])

res = pd.DataFrame(rows)
res.to_csv(f"{OUT_DIR}\\l1_ic_results.csv", index=False, encoding="utf-8-sig")
feats.to_parquet(f"{OUT_DIR}\\l1_features_sample.parquet", index=False)
with open(f"{OUT_DIR}\\l1_run_meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "seed": 42, "n_stocks": len(syms), "eval_dates": [str(dates_ev[0]), str(dates_ev[-1])],
        "warmup_days": WARMUP, "label": "r5=close_hfq(t+5)/close_hfq(t)-1",
        "limit_rule": "close/pre_close-1>=thr, 30/68:19.8% else 9.8% (approx; ST/BSE/rounding ignored)",
    }, f, ensure_ascii=False, indent=2)
print("[done] saved l1_ic_results.csv / l1_features_sample.parquet")
