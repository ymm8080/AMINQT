# -*- coding: utf-8 -*-
"""flat_days 全宇宙复验 (无抽样)
完全复用 run_l1_eval.py 口径:
  flat_days = |10日累计涨幅(close_hfq)|<3% 的连续天数 (横盘计数)
  标签 r5 = close_hfq(t+5)/close_hfq(t)-1; TS Spearman IC + 简单 t;
  残差IC = 逐日截面 OLS 中性化 [mom5, close_vs_low_ma5] 后与 r5 的 Spearman;
  前后半窗各复算; 最近500交易日评估 + 前40日仅预热。
差异: 全部股票 (~1780) 替代 seed42 抽样 400; 列最小化 symbol/date/close/low/close_hfq。
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
MIN_CS = 30          # 截面最少股票数

# ---------- 1. 全宇宙日期窗 (无抽样) ----------
key = pq.read_table(PANEL, columns=["symbol", "date"])
kdf = key.to_pandas()
dates_all = np.sort(kdf["date"].unique())
dates_win = dates_all[-(EVAL_DAYS + WARMUP):]
cutoff = dates_win[0]
n_syms = kdf["symbol"].nunique()
del key, kdf
print(f"[sample] dates={len(dates_win)} ({dates_win[0]} .. {dates_win[-1]}), stocks={n_syms} (FULL universe)")

# ---------- 2. 读面板 (最小列集 pyarrow 预过滤) ----------
cols = ["symbol", "date", "close", "low", "close_hfq"]
tbl = pq.read_table(PANEL, columns=cols, filters=[("date", ">=", cutoff)])
df = tbl.to_pandas()
del tbl
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"[load] rows={len(df)}, mem={df.memory_usage(deep=True).sum()/1e9:.2f} GB")

# ---------- 3. 特征构造 (组内全向量化, 口径与 run_l1_eval.py 逐行一致) ----------
def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, low = g["close"], g["low"]
    hfq = g["close_hfq"]
    out = pd.DataFrame(index=g.index)
    sym = str(df.at[g.index[0], "symbol"])

    # 4) flat_days: |10日累计涨幅|<3% 的连续天数 (横盘计数)
    cond = (hfq / hfq.shift(10) - 1).abs() < 0.03
    cond = cond.fillna(False)
    rid = (cond != cond.shift(1)).cumsum()
    out["flat_days"] = pd.Series(
        np.where(cond, cond.groupby(rid).cumcount() + 1, 0), index=g.index
    )

    # --- 基准特征 (残差中性化用, 与 v35 引擎同口径, 原始价) ---
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

feat_cols = ["flat_days", "close_vs_low_ma5"]  # 基准一并复算验证口径

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
for col in feat_cols:
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
res.to_csv(f"{OUT_DIR}\\_flatdays_full_results.csv", index=False, encoding="utf-8-sig")
with open(f"{OUT_DIR}\\_flatdays_full_meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "universe": "FULL (no sampling)", "n_stocks": int(n_syms),
        "eval_dates": [str(dates_ev[0]), str(dates_ev[-1])],
        "warmup_days": WARMUP, "label": "r5=close_hfq(t+5)/close_hfq(t)-1",
        "flat_days_rule": "|close_hfq/close_hfq.shift(10)-1|<0.03 consecutive days count",
        "prior_sample_run": "seed42 400 stocks: ic=0.0238 t=5.1 h2_t=1.79 resid=-0.0226 t=-5.26",
    }, f, ensure_ascii=False, indent=2)
print("[done] saved _flatdays_full_results.csv / _flatdays_full_meta.json")
