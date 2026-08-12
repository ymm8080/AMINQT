"""验证 cls (概率) 头修复: 修复版 split (反锚) vs 部署版 (陈旧窗口).

用户观察: prob_up 跨股票几乎相同 (dual 完全常数). 根因 = 陈旧窗口 → cls 头塌缩.
测: 末 60 日 OOS, 修复版 vs 部署版 的 prob 截面 std / 分位分散 / rank IC (对真实前向收益).
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

BRD = sys.argv[1] if len(sys.argv) > 1 else "dual"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BUNDLE = f"models/pipeline1/{BRD}_20260810.pkl"

pf = pq.ParquetFile(f"data/_diag_stage_{BRD}_3y.parquet")
schema = set(pf.schema.names)
bundle = pickle.load(open(BUNDLE, "rb"))
cols = [c for c in bundle["feature_cols"] if c in schema]
lbl_reg = f"label_pm_{K}d_net"
lbl_cls = f"label_pm_{K}d_cls_net"
df = pf.read(columns=["date", "symbol"] + cols + [lbl_reg, lbl_cls]).to_pandas()
for c in cols:
    df[c] = df[c].astype("float32")
print(f"frame rows={len(df)}", flush=True)

dates = sorted(df["date"].unique())
n = len(dates)
d = df["date"].values
train_d, es_d, test_d = dates[:-80], dates[-80:-60], dates[-60:]


def arr(seg_d, lbl):
    m = (
        np.isin(d, np.array(seg_d).astype("datetime64[ns]"))
        & df[lbl].notna().to_numpy()
    )
    ix = np.flatnonzero(m)
    X = np.nan_to_num(df.iloc[ix].loc[:, cols].to_numpy(np.float32), nan=0.0)
    return (
        X,
        df.iloc[ix][lbl].to_numpy(),
        df.iloc[ix]["date"].to_numpy(),
        df.iloc[ix]["symbol"].to_numpy(),
    )


Xtr, ytr, dtr, _ = arr(train_d, lbl_cls)
Xes, yes, _, _ = arr(es_d, lbl_cls)
# test 用单一行掩码 (cls 标签非空), 同时取回归标签做 rank IC 参照
mte = (
    np.isin(d, np.array(test_d).astype("datetime64[ns]"))
    & df[lbl_cls].notna().to_numpy()
    & df[lbl_reg].notna().to_numpy()
)
ixte = np.flatnonzero(mte)
Xte = np.nan_to_num(df.iloc[ixte].loc[:, cols].to_numpy(np.float32), nan=0.0)
dte, ste = df.iloc[ixte]["date"].to_numpy(), df.iloc[ixte]["symbol"].to_numpy()
rte = df.iloc[ixte][lbl_reg].to_numpy()


def fit_cls(X, y, Xes, yes):
    ds = lgb.Dataset(X, y)
    bst = lgb.train(
        dict(
            objective="binary", learning_rate=0.05, num_leaves=15, seed=42, verbosity=-1
        ),
        ds,
        valid_sets=[lgb.Dataset(Xes, yes)],
        num_boost_round=1000,
        callbacks=[lgb.record_evaluation({}), lgb.early_stopping(100)],
    )
    return bst


def metrics(name, prob, te_dates, te_syms, te_ret):
    f = pd.DataFrame({"d": te_dates, "s": te_syms, "p": prob, "r": te_ret})
    daily = f.groupby("d").apply(
        lambda g: spearmanr(g["p"], g["r"])[0], include_groups=False
    )
    f.groupby("d")["p"].apply(lambda s: np.percentile(s, 50))
    p90 = f.groupby("d")["p"].apply(lambda s: np.percentile(s, 90))
    p10 = f.groupby("d")["p"].apply(lambda s: np.percentile(s, 10))
    spread = (p90 - p10).mean()
    top_hit = f.groupby("d").apply(
        lambda g: (g.nlargest(20, "p")["r"] > 0).mean(), include_groups=False
    )
    pool_hit = f.groupby("d").apply(lambda g: (g["r"] > 0).mean(), include_groups=False)
    print(
        f"  [{name}] trees={bst.num_trees() if name.startswith('修复') else '部署'} "
        f"prob_std={prob.std():.5f} | 分位90-10={spread:.4f} "
        f"| rankIC={daily.mean():.4f} | TOP20命中={top_hit.mean():.3f} vs 池={pool_hit.mean():.3f}",
        flush=True,
    )


# 部署版
m, _ = bundle["models"][f"{K}d_cls"]
cal = bundle["calibrators"].get(K)
raw_dep = m.predict_proba(Xte)[:, 1]
p_dep = cal.predict_proba(raw_dep) if cal else raw_dep
print(f"--- {BRD} {K}d_cls 末60日 ({test_d[0]:%Y-%m-%d}..{test_d[-1]:%Y-%m-%d}) ---")
metrics("部署版", p_dep, dte, ste, rte)

# 修复版
bst = fit_cls(Xtr, ytr, Xes, yes)
raw_fix = bst.predict(Xte)
p_fix = raw_fix  # 不做校准, 直接看原始 prob 区分度
metrics("修复版", p_fix, dte, ste, rte)
