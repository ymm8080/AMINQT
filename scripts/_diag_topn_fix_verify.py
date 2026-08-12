"""验证 10d 头修复对 TOP-N 质量的提升 (2026-08-11).

对比: 部署版 (train 止 03-16, 3树近常数) vs FIX-A (train 止 04-14, 400树).
评估窗 = 诚实 OOS 末 60 交易日 (05-18..08-10), 逐日 TOP-15 按 pred_ret_10d 排名.
指标: 命中率 (label_pm_10d_net>0), 平均前向收益, 相对池基线的超额.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
import gc
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FRAME = r"data/_diag_stage_main_3y.parquet"
BUNDLE = r"models/pipeline1/main_20260810.pkl"

pf = pq.ParquetFile(FRAME)
schema = set(pf.schema.names)
bundle = pickle.load(open(BUNDLE, "rb"))
cols = [c for c in bundle["feature_cols"] if c in schema]
need = ["date", "symbol"] + cols + ["label_pm_10d_net"]
df = pf.read(columns=need).to_pandas()
for c in cols:
    df[c] = df[c].astype("float32")
print(f"frame rows={len(df)} cols={len(cols)}", flush=True)

dates = sorted(df["date"].unique())
n = len(dates)
d = df["date"].values
LBL = "label_pm_10d_net"
test_dates = dates[-60:]
test_mask = (
    np.isin(d, np.array(test_dates).astype("datetime64[ns]"))
    & df[LBL].notna().to_numpy()
)
test_idx = np.flatnonzero(test_mask)
X_test = df.iloc[test_idx].loc[:, cols].to_numpy(np.float32)
y_test = df.iloc[test_idx][LBL].to_numpy(float)
test_symbols = df.iloc[test_idx]["symbol"].to_numpy()
test_dates_arr = df.iloc[test_idx]["date"].to_numpy()
print(
    f"test rows={len(X_test)} ({test_dates[0]:%Y-%m-%d}..{test_dates[-1]:%Y-%m-%d})",
    flush=True,
)


def topn_metrics(pred, labels, syms, dtarr, top_n=15, min_pool=20):
    """逐日 TOP-N 命中率 + 平均前向收益, 与池基线对比."""
    frame = pd.DataFrame({"d": dtarr, "s": syms, "p": pred, "y": labels})
    hit = []
    ret_top, ret_pool = [], []
    for _day, g in frame.groupby("d"):
        if len(g) < min_pool:
            continue
        top = g.nlargest(top_n, "p")
        hit.append((top["y"] > 0).mean())
        ret_top.append(top["y"].mean())
        ret_pool.append(g["y"].mean())
    return {
        "hit_rate": float(np.mean(hit)),
        "top_ret": float(np.mean(ret_top)),
        "pool_ret": float(np.mean(ret_pool)),
        "excess": float(np.mean(ret_top) - np.mean(ret_pool)),
        "n_days": len(hit),
    }


def train_fix_a():
    """FIX-A: train 止 04-14 (dates[:-80]), es=dates[-80:-60], test=last 60."""
    train_d = dates[:-80]
    es_d = dates[-80:-60]
    keep = (
        np.isin(d, np.array(train_d).astype("datetime64[ns]"))
        & df[LBL].notna().to_numpy()
    )
    idx = np.flatnonzero(keep)
    X, y = (
        df.iloc[idx].loc[:, cols].to_numpy(np.float32),
        df.iloc[idx][LBL].to_numpy(float),
    )
    es_keep = (
        np.isin(d, np.array(es_d).astype("datetime64[ns]")) & df[LBL].notna().to_numpy()
    )
    eidx = np.flatnonzero(es_keep)
    Xe, ye = (
        df.iloc[eidx].loc[:, cols].to_numpy(np.float32),
        df.iloc[eidx][LBL].to_numpy(float),
    )
    # time_weights (半衰期 250) 复刻生产
    tr_dates_sub = sorted(set(train_d))
    w_map = {
        dp: 0.5 ** ((len(tr_dates_sub) - 1 - i) / 250)
        for i, dp in enumerate(tr_dates_sub)
    }
    w = np.array([w_map[dp] for dp in d[idx]])
    ds = lgb.Dataset(X, y)
    ds.set_weight(w)
    bst = lgb.train(
        dict(
            objective="huber", learning_rate=0.05, num_leaves=31, seed=42, verbosity=-1
        ),
        ds,
        valid_sets=[lgb.Dataset(Xe, ye)],
        num_boost_round=1000,
        callbacks=[lgb.record_evaluation({}), lgb.early_stopping(100)],
    )
    return bst, X_test, y_test


# --- 部署版 (3树近常数) ---
m10, label = bundle["models"]["10d_reg"]
pred_deployed = m10.predict(np.nan_to_num(X_test, nan=0.0))
md = topn_metrics(pred_deployed, y_test, test_symbols, test_dates_arr)
print(
    f"\n[部署版 10d_reg trees={m10.booster_.num_trees()}] pred_std={np.std(pred_deployed):.6f}"
)
print(
    f"  TOP-15: hit={md['hit_rate']:.3f} top_ret={md['top_ret']:+.4f} pool={md['pool_ret']:+.4f} excess={md['excess']:+.4f} ({md['n_days']}d)"
)
del m10, pred_deployed
gc.collect()

# --- FIX-A 新模型 ---
bst, _, _ = train_fix_a()
pred_fix = bst.predict(np.nan_to_num(X_test, nan=0.0))
mf = topn_metrics(pred_fix, y_test, test_symbols, test_dates_arr)
print(f"\n[FIX-A 10d_reg trees={bst.num_trees()}] pred_std={np.std(pred_fix):.6f}")
print(
    f"  TOP-15: hit={mf['hit_rate']:.3f} top_ret={mf['top_ret']:+.4f} pool={mf['pool_ret']:+.4f} excess={mf['excess']:+.4f} ({mf['n_days']}d)"
)

# rank IC 对比
from scipy.stats import spearmanr


def rank_ic(pred, labels, dtarr):
    f = pd.DataFrame({"d": dtarr, "p": pred, "y": labels})
    return (
        f.groupby("d")
        .apply(lambda g: spearmanr(g["p"], g["y"])[0], include_groups=False)
        .mean()
    )


# rank IC 对比 (重新计算部署版, 避免变量已被 del)
m10b, _ = bundle["models"]["10d_reg"]
pred_dep2 = m10b.predict(np.nan_to_num(X_test, nan=0.0))
print(
    f"\nrank IC (末60d): deployed={rank_ic(pred_dep2, y_test, test_dates_arr):.4f} fix_a={rank_ic(pred_fix, y_test, test_dates_arr):.4f}"
)
