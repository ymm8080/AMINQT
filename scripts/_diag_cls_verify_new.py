"""验证重训新 bundle: 反锚切分 + dual cls AUC 早停 (2026-08-11).

对每个板块新 bundle (tag 20260811_fix):
- 12 头的树数 (dual cls 应从 1-2 树恢复, main 10d_reg 应从 3 树恢复)
- cls 头: prob_up 末 60 日截面 std / 唯一值 (dual 不应再常数) + rankIC
- 10d_reg 头: pred 截面 std (spread 恢复)
用法: python scripts/_diag_cls_verify_new.py [tag]
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

TAG = sys.argv[1] if len(sys.argv) > 1 else "20260811_fix"


def trees(b, k):
    m = b["models"].get(k)
    return m[0].booster_.num_trees() if m else None


for BRD in ("main", "dual"):
    pf = pq.ParquetFile(f"data/_diag_stage_{BRD}_3y.parquet")
    schema = set(pf.schema.names)
    b = pickle.load(open(f"models/pipeline1/{BRD}_{TAG}.pkl", "rb"))
    cols = [c for c in b["feature_cols"] if c in schema]
    need = ["date", "symbol"] + cols
    for k in (3, 5, 10):
        need += [f"label_pm_{k}d_cls_net", f"label_pm_{k}d_net"]
    df = pf.read(columns=need).to_pandas()
    for c in cols:
        df[c] = df[c].astype("float32")
    dates = sorted(df["date"].unique())
    test_d = dates[-60:]
    mte = np.isin(df["date"].values, np.array(test_d).astype("datetime64[ns]"))
    print(f"=== {BRD} 新bundle tag={TAG} ===", flush=True)
    tr = {f"{k}d_reg": trees(b, f"{k}d_reg") for k in (3, 5, 10)}
    tc = {f"{k}d_cls": trees(b, f"{k}d_cls") for k in (3, 5, 10)}
    print(f"  reg trees: {tr}", flush=True)
    print(f"  cls trees: {tc}", flush=True)
    for K in (3, 5, 10):
        lc, lg = f"label_pm_{K}d_cls_net", f"label_pm_{K}d_net"
        m = mte & df[lc].notna().to_numpy() & df[lg].notna().to_numpy()
        ix = np.flatnonzero(m)
        X = np.nan_to_num(df.iloc[ix].loc[:, cols].to_numpy(np.float32), nan=0.0)
        model, _ = b["models"][f"{K}d_cls"]
        cal = b["calibrators"].get(K)
        raw = model.predict_proba(X)[:, 1]
        p = cal.predict_proba(raw) if cal else raw
        r = df.iloc[ix][lg].to_numpy()
        dt = df.iloc[ix]["date"].to_numpy()
        ric = (
            pd.DataFrame({"d": dt, "p": p, "r": r})
            .groupby("d")
            .apply(lambda g: spearmanr(g["p"], g["r"])[0], include_groups=False)
            .mean()
        )
        print(
            f"  {K}d_cls: 校准std={p.std():.5f} 唯一值={len(np.unique(np.round(p, 3)))} "
            f"rankIC={ric:.4f}",
            flush=True,
        )
        del X
    for K in (3, 5, 10):
        lg = f"label_pm_{K}d_net"
        m = mte & df[lg].notna().to_numpy()
        ix = np.flatnonzero(m)
        X = np.nan_to_num(df.iloc[ix].loc[:, cols].to_numpy(np.float32), nan=0.0)
        model, _ = b["models"][f"{K}d_reg"]
        pr = model.predict(X)
        print(f"  {K}d_reg: pred_std={np.std(pr):.5f}", flush=True)
        del X, pr
    del df
