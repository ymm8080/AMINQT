"""全 cls 头修复验证: 反锚窗口 + AUC 早停 vs 部署版 (2026-08-11).

每个 cls 头: 用反锚切分 (train 止 04-14, es 04-15..05-15) + AUC 早停训练,
测末 60 日 OOS: 树数 / prob 截面 std / rank IC (对真实前向收益).
对比部署版树数 (main 3d=51/5d=27/10d=17; dual 3d=1/5d=2/10d=1).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from scipy.stats import spearmanr

for BRD in ("main", "dual"):
    pf = pq.ParquetFile(f"data/_diag_stage_{BRD}_3y.parquet")
    schema = set(pf.schema.names)
    b = pickle.load(open(f"models/pipeline1/{BRD}_20260810.pkl", "rb"))
    cols = [c for c in b["feature_cols"] if c in schema]
    need = ["date", "symbol"] + cols
    for k in (3, 5, 10):
        need += [f"label_pm_{k}d_cls_net", f"label_pm_{k}d_net"]
    df = pf.read(columns=need).to_pandas()
    for c in cols:
        df[c] = df[c].astype("float32")
    dates = sorted(df["date"].unique())
    n = len(dates)
    d = df["date"].values
    train_d, es_d, test_d = dates[:-80], dates[-80:-60], dates[-60:]

    def arr(seg_d, lbl_cls, lbl_reg):
        m = np.isin(d, np.array(seg_d).astype("datetime64[ns]"))
        m &= df[lbl_cls].notna().to_numpy() & df[lbl_reg].notna().to_numpy()
        ix = np.flatnonzero(m)
        X = np.nan_to_num(df.iloc[ix].loc[:, cols].to_numpy(np.float32), nan=0.0)
        return X, df.iloc[ix][lbl_cls].to_numpy(), df.iloc[ix]["date"].to_numpy(), df.iloc[ix][lbl_reg].to_numpy()

    print(f"=== {BRD} (n={n}) ===", flush=True)
    for K in (3, 5, 10):
        lbl_cls, lbl_reg = f"label_pm_{K}d_cls_net", f"label_pm_{K}d_net"
        Xtr, ytr, _, _ = arr(train_d, lbl_cls, lbl_reg)
        Xes, yes, _, _ = arr(es_d, lbl_cls, lbl_reg)
        Xte, yte, dte, rte = arr(test_d, lbl_cls, lbl_reg)
        dep = b["models"].get(f"{K}d_cls")
        dep_trees = dep[0].booster_.num_trees() if dep else "?"
        # AUC 早停 (pat=100), 反锚切分
        bst = lgb.train(
            dict(objective="binary", learning_rate=0.05, num_leaves=31,
                 seed=42, verbosity=-1, metric="auc"),
            lgb.Dataset(Xtr, ytr), valid_sets=[lgb.Dataset(Xes, yes)],
            num_boost_round=1000,
            callbacks=[lgb.record_evaluation({}), lgb.early_stopping(100)],
        )
        p = bst.predict(Xte)
        ric = pd.DataFrame({"d": dte, "p": p, "r": rte}).groupby("d").apply(
            lambda g: spearmanr(g["p"], g["r"])[0], include_groups=False).mean()
        # 校准后 spread (用 es 拟合 Platt, 避免 isotonic 桶化)
        from app.pipeline1.prob_calibrator import ProbCalibrator
        cal = ProbCalibrator(method="platt").fit(bst.predict(Xes), yes)
        pcal = cal.predict_proba(p)
        print(f"  {K}d_cls: 部署trees={dep_trees} | 修复trees={bst.num_trees()} "
              f"raw_std={p.std():.5f} cal_std={pcal.std():.5f} "
              f"cal_unique={len(np.unique(np.round(pcal, 3)))} rankIC={ric:.4f}", flush=True)
        del Xtr, ytr, Xes, yes, Xte, yte, dte, rte
