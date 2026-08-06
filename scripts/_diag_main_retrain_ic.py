# -*- coding: utf-8 -*-
"""_diag_main_retrain_ic.py — 判断 main 重训值不值: 新选中特征在 OOS 尾部有无预测力.

对 selected_main_*.json 的选中特征, 在 features_main 末 250 交易日 (OOS) 逐日算
横截面 Rank IC vs label_pm_{1,3,5}d_net, 输出单特征 IC 分布 + 复合分 IC (等权
横截面 pct-rank 均值). RAM 安全: 分块读列, 逐 chunk 累加滚动统计 (不存全量 IC).

用法: python scripts/_diag_main_retrain_ic.py [--features <parquet>] [--oos 250]
输出: data/_diag_main_retrain_ic_{ts}.json (WORM)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

REGISTRY = "D:/AMINQT/DATA OTHERS/factor_registry"
LOCAL_REGISTRY = "data/factor_registry"
LABELS = ["label_pm_1d_net", "label_pm_3d_net", "label_pm_5d_net"]
CHUNK = 40


def latest_features(dirpath):
    files = sorted(f for f in os.listdir(dirpath) if f.startswith("features_main_"))
    if not files:
        raise FileNotFoundError(f"No features_main_*.parquet in {dirpath}")
    return os.path.join(dirpath, files[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--oos", type=int, default=250)
    args = ap.parse_args()

    feats_path = args.features or latest_features(LOCAL_REGISTRY)
    sel_path = os.path.join(REGISTRY, "selected_main_20260805T190005.json")
    sel = json.load(open(sel_path, encoding="utf-8"))
    selected = sel["features"]
    sel_ts = sel.get("created", "?")

    pf = pq.ParquetFile(feats_path)
    all_cols = [pf.schema_arrow.field(i).name for i in range(pf.metadata.num_columns)]
    feats = [f for f in selected if f in all_cols]
    missing = [f for f in selected if f not in all_cols]
    labels = [l for l in LABELS if l in all_cols]
    brute_set = {f for f in feats if "_brute_" in f}

    t0 = time.time()
    d = pd.read_parquet(feats_path, columns=["date"])
    dates = sorted(d["date"].unique())
    oos = set(dates[-args.oos:])
    print(
        f"[info] features={os.path.basename(feats_path)} rows={pf.metadata.num_rows:,} "
        f"dates={dates[0].date()}..{dates[-1].date()} oos_days={len(oos)}",
        flush=True,
    )
    print(
        f"[info] selected={len(selected)} present={len(feats)} missing={len(missing)} "
        f"brute_only={len(brute_set)} labels={labels}",
        flush=True,
    )
    if not feats:
        raise RuntimeError("No selected features present in features parquet")

    # 每 (date, symbol) 行的复合分累加 (全选中 / brute 子集) + label 原始值.
    acc = pd.read_parquet(feats_path, columns=["date", "symbol"])
    acc = acc[acc["date"].isin(oos)].reset_index(drop=True)
    acc["comp_all"] = 0.0
    acc["comp_brute"] = 0.0
    acc["n_all"] = 0
    acc["n_brute"] = 0
    for lab in labels:
        acc[lab] = np.nan

    lab_df = pd.read_parquet(feats_path, columns=["date"] + labels)
    lab_df = lab_df[lab_df["date"].isin(oos)].reset_index(drop=True)
    for lab in labels:
        acc[lab] = lab_df[lab].values

    # 逐特征滚动统计: sum_ic / sum_sq / count / pos_count per label.
    stats = {lab: {} for lab in labels}

    def upd(stats_lab, feat, s, sq, c, p):
        prev = stats_lab.get(feat)
        if prev is None:
            stats_lab[feat] = [s, sq, c, p]
        else:
            prev[0] += s
            prev[1] += sq
            prev[2] += c
            prev[3] += p

    n_feats = len(feats)
    for start in range(0, n_feats, CHUNK):
        chunk = feats[start : start + CHUNK]
        cdf = pd.read_parquet(feats_path, columns=["date", "symbol"] + chunk)
        cdf = cdf[cdf["date"].isin(oos)].reset_index(drop=True)
        g = cdf.groupby("date")

        ranks = g[chunk].rank(pct=True)
        rmean = g[chunk].transform("mean")
        rstd = g[chunk].transform("std")
        rz = (ranks - rmean) / rstd

        # 复合分: 等权 pct-rank 均值 (skipna, 缺失特征不影响行权重).
        acc["comp_all"] += ranks.mean(axis=1, skipna=True).values
        acc["n_all"] += ranks.notna().sum(axis=1).values
        bmask = [c for c in chunk if c in brute_set]
        if bmask:
            acc["comp_brute"] += ranks[bmask].mean(axis=1, skipna=True).values
            acc["n_brute"] += ranks[bmask].notna().sum(axis=1).values

        for lab in labels:
            lr = g[lab].rank(pct=True)
            lmean = g[lab].transform("mean")
            lstd = g[lab].transform("std")
            lz = (lr - lmean) / lstd
            daily = (rz.multiply(lz, axis=0)).groupby(cdf["date"]).mean()
            s = daily.sum()
            sq = (daily ** 2).sum()
            c = daily.count()
            p = (daily > 0).sum()
            st = stats[lab]
            for i, f in enumerate(chunk):
                upd(st, f, s.iloc[i], sq.iloc[i], c.iloc[i], p.iloc[i])

        del cdf, ranks, rz
        print(f"  chunk {start + len(chunk)}/{n_feats} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 汇总 ----
    def dist(st):
        vals = {f: v[0] / v[2] for f, v in st.items() if v[2] > 30}
        if not vals:
            return None
        arr = np.array(list(vals.values()))
        return {
            "n_feats": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.quantile(arr, 0.25)),
            "p75": float(np.quantile(arr, 0.75)),
            "pct_pos": float((arr > 0).mean()),
            "pct_strong_pos": float((arr > 0.02).mean()),
            "best": {k: float(v) for k, v in sorted(vals.items(), key=lambda x: -x[1])[:5]},
        }

    per_feature = {}
    for lab in labels:
        per_feature[lab] = {
            "all": dist(stats[lab]),
            "brute_only": dist(
                {f: v for f, v in stats[lab].items() if f in brute_set}
            ),
        }

    # ---- 复合分逐日 IC ----
    comp = {}
    for lab in labels:
        out = {}
        for tag, col in (("comp_all", "comp_all"), ("comp_brute", "comp_brute")):
            sub = acc[["date", col, lab]].dropna()
            if len(sub) < 50:
                out[tag] = None
                continue
            g = sub.groupby("date")
            cval = g[col].rank(pct=True)
            cm = g[col].transform("mean")
            cs = g[col].transform("std")
            cz = (cval - cm) / cs
            lval = g[lab].rank(pct=True)
            lm = g[lab].transform("mean")
            ls = g[lab].transform("std")
            lz = (lval - lm) / ls
            daily = (cz * lz).groupby(sub["date"]).mean()
            out[tag] = {
                "mean_ic": float(daily.mean()),
                "pos_day_ratio": float((daily > 0).mean()),
                "n_days": int(len(daily)),
                "last_mean_60d": float(daily.tail(60).mean()),
            }
        comp[lab] = out

    # ---- 结论 ----
    verdicts = {}
    for lab in labels:
        c = comp[lab]["comp_all"]
        if c is None:
            verdicts[lab] = "无数据"
            continue
        strong = c["mean_ic"] > 0.02 and c["pos_day_ratio"] > 0.60
        weak_pos = c["mean_ic"] > 0 and c["pos_day_ratio"] > 0.50
        verdicts[lab] = (
            "池有信号, 重训值得"
            if strong
            else "池信号弱但为正, 重训可能小幅改善"
            if weak_pos
            else "池信号≈0或负, 重训无济于事"
        )

    result = {
        "meta": {
            "features_parquet": os.path.basename(feats_path),
            "selected_json": os.path.basename(sel_path),
            "selected_created": sel_ts,
            "oos_days": len(oos),
            "selected_total": len(selected),
            "present": len(feats),
            "missing": len(missing),
            "missing_sample": missing[:15],
            "run_at": pd.Timestamp.now().isoformat(),
        },
        "per_feature": per_feature,
        "composite": comp,
        "verdict": verdicts,
    }
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/_diag_main_retrain_ic_{ts}.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[verdict] {json.dumps(verdicts, ensure_ascii=False, indent=2)}", flush=True)
    print(f"[saved] {out_path}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
