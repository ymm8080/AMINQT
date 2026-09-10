# -*- coding: utf-8 -*-
"""小型回测 v2: 10d 秩目标头 多窗口 walk-forward 对拍生产 (v1 单窗口 FAIL 后的用户路线).

v1 (ranktarget_minibacktest_0910.py) 单窗口 (06-17→08-26) 结论:
  校准缺口 −1.39pp vs prod −4.20pp ✓, RankIC 0.129 vs 0.090 ✓,
  但 TOP10 实得 +3.05% vs prod +9.79% ✗ → FAIL。
  而匹配视界审计 (至 09-09) prod TOP10 实得≈0 — 两窗口打架 → 需多窗口判定
  prod 的 TOP10 优势是稳定优势还是行情段运气。

设计: 对最近 N 个不重叠 60 交易日窗各做一次 walk-forward (train 段止于该窗
es 之前, 无泄漏), 生产同款 _train_one 超参 + 生产 bundle 特征列, 唯一变量 =
训练目标。dual 板: 标签为 per-date 去均值超额口径 — 秩目标对 per-date 常数
平移不变 (同日内排序不变), 映射/承诺/实得均在去均值空间内部自洽, 加回
mkt_expected_10d 仅作展示平移, 对缺口/差值/RankIC 无影响。

总体判据 (预设): 所有窗 gt6 缺口 <3pp 且 IC 均值不降 (−0.005) 且 spread
均值不恶化 (−0.5pp)。

用法: python tmp_t/_ranktarget_minibacktest_v2_0910.py [--board main|dual] [--windows 5]
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

import config.settings as rank_cfg

rank_cfg.LEGACY_10D_RANK_TARGET["enable"] = True

from app.pipeline1.dual_track_trainer import (  # noqa: E402
    DualTrackTrainer,
    fit_rank_map,
    rank_map_apply,
    risk_filter,
)
from config.settings import data_others_path  # noqa: E402

STAGE = {
    "main": "data/_diag_stage_main_3y.parquet",
    "dual": "data/_diag_stage_dual_3y.parquet",
}
MODEL_DIR = "models/pipeline1"
META_COLS = ["date", "symbol", "label_10d_net", "close_hfq"]
WIN = 60  # 每 test 窗交易日数 (与 split_window test 段一致)
ES_N = 20


def _rank_ic(pred, real, dates):
    from scipy.stats import spearmanr

    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 5:
            continue
        r = spearmanr(pred[m], real[m])[0]
        if np.isfinite(r):
            ics.append(r)
    return (float(np.mean(ics)) if ics else float("nan")), len(ics)


def eval_window(trainer, prod, cols, df, board, k):
    dates = sorted(df["date"].unique())
    end = len(dates) - k * WIN
    if end - WIN - ES_N < 60:
        return None
    test_d = set(dates[end - WIN : end])
    es_d = set(dates[end - WIN - ES_N : end - WIN])
    train_d = set(dates[: end - WIN - ES_N])
    tr_df = df[df["date"].isin(train_d)]
    es_df = df[df["date"].isin(es_d)]
    te_df = df[df["date"].isin(test_d)].copy()

    model_new, label = trainer._train_one(
        "10d_reg", {"train": tr_df, "es": es_df}, cols, board
    )
    trf = risk_filter(tr_df.dropna(subset=[label]))
    rmap = fit_rank_map(model_new, trf, label, cols, bins=20)
    del tr_df, es_df, trf
    gc.collect()

    X = np.nan_to_num(te_df[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    new_pred = rank_map_apply(rmap, model_new.predict(X))
    prod_pred = prod["models"]["10d_reg"][0].predict(X)
    d = pd.DataFrame(
        {
            "date": te_df["date"].values,
            "symbol": te_df["symbol"].astype(str).values,
            "new": new_pred,
            "prod": prod_pred,
            "real": te_df[label].values,
            "r5": te_df["r5"].values,
        }
    ).dropna(subset=["real"])

    rep = {
        "window": k,
        "test_span": [str(pd.Timestamp(d["date"].min()).date()),
                      str(pd.Timestamp(d["date"].max()).date())],
        "n_eval": int(len(d)),
    }
    for name, (lo, hi) in (("gt6", (0.06, 10.0)), ("3_6", (0.03, 0.06)), ("0_3", (0.0, 0.03))):
        blk = {}
        for head in ("new", "prod"):
            g = d[(d[head] > lo) & (d[head] <= hi)]
            blk[head] = {
                "n": int(len(g)),
                "promise": float(g[head].mean()) if len(g) else None,
                "real": float(g["real"].mean()) if len(g) else None,
                "gap_pp": float((g["real"] - g[head]).mean() * 100) if len(g) else None,
            }
        rep[f"calib_{name}"] = blk
    for head in ("new", "prod"):
        d[f"rk_{head}"] = d.groupby("date")[head].rank(ascending=False, method="first")
        top = d[d[f"rk_{head}"] <= 10]
        mid = d[(d[f"rk_{head}"] > 10) & (d[f"rk_{head}"] <= 30)]
        rep[f"rank_{head}"] = {
            "top10_real": float(top["real"].mean()),
            "r11_30_real": float(mid["real"].mean()),
            "spread_pp": float((top["real"].mean() - mid["real"].mean()) * 100),
        }
    real_v = d["real"].to_numpy(dtype=float)
    for head in ("new", "prod"):
        ic, n = _rank_ic(d[head].to_numpy(dtype=float), real_v, d["date"].to_numpy())
        rep[f"rankic_{head}"] = {"ic": ic, "n_days": n}
    dn = d[d["r5"] < 0]
    rep["down_slice"] = {
        "n_down": int(len(dn)),
        "new_top10_down": int((dn["rk_new"] <= 10).sum()),
        "prod_top10_down": int((dn["rk_prod"] <= 10).sum()),
        "new_top10_down_real": float(dn.loc[dn["rk_new"] <= 10, "real"].mean()) if (dn["rk_new"] <= 10).any() else None,
        "prod_top10_down_real": float(dn.loc[dn["rk_prod"] <= 10, "real"].mean()) if (dn["rk_prod"] <= 10).any() else None,
    }
    g6n = (rep["calib_gt6"]["new"] or {}).get("gap_pp")
    rep["window_PASS_gap"] = bool(g6n is not None and g6n < 3.0)
    return rep


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=("main", "dual"), default="main")
    ap.add_argument("--windows", type=int, default=5)
    args = ap.parse_args()
    board = args.board

    import psutil

    free_gb = psutil.virtual_memory().available / 1024**3
    if free_gb < 5.0:
        print(f"[ram] 空闲 {free_gb:.1f}GB < 5GB, 退出 (rc=3)", flush=True)
        return 3

    prod = DualTrackTrainer.load(os.path.join(MODEL_DIR, f"{board}_current.pkl"))
    cols = list(prod["feature_cols"])
    import pyarrow.parquet as pq

    stage = STAGE[board]
    stage_names = set(pq.ParquetFile(stage).schema_arrow.names)
    missing = [c for c in cols + META_COLS if c not in stage_names]
    if missing:
        print(f"[stage] 缺列 {len(missing)}: {missing[:5]}", flush=True)
        return 4

    print(f"[load] {stage}", flush=True)
    df = pd.read_parquet(stage, columns=sorted(set(cols + META_COLS)))
    df[cols] = df[cols].astype("float32", copy=False)
    df = df.sort_values(["symbol", "date"])
    df["r5"] = df.groupby("symbol")["close_hfq"].pct_change(5)
    print(f"[load] {len(df)} rows x {df.shape[1]} cols ({time.time() - t0:.0f}s)", flush=True)

    trainer = DualTrackTrainer(model_dir=MODEL_DIR)
    reps = []
    for k in range(args.windows):
        r = eval_window(trainer, prod, cols, df, board, k)
        if r is None:
            break
        reps.append(r)
        print(
            f"[win{k}] {r['test_span'][0]}→{r['test_span'][1]} "
            f"gt6缺口 new {r['calib_gt6']['new']['gap_pp']:.2f} vs prod {r['calib_gt6']['prod']['gap_pp']:.2f} | "
            f"top10 new {r['rank_new']['top10_real']*100:.2f}% vs prod {r['rank_prod']['top10_real']*100:.2f}% | "
            f"spread new {r['rank_new']['spread_pp']:.2f} vs prod {r['rank_prod']['spread_pp']:.2f} | "
            f"IC new {r['rankic_new']['ic']:.4f} vs prod {r['rankic_prod']['ic']:.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        gc.collect()

    n = len(reps)
    agg = {
        "board": board,
        "n_windows": n,
        "gap_all_lt3": all(r["window_PASS_gap"] for r in reps),
        "ic_mean_new": float(np.mean([r["rankic_new"]["ic"] for r in reps])) if n else None,
        "ic_mean_prod": float(np.mean([r["rankic_prod"]["ic"] for r in reps])) if n else None,
        "spread_mean_new": float(np.mean([r["rank_new"]["spread_pp"] for r in reps])) if n else None,
        "spread_mean_prod": float(np.mean([r["rank_prod"]["spread_pp"] for r in reps])) if n else None,
        "top10_real_mean_new": float(np.mean([r["rank_new"]["top10_real"] for r in reps])) if n else None,
        "top10_real_mean_prod": float(np.mean([r["rank_prod"]["top10_real"] for r in reps])) if n else None,
        "windows": reps,
    }
    if n:
        agg["PASS"] = bool(
            agg["gap_all_lt3"]
            and agg["ic_mean_new"] >= agg["ic_mean_prod"] - 0.005
            and agg["spread_mean_new"] >= agg["spread_mean_prod"] - 0.5
        )
        wins_top10 = sum(1 for r in reps if r["rank_new"]["top10_real"] >= r["rank_prod"]["top10_real"])
        agg["top10_new_wins_k_of_n"] = f"{wins_top10}/{n}"

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"ranktarget_minibacktest_v2_{board}_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2, default=str)
    if n:
        print(
            f"[verdict] {board} 达线 {'PASS' if agg['PASS'] else 'FAIL'} | "
            f"全窗缺口<3pp: {agg['gap_all_lt3']} | IC均值 new {agg['ic_mean_new']:.4f} vs prod {agg['ic_mean_prod']:.4f} | "
            f"spread均值 new {agg['spread_mean_new']:.2f}pp vs prod {agg['spread_mean_prod']:.2f}pp | "
            f"top10实得均值 new {agg['top10_real_mean_new']*100:.2f}% vs prod {agg['top10_real_mean_prod']*100:.2f}% "
            f"(新头胜 {agg['top10_new_wins_k_of_n']} 窗) | ({time.time() - t0:.0f}s)",
            flush=True,
        )
    print(f"[worm] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
