# -*- coding: utf-8 -*-
"""小型回测 v3: 10d 秩目标 vs 幅度目标 双头对照 (v2 基线缺陷修正版).

v2 缺陷: 用生产 current 包 (09-05 训) 当对照头, 其训练数据覆盖全部测试窗
→ 生产头在窗内是"在样本内"被打分 (win1/win2 IC 0.43~0.44 = 背答案特征),
对照无效。v3 修正: 每个窗内同训两个头 —
  old: LEGACY_10D_RANK_TARGET.enable=False (幅度 Huber, 生产配方)
  new: enable=True  (per-date 截面百分位 + 桶中位映射)
唯一变量 = 训练目标; 两头都只用窗前数据 → 都是真 OOS。

有效性自检: old 头在最近窗 (win0) 的 gt6 缺口应为明显负值 (审计
−9.7~−11.8pp 同向; 真 OOS 幅度头过度承诺是生产既确认性状); 若为正则
SUSPECT (staging 帧或切分可疑)。

总体判据 (预设, 与 v2 相同): 所有窗 new gt6 缺口 <3pp 且 IC 均值不降
(−0.005) 且 spread 均值不恶化 (−0.5pp)。

用法: python tmp_t/_ranktarget_minibacktest_v3_0910.py [--board main|dual] [--windows 5]
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
META_COLS = ["date", "symbol", "label_10d_net", "close_hfq"]
WIN = 60
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


def _metrics(d, heads=("new", "old")):
    rep = {}
    for name, (lo, hi) in (("gt6", (0.06, 10.0)), ("3_6", (0.03, 0.06)), ("0_3", (0.0, 0.03))):
        blk = {}
        for head in heads:
            g = d[(d[head] > lo) & (d[head] <= hi)]
            blk[head] = {
                "n": int(len(g)),
                "promise": float(g[head].mean()) if len(g) else None,
                "real": float(g["real"].mean()) if len(g) else None,
                "gap_pp": float((g["real"] - g[head]).mean() * 100) if len(g) else None,
            }
        rep[f"calib_{name}"] = blk
    for head in heads:
        d[f"rk_{head}"] = d.groupby("date")[head].rank(ascending=False, method="first")
        top = d[d[f"rk_{head}"] <= 10]
        mid = d[(d[f"rk_{head}"] > 10) & (d[f"rk_{head}"] <= 30)]
        rep[f"rank_{head}"] = {
            "top10_real": float(top["real"].mean()),
            "r11_30_real": float(mid["real"].mean()),
            "spread_pp": float((top["real"].mean() - mid["real"].mean()) * 100),
        }
    real_v = d["real"].to_numpy(dtype=float)
    for head in heads:
        ic, n = _rank_ic(d[head].to_numpy(dtype=float), real_v, d["date"].to_numpy())
        rep[f"rankic_{head}"] = {"ic": ic, "n_days": n}
    dn = d[d["r5"] < 0]
    rep["down_slice"] = {
        "n_down": int(len(dn)),
        "new_top10_down": int((dn["rk_new"] <= 10).sum()),
        "old_top10_down": int((dn["rk_old"] <= 10).sum()),
        "new_top10_down_real": float(dn.loc[dn["rk_new"] <= 10, "real"].mean()) if (dn["rk_new"] <= 10).any() else None,
        "old_top10_down_real": float(dn.loc[dn["rk_old"] <= 10, "real"].mean()) if (dn["rk_old"] <= 10).any() else None,
    }
    return rep


def eval_window(trainer, cols, df, board, k):
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

    # ── new 头: 秩目标 (flag ON) ──
    rank_cfg.LEGACY_10D_RANK_TARGET["enable"] = True
    model_new, label_new = trainer._train_one(
        "10d_reg", {"train": tr_df, "es": es_df}, cols, board
    )
    trf = risk_filter(tr_df.dropna(subset=[label_new]))
    rmap = fit_rank_map(model_new, trf, label_new, cols, bins=20)
    del trf
    gc.collect()

    # ── old 头: 幅度 Huber (生产配方, flag OFF) ──
    rank_cfg.LEGACY_10D_RANK_TARGET["enable"] = False
    model_old, label_old = trainer._train_one(
        "10d_reg", {"train": tr_df, "es": es_df}, cols, board
    )
    rank_cfg.LEGACY_10D_RANK_TARGET["enable"] = True
    del tr_df, es_df
    gc.collect()

    X = np.nan_to_num(te_df[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    new_pred = rank_map_apply(rmap, model_new.predict(X))
    old_pred = model_old.predict(X)
    del model_new, model_old, rmap, X
    gc.collect()

    d = pd.DataFrame(
        {
            "date": te_df["date"].values,
            "symbol": te_df["symbol"].astype(str).values,
            "new": new_pred,
            "old": old_pred,
            "real": te_df["label_10d_net"].values,
            "r5": te_df["r5"].values,
        }
    ).dropna(subset=["real"])
    del te_df
    gc.collect()

    rep = {
        "window": k,
        "test_span": [str(pd.Timestamp(d["date"].min()).date()),
                      str(pd.Timestamp(d["date"].max()).date())],
        "n_eval": int(len(d)),
        "train_days": len(train_d),
    }
    rep.update(_metrics(d))
    g6n = (rep["calib_gt6"]["new"] or {}).get("gap_pp")
    g6o = (rep["calib_gt6"]["old"] or {}).get("gap_pp")
    rep["window_PASS_gap"] = bool(g6n is not None and g6n < 3.0)
    rep["old_gt6_gap_pp"] = g6o
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

    import pyarrow.parquet as pq

    stage = STAGE[board]
    stage_names = set(pq.ParquetFile(stage).schema_arrow.names)
    if [c for c in META_COLS if c not in stage_names]:
        print("[stage] 缺元列, staging 帧不可用 (rc=4)", flush=True)
        return 4

    # 生产 bundle 仅取特征列清单 (不加载模型, 对照头全部现训)
    import pickle

    with open(os.path.join("models", "pipeline1", f"{board}_current.pkl"), "rb") as fh:
        prod_meta = pickle.load(fh)
    cols = list(prod_meta["feature_cols"])
    del prod_meta
    missing = [c for c in cols if c not in stage_names]
    if missing:
        print(f"[stage] 缺特征列 {len(missing)}: {missing[:5]} (rc=4)", flush=True)
        return 4

    print(f"[load] {stage}", flush=True)
    df = pd.read_parquet(stage, columns=sorted(set(cols + META_COLS)))
    df[cols] = df[cols].astype("float32", copy=False)
    df = df.sort_values(["symbol", "date"])
    df["r5"] = df.groupby("symbol")["close_hfq"].pct_change(5)
    print(f"[load] {len(df)} rows x {df.shape[1]} cols ({time.time() - t0:.0f}s)", flush=True)

    trainer = DualTrackTrainer(model_dir=os.path.join("models", "pipeline1"))
    reps = []
    for k in range(args.windows):
        r = eval_window(trainer, cols, df, board, k)
        if r is None:
            break
        reps.append(r)
        print(
            f"[win{k}] {r['test_span'][0]}→{r['test_span'][1]} "
            f"gt6缺口 new {r['calib_gt6']['new']['gap_pp']:.2f} vs old {r['calib_gt6']['old']['gap_pp']:.2f} | "
            f"top10 new {r['rank_new']['top10_real']*100:.2f}% vs old {r['rank_old']['top10_real']*100:.2f}% | "
            f"IC new {r['rankic_new']['ic']:.4f} vs old {r['rankic_old']['ic']:.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        gc.collect()

    n = len(reps)
    agg = {
        "board": board,
        "n_windows": n,
        "gap_all_lt3": all(r["window_PASS_gap"] for r in reps) if n else None,
        "old_win0_gt6_gap_pp": reps[0]["old_gt6_gap_pp"] if n else None,
        "ic_mean_new": float(np.mean([r["rankic_new"]["ic"] for r in reps])) if n else None,
        "ic_mean_old": float(np.mean([r["rankic_old"]["ic"] for r in reps])) if n else None,
        "spread_mean_new": float(np.mean([r["rank_new"]["spread_pp"] for r in reps])) if n else None,
        "spread_mean_old": float(np.mean([r["rank_old"]["spread_pp"] for r in reps])) if n else None,
        "top10_real_mean_new": float(np.mean([r["rank_new"]["top10_real"] for r in reps])) if n else None,
        "top10_real_mean_old": float(np.mean([r["rank_old"]["top10_real"] for r in reps])) if n else None,
        "windows": reps,
    }
    if n:
        agg["PASS"] = bool(
            agg["gap_all_lt3"]
            and agg["ic_mean_new"] >= agg["ic_mean_old"] - 0.005
            and agg["spread_mean_new"] >= agg["spread_mean_old"] - 0.5
        )
        wins_top10 = sum(1 for r in reps if r["rank_new"]["top10_real"] >= r["rank_old"]["top10_real"])
        agg["top10_new_wins_k_of_n"] = f"{wins_top10}/{n}"
        o0 = agg["old_win0_gt6_gap_pp"]
        agg["validity"] = "OK" if (o0 is not None and o0 <= -4.0) else "SUSPECT(old头win0缺口非负,幅度头过度承诺未现)"

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"ranktarget_minibacktest_v3_{board}_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2, default=str)
    if n:
        print(
            f"[verdict] {board} 达线 {'PASS' if agg['PASS'] else 'FAIL'} | "
            f"全窗new缺口<3pp: {agg['gap_all_lt3']} | old头win0缺口 {agg['old_win0_gt6_gap_pp']:.2f}pp ({agg['validity']}) | "
            f"IC均值 new {agg['ic_mean_new']:.4f} vs old {agg['ic_mean_old']:.4f} | "
            f"spread均值 new {agg['spread_mean_new']:.2f}pp vs old {agg['spread_mean_old']:.2f}pp | "
            f"top10实得均值 new {agg['top10_real_mean_new']*100:.2f}% vs old {agg['top10_real_mean_old']*100:.2f}% "
            f"(new胜 {agg['top10_new_wins_k_of_n']} 窗) | ({time.time() - t0:.0f}s)",
            flush=True,
        )
    print(f"[worm] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
