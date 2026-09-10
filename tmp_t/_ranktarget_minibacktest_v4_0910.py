# -*- coding: utf-8 -*-
"""小型回测 v4: 三头对拍 — 衰减陡度 × 目标形状 (用户指令: 最近5天权重最大).

背景:
- v2 缺陷: 生产 current 包 (09-05 训) 当对照头 = 在样本内被打分, 无效。
- v3 修正为每窗现训双头 (幅度 vs 秩, 均真 OOS), 但只覆盖目标形状一个轴。
- 生产 reg 头自 09-03 起已有时间衰减样本权重 (prob_head.HALF_LIFE_DAYS=60
  自然日半衰期); 5 天前权重 0.94 ≈ 无差别 — 用户指令: 最近 5 天权重应最大
  → 半衰期 ~5 自然日 (d-5 权重 0.5, d-10 0.25, d-30 0.03)。

设计: 5×60 交易日 walk-forward, 每窗现训三头, 唯一变量 = 衰减陡度/目标形状:
  mag60 : 幅度 Huber + 衰减 60 自然日 (= 现生产配方)
  mag5  : 幅度 Huber + 衰减 5 自然日  (= 用户处方)
  rank60: 秩目标 + 衰减 60 自然日     (= LEGACY_10D_RANK_TARGET flag ON)
全部只用窗前数据 → 全部真 OOS。半衰期经 prob_head 模块属性进程内切换
(_train_one 函数体内 from-import, 每次调用取当前值)。

判决:
  RANK  : 所有窗 rank60 gt6 缺口 <3pp 且 IC 均值 ≥ mag60−0.005 且 spread ≥
          mag60−0.5 (同 v2/v3 判据)
  DECAY : mag5 vs mag60 — IC 均值不降 (−0.005) 且 spread 不恶化 (−0.5pp)
  有效性: mag60 在 win0 (最近窗) 的 gt6 缺口应为明显负值 (审计 −9.7~−11.8pp
  同向); 非负 → SUSPECT。

用法: python tmp_t/_ranktarget_minibacktest_v4_0910.py [--board main|dual] [--windows 5]
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

import app.pipeline_parallel.prob_head as prob_head_mod
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
PROD_HL = 60
ARMS = (("mag60", False, PROD_HL), ("mag5", False, 5), ("rank60", True, PROD_HL))
ARM_NAMES = tuple(a[0] for a in ARMS)


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


def _metrics(d, heads):
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
    ds = {"n_down": int(len(dn))}
    for head in heads:
        ds[f"{head}_top10_down"] = int((dn[f"rk_{head}"] <= 10).sum())
        ds[f"{head}_top10_down_real"] = (
            float(dn.loc[dn[f"rk_{head}"] <= 10, "real"].mean())
            if (dn[f"rk_{head}"] <= 10).any()
            else None
        )
    rep["down_slice"] = ds
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

    preds = {}
    for name, use_rank, hl in ARMS:
        rank_cfg.LEGACY_10D_RANK_TARGET["enable"] = bool(use_rank)
        prob_head_mod.HALF_LIFE_DAYS = hl
        model, label = trainer._train_one(
            "10d_reg", {"train": tr_df, "es": es_df}, cols, board
        )
        if use_rank:
            trf = risk_filter(tr_df.dropna(subset=[label]))
            rmap = fit_rank_map(model, trf, label, cols, bins=20)
            del trf
            gc.collect()
        X = np.nan_to_num(te_df[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        raw = model.predict(X)
        preds[name] = rank_map_apply(rmap, raw) if use_rank else raw
        del model, X, raw
        gc.collect()
    rank_cfg.LEGACY_10D_RANK_TARGET["enable"] = False
    prob_head_mod.HALF_LIFE_DAYS = PROD_HL
    del tr_df, es_df
    gc.collect()

    d = pd.DataFrame(
        {
            "date": te_df["date"].values,
            "symbol": te_df["symbol"].astype(str).values,
            "real": te_df["label_10d_net"].values,
            "r5": te_df["r5"].values,
            **preds,
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
    rep.update(_metrics(d, ARM_NAMES))
    g6r = (rep["calib_gt6"]["rank60"] or {}).get("gap_pp")
    rep["window_PASS_gap"] = bool(g6r is not None and g6r < 3.0)
    rep["mag60_gt6_gap_pp"] = (rep["calib_gt6"]["mag60"] or {}).get("gap_pp")
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

    import pickle

    import pyarrow.parquet as pq

    stage = STAGE[board]
    stage_names = set(pq.ParquetFile(stage).schema_arrow.names)
    with open(os.path.join("models", "pipeline1", f"{board}_current.pkl"), "rb") as fh:
        prod_meta = pickle.load(fh)
    cols = list(prod_meta["feature_cols"])
    del prod_meta
    missing = [c for c in cols + META_COLS if c not in stage_names]
    if missing:
        print(f"[stage] 缺列 {len(missing)}: {missing[:5]} (rc=4)", flush=True)
        return 4

    print(f"[load] {stage}", flush=True)
    df = pd.read_parquet(stage, columns=sorted(set(cols + META_COLS)))
    df[cols] = df[cols].astype("float32", copy=False)
    df = df.sort_values(["symbol", "date"])
    df["r5"] = df.groupby("symbol")["close_hfq"].pct_change(5)
    print(f"[load] {len(df)} rows x {df.shape[1]} cols ({time.time() - t0:.0f}s)", flush=True)

    trainer = DualTrackTrainer(model_dir=os.path.join("models", "pipeline1"))
    reps = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"ranktarget_decayrank_v4_{board}_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for k in range(args.windows):
        r = eval_window(trainer, cols, df, board, k)
        if r is None:
            break
        reps.append(r)
        seg = " | ".join(
            f"{a} gap {r['calib_gt6'][a]['gap_pp']:.2f} top10 {r[f'rank_{a}']['top10_real']*100:.2f}% "
            f"IC {r[f'rankic_{a}']['ic']:.4f}"
            for a in ARM_NAMES
        )
        print(f"[win{k}] {r['test_span'][0]}→{r['test_span'][1]} {seg} ({time.time() - t0:.0f}s)", flush=True)
        # 每窗先落盘 (聚合步崩溃不丢窗口数据)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"board": board, "n_windows": len(reps), "windows": reps}, fh, ensure_ascii=False, indent=2, default=str)
        gc.collect()

    n = len(reps)

    def mean_over(key_path):
        vals = []
        for r in reps:
            v = r
            for kk in key_path:
                v = v[kk]
            vals.append(v)
        return float(np.mean(vals)) if vals else None

    agg = {
        "board": board,
        "n_windows": n,
        "arms": list(ARM_NAMES),
        "ic_mean": {a: mean_over((f"rankic_{a}", "ic")) for a in ARM_NAMES},
        "spread_mean_pp": {a: mean_over((f"rank_{a}", "spread_pp")) for a in ARM_NAMES},
        "top10_real_mean": {a: mean_over((f"rank_{a}", "top10_real")) for a in ARM_NAMES},
        "mag60_win0_gt6_gap_pp": reps[0]["mag60_gt6_gap_pp"] if n else None,
        "windows": reps,
    }
    if n:
        agg["gap_all_lt3_rank60"] = all(r["window_PASS_gap"] for r in reps)
        agg["PASS_rank60"] = bool(
            agg["gap_all_lt3_rank60"]
            and agg["ic_mean"]["rank60"] >= agg["ic_mean"]["mag60"] - 0.005
            and agg["spread_mean_pp"]["rank60"] >= agg["spread_mean_pp"]["mag60"] - 0.5
        )
        agg["PASS_decay5"] = bool(
            agg["ic_mean"]["mag5"] >= agg["ic_mean"]["mag60"] - 0.005
            and agg["spread_mean_pp"]["mag5"] >= agg["spread_mean_pp"]["mag60"] - 0.5
        )
        wins_top10 = {
            a: sum(1 for r in reps if r[f"rank_{a}"]["top10_real"] >= max(
                r[f"rank_{b}"]["top10_real"] for b in ARM_NAMES if b != a
            ))
            for a in ARM_NAMES
        }
        agg["top10_arm_wins_k_of_n"] = {a: f"{v}/{n}" for a, v in wins_top10.items()}
        o0 = agg["mag60_win0_gt6_gap_pp"]
        agg["validity"] = (
            "OK" if (o0 is not None and o0 <= -4.0) else "SUSPECT(mag60头win0缺口非负,幅度头过度承诺未现)"
        )

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2, default=str)
    if n:
        print(
            f"[verdict] {board} | RANK(秩目标) {'PASS' if agg['PASS_rank60'] else 'FAIL'} "
            f"(全窗缺口<3pp: {agg['gap_all_lt3_rank60']}) | DECAY5(半衰期5日) "
            f"{'PASS' if agg['PASS_decay5'] else 'FAIL'} | mag60 win0缺口 {agg['mag60_win0_gt6_gap_pp']:.2f}pp "
            f"({agg['validity']}) | "
            + " | ".join(
                f"{a}: IC {agg['ic_mean'][a]:.4f} spread {agg['spread_mean_pp'][a]:.2f}pp "
                f"top10 {agg['top10_real_mean'][a]*100:.2f}% (胜{agg['top10_arm_wins_k_of_n'][a]})"
                for a in ARM_NAMES
            )
            + f" | ({time.time() - t0:.0f}s)",
            flush=True,
        )
    print(f"[worm] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
