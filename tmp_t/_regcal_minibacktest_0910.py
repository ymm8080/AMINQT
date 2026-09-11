# -*- coding: utf-8 -*-
"""小型回测 v5: 行情条件化承诺校准 (计划见 memory/regcal-v5-plan-0910.md).

v4 定位: 幅度头真病 = 承诺缺口的行情依赖 (弱市超诺 4.7~5.2pp / 强市少诺 0.8pp,
5 窗均值 −2.81pp)。v5 检验: 树模型不动 (mag60 现产配方原样), 训练段内按
{弱市日, 强市日} 各拟合一条 isotonic pred→label 映射, 测试日按当日行情状态套用。

行情状态 (零前视): 当日截面 market mean r5 (全帧 close_hfq pct_change(5) 的
per-date 均值, 预测日收盘即已知); < 0 → 弱市日, 否则强市日。

臂:
  mag60        : 幅度 Huber + 60 自然日衰减, 原样预测 (基线, 与 v4 同配方同切分可对表)
  mag60_regcal : 同一模型, 预测过按行情分套的 isotonic 校准后输出

达线判据 (预设, 勿临场改):
  1) regcal gt6 缺口均值收敛到 ±1.5pp 内 (v4 基线 −2.81pp 至少收敛一半);
  2) IC 均值 ≥ mag60−0.005 且 TOP10 实得 ≥ mag60−0.5pp 且 spread ≥ mag60−0.5pp;
  3) 无单窗 gt6 缺口 < −4pp。
  PASS → 生产化 (bundle 存两条 isotonic + 状态函数, config 门控, 受控 A/B);
  FAIL → 行情依赖非单调校准可修形状, 回病灶假设, 勿硬扫参数。

用法: python tmp_t/_regcal_minibacktest_0910.py [--board main|dual] [--windows 5]
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

from app.pipeline1.dual_track_trainer import DualTrackTrainer  # noqa: E402
from config.settings import data_others_path  # noqa: E402

STAGE = {
    "main": "data/_diag_stage_main_3y.parquet",
    "dual": "data/_diag_stage_dual_3y.parquet",
}
META_COLS = ["date", "symbol", "label_10d_net", "close_hfq"]
WIN = 60
ES_N = 20
ARMS = ("mag60", "regcal")
# v4 主板参照 (同切分, 对表用, 非闸): gap −2.81pp / IC 0.0550 / top10 5.91%
V4_MAIN_REF = {"gap_pp": -2.81, "ic": 0.0550, "top10_pct": 5.91}


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


def _metrics(d):
    rep = {}
    for name, (lo, hi) in (("gt6", (0.06, 10.0)), ("3_6", (0.03, 0.06)), ("0_3", (0.0, 0.03))):
        blk = {}
        for head in ARMS:
            g = d[(d[head] > lo) & (d[head] <= hi)]
            blk[head] = {
                "n": int(len(g)),
                "promise": float(g[head].mean()) if len(g) else None,
                "real": float(g["real"].mean()) if len(g) else None,
                "gap_pp": float((g["real"] - g[head]).mean() * 100) if len(g) else None,
            }
        rep[f"calib_{name}"] = blk
    for head in ARMS:
        d[f"rk_{head}"] = d.groupby("date")[head].rank(ascending=False, method="first")
        top = d[d[f"rk_{head}"] <= 10]
        mid = d[(d[f"rk_{head}"] > 10) & (d[f"rk_{head}"] <= 30)]
        rep[f"rank_{head}"] = {
            "top10_real": float(top["real"].mean()),
            "r11_30_real": float(mid["real"].mean()),
            "spread_pp": float((top["real"].mean() - mid["real"].mean()) * 100),
        }
    real_v = d["real"].to_numpy(dtype=float)
    for head in ARMS:
        ic, n = _rank_ic(d[head].to_numpy(dtype=float), real_v, d["date"].to_numpy())
        rep[f"rankic_{head}"] = {"ic": ic, "n_days": n}
    # 行情分桶读数: 弱/强测试日各自的 gt6 缺口 + 全样本缺口 (本实验的直接靶点)
    for reg, m in (("weak", d["regime"] < 0), ("strong", d["regime"] >= 0)):
        blk = {}
        for head in ARMS:
            g = d[m & (d[head] > 0.06)]
            blk[head] = {
                "n": int(len(g)),
                "gap_pp": float((g["real"] - g[head]).mean() * 100) if len(g) else None,
            }
        blk["n_days"] = int((d["regime"] < 0).sum() if reg == "weak" else (d["regime"] >= 0).sum())
        rep[f"regime_gt6_{reg}"] = blk
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

    # ── 单次训练: mag60 现产配方 (幅度 Huber + 60 自然日衰减, prob_head 默认未动) ──
    model, label = trainer._train_one("10d_reg", {"train": tr_df, "es": es_df}, cols, board)

    from sklearn.isotonic import IsotonicRegression

    X_tr = np.nan_to_num(tr_df[cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pred_tr = model.predict(X_tr)
    del X_tr
    gc.collect()
    lab_tr = tr_df[label].to_numpy(dtype=float)
    reg_tr = tr_df["date"].map(mkt_r5).to_numpy(dtype=float)  # 训练日行情状态

    iso = {}
    fit_n = {}
    for reg, m in (("weak", reg_tr < 0), ("strong", ~(reg_tr < 0) & np.isfinite(reg_tr))):
        mm = m & np.isfinite(lab_tr) & np.isfinite(pred_tr)
        iso[reg] = IsotonicRegression(out_of_bounds="clip")
        iso[reg].fit(pred_tr[mm], lab_tr[mm])
        fit_n[reg] = int(mm.sum())
    del pred_tr, lab_tr, reg_tr
    gc.collect()

    X = np.nan_to_num(te_df[cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    raw = model.predict(X)
    del model, X
    gc.collect()

    te_reg = te_df["date"].map(mkt_r5).fillna(0.0).to_numpy(dtype=float)  # NaN 默认强市
    cal = np.empty_like(raw)
    for reg, m in (("weak", te_reg < 0), ("strong", te_reg >= 0)):
        cal[m] = iso[reg].predict(raw[m])

    d = pd.DataFrame(
        {
            "date": te_df["date"].values,
            "symbol": te_df["symbol"].astype(str).values,
            "mag60": raw,
            "regcal": cal,
            "real": te_df["label_10d_net"].values,
            "r5": te_df["r5"].values,
            "regime": te_reg,
        }
    ).dropna(subset=["real"])
    del te_df, raw, cal
    gc.collect()

    rep = {
        "window": k,
        "test_span": [str(pd.Timestamp(d["date"].min()).date()),
                      str(pd.Timestamp(d["date"].max()).date())],
        "n_eval": int(len(d)),
        "train_days": len(train_d),
        "calib_fit_rows": fit_n,
    }
    rep.update(_metrics(d))
    g6r = (rep["calib_gt6"]["regcal"] or {}).get("gap_pp")
    rep["window_gap_ok"] = bool(g6r is not None and g6r > -4.0)
    rep["mag60_gt6_gap_pp"] = (rep["calib_gt6"]["mag60"] or {}).get("gap_pp")
    return rep


def main() -> int:
    global mkt_r5
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
    mkt_r5 = df.groupby("date")["r5"].mean()  # 当日截面市场平均 r5 (零前视)
    print(f"[load] {len(df)} rows x {df.shape[1]} cols; 弱市日占比 "
          f"{(mkt_r5 < 0).mean():.2%} ({time.time() - t0:.0f}s)", flush=True)

    trainer = DualTrackTrainer(model_dir=os.path.join("models", "pipeline1"))
    reps = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"ranktarget_regcal_v5_{board}_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for k in range(args.windows):
        r = eval_window(trainer, cols, df, board, k)
        if r is None:
            break
        reps.append(r)
        # 每窗先落盘 (v4 教训: 打印/聚合任何一步崩溃都不丢窗口数据)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"board": board, "n_windows": len(reps), "windows": reps}, fh,
                      ensure_ascii=False, indent=2, default=str)

        def _fmt(v):
            return f"{v:.2f}" if v is not None else "n/a"

        seg = " | ".join(
            f"{a} gap {_fmt(r['calib_gt6'][a]['gap_pp'])} "
            f"top10 {_fmt(r[f'rank_{a}']['top10_real'] * 100)}% "
            f"IC {_fmt(r[f'rankic_{a}']['ic'])}"
            for a in ARMS
        )
        rw = r["regime_gt6_weak"]
        rs = r["regime_gt6_strong"]
        print(
            f"[win{k}] {r['test_span'][0]}→{r['test_span'][1]} {seg} | "
            f"弱市日 gap mag60 {_fmt(rw['mag60']['gap_pp'])}→regcal {_fmt(rw['regcal']['gap_pp'])} | "
            f"强市日 gap mag60 {_fmt(rs['mag60']['gap_pp'])}→regcal {_fmt(rs['regcal']['gap_pp'])} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
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
        "arms": list(ARMS),
        "ic_mean": {a: mean_over((f"rankic_{a}", "ic")) for a in ARMS},
        "spread_mean_pp": {a: mean_over((f"rank_{a}", "spread_pp")) for a in ARMS},
        "top10_real_mean": {a: mean_over((f"rank_{a}", "top10_real")) for a in ARMS},
        "gt6_gap_mean_pp": {a: mean_over(("calib_gt6", a, "gap_pp")) for a in ARMS},
        "regime_gap_mean_pp": {
            reg: {a: mean_over((f"regime_gt6_{reg}", a, "gap_pp")) for a in ARMS}
            for reg in ("weak", "strong")
        },
        "windows": reps,
    }
    if n:
        gap_ok = all(r["window_gap_ok"] for r in reps)
        base_gap = agg["gt6_gap_mean_pp"]["mag60"]
        rc_gap = agg["gt6_gap_mean_pp"]["regcal"]
        agg["gate"] = {
            "gap_mean_abs_le_1.5": bool(abs(rc_gap) <= 1.5),
            "ic_not_hurt": bool(agg["ic_mean"]["regcal"] >= agg["ic_mean"]["mag60"] - 0.005),
            "top10_not_hurt": bool(
                agg["top10_real_mean"]["regcal"] >= agg["top10_real_mean"]["mag60"] - 0.005
            ),
            "spread_not_hurt": bool(agg["spread_mean_pp"]["regcal"] >= agg["spread_mean_pp"]["mag60"] - 0.5),
            "no_window_gap_lt_-4": gap_ok,
        }
        agg["PASS"] = bool(all(agg["gate"].values()))
        agg["v4_ref_crosscheck"] = {
            "v4_main_ref": V4_MAIN_REF,
            "note": "mag60 本跑与 v4 同配方同切分, 数字应接近 (LGBM 种子相同)",
        }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2, default=str)
    if n:
        rg = agg["regime_gap_mean_pp"]

        def _fmt(v, nd=2):
            return f"{v:.{nd}f}" if v is not None else "n/a"

        print(
            f"[verdict] {board} v5 行情条件化校准 {'PASS' if agg['PASS'] else 'FAIL'} | "
            f"gt6缺口均值 mag60 {_fmt(base_gap)}pp → regcal {_fmt(rc_gap)}pp (闸±1.5) | "
            f"弱市 {_fmt(rg['weak']['mag60'])}→{_fmt(rg['weak']['regcal'])} | "
            f"强市 {_fmt(rg['strong']['mag60'])}→{_fmt(rg['strong']['regcal'])} | "
            f"IC {_fmt(agg['ic_mean']['mag60'], 4)}→{_fmt(agg['ic_mean']['regcal'], 4)} | "
            f"top10 {_fmt(agg['top10_real_mean']['mag60'] * 100)}%"
            f"→{_fmt(agg['top10_real_mean']['regcal'] * 100)}% | "
            f"单窗不爆(>-4pp): {gap_ok} | "
            + " ".join(f"{k}={v}" for k, v in agg["gate"].items())
            + f" | ({time.time() - t0:.0f}s)",
            flush=True,
        )
    print(f"[worm] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
