# -*- coding: utf-8 -*-
"""小型回测: 10d 秩目标头 OOS 达线判定 (688228 案; 用户指令: BACKTESTING 小测试定进产).

数据: data/_diag_stage_main_3y.parquet (09-09 22:28 诊断遗留全特征帧, 585 列,
max=2026-09-09, 与当前面板同源) — 跳过 ~64min 特征构建, 全程 ~20min。
设计: 生产同款特征列 (main_current 268) + 同款 split_window + 同款 _train_one
超参, 唯一变量 = 训练目标 (幅度 Huber → per-date 截面百分位 + 桶中位映射)。
有效自检: 生产头在本 test 段 gt6 校准缺口应 ≈ −4pp 以下 (审计 −9.7~−11.8pp 同向);
偏离过大 = staging 帧可疑, 输出降级标记。

判据: pred10>6% 桶校准缺口 <3pp 且 RankIC 不降 (容差 −0.005) 且
top10-vs-11-30 不恶化 (容差 −0.5pp)。
达线 → settings enable=True → 周五夜链 (RETRAIN_WEEKDAY=4, 自动带 09-10/09-11
数据) 新目标进生产, IC 发布闸兜底。

用法: 链退出后 python tmp_t/_ranktarget_minibacktest_0910.py
"""

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

# 进程内打开秩目标 (生产 settings.py 默认 False; 小测试专用, 不落盘)
import config.settings as rank_cfg

rank_cfg.LEGACY_10D_RANK_TARGET["enable"] = True

from app.pipeline1.dual_track_trainer import (  # noqa: E402
    DualTrackTrainer,
    fit_rank_map,
    rank_map_apply,
    risk_filter,
)
from config.settings import data_others_path  # noqa: E402

STAGE_PARQUET = "data/_diag_stage_main_3y.parquet"
MODEL_DIR = "models/pipeline1"
META_COLS = ["date", "symbol", "label_10d_net", "close_hfq"]


def _rank_ic(pred: np.ndarray, real: np.ndarray, dates: np.ndarray) -> tuple[float, int]:
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


def main() -> int:
    t0 = time.time()
    import psutil

    free_gb = psutil.virtual_memory().available / 1024**3
    if free_gb < 5.0:
        print(f"[ram] 空闲 {free_gb:.1f}GB < 5GB, 有并发重活, 退出 (rc=3)", flush=True)
        return 3

    prod = DualTrackTrainer.load(os.path.join(MODEL_DIR, "main_current.pkl"))
    cols = list(prod["feature_cols"])
    import pyarrow.parquet as pq

    stage_names = set(pq.ParquetFile(STAGE_PARQUET).schema_arrow.names)
    missing = [c for c in cols + META_COLS if c not in stage_names]
    if missing:
        print(f"[stage] 缺列 {len(missing)}: {missing[:5]} — staging 帧不可用, 改跑全量影子", flush=True)
        return 4

    print(f"[load] {STAGE_PARQUET} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.read_parquet(STAGE_PARQUET, columns=sorted(set(cols + META_COLS)))
    df[cols] = df[cols].astype("float32", copy=False)
    print(f"[load] {len(df)} rows x {df.shape[1]} cols ({time.time() - t0:.0f}s)", flush=True)

    # r5 在全窗上算 (test 段头几天有历史上下文), 再切分
    df = df.sort_values(["symbol", "date"])
    df["r5"] = df.groupby("symbol")["close_hfq"].pct_change(5)

    segs = DualTrackTrainer.split_window(df)
    del df
    gc.collect()
    test = segs["test"].copy()
    print(f"[split] test={test['date'].nunique()} 日 ({time.time() - t0:.0f}s)", flush=True)

    # ── 新头: 生产同款 _train_one, 唯一变量=秩目标 (flag 进程内 ON) ──
    trainer = DualTrackTrainer(model_dir=MODEL_DIR)
    model_new, label = trainer._train_one(
        "10d_reg", {"train": segs["train"], "es": segs["es"]}, cols, "main"
    )
    tr = risk_filter(segs["train"].dropna(subset=[label]))
    rmap = fit_rank_map(model_new, tr, label, cols, bins=20)
    del segs, tr
    gc.collect()
    print(f"[train] 10d 秩目标头 + map 完成 ({time.time() - t0:.0f}s)", flush=True)

    X = np.nan_to_num(test[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    new_pred = rank_map_apply(rmap, model_new.predict(X))
    prod_pred = prod["models"]["10d_reg"][0].predict(X)

    d = pd.DataFrame(
        {
            "date": test["date"].values,
            "symbol": test["symbol"].astype(str).values,
            "new": new_pred,
            "prod": prod_pred,
            "real": test["label_10d_net"].values,
            "r5": test["r5"].values,
        }
    ).dropna(subset=["real"])  # 末 ~10 日 label 未成熟, 双头公平剔除
    rep: dict = {"board": "main", "n_eval": int(len(d)),
                 "n_days": int(d["date"].nunique()),
                 "test_span": [str(pd.Timestamp(d['date'].min()).date()),
                               str(pd.Timestamp(d['date'].max()).date())]}

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

    # ── 688228 个案 (末 test 日) + 生产头 TreeSHAP (回答"HOW 排进 TOP10") ──
    g = d[d["symbol"].str.startswith("688228")].sort_values("date")
    if len(g):
        last = g.iloc[-1]
        rep["case_688228"] = {
            "date": str(pd.Timestamp(last["date"]).date()),
            "new_promise": float(last["new"]),
            "prod_promise": float(last["prod"]),
            "real_fwd10": float(last["real"]),
            "r5_at_day": None if pd.isna(last["r5"]) else float(last["r5"]),
        }
    full = pd.read_parquet(STAGE_PARQUET, columns=sorted(set(cols + ["date", "symbol"])))
    crow = (
        full[full["symbol"].astype(str).str.startswith("688228")]
        .sort_values("date")
        .tail(1)
    )
    del full
    gc.collect()
    if len(crow):
        Xc = np.nan_to_num(crow[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        m = prod["models"]["10d_reg"][0]
        contrib = m.predict(Xc, pred_contrib=True)[0]
        vals = contrib[:-1]
        order = np.argsort(-np.abs(vals))[:12]
        rep["shap_688228_prod"] = {
            "date": str(pd.Timestamp(crow["date"].iloc[0]).date()),
            "pred10_prod": float(m.predict(Xc)[0]),
            "bias": float(contrib[-1]),
            "top_contrib": [{"feature": cols[i], "contrib": float(vals[i])} for i in order],
        }

    # ── 判决 ──
    pg = (rep["calib_gt6"]["prod"] or {}).get("gap_pp")
    ng = (rep["calib_gt6"]["new"] or {}).get("gap_pp")
    ic_n = rep["rankic_new"]["ic"]
    ic_p = rep["rankic_prod"]["ic"]
    sp_n = rep["rank_new"]["spread_pp"]
    sp_p = rep["rank_prod"]["spread_pp"]
    rep["validity"] = "OK" if (pg is not None and pg <= -4.0) else "SUSPECT(staging帧与审计缺口偏离)"
    rep["verdict"] = {
        "gap_new_pp": ng,
        "gap_prod_pp": pg,
        "ic_new": ic_n,
        "ic_prod": ic_p,
        "spread_new_pp": sp_n,
        "spread_prod_pp": sp_p,
        "PASS": bool(ng is not None and ng < 3.0 and ic_n >= ic_p - 0.005 and sp_n >= sp_p - 0.5),
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"ranktarget_minibacktest_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(rep, ensure_ascii=False, indent=2, default=str), flush=True)
    v = rep["verdict"]
    print(
        f"[verdict] 达线 {'PASS' if v['PASS'] else 'FAIL'} | gt6缺口 new {ng}pp vs prod {pg}pp | "
        f"RankIC new {ic_n:.4f} vs prod {ic_p:.4f} | spread new {sp_n}pp vs prod {sp_p}pp | "
        f"validity={rep['validity']} | ({time.time() - t0:.0f}s)",
        flush=True,
    )
    print(f"[worm] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
