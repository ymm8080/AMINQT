# -*- coding: utf-8 -*-
"""pin-brute 恢复受控 A/B — base (pin_allow_brute=False, 现产 268 口径) vs brute85 (True, 恢复 85 brute).

背景 (main-pin-brute-drop-bug-0910): 08-31 pin 冻结后, _run_bruteforce_dedup pin
路径裸 `f in df.columns` 把快照里 85 个 brute 名整批误杀 (选择期 df 无 brute 列,
训练端才 post-injection 物化) → main 实训 268 列零 brute. 修复 = 逃生舱
pin_allow_brute (commit 04fc6725, 默认关); 本 A/B 是翻门前置 (用户协议:
A/B PASS 才接线进生产).

设计 (与 bkd v6 harness 同源):
- 两臂唯一变量 = FeatureSelector main cfg pin_allow_brute False/True, 真实 select()
  出列 (nan 门 / force_include / missing 报告与生产同一条代码路径).
- stage 帧无 ths 列 → 两臂 force_include 注入对称缺席 (单变量干净).
- brute 列在 select 之后物化 (inject_missing_brute 全历史帧, 后视变换无泄漏),
  训练端生产同函数.
- 判据 (预设, 恢复线非治病线 — 恢复的是已过 250d OOS 季度重选的冻结设计):
  1. 不伤: IC >= base-0.005 且 top10 >= base-0.5pp 且 spread >= base-0.5pp;
  2. 不爆: 无单窗 gt6 缺口比 base 差 > 1pp;
  3. 个案 (报数非硬闸): 688228@09-09 重打分.
  PASS = 1&2 → 翻 pin_allow_brute=true + main 重训生效; FAIL → 维持 268 口径.

用法: python tmp_t/_pin_brute_ab_0910.py [--windows 5]   (--windows 0 = 仅选择+注入干跑)
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

from scripts._run_guard import find_conflicts  # noqa: E402

STAGE = "data/_diag_stage_main_3y.parquet"
PIN_FILE = "selected_main_pinned.json"
META_COLS = ["date", "symbol", "label_10d_net", "close_hfq"]
WIN = 60
ES_N = 20
CASE_SYM, CASE_DATE = "688228", "2026-09-09"
ARMS = ("base", "brute85")
BRUTE_RESTORE_MIN = 50  # brute 臂 brute 名低于此 = 恢复未生效 (配置/基列断链)


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


def _metrics(d, arms):
    rep = {}
    mkt_r5 = d.groupby("date")["r5"].mean()
    weak_days = set(mkt_r5[mkt_r5 < 0].index)
    rep["n_weak_days"] = len(weak_days)
    rep["n_days"] = int(d["date"].nunique())
    for name, (lo, hi) in (("gt6", (0.06, 10.0)), ("3_6", (0.03, 0.06)), ("0_3", (0.0, 0.03))):
        blk = {}
        for a in arms:
            g = d[(d[a] > lo) & (d[a] <= hi)]
            gw = d[(d[a] > lo) & (d[a] <= hi) & d["date"].isin(weak_days)]
            blk[a] = {
                "n": int(len(g)),
                "promise": float(g[a].mean()) if len(g) else None,
                "real": float(g["real"].mean()) if len(g) else None,
                "gap_pp": float((g["real"] - g[a]).mean() * 100) if len(g) else None,
                "weak_gap_pp": float((gw["real"] - gw[a]).mean() * 100) if len(gw) else None,
                "weak_n": int(len(gw)),
            }
        rep[f"calib_{name}"] = blk
    for a in arms:
        d[f"rk_{a}"] = d.groupby("date")[a].rank(ascending=False, method="first")
        top = d[d[f"rk_{a}"] <= 10]
        mid = d[(d[f"rk_{a}"] > 10) & (d[f"rk_{a}"] <= 30)]
        rep[f"rank_{a}"] = {
            "top10_real": float(top["real"].mean()),
            "r11_30_real": float(mid["real"].mean()),
            "spread_pp": float((top["real"].mean() - mid["real"].mean()) * 100),
        }
        dnsl = d[(d["r5"] < 0) & (d[f"rk_{a}"] <= 10)]
        rep[f"downslice_{a}"] = {
            "n": int(len(dnsl)),
            "promise": float(dnsl[a].mean()) if len(dnsl) else None,
            "gap_pp": float((dnsl["real"] - dnsl[a]).mean() * 100) if len(dnsl) else None,
        }
    real_v = d["real"].to_numpy(dtype=float)
    for a in arms:
        ic, n = _rank_ic(d[a].to_numpy(dtype=float), real_v, d["date"].to_numpy())
        rep[f"rankic_{a}"] = {"ic": ic, "n_days": n}
    return rep


def eval_window(trainer, cols_by_arm, df, k):
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
    case_preds = {}
    case_mask = (te_df["symbol"] == CASE_SYM) & (te_df["date"] == CASE_DATE)
    for a in ARMS:
        model, _ = trainer._train_one(
            "10d_reg", {"train": tr_df, "es": es_df}, cols_by_arm[a], "main"
        )
        X = np.nan_to_num(
            te_df[cols_by_arm[a]].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        raw = model.predict(X)
        preds[a] = raw
        if case_mask.any():
            ci = int(np.flatnonzero(case_mask.to_numpy())[0])
            case_preds[a] = float(raw[ci])
        del model, X, raw
        gc.collect()
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
        "case_688228_pred": case_preds,
    }
    rep.update(_metrics(d, ARMS))
    g6b = rep["calib_gt6"]["base"]["gap_pp"]
    rep["window_85_gt6_worse_than_base"] = bool(
        g6b is not None and rep["calib_gt6"]["brute85"]["gap_pp"] is not None
        and rep["calib_gt6"]["brute85"]["gap_pp"] < g6b - 1.0
    )
    return rep


def _f(v, nd=2):
    return "NA" if v is None else f"{v:.{nd}f}"


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=5)
    args = ap.parse_args()

    others = find_conflicts()
    if others:
        for c in others:
            print(f"[guard] 冲突进程: {c['sentinel']} (PID {c['pid']})", flush=True)
        print(f"[guard] 已有 {len(others)} 个重活进程在跑, 本实例退出 (rc=3)", flush=True)
        return 3

    import psutil

    free_gb = psutil.virtual_memory().available / 1024**3
    if free_gb < 5.0:
        print(f"[ram] 空闲 {free_gb:.1f}GB < 5GB, 退出 (rc=3)", flush=True)
        return 3

    import pyarrow.parquet as pq

    from app.pipeline1.dual_track_trainer import DualTrackTrainer
    from app.pipeline1.feature_selector import FeatureSelector, inject_missing_brute

    st_names = set(pq.ParquetFile(STAGE).schema_arrow.names)
    miss_meta = [c for c in META_COLS if c not in st_names]
    if miss_meta:
        print(f"[stage] meta 缺列 {miss_meta} (rc=4)", flush=True)
        return 4

    sel_off = FeatureSelector()  # 默认配置 = 生产口径 (pin_allow_brute=False)
    reg_dir = sel_off.registry_dir
    with open(os.path.join(reg_dir, PIN_FILE), encoding="utf-8") as fh:
        pin = json.load(fh).get("features", [])
    bases = sorted({f.split("_brute_")[0] for f in pin if "_brute_" in f})
    base_miss = [b for b in bases if b not in st_names]
    if base_miss:
        print(f"[stage] brute 基列缺 {len(base_miss)}: {base_miss[:5]} (rc=4)", flush=True)
        return 4

    need = [c for c in pin if "_brute_" not in c and c in st_names]
    need += [b for b in bases if b not in need]
    fi = [c for c in sel_off.config["main"].get("force_include") or []]
    need += [c for c in fi if c in st_names and c not in need]
    print(f"[load] {STAGE} (need {len(need)} cols)", flush=True)
    df = pd.read_parquet(STAGE, columns=sorted(set(need + META_COLS)))
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    df["r5"] = df.groupby("symbol")["close_hfq"].pct_change(5)

    # 两臂出列: 真实 select() 同生产路径 (会各写一份 WORM 快照 selected_main_{ts}.json)
    cols_off = sel_off.select(df, "main")
    cfg_on = json.loads(json.dumps(sel_off.config))
    cfg_on["main"]["pin_allow_brute"] = True
    sel_on = FeatureSelector(config=cfg_on, registry_dir=reg_dir)
    cols_on = sel_on.select(df, "main")
    off_brute = sum(1 for c in cols_off if "_brute_" in c)
    on_brute = sum(1 for c in cols_on if "_brute_" in c)
    print(
        f"[sel] base {len(cols_off)} 列 (brute {off_brute}) | brute85 臂 "
        f"{len(cols_on)} 列 (brute {on_brute}) ({time.time() - t0:.0f}s)",
        flush=True,
    )
    if on_brute < BRUTE_RESTORE_MIN:
        print(f"[sel] brute 臂 brute 名 {on_brute} < {BRUTE_RESTORE_MIN}, 恢复未生效 rc=5", flush=True)
        return 5

    # select 之后物化 brute 列 (全历史帧, 后视变换无泄漏; 与生产训练/推理同函数)
    still = inject_missing_brute(df, cols_on)
    if still:
        print(f"[inject] 仍缺 {len(still)}: {still[:5]} (rc=6)", flush=True)
        return 6
    gc.collect()
    print(f"[inject] brute 列全物化 ({time.time() - t0:.0f}s)", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    from config.settings import data_others_path

    out = data_others_path("diag") / f"pin_brute_ab_main_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.windows <= 0:
        report = {
            "mode": "dry_selection_only",
            "base_cols": len(cols_off),
            "base_brute": off_brute,
            "brute85_cols": len(cols_on),
            "brute85_brute": on_brute,
            "pin_total": len(pin),
            "pin_brute": sum(1 for f in pin if "_brute_" in f),
        }
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
        print(f"[dry] 选择+注入干跑 OK, 报告 {out} ({time.time() - t0:.0f}s)", flush=True)
        return 0

    cols_by_arm = {"base": list(cols_off), "brute85": list(cols_on)}
    trainer = DualTrackTrainer(model_dir=os.path.join("models", "pipeline1"))
    reps = []
    for k in range(args.windows):
        r = eval_window(trainer, cols_by_arm, df, k)
        if r is None:
            break
        reps.append(r)
        seg = " | ".join(
            f"{a} gap {_f(r['calib_gt6'][a]['gap_pp'])} weak {_f(r['calib_gt6'][a]['weak_gap_pp'])} "
            f"top10 {_f(r[f'rank_{a}']['top10_real'] * 100)}% IC {_f(r[f'rankic_{a}']['ic'], 4)}"
            for a in ARMS
        )
        cp = r.get("case_688228_pred") or {}
        cp_s = (
            f" case688228 {cp.get('base', float('nan')):+.4f}->{cp.get('brute85', float('nan')):+.4f}"
            if cp
            else ""
        )
        print(
            f"[win{k}] {r['test_span'][0]}→{r['test_span'][1]} {seg}{cp_s} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"board": "main", "n_windows": len(reps), "windows": reps}, fh,
                      ensure_ascii=False, indent=2, default=str)
        gc.collect()

    n = len(reps)

    def mean_over(key_path):
        vals = []
        for r in reps:
            v = r
            for kk in key_path:
                v = v[kk]
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    agg = {
        "board": "main",
        "n_windows": n,
        "arms": list(ARMS),
        "base_cols": len(cols_off),
        "brute85_cols": len(cols_on),
        "restored_brute_n": on_brute - off_brute,
        "ic_mean": {a: mean_over((f"rankic_{a}", "ic")) for a in ARMS},
        "spread_mean_pp": {a: mean_over((f"rank_{a}", "spread_pp")) for a in ARMS},
        "top10_real_mean": {a: mean_over((f"rank_{a}", "top10_real")) for a in ARMS},
        "gt6_gap_mean_pp": {a: mean_over(("calib_gt6", a, "gap_pp")) for a in ARMS},
        "downslice_gap_mean_pp": {a: mean_over((f"downslice_{a}", "gap_pp")) for a in ARMS},
        "windows": reps,
    }
    if n:
        agg["case_688228_win0"] = reps[0].get("case_688228_pred") or {}
        agg["g1_not_hurt"] = bool(
            agg["ic_mean"]["brute85"] >= agg["ic_mean"]["base"] - 0.005
            and agg["top10_real_mean"]["brute85"] >= agg["top10_real_mean"]["base"] - 0.005
            and agg["spread_mean_pp"]["brute85"] >= agg["spread_mean_pp"]["base"] - 0.5
        )
        agg["g3_no_blowup"] = not any(r["window_85_gt6_worse_than_base"] for r in reps)
        agg["PASS"] = bool(agg["g1_not_hurt"] and agg["g3_no_blowup"])
        g0 = agg["gt6_gap_mean_pp"]["base"]
        agg["validity"] = "OK" if (g0 is not None and g0 <= -1.0) else "SUSPECT(base承诺超诺未现)"

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2, default=str)
    if n:
        print(
            f"[verdict] main | 不伤 {'PASS' if agg['g1_not_hurt'] else 'FAIL'} "
            f"(IC {_f(agg['ic_mean']['base'], 4)}->{_f(agg['ic_mean']['brute85'], 4)}, "
            f"top10 {_f(agg['top10_real_mean']['base'] * 100)}%->{_f(agg['top10_real_mean']['brute85'] * 100)}%, "
            f"spread {_f(agg['spread_mean_pp']['base'])}->{_f(agg['spread_mean_pp']['brute85'])}) | "
            f"不爆 {'PASS' if agg['g3_no_blowup'] else 'FAIL'} | "
            f"恢复 brute {on_brute - off_brute} 个 | "
            f"case688228 win0 {agg['case_688228_win0']} | "
            f"{'OVERALL PASS' if agg['PASS'] else 'OVERALL FAIL'} | ({time.time() - t0:.0f}s)",
            flush=True,
        )
    print(f"[worm] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
