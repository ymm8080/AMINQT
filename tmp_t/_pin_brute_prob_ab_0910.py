# -*- coding: utf-8 -*-
"""pin-brute 排名头(概率头)受控 A/B — base (prob 270 列) vs prob85 (prob 270+85 brute).

背景 (main-pin-brute-drop-bug-0910 + 用户 0910 指令 "COMPARE 进排名头/进回归头
哪个增益最大 THEN REALIZE IT IN PIPELINE"):
- 产线 LEGACY_SELECTION mode="prob10_pull" (09-07 起): 交付 Top10 = 纯 prob_up_10d
  板内降序 → 截 board_top_n=10 → 回撤闸 (pull_min=-0.10, 真删不补齐)。
  回归头 (pred_ret_10d) 在此模式下不进清单 → 12:41 的 reg85 A/B (IC -0.0055 FAIL)
  对交付对象是间接证据; 概率头那一臂从未量过 — 本脚本补上。
- 概率头特征 = reg bundle feature_cols (scripts/_train_legacy_prob_head.py:109) —
  pin bug 同时掐死了两头的 brute; "只进排名头"接线点 = 概率头脚本侧扩列。
- 判据预设 (终判对象 = Top10, 用户 0910 裁决):
  g1 不伤: top10(回撤闸后) >= base-0.5pp 且 spread >= base-0.5pp 且
           Brier <= base+0.002 (概率头自身的 mfe 达标目标);
  g3 不爆: 无单窗 top10(闸后) < base-1.0pp。
  PASS = g1&g3 → 概率头脚本侧接 brute 扩列 (reg 头维持 270 零风险);
  增益对比 = 本臂数字 vs 12:41 reg85 A/B (自动读 WORM 并入 comparison 块)。

设计:
- 零 select() 调用 → 不写任何 selected_main_{ts}.json (快照劫持雷免疫)。
  base 270 列 = 隔离区 12:41 A/B base 臂快照 (生产 nan 门后的精确清单);
  brute 85 名 = selected_main_pinned.json (基列须在 stage)。
- 训练/配方 = 生产 prob_head 逐字节复用: LGB_PARAMS + _fit_with_es + mfe_3d 口径
  (_add_mfe_3d, abs_target=0.03) + predict_proba。偏差仅一处: 训练行过滤无
  label_pain 门 (两臂对称, 不改变单变量性)。
- walk-forward 5 窗与 12:41 reg A/B 同几何 (test span 相同 → 增益可比)。
- 回撤闸镜像: close/rolling(10,min_periods=2)max - 1 < pull_min → 剔 (真删不补齐)。

用法: python tmp_t/_pin_brute_prob_ab_0910.py [--windows 5]
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
QUARANTINE = os.path.join("data", "factor_registry", "ab_dryrun_quarantine_0910")
PIN_FILE = "selected_main_pinned.json"
META_COLS = ["date", "symbol", "label_10d_net", "close_hfq", "high_hfq", "amount"]
CASE_SYM, CASE_DATE = "688228", "2026-09-09"
ARMS = ("base", "prob85")
WIN = 60
BASE270_LEN = 270  # base 臂快照校验 (nan 门后生产口径)


def _find_base_snapshot() -> str:
    """隔离区里挑 base 臂快照 (len==270 且零 brute); 找不到 rc=4 大声退出."""
    best = None
    for fn in sorted(os.listdir(QUARANTINE)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(QUARANTINE, fn), encoding="utf-8") as fh:
            feats = json.load(fh).get("features") or []
        n_brute = sum(1 for f in feats if "_brute_" in f)
        if len(feats) == BASE270_LEN and n_brute == 0:
            best = os.path.join(QUARANTINE, fn)
            break
    if best is None:
        print(f"[snapshot] 隔离区无 {BASE270_LEN} 列零 brute 快照 (rc=4)", flush=True)
        raise SystemExit(4)
    return best


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

    from lightgbm import LGBMClassifier

    from app.pipeline1.feature_selector import FeatureSelector, inject_missing_brute
    from app.pipeline1.prob_head import LGB_PARAMS, _add_mfe_3d, _fit_with_es
    from config.settings import LEGACY_PROB_GATE, LEGACY_SELECTION

    abs_target = float(LEGACY_PROB_GATE["abs_target"])
    pull_min = float(LEGACY_SELECTION["pull_min"])
    top_n = int(LEGACY_SELECTION["board_top_n"])

    # 列名解析: base270 (隔离快照) + brute85 (pin 快照, 基列须在 stage)
    st_names = set(pq.ParquetFile(STAGE).schema_arrow.names)
    with open(_find_base_snapshot(), encoding="utf-8") as fh:
        base_cols = json.load(fh).get("features") or []
    sel = FeatureSelector()
    with open(os.path.join(sel.registry_dir, PIN_FILE), encoding="utf-8") as fh:
        pin = json.load(fh).get("features", [])
    brute_names = [f for f in pin if "_brute_" in f]
    bases = sorted({f.split("_brute_")[0] for f in brute_names})
    base_miss = [b for b in bases if b not in st_names]
    if base_miss:
        print(f"[stage] brute 基列缺 {len(base_miss)}: {base_miss[:5]} (rc=4)", flush=True)
        return 4
    cols_by_arm = {"base": list(base_cols), "prob85": base_cols + brute_names}
    miss = [c for c in base_cols if c not in st_names]
    if miss:
        print(f"[stage] base270 缸缺 {len(miss)}: {miss[:5]} (rc=4)", flush=True)
        return 4
    miss_meta = [c for c in META_COLS if c not in st_names]
    if miss_meta:
        print(f"[stage] meta 缺列 {miss_meta} (rc=4)", flush=True)
        return 4

    need = sorted(set(base_cols) | set(bases) | set(META_COLS) | {"adv20"})
    print(f"[load] {STAGE} (need {len(need)} cols)", flush=True)
    df = pd.read_parquet(STAGE, columns=need)
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    if "adv20" not in df.columns or df["adv20"].isna().all():
        df["adv20"] = (
            df.groupby("symbol")["amount"].rolling(20, min_periods=20).mean()
            .reset_index(level=0, drop=True)
        )
    raw = df[["symbol", "date", "close_hfq", "high_hfq", "adv20"]].copy()
    raw["symbol"] = raw["symbol"].astype(str)
    df["mfe_3d"] = _add_mfe_3d(raw)["mfe_3d"].to_numpy()
    del raw
    # 回撤闸镜像 (逐行, 生产同公式: close / 10日滚动高 - 1)
    g = df.groupby("symbol", sort=False)["close_hfq"]
    df["pull"] = df["close_hfq"] / g.rolling(10, min_periods=2).max().reset_index(
        level=0, drop=True
    ) - 1.0
    gc.collect()

    still = inject_missing_brute(df, cols_by_arm["prob85"])
    if still:
        print(f"[inject] 仍缺 {len(still)}: {still[:5]} (rc=6)", flush=True)
        return 6
    gc.collect()
    print(
        f"[inject] brute 全物化: base {len(base_cols)} 列 | prob85 {len(cols_by_arm['prob85'])} 列 "
        f"(brute {len(brute_names)}) ({time.time() - t0:.0f}s)",
        flush=True,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    from config.settings import data_others_path

    out = data_others_path("diag") / f"pin_brute_prob_ab_main_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    dates = np.array(sorted(pd.to_datetime(df["date"]).unique()))
    ok_all = df["mfe_3d"].notna().to_numpy()
    reps = []
    for k in range(args.windows):
        end = len(dates) - k * WIN
        if end - WIN < 60:
            break
        test_d = set(dates[end - WIN : end])
        train_d = set(dates[: end - WIN])
        tr_df = df[df["date"].isin(train_d)]
        te_df = df[df["date"].isin(test_d)].copy()
        case_mask = (te_df["symbol"] == CASE_SYM) & (te_df["date"] == CASE_DATE)
        preds = {}
        case_preds = {}
        for a in ARMS:
            cols = cols_by_arm[a]
            ok = tr_df["mfe_3d"].notna().to_numpy()
            x = tr_df.loc[ok, cols].to_numpy(dtype="float32")
            y = (tr_df.loc[ok, "mfe_3d"] >= abs_target).astype(float)
            model = LGBMClassifier(**LGB_PARAMS)
            model = _fit_with_es(
                "main", model, x, y.to_numpy(), tr_df.loc[ok, "date"].to_numpy()
            )
            xt = te_df[cols].to_numpy(dtype="float32")
            preds[a] = model.predict_proba(xt)[:, 1]
            if case_mask.any():
                ci = int(np.flatnonzero(case_mask.to_numpy())[0])
                case_preds[a] = float(preds[a][ci])
            del model, x, xt, ok
            gc.collect()
        del tr_df
        gc.collect()

        d = pd.DataFrame(
            {
                "date": te_df["date"].values,
                "symbol": te_df["symbol"].astype(str).values,
                "real": te_df["label_10d_net"].values,
                "mfe": te_df["mfe_3d"].values,
                "pull": te_df["pull"].values,
                **preds,
            }
        )
        del te_df
        gc.collect()
        d = d.dropna(subset=["real"])
        y_real = (d["mfe"] >= abs_target).astype(float)
        mkt_r5 = d.groupby("date")["real"].mean()
        weak_days = set(mkt_r5[mkt_r5 < 0].index)

        rep = {
            "window": k,
            "test_span": [str(pd.Timestamp(d["date"].min()).date()),
                          str(pd.Timestamp(d["date"].max()).date())],
            "n_eval": int(len(d)),
            "case_688228_prob": case_preds,
        }
        for a in ARMS:
            p = d[a].to_numpy(dtype=float)
            rk = d.groupby("date")[a].rank(ascending=False, method="first")
            top = d[rk <= top_n]
            mid = d[(rk > top_n) & (rk <= top_n * 3)]
            # 终判对象: prob10_pull 全镜像 — top10 截取后过回撤闸 (真删不补齐)
            kept = top[top["pull"] >= pull_min]
            weak = top[top["date"].isin(weak_days)]
            ic, n_ic = _rank_ic(p, d["real"].to_numpy(dtype=float), d["date"].to_numpy())
            rep[f"obj_{a}"] = {
                "top10_real": float(top["real"].mean()),
                "top10_pull_real": float(kept["real"].mean()),
                "n_after_pull": int(len(kept)),
                "spread_pp": float((top["real"].mean() - mid["real"].mean()) * 100),
                "weak_top10_real": float(weak["real"].mean()) if len(weak) else None,
                "rankic": ic,
                "brier": float(np.mean((p - y_real.to_numpy(dtype=float)) ** 2)),
                "prob_mean": float(np.mean(p)),
                "mfe_hit_rate": float(y_real.mean()),
            }
        rep["window_prob85_top10_worse_than_base"] = bool(
            rep["obj_prob85"]["top10_pull_real"]
            < rep["obj_base"]["top10_pull_real"] - 0.01
        )
        reps.append(rep)
        seg = " | ".join(
            f"{a} top10闸后 {_f(rep[f'obj_{a}']['top10_pull_real'] * 100)}% "
            f"spread {_f(rep[f'obj_{a}']['spread_pp'])} IC {_f(rep[f'obj_{a}']['rankic'], 4)} "
            f"Brier {_f(rep[f'obj_{a}']['brier'], 4)}"
            for a in ARMS
        )
        print(f"[win{k}] {rep['test_span'][0]}→{rep['test_span'][1]} {seg} ({time.time() - t0:.0f}s)", flush=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"board": "main_prob", "n_windows": len(reps), "windows": reps},
                      fh, ensure_ascii=False, indent=2, default=str)
        del d, y_real
        gc.collect()

    n = len(reps)

    def mean_over(path):
        vals = []
        for r in reps:
            v = r
            for kk in path:
                v = v[kk]
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    agg = {
        "board": "main_prob",
        "n_windows": n,
        "arms": list(ARMS),
        "base_cols": len(base_cols),
        "prob85_cols": len(cols_by_arm["prob85"]),
        "restored_brute_n": len(brute_names),
        "top10_pull_real_mean": {a: mean_over((f"obj_{a}", "top10_pull_real")) for a in ARMS},
        "top10_real_mean": {a: mean_over((f"obj_{a}", "top10_real")) for a in ARMS},
        "spread_mean_pp": {a: mean_over((f"obj_{a}", "spread_pp")) for a in ARMS},
        "rankic_mean": {a: mean_over((f"obj_{a}", "rankic")) for a in ARMS},
        "brier_mean": {a: mean_over((f"obj_{a}", "brier")) for a in ARMS},
        "windows": reps,
    }
    if n:
        agg["case_688228_win0"] = reps[0].get("case_688228_prob") or {}
        agg["g1_not_hurt"] = bool(
            agg["top10_pull_real_mean"]["prob85"] >= agg["top10_pull_real_mean"]["base"] - 0.005
            and agg["spread_mean_pp"]["prob85"] >= agg["spread_mean_pp"]["base"] - 0.5
            and agg["brier_mean"]["prob85"] <= agg["brier_mean"]["base"] + 0.002
        )
        agg["g3_no_blowup"] = not any(r["window_prob85_top10_worse_than_base"] for r in reps)
        agg["PASS"] = bool(agg["g1_not_hurt"] and agg["g3_no_blowup"])
        reg_worm = data_others_path("diag") / "pin_brute_ab_main_20260910_124139.json"
        if reg_worm.exists():
            with open(reg_worm, encoding="utf-8") as fh:
                reg = json.load(fh)
            agg["comparison_vs_reg85"] = {
                "reg85_ic_mean": (reg.get("ic_mean") or {}).get("brute85"),
                "reg85_top10_mean": (reg.get("top10_real_mean") or {}).get("brute85"),
                "reg85_spread_pp": (reg.get("spread_mean_pp") or {}).get("brute85"),
                "reg85_gt6_gap_pp": (reg.get("gt6_gap_mean_pp") or {}).get("brute85"),
                "note": "reg85 = 进回归头臂 (12:41 A/B, 排名键当时=reg pred); "
                        "prob85 = 进排名头臂 (本 A/B, 排名键=prob_up_10d 生产现役)",
            }
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(agg, fh, ensure_ascii=False, indent=2, default=str)
        print(
            f"[verdict] main_prob | 不伤 {'PASS' if agg['g1_not_hurt'] else 'FAIL'} "
            f"(top10闸后 {_f(agg['top10_pull_real_mean']['base'] * 100)}%->"
            f"{_f(agg['top10_pull_real_mean']['prob85'] * 100)}%, "
            f"spread {_f(agg['spread_mean_pp']['base'])}->{_f(agg['spread_mean_pp']['prob85'])}, "
            f"Brier {_f(agg['brier_mean']['base'], 4)}->{_f(agg['brier_mean']['prob85'], 4)}, "
            f"IC {_f(agg['rankic_mean']['base'], 4)}->{_f(agg['rankic_mean']['prob85'], 4)}) | "
            f"不爆 {'PASS' if agg['g3_no_blowup'] else 'FAIL'} | "
            f"{'OVERALL PASS' if agg['PASS'] else 'OVERALL FAIL'} | ({time.time() - t0:.0f}s)",
            flush=True,
        )
    print(f"[worm] {out}", flush=True)
    return 0


def _f(v, nd=2):
    return "NA" if v is None else f"{v:.{nd}f}"


if __name__ == "__main__":
    raise SystemExit(main())
