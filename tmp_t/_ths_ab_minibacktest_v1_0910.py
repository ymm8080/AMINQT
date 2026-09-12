# -*- coding: utf-8 -*-
"""小型回测 v1: THS问财看涨池特征 A/B — base (现产 pin 配方) vs ths (+6 看涨特征).

背景 (09-10 用户定案):
- 16:26 cron 曾把 11 名 THS 特征强注入生产被叫停; 定案 = A/B PASS 才入池.
- A/B #1 只评看涨侧 6 特征 (ths_bull 672d 真实回填); ths_bear 个股级仅 0910 一天
  有真值、ths_bear_pool 有 2 年空洞 → 等 bear pass-2 回填后 A/B #2.
- THS 列不在 stage 帧里 → 从生产面板按 (symbol,date) join, brute 变体按生产
  _family_for_symbol 语义复刻 (pct1=(s-s1)/|s1|*100 首1值NaN; ma=rolling
  min_periods=1; inf→NaN; float32).

判据 (预设, 勿临场改):
  1. 不伤线: IC >= base-0.005 且 top10 >= base-0.5pp 且 spread >= base-0.5pp;
  2. 有益线 (主): top10 改善 >= +0.5pp 或 IC 改善 >= +0.005;
  3. 不爆线: 无单窗 top10 比 base 差 > 1.5pp;
  PASS = 三线全过 → 11 名回填 force_include (bear 5 名待 A/B #2 单独过审).

设计: 与 bkd v6 同源 — staging 帧 + 5x60 交易日 walk-forward, 每窗现训两臂
(唯一变量 = 特征列), 全真 OOS。每窗先落盘 (WORM), RAM guard, 不与重活并发.
用法: python tmp_t/_ths_ab_minibacktest_v1_0910.py [--board main|dual] [--windows 5]
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
PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
META_COLS = ["date", "symbol", "label_10d_net", "close_hfq"]
WIN = 60
ES_N = 20
THS_BASE_COLS = ["ths_bull", "ths_bull_buy_sig_n", "ths_bull_tech_n"]
FAMILY = [
    "ths_bull",
    "ths_bull_brute_pct1",
    "ths_bull_brute_ma5",
    "ths_bull_brute_ma20",
    "ths_bull_buy_sig_n_brute_ma5",
    "ths_bull_tech_n_brute_ma5",
]
ARMS = ("base", "ths")


def ths_brute(df: pd.DataFrame) -> pd.DataFrame:
    """5 个 brute 变体, 与生产 _family_for_symbol 数学逐字节一致 (ths_bull 基列由 join 提供)."""
    out = pd.DataFrame(index=df.index)
    for col, want_pct, wants in (
        ("ths_bull", True, (5, 20)),
        ("ths_bull_buy_sig_n", False, (5,)),
        ("ths_bull_tech_n", False, (5,)),
    ):
        g = df.groupby("symbol", sort=False)[col]
        s = df[col].astype(float)
        if want_pct:
            pc = g.shift(1)
            v = (s - pc) / pc.abs() * 100.0
            out[f"{col}_brute_pct1"] = v.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        for w in wants:
            out[f"{col}_brute_ma{w}"] = g.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            ).astype(np.float32)
    return out


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


def eval_window(trainer, cols_by_arm, df, k, board):
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
    for a in ARMS:
        model, _ = trainer._train_one(
            "10d_reg", {"train": tr_df, "es": es_df}, cols_by_arm[a], board
        )
        X = np.nan_to_num(
            te_df[cols_by_arm[a]].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        preds[a] = model.predict(X)
        del model, X
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
    }
    rep.update(_metrics(d, ARMS))
    t0b = rep["rank_base"]["top10_real"]
    rep["window_ths_top10_worse_than_base"] = bool(
        rep["rank_ths"]["top10_real"] < t0b - 0.015
    )
    return rep


def _f(v, nd=2):
    return "NA" if v is None else f"{v:.{nd}f}"


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
    cols_base = [c for c in prod_meta["feature_cols"] if c in stage_names]
    dropped = [c for c in prod_meta["feature_cols"] if c not in stage_names]
    if dropped:
        print(f"[stage] 现产列缺 {len(dropped)} 个 (按 stage 实际列训练两臂一致)", flush=True)
    del prod_meta
    raw_need = sorted(set(META_COLS))
    missing = [c for c in raw_need if c not in stage_names]
    ths_in_stage = [c for c in THS_BASE_COLS if c in stage_names]
    if ths_in_stage:
        print(f"[stage] stage 已含 THS 列 {ths_in_stage} (直接用, 不再 join)", flush=True)
    if missing:
        print(f"[stage] 缺列 {len(missing)}: {missing[:5]} (rc=4)", flush=True)
        return 4

    load_cols = sorted(set(cols_base + raw_need + THS_BASE_COLS))
    print(f"[load] {stage}", flush=True)
    df = pd.read_parquet(stage, columns=load_cols)

    if not ths_in_stage:
        pt = pd.read_parquet(PANEL, columns=["symbol", "date"] + THS_BASE_COLS)
        pt["symbol"] = pt["symbol"].astype(str)
        pt["date"] = pd.to_datetime(pt["date"])
        df["symbol"] = df["symbol"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        n0 = len(df)
        df = df.merge(pt.drop_duplicates(subset=["symbol", "date"]),
                      on=["symbol", "date"], how="left")
        del pt
        cov = float(df["ths_bull"].notna().mean())
        print(f"[join] THS 列覆盖 {cov:.1%} (rows {n0}->{len(df)})", flush=True)
        if cov < 0.90:
            print(f"[join] 覆盖 <90%, 疑似键错位, 放弃 (rc=5)", flush=True)
            return 5

    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    df["r5"] = df.groupby("symbol")["close_hfq"].pct_change(5)
    br = ths_brute(df)
    df = pd.concat([df, br], axis=1)
    del br
    gc.collect()
    pool_daily = df.groupby(df["date"].dt.strftime("%Y%m%d"))["ths_bull"].sum()
    print(f"[sanity] ths_bull==1 日均 {pool_daily.mean():.1f} 只 / 中位 {pool_daily.median():.0f} "
          f"/ 有池天数 {int((pool_daily > 0).sum())}", flush=True)
    cols_by_arm = {"base": list(cols_base), "ths": list(cols_base) + FAMILY}
    print(f"[load] {len(df)} rows | base {len(cols_base)} cols -> ths {len(cols_by_arm['ths'])} "
          f"({time.time() - t0:.0f}s)", flush=True)

    trainer = DualTrackTrainer(model_dir=os.path.join("models", "pipeline1"))
    reps = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"ths_ab_v1_{board}_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for k in range(args.windows):
        r = eval_window(trainer, cols_by_arm, df, k, board)
        if r is None:
            break
        reps.append(r)
        seg = " | ".join(
            f"{a} gap {_f(r['calib_gt6'][a]['gap_pp'])} top10 {_f(r[f'rank_{a}']['top10_real']*100)}% "
            f"IC {_f(r[f'rankic_{a}']['ic'], 4)}"
            for a in ARMS
        )
        print(f"[win{k}] {r['test_span'][0]}→{r['test_span'][1]} {seg} ({time.time() - t0:.0f}s)", flush=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"board": board, "n_windows": len(reps), "windows": reps}, fh,
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
        "board": board,
        "n_windows": n,
        "arms": list(ARMS),
        "family_n": len(FAMILY),
        "ic_mean": {a: mean_over((f"rankic_{a}", "ic")) for a in ARMS},
        "spread_mean_pp": {a: mean_over((f"rank_{a}", "spread_pp")) for a in ARMS},
        "top10_real_mean": {a: mean_over((f"rank_{a}", "top10_real")) for a in ARMS},
        "gt6_gap_mean_pp": {a: mean_over(("calib_gt6", a, "gap_pp")) for a in ARMS},
        "windows": reps,
    }
    if n:
        d_top10 = (agg["top10_real_mean"]["ths"] or 0) - (agg["top10_real_mean"]["base"] or 0)
        d_ic = (agg["ic_mean"]["ths"] or 0) - (agg["ic_mean"]["base"] or 0)
        agg["gain_top10_pp"] = round(d_top10 * 100, 3)
        agg["gain_ic"] = round(d_ic, 4)
        agg["g1_not_hurt"] = bool(
            agg["ic_mean"]["ths"] >= agg["ic_mean"]["base"] - 0.005
            and agg["top10_real_mean"]["ths"] >= agg["top10_real_mean"]["base"] - 0.005
            and agg["spread_mean_pp"]["ths"] >= agg["spread_mean_pp"]["base"] - 0.5
        )
        agg["g2_gain"] = bool(d_top10 * 100 >= 0.5 or d_ic >= 0.005)
        agg["g3_no_blowup"] = not any(r["window_ths_top10_worse_than_base"] for r in reps)
        agg["PASS"] = bool(agg["g1_not_hurt"] and agg["g2_gain"] and agg["g3_no_blowup"])
        g0 = agg["gt6_gap_mean_pp"]["base"]
        agg["validity"] = "OK" if (g0 is not None and g0 <= -1.0) else "SUSPECT(base承诺超诺未现)"

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2, default=str)
    if n:
        print(
            f"[verdict] {board} | 不伤 {'PASS' if agg['g1_not_hurt'] else 'FAIL'} "
            f"(IC {_f(agg['ic_mean']['base'], 4)}->{_f(agg['ic_mean']['ths'], 4)}, "
            f"top10 {_f(agg['top10_real_mean']['base']*100)}%->{_f(agg['top10_real_mean']['ths']*100)}%, "
            f"spread {_f(agg['spread_mean_pp']['base'])}->{_f(agg['spread_mean_pp']['ths'])}) | "
            f"有益 {'PASS' if agg['g2_gain'] else 'FAIL'} "
            f"(top10 {agg['gain_top10_pp']:+.2f}pp, IC {agg['gain_ic']:+.4f}) | "
            f"不爆 {'PASS' if agg['g3_no_blowup'] else 'FAIL'} | "
            f"{'OVERALL PASS' if agg['PASS'] else 'OVERALL FAIL'} | ({time.time() - t0:.0f}s)",
            flush=True,
        )
    print(f"[worm] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
