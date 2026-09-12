# -*- coding: utf-8 -*-
"""小型回测 v6: 双向短期特征族 A/B — base (mag60 现产配方) vs bkd (+14 特征).

背景 (bkd 计划 Phase 3, 688228 案):
- v4/v5 终判后仅剩"换特征结构"路线; IC 筛选 (bkd_up_icscreen_{board}_*.json) 保留
  14 个双向短期特征 (bkd_ 破位 x8 + up_ 上涨 x6, bkd_dnshrink5 IC 死淘汰,
  bkd_pos_range20 共线淘汰)。
- 判据 (预设, 勿临场改):
  1. 不伤线: IC >= base-0.005 且 top10 >= base-0.5pp 且 spread >= base-0.5pp;
  2. 治病线 (主): 弱市日 gt6 缺口改善 >= 1.5pp 且 下杀切片 (r5<0 的 top10 行)
     承诺缺口改善 >= 1pp;
  3. 不爆线: 无单窗整体 gt6 缺口比 base 差 > 1pp;
  4. 个案线 (报数非硬闸): win0 模型对 688228@09-09 重打分, +5.27% 承诺应收缩。

设计: 与 v4 同源 — staging 帧 + 5x60 交易日 walk-forward, 每窗现训两臂 (唯一变量 =
特征列), 全真 OOS。每窗先落盘 (WORM), RAM guard, 不与重活并发。
用法: python tmp_t/_bkd_ab_minibacktest_v6_0910.py [--board main|dual] [--windows 5]
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
FAMILY = [
    "bkd_dn_streak", "bkd_dn_days5", "bkd_dd5_high20", "bkd_dd_high60",
    "bkd_min10_dist", "bkd_below_ma_cnt", "bkd_ma_bear_align", "bkd_ma5_slope5",
    "up_vol_confirm5", "up_body5", "up_break20_vol", "up_followthrough",
    "up_pullback_depth", "up_gap_hold",
]
CASE_SYM, CASE_DATE = "688228", "2026-09-09"
ARMS = ("base", "bkd")


def g_apply_rolling(series: pd.Series, symbol: pd.Series, win: int, how: str) -> pd.Series:
    return series.groupby(symbol, sort=False).transform(lambda s: getattr(s.rolling(win, min_periods=win), how)())


def bkd_up_features(df: pd.DataFrame) -> pd.DataFrame:
    """与 _bkd_up_ic_screen_0910.py 同一实现 (保留族), 输入须按 symbol,date 排序。"""
    g = df.groupby("symbol", sort=False)
    c = df["close_hfq"]
    pc = g["close_hfq"].shift(1)
    dn = (c < pc).astype("float64")
    up = (c > pc).astype("float64")
    ma5 = g["close_hfq"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma10 = g["close_hfq"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    ma20 = g["close_hfq"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    hi20 = g["high_hfq"].transform(lambda s: s.rolling(20, min_periods=20).max())
    hi20p = g["high_hfq"].shift(1).groupby(df["symbol"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=20).max()
    )
    hi5 = g["close_hfq"].transform(lambda s: s.rolling(5, min_periods=5).max())
    lo10 = g["low_hfq"].transform(lambda s: s.rolling(10, min_periods=10).min())
    hi60 = g["high_hfq"].transform(lambda s: s.rolling(60, min_periods=60).max())
    v20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())

    f = pd.DataFrame(index=df.index)
    dn_prev = dn.groupby(df["symbol"], sort=False).shift(1)
    blk = (dn != dn_prev).groupby(df["symbol"], sort=False).cumsum()
    streak = dn.groupby([df["symbol"], blk], sort=False).cumsum()
    f["bkd_dn_streak"] = streak.clip(upper=10) * dn
    f["bkd_dn_days5"] = g_apply_rolling(dn, df["symbol"], 5, "sum")
    f["bkd_dd5_high20"] = c / hi20 - 1.0
    f["bkd_dd_high60"] = c / hi60 - 1.0
    f["bkd_min10_dist"] = c / lo10 - 1.0
    f["bkd_below_ma_cnt"] = (
        (c < ma5).astype("float64") + (c < ma10).astype("float64") + (c < ma20).astype("float64")
    )
    f["bkd_ma_bear_align"] = (
        (ma5 < ma10).astype("float64") + (ma10 < ma20).astype("float64") + (c < ma5).astype("float64")
    ) / 3.0
    f["bkd_ma5_slope5"] = ma5.groupby(df["symbol"], sort=False).pct_change(5, fill_method=None)
    up5 = g_apply_rolling(up, df["symbol"], 5, "sum")
    upvol5 = g_apply_rolling(up * df["volume"], df["symbol"], 5, "sum")
    f["up_vol_confirm5"] = (upvol5 / up5.replace(0.0, np.nan)) / v20
    body = (df["close_hfq"] - df["open_hfq"]) / pc
    f["up_body5"] = g_apply_rolling(body, df["symbol"], 5, "mean")
    vr = df["volume"] / v20
    f["up_break20_vol"] = (c / hi20p - 1.0) * vr
    gap = df["open_hfq"] / pc - 1.0
    up_prev = (pc > g["close_hfq"].shift(2)).astype("float64")
    f["up_followthrough"] = g_apply_rolling((gap * up_prev).fillna(0.0), df["symbol"], 5, "mean")
    f["up_pullback_depth"] = c / hi5 - 1.0
    upgap_hold = (df["low_hfq"] > pc).astype("float64")
    f["up_gap_hold"] = g_apply_rolling(upgap_hold, df["symbol"], 10, "mean")
    return f[FAMILY]


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
        # 下杀切片: r5<0 且该臂 top10 入选行的承诺缺口 (跌着还被选中的票, 诺 vs 实)
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
    case_preds = {}
    case_mask = (te_df["symbol"] == CASE_SYM) & (te_df["date"] == CASE_DATE)
    for a in ARMS:
        model, _ = trainer._train_one(
            "10d_reg", {"train": tr_df, "es": es_df}, cols_by_arm[a], board
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
    rep["window_bkd_gt6_worse_than_base"] = bool(
        g6b is not None and rep["calib_gt6"]["bkd"]["gap_pp"] is not None
        and rep["calib_gt6"]["bkd"]["gap_pp"] < g6b - 1.0
    )
    return rep


def arms_cols(cols_by_arm):
    return cols_by_arm


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
    raw_need = sorted(set(META_COLS + ["open_hfq", "high_hfq", "low_hfq", "volume"]))
    missing = [c for c in cols_base + raw_need if c not in stage_names]
    if missing:
        print(f"[stage] 缺列 {len(missing)}: {missing[:5]} (rc=4)", flush=True)
        return 4

    print(f"[load] {stage}", flush=True)
    df = pd.read_parquet(stage, columns=sorted(set(cols_base + raw_need)))
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    df["r5"] = df.groupby("symbol")["close_hfq"].pct_change(5)
    fam = bkd_up_features(df)
    df = pd.concat([df, fam], axis=1)
    del fam
    gc.collect()
    # 连跌断言 (同筛选脚本): 算法回归防线
    cm = (df["symbol"] == CASE_SYM) & (df["date"] == CASE_DATE)
    if cm.any():
        ci = int(np.flatnonzero(cm.to_numpy())[0])
        assert float(df["bkd_dn_streak"].iloc[ci]) == 4.0, "连跌断言失败 (应为 4)"
        print(f"[assert] {CASE_SYM}@{CASE_DATE} streak=4 OK", flush=True)
    cols_by_arm = {"base": list(cols_base), "bkd": list(cols_base) + FAMILY}
    print(f"[load] {len(df)} rows | base {len(cols_base)} cols -> bkd {len(cols_by_arm['bkd'])} "
          f"({time.time() - t0:.0f}s)", flush=True)

    trainer = DualTrackTrainer(model_dir=os.path.join("models", "pipeline1"))
    reps = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"bkd_ab_v6_{board}_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for k in range(args.windows):
        r = eval_window(trainer, cols_by_arm, df, k, board)
        if r is None:
            break
        reps.append(r)
        seg = " | ".join(
            f"{a} gap {_f(r['calib_gt6'][a]['gap_pp'])} weak {_f(r['calib_gt6'][a]['weak_gap_pp'])} "
            f"top10 {_f(r[f'rank_{a}']['top10_real']*100)}% IC {_f(r[f'rankic_{a}']['ic'], 4)}"
            for a in ARMS
        )
        cp = r.get("case_688228_pred") or {}
        cp_s = f" case688228 {cp.get('base', float('nan')):+.4f}->{cp.get('bkd', float('nan')):+.4f}" if cp else ""
        print(f"[win{k}] {r['test_span'][0]}→{r['test_span'][1]} {seg}{cp_s} ({time.time() - t0:.0f}s)", flush=True)
        # 每窗先落盘 (聚合步崩溃不丢窗口数据)
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
        "gt6_weak_gap_mean_pp": {a: mean_over(("calib_gt6", a, "weak_gap_pp")) for a in ARMS},
        "downslice_gap_mean_pp": {a: mean_over((f"downslice_{a}", "gap_pp")) for a in ARMS},
        "windows": reps,
    }
    if n:
        agg["case_688228_win0"] = reps[0].get("case_688228_pred") or {}
        imp_weak = (agg["gt6_weak_gap_mean_pp"]["bkd"] or 0) - (agg["gt6_weak_gap_mean_pp"]["base"] or 0)
        imp_dn = (agg["downslice_gap_mean_pp"]["bkd"] or 0) - (agg["downslice_gap_mean_pp"]["base"] or 0)
        agg["cure_weak_improvement_pp"] = round(imp_weak, 3)
        agg["cure_downslice_improvement_pp"] = round(imp_dn, 3)
        agg["g1_not_hurt"] = bool(
            agg["ic_mean"]["bkd"] >= agg["ic_mean"]["base"] - 0.005
            and agg["top10_real_mean"]["bkd"] >= agg["top10_real_mean"]["base"] - 0.005
            and agg["spread_mean_pp"]["bkd"] >= agg["spread_mean_pp"]["base"] - 0.5
        )
        agg["g2_cure"] = bool(imp_weak >= 1.5 and imp_dn >= 1.0)
        agg["g3_no_blowup"] = not any(r["window_bkd_gt6_worse_than_base"] for r in reps)
        agg["PASS"] = bool(agg["g1_not_hurt"] and agg["g2_cure"] and agg["g3_no_blowup"])
        g0 = agg["gt6_gap_mean_pp"]["base"]
        agg["validity"] = "OK" if (g0 is not None and g0 <= -1.0) else "SUSPECT(base承诺超诺未现)"

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2, default=str)
    if n:
        print(
            f"[verdict] {board} | 不伤 {'PASS' if agg['g1_not_hurt'] else 'FAIL'} "
            f"(IC {_f(agg['ic_mean']['base'], 4)}->{_f(agg['ic_mean']['bkd'], 4)}, "
            f"top10 {_f(agg['top10_real_mean']['base']*100)}%->{_f(agg['top10_real_mean']['bkd']*100)}%, "
            f"spread {_f(agg['spread_mean_pp']['base'])}->{_f(agg['spread_mean_pp']['bkd'])}) | "
            f"治病 {'PASS' if agg['g2_cure'] else 'FAIL'} "
            f"(弱市gt6缺口 {_f(agg['gt6_weak_gap_mean_pp']['base'])}->{_f(agg['gt6_weak_gap_mean_pp']['bkd'])}, "
            f"改善 {imp_weak:+.2f}; 下杀切片缺口 {_f(agg['downslice_gap_mean_pp']['base'])}->"
            f"{_f(agg['downslice_gap_mean_pp']['bkd'])}, 改善 {imp_dn:+.2f}) | "
            f"不爆 {'PASS' if agg['g3_no_blowup'] else 'FAIL'} | "
            f"case688228 win0 {agg['case_688228_win0']} | "
            f"{'OVERALL PASS' if agg['PASS'] else 'OVERALL FAIL'} | ({time.time() - t0:.0f}s)",
            flush=True,
        )
    print(f"[worm] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
