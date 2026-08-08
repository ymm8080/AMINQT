"""_diag_tiered_topn_ab.py — MAIN 池分层 A/B/C TopN 金标准验证 (只读, 不改训练逻辑).

比较 (同一训练协议, 唯一差别 = 特征列表):
  BASE = FeatureSelector(DEFAULT_CONFIG, tier.enable=False) → 全 eligible brute 展开 (~3525 池)
  TIER = 同 selector + tier.enable=True → 只 A 层 brute 展开 (~2901 池, 当前池的严格子集)

流水线 (与生产一致): CleaningPipeline.run_train(board="main") → prepare_board_frame
(V35 全窗 build → 标签 → 停牌/近端掩码) → 裁剪到末 TRAIN_WINDOW_DAYS 交易日 →
select_features (选择 + event scope screens + brute 后注入) × 2 (BASE/TIER) →
risk_filter → OOS 边界切分 (训练严格早于 OOS, 诚实无重叠) →
model_params("main","reg"/"cls") LGBM (训练尾部 es_days 早停) →
度量: Top5/10/20 命中率+均值 (金标准), dir_acc/AUC/校准, OOS 加权 IC, OOS 子窗口稳定性.

Gate (全部满足才允许落地 tier.enable=True, 否则保持 False 并交用户定夺):
  主判据: TIER Top10 hit/mean >= BASE (允许 -0.5pp 噪声), Top5/Top20 同向不恶化
  副判据: dir_acc / AUC / 校准误差不劣化 >1pp; OOS weighted_IC 不劣化 >0.005
  稳定性: OOS 按时间三等分, TIER>=BASE (容差) 的方向须 >= 2/3 子窗稳定

产出 (WORM): data/_diag_tiered_topn_ab_<ts>.json + 控制台对比表 + 明确 GATE 结论.

用法: python scripts/_diag_tiered_topn_ab.py [--window-days 380] [--oos-days 120]
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from config.settings import PANEL_V3_PATH  # noqa: E402
from app.pipeline1.cleaning_pipeline import CleaningPipeline  # noqa: E402
from app.pipeline1.dual_track_trainer import model_params, risk_filter  # noqa: E402
from app.pipeline1.feature_engine_v35 import FeatureEngineV35  # noqa: E402
from app.pipeline1.feature_selector import FeatureSelector  # noqa: E402
from app.pipeline1.ic_screener import ICScreener  # noqa: E402
from app.pipeline1.train_runner import prepare_board_frame, select_features  # noqa: E402

logging.disable(logging.CRITICAL)

ES_DAYS = 40  # 训练尾部早停验证段 (与生产 es 段一致)
PREP_DAYS = 750  # 特征构建窗 = 末 ~750 交易日 (3 年, 与生产 assemble_panel(years=3) 一致)
HORIZONS = (2, 3, 5)  # 与 LABEL_WEIGHTS 主权重对齐
TOPKS = (5, 10, 20)
W_IC = {2: 0.45, 3: 0.35, 5: 0.2}  # 归一化 2/3/5d 权重 (LABEL_WEIGHTS 主项)
HIT_TOL = 0.005  # -0.5pp 噪声容差 (命中率/均值)
WIC_TOL = 0.005  # OOS weighted_IC 劣化容差
PP_TOL = 0.01  # dir_acc/AUC/校准 1pp 劣化容差
TMP_REGISTRY = os.path.join(ROOT, "data", "_diag_tier_registry")


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    return df


def _mem(label: str) -> None:
    try:
        import psutil

        print(
            f"  [mem] {label}: RSS={psutil.Process().memory_info().rss / 1e9:.2f}GB",
            flush=True,
        )
    except Exception:
        pass


def _fit_model(rows, Xcols, kind, k, es_days=ES_DAYS):
    """单视界单类型 LGBM (训练尾部 es_days 早停), None 当样本不足."""
    y_col = f"label_pm_{k}d_net"
    data = rows.dropna(subset=[y_col])
    if len(data) < 5000:
        return None
    data = data.sort_values("date")
    es_cut = sorted(data["date"].unique())[-es_days]
    fit = data[data["date"] < es_cut]
    es = data[data["date"] >= es_cut]
    X = np.nan_to_num(fit[Xcols].to_numpy(dtype=float), nan=0.0)
    y = fit[y_col].to_numpy(dtype=float)
    X_es = np.nan_to_num(es[Xcols].to_numpy(dtype=float), nan=0.0)
    y_es = es[y_col].to_numpy(dtype=float)
    if kind == "cls":
        y = (y > 0).astype(int)
        y_es = (y_es > 0).astype(int)
        model = lgb.LGBMClassifier(**model_params("main", "cls"))
    else:
        model = lgb.LGBMRegressor(**model_params("main", "reg"))
    model.fit(
        X,
        y,
        eval_set=[(X_es, y_es)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _eval(te, Xcols, cache, tag, k):
    m = {
        "n": 0,
        "dir_acc": np.nan,
        "mae": np.nan,
        "bias": np.nan,
        "auc": np.nan,
        "calib_err": np.nan,
        "hit50": np.nan,
        "hit55": np.nan,
        "hit60": np.nan,
    }
    reg = cache[(tag, k, "reg")]
    cls = cache[(tag, k, "cls")]
    if reg is None or cls is None:
        return m
    y_col = f"label_pm_{k}d_net"
    data = te.dropna(subset=[y_col]).copy()
    if len(data) < 50:
        return m
    X = np.nan_to_num(data[Xcols].to_numpy(dtype=float), nan=0.0)
    actual = data[y_col].to_numpy(dtype=float)
    pred = reg.predict(X)
    prob = cls.predict_proba(X)[:, 1]
    m["n"] = int(len(actual))
    m["dir_acc"] = float(np.mean(np.sign(pred) == np.sign(actual)))
    m["mae"] = float(np.mean(np.abs(pred - actual)))
    m["bias"] = float(np.mean(pred - actual))
    up = actual > 0
    if up.sum() >= 10 and (~up).sum() >= 10:
        try:
            m["auc"] = float(roc_auc_score(up, prob))
        except ValueError:
            pass
    dfp = pd.DataFrame({"p": prob, "y": up}).dropna()
    if len(dfp) >= 200:
        try:
            dfp["b"] = pd.qcut(dfp["p"], 10, duplicates="drop")
            g = dfp.groupby("b", observed=True).agg(
                mean_p=("p", "mean"), rate=("y", "mean")
            )
            m["calib_err"] = float(np.mean((g["rate"] - g["mean_p"]).abs()))
        except Exception:
            pass
    for t, key in ((0.5, "hit50"), (0.55, "hit55"), (0.60, "hit60")):
        sub = actual[prob >= t]
        if len(sub) >= 30:
            m[key] = float(np.mean(sub > 0))
    return m


def _topn_eval(te, Xcols, cache, tag, k=5):
    """每日按 pred_ret_5d 取 TOP5/10/20 → 命中率 + 均值 (金标准口径)."""
    y_col = f"label_pm_{k}d_net"
    reg = cache[(tag, k, "reg")]
    cls = cache[(tag, k, "cls")]
    data = te.dropna(subset=[y_col]).copy()
    res = {}
    if reg is None or cls is None or len(data) < 200:
        return res
    X = np.nan_to_num(data[Xcols].to_numpy(dtype=float), nan=0.0)
    data["p_ret"] = reg.predict(X)
    data["p_prob"] = cls.predict_proba(X)[:, 1]
    res["all"] = {
        "hit": float(np.mean(data[y_col] > 0)),
        "mean": float(np.mean(data[y_col])),
    }
    for topk in TOPKS:
        hits, means = [], []
        for _d, g in data.groupby("date"):
            top = g.nlargest(topk, "p_ret")
            hits.append(float((top[y_col] > 0).mean()))
            means.append(float(top[y_col].mean()))
        res[f"top{topk}_ret"] = {
            "hit": float(np.mean(hits)),
            "mean": float(np.mean(means)),
        }
    return res


def _oos_ic(te, Xcols, cache, tag):
    """OOS 段横截面 Rank IC (日度 Spearman 时间均值), 2/3/5d 按 W_IC 加权."""
    ics = {}
    for k in HORIZONS:
        y_col = f"label_pm_{k}d_net"
        reg = cache[(tag, k, "reg")]
        sub = te.dropna(subset=[y_col]).copy()
        if reg is None or len(sub) < 30:
            ics[k] = 0.0
            continue
        sub["_pred"] = reg.predict(
            np.nan_to_num(sub[Xcols].to_numpy(dtype=float), nan=0.0)
        )
        ics[k] = ICScreener.rank_ic(
            sub.rename(columns={"_pred": "score"}), "score", y_col
        )
    tw = sum(W_IC.values())
    wic = sum(W_IC[k] * ics[k] for k in W_IC) / tw
    return ics, wic


def _subwindow_topn(te, Xcols, cache, tag, k=5):
    """OOS 按时间三等分, 每段 Top10 hit/mean (稳定性判据)."""
    dates = sorted(te["date"].unique())
    n = len(dates)
    edges = [0, n // 3, 2 * n // 3, n]
    out = {}
    for i in range(3):
        seg = te[te["date"].isin(dates[edges[i]: edges[i + 1]])]
        r = _topn_eval(seg, Xcols, cache, tag, k=k)
        out[f"seg{i + 1}"] = r.get("top10_ret", {"hit": np.nan, "mean": np.nan})
    return out


def _run_arm(df, tag, oos_start, tier_enable):
    print(f"[3/5] 选择特征 {tag} (tier.enable={tier_enable}) ...", flush=True)
    cfg = copy.deepcopy(FeatureSelector.DEFAULT_CONFIG)
    cfg["main"]["tier"]["enable"] = tier_enable
    os.makedirs(TMP_REGISTRY, exist_ok=True)
    sel = FeatureSelector(config=cfg, registry_dir=TMP_REGISTRY)
    t0 = time.time()
    picked, aug = select_features(df, "main", f"diag_{tag}", sel)
    print(f"  {tag} selected={len(picked)} ({time.time() - t0:.1f}s)", flush=True)
    keep = ["symbol", "date", "is_suspended"] + list(picked)
    keep += [f"label_pm_{k}d_net" for k in HORIZONS]
    keep = [c for c in keep if c in aug.columns]
    aug = aug[keep].copy()
    _downcast(aug)
    del df
    gc.collect()
    tr_raw = aug[aug["date"] < oos_start]
    te_raw = aug[aug["date"] >= oos_start]
    tr = risk_filter(tr_raw)
    te = risk_filter(te_raw)
    del aug, tr_raw, te_raw
    gc.collect()
    print(
        f"  {tag} train={len(tr)} ({tr['date'].nunique()}日) "
        f"test={len(te)} ({te['date'].nunique()}日)",
        flush=True,
    )
    _mem(f"arm {tag}")
    return picked, tr, te


def _fmt_pp(v):
    return f"{v * 100:+.2f}pp"


def _print_table(b, t):
    print("\n" + "=" * 96)
    print("MAIN 池分层 A/B — OOS 对比 (BASE=全 brute 展开 vs TIER=只 A brute)")
    print("=" * 96)
    print(f"{'metric':<14s} {'BASE':>12s} {'TIER':>12s} {'Δ':>10s}  {'判据'}")
    print("-" * 96)

    print(f"{'n_picked':<14s} {b['n_picked']:>12d} {t['n_picked']:>12d} {t['n_picked'] - b['n_picked']:>+10d}")
    print(f"{'oos_weighted_IC':<14s} {b['oos_wic']:>12.4f} {t['oos_wic']:>12.4f} {t['oos_wic'] - b['oos_wic']:>+10.4f}  {'Δ>=%.3f' % (-WIC_TOL)}")

    for key, lab in (("top5_ret", "TOP5 hit"), ("top10_ret", "TOP10 hit"), ("top20_ret", "TOP20 hit")):
        bh, th = b["topn5"][key]["hit"], t["topn5"][key]["hit"]
        ok = "PASS" if th >= bh - HIT_TOL else "FAIL"
        print(f"{lab:<14s} {bh * 100:>11.2f}% {th * 100:>11.2f}% {(th - bh) * 100:>+9.2f}pp  {ok}")
    for key, lab in (("top5_ret", "TOP5 mean"), ("top10_ret", "TOP10 mean"), ("top20_ret", "TOP20 mean")):
        bm, tm = b["topn5"][key]["mean"], t["topn5"][key]["mean"]
        ok = "PASS" if tm >= bm - HIT_TOL else "FAIL"
        print(f"{lab:<14s} {bm * 100:>11.3f}% {tm * 100:>11.3f}% {(tm - bm) * 100:>+9.3f}pp  {ok}")
    bm0, tm0 = b["topn5"]["all"]["mean"], t["topn5"]["all"]["mean"]
    print(f"{'ALL mean':<14s} {bm0 * 100:>11.3f}% {tm0 * 100:>11.3f}% {(tm0 - bm0) * 100:>+9.3f}pp")

    for k in HORIZONS:
        print(f"\n--- {k}d  (OOS n={b['per_horizon'][k]['n']}) ---")
        for metric, is_pct in (("dir_acc", True), ("auc", True), ("calib_err", False), ("hit50", True), ("hit55", True), ("hit60", True)):
            bv = b["per_horizon"][k].get(metric, np.nan)
            tv = t["per_horizon"][k].get(metric, np.nan)
            if np.isnan(bv) or np.isnan(tv):
                continue
            if is_pct:
                print(f"  {metric:<10s} {bv * 100:>10.2f}% {tv * 100:>10.2f}% {(tv - bv) * 100:>+8.2f}pp")
            else:
                print(f"  {metric:<10s} {bv:>12.4f} {tv:>12.4f} {(tv - bv):>+9.4f}")

    print("\n--- OOS 子窗口稳定性 (Top10 hit) ---")
    for seg in ("seg1", "seg2", "seg3"):
        bv = b["subwin"][seg]["hit"]
        tv = t["subwin"][seg]["hit"]
        if np.isnan(bv) or np.isnan(tv):
            print(f"  {seg:<6s}  n/a")
            continue
        print(
            f"  {seg:<6s} BASE={bv * 100:>7.2f}% TIER={tv * 100:>7.2f}% "
            f"Δ={(tv - bv) * 100:>+6.2f}pp  {'PASS' if tv >= bv - HIT_TOL else 'FAIL'}"
        )
    print()


def _apply_gate(b, t):
    gate = {}

    # 主判据: Top10 hit/mean (容差 -0.5pp), Top5/Top20 同向不恶化
    topn_keys = ("top5_ret", "top10_ret", "top20_ret")
    for key in topn_keys:
        bh, th = b["topn5"][key]["hit"], t["topn5"][key]["hit"]
        bm, tm = b["topn5"][key]["mean"], t["topn5"][key]["mean"]
        gate[key] = {
            "hit_delta_pp": (th - bh) * 100,
            "mean_delta_pp": (tm - bm) * 100,
            "hit_pass": bool(th >= bh - HIT_TOL),
            "mean_pass": bool(tm >= bm - HIT_TOL),
        }
    main_pass = (
        gate["top10_ret"]["hit_pass"]
        and gate["top10_ret"]["mean_pass"]
        and gate["top5_ret"]["hit_pass"]
        and gate["top5_ret"]["mean_pass"]
        and gate["top20_ret"]["hit_pass"]
        and gate["top20_ret"]["mean_pass"]
    )

    # 副判据: dir_acc/AUC/校准不劣化 >1pp; OOS weighted_IC 不劣化 >0.005
    sub_ok = True
    for k in HORIZONS:
        bm = b["per_horizon"][k]
        tm = t["per_horizon"][k]
        for metric in ("dir_acc", "auc"):
            bv, tv = bm.get(metric, np.nan), tm.get(metric, np.nan)
            if not np.isnan(bv) and not np.isnan(tv) and (tv - bv) < -PP_TOL:
                sub_ok = False
        bv, tv = bm.get("calib_err", np.nan), tm.get("calib_err", np.nan)
        if not np.isnan(bv) and not np.isnan(tv) and (tv - bv) > PP_TOL:
            sub_ok = False
    wic_ok = t["oos_wic"] >= b["oos_wic"] - WIC_TOL
    gate["sub_metrics_ok"] = bool(sub_ok)
    gate["wic_ok"] = bool(wic_ok)

    # 稳定性: TIER>=BASE (容差) 子窗 >= 2/3
    stable = 0
    for seg in ("seg1", "seg2", "seg3"):
        bv = b["subwin"][seg]["hit"]
        tv = t["subwin"][seg]["hit"]
        if not np.isnan(bv) and not np.isnan(tv) and tv >= bv - HIT_TOL:
            stable += 1
    gate["stable_windows"] = stable
    gate["stable_ok"] = bool(stable >= 2)

    gate["pass"] = bool(main_pass and wic_ok and sub_ok and gate["stable_ok"])
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description="MAIN 池分层 TopN A/B 验证")
    parser.add_argument("--window-days", type=int, default=380)
    parser.add_argument("--oos-days", type=int, default=120)
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    t_start = time.time()
    print("[1/5] 载入面板 + 清洗主板 ...", flush=True)
    panel = pd.read_parquet(PANEL_V3_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    all_dates = np.array(sorted(panel["date"].unique()))
    window_start = pd.Timestamp(all_dates[-args.window_days])
    oos_start = pd.Timestamp(all_dates[-args.oos_days])
    print(
        f"  全窗日期 {all_dates[0].date()}..{all_dates[-1].date()} | "
        f"分析窗 {window_start.date()}..{all_dates[-1].date()} | OOS 从 {oos_start.date()} 起",
        flush=True,
    )
    assert args.window_days > args.oos_days, "--window-days 必须 > --oos-days"
    assert args.window_days <= PREP_DAYS, f"--window-days 不能超过特征构建窗 {PREP_DAYS}"
    cleaner = CleaningPipeline()
    main_df, _ = cleaner.run_train(panel, board="main")
    del panel
    gc.collect()
    m_dates = np.array(sorted(main_df["date"].unique()))
    prep_start = pd.Timestamp(m_dates[-PREP_DAYS])
    main_df = main_df[main_df["date"] >= prep_start].reset_index(drop=True)
    _downcast(main_df)  # 内存: build 输入 float32 (两臂同口径, 不影响 A/B 公平)
    print(
        f"  清洗后 3y 窗 rows={len(main_df)} stocks={main_df['symbol'].nunique()} "
        f"({prep_start.date()}..)",
        flush=True,
    )
    _mem("cleaned")

    print("[2/5] prepare_board_frame (V35 3y build + 标签 + 掩码) ...", flush=True)
    df = prepare_board_frame(main_df, FeatureEngineV35())
    del main_df
    gc.collect()
    _downcast(df)
    print(f"  board_frame rows={len(df)} cols={len(df.columns)}", flush=True)
    df = df[df["date"] >= window_start].reset_index(drop=True)
    print(f"  裁剪到分析窗 rows={len(df)} ({df['date'].nunique()}日)", flush=True)
    gc.collect()
    _mem("board_frame")

    arms = {}
    for tag, tier_enable in (("BASE", False), ("TIER", True)):
        picked, tr, te = _run_arm(df, tag, oos_start, tier_enable)
        arms[tag] = {"picked": picked, "tr": tr, "te": te}

    print("[4/5] 训练 LGBM (2/3/5d × reg/cls) ...", flush=True)
    cache = {}
    for tag in ("BASE", "TIER"):
        for k in HORIZONS:
            for kind in ("reg", "cls"):
                t0 = time.time()
                cache[(tag, k, kind)] = _fit_model(arms[tag]["tr"], arms[tag]["picked"], kind, k)
                print(f"  fit {tag} {k}d_{kind} {time.time() - t0:.1f}s", flush=True)
    for tag in arms:
        del arms[tag]["tr"]
    gc.collect()

    print("[5/5] 评估 + Gate ...", flush=True)
    report = {}
    for tag in ("BASE", "TIER"):
        te = arms[tag]["te"]
        picked = arms[tag]["picked"]
        ics, wic = _oos_ic(te, picked, cache, tag)
        per_h = {k: _eval(te, picked, cache, tag, k) for k in HORIZONS}
        topn = _topn_eval(te, picked, cache, tag, k=5)
        subwin = _subwindow_topn(te, picked, cache, tag, k=5)
        report[tag] = {
            "n_picked": len(picked),
            "oos_ics": ics,
            "oos_wic": wic,
            "per_horizon": per_h,
            "topn5": topn,
            "subwin": subwin,
        }
        _mem(f"report {tag}")

    _print_table(report["BASE"], report["TIER"])
    gate = _apply_gate(report["BASE"], report["TIER"])

    print("=" * 96)
    print("GATE 判定 (全部满足才允许落地 tier.enable=True):")
    print(f"  主判据 Top5/10/20 hit+mean 不劣化:      {'PASS' if all(g['hit_pass'] and g['mean_pass'] for g in [gate['top5_ret'], gate['top10_ret'], gate['top20_ret']]) else 'FAIL'}")
    print(f"  副判据 dir_acc/AUC/校准 不劣化>1pp:     {'PASS' if gate['sub_metrics_ok'] else 'FAIL'}")
    print(f"  副判据 OOS weighted_IC 不劣化>0.005:   {'PASS' if gate['wic_ok'] else 'FAIL'}  (Δ={report['TIER']['oos_wic'] - report['BASE']['oos_wic']:+.4f})")
    print(f"  稳定性 子窗 TIER>=BASE >= 2/3:          {'PASS' if gate['stable_ok'] else 'FAIL'}  ({gate['stable_windows']}/3)")
    verdict = "PASS → 可落地 tier.enable=True" if gate["pass"] else "FAIL → 保持 tier.enable=False, 交用户定夺"
    print(f"  ==> {'=' * 4} {verdict} {'=' * 4}")
    print("=" * 96)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(ROOT, "data", f"_diag_tiered_topn_ab_{ts}.json")
    payload = {
        "created": ts,
        "window_days": args.window_days,
        "oos_days": args.oos_days,
        "oos_start": str(oos_start.date()),
        "elapsed_s": round(time.time() - t_start, 1),
        "gate": gate,
        "report": report,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n报告落盘 (WORM): {out_path}")


if __name__ == "__main__":
    main()
