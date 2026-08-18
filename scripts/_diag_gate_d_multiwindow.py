"""gate_d 多窗口消融降噪扫描 (特征集季度重选协议 Phase 1, 2026-08-18 定案).

背景: 单窗 ICIR 曲线是平坦噪声 (n=50 在 08-16 是峰 0.3382 / 08-17 是谷 0.2048,
面板仅差 1-3 天), gate_d 每周重选 = 掷骰子 (记录 30/200/325/50/50/200,
大签 ~50% 需人工回退). 特征集已冻结 pin (selected_dual_pinned.json, 08-16 的
50 特征). 本脚本回答: 有没有比 pin 50 更稳的集合 / pin 是否在噪声底内.

方法: 对末 ~window_days 交易日每 ~window_step 日取一个窗端点 W, 每次把面板
切到 date<=W 后原样跑 gate_d_ablation (生产同参数: deterministic LGBM,
300 树全模型 + 200 树消融, 内部 80/20 切分, min_feats=30, sat_pct=0.95,
label=label_pm_1d_net, random_state=42), 记录每窗 ICIR 曲线/选中集/gain 排名.
聚合:
  - 噪声底: best_ir / sat_n / n_selected 跨窗 min/max/mean/std
  - 平均曲线: 各 n 点跨窗平均 ICIR → 饱和点 = 候选A 个数, 特征 = 跨窗平均
    gain 排名 Top 候选A
  - 稳定核心: 跨窗选中频率 Top-50 = 候选B
  - pin 对照: 同一切分下 pin 集 ICIR vs 每窗实际选中集 (win rate / mean delta)

面板 = 生产同源 (load_panel_v3 + 3y 窗 + CleaningPipeline 清洗 +
prepare_board_frame 双创截面排名 + registry 门控), 特征池 = nan_filter(0.95)
同 _run_gate_d. 已知偏差: 数据质量回填 (07-27/28/cyq/LHB 等) 使早窗数据略优于
当时真实快照, 协议按当前面板切片. 事件池特征 (EVENT_SCOPE_SCREENS) 生产侧在
任何选中集之上等量附加, 正交, 本扫描不含.

用法: python scripts/_diag_gate_d_multiwindow.py [--window-days 300]
                                           [--window-step 10] [--pin PATH]
WORM: <DATA OTHERS>/diag/gate_d_multiwindow_<ts>/ (json + freq csv)
耗时: 每窗 ~10 个 LGBM fit (1×300 + ~9×200) + 1×200 pin 对照; 31 窗 ≈ 3-6h,
建议后台跑, 期间勿跑重训 (RAM 独占闸).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import lightgbm as lgb
from scipy.stats import spearmanr

from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_selector import (
    FeatureSelector,
    gate_d_ablation,
    nan_filter,
)
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import PANEL_V3_PATH, data_others_path

logger = logging.getLogger(__name__)

PIN_DEFAULT = "selected_dual_pinned.json"
CAND_B_TOP_N = 50


def _eval_icir(te, preds, label_col):
    """与 gate_d_ablation 内部 _eval_icir 同口径: 日截面 Spearman IC 均值/标准差."""
    df_e = te.copy()
    df_e["pred"] = preds
    ics = [
        spearmanr(g["pred"], g[label_col])[0]
        for _, g in df_e.groupby("date")
        if len(g) >= 10
    ]
    a = np.array([x for x in ics if not np.isnan(x)])
    if len(a) > 1:
        sd = a.std()
        if sd > 0:
            return float(round(a.mean() / sd, 4))
    return 0.0


def _lgbm_params(n_estimators):
    """镜像 gate_d_ablation.base_params (生产同参, deterministic 固定种子)."""
    return dict(
        n_estimators=n_estimators,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        deterministic=True,
        force_col_wise=True,
        bagging_seed=43,
        feature_fraction_seed=44,
        drop_seed=45,
        n_jobs=-1,
        verbose=-1,
    )


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    return float(len(sa & sb) / len(sa | sb)) if sa and sb else 0.0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    ap = argparse.ArgumentParser(description="gate_d 多窗口消融扫描 (Phase 1)")
    ap.add_argument("--window-days", type=int, default=300, help="扫描覆盖末 N 交易日")
    ap.add_argument("--window-step", type=int, default=10, help="窗端点间隔交易日")
    ap.add_argument(
        "--min-window-days", type=int, default=120, help="窗内最少交易日, 不足跳过"
    )
    ap.add_argument(
        "--pin",
        default=None,
        help=f"pin 文件路径 (默认 registry 下 {PIN_DEFAULT})",
    )
    args = ap.parse_args()
    if args.window_step < 1 or args.window_days < 1:
        raise SystemExit("--window-days / --window-step 必须 >= 1")

    # ── 生产同源面板 (run_training 同路径: V3 直读 + 3y 窗 + 清洗 + 特征 + 标签) ──
    t0 = time.time()
    print(f"[load] 读面板 {PANEL_V3_PATH} ...", flush=True)
    panel = load_panel_v3(path=PANEL_V3_PATH)
    cut = panel["date"].max() - pd.DateOffset(years=3)
    panel = panel[panel["date"] >= cut]
    print(
        f"[load] 3y 窗: {len(panel)} rows max={panel['date'].max().date()}", flush=True
    )
    board_df = CleaningPipeline().run_train(panel, board="dual")[1]
    del panel
    gc.collect()
    print(f"[load] dual 清洗后: {len(board_df)} rows", flush=True)
    registry = FeatureRegistry()
    df = prepare_board_frame(
        board_df,
        FeatureEngineV35(),
        None,
        cross_sectional_rank=True,
        registry=registry,
    )
    del board_df
    gc.collect()

    # ── 特征池 (同 _run_gate_d: feature_columns → nan_filter) ──
    gcfg = FeatureSelector.DEFAULT_CONFIG["dual"]["gate_d"]
    label_col = gcfg.get("label", "label_pm_1d_net")
    if label_col not in df.columns:
        label_col = "label_1d_net"
    feats = nan_filter(
        FeatureEngineV35.feature_columns(df),
        df,
        FeatureSelector.DEFAULT_CONFIG["dual"].get("nan_threshold", 0.95),
    )
    df = df[["date", label_col] + feats].copy()
    gc.collect()
    print(
        f"[pool] {len(df)} rows × {len(feats)} features, label={label_col}, "
        f"用时 {time.time() - t0:.0f}s",
        flush=True,
    )

    # ── pin ──
    pin_path = args.pin or str(data_others_path("data/factor_registry") / PIN_DEFAULT)
    if not os.path.exists(pin_path):
        raise SystemExit(
            f"pin 文件不存在: {pin_path} (Phase 1 的对照基准缺失, 拒绝扫描)"
        )
    with open(pin_path, encoding="utf-8") as fh:
        pin_feats = [f for f in json.load(fh).get("features", []) if f in feats]
    print(f"[pin] {os.path.basename(pin_path)}: {len(pin_feats)}/50 在池内", flush=True)

    # ── 窗端点: 末 window_days 交易日, 每 window_step 一个, 含最新 ──
    dates = np.sort(df["date"].unique())
    n_dates = len(dates)
    n_wins = (args.window_days + args.window_step - 1) // args.window_step
    endpoints = [
        pd.Timestamp(dates[n_dates - 1 - k * args.window_step])
        for k in range(n_wins)
        if n_dates - 1 - k * args.window_step >= 0
    ]
    if len(endpoints) < 2:
        raise SystemExit(
            "窗端点 < 2, 无法聚合 (调大 --window-days 或调小 --window-step)"
        )
    print(
        f"[windows] {len(endpoints)} 端点: {endpoints[-1].date()} .. {endpoints[0].date()}",
        flush=True,
    )

    results = []
    for i, w_end in enumerate(endpoints):
        w_t0 = time.time()
        df_w = df[df["date"] <= w_end]
        n_dates = int(df_w["date"].nunique())
        if n_dates < args.min_window_days:
            logger.warning(
                "跳过窗 %s (%d 交易日 < %d)",
                w_end.date(),
                n_dates,
                args.min_window_days,
            )
            continue
        metrics: dict = {}
        selected = gate_d_ablation(
            feats,
            df_w,
            label_col=label_col,
            min_feats=gcfg.get("min_features", 30),
            sat_pct=gcfg.get("saturation_pct", 0.95),
            metrics_out=metrics,
        )
        sat_icir = next(
            (
                e["icir"]
                for e in metrics.get("ablation_log", [])
                if e["n"] == metrics.get("sat_n")
            ),
            None,
        )
        rec: dict = {
            "window_end": w_end.strftime("%Y-%m-%d"),
            "n_dates": n_dates,
            "n_candidates": metrics.get("n_candidates"),
            "best_ir": metrics.get("best_ir"),
            "best_n": metrics.get("best_n"),
            "sat_n": metrics.get("sat_n"),
            "sat_icir": sat_icir,
            "n_selected": metrics.get("n_selected"),
            "selected": selected,
            "gain_rank": metrics.get("gain_rank", []),
            "ablation_log": metrics.get("ablation_log", []),
            "pin_icir": None,
            "pin_n_avail": None,
        }
        # ── pin 对照: 同 80/20 切分, 200 树, 与消融模型同参 ──
        if pin_feats:
            rec["pin_n_avail"] = len(pin_feats)
            if len(pin_feats) >= 10:
                w_dates = np.sort(df_w["date"].unique())
                split = int(len(w_dates) * 0.8)
                tr = df_w[df_w["date"].isin(w_dates[:split])].dropna(subset=[label_col])
                te = df_w[df_w["date"].isin(w_dates[split:])].dropna(subset=[label_col])
                m = lgb.LGBMRegressor(**_lgbm_params(200))
                m.fit(tr[pin_feats], tr[label_col])
                rec["pin_icir"] = _eval_icir(te, m.predict(te[pin_feats]), label_col)
                del m
        results.append(rec)
        n_done = len(results)
        per_w = (time.time() - w_t0) / n_done
        eta = per_w * (len(endpoints) - i - 1)
        print(
            f"[{n_done}/{len(endpoints)}] {rec['window_end']}: "
            f"n_selected={rec['n_selected']} best_ir={rec['best_ir']} "
            f"sat_n={rec['sat_n']} sat_icir={rec['sat_icir']} "
            f"pin_icir={rec['pin_icir']} ({time.time() - w_t0:.0f}s, 余 ~{eta / 60:.0f}min)",
            flush=True,
        )
        del df_w, metrics
        gc.collect()

    if len(results) < 2:
        print("[fail] 有效窗 < 2, 无聚合意义", flush=True)
        return 1
    n_w = len(results)

    # ── 聚合 ──
    best_irs = np.array([r["best_ir"] for r in results])
    sat_ns = np.array([r["sat_n"] for r in results])
    n_sels = np.array([r["n_selected"] for r in results])
    noise_floor = {
        "n_windows": n_w,
        "best_ir": {
            "min": float(best_irs.min()),
            "max": float(best_irs.max()),
            "mean": float(best_irs.mean()),
            "std": float(best_irs.std()),
        },
        "sat_n": {
            "min": int(sat_ns.min()),
            "max": int(sat_ns.max()),
            "mean": float(sat_ns.mean()),
        },
        "n_selected": {
            "min": int(n_sels.min()),
            "max": int(n_sels.max()),
            "mean": float(n_sels.mean()),
        },
    }

    # 平均曲线 → 候选A (饱和点 + 跨窗平均 gain 排名)
    n_grid = sorted({e["n"] for r in results for e in r["ablation_log"]})
    curve_mean = {
        n: float(
            np.mean(
                [e["icir"] for r in results for e in r["ablation_log"] if e["n"] == n]
            )
        )
        for n in n_grid
    }
    max_ir = max(curve_mean.values())
    cand_a_n = next(
        n for n in n_grid if curve_mean[n] >= max_ir * gcfg.get("saturation_pct", 0.95)
    )
    cand_a_n = max(cand_a_n, gcfg.get("min_features", 30))

    rank_sum: dict[str, float] = {}
    for r in results:
        for pos, f in enumerate(r["gain_rank"]):
            rank_sum[f] = rank_sum.get(f, 0.0) + pos
    mean_rank = {f: v / n_w for f, v in rank_sum.items()}
    cand_a = sorted(mean_rank, key=mean_rank.get)[:cand_a_n]

    # 稳定核心 → 候选B (跨窗选中频率 Top-50, 平手按平均排名)
    freq: dict[str, float] = {}
    for r in results:
        for f in r["selected"]:
            freq[f] = freq.get(f, 0) + 1
    freq = {f: v / n_w for f, v in freq.items()}
    cand_b = sorted(freq, key=lambda f: (-freq[f], mean_rank.get(f, 1e9)))[
        :CAND_B_TOP_N
    ]

    # pin 对照
    pairs = [
        (r["pin_icir"], r["sat_icir"] or r["best_ir"], r["best_ir"])
        for r in results
        if r["pin_icir"] is not None
    ]
    pin_comp = {}
    if pairs:
        pin_arr = np.array([p[0] for p in pairs])
        sat_arr = np.array([p[1] for p in pairs])
        best_arr = np.array([p[2] for p in pairs])
        pin_comp = {
            "n_windows": int(len(pairs)),
            "pin_mean_icir": float(pin_arr.mean()),
            "sat_mean_icir": float(sat_arr.mean()),
            "best_mean_icir": float(best_arr.mean()),
            "pin_win_vs_sat": float((pin_arr >= sat_arr).mean()),
            "pin_win_vs_best": float((pin_arr >= best_arr).mean()),
            "mean_delta_pin_minus_sat": float((pin_arr - sat_arr).mean()),
            "mean_delta_pin_minus_best": float((pin_arr - best_arr).mean()),
        }
    pin_freq = (
        float(np.mean([freq[f] for f in pin_feats if f in freq])) if pin_feats else None
    )

    # ── WORM 落盘 ──
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag") / f"gate_d_multiwindow_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "script": "_diag_gate_d_multiwindow.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "window_days": args.window_days,
            "window_step": args.window_step,
            "min_window_days": args.min_window_days,
            "n_endpoints": len(endpoints),
            "pool_size": len(feats),
            "label": label_col,
            "min_feats": gcfg.get("min_features", 30),
            "sat_pct": gcfg.get("saturation_pct", 0.95),
            "pin_file": os.path.basename(pin_path),
        },
        "noise_floor": noise_floor,
        "averaged_curve": curve_mean,
        "candidates": {
            "A": {"n": cand_a_n, "features": cand_a},
            "B": {"features": cand_b},
        },
        "pin": {
            "n_pool": len(pin_feats),
            "mean_freq": pin_freq,
            "jaccard_A": _jaccard(pin_feats, cand_a),
            "jaccard_B": _jaccard(pin_feats, cand_b),
            "comparison": pin_comp,
        },
        "windows": results,
        "summary": {
            "noise_floor": noise_floor,
            "averaged_curve_peak": max_ir,
            "cand_a_n": cand_a_n,
            "pin_comp": pin_comp,
        },
    }
    with open(out_dir / f"gate_d_multiwindow_{ts}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    with open(
        out_dir / f"gate_d_multiwindow_{ts}_freq.csv", "w", encoding="utf-8"
    ) as fh:
        fh.write("feature,freq,mean_rank\n")
        for f in sorted(freq, key=lambda f: (-freq[f], mean_rank.get(f, 1e9))):
            fh.write(f"{f},{freq[f]:.4f},{mean_rank.get(f, 0.0):.1f}\n")

    # ── 摘要 ──
    print("\n════════ Phase 1 结果 ════════", flush=True)
    print(
        f"噪声底 ({n_w} 窗): best_ir "
        f"[{noise_floor['best_ir']['min']:.4f} .. {noise_floor['best_ir']['max']:.4f}] "
        f"mean={noise_floor['best_ir']['mean']:.4f} std={noise_floor['best_ir']['std']:.4f} | "
        f"sat_n [{noise_floor['sat_n']['min']} .. {noise_floor['sat_n']['max']}] "
        f"mean={noise_floor['sat_n']['mean']:.1f}",
        flush=True,
    )
    print(
        f"平均曲线: 峰 {max_ir:.4f} @ n={max(curve_mean, key=curve_mean.get)} | "
        f"饱和点 n={cand_a_n} (候选A)",
        flush=True,
    )
    print(
        f"候选A (平均排名 Top {cand_a_n}): 与 pin 重合 {_jaccard(pin_feats, cand_a):.0%}",
        flush=True,
    )
    print(
        f"候选B (选中频率 Top {CAND_B_TOP_N}): 与 pin 重合 {_jaccard(pin_feats, cand_b):.0%} | "
        f"pin 平均选中频率 {pin_freq:.0%}",
        flush=True,
    )
    if pin_comp:
        print(
            f"pin 对照: mean pin={pin_comp['pin_mean_icir']:.4f} vs "
            f"选中集={pin_comp['sat_mean_icir']:.4f} vs 每窗最佳={pin_comp['best_mean_icir']:.4f} | "
            f"pin 赢率 vs 选中集 {pin_comp['pin_win_vs_sat']:.0%} ({pin_comp['n_windows']} 窗)",
            flush=True,
        )
    print(f"WORM: {out_dir}", flush=True)
    print(f"总用时 {(time.time() - t0) / 60:.0f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
