"""_diag_weight_ab.py — legacy 选股方案 A/B: 纯 10d vs 固定 compound vs 自适应权重.

用户 2026-08-08 裁决权重 (弃 2d / 3d 相对 5d 最小化 / 10d 最高), 并改 list_generator 用
10d_pred 进 compound. 用户 2026-08-09 追问: 为什么准入门用 compound 而非纯 10d_pred?
以及 compound 是否该用自适应权重. 本脚本用 OOS 回测量化, 同模型 / 同池 / 同排名键
(pred_ret_10d), 唯一差异 = 准入 gate 的 return 条件, 度量三个方案的清单实得表现:

    PURE10: gate 需要 pred_ret_10d > 0                 (纯资金窗口口径)
    OLD:    gate 需要 0.45*pred_2d+0.35*pred_3d+0.20*pred_5d > 0    (含 2d, 2d 最高)
    NEW:    gate 需要 0.10*pred_3d+0.40*pred_5d+0.50*pred_10d > 0   (弃 2d / 10d 最高)
    ADAPT:  gate 需要自适应 compound > 0; 权重逐日 walk-forward, 各视界按最近
            adapt_w 交易日的 Rank IC (pred vs 实得) 正值归一; 全非正回退 NEW.

流水线 (与 _diag_icdedup_topn_ab 一致): CleaningPipeline.run_train(main) →
prepare_board_frame (V35 build, 可 --board-cache 复用) → BASE 选择 (方差 dedup, 生产默认)
→ risk_filter → OOS 边界切分 (训练严格早于 OOS) → LGBM reg/cls × 2/3/5/10d →
逐日走 OOS: 预测 → 三方案各自算准入门 → 按 pred_ret_10d 降序取 TOP10 →
度量清单命中率+均值 (T+10 资金窗口为主) + 逐日 Rank IC (预测质量) + OOS 子窗稳定性.

Gate: 相对 PURE10 基线, T+10 命中率+均值 >= (容差 -0.5pp) → 该方案不劣化;
ADAPT 平均权重诊断是否收敛到纯 10d.

产出 (WORM): data/_diag_weight_ab_<ts>.json + 控制台对比表 + 结论.
用法: python scripts/_diag_weight_ab.py [--frame ...] [--window-days 250] [--oos-days 120] [--adapt-w 60]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

from app.pipeline1.cleaning_pipeline import CleaningPipeline  # noqa: E402
from app.pipeline1.feature_engine_v35 import FeatureEngineV35  # noqa: E402
from app.pipeline1.feature_selector import FeatureSelector  # noqa: E402
from app.pipeline1.train_runner import (  # noqa: E402
    prepare_board_frame,
    select_features,
)
from config.settings import PANEL_V3_PATH  # noqa: E402
from scripts._diag_icdedup_topn_ab import (  # noqa: E402
    PREP_DAYS,
    TMP_REGISTRY,
    _downcast,
    _fit_model,
    _mem,
)

logging.disable(logging.ERROR)

HORIZONS = (2, 3, 5, 10)
TOP_N = 10
# OLD (2026-08-08 前的生产权重, 含 2d 且 2d 最高) vs NEW (弃 2d / 10d 最高, 生产现行).
OLD_W = {2: 0.45, 3: 0.35, 5: 0.20}
NEW_W = {3: 0.10, 5: 0.40, 10: 0.50}
HIT_TOL = 0.005  # -0.5pp 噪声容差


def _run_arm(df, oos_start):
    cfg = FeatureSelector.DEFAULT_CONFIG  # BASE: 方差 dedup, 生产默认
    os.makedirs(TMP_REGISTRY, exist_ok=True)
    sel = FeatureSelector(config=cfg, registry_dir=TMP_REGISTRY)
    picked, aug = select_features(df, "main", "diag_wab", sel)
    # select_features 注入 brute 列 → aug 才有 picked 全集 (勿用 df[keep] 过滤).
    picked = [c for c in picked if c in aug.columns]
    aug = aug.reset_index(drop=True)
    _downcast(aug)
    tr = aug[aug["date"] < oos_start].copy()
    te = aug[aug["date"] >= oos_start].copy()
    print(
        f"  picked={len(picked)} train={len(tr)} ({tr['date'].nunique()}日) "
        f"test={len(te)} ({te['date'].nunique()}日)",
        flush=True,
    )
    return picked, tr, te, aug


def _arm_daily(
    rows: pd.DataFrame,
    base_rate_ser: dict,
    gate_fn,
) -> pd.DataFrame:
    """逐日对 gate 通过集按 pred_ret_10d 取 TOP_N, 返回逐日清单度量."""
    recs = []
    for d, g in rows.groupby("date"):
        base = float(base_rate_ser.get(d, 0.5))
        ok = gate_fn(g, base, d)
        pool = g[ok]
        if len(pool) == 0:
            continue
        top = pool.nlargest(TOP_N, "pred_10d")
        recs.append(
            {
                "date": d,
                "n_pick": int(len(top)),
                "hit10": float((top["label_pm_10d_net"] > 0).mean()),
                "mean10": float(top["label_pm_10d_net"].mean()),
            }
        )
    return pd.DataFrame(recs)


def main() -> None:
    parser = argparse.ArgumentParser(description="legacy 跨视界权重 A/B")
    parser.add_argument("--window-days", type=int, default=380)
    parser.add_argument("--oos-days", type=int, default=120)
    parser.add_argument("--board-cache", default=None)
    parser.add_argument(
        "--frame",
        default=None,
        help="预构建 frame parquet (跳过 3.3h build, 复用 _diag_stage_main_3y.parquet)",
    )
    parser.add_argument(
        "--adapt-w",
        type=int,
        default=60,
        help="ADAPT 自适应权重 trailing 窗口 (交易日, 默认 60)",
    )
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 快速路径: 直接载入已构建的 frame (跳过 3.3h CleaningPipeline + prepare_board_frame).
    # 复用 _diag_stage_main_3y.parquet (2026-08-07 生成, 含全标签 + is_suspended).
    if args.frame and os.path.exists(args.frame):
        df = pd.read_parquet(args.frame)
        df["date"] = pd.to_datetime(df["date"])
        all_dates = np.array(sorted(df["date"].unique()))
        window_start = pd.Timestamp(all_dates[-args.window_days])
        oos_start = pd.Timestamp(all_dates[-args.oos_days])
        print(f"[1/2] 载入预构建 frame {args.frame}: rows={len(df)} cols={len(df.columns)}", flush=True)
        print(
            f"  窗 {window_start.date()}..{all_dates[-1].date()} | OOS {oos_start.date()}..",
            flush=True,
        )
        assert args.window_days > args.oos_days
        assert args.window_days <= PREP_DAYS
        _downcast(df)
        df = df[df["date"] >= window_start].reset_index(drop=True)
        print(f"  裁剪到分析窗 rows={len(df)} ({df['date'].nunique()}日)", flush=True)
        gc.collect()
        _mem("frame")
    else:
        print("[1/4] 载入面板 + 清洗主板 ...", flush=True)
        panel = pd.read_parquet(PANEL_V3_PATH)
        panel["date"] = pd.to_datetime(panel["date"])
        all_dates = np.array(sorted(panel["date"].unique()))
        window_start = pd.Timestamp(all_dates[-args.window_days])
        oos_start = pd.Timestamp(all_dates[-args.oos_days])
        print(
            f"  全窗 {all_dates[0].date()}..{all_dates[-1].date()} | "
            f"窗 {window_start.date()}.. | OOS {oos_start.date()}..",
            flush=True,
        )
        assert args.window_days > args.oos_days
        assert args.window_days <= PREP_DAYS
        cleaner = CleaningPipeline()
        main_df, _ = cleaner.run_train(panel, board="main")
        del panel
        gc.collect()
        m_dates = np.array(sorted(main_df["date"].unique()))
        prep_start = pd.Timestamp(m_dates[-PREP_DAYS])
        main_df = main_df[main_df["date"] >= prep_start].reset_index(drop=True)
        _downcast(main_df)
        print(
            f"  清洗后 3y 窗 rows={len(main_df)} ({prep_start.date()}..)",
            flush=True,
        )
        _mem("cleaned")

        print("[2/4] prepare_board_frame (V35 3y build + 标签) ...", flush=True)
        if args.board_cache and os.path.exists(args.board_cache):
            df = pd.read_parquet(args.board_cache)
            print(f"  [cache] 载入 board frame rows={len(df)} cols={len(df.columns)}", flush=True)
            del main_df
            gc.collect()
        else:
            df = prepare_board_frame(main_df, FeatureEngineV35())
            if args.board_cache:
                os.makedirs(os.path.dirname(args.board_cache), exist_ok=True)
                df.to_parquet(args.board_cache)
                print(f"  [cache] 已保存 board frame ({len(df)} rows)", flush=True)
            del main_df
            gc.collect()
        _downcast(df)
        df = df[df["date"] >= window_start].reset_index(drop=True)
        print(f"  board_frame rows={len(df)} ({df['date'].nunique()}日)", flush=True)
        gc.collect()
        _mem("board_frame")

    print("[2/3] 选择特征 + 训练 LGBM ...", flush=True)
    picked, tr, te, aug = _run_arm(df, oos_start)
    del df
    gc.collect()
    cache = {}
    for k in HORIZONS:
        for kind in ("reg", "cls"):
            t0 = time.time()
            cache[(kind, k)] = _fit_model(tr, picked, kind, k)
            print(f"  fit {k}d_{kind} {time.time() - t0:.1f}s", flush=True)
    del tr
    gc.collect()

    print("[3/3] OOS 逐日走查 (PURE10 / NEW 固定 / ADAPT 自适应) ...", flush=True)
    oos_dates = sorted(te["date"].unique())
    rows = te.dropna(subset=["label_pm_10d_net"]).copy()
    if len(rows) < 200:
        print("  OOS 有效行不足, 退出")
        return
    X_te = np.nan_to_num(rows[picked].to_numpy(dtype=float), nan=0.0)
    rows["pred_10d"] = cache[("reg", 10)].predict(X_te)
    for k in HORIZONS:
        rows[f"pred_{k}d"] = cache[("reg", k)].predict(X_te)
    rows["prob10"] = cache[("cls", 10)].predict_proba(X_te)[:, 1]

    # base_rate = 逐日 prob10 均值序列 (B4 20日滚动均值; 前窗用扩张均值)
    daily_prob = rows.groupby("date")["prob10"].mean()
    base_hist = []
    base_rate_ser = {}
    for d in oos_dates:
        hist = base_hist[-20:] if base_hist else []
        base_rate_ser[d] = float(np.mean(hist)) if hist else 0.5
        base_hist.append(float(daily_prob.get(d, 0.5)))

    # OLD / NEW 固定 compound (present 归一; 全部在 → /1.0)
    rows["compound_old"] = (
        OLD_W[2] * rows["pred_2d"] + OLD_W[3] * rows["pred_3d"] + OLD_W[5] * rows["pred_5d"]
    ) / sum(OLD_W.values())
    rows["compound_new"] = (
        NEW_W[3] * rows["pred_3d"] + NEW_W[5] * rows["pred_5d"] + NEW_W[10] * rows["pred_10d"]
    ) / sum(NEW_W.values())

    # ADAPT: 逐日自适应权重 (walk-forward, 严格早于当日) = 各视界最近 adapt_w 交易日
    # 的逐日 Rank IC (pred vs 实得) 正值归一; 全非正 → 回退 NEW 固定权重.
    print(f"  [ADAPT] 逐日自适应权重 (trail={args.adapt_w}日, 无 look-ahead) ...", flush=True)
    X_all = np.nan_to_num(aug[picked].to_numpy(dtype=float), nan=0.0)
    all_df = pd.DataFrame({"date": aug["date"].values})
    for k in HORIZONS:
        all_df[f"pred_{k}d"] = cache[("reg", k)].predict(X_all)
        all_df[f"lab_{k}"] = aug.get(f"label_pm_{k}d_net")
    ic_by_day = {}
    for k in HORIZONS:
        sub = all_df.dropna(subset=[f"lab_{k}"]).copy()
        tmp = {}
        for dd, gg in sub.groupby("date"):
            if len(gg) >= 20:
                r = gg[[f"pred_{k}d", f"lab_{k}"]].rank(pct=True)
                ic = float(r[f"pred_{k}d"].corr(r[f"lab_{k}"]))
                if np.isfinite(ic):  # 常数分组 → NaN, 不入 trailing 窗
                    tmp[dd] = ic
        ic_by_day[k] = tmp
    all_dates = sorted(aug["date"].unique())
    pos = {dd: i for i, dd in enumerate(all_dates)}
    adapt_w_by_date = {}
    for d in oos_dates:
        i = pos[d]
        lo = max(0, i - args.adapt_w)
        w = {}
        for k in HORIZONS:
            vals = [
                ic_by_day[k][all_dates[j]]
                for j in range(lo, i)
                if all_dates[j] in ic_by_day[k]
            ]
            w[k] = float(np.mean(vals)) if vals else 0.0
        w = {k: max(v, 0.0) for k, v in w.items()}
        tot = sum(w.values())
        adapt_w_by_date[d] = {k: v / tot for k, v in w.items()} if tot > 1e-12 else dict(NEW_W)
    avg_w = {
        k: float(np.mean([adapt_w_by_date[d].get(k, 0.0) for d in oos_dates])) for k in HORIZONS
    }

    def pure_gate(g, base, d):
        return (g["prob10"] > base) & (g["pred_10d"] > 0)

    def old_gate(g, base, d):
        return (g["prob10"] > base) & (g["compound_old"] > 0)

    def new_gate(g, base, d):
        return (g["prob10"] > base) & (g["compound_new"] > 0)

    def adapt_gate(g, base, d):
        w = adapt_w_by_date[d]
        comp = sum(w[k] * g[f"pred_{k}d"] for k in w)
        return (g["prob10"] > base) & (comp > 0)

    pure_df = _arm_daily(rows, base_rate_ser, pure_gate)
    old_df = _arm_daily(rows, base_rate_ser, old_gate)
    new_df = _arm_daily(rows, base_rate_ser, new_gate)
    adapt_df = _arm_daily(rows, base_rate_ser, adapt_gate)

    # 预测质量 (与 gate 无关): 逐日 Rank IC pred_10d vs label_pm_10d_net
    ics = []
    for _d, g in rows.groupby("date"):
        if len(g) >= 20:
            r = g[["pred_10d", "label_pm_10d_net"]].rank(pct=True)
            ic = float(r["pred_10d"].corr(r["label_pm_10d_net"]))
            if np.isfinite(ic):
                ics.append(ic)
    forecast_ic = float(np.mean(ics)) if ics else np.nan

    def summarize(d0: pd.DataFrame, name: str):
        if len(d0) == 0:
            print(f"  {name}: 无清单日")
            return {"days": 0}
        return {
            "days": int(len(d0)),
            "avg_n_pick": float(d0["n_pick"].mean()),
            "hit10": float(d0["hit10"].mean()),
            "mean10": float(d0["mean10"].mean()),
        }

    arms = {
        "PURE10": summarize(pure_df, "PURE10"),
        "OLD": summarize(old_df, "OLD"),
        "NEW": summarize(new_df, "NEW"),
        "ADAPT": summarize(adapt_df, "ADAPT"),
    }

    # 子窗稳定性 (OOS 三等分)
    chunks = np.array_split(np.array(oos_dates), 3)
    subwin = {}
    for name, df_ in (
        ("PURE10", pure_df),
        ("OLD", old_df),
        ("NEW", new_df),
        ("ADAPT", adapt_df),
    ):
        subwin[name] = []
        for ch in chunks:
            sub = df_[df_["date"].isin(set(ch))]
            subwin[name].append(
                [float(sub["hit10"].mean()), float(sub["mean10"].mean())]
                if len(sub) else [None, None]
            )

    print(f"\n  OOS 逐日 Rank IC (pred_10d vs 实得): {forecast_ic:.4f}")
    print("  ADAPT 平均自适应权重: " + "  ".join(f"{k}d={avg_w.get(k, 0.0):.3f}" for k in HORIZONS))

    print("\n" + "=" * 96)
    print("legacy 选股方案 A/B — OOS 对比 (同模型/同池/同排名键 pred_10d, 仅准入门不同)")
    print("   PURE10: pred_10d>0 | OLD: 2d-heavy compound>0 | NEW: 10d-dominant compound>0 | ADAPT: 自适应")
    print("=" * 96)
    print(f"{'metric':<14s} {'PURE10':>11s} {'OLD':>10s} {'NEW':>10s} {'ADAPT':>10s} {'ΔNEW-OLD':>12s}")
    print("-" * 96)
    for k, lab in (("days", "清单日数"), ("avg_n_pick", "日均选股数"), ("hit10", "T+10 命中率"), ("mean10", "T+10 均值")):
        pv = arms["PURE10"].get(k, 0.0)
        ov = arms["OLD"].get(k, 0.0)
        nv = arms["NEW"].get(k, 0.0)
        av = arms["ADAPT"].get(k, 0.0)
        if k == "hit10":
            print(f"{lab:<14s} {pv * 100:>10.2f}% {ov * 100:>9.2f}% {nv * 100:>9.2f}% {av * 100:>9.2f}% {(nv - ov) * 100:>+11.2f}pp")
        elif k == "mean10":
            print(f"{lab:<14s} {pv * 100:>10.3f}% {ov * 100:>9.3f}% {nv * 100:>9.3f}% {av * 100:>9.3f}% {(nv - ov) * 100:>+11.3f}pp")
        else:
            print(f"{lab:<14s} {pv:>11.2f} {ov:>10.2f} {nv:>10.2f} {av:>10.2f} {(nv - ov):>+12.2f}")
    new_ge_old = (
        arms["NEW"]["hit10"] >= arms["OLD"]["hit10"] - HIT_TOL
        and arms["NEW"]["mean10"] >= arms["OLD"]["mean10"] - HIT_TOL
    )
    hit_pass = {
        n: arms[n]["hit10"] >= arms["PURE10"]["hit10"] - HIT_TOL for n in ("NEW", "ADAPT")
    }
    mean_pass = {
        n: arms[n]["mean10"] >= arms["PURE10"]["mean10"] - HIT_TOL for n in ("NEW", "ADAPT")
    }
    adapt_vs_new = (
        arms["ADAPT"]["hit10"] >= arms["NEW"]["hit10"] - HIT_TOL
        and arms["ADAPT"]["mean10"] >= arms["NEW"]["mean10"] - HIT_TOL
    )
    print("=" * 96)
    print(f"  OLD(含2d) vs NEW: {'NEW ≥ OLD → 弃 2d 决定成立' if new_ge_old else 'NEW 劣于 OLD → 弃 2d 需复核'}")
    for n in ("NEW", "ADAPT"):
        print(f"  {n:<6s} vs PURE10: T+10 命中率 {'PASS' if hit_pass[n] else 'FAIL'} / 均值 {'PASS' if mean_pass[n] else 'FAIL'}")
    print(f"  ADAPT vs NEW: {'ADAPT 不劣于 NEW' if adapt_vs_new else 'ADAPT 劣于 NEW'}")
    if new_ge_old:
        verdict = "PASS → 弃 2d 成立: NEW(10d-dominant) ≥ OLD(含2d)"
    elif arms["OLD"]["hit10"] > arms["NEW"]["hit10"] + HIT_TOL or arms["OLD"]["mean10"] > arms["NEW"]["mean10"] + HIT_TOL:
        verdict = "FAIL → OLD(含2d) 显著更优, 弃 2d 决定需人工复核"
    else:
        verdict = "OLD vs NEW 在噪声容差内打平, 维持现行 NEW"
    print(f"  ==> {verdict}")
    print("=" * 96)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", f"_diag_weight_ab_{ts}.json")
    payload = {
        "created": ts,
        "window_days": args.window_days,
        "oos_days": args.oos_days,
        "oos_start": str(oos_start.date()),
        "adapt_w": args.adapt_w,
        "old_w": OLD_W,
        "new_w": NEW_W,
        "forecast_ic_10d": forecast_ic,
        "arms": arms,
        "subwindow": subwin,
        "avg_adapt_w": avg_w,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n报告落盘 (WORM): {out_path}")


if __name__ == "__main__":
    main()
