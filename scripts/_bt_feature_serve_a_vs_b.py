"""特征服务方式 A(训练一致/全板块) vs B(top-N/每日清单) 预测质量 回测.

用户要求回测证据支持"训练一致特征比 top-N 特征产生更好预测"的主张
(特征落盘改造的前置验证, 否则改造不推进).

设计 (隔离唯一变量 = 截面参照系):
  Mode A  训练语义:  run_train(全板块, 无 top-N) → prepare_board_frame(registry 门控)
          -> 截面列 (xrank / dim04/06/14/15/17/21/27/28/31 逐日聚合) 在全板块上计算.
  Mode B  推理语义:  run_inference(逐 date+board 取流动性 top-400 + step4 + step5 E6)
          -> features.build(inference_cols=UNION, registry=None)  ← 与 daily_pipeline 同款.
          -> 截面列在 top-N 子集上计算.

  同一 bundle、同一 (date,symbol) 可成交集合、同一 60d OOS 窗口 → 仅特征值不同.
  指标: validate_oos weighted_IC + 子窗口稳定段 + top-N 实得收益 (验收同款 machinery).

护栏: 只读 bundle (不改 current_meta / 不 pin 模型); 评估窗口严格在 bundle 训练截止后;
      标签 t+3/5/10 前向; 帧间 del + gc.collect() 防 OOM; 结果 WORM.

用法: python scripts/_bt_feature_serve_a_vs_b.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.label_engine import MASK_RECENT_DAYS, LabelEngine
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import BACKTEST_RESULT_DIR, data_others_path

MODEL_DIR = "models/pipeline1"
BUNDLES = ("dual_20260811b.pkl", "dual_20260812.pkl")
WARMUP_DAYS = 330  # 270 特征暖机 + 60 评估
EVAL_DAYS = 60
N_SUB = 3
TOPN = (5, 10, 20)
HORIZONS = (3, 5, 10)


def topn_metrics(test: pd.DataFrame, models: dict, cols: list[str]) -> dict:
    """按 10d_reg 预测排名, 逐日 top-N 已实现收益均值 + 上涨率."""
    reg10, _label10 = models["10d_reg"]
    X = np.nan_to_num(test[cols].values, nan=0.0)
    test = test.copy()
    test["_pred_10d"] = reg10.predict(X)
    out: dict = {}
    for h in HORIZONS:
        lab = f"label_pm_{h}d_net"
        sub = test.dropna(subset=[lab]).copy()
        for n in TOPN:
            top = (
                sub.sort_values("_pred_10d", ascending=False)
                .groupby("date")
                .head(n)
            )
            out[f"{h}d_n{n}"] = {
                "mean_ret": float(top[lab].mean()),
                "hit": float((top[lab] > 0).mean()),
                "rows": int(len(top)),
            }
    return out


def pred_10d(test: pd.DataFrame, models: dict, cols: list[str]) -> np.ndarray:
    reg10, _ = models["10d_reg"]
    X = np.nan_to_num(test[cols].values, nan=0.0)
    return reg10.predict(X)


def eval_bundle(
    bundle: dict,
    cols: list[str],
    test: pd.DataFrame,
) -> dict:
    """单个 bundle 在单个模式帧上的 wIC + 子窗口 + top-N 评估 (验收同款)."""
    trained = {"segs": {"test": test}, "feature_cols": cols, "models": bundle["models"]}
    oos = DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(trained)

    sub_dates = sorted(test["date"].unique())
    step = len(sub_dates) // N_SUB
    sub_ics: list[float] = []
    for i in range(N_SUB):
        s0, s1 = i * step, len(sub_dates) if i == N_SUB - 1 else (i + 1) * step
        sub_df = test[test["date"].isin(sub_dates[s0:s1])]
        tsub = {"segs": {"test": sub_df}, "feature_cols": cols, "models": bundle["models"]}
        sub_ics.append(DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(tsub)["weighted_ic"])

    return {
        "weighted_ic": oos["weighted_ic"],
        "ics": oos["ics"],
        "sub_window_ic": sub_ics,
        "sub_window_ic_mean": float(np.mean(sub_ics)),
        "sub_window_ic_std": float(np.std(sub_ics)),
        "topn": topn_metrics(test, bundle["models"], cols),
    }


def main() -> int:
    t0 = time.time()

    # ---- 先取两 bundle 的 feature_cols 并集 (Mode B 构建一次供两 bundle 复用) ----
    union_cols: list[str] = []
    bundle_cols: dict[str, list[str]] = {}
    for fname in BUNDLES:
        tag = fname[len("dual_"): -len(".pkl")]
        b = DualTrackTrainer.load(os.path.join(MODEL_DIR, fname))
        bundle_cols[tag] = list(b["feature_cols"])
        union_cols = list(dict.fromkeys(union_cols + bundle_cols[tag]))
        del b
        gc.collect()
    print(f"[cols] union={len(union_cols)} (38={len(bundle_cols['20260811b'])}, "
          f"328={len(bundle_cols['20260812'])})", flush=True)

    panel = load_panel_v3()
    dates = sorted(panel["date"].unique())
    cut = dates[-WARMUP_DAYS]
    panel = panel[panel["date"] >= cut]
    print(f"[panel] sliced last {WARMUP_DAYS} trading days -> {len(panel):,}r", flush=True)

    features = FeatureEngineV35()
    cleaner = CleaningPipeline()

    # ================= Mode B (top-N / 每日清单语义) =================
    main_b, dual_b, state = cleaner.run_inference(panel)
    del main_b
    gc.collect()
    print(f"[B] run_inference dual 清洗后 {len(dual_b):,}r (valve={state})", flush=True)
    dfB = features.build(dual_b, None, inference_cols=union_cols, cross_sectional_rank=True)
    del dual_b
    gc.collect()
    dfB = LabelEngine.build_path_labels(dfB)
    dfB = LabelEngine.build_labels(dfB)
    dfB = LabelEngine.mask_suspension(dfB)
    dfB = LabelEngine.mask_recent_days(dfB, days=MASK_RECENT_DAYS)
    print(f"[B] 特征帧 {len(dfB):,}r x {len(dfB.columns)}c", flush=True)

    b_dates = sorted(dfB["date"].unique())
    eval_start = b_dates[-EVAL_DAYS]
    keep_B = dfB[["date", "symbol"]][dfB["date"] >= eval_start].drop_duplicates()
    n_B = len(keep_B)
    print(f"[B] eval 窗口 {eval_start:%Y-%m-%d}..{b_dates[-1]:%Y-%m-%d} "
          f"{len(sorted(dfB[dfB['date']>=eval_start]['date'].unique()))}d, {n_B:,} 行",
          flush=True)

    # ================= Mode A (训练一致 / 全板块语义) =================
    main_a, dual_a = cleaner.run_train(panel)
    del main_a, panel
    gc.collect()
    print(f"[A] run_train dual 清洗后 {len(dual_a):,}r", flush=True)
    registry = FeatureRegistry(
        path=os.path.join(str(data_others_path("data/factor_registry")), "feature_registry.json")
    )
    dfA = prepare_board_frame(dual_a, features, cross_sectional_rank=True, registry=registry)
    del dual_a
    gc.collect()
    print(f"[A] 特征帧 {len(dfA):,}r x {len(dfA.columns)}c", flush=True)

    eval_A = dfA[dfA["date"] >= eval_start].merge(keep_B, on=["date", "symbol"], how="inner")
    del dfA
    gc.collect()
    print(f"[A] 对齐到 top-N 可成交集合后 {len(eval_A):,}r", flush=True)

    # ---- 逐 date 行数一致性校验 (对齐失败 = 实验失效, 失败要大声) ----
    cnt_B = keep_B.groupby("date").size()
    cnt_A = eval_A.groupby("date").size()
    mism = (cnt_A.index != cnt_B.index).any() or (cnt_A.values != cnt_B.values).any()
    if mism:
        print("[FATAL] Mode A/B 逐 date 行数不一致 → 实验失效, abort", flush=True)
        return 3
    print(f"[align] 两帧逐 date 行数一致, 每日期望 ≈{int(cnt_B.mean())} 行", flush=True)

    # ================= 逐 bundle 评估两模式 =================
    results: dict = {}
    for tag, cols in bundle_cols.items():
        bundle = DualTrackTrainer.load(os.path.join(MODEL_DIR, f"dual_{tag}.pkl"))
        labels = sorted({lbl for _, (_, lbl) in bundle["models"].items()})
        keep_cols = [c for c in (cols + labels) if c in eval_A.columns] + ["date", "symbol"]
        missingA = [c for c in cols if c not in eval_A.columns]
        missingB = [c for c in cols if c not in dfB.columns]
        if missingA or missingB:
            print(f"[FATAL] {tag} 缺失特征列 A={missingA[:5]} B={missingB[:5]} → abort",
                  flush=True)
            return 3
        testA = eval_A[keep_cols].sort_values(["date", "symbol"]).reset_index(drop=True)
        testB = dfB[dfB["date"] >= eval_start][keep_cols].sort_values(
            ["date", "symbol"]
        ).reset_index(drop=True)
        if len(testA) != len(testB):
            print(f"[FATAL] {tag} 评估行数 A={len(testA)} B={len(testB)} → abort", flush=True)
            return 3

        resA = eval_bundle(bundle, cols, testA)
        resB = eval_bundle(bundle, cols, testB)

        # 特征漂移佐证: 两模式预测排名 spearman (0=无关, 1=特征等价) + 特征位移指数
        pA = pd.Series(pred_10d(testA, bundle["models"], cols))
        pB = pd.Series(pred_10d(testB, bundle["models"], cols))
        rank_spearman = float(pA.rank().corr(pB.rank()))
        delta_std = (testA[cols] - testB[cols]).abs().mean()
        stdA = testA[cols].std()
        shift_idx = float(
            (delta_std / stdA).replace([np.inf, -np.inf], np.nan).mean()
        )

        results[tag] = {
            "n_features": len(cols),
            "mode_A_full_board": resA,
            "mode_B_topn": resB,
            "delta_wic": float(resA["weighted_ic"] - resB["weighted_ic"]),
            "pred_rank_spearman_A_vs_B": rank_spearman,
            "feature_shift_index": shift_idx,  # 0=同构; 越大=截面参照系越影响特征
        }
        wA, wB = resA["weighted_ic"], resB["weighted_ic"]
        tA10, tB10 = resA["topn"]["10d_n10"]["mean_ret"], resB["topn"]["10d_n10"]["mean_ret"]
        print(
            f"[{tag}] wIC A={wA:.4f} B={wB:.4f} (Δ={wA - wB:+.4f}) | "
            f"top10 实得 10d A={tA10:+.3f} B={tB10:+.3f} | "
            f"pred_rank_spearman={rank_spearman:.3f} shift={shift_idx:.3f}",
            flush=True,
        )
        del bundle, testA, testB
        gc.collect()

    # ---- 判定 ----
    verdict = {"note": "主张=训练一致(全板块)特征 ≥ top-N 特征; 每 bundle 看 wIC + top10 双指标"}
    for tag, r in results.items():
        a10, b10 = r["mode_A_full_board"]["topn"]["10d_n10"]["mean_ret"], \
                   r["mode_B_topn"]["topn"]["10d_n10"]["mean_ret"]
        verdict[tag] = {
            "A_wins_ic": r["mode_A_full_board"]["weighted_ic"] >= r["mode_B_topn"]["weighted_ic"],
            "A_wins_top10": a10 >= b10,
            "delta_wic": r["delta_wic"],
            "delta_top10_ret": float(a10 - b10),
        }
    results["_verdict"] = verdict

    out_dir = Path(BACKTEST_RESULT_DIR) / f"feat_serve_a_vs_b_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] 结果 WORM -> {out_dir}  ({time.time() - t0:.0f}s)", flush=True)
    for tag, v in verdict.items():
        if tag == "note":
            continue
        print(f"[verdict {tag}] ic={'A' if v['A_wins_ic'] else 'B'} "
              f"top10={'A' if v['A_wins_top10'] else 'B'} "
              f"Δwic={v['delta_wic']:+.4f} Δtop10={v['delta_top10_ret']:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
