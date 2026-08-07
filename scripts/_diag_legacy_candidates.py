"""Legacy 候选诊断: 复现 daily_pipeline.run() 但逐候选拆解 E7 准入闸.

背景 (2026-08-05): main_current/dual_current 已回滚为 8-03 干净模型,
legacy 全链路可跑, 但 E7 计算型准入输出 0 只 → 今日空清单.
本脚本回答: 0 只是"模型真没看到机会"还是"某道闸误杀"? 输出每只候选的
score / prob / pred_ret / 各闸通过与否, 供用户判断.

用法: python scripts/_diag_legacy_candidates.py [YYYYMMDD]
输出: DATA_OTHERS/diag/legacy_candidates_{date}.csv (含闸标记) + stdout 摘要
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gc

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import PANEL_V3_PATH, data_others_path

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}


def main():
    trade_date = sys.argv[1] if len(sys.argv) > 1 else "20260805"
    t0 = time.time()
    predictor = V35Predictor(BUNDLES)
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    lister = ListGenerator()

    panel = pd.read_parquet(str(PANEL_V3_PATH))
    print(
        f"[panel] {len(panel):,}r max={panel['date'].max():%Y-%m-%d} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    dates = sorted(panel["date"].unique())
    cut = dates[-300]
    panel = panel[panel["date"] >= cut]
    print(
        f"[slice] {cut.date()}.. {len(panel):,}r ({time.time() - t0:.0f}s)", flush=True
    )

    main_df, dual_df, valve_state = cleaner.run_inference(panel)
    del panel
    gc.collect()
    print(
        f"[clean] valve={valve_state} main={len(main_df):,} dual={len(dual_df):,} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    latest_date = None
    frames = []
    for board, df in (("main", main_df), ("dual", dual_df)):
        if len(df) == 0 or board not in predictor.bundles:
            print(f"[{board}] skip", flush=True)
            continue
        cols = predictor.bundles[board]["feature_cols"]
        feat = features.build(
            df,
            None,
            inference_cols=cols,
            cross_sectional_rank=(board == "dual"),
        )
        latest_symbols = df[df["date"] == df["date"].max()]["symbol"]
        today_feat = feat[feat["symbol"].isin(set(latest_symbols))]
        latest_date = df["date"].max()
        pred = predictor.predict(today_feat, board)
        frames.append(pred)
        print(
            f"[{board}] feat={len(today_feat)} pred={len(pred)} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        del feat, df
        gc.collect()
    if not frames:
        raise SystemExit("无候选")
    candidates = pd.concat(frames, ignore_index=True)

    # compute_scores (与 list_generator.emit 相同入口) → base_rate/compound_ret/score
    scored = lister.compute_scores(candidates)

    # 逐闸拆解 (镜像 entry_filter 逻辑; 非 bear)
    ok_prob = scored["compound_prob"] > scored["base_rate"]
    ok_ret = scored["compound_ret"] > 0
    ok_q50 = True
    if "pred_q50" in scored.columns and scored["pred_q50"].notna().any():
        ok_q50 = scored["pred_q50"].fillna(scored["compound_ret"]) > 0
    ok_pain = True
    if "pain_prob" in scored.columns:
        ok_pain = scored["pain_prob"].fillna(0) <= 0.5
    passed_all = ok_prob & ok_ret & ok_q50 & ok_pain

    out = scored.copy()
    out["gate_prob"] = ok_prob
    out["gate_ret"] = ok_ret
    out["gate_q50"] = ok_q50
    out["gate_pain"] = ok_pain
    out["passed_all"] = passed_all

    show_cols = [
        "symbol",
        "board",
        "score",
        "prob_up",
        "compound_prob",
        "base_rate",
        "pred_ret_1d",
        "pred_ret_2d",
        "pred_ret_3d",
        "pred_ret_5d",
        "compound_ret",
        "pred_q50",
        "pain_prob",
        "gate_prob",
        "gate_ret",
        "gate_q50",
        "gate_pain",
        "passed_all",
    ]
    show_cols = [c for c in show_cols if c in out.columns]

    print(
        f"\n=== 候选统计 (trade_date={trade_date}, data_date={latest_date:%Y-%m-%d}) ===",
        flush=True,
    )
    print(f"候选总数: {len(out)} | 全闸通过: {int(passed_all.sum())}", flush=True)
    print(f"闸1 prob>base_rate: {int(ok_prob.sum())}/{len(out)}", flush=True)
    print(f"闸2 compound_ret>0: {int(ok_ret.sum())}/{len(out)}", flush=True)
    print(f"闸3 pred_q50>0: {int(ok_q50.sum())}/{len(out)}", flush=True)
    print(f"闸4 pain<=0.5: {int(ok_pain.sum())}/{len(out)}", flush=True)
    print(
        f"\nprob_up 分布: {np.percentile(out['prob_up'], [10, 25, 50, 75, 90]).round(3)}",
        flush=True,
    )
    print(
        f"compound_ret 分布: {np.percentile(out['compound_ret'], [10, 25, 50, 75, 90]).round(4)}",
        flush=True,
    )
    print(
        f"pred_ret_1d 分布: {np.percentile(out['pred_ret_1d'], [10, 25, 50, 75, 90]).round(4)}",
        flush=True,
    )
    print(f"base_rate={float(scored['base_rate'].iloc[0]):.4f}", flush=True)

    print("\n=== 按 score 前 20 (near-miss, 各闸标记) ===", flush=True)
    top = out.sort_values("score", ascending=False).head(20)
    print(top[show_cols].to_string(index=False), flush=True)

    diag_dir = data_others_path("data/diag")
    os.makedirs(diag_dir, exist_ok=True)
    out[show_cols].to_csv(
        os.path.join(str(diag_dir), f"legacy_candidates_{trade_date}.csv"),
        index=False,
    )
    print(
        f"\n[saved] legacy_candidates_{trade_date}.csv ({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
