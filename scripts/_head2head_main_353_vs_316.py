"""Head-to-head OOS 证明: main 20260812 (353 特征, 生产) vs 20260811b (316 特征, 旧).

用户问"main 板块 38 vs 328" — 但 main 板块无 38 特征版本 (38/328 是 dual 的特征数).
main 实际可对比: 20260812 (353 特征, 生产, 过闸发布) vs 20260811b (316 特征, 旧版).
两 bundle 结构一致 (3d/5d/10d reg+cls, 同 label), 唯一差异是特征集大小.

共享 OOS 窗口: 面板最近 EVAL_DAYS 个交易日 (对两个 bundle 均严格 OOS:
20260812 的 test=最后60日, 20260811b 训练截止 08-11 早于该窗口).
指标 (与验收同款 machinery, 不用新评估代码):
  1. weighted_IC  = validate_oos 的跨视界加权 IC (LABEL_WEIGHTS, 3d/5d/10d reg)
  2. 子窗口稳定性  = 把 eval 窗口均分 N_SUB 段, 逐段 weighted_IC (选稳定>最高)
  3. top-N 命中/收益 = 按 10d_reg 预测 (legacy main 生产排名键) 取 top-N,
                       已实现 label_pm_{3,5,10}d_net 均值 + 上涨率

只读 (不改 current_meta/模型), 结果 WORM -> BACKTEST_RESULT_DIR/main_h2h_353_vs_316_<ts>/
用法: python scripts/_head2head_main_353_vs_316.py
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
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import BACKTEST_RESULT_DIR, data_others_path

MODEL_DIR = "models/pipeline1"
BUNDLES = ("main_20260812.pkl", "main_20260811b.pkl")
WARMUP_DAYS = 330  # 特征回看窗口 (年线特征 ~252d) 暖机; 330=270 暖机 + 60 评估
EVAL_DAYS = 60  # 共享 OOS 窗口
N_SUB = 3  # 子窗口段数
TOPN = (5, 10, 20)
HORIZONS = (3, 5, 10)


def topn_metrics(test: pd.DataFrame, models: dict, cols: list[str]) -> dict:
    """按 10d_reg 预测排名, 逐日 top-N 已实现收益均值 + 上涨率."""
    reg10, label10 = models["10d_reg"]
    X = np.nan_to_num(test[cols].values, nan=0.0)
    test = test.copy()
    test["_pred_10d"] = reg10.predict(X)
    out: dict = {}
    for h in HORIZONS:
        lab = f"label_pm_{h}d_net"
        sub = test.dropna(subset=[lab]).copy()
        for n in TOPN:
            top = sub.sort_values("_pred_10d", ascending=False).groupby("date").head(n)
            out[f"{h}d_n{n}"] = {
                "mean_ret": float(top[lab].mean()),
                "hit": float((top[lab] > 0).mean()),
                "rows": int(len(top)),
            }
    return out


def main() -> int:
    panel = load_panel_v3()
    dates = sorted(panel["date"].unique())
    print(
        f"[panel] {len(panel):,}r {len(dates)}d max={panel['date'].max():%Y-%m-%d}",
        flush=True,
    )
    cut = dates[-WARMUP_DAYS]
    panel = panel[panel["date"] >= cut]
    print(f"[slice] last {WARMUP_DAYS} trading days -> {len(panel):,}r", flush=True)

    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)
    del dual_df, panel
    gc.collect()
    print(f"[main] 清洗后 {len(main_df):,}r", flush=True)

    features = FeatureEngineV35()
    # 与训练同款 registry 门控: 只构建活跃 dim, 列少不碎片 (否则全量 dim 380+ 列
    # 高度碎片化 → _consolidate_inplace OOM, 且与 353 特征训练时的特征空间不一致).
    registry = FeatureRegistry(
        path=os.path.join(
            str(data_others_path("data/factor_registry")), "feature_registry.json"
        )
    )
    df = prepare_board_frame(
        main_df, features, cross_sectional_rank=True, registry=registry
    )
    del main_df
    gc.collect()
    print(f"[feat] main 特征帧 {len(df):,}r x {len(df.columns)}c", flush=True)

    ddates = sorted(df["date"].unique())
    eval_start = ddates[-EVAL_DAYS]
    eval_df = df[df["date"] >= eval_start].copy()
    del df
    gc.collect()
    print(
        f"[eval] 窗口 {eval_start:%Y-%m-%d}..{ddates[-1]:%Y-%m-%d} "
        f"= {len(eval_df):,}r ({len(sorted(eval_df['date'].unique()))}d, 扣 mask 尾)",
        flush=True,
    )

    results: dict = {}
    for fname in BUNDLES:
        tag = fname[len("main_") : -len(".pkl")]
        bundle = DualTrackTrainer.load(os.path.join(MODEL_DIR, fname))
        cols = bundle["feature_cols"]
        present = [c for c in cols if c in eval_df.columns]
        missing = [c for c in cols if c not in eval_df.columns]
        labels = sorted({lbl for _, (_, lbl) in bundle["models"].items()})
        keep = [c for c in (present + labels) if c in eval_df.columns] + ["date", "symbol"]
        test = eval_df[keep].copy()
        # 推理端无法复现训练注入的 _brute_ 特征 → 补 0 (与 predictor.py 生产行为一致).
        # 否则这些列缺失 → validate_oos/topn 的 sub[cols] 直接 KeyError 崩溃;
        # 补 0 使 OOS 评估反映真实生产预测 (训练时这些列有真值, 生产全为 0).
        for c in missing:
            test[c] = 0.0
        print(
            f"[{tag}] {len(cols)} 特征 (有效 {len(present)} + 补0 {len(missing)}), 帧列 {len(keep)}",
            flush=True,
        )
        trained = {
            "segs": {"test": test},
            "feature_cols": cols,
            "models": bundle["models"],
        }
        oos = DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(trained)

        sub_dates = sorted(test["date"].unique())
        step = len(sub_dates) // N_SUB
        sub_ics: list[float] = []
        for i in range(N_SUB):
            s0, s1 = i * step, len(sub_dates) if i == N_SUB - 1 else (i + 1) * step
            sub_df = test[test["date"].isin(sub_dates[s0:s1])]
            tsub = {
                "segs": {"test": sub_df},
                "feature_cols": cols,
                "models": bundle["models"],
            }
            sub_ics.append(
                DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(tsub)["weighted_ic"]
            )

        results[tag] = {
            "n_features": len(cols),
            "effective_n": len(present),
            "n_zero_filled": len(missing),
            "weighted_ic": oos["weighted_ic"],
            "ics": oos["ics"],
            "sub_window_ic": sub_ics,
            "sub_window_ic_mean": float(np.mean(sub_ics)),
            "sub_window_ic_std": float(np.std(sub_ics)),
            "oos_pass": oos["pass"],
            "topn": topn_metrics(test, bundle["models"], cols),
        }
        print(
            f"[{tag}] wIC={oos['weighted_ic']:.4f} sub={[round(x, 4) for x in sub_ics]} "
            f"(mean {np.mean(sub_ics):.4f} std {np.std(sub_ics):.4f})",
            flush=True,
        )
        del test, trained, bundle
        gc.collect()

    # ---- 对比结论 ----
    tags = list(results)
    winner_ic = max(tags, key=lambda t: results[t]["weighted_ic"])
    pair_wins = {t: 0 for t in tags}
    for i in range(N_SUB):
        best = max(tags, key=lambda t: results[t]["sub_window_ic"][i])
        pair_wins[best] += 1
    verdict = {
        "weighted_ic_winner": winner_ic,
        "sub_window_majority_winner": max(pair_wins, key=pair_wins.get),
        "sub_window_wins": pair_wins,
        "note": "353 即 main_20260812 (生产); 316 即 main_20260811b (旧)",
    }
    results["_verdict"] = verdict

    out_dir = (
        Path(BACKTEST_RESULT_DIR)
        / f"main_h2h_353_vs_316_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] 结果 WORM -> {out_dir}", flush=True)
    print(
        f"[verdict] wIC: {tags[0]}={results[tags[0]]['weighted_ic']:.4f} vs "
        f"{tags[1]}={results[tags[1]]['weighted_ic']:.4f} -> {verdict['weighted_ic_winner']} "
        f"| 子窗口多数胜: {verdict['sub_window_majority_winner']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
