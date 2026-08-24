"""原型: reg 派生概率 vs 生产 Platt cls 概率 — rank IC + ECE 对比 (2026-08-24)
==================================================================================
问题: 生产 cls 头 raw Rank IC ≈ 0 (main OOS 3d/5d/10d_cls 全 0.0), prob 列不排序股票.
候选: 用回归模型的残差分布派生概率 p_reg = E[1(r_hat + e > CLS_THRESHOLD)],
      继承 reg 头 IC (5d 0.074 / 10d 0.105) 而非近常数分类头.

本脚本在 test 段 (与重校同管道) 对比:
  - p_cls = 生产 Platt cls 概率 (predictor 实际口径)
  - p_reg = 回归残差派生概率 (calib 段残差经验分布, 无未来函数)
  - p_reg_gauss = p_reg 的 Gaussian 变体 (稳健性对照, Φ((r_hat-TH)/σ_e))
指标:
  - rank_ic(prob, 连续实际收益) + rank_ic(prob, up 指示)   — 判别力 (日截面 Spearman 均值)
  - ECE(prob, up 指示)                                       — 校准度 (10 桶)
评估事件统一为 up = (reg 净收益标签 > CLS_THRESHOLD), 与 reg 模型同口径.
p_cls 名义事件为毛收益>0.5% (≈净收益>0.2%), 阈值差 ~0.2pp, rank IC 不受影响, ECE 有此小偏移.

用法: python scripts/_diag_prob_reg_derived.py [--boards main,dual]
输出: 每 board × 每视界 (3/5/10) 对比表; 结果 JSON 落 data/_diag_prob_reg_derived.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy.stats import norm

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_selector import BRUTE_FAMILIES, BruteForceGenerator
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.label_engine import CLS_THRESHOLD
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import PANEL_V3_PATH, data_others_path

MODEL_DIR = "models/pipeline1"
REGISTRY_PATH = str(data_others_path("data/factor_registry"))
OUT_JSON = "data/_diag_prob_reg_derived.json"
TH = CLS_THRESHOLD


def build_board_df(
    board_df: pd.DataFrame,
    features: FeatureEngineV35,
    registry: FeatureRegistry,
    bundle: dict,
    board: str,
) -> pd.DataFrame:
    """prepare_board_frame + BruteForce 注入 (镜像重校脚本管道)."""
    df = prepare_board_frame(
        board_df,
        features,
        cross_sectional_rank=(board != "main"),
        registry=registry,
    )
    missing = [c for c in bundle["feature_cols"] if c not in df.columns]
    brute_missing = [c for c in missing if "_brute_" in c]
    if brute_missing:
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        need = set(brute_missing)
        picks = []
        for fam in BRUTE_FAMILIES:
            new = gen.generate_columns(
                df, fam, need, raw_cols=raw_cols, dtype="float32"
            )
            if new is None or not len(new.columns):
                continue
            picks.append(new)
        if picks:
            _brute = pd.concat(picks, axis=1)
            for _c in _brute.columns:
                df[_c] = _brute[_c].to_numpy()
            print(
                f"[{board}] BruteForce 注入 {len(_brute.columns)} 列 "
                f"(缺 {len(brute_missing)})",
                flush=True,
            )
    still = [c for c in bundle["feature_cols"] if c not in df.columns]
    if still:
        raise RuntimeError(
            f"[{board}] 缺 {len(still)} 特征列: {still[:8]} (特征引擎/注册表不一致)"
        )
    return df


def prob_from_residuals(r_hat: np.ndarray, e: np.ndarray, th: float = TH) -> np.ndarray:
    """经验残差分布: p = P(r_hat + e > th) = 1 - F_e(th - r_hat)."""
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 30:
        return np.full(len(r_hat), np.nan)
    e_sorted = np.sort(e)
    p = 1.0 - np.searchsorted(e_sorted, th - np.asarray(r_hat, dtype=float)) / len(e)
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def ece(prob: np.ndarray, up: np.ndarray, n_bins: int = 10) -> float:
    mask = np.isfinite(prob) & np.isfinite(up)
    prob, up = prob[mask], up[mask]
    if len(prob) < 30:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    val = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.sum():
            continue
        w = sel.sum() / len(prob)
        val += w * abs(prob[sel].mean() - up[sel].mean())
    return float(val)


def compare_horizon(board: str, trained: dict, k: int) -> dict | None:
    cols = trained["feature_cols"]
    models = trained["models"]
    kind_cls, kind_reg = f"{k}d_cls", f"{k}d_reg"
    if kind_cls not in models or kind_reg not in models:
        return None
    cls_model, cls_label = models[kind_cls]
    reg_model, reg_label = models[kind_reg]
    cal = trained["calibrators"].get(k)
    segs = trained["segs"]

    test = segs["test"].dropna(subset=[cls_label, reg_label]).copy()
    if len(test) < 30:
        return None
    X_test = np.nan_to_num(test[cols].values, nan=0.0)

    # p_cls = 生产 Platt cls 概率
    raw_cls = cls_model.predict_proba(X_test)[:, 1]
    p_cls = cal.predict_proba(raw_cls) if cal is not None else raw_cls

    # p_reg: calib 段回归残差经验分布 (无未来函数)
    calib = segs["calib"].dropna(subset=[reg_label]).copy()
    if len(calib) < 30:
        return None
    X_calib = np.nan_to_num(calib[cols].values, nan=0.0)
    r_calib = reg_model.predict(X_calib)
    e = calib[reg_label].values - r_calib
    r_test = reg_model.predict(X_test)
    p_reg = prob_from_residuals(r_test, e)

    sd = float(np.nanstd(e))
    if sd > 0:
        p_reg_gauss = np.clip(norm.cdf((r_test - TH) / sd), 1e-6, 1.0 - 1e-6)
    else:
        p_reg_gauss = p_reg

    ret = test[reg_label].values.astype(float)
    up = (ret > TH).astype(float)
    cmp = pd.DataFrame(
        {
            "date": test["date"].values,
            "p_cls": np.asarray(p_cls, dtype=float),
            "p_reg": np.asarray(p_reg, dtype=float),
            "p_reg_gauss": np.asarray(p_reg_gauss, dtype=float),
            "ret": ret,
            "up": up,
        }
    )
    out: dict = {
        "horizon": k,
        "n": int(len(cmp)),
        "up_rate": float(up.mean()),
        "cls_label": cls_label,
        "reg_label": reg_label,
    }
    for name in ("p_cls", "p_reg", "p_reg_gauss"):
        p = cmp[name].values
        out[f"{name}_range"] = [float(np.min(p)), float(np.max(p))]
        out[f"{name}_std"] = float(np.std(p))
        out[f"{name}_ic_ret"] = ICScreener.rank_ic(cmp, name, "ret")
        out[f"{name}_ic_up"] = ICScreener.rank_ic(cmp, name, "up")
        out[f"{name}_ece"] = ece(p, up)
    print(
        f"[{board}] {k}d: p_cls IC_ret={out['p_cls_ic_ret']:+.4f} IC_up={out['p_cls_ic_up']:+.4f} "
        f"ECE={out['p_cls_ece']:.4f} std={out['p_cls_std']:.4f} | "
        f"p_reg IC_ret={out['p_reg_ic_ret']:+.4f} IC_up={out['p_reg_ic_up']:+.4f} "
        f"ECE={out['p_reg_ece']:.4f} std={out['p_reg_std']:.4f}",
        flush=True,
    )
    return out


def main() -> int:
    args = sys.argv[1:]
    boards = ["main", "dual"]
    for i, a in enumerate(args):
        if a == "--boards" and i + 1 < len(args):
            boards = [b.strip() for b in args[i + 1].split(",")]
    t0 = time.time()
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    cut = panel["date"].max() - pd.DateOffset(years=3)
    panel = panel[panel["date"] >= cut]
    print(
        f"[panel] {len(panel):,}r max={panel['date'].max():%Y-%m-%d} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    registry = FeatureRegistry(
        path=os.path.join(REGISTRY_PATH, "feature_registry.json")
    )
    main_df, dual_df = cleaner.run_train(panel)
    del panel
    gc.collect()
    print(
        f"[clean] main={len(main_df):,} dual={len(dual_df):,} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    trainer = DualTrackTrainer(model_dir=MODEL_DIR)
    results: dict[str, list] = {}
    for board, board_df in (("main", main_df), ("dual", dual_df)):
        if board not in boards or len(board_df) == 0:
            continue
        cur = os.path.join(MODEL_DIR, f"{board}_current.pkl")
        bundle = DualTrackTrainer.load(cur)
        df = build_board_df(board_df, features, registry, bundle, board)
        del board_df
        gc.collect()
        segs = trainer.split_window(df)
        del df
        gc.collect()
        trained = {
            "board": board,
            "feature_cols": bundle["feature_cols"],
            "models": bundle["models"],
            "segs": segs,
        }
        trained["calibrators"] = bundle.get("calibrators", {})
        rows = []
        for k in (3, 5, 10):
            row = compare_horizon(board, trained, k)
            if row is not None:
                rows.append(row)
        results[board] = rows
        del segs, trained, bundle
        gc.collect()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[saved] {OUT_JSON}", flush=True)
    print(f"[done] ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
