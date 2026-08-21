"""_diag_wick_wf_train.py — wick_probe 特征变体 walk-forward 训练 (2026-08-19).

A/B 对比 legacy prob_head (208 特征) vs +wick_probe (209 特征):
  baseline 检查点 = data/_diag_legacy_wf_pred_dual_e250.parquet (已有)
  new 检查点     = data/_diag_wick_wf_pred_dual_e250.parquet (本脚本)

数据流与 _diag_legacy_prob_head_replay.py 的 dual walk-forward 分支逐字一致
(LGB_PARAMS / REFIT_EVERY=21 / 训练掩码 idx<pos-4 防前瞻 / 检查点恢复),
唯一差异:
  1. 特征帧注入 wick_probe = 放量(vr>1.5) & 收涨(ret>0) &
     (长上影≥3% | (上影≥2% & 下影≥2%))   — 用户"上影加下影=试盘线"信号
  2. eval_days = WORM replay CSV 的日期集 (2025-07-21..2026-07-30, 250 天)
     → 与 baseline 检查点日期对齐, 候选池/标签直接复用 WORM CSV.

用法: python scripts/_diag_wick_wf_train.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from app.pipeline1 import prob_head
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine, _ensure_sorted
from app.pipeline1.predictor import V35Predictor
from config.settings import DATA_DIR, LEGACY_PROB_GATE, PANEL_V3_PATH

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
REPLAY_CSV = r"D:\AMINQT\DATA OTHERS\diag\legacy_prob_head_replay_20260816_220029.csv"
SLICE = 420
EVAL = 250
REFIT_EVERY = LEGACY_PROB_GATE["refit_every_days"]
ABS_TARGET = LEGACY_PROB_GATE["abs_target"]
CKPT = DATA_DIR / "_diag_wick_wf_pred_dual_e250.parquet"


def _add_wick_probe(feat: pd.DataFrame) -> pd.DataFrame:
    """放量+收涨+(长上影|双影) 试盘线信号 (0/1, T 日收盘可得)."""
    body_hi = feat[["open", "close"]].max(axis=1)
    body_lo = feat[["open", "close"]].min(axis=1)
    upper = (feat["high"] - body_hi) / feat["close"]
    lower = (body_lo - feat["low"]) / feat["close"]
    vma = feat.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    vr = feat["volume"] / vma
    ret = feat.groupby("symbol")["close_hfq"].pct_change()
    signal = (
        (vr > 1.5) & (ret > 0) & ((upper >= 0.03) | ((upper >= 0.02) & (lower >= 0.02)))
    )
    feat["wick_probe"] = signal.astype(float)
    return feat


def _build_raw_labels(dfb: pd.DataFrame) -> pd.DataFrame:
    if "adv20" not in dfb.columns:
        if "amount" not in dfb.columns:
            raise ValueError("清洗帧缺 amount")
        dfb = _ensure_sorted(dfb)
        dfb["adv20"] = (
            dfb.groupby("symbol")["amount"]
            .rolling(20, min_periods=20)
            .mean()
            .reset_index(level=0, drop=True)
        )
    raw = dfb[["symbol", "date", "close_hfq", "high_hfq", "low_hfq", "adv20"]].copy()
    raw["symbol"] = raw["symbol"].astype(str)
    raw = prob_head._add_mfe_3d(raw)
    pain = LabelEngine.build_path_labels(raw)["label_pain"]
    if "is_suspended" in dfb.columns:
        rs = (
            dfb.groupby("symbol")["is_suspended"]
            .rolling(5)
            .sum()
            .reset_index(level=0, drop=True)
        )
        vals = rs.values
        susp = np.zeros(len(vals), dtype=bool)
        if len(vals) > 4:
            susp[: len(vals) - 4] = vals[4:] > 0
        pain = pain.where(~pd.Series(susp, index=rs.index), np.nan)
    raw["label_pain"] = pain
    return raw


def main() -> int:
    t0 = time.time()
    replay = pd.read_csv(REPLAY_CSV)
    eval_days = sorted(pd.unique(pd.to_datetime(replay["date"])))
    print(
        f"[eval] {len(eval_days)} 日 {pd.Timestamp(eval_days[0]).date()}.."
        f"{pd.Timestamp(eval_days[-1]).date()}",
        flush=True,
    )

    predictor = V35Predictor(BUNDLES)
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()

    panel = pd.read_parquet(str(PANEL_V3_PATH))
    dates = sorted(pd.unique(pd.to_datetime(panel["date"])))
    cut = dates[-SLICE]
    panel = panel[pd.to_datetime(panel["date"]) >= cut].reset_index(drop=True)
    print(f"[slice] {pd.Timestamp(cut).date()}.. {len(panel):,}r", flush=True)

    _main_df, dual_df, _state = cleaner.run_inference(panel)
    del panel
    gc.collect()

    cols = predictor.bundles["dual"]["feature_cols"]
    feat = features.build(dual_df, None, inference_cols=cols, cross_sectional_rank=True)
    feat = _add_wick_probe(feat)
    print(
        f"[feat] {len(feat):,}r {len(feat.columns)}c (+wick_probe, {time.time() - t0:.0f}s)",
        flush=True,
    )

    raw = _build_raw_labels(dual_df)
    del dual_df
    gc.collect()

    feat["symbol"] = feat["symbol"].astype(str)
    feat["date"] = pd.to_datetime(feat["date"])
    meta = feat[["symbol", "date"]].reset_index(drop=True)
    feat = feat.merge(
        raw[["symbol", "date", "mfe_3d", "label_pain"]],
        on=["symbol", "date"],
        how="left",
    )
    feat_cols = prob_head.feature_cols(feat)
    assert "wick_probe" in feat_cols, "wick_probe 未被 feature_cols 选中!"
    y = (feat["mfe_3d"] >= ABS_TARGET).astype(float)
    ok = y.notna() & feat["label_pain"].notna()
    x_all = feat[feat_cols].to_numpy(dtype="float32")
    board_dates_arr = np.array(
        pd.to_datetime(np.unique(pd.to_datetime(feat["date"]).values))
    )
    idx = np.searchsorted(board_dates_arr, feat["date"].values)
    ok_arr = ok.to_numpy()
    del feat
    gc.collect()
    print(f"[wf] 特征 {len(feat_cols)} 列 ({time.time() - t0:.0f}s)", flush=True)

    if CKPT.exists():
        cp = pd.read_parquet(str(CKPT))
        print(f"[ckpt] 恢复 {len(cp):,} 行 — 已存在, 直接退出", flush=True)
        return 0

    model = None
    wf_rows: list[pd.DataFrame] = []
    n_refits = 0
    for k, d in enumerate(eval_days):
        pos = int(np.searchsorted(board_dates_arr, np.datetime64(d)))
        if model is None or k % REFIT_EVERY == 0:
            tr = (idx < pos - 4) & ok_arr
            model = LGBMClassifier(**prob_head.LGB_PARAMS)
            model.fit(x_all[tr], y.loc[tr].to_numpy())
            n_refits += 1
        te = idx == pos
        if not te.any():
            continue
        pr = model.predict_proba(x_all[te])[:, 1]
        wf_rows.append(meta.loc[te].assign(pred=pr).reset_index(drop=True))
        if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
            print(
                f"[wf] {k + 1}/{len(eval_days)} (refits={n_refits}, {time.time() - t0:.0f}s)",
                flush=True,
            )

    pd.concat(wf_rows, ignore_index=True).to_parquet(CKPT)
    print(
        f"[wf] 完成 {n_refits} 次重训 → {CKPT.name} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
