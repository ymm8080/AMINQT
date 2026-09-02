"""_diag_q50_ensemble_ab.py — q50 多 seed 中位集成 A/B (2026-09-02).

背景: E7 闸3 读 pred_q50_3d/5d>0; retrain_20260903_ms 实测 q50 早停 1 树 →
地板 30 树盲重训仍贴零摆动随机翻闸 (002295 q50_3d=-0.07% 挡门外). 修复 =
QUANTILE_ENSEMBLE (q50 按 3 seed 独立重训, 推理取中位; 同
LEGACY_TOP10_SECOND_VOTE multi_seed 机制, 成员多样性来自重训级浮点非确定).

按生产重训同配方重建数据 (panel 3y → run_train 清洗 → 推理列特征 + 标签 →
split_window segs), 在 test 段对比:
  single = 主 seed 单模型 (修复前生产行为) / m1..m3 = 各成员 / ens = 成员中位 (修复后)
量: 成员离散度与判词分歧 / 闸3 翻闸率与 margin 带 / 翻闸股实得
(label_pm_{k}d_net) / 日过闸数与零过闸日.

口径: test 段在训练窗内 (非 walk-forward 绝对收益), 两臂同窗同数据, delta 可比;
绝对水平含"训练记忆"上偏, 结论只取相对方向.

WORM: DATA OTHERS/diag/q50_ensemble_ab_<ts>.json + q50_ensemble_ab_rows_<ts>.parquet

用法:
  python scripts/_diag_q50_ensemble_ab.py                # main 板
  python scripts/_diag_q50_ensemble_ab.py --board dual   # dual 板
  python scripts/_diag_q50_ensemble_ab.py --board both
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import joblib
import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.dual_track_trainer import (
    LGB_PARAMS_REG,
    QUANTILE_ES_PATIENCE,
    WINDOW_TOTAL,
    DualTrackTrainer,
    risk_filter,
)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import MASK_RECENT_DAYS, LabelEngine
from app.pipeline1.quantile_models import QuantileModelSet
from app.pipeline1.ram_guard import check_startup_gate, start_monitor
from app.pipeline1.train_runner import (
    LEGACY_MKT_EXPECT_WINDOW,
    compute_mkt_expected,
    demean_excess_labels,
)
from config.settings import (
    LEGACY_EXCESS_LABEL_BOARDS,
    PANEL_V3_PATH,
    RETRAIN_RAM_GUARD_MIN_FREE_GB,
    RETRAIN_RAM_GUARD_POLL_S,
    data_others_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("_diag_q50_ensemble_ab")

MODEL_DIR = "models/pipeline1"


def _board_frame(board: str, board_df: pd.DataFrame, features: FeatureEngineV35):
    """推理列特征 + 标签 (复刻 prepare_board_frame, 但 build 用 bundle 推理列省内存)."""
    meta = json.load(
        open(os.path.join(MODEL_DIR, "current_meta.json"), encoding="utf-8")
    )
    bundle = joblib.load(os.path.join(MODEL_DIR, meta[board]["file"]))
    cols = list(bundle["feature_cols"])
    tag = meta[board]["tag"]
    del bundle
    gc.collect()
    logger.info("[%s] bundle=%s 推理列 %d", board, tag, len(cols))

    use_xrank = board == "dual"  # 仅双创开截面排名 (与训练一致)
    df = features.build(
        board_df, None, cross_sectional_rank=use_xrank, inference_cols=cols
    )
    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    if board in LEGACY_EXCESS_LABEL_BOARDS:
        mkt = compute_mkt_expected(df, LEGACY_MKT_EXPECT_WINDOW)
        df = demean_excess_labels(df)
        df.attrs["mkt_expected"] = mkt
        logger.info("[%s] 超额标签去均值: %s", board, mkt)
    return df, cols, tag


def _xy(seg: pd.DataFrame, label: str, cols: list[str]):
    sub = risk_filter(seg.dropna(subset=[label]))
    cols_present = [c for c in cols if c in sub.columns]
    if cols_present:
        sub[cols_present] = sub[cols_present].astype("float32", copy=False)
    X = np.nan_to_num(sub[cols_present].to_numpy(dtype=np.float32), nan=0.0)
    return sub, X, sub[label].to_numpy()


def _train_qset(board: str, horizon: int, segs: dict, cols: list[str], trainer):
    """按 _train_extras 同配方训练 qset (q50 经 QUANTILE_ENSEMBLE = 3 成员)."""
    label = next(
        (
            c
            for c in (
                f"label_pm_{horizon}d_net",
                f"label_{horizon}d_net",
                f"label_{horizon}d",
            )
            if c in segs["train"].columns
        ),
        None,
    )
    if label is None:
        logger.warning("[%s] %dd 标签缺失, 跳过", board, horizon)
        return None, None

    train, X, y = _xy(segs["train"], label, cols)
    _, X_es, y_es = _xy(segs["es"], label, cols)
    params = {k: v for k, v in LGB_PARAMS_REG.items() if k != "objective"}
    t0 = time.time()
    qset = QuantileModelSet(params).fit(
        X,
        y,
        sample_weight=trainer.time_weights(train),
        eval_set=(X_es, y_es) if len(y_es) else None,
        es_patience=QUANTILE_ES_PATIENCE,
    )
    logger.info(
        "[%s] q50_%dd qset 训练完成 %.0fs (样本 %d, 成员 %d)",
        board,
        horizon,
        time.time() - t0,
        len(y),
        len(qset.ensemble_members.get(0.50, [])),
    )

    test_sub, X_t, _ = _xy(segs["test"], label, cols)
    members = qset.ensemble_members.get(0.50) or [qset.models[0.50]]
    pm = np.column_stack([m.predict(X_t) for m in members])
    out = test_sub[["symbol", "date"]].copy()
    out["q50_single"] = pm[:, 0]
    for i in range(pm.shape[1]):
        out[f"q50_m{i + 1}"] = pm[:, i]
    out["q50_ens"] = np.median(pm, axis=1)
    out["label"] = test_sub[label].to_numpy()
    out["horizon"] = horizon
    return out, len(y)


def analyze(board: str, frames: list[pd.DataFrame]) -> dict:
    """闸3 (3d>0 & 5d>0) 三臂对比 + 翻闸实得."""
    wide = frames[0][["symbol", "date", "q50_single", "q50_ens", "label"]].rename(
        columns={
            "q50_single": "s3",
            "q50_ens": "e3",
            "label": "lab3",
        }
    )
    mcols3 = [c for c in frames[0].columns if c.startswith("q50_m")]
    mcols5 = [c for c in frames[1].columns if c.startswith("q50_m")]
    w3 = frames[0][["symbol", "date"] + mcols3].rename(
        columns={c: c + "_3" for c in mcols3}
    )
    wide = wide.merge(w3, on=["symbol", "date"])
    w5 = frames[1][
        ["symbol", "date", "q50_single", "q50_ens", "label"] + mcols5
    ].rename(
        columns={
            "q50_single": "s5",
            "q50_ens": "e5",
            "label": "lab5",
            **{c: c + "_5" for c in mcols5},
        }
    )
    wide = wide.merge(w5, on=["symbol", "date"])

    ps = (wide["s3"] > 0) & (wide["s5"] > 0)
    pe = (wide["e3"] > 0) & (wide["e5"] > 0)
    to_pass = pe & ~ps
    to_block = ps & ~pe
    margin = np.minimum(wide["s3"].abs(), wide["s5"].abs())

    def _realized(mask, col):
        v = wide.loc[mask, col]
        return float(v.mean()) if v.notna().any() else None

    # 成员判词分歧 (每 horizon 任一成员翻符号的比率) + 行内 std
    m3 = wide[[c + "_3" for c in mcols3]].to_numpy()
    m5 = wide[[c + "_5" for c in mcols5]].to_numpy()
    dis3 = float(np.mean(np.ptp(np.sign(m3), axis=1) != 0))
    dis5 = float(np.mean(np.ptp(np.sign(m5), axis=1) != 0))
    spread3 = float(np.mean(np.std(m3, axis=1)))
    spread5 = float(np.mean(np.std(m5, axis=1)))

    daily_s = ps.groupby(wide["date"]).sum()
    daily_e = pe.groupby(wide["date"]).sum()

    return {
        "rows": int(len(wide)),
        "days": int(wide["date"].nunique()),
        "member_sign_disagreement_3d": dis3,
        "member_sign_disagreement_5d": dis5,
        "member_mean_spread_3d": spread3,
        "member_mean_spread_5d": spread5,
        "pass_single": int(ps.sum()),
        "pass_ens": int(pe.sum()),
        "flip_to_pass": int(to_pass.sum()),
        "flip_to_block": int(to_block.sum()),
        "flip_margin_lt_0.5pp": int((margin[to_pass | to_block] < 0.005).sum()),
        "realized_3d_flip_to_pass": _realized(to_pass, "lab3"),
        "realized_3d_flip_to_block": _realized(to_block, "lab3"),
        "realized_3d_pass_single": _realized(ps, "lab3"),
        "realized_3d_pass_ens": _realized(pe, "lab3"),
        "realized_5d_flip_to_pass": _realized(to_pass, "lab5"),
        "realized_5d_flip_to_block": _realized(to_block, "lab5"),
        "zero_pass_days_single": int((daily_s == 0).sum()),
        "zero_pass_days_ens": int((daily_e == 0).sum()),
        "mean_pass_per_day_single": float(daily_s.mean()),
        "mean_pass_per_day_ens": float(daily_e.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=("main", "dual", "both"), default="main")
    args = ap.parse_args()
    boards = ("main", "dual") if args.board == "both" else (args.board,)

    check_startup_gate(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3)
    start_monitor(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3, RETRAIN_RAM_GUARD_POLL_S)

    t0 = time.time()
    logger.info("[load] panel %s", PANEL_V3_PATH)
    panel = load_panel_v3(path=str(PANEL_V3_PATH))
    cut = panel["date"].max() - pd.DateOffset(years=3)
    panel = panel[panel["date"] >= cut]
    logger.info("[load] 3y 窗 %d rows (%.0fs)", len(panel), time.time() - t0)

    board_dfs = dict(zip(("main", "dual"), CleaningPipeline().run_train(panel)))
    del panel
    gc.collect()

    trainer = DualTrackTrainer(model_dir=MODEL_DIR)
    features = FeatureEngineV35()
    all_rows: list[pd.DataFrame] = []
    report: dict = {"ts": time.strftime("%Y%m%d_%H%M%S"), "boards": {}}
    for board in boards:
        bdf = board_dfs.pop(board)
        if not len(bdf):
            logger.warning("[%s] 清洗后空, 跳过", board)
            continue
        df, cols, tag = _board_frame(board, bdf, features)
        del bdf
        gc.collect()
        segs = trainer.split_window(df, WINDOW_TOTAL)
        del df
        gc.collect()
        logger.info(
            "[%s] segs train=%d es=%d test=%d (%.0fs)",
            board,
            len(segs["train"]),
            len(segs["es"]),
            len(segs["test"]),
            time.time() - t0,
        )

        frames = []
        for horizon in (3, 5):
            out, n = _train_qset(board, horizon, segs, cols, trainer)
            if out is not None:
                out["board"] = board
                frames.append(out)
                all_rows.append(out)
                logger.info("[%s] q50_%dd test 行 %d", board, horizon, len(out))
        del segs
        gc.collect()
        if len(frames) == 2:
            report["boards"][board] = {"bundle_tag": tag, **analyze(board, frames)}
            logger.info("[%s] A/B: %s", board, json.dumps(report["boards"][board]))

    if not all_rows:
        logger.error("无可用结果")
        return 1

    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    ts = report["ts"]
    pd.concat(all_rows, ignore_index=True).to_parquet(
        out_dir / f"q50_ensemble_ab_rows_{ts}.parquet", index=False
    )
    (out_dir / f"q50_ensemble_ab_{ts}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[done] WORM %s (%.0fs)", ts, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
