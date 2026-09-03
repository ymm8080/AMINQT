"""_diag_reg_decay_ab.py — legacy 幅度头 (10d_reg) 时间衰减半衰期 A/B (2026-09-03).

用户: 概率头 hl60/hl30 过闸且 30+60 集成更优后, 幅度头同机制补 15/7 档对拍.
臂: base(原 B10 time_weights=半衰期250交易日, HALF_LIFE_DAYS=None) / hl60 / hl30 /
    hl15 / hl7 + 累进集成 ens2=mean(hl60,hl30) / ens3=+hl15 / ens4=+hl7 (预测均值).

配对协议: 同 seed (第二票闸 SEEDS[0], 相位 0) 同 walk-forward (REFIT_EVERY /
    REG_EMBARGO / split_window / _train_one 全生产路径, 唯一衰减开关运行时改
    prob_head.HALF_LIFE_DAYS), 同池同评估日 — 臂间唯一差异 = 样本权重.
    单 seed 配对消窗相位方差 (multi-seed 教训: ±0.04/日 run-to-run 方差来自训练窗
    差 1 日, 配对下逐臂同窗抵消). 判据 (预登记): delta>0 且 4 子窗 ≥3 正 且双半窗
    同号; depths 5/15 作扰动佐证.
池: data/_diag_rankkey_scored_{board}_e125.parquet 经 _load_pool_from_ckpt 重建
    (E7 池口径, realized_net/base_prod 已并, 免 3h 推理); 基线键两根:
    key:reg_base (改动前生产行为, 主基线) 与 key:mag_prod (生产包 pred_ret_10d, 参考).
检查点: data/_diag_regdecay_wf_{arm}_{board}_e{eval}.parquet (臂×板, 崩溃重跑免重训).
WORM: DATA OTHERS/diag/reg_decay_ab_{ts}.csv/.json + *_daily_{ts}.csv

用法:
  python scripts/_diag_reg_decay_ab.py                  # 全量 (slice 420 / eval 125)
  python scripts/_diag_reg_decay_ab.py --slice 120 --eval 20   # 冒烟 (~30min)
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

import numpy as np
import pandas as pd

import app.pipeline_parallel.prob_head as prob_head
from app.pipeline1 import dual_track_trainer as dtt
from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.predictor import V35Predictor
from config.settings import DATA_DIR, PANEL_V3_PATH, data_others_path
from scripts._rankkey_multiseed_sweep import (
    COST,
    REFIT_EVERY,
    REG_EMBARGO,
    REG_HORIZON,
    SEEDS,
    _build_raw_labels,
    _build_realized_pivot,
    _daily_net,
    _daily_topn,
    _load_pool_from_ckpt,
    _reg_labels_from_matrix,
    _sub_means,
    _unpin_deterministic,
)
from scripts._run_guard import find_conflicts

ARMS = (("base", None), ("hl60", 60), ("hl30", 30), ("hl15", 15), ("hl7", 7))
ARM_PRED = {arm: f"pred_{arm}" for arm, _ in ARMS}
SEED = SEEDS[0]

KEYS = (
    "key:mag_prod",
    "key:reg_base",
    "key:hl60",
    "key:hl30",
    "key:hl15",
    "key:hl7",
    "key:ens2",
    "key:ens3",
    "key:ens4",
)
KEY_BASELINE = "key:reg_base"
KEY_LABELS = {
    "key:mag_prod": "mag_prod(生产包 pred_ret_10d, 参考)",
    "key:reg_base": "reg_base(原250交易日权重=改动前生产, 主基线)",
    "key:hl60": "hl60(60自然日半衰)",
    "key:hl30": "hl30",
    "key:hl15": "hl15",
    "key:hl7": "hl7",
    "key:ens2": "ens2=mean(hl60,hl30)",
    "key:ens3": "ens3=mean(hl60,hl30,hl15)",
    "key:ens4": "ens4=mean(hl60,hl30,hl15,hl7)",
}

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)


def _wf_reg_arm(
    feat: pd.DataFrame,
    feat_cols: list[str],
    eval_days: list,
    board_dates_arr: np.ndarray,
    idx: np.ndarray,
    board: str,
    eval_n: int,
    arm: str,
    hl: int | None,
) -> None:
    """单臂 reg 头 walk-forward (生产 _train_one 全路径, 运行时改衰减开关)."""
    ckpt = DATA_DIR / f"_diag_regdecay_wf_{arm}_{board}_e{eval_n}.parquet"
    if ckpt.exists():
        logger.info("[wf:%s/%s] 从检查点恢复 %s", board, arm, ckpt.name)
        return
    prob_head.HALF_LIFE_DAYS = hl
    try:
        trainer = dtt.DualTrackTrainer()
        model = None
        rows: list[pd.DataFrame] = []
        n_refits = 0
        t0 = time.time()
        for k, d in enumerate(eval_days):
            pos = int(np.searchsorted(board_dates_arr, np.datetime64(d)))
            if model is None or k % REFIT_EVERY == 0:
                tr_mask = idx < (pos - REG_EMBARGO)
                segs = trainer.split_window(feat.loc[tr_mask])
                model, _label = trainer._train_one(
                    "10d_reg", segs, feat_cols, board, seed=SEED
                )
                del segs
                gc.collect()
                n_refits += 1
            te = idx == pos
            if not te.any():
                continue
            pred = model.predict(
                np.nan_to_num(feat.loc[te, feat_cols].to_numpy(), nan=0.0)
            )
            rows.append(
                feat.loc[te, ["symbol", "date"]]
                .assign(pred=pred)
                .reset_index(drop=True)
            )
            if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
                logger.info(
                    "[wf:%s/%s] %s/%s (refits=%s, %.0fs)",
                    board,
                    arm,
                    k + 1,
                    len(eval_days),
                    n_refits,
                    time.time() - t0,
                )
        try:
            pd.concat(rows, ignore_index=True).to_parquet(str(ckpt))
        except Exception as exc:
            logger.error("[wf:%s/%s] 检查点写入失败: %s", board, arm, exc)
            raise
        logger.info(
            "[wf:%s/%s] 完成: %s 次重训 → %s (%.0fs)",
            board,
            arm,
            n_refits,
            ckpt.name,
            time.time() - t0,
        )
    finally:
        prob_head.HALF_LIFE_DAYS = None


def _analyze(pool_df: pd.DataFrame, eval_n: int, t0: float) -> int:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)

    rows_out: list[dict] = []
    verdicts: dict = {}
    daily_out: list[pd.DataFrame] = []

    for board in ("main", "dual"):
        b = pool_df[pool_df["board"] == board].copy()
        e7 = b[~b["pain_excluded"].fillna(False)].copy()
        s = e7
        for arm, _ in ARMS:
            ck = pd.read_parquet(
                str(DATA_DIR / f"_diag_regdecay_wf_{arm}_{board}_e{eval_n}.parquet")
            )
            ck["date"] = pd.to_datetime(ck["date"])
            s = s.merge(
                ck.rename(columns={"pred": ARM_PRED[arm]}),
                on=["symbol", "date"],
                how="left",
            )
            miss = float(s[ARM_PRED[arm]].isna().mean())
            logger.info("[merge:%s/%s] pred 缺失率 %.2f%%", board, arm, 100 * miss)
        s["key:mag_prod"] = s["pred_ret_10d"]
        s["key:reg_base"] = s[ARM_PRED["base"]]  # 臂名 base, 判据键名 reg_base
        for arm, _ in ARMS[1:]:
            s[f"key:{arm}"] = s[ARM_PRED[arm]]
        s["key:ens2"] = s[["pred_hl60", "pred_hl30"]].mean(axis=1)
        s["key:ens3"] = s[["pred_hl60", "pred_hl30", "pred_hl15"]].mean(axis=1)
        s["key:ens4"] = s[["pred_hl60", "pred_hl30", "pred_hl15", "pred_hl7"]].mean(
            axis=1
        )

        depths = (5, 10, 15)
        key_series: dict = {}
        for key in KEYS:
            sk = s[s[key].notna()]  # 无预测日不可入围 (_daily_topn 对 NaN 不剔行)
            for depth in depths:
                net = _daily_net(_daily_topn(sk, key, depth))
                key_series[(key, depth)] = net
                r = net.dropna()
                rows_out.append(
                    {
                        "board": board,
                        "key": key,
                        "depth": depth,
                        "n_days": int(net.notna().sum()),
                        "mean": float(r.mean()) if len(r) else np.nan,
                        "hit": float((r > 0).mean()) if len(r) else np.nan,
                        "ge5": float((r >= 0.05).mean()) if len(r) else np.nan,
                    }
                )

        logger.info("\n===== %s (depth=10, %s 评估日, seed=%s 配对) =====", board, len(s["date"].unique()), SEED)
        logger.info("  %s%s%s%s%s  判定", f"{'键':<38}", f"{'日均净':>9}", f"{'delta':>9}", f"{'半窗h1/h2':>16}", f"{'子窗正':>7}")
        verdicts[board] = {}
        bs = key_series[(KEY_BASELINE, 10)]
        for key in KEYS:
            net = key_series[(key, 10)]
            full = float(net.dropna().mean())
            if key == KEY_BASELINE:
                logger.info("  %s%s  基线", f"{KEY_LABELS[key]:<38}", f"{full:>+9.2%}")
                continue
            delta = (net - bs).dropna()
            d = float(delta.mean())
            half = len(delta) // 2
            h1 = float(delta.iloc[:half].mean())
            h2 = float(delta.iloc[half:].mean())
            subs = _sub_means(delta)
            subs_pos = sum(1 for x in subs if np.isfinite(x) and x > 0)
            n_valid = sum(1 for x in subs if np.isfinite(x))
            passed = bool(
                np.isfinite(d) and d > 0 and n_valid > 0 and subs_pos * 4 >= 3 * n_valid
                and np.isfinite(h1) and np.isfinite(h2) and h1 > 0 and h2 > 0
            )
            verdicts[board][key] = {
                "delta": d,
                "h1": h1,
                "h2": h2,
                "subs": subs,
                "subs_pos": subs_pos,
                "n_subs_valid": n_valid,
                "pass": passed,
                "full": full,
            }
            logger.info(
                "  %s%s%s%s/%s%s/%s  %s [子窗 %s]",
                f"{KEY_LABELS[key]:<38}",
                f"{full:>+9.2%}",
                f"{d:>+9.2%}",
                f"{h1:>+7.2%}",
                f"{h2:>+7.2%}",
                subs_pos,
                n_valid,
                "通过" if passed else "不通过",
                "/".join(f"{x:+.2%}" for x in subs),
            )
            daily_out.append(
                pd.DataFrame(
                    {
                        "board": board,
                        "key": key,
                        "date": delta.index,
                        "delta": delta.to_numpy(),
                        "net": net.reindex(delta.index).to_numpy(),
                    }
                )
            )

    try:
        pd.DataFrame(rows_out).to_csv(out_dir / f"reg_decay_ab_{ts}.csv", index=False)
        (out_dir / f"reg_decay_ab_{ts}.json").write_text(
            json.dumps(
                {
                    "ts": ts,
                    "eval": eval_n,
                    "seed": SEED,
                    "refit_every": REFIT_EVERY,
                    "embargo_reg": REG_EMBARGO,
                    "arms": [a for a, _ in ARMS],
                    "halves": {a: v for a, v in ARMS},
                    "keys": KEY_LABELS,
                    "verdict_rule": (
                        "配对单seed: delta>0 且 ≥3/4 子窗正 且双半窗同正 (depth=10, "
                        "基线=key:reg_base 即改动前生产行为)"
                    ),
                    "verdicts": verdicts,
                    "n_pool": len(pool_df),
                    "runtime_s": round(time.time() - t0, 0),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        pd.concat(daily_out, ignore_index=True).to_csv(
            out_dir / f"reg_decay_ab_daily_{ts}.csv", index=False
        )
    except Exception as exc:
        logger.error("WORM 写入失败: %s", exc)
        raise
    logger.info(
        "\n[saved] %s/reg_decay_ab_%s.csv/.json + *_daily_%s.csv (%.0fs)",
        out_dir,
        ts,
        ts,
        time.time() - t0,
    )
    logger.info("=== DONE ===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420, help="面板切片交易日数")
    ap.add_argument("--eval", type=int, default=125, help="评估的已实现决策日数")
    ap.add_argument("--pool-tag", type=int, default=125, help="scored 检查点 eval 标签")
    ap.add_argument(
        "--analyze-only",
        action="store_true",
        help="跳过特征构建与 walk-forward, 只重算分析段 (检查点必须已存在)",
    )
    args = ap.parse_args()

    hits = find_conflicts()
    if hits:
        logger.error("[guard] 存活重活进程冲突, 退出: %s", hits)
        return 2

    _unpin_deterministic()

    if args.analyze_only:
        pool_df, _ = _load_pool_from_ckpt(args.pool_tag)
        pool_df["date"] = pd.to_datetime(pool_df["date"])
        logger.info("[analyze-only] 池 %s 行, 检查点 e%s", f"{len(pool_df):,}", args.eval)
        return _analyze(pool_df, args.eval, time.time())

    t0 = time.time()
    logger.info(
        "[cfg] slice=%s eval=%s pool_tag=%s seed=%s refit_every=%s embargo_reg=%s arms=%s",
        args.slice,
        args.eval,
        args.pool_tag,
        SEED,
        REFIT_EVERY,
        REG_EMBARGO,
        [a for a, _ in ARMS],
    )

    pool_df, base_maps = _load_pool_from_ckpt(args.pool_tag)
    pool_df["date"] = pd.to_datetime(pool_df["date"])
    logger.info("[pool] %s 行 (%.0fs)", f"{len(pool_df):,}", time.time() - t0)

    predictor = V35Predictor(
        {
            "main": "models/pipeline1/main_current.pkl",
            "dual": "models/pipeline1/dual_current.pkl",
        }
    )
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()

    panel = load_panel_v3(path=PANEL_V3_PATH)
    dates_all = sorted(pd.unique(pd.to_datetime(panel["date"])))
    cut = dates_all[-args.slice]
    panel = panel[pd.to_datetime(panel["date"]) >= cut].reset_index(drop=True)
    logger.info(
        "[slice] %s.. %sr (%.0fs)",
        pd.Timestamp(cut).date(),
        f"{len(panel):,}",
        time.time() - t0,
    )
    pivot, cal = _build_realized_pivot(panel)
    logger.info(
        "[pivot] symbols=%s days=%s (%.0fs)",
        len(pivot),
        len(cal),
        time.time() - t0,
    )

    main_df, dual_df, state = cleaner.run_inference(panel)
    logger.info(
        "[clean] valve=%s main=%sr dual=%sr (%.0fs)",
        state,
        f"{len(main_df):,}",
        f"{len(dual_df):,}",
        time.time() - t0,
    )
    del panel
    gc.collect()

    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        pool_b = pool_df[pool_df["board"] == board]
        eval_days = sorted(pool_b["date"].unique())[-args.eval :]
        cols = predictor.bundles[board]["feature_cols"]

        # 与 rankkey 主流程同序: raw 标签 (mfe_3d/label_pain, 训练帧并表同构)
        # → 特征帧 → merge → reg 净标签 (pivot 向量化) → walk-forward
        raw = _build_raw_labels(dfb)
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        feat = feat.reset_index(drop=True)
        feat["symbol"] = feat["symbol"].astype(str)
        feat["date"] = pd.to_datetime(feat["date"])
        del dfb
        gc.collect()
        logger.info(
            "[feat:%s] %sr %sc (%.0fs)",
            board,
            f"{len(feat):,}",
            len(feat.columns),
            time.time() - t0,
        )

        day_dates = sorted(pd.unique(feat["date"]))
        day_set = set(day_dates)
        missing = [d for d in eval_days if d not in day_set]
        if missing:
            logger.error("[%s] 池评估日 %s 不在特征帧, FAIL", board, missing[:3])
            return 2

        feat = feat.merge(
            raw[["symbol", "date", "mfe_3d", "label_pain"]],
            on=["symbol", "date"],
            how="left",
        )
        del raw
        gc.collect()
        price = pivot.to_numpy(dtype="float64")
        sym_rows = pivot.index.get_indexer(feat["symbol"].to_numpy())
        j_cols = np.searchsorted(cal, feat["date"].to_numpy())
        if not np.all(cal[j_cols] == feat["date"].to_numpy()):
            logger.error("[%s] FAIL 特征帧日期不在 pivot 日历中", board)
            return 2
        feat["label_10d_net"] = _reg_labels_from_matrix(
            price, sym_rows, j_cols, COST, horizon=REG_HORIZON
        )
        logger.info(
            "[%s] reg 净标签 %s/%s 行可用 (%.0fs)",
            board,
            f"{feat['label_10d_net'].notna().sum():,}",
            f"{len(feat):,}",
            time.time() - t0,
        )

        board_dates_arr = np.array(pd.to_datetime(day_dates))
        idx = np.searchsorted(board_dates_arr, feat["date"].values)
        for arm, hl in ARMS:
            _wf_reg_arm(
                feat, cols, eval_days, board_dates_arr, idx, board, args.eval, arm, hl
            )
        del feat
        gc.collect()

    del pivot
    gc.collect()
    return _analyze(pool_df, args.eval, t0)


if __name__ == "__main__":
    raise SystemExit(main())
