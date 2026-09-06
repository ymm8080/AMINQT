"""_diag_widepool_valve_replay.py — 放宽 serving 池的 125d 打分回放 (2026-09-06).

背景: 用户问"训练保持流动性闸, 预测端不因 5000万/8000万 不入池, 质量更好吗".
现有 _diag_rankkey_scored_*_e125.parquet 生成时池底=8e7 (step4 阀), 0.5-0.8亿
行数=0, 无法回答. 本脚本用最宽池口径 (读墙 min_amount=0 + 池底 floor=0) 重跑
生产包逐日推理, 落一份带 amount 的宽池打分文件; 离线 A/B 取子池即可同时回答:
  ≥8e7  = 现状池 (sanity 锚, main 应与旧文件同分布)
  ≥5e7  = 撤 8000万 (池底降档)
  全部   = 撤 5000万 (读墙放开, <5e7 行对模型 OOD — 回测直接量其伤害)

口径与 _rankkey_multiseed_sweep 推理段一致: slice 420d → run_inference (全宽
config) → FeatureEngineV35.build(inference_cols) → 35d base_rate 预热 → 125d
逐日 V35Predictor.predict + compute_scores, realized_net=T+10 c2c 净(0.2% 成本).
dual 的 cross_sectional_rank 在宽截面上计算 = 真实放宽 serving 的反事实.
dual 不做 pool_blend_cut (与 sweep 同; main TOP15 是主判据, dual 为次口径).

输出: data/_diag_rankkey_scored_wide_e125.parquet
      (date/board/symbol/amount/pred_ret_10d/prob/base_rate/pain_prob/
       pred_q50_3d/pred_q50_5d/realized_net)

用法:
  python scripts/_diag_widepool_valve_replay.py            # 125d 全量 (~40min)
  python scripts/_diag_widepool_valve_replay.py --slice 180 --eval 40  # 冒烟
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import DATA_DIR, PANEL_V3_PATH
from scripts._rankkey_multiseed_sweep import (
    REALIZED_SELL_LAG,
    _build_realized_pivot,
    _realized_net,
)

REPO = Path(__file__).resolve().parent.parent
BUNDLES = {
    "main": str(REPO / "models" / "pipeline1" / "main_current.pkl"),
    "dual": str(REPO / "models" / "pipeline1" / "dual_current.pkl"),
}
WARM_DAYS = 35
OUT = "_diag_rankkey_scored_wide_e{eval}.parquet"


def _log(msg: str) -> None:
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def main() -> int:
    from scripts._run_guard import find_conflicts

    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420, help="面板切片交易日数")
    ap.add_argument("--eval", type=int, default=125, help="评估已实现决策日数")
    args = ap.parse_args()

    hits = find_conflicts()
    if hits:
        _log(f"[guard] 存活重活进程冲突, 退出: {hits}")
        return 2

    t0 = time.time()

    # ── 1) 宽读取: 撤 5000万读墙 (只保留非停牌), 训练闸不动 — 本脚本只测 serving ──
    filters = (pc.field("amount") >= 0.0) & (
        pc.field("is_suspended").cast(pa.int64()) == 0
    )
    _log(f"[read] 宽读取 {PANEL_V3_PATH} (amount>=0, 非停牌)")
    panel = pq.read_table(str(PANEL_V3_PATH), filters=filters).to_pandas()
    _log(f"[read] {len(panel):,}r max={panel['date'].max()} ({time.time()-t0:.0f}s)")

    dates_all = sorted(pd.unique(pd.to_datetime(panel["date"])))
    cut = dates_all[-args.slice]
    panel = panel[pd.to_datetime(panel["date"]) >= cut].reset_index(drop=True)
    _log(f"[slice] {pd.Timestamp(cut).date()}.. {len(panel):,}r ({time.time()-t0:.0f}s)")

    pivot, cal = _build_realized_pivot(panel)
    _log(f"[pivot] symbols={len(pivot)} days={len(cal)} ({time.time()-t0:.0f}s)")

    # ── 2) 全宽清洗: 读墙与池底都放开 (real counterfactual serving pool) ──
    cleaner = CleaningPipeline(CleaningConfig(min_amount=0.0, abs_amount_floor=0.0))
    main_df, dual_df, state = cleaner.run_inference(panel)
    _log(
        f"[clean] valve={state} main={len(main_df):,}r dual={len(dual_df):,}r "
        f"({time.time()-t0:.0f}s)"
    )
    del panel
    gc.collect()

    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}

    predictor = V35Predictor(BUNDLES)
    features = FeatureEngineV35()
    lister = ListGenerator()

    scored_frames: dict[str, list] = {}
    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        cols = predictor.bundles[board]["feature_cols"]
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        feat = feat.reset_index(drop=True)
        feat["symbol"] = feat["symbol"].astype(str)
        feat["date"] = pd.to_datetime(feat["date"])
        _log(f"[feat:{board}] {len(feat):,}r {len(feat.columns)}c ({time.time()-t0:.0f}s)")

        amt_map = dfb.assign(
            symbol=dfb["symbol"].astype(str), date=pd.to_datetime(dfb["date"])
        ).set_index(["symbol", "date"])["amount"]

        day_dates = sorted(pd.unique(feat["date"]))
        eval_days = [
            d
            for d in day_dates
            if d in i_of and i_of[d] + REALIZED_SELL_LAG < len(all_cal)
        ][-args.eval :]
        _log(
            f"[{board}] eval {len(eval_days)}d "
            f"{pd.Timestamp(eval_days[0]).date()}..{pd.Timestamp(eval_days[-1]).date()}"
        )

        for d in [x for x in day_dates if x < eval_days[0]]:
            day_feat = feat[feat["date"] == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
                if not pred.empty:
                    lister.compute_scores(pred)
            except Exception:
                pass
        _log(f"[{board}] base_rate 预热完成 ({time.time()-t0:.0f}s)")

        for k, d in enumerate(eval_days):
            di = i_of[d]
            day_feat = feat[feat["date"] == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
            except Exception as exc:
                _log(f"[{board}] {pd.Timestamp(d).date()} predict err: {exc}")
                continue
            if pred.empty:
                continue
            scored = lister.compute_scores(pred)
            scored["date"] = d
            scored["board"] = board
            sym_key = list(
                zip(scored["symbol"].astype(str), pd.to_datetime(scored["date"]))
            )
            cp = (
                scored["compound_prob"]
                if "compound_prob" in scored.columns
                else scored["prob_up"]
            )
            cr = (
                scored["compound_ret"]
                if "compound_ret" in scored.columns
                else scored["pred_ret_10d"]
            )
            ing = pd.DataFrame(
                {
                    "date": d,
                    "board": board,
                    "symbol": scored["symbol"].astype(str).to_numpy(),
                    "amount": [float(amt_map.get(sy, np.nan)) for sy in sym_key],
                    "pred_ret_10d": cr.to_numpy(),
                    "prob": cp.to_numpy(),
                    "base_rate": (
                        scored["base_rate"].to_numpy()
                        if "base_rate" in scored.columns
                        else np.nan
                    ),
                    "pain_prob": (
                        scored["pain_prob"].to_numpy()
                        if "pain_prob" in scored.columns
                        else np.nan
                    ),
                    "pred_q50_3d": (
                        scored["pred_q50_3d"].to_numpy()
                        if "pred_q50_3d" in scored.columns
                        else np.nan
                    ),
                    "pred_q50_5d": (
                        scored["pred_q50_5d"].to_numpy()
                        if "pred_q50_5d" in scored.columns
                        else np.nan
                    ),
                    "realized_net": [
                        _realized_net(pivot, cal, di, str(s)) for s in scored["symbol"]
                    ],
                }
            )
            scored_frames.setdefault(board, []).append(ing)
            if (k + 1) % 25 == 0:
                _log(f"[{board}] eval {k+1}/{len(eval_days)} ({time.time()-t0:.0f}s)")

        del feat, dfb
        gc.collect()

    if not scored_frames:
        _log("[save] 无任何打分行 — FAIL")
        return 3
    out = pd.concat(
        [f for rows in scored_frames.values() for f in rows], ignore_index=True
    )
    out_path = DATA_DIR / OUT.format(eval=args.eval)
    out.to_parquet(str(out_path))
    band = pd.cut(
        out["amount"],
        [0, 5e7, 8e7, 1.5e8, np.inf],
        labels=["<0.5亿", "0.5-0.8亿", "0.8-1.5亿", ">=1.5亿"],
    )
    _log(
        f"[save] {out_path} {len(out):,}r 档分布: "
        f"{band.value_counts().to_dict()} ({time.time()-t0:.0f}s)"
    )
    _log("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
