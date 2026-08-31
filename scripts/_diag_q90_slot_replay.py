"""_diag_q90_slot_replay.py — q90 彩票槽长窗回放: 全池逐股双头预测 (2026-08-30).

背景 (08-29 TOP10 质量诊断): 140 只 3d 真赢家 legacy 命中 1; 漏网股 42% 在池内,
均值头只排 26 分位但 q90 分位头排 81 分位 → 彩票槽候选杠杆. 原 14 天稳定性检验
窗口太薄, 本脚本产 125 日全池明细供离线验证 (槽位 A/B / 真赢家覆盖 / 名次衰减 /
闸归因).

与 _diag_legacy_hitrate_topn.py 同构 (清洗→特征→逐日 predict→compute_scores),
区别: 记录**全池**行 (非仅过闸), 附 q90/q75/q50 预测列 + 3d/10d 双视界实得 + 锚日
amount. 闸分量离线按当期 config 重建 (记录原始列, 不在回放内固化闸).

口径警告: 影子回放 — 当前包含训练数据的包重放历史, 绝对水平上偏; 槽位 A/B 两臂
共用同一预测, delta 仍可比但 q90 头对赢家的"记忆"成分无法剔除, 结论须与 08-06 起
诚实生产清单窗 (walk-forward) 互证.

实得 = buy D+1 close / sell D+(1+h) close − COST (与 hitrate 回放同口径).

WORM: DATA_OTHERS/diag/q90_slot_replay_<ts>.parquet + .json (meta)

用法:
  python scripts/_diag_q90_slot_replay.py                # 全量 --slice 420 --eval 125
  python scripts/_diag_q90_slot_replay.py --eval 10      # 冒烟
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gc

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import PANEL_V3_PATH, data_others_path

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
COST = 0.0020  # 与 hitrate 回放一致: 佣金+印花税+滑点 ≈ 0.2% 往返
KEEP_PRED = [
    "symbol", "pred_ret_10d", "pred_ret_3d", "prob_up", "prob_up_10d",
    "base_rate", "pain_prob", "pred_q50_3d", "pred_q50_5d",
    "pred_q75_3d", "pred_q90_3d",
]


def _pivots(panel: pd.DataFrame):
    """symbol×date 宽表: close_hfq (ffill) + amount, 返回 (px, amt, cal)."""
    cal = np.sort(np.unique(pd.to_datetime(panel["date"].to_numpy()).normalize().to_numpy()))
    dt = pd.to_datetime(panel["date"]).dt.normalize()
    px = (
        panel.assign(dt=dt).pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last").sort_index()
        .reindex(columns=pd.to_datetime(cal)).ffill(axis=1)
    )
    amt = (
        panel.assign(dt=dt).pivot_table(index="symbol", columns="dt", values="amount", aggfunc="last").sort_index()
        .reindex(columns=pd.to_datetime(cal))
    )
    return px, amt, cal


def _net_vec(px: pd.DataFrame, symbols: pd.Series, buy_dt, sell_dt) -> np.ndarray:
    """buy/sell 收盘价按 symbol 对齐 → 净收益 (NaN 安全, pb<=0 → NaN)."""
    pb = px[buy_dt].reindex(symbols).to_numpy(dtype=float)
    ps = px[sell_dt].reindex(symbols).to_numpy(dtype=float)
    out = ps / pb - 1.0 - COST
    out[~(pb > 0)] = np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420)
    ap.add_argument("--eval", type=int, default=125)
    args = ap.parse_args()

    t0 = time.time()
    predictor = V35Predictor(BUNDLES)
    features = FeatureEngineV35()
    lister = ListGenerator()

    print(f"[load] panel {PANEL_V3_PATH}", flush=True)
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    print(f"[load] {len(panel):,}r max={panel['date'].max()} ({time.time() - t0:.0f}s)", flush=True)
    dates = sorted(pd.unique(pd.to_datetime(panel["date"])))
    cut = dates[-args.slice]
    panel = panel[pd.to_datetime(panel["date"]) >= cut].reset_index(drop=True)
    print(f"[slice] {pd.Timestamp(cut).date()}.. {len(panel):,}r ({time.time() - t0:.0f}s)", flush=True)

    px, amt, cal = _pivots(panel)
    print(f"[pivot] symbols={len(px)} days={len(cal)} ({time.time() - t0:.0f}s)", flush=True)
    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}
    max_lag = 11  # 10d 实得需 i+11; 3d 只需 i+4

    main_df, dual_df, state = CleaningPipeline(CleaningConfig()).run_inference(panel)
    print(f"[clean] valve={state} main={len(main_df):,} dual={len(dual_df):,} ({time.time() - t0:.0f}s)", flush=True)
    del panel
    gc.collect()

    rows: list[dict] = []
    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        cols = predictor.bundles[board]["feature_cols"]
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        print(f"[feat:{board}] {len(feat):,}r {len(feat.columns)}c ({time.time() - t0:.0f}s)", flush=True)
        del dfb
        gc.collect()

        day_dates = sorted(pd.unique(pd.to_datetime(feat["date"])))
        eval_days = [d for d in day_dates if d in i_of and i_of[d] + max_lag < len(all_cal)][-args.eval:]
        for d in day_dates:
            if d >= eval_days[0]:
                break
            day_feat = feat[pd.to_datetime(feat["date"]) == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
                if not pred.empty:
                    lister.compute_scores(pred)
            except Exception:
                pass
        print(f"[{board}] eval {len(eval_days)}d | {pd.Timestamp(eval_days[0]).date()}..{pd.Timestamp(eval_days[-1]).date()}", flush=True)

        for k, d in enumerate(eval_days):
            di = i_of[d]
            day_feat = feat[pd.to_datetime(feat["date"]) == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
            except Exception as exc:
                print(f"[{board}] {pd.Timestamp(d).date()} predict err: {exc}", flush=True)
                continue
            if pred.empty:
                continue
            scored = lister.compute_scores(pred)
            if "compound_ret" in scored.columns:
                scored["pred_ret_10d"] = scored["compound_ret"]
            if "compound_prob" in scored.columns:
                scored["prob_up"] = scored["compound_prob"]
            have = [c for c in KEEP_PRED if c in scored.columns]
            sub = scored[have].copy()
            sub["symbol"] = sub["symbol"].astype(str).str.zfill(6)
            b1, s3, s10 = all_cal[di + 1], all_cal[di + 4], all_cal[di + 11]
            sub["net_3d"] = _net_vec(px, sub["symbol"], b1, s3)
            sub["net_10d"] = _net_vec(px, sub["symbol"], b1, s10)
            if d in amt.columns:
                sub["amount"] = amt[d].reindex(sub["symbol"]).to_numpy(dtype=float)
            else:
                sub["amount"] = np.nan
            sub["date"] = str(pd.Timestamp(d).date())
            sub["board"] = board
            rows.extend(sub.to_dict("records"))
            if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
                print(f"[{board}] {k + 1}/{len(eval_days)} rows={len(rows):,} ({time.time() - t0:.0f}s)", flush=True)
        del feat
        gc.collect()

    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pq_path = out_dir / f"q90_slot_replay_{ts}.parquet"
    df.to_parquet(pq_path, index=False)
    (out_dir / f"q90_slot_replay_{ts}.json").write_text(
        json.dumps({
            "ts": ts, "slice": args.slice, "eval": args.eval, "cost": COST,
            "bundles": BUNDLES, "rows": int(len(df)),
            "days": int(df["date"].nunique()),
            "range": [str(df["date"].min()), str(df["date"].max())],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {pq_path} rows={len(df):,} ({time.time() - t0:.0f}s)", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
