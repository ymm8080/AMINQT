"""_diag_time_decay_ab.py — 时间衰减样本加权 A/B (125d 全池 walk-forward, 2026-09-03).

动机 (用户 2026-09-02): 「加进时间权重, 看看有什么收益」— 给概率头训练样本加
指数时间衰减权重 (半衰期 60 交易日), 近期样本权重大, 检验能否改善 regime 漂移
下的 TOP10 质量. 双臂唯一差异 = LGBM fit 的 sample_weight:

  ARM BASE = 无权重 (生产现状)
  ARM TW   = w_i = 0.5 ** ((refit日 − 样本日) / 60)

协议 (对齐 _diag_vp_family_ab.py, 除权重外双臂共享全部环节):
  score = max(狙击, 融合) 截面分位; mag = calibrate_mag10d walk-forward (双臂同);
  prob  = LGBM(label_mfe_10d_net>=0.06) 每 21 交易日扩窗重拟合;
  rank_blend = mag × prob; TOP10 per board/day.

评估: net3/net10, hit3, 赢家重叠, 涨停前捕获, 半窗拆分 — 与 vp A/B 同四闸:
  Δnet3 双半窗同向为必要条件, 涨停前捕获不降为必要条件.

WORM: DATA OTHERS/diag/time_decay_ab_<ts>.parquet + .json
用法:
  python scripts/_diag_time_decay_ab.py                    # slice 420, eval 125
  python scripts/_diag_time_decay_ab.py --slice 260 --eval 10   # 冒烟
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from app.pipeline_parallel.backtest import add_mfe_labels, tradability_gate
from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, PANEL, SNIPER
from app.pipeline_parallel.prob_head import LGB_PARAMS, feature_cols
from app.pipeline_parallel.scoring import pool_score
from config.settings import PANEL_V3_PATH, data_others_path
from scripts._reclassify_all_features import _finalize_slice

COST = 0.0020
PROB_TARGET = 0.06
PROB_REFIT_EVERY = 21
PROB_REFIT_FROM = 40
PROB_MIN_ROWS = 500
TOP_N = 10
LU_THR = {"main": 0.095, "dual": 0.19}
MAX_LAG = 11
HALF_LIFE_DAYS = 60


def _lgbm_walkforward(
    X: np.ndarray,
    y: np.ndarray,
    fit_ok: np.ndarray,
    dv: np.ndarray,
    dates: np.ndarray,
    half_life: int | None,
) -> np.ndarray:
    """扩窗 LGBM: refit 日用 ≤当日已实现行拟合, 应用到 [refit, 下个 refit).

    half_life=None → 无权重 (BASE 臂); 否则指数时间衰减 sample_weight.
    """
    out = np.full(len(dv), np.nan, dtype=float)
    model = None
    for i, d in enumerate(dates):
        if i >= PROB_REFIT_FROM and i % PROB_REFIT_EVERY == 0:
            m = (dv <= d) & fit_ok
            if int(m.sum()) >= PROB_MIN_ROWS:
                model = LGBMClassifier(**LGB_PARAMS)
                if half_life is not None:
                    age_days = (d - dv[m]).astype("timedelta64[D]").astype(int)
                    w = np.power(0.5, age_days / float(half_life))
                    model.fit(X[m], y[m], sample_weight=w)
                else:
                    model.fit(X[m], y[m])
        if model is None:
            continue
        rows = np.nonzero(dv == d)[0]
        if len(rows):
            out[rows] = model.predict_proba(X[rows])[:, 1]
    return out


def _pivots(panel: pd.DataFrame):
    cal = np.sort(
        np.unique(pd.to_datetime(panel["date"].to_numpy()).normalize().to_numpy())
    )
    dt = pd.to_datetime(panel["date"]).dt.normalize()
    px = (
        panel.assign(dt=dt)
        .pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
        .sort_index()
        .reindex(columns=pd.to_datetime(cal))
        .ffill(axis=1)
    )
    px.index = px.index.astype(str).str.zfill(6)
    return px, cal


def _panel_pivots(cutoff):
    panel = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=["symbol", "date", "close_hfq"],
        filters=[("date", ">=", cutoff)],
    )
    return _pivots(panel)


def _net_vec(px: pd.DataFrame, symbols: pd.Series, buy_dt, sell_dt) -> np.ndarray:
    pb = px[buy_dt].reindex(symbols).to_numpy(dtype=float)
    ps = px[sell_dt].reindex(symbols).to_numpy(dtype=float)
    out = ps / pb - 1.0 - COST
    out[~(pb > 0)] = np.nan
    return out


def _top10_metrics(day_rows: pd.DataFrame, winners: set[str]) -> dict:
    net3 = day_rows["net_3d"].dropna()
    return {
        "n": int(len(day_rows)),
        "net3": float(net3.mean()) if len(net3) else np.nan,
        "net10": float(day_rows["net_10d"].dropna().mean()) if len(net3) else np.nan,
        "hit3": float((net3 > 0).mean()) if len(net3) else np.nan,
        "winner_overlap": float(len(set(day_rows["symbol"]) & winners)),
    }


def process_board(board: str, cutoff: pd.Timestamp, eval_n: int, t0: float, px, cal):
    ckpt = PANEL.main_checkpoint if board == "main" else PANEL.dual_checkpoint
    print(f"[{board}] read {ckpt}", flush=True)
    df = pd.read_parquet(ckpt, filters=[("date", ">=", cutoff)])
    df = _finalize_slice(df)
    df = add_mfe_labels(df, horizons=(10,), already_sorted=True)
    df, gate = tradability_gate(df)
    print(f"[{board}] rows {len(df):,} gate -{gate['removed_rows']:,} ({time.time()-t0:.0f}s)", flush=True)
    df["board"] = board
    df["_rid"] = np.arange(len(df))
    base_cols = [c for c in feature_cols(df) if c != "_rid"]

    score_s = pool_score(df, SNIPER.pool)
    score_f = pool_score(df, FUSION.pool)
    scored = df[
        ["_rid", "symbol", "date", "board", "label_pm_10d_net", "label_mfe_10d_net"]
    ].copy()
    scored["score"] = np.maximum(score_s.values, score_f.values)
    del score_s, score_f
    gc.collect()
    scored = scored.dropna(subset=["score"])
    mag = calibrate_mag10d(scored, score_col="score", target_col="label_pm_10d_net")
    scored = scored.merge(mag[["symbol", "date", "mag"]], on=["symbol", "date"], how="inner")
    del mag
    gc.collect()
    print(f"[{board}] scored {len(scored):,}r ({time.time()-t0:.0f}s)", flush=True)

    sub = df.set_index("_rid").loc[scored["_rid"].to_numpy()]
    del df
    gc.collect()
    X = sub[base_cols].to_numpy(dtype="float32")
    del sub
    gc.collect()
    y = (scored["label_mfe_10d_net"].to_numpy(dtype=float) >= PROB_TARGET).astype(int)
    fit_ok = scored["label_mfe_10d_net"].notna().to_numpy()
    dates = np.sort(pd.to_datetime(scored["date"]).unique())
    dv = pd.to_datetime(scored["date"]).to_numpy()
    print(f"[{board}] X {X.shape} fit_rows={int(fit_ok.sum()):,} ({time.time()-t0:.0f}s)", flush=True)

    prob_base = _lgbm_walkforward(X, y, fit_ok, dv, dates, half_life=None)
    prob_tw = _lgbm_walkforward(X, y, fit_ok, dv, dates, half_life=HALF_LIFE_DAYS)
    del X
    gc.collect()

    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}
    day_dates = sorted(pd.unique(pd.to_datetime(scored["date"])))
    eval_days = [d for d in day_dates if d in i_of and i_of[d] + MAX_LAG < len(all_cal)][-eval_n:]
    sym_key = scored["symbol"].astype(str).str.zfill(6)

    pct_px = px.pct_change(axis=1, fill_method=None)
    thr = LU_THR[board]
    lu_days = set()
    mask = pct_px >= thr
    for d in mask.columns:
        for sym in mask.index[mask[d]]:
            lu_days.add((sym, pd.Timestamp(d)))

    frames = []
    for k, d in enumerate(eval_days):
        di = i_of[d]
        idx = np.nonzero((dv == np.datetime64(d)))[0]
        if not len(idx):
            continue
        b1, s3, s10 = all_cal[di + 1], all_cal[di + 4], all_cal[di + 11]
        day = pd.DataFrame(
            {
                "symbol": sym_key.iloc[idx].to_numpy(),
                "score": scored["score"].to_numpy()[idx],
                "mag": scored["mag"].to_numpy()[idx],
                "prob_base": prob_base[idx],
                "prob_tw": prob_tw[idx],
            }
        )
        day["net_3d"] = _net_vec(px, day["symbol"], b1, s3)
        day["net_10d"] = _net_vec(px, day["symbol"], b1, s10)
        day["date"] = str(pd.Timestamp(d).date())
        day["board"] = board
        frames.append(day)
        if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
            print(f"[{board}] eval {k+1}/{len(eval_days)} ({time.time()-t0:.0f}s)", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res["rank_blend_base"] = res["mag"] * res["prob_base"]
    res["rank_blend_tw"] = res["mag"] * res["prob_tw"]
    del scored, prob_base, prob_tw
    gc.collect()

    w0 = i_of[eval_days[0]]
    lu_all = []
    for arm in ("base", "tw"):
        col = f"rank_blend_{arm}"
        by_day = {d: set(sub_df.nlargest(TOP_N, col)["symbol"]) for d, sub_df in res.groupby("date")}
        captured, leads, missed, events = 0, [], 0, 0
        for (sym, L) in lu_days:
            Li = i_of.get(pd.Timestamp(L))
            if Li is None or not (w0 + 3 <= Li <= i_of[eval_days[-1]]):
                continue
            events += 1
            memb = {
                off: (sym in by_day.get(str(all_cal[Li - off].date()), set()))
                for off in (1, 2, 3)
            }
            if any(memb.values()):
                captured += 1
                leads.append(min(o for o, v in memb.items() if v))
            else:
                missed += 1
        lu_all.append(
            {
                "arm": arm,
                "board": board,
                "lu_events": events,
                "captured_before": captured,
                "capture_rate": round(captured / events, 4) if events else None,
                "mean_lead_days": round(float(np.mean(leads)), 2) if leads else None,
                "missed": missed,
            }
        )
    return res, lu_all


def main() -> int:
    global HALF_LIFE_DAYS
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420)
    ap.add_argument("--eval", type=int, default=125)
    ap.add_argument("--half-life", type=int, default=HALF_LIFE_DAYS)
    args = ap.parse_args()
    HALF_LIFE_DAYS = args.half_life
    t0 = time.time()
    print(f"[half-life] {HALF_LIFE_DAYS}d", flush=True)

    dts = pd.read_parquet(PANEL.main_checkpoint, columns=["date"])["date"].unique()
    cutoff = np.sort(pd.to_datetime(pd.Series(dts)))[-args.slice]
    print(f"[cutoff] {pd.Timestamp(cutoff).date()} slice={args.slice}", flush=True)
    px, cal = _panel_pivots(cutoff)
    print(f"[panel-pivot] symbols={len(px)} days={len(cal)} ({time.time()-t0:.0f}s)", flush=True)
    all_cal = pd.to_datetime(cal)

    out_frames, lu_all = [], []
    for board in ("main", "dual"):
        res, ev = process_board(board, cutoff, args.eval, t0, px, all_cal)
        out_frames.append(res)
        lu_all.extend(ev)
        del res
        gc.collect()
    res = pd.concat(out_frames, ignore_index=True)
    del out_frames
    gc.collect()

    days = sorted(res["date"].unique())
    half = len(days) // 2
    wins = {
        (board, d): set(
            res[(res["date"] == d) & (res["board"] == board)]
            .nlargest(TOP_N, "net_3d")["symbol"]
        )
        for board in ("main", "dual")
        for d in days
    }
    summary = {"arms": {}}
    for arm in ("base", "tw"):
        col = f"rank_blend_{arm}"
        rows = []
        for d in days:
            dd = res[res["date"] == d]
            for board in ("main", "dual"):
                sub = dd[dd["board"] == board].nlargest(TOP_N, col).copy()
                sub["arm"], sub["d"] = arm, d
                rows.append(sub)
        sel = pd.concat(rows, ignore_index=True)
        for board in ("main", "dual"):
            b = sel[sel["board"] == board]
            per_day = []
            for d in days:
                bd = b[b["date"] == d]
                m = _top10_metrics(bd, wins[(board, d)])
                m["date"] = d
                per_day.append(m)
            pdf = pd.DataFrame(per_day)
            h1, h2 = pdf.iloc[:half], pdf.iloc[half:]
            summary["arms"].setdefault(arm, {})[board] = {
                "net3_mean": float(pdf["net3"].mean()),
                "net10_mean": float(pdf["net10"].mean()),
                "hit3_mean": float(pdf["hit3"].mean()),
                "winner_overlap_mean": float(pdf["winner_overlap"].mean()),
                "net3_h1": float(h1["net3"].mean()),
                "net3_h2": float(h2["net3"].mean()),
                "days": int(len(pdf)),
            }

    deltas = {}
    for board in ("main", "dual"):
        b_ = summary["arms"]["base"][board]
        f_ = summary["arms"]["tw"][board]
        deltas[board] = {
            "d_net3": round(f_["net3_mean"] - b_["net3_mean"], 5),
            "d_net3_h1": round(f_["net3_h1"] - b_["net3_h1"], 5),
            "d_net3_h2": round(f_["net3_h2"] - b_["net3_h2"], 5),
            "d_hit3": round(f_["hit3_mean"] - b_["hit3_mean"], 5),
            "d_winners": round(f_["winner_overlap_mean"] - b_["winner_overlap_mean"], 4),
        }
    summary["deltas_tw_minus_base"] = deltas
    summary["limitup_capture"] = lu_all
    summary["gate"] = {
        "necessary": [
            "d_net3_h1 与 d_net3_h2 同向且 >0 (至少一板, 另一板不负)",
            "tw 涨停前捕获率 >= base (同板)",
        ],
        "note": "必要条件全过才进人工终审; 单板驱动/半窗翻面即判死",
    }

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pq_path = out_dir / f"time_decay_ab_{ts}.parquet"
    res.to_parquet(pq_path, index=False)
    meta = {
        "ts": ts,
        "slice": args.slice,
        "eval": args.eval,
        "cost": COST,
        "prob_target": PROB_TARGET,
        "half_life_days": HALF_LIFE_DAYS,
        "protocol_deviations": [
            "prob=池内 LGBM 扩窗重拟合 (10d/6% 靶), 生产=3d/3% 靶+OOS 拟合",
            "EMA/迟滞/制度门/解禁过滤/LGBM 闸省略 (双臂同担, Δ 可比)",
            "实得价基=原始面板窄读 (与 fullpool_replay 同源)",
        ],
        "summary": summary,
    }
    (out_dir / f"time_decay_ab_{ts}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {pq_path} rows={len(res):,} ({time.time()-t0:.0f}s)", flush=True)
    print(json.dumps({"deltas": deltas, "limitup": lu_all}, ensure_ascii=False, indent=1))
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
