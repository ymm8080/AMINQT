"""_diag_vp_family_ab.py — 量价配合特征族 A/B (125d 全池 walk-forward, 2026-09-02).

动机 (002881 探针, 2026-09-02): 模型能看见上涨腿 (08-17/18 top 分位) 但分不清
真假腿 — 假腿=缩量上涨 (vol5/vol20 0.69), 真腿=持续放量且穿越前 20 日基准
(0.77→1.25), 量能领先价格 1-3 天. 本 A/B 检验: 给概率头加「量价配合」特征族
能否在涨停前捕获此类票并提升 TOP10 质量.

特征族 (7 列, 全部向后看, t 日 bar 可用于 t 日决策, D+1 买入 — 无前瞻):
  vp_regime      vol5/vol20 量能状态 (基准=前 20 日均值 shift 5, 不含当前 5 日窗)
  vp_regime_sl5  量能状态 5 日斜率
  vp_upexp10     近 10 日「放量上涨日」占比
  vp_dndry10     近 10 日「缩量回调日」占比
  vp_upvratio10  近 10 日上涨日平均量比 (vol / vol20 基准)
  vp_updn_vol10  近 10 日上涨日总量 / 下跌日总量
  vp_dd20_x_reg  距 20 日高回撤 × 量能状态

协议 (对齐 _diag_parallel_fullpool_replay.py, 双臂共享除 prob 特征外全部环节):
  score = max(狙击, 融合) 截面分位分; mag = calibrate_mag10d walk-forward (双臂同);
  prob  = LGBM(label_mfe_10d_net>=0.06) 每 21 交易日扩窗重拟合 (生产 prob_head
          通道的回放化; 生产=3d/3% 靶+OOS 清单拟合, 此处=10d/6% 靶+池内已实现行,
          偏差已在 meta 声明); rank_blend = mag × prob; TOP10 per board/day.
  ARM BASE = 现有特征空间 (vp_ 列剔除); ARM FAM = 现有 + 7 列量价族.

评估指标:
  net3/net10 TOP10 均值, hit3, 真赢家重叠 (当日全池 net3 前 10), 半窗拆分;
  **涨停前捕获** (用户主指标): 对评估窗内每个涨停事件 (close_hfq 日涨幅
  main>=9.5% / dual>=19%), 统计涨停日之前 1-3 交易日该臂 TOP10 已含该股的比例
  与平均领先天数. 判闸: Δnet3 双半窗同向为必要条件, 涨停前捕获不降为必要条件.

WORM: DATA OTHERS/diag/vp_family_ab_<ts>.parquet + .json
用法:
  python scripts/_diag_vp_family_ab.py                    # slice 420, eval 125
  python scripts/_diag_vp_family_ab.py --slice 260 --eval 10   # 冒烟
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
VP_COLS = [
    "vp_regime",
    "vp_regime_sl5",
    "vp_upexp10",
    "vp_dndry10",
    "vp_upvratio10",
    "vp_updn_vol10",
    "vp_dd20_x_reg",
]


def add_vp_family(df: pd.DataFrame) -> pd.DataFrame:
    """量价配合特征族, groupby 向量化 (无逐股循环). 需 volume/close_hfq 列."""
    g = df.groupby("symbol", sort=False)
    vol20_base = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(5)
    )
    vol5 = g["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["vp_regime"] = vol5 / vol20_base
    df["vp_regime_sl5"] = df["vp_regime"] - g["vp_regime"].transform(lambda s: s.shift(5))
    pct = g["close_hfq"].pct_change()
    up, dn = (pct > 0), (pct < 0)
    volexp, voldry = df["volume"] > vol20_base, df["volume"] < vol20_base
    df["_ue"] = (up & volexp).astype(float)
    df["_dd"] = (dn & voldry).astype(float)
    df["vp_upexp10"] = g["_ue"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    df["vp_dndry10"] = g["_dd"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    df["_uvr"] = df["vp_regime"].where(up)
    df["vp_upvratio10"] = g["_uvr"].transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    df["_uv"] = (df["volume"] * up.astype(float)).where(up, 0.0)
    df["_dv"] = (df["volume"] * dn.astype(float)).where(dn, 0.0)
    uv = g["_uv"].transform(lambda s: s.rolling(10, min_periods=3).sum())
    dvv = g["_dv"].transform(lambda s: s.rolling(10, min_periods=3).sum())
    df["vp_updn_vol10"] = uv / dvv.replace(0.0, np.nan)
    hh = g["close_hfq"].transform(lambda s: s.rolling(20, min_periods=10).max())
    df["vp_dd20_x_reg"] = (df["close_hfq"] / hh - 1.0) * df["vp_regime"]
    return df.drop(columns=["_ue", "_dd", "_uvr", "_uv", "_dv"])


def _lgbm_walkforward(
    X: np.ndarray, y: np.ndarray, fit_ok: np.ndarray, dv: np.ndarray, dates: np.ndarray
) -> np.ndarray:
    """扩窗 LGBM: refit 日用 ≤当日已实现行拟合, 应用到 [refit, 下个 refit)."""
    out = np.full(len(dv), np.nan, dtype=float)
    model = None
    for i, d in enumerate(dates):
        if i >= PROB_REFIT_FROM and i % PROB_REFIT_EVERY == 0:
            m = (dv <= d) & fit_ok
            if int(m.sum()) >= PROB_MIN_ROWS:
                model = LGBMClassifier(**LGB_PARAMS)
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
    df = add_vp_family(df)
    df["_rid"] = np.arange(len(df))
    base_cols = [
        c for c in feature_cols(df) if not c.startswith("vp_") and c != "_rid"
    ]
    print(f"[{board}] feat base={len(base_cols)} fam+{len(VP_COLS)} ({time.time()-t0:.0f}s)", flush=True)

    score_s = pool_score(df, SNIPER.pool)
    score_f = pool_score(df, FUSION.pool)
    scored = df[
        ["_rid", "symbol", "date", "board", "label_pm_10d_net", "label_mfe_10d_net"]
    ].copy()
    scored["score"] = np.maximum(score_s.values, score_f.values)
    del score_s, score_f
    scored = scored.dropna(subset=["score"])
    mag = calibrate_mag10d(scored, score_col="score", target_col="label_pm_10d_net")
    scored = scored.merge(mag[["symbol", "date", "mag"]], on=["symbol", "date"], how="inner")
    del mag
    gc.collect()
    print(f"[{board}] scored {len(scored):,}r ({time.time()-t0:.0f}s)", flush=True)

    sub = df.set_index("_rid").loc[scored["_rid"].to_numpy()]
    del df
    gc.collect()
    X_fam_part = sub[VP_COLS].to_numpy(dtype="float32")
    X_base = sub[base_cols].to_numpy(dtype="float32")
    del sub
    gc.collect()
    X_fam = np.hstack([X_base, X_fam_part])
    del X_fam_part
    gc.collect()
    y = (scored["label_mfe_10d_net"].to_numpy(dtype=float) >= PROB_TARGET).astype(int)
    fit_ok = scored["label_mfe_10d_net"].notna().to_numpy()
    dates = np.sort(pd.to_datetime(scored["date"]).unique())
    dv = pd.to_datetime(scored["date"]).to_numpy()
    print(f"[{board}] X_base {X_base.shape} fit_rows={int(fit_ok.sum()):,} ({time.time()-t0:.0f}s)", flush=True)

    prob_base = _lgbm_walkforward(X_base, y, fit_ok, dv, dates)
    del X_base
    gc.collect()
    prob_fam = _lgbm_walkforward(X_fam, y, fit_ok, dv, dates)
    del X_fam
    gc.collect()

    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}
    day_dates = sorted(pd.unique(pd.to_datetime(scored["date"])))
    eval_days = [d for d in day_dates if d in i_of and i_of[d] + MAX_LAG < len(all_cal)][-eval_n:]
    sym_key = scored["symbol"].astype(str).str.zfill(6)

    # 涨停日集合 (close_hfq 日涨幅, board 阈值); axis=1 = 逐交易日, px 已 ffill 无内部 NaN
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
                "prob_fam": prob_fam[idx],
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
    res["rank_blend_fam"] = res["mag"] * res["prob_fam"]
    del scored, prob_base, prob_fam
    gc.collect()

    # 涨停前捕获: 事件日 L ∈ eval 窗且 L-1..L-3 都在窗内, 检查各臂 TOP10 成员
    w0 = i_of[eval_days[0]]
    lu_all = []
    for arm in ("base", "fam"):
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
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420)
    ap.add_argument("--eval", type=int, default=125)
    args = ap.parse_args()
    t0 = time.time()

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

    # TOP10 指标 per arm/board + 半窗
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
    for arm in ("base", "fam"):
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
        f_ = summary["arms"]["fam"][board]
        deltas[board] = {
            "d_net3": round(f_["net3_mean"] - b_["net3_mean"], 5),
            "d_net3_h1": round(f_["net3_h1"] - b_["net3_h1"], 5),
            "d_net3_h2": round(f_["net3_h2"] - b_["net3_h2"], 5),
            "d_hit3": round(f_["hit3_mean"] - b_["hit3_mean"], 5),
            "d_winners": round(f_["winner_overlap_mean"] - b_["winner_overlap_mean"], 4),
        }
    summary["deltas_fam_minus_base"] = deltas
    summary["limitup_capture"] = lu_all
    summary["gate"] = {
        "necessary": [
            "d_net3_h1 与 d_net3_h2 同向且 >0 (至少一板, 另一板不负)",
            "fam 涨停前捕获率 >= base (同板)",
        ],
        "note": "必要条件全过才进人工终审; 单板驱动/半窗翻面即判死",
    }

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pq_path = out_dir / f"vp_family_ab_{ts}.parquet"
    res.to_parquet(pq_path, index=False)
    meta = {
        "ts": ts,
        "slice": args.slice,
        "eval": args.eval,
        "cost": COST,
        "prob_target": PROB_TARGET,
        "vp_cols": VP_COLS,
        "protocol_deviations": [
            "prob=池内 LGBM 扩窗重拟合 (10d/6% 靶), 生产=3d/3% 靶+OOS 拟合",
            "EMA/迟滞/制度门/解禁过滤/LGBM 闸省略 (双臂同担, Δ 可比)",
            "vp_ 族在检查点清洗行上滚动 (gate 剔行→窗口跨日历缺口), 基线特征为构建期全史",
            "实得价基=原始面板窄读 (与 fullpool_replay 同源)",
        ],
        "summary": summary,
    }
    (out_dir / f"vp_family_ab_{ts}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {pq_path} rows={len(res):,} ({time.time()-t0:.0f}s)", flush=True)
    print(json.dumps({"deltas": deltas, "limitup": lu_all}, ensure_ascii=False, indent=1))
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
