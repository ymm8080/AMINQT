"""_diag_preflush_detector.py — 放量下跌预测器 (④镜像, 2026-09-03).

动机 (用户 09-03): 「怎么预测放量下跌」— ⑤已证放量下跌偏延续 (42% 五日内再大跌,
A4 高分组的放量下跌 65% 再大跌). 若能事前识别谁将放量下跌, 可用作回避过滤
(A股无便利做空, 只做减法).

事件: 放量下跌日 = close_hfq 跌幅 <= -5% 且 turnover_rate 截面秩 >= 0.80.
六臂 (⑤证据+审计映射, 无扫参, 全为日频可计算):
  F1_hotmom   = bias_20/60/120/250 秩均值 (高位股)
  F2_hotvol   = amount/turnover_rate/volume_ratio/ma_vol_ratio_5_20 秩均值 (量能热)
  F3_combined = mean(F1, F2)
  F4_runup    = mean(F1, peak_roc_20d 秩) — 近 20 日冲高速度
  S1_shape    = T-1 日形状坏分秩: 大振幅+长上影+冲高回落+弱收 (⑤"延续/出货"向)
  F5_heatshape = mean(F1, F2, S1) — 热度×形状合成 (09-03 用户: 判断纳入日线形状)

评估 (近 250 交易日, PIT):
  recall   未来 5 日出现放量下跌的事件中, 此前 5 日内曾入该臂 top10% 的比例
  precision 入 top10% 的股票 5 日内放量下跌概率 vs 全池基率
  fwd3     top10% 成员次日买入 3 日净收益 (应为负 = 风险确认)
  半窗拆分 + 当日 top15 风险清单 (asof 面板最后交易日)

WORM: DATA OTHERS/diag/preflush_detector_<ts>.parquet + .json
用法: python scripts/_diag_preflush_detector.py [--days 550] [--eval 250]
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

from config.settings import PANEL_V3_PATH, data_others_path

COST = 0.0020
FLUSH_THR = -0.05
VOL_Q = 0.80
TOP_Q = 0.10
HORIZON = 5

MOM_COLS = ("bias_20", "bias_60", "bias_120", "bias_250")
HOTVOL_COLS = ("amount", "turnover_rate", "volume_ratio", "ma_vol_ratio_5_20")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=550)
    ap.add_argument("--eval", type=int, default=250)
    args = ap.parse_args()
    t0 = time.time()

    dts = pd.read_parquet(PANEL_V3_PATH, columns=["date"])["date"].unique()
    cal_all = np.sort(pd.to_datetime(pd.Series(dts)).unique())
    cutoff = cal_all[-args.days]
    print(f"[cutoff] {pd.Timestamp(cutoff).date()} days={args.days}", flush=True)

    read_cols = [
        "symbol",
        "date",
        "close_hfq",
        "open",
        "high",
        "low",
        "close",
        *MOM_COLS,
        *HOTVOL_COLS,
        "peak_roc_20d",
    ]  # turnover_rate ∈ HOTVOL_COLS, 勿重复
    panel = pd.read_parquet(
        str(PANEL_V3_PATH), columns=read_cols, filters=[("date", ">=", cutoff)]
    )
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    dt = pd.to_datetime(panel["date"]).dt.normalize()
    cal = np.sort(dt.unique())
    symbols = np.sort(panel["symbol"].unique())
    n_days = len(cal)
    print(
        f"[panel] {len(symbols)} syms x {n_days} days ({time.time() - t0:.0f}s)",
        flush=True,
    )

    px_wide = (
        panel.assign(d=dt)
        .pivot_table(index="symbol", columns="d", values="close_hfq", aggfunc="last")
        .sort_index()
        .reindex(index=symbols, columns=pd.DatetimeIndex(cal))
    )
    px = px_wide.ffill(axis=1).to_numpy(dtype="float64")
    del px_wide
    gc.collect()
    pct = px / np.roll(px, 1, axis=1) - 1.0
    pct[:, 0] = np.nan
    fwd3 = px[:, 4:] / px[:, 1:-3] - 1.0 - COST
    n_eval = fwd3.shape[1]

    rank_mats: dict[str, np.ndarray] = {}
    for col in [*MOM_COLS, *HOTVOL_COLS, "peak_roc_20d"]:
        df = pd.DataFrame({"s": panel["symbol"], "d": dt, "v": panel[col]})
        df["rk"] = df.groupby("d")["v"].rank(pct=True)
        rank_mats[col] = (
            df.pivot(index="s", columns="d", values="rk")
            .sort_index()
            .to_numpy("float32")
        )
        del df
    turn_rk = rank_mats["turnover_rate"]
    flush = (pct <= FLUSH_THR) & (turn_rk >= VOL_Q)
    flush[:, 0] = False

    # ---- 日线形状衍生 (用户 09-03: 判断纳入日线形状, 方向均取自⑤事件研究"延续/出货"向) ----
    def _raw_pivot(col: str) -> np.ndarray:
        w = (
            panel.assign(d=dt)
            .pivot_table(index="symbol", columns="d", values=col, aggfunc="last")
            .sort_index()
            .reindex(index=symbols, columns=pd.DatetimeIndex(cal))
        )
        return w.to_numpy(dtype="float64")

    O = _raw_pivot("open")
    H = _raw_pivot("high")
    L = _raw_pivot("low")
    C = _raw_pivot("close")
    prevC = np.roll(C, 1, axis=1)
    prevC[:, 0] = np.nan
    rng = H - L
    with np.errstate(invalid="ignore", divide="ignore"):
        shapes = {
            "amp_pct": (H - L) / prevC,
            "upper_wick": np.where(rng > 0, (H - np.maximum(O, C)) / rng, np.nan),
            "spike_rev": C / H - 1.0,
            "close_pos": np.where(rng > 0, (C - L) / rng, np.nan),
        }
    del O, H, L, C
    gc.collect()
    shape_rk = {}
    for nm, arr in shapes.items():
        shape_rk[nm] = pd.DataFrame(arr).rank(axis=0, pct=True).to_numpy("float32")
        del arr
    del shapes
    gc.collect()
    # 形状坏分: 大振幅/长上影/冲高回落高分坏, 弱收(收在下半, close_pos 秩低)坏
    shape_bad = np.nanmean(
        np.stack(
            [
                shape_rk["amp_pct"],
                shape_rk["upper_wick"],
                shape_rk["spike_rev"],
                1.0 - shape_rk["close_pos"],
            ]
        ),
        axis=0,
    ).astype("float32")
    del shape_rk
    S1 = np.full_like(shape_bad, np.nan)
    S1[:, 1:] = shape_bad[:, :-1]  # T-1 日形状, PIT 安全
    del shape_bad
    gc.collect()

    del panel
    gc.collect()

    def nanmean_mats(cols):
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.stack([rank_mats[c] for c in cols]), axis=0).astype(
                "float32"
            )

    F1 = nanmean_mats(MOM_COLS)
    F2 = nanmean_mats(HOTVOL_COLS)
    F4 = nanmean_mats([*MOM_COLS, "peak_roc_20d"])
    F3 = np.nanmean(np.stack([F1, F2]), axis=0).astype("float32")
    F5 = np.nanmean(np.stack([F1, F2, S1]), axis=0).astype("float32")
    arms = {
        "F1_hotmom": F1,
        "F2_hotvol": F2,
        "F3_combined": F3,
        "F4_runup": F4,
        "S1_shape": S1,
        "F5_heatshape": F5,
    }

    eval_lo = n_eval - args.eval
    days = np.arange(max(1, eval_lo), n_eval)
    half = len(days) // 2

    # 未来 5 日内放量下跌 (滚动 OR)
    flush_next = np.zeros_like(flush)
    for off in range(1, HORIZON + 1):
        sh = np.zeros_like(flush)
        sh[:, :-off] = flush[:, off:]
        flush_next |= sh

    rows = []
    for arm, W in arms.items():
        TM = np.zeros_like(flush)
        for j in days:
            w = W[:, j]
            ok = np.isfinite(w)
            if ok.sum() < 100:
                continue
            q = np.quantile(w[ok], 1 - TOP_Q)
            TM[ok, j] = w[ok] >= q
        prev_top = np.zeros_like(TM)
        for off in range(1, HORIZON + 1):
            sh = np.zeros_like(TM)
            sh[:, off:] = TM[:, :-off]
            prev_top |= sh

        fmask = flush[:, days]
        pmask = prev_top[:, days]
        recall = float(pmask[fmask].mean()) if fmask.any() else np.nan
        flush_next[:, days] & TM[:, days]
        prec = (
            float(flush_next[:, days][TM[:, days]].mean())
            if TM[:, days].any()
            else np.nan
        )
        base_prec = float(flush_next[:, days].mean())
        f3v = fwd3[:, days]
        top_f = f3v[TM[:, days]]
        top_fwd3 = float(np.nanmean(top_f)) if np.isfinite(top_f).any() else np.nan
        h1m = np.zeros(len(days), dtype=bool)
        h1m[:half] = True
        rec_h1 = (
            float(pmask[:, h1m][fmask[:, h1m]].mean())
            if fmask[:, h1m].any()
            else np.nan
        )
        rec_h2 = (
            float(pmask[:, ~h1m][fmask[:, ~h1m]].mean())
            if fmask[:, ~h1m].any()
            else np.nan
        )
        rows.append(
            {
                "arm": arm,
                "recall_5d": round(recall, 4),
                "recall_h1": round(rec_h1, 4),
                "recall_h2": round(rec_h2, 4),
                "precision_5d": round(prec, 4),
                "base_rate_5d": round(base_prec, 4),
                "top_fwd3": round(top_fwd3, 5),
                "top_n": int(TM[:, days].sum()),
            }
        )

    res = pd.DataFrame(rows).sort_values("recall_5d", ascending=False)
    print(res.to_string(index=False), flush=True)

    # 当日风险清单: 面板最后交易日, 各臂 top15
    last_j = n_days - 1
    last_date = str(pd.Timestamp(cal[last_j]).date())
    pred_rows = []
    for arm, W in arms.items():
        w = W[:, last_j]
        ok = np.isfinite(w)
        cand = pd.DataFrame(
            {"symbol": symbols[ok], "score": w[ok], "pct_today": pct[ok, last_j]}
        )
        for rank, (_, r) in enumerate(cand.nlargest(15, "score").iterrows(), 1):
            pred_rows.append(
                {"arm": arm, "rank": rank, **r.to_dict(), "asof": last_date}
            )
    pred = pd.DataFrame(pred_rows)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pq_path = out_dir / f"preflush_detector_{ts}.parquet"
    res.to_parquet(pq_path, index=False)
    pred.to_parquet(out_dir / f"preflush_detector_{ts}_picks.parquet", index=False)
    meta = {
        "ts": ts,
        "days": args.days,
        "eval": args.eval,
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "asof": last_date,
        "flush_thr": FLUSH_THR,
        "vol_q": VOL_Q,
        "cost": COST,
        "summary": res.to_dict("records"),
        "protocol_deviations": [
            "事件=跌幅<=-5% 且换手秩>=0.80 (双创同阈)",
            "recall=事件前5日曾入top10%比例; precision=top10%成员5日内放量下跌率",
            "用途=回避/硬过滤 (A股无便利做空)",
            "09-03 增形状臂: S1=T-1日(大振幅+长上影+冲高回落+弱收)秩均值, F5=mean(F1,F2,S1)",
        ],
    }
    (out_dir / f"preflush_detector_{ts}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {pq_path} ({time.time() - t0:.0f}s)", flush=True)
    print(pred[pred["arm"] == "F3_combined"].to_string(index=False), flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
