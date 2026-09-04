"""_diag_volflush_study.py — 放量下跌事件研究 + 多指标交叉判别 (2026-09-03).

动机 (用户 09-03): 涨前预热捕获器 (prerise_detector) 的高分含"放量下跌"名字
(高位放量下跌也吃高 vol 秩). 问题: 放量下跌股票后续是跌势延续 (出货) 还是
止跌反弹 (洗盘)? 哪些已计算指标能交叉判别? 捕获器清单里的下跌名字是拖累还是金子?

事件 (L = 事件日): 当日 close_hfq 跌幅 <= -5% 且 turnover_rate 截面秩 >= 0.80.
标签: 腿后 3 日净回报 fwd3 = px[L+4]/px[L+1]-1-COST
  CONT   (跌势延续) fwd3 <= -2%
  BOUNCE (止跌反弹) fwd3 >= +2%
  中间剔除. 双标签外全体事件另报: 未来 5 日再大跌率 / 未来 5 日强涨日率.

交叉判别指标 (事件日 L 与 L-1 两个时点的截面秩, AUC, CONT=1):
  mom(bias_20/60/120/250) vol(turnover/amount/free_float_turn/vol_ratio/ma_vol_5_20)
  chip(winner_ratio/pct_90_con/chip_skew_dist/cost_bias/resistance_dist/
       conc_trend_20d/chip_gini/peak_roc_20d/chip_entropy)
  holder(sh_net_change_sign/sh_net_sign/sh_change_vol)
  margin(margin_buy_amt/margin_balance) val(dv_ttm/dv_ratio/pe_ttm)

捕获器交叉验证: 事件按 A4_warm5 当日是否 top10% 分组, 比较 fwd3/未来5日强涨率
  — 量化"清单里放量下跌名字"的拖累或增益.
半窗拆分稳定性 (前/后 125 交易日 AUC).

WORM: DATA OTHERS/diag/volflush_study_<ts>.parquet + .json
用法: python scripts/_diag_volflush_study.py [--days 550] [--eval 250]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

COST = 0.0020
FLUSH_THR = -0.05
VOL_Q = 0.80
CONT_THR = -0.02
BOUNCE_THR = 0.02
NEXT_WIN = 5
LU_MAIN, LU_DUAL = 0.095, 0.19

MOM_COLS = ("bias_20", "bias_60", "bias_120", "bias_250")
VOL_COLS = (
    "turnover_rate",
    "amount",
    "free_float_turnover_rate",
    "volume_ratio",
    "ma_vol_ratio_5_20",
)
CHIP_COLS = (
    "winner_ratio",
    "pct_90_con",
    "chip_skew_dist",
    "cost_bias",
    "resistance_dist",
    "conc_trend_20d",
    "chip_gini",
    "peak_roc_20d",
    "chip_entropy",
)
HOLDER_COLS = ("sh_net_change_sign", "sh_net_sign", "sh_change_vol")
MARGIN_COLS = ("margin_buy_amt", "margin_balance")
VAL_COLS = ("dv_ttm", "dv_ratio", "pe_ttm")
FAM = {
    "mom": MOM_COLS,
    "vol": VOL_COLS,
    "chip": CHIP_COLS,
    "holder": HOLDER_COLS,
    "margin": MARGIN_COLS,
    "val": VAL_COLS,
}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    ok = np.isfinite(scores)
    s, y = scores[ok], labels[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = ranks[y == 1].sum()
    return float((rp - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


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

    all_cols = sorted({c for cs in FAM.values() for c in cs})
    panel = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=[
            "symbol",
            "date",
            "close_hfq",
            "open",
            "high",
            "low",
            "close",
            "volume",
            *all_cols,
        ],
        filters=[("date", ">=", cutoff)],
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
    fwd3 = px[:, 4:] / px[:, 1:-3] - 1.0 - COST  # 列 j ↔ 信号日 cal[j]
    n_eval = fwd3.shape[1]
    dual_mask = np.array([s[:2] in ("30", "68") for s in symbols])
    lu_thr = np.where(dual_mask, LU_DUAL, LU_MAIN)[:, None]

    # 指标秩矩阵 (symbols × cal, 与 px 对齐)
    rank_mats: dict[str, np.ndarray] = {}
    for k, col in enumerate(all_cols):
        df = pd.DataFrame({"s": panel["symbol"], "d": dt, "v": panel[col]})
        df["rk"] = df.groupby("d")["v"].rank(pct=True)
        rank_mats[col] = (
            df.pivot(index="s", columns="d", values="rk")
            .sort_index()
            .to_numpy(dtype="float32")
        )
        del df
        if (k + 1) % 10 == 0:
            print(
                f"  [rank] {k + 1}/{len(all_cols)} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    # 事件: 跌 <=-5% 且 换手秩 >=0.8
    turn_rk = rank_mats["turnover_rate"]
    ev_s, ev_j = [], []
    for j in range(1, n_eval):
        m = (pct[:, j] <= FLUSH_THR) & (turn_rk[:, j] >= VOL_Q)
        idx = np.nonzero(m)[0]
        ev_s.extend(idx.tolist())
        ev_j.extend([j] * len(idx))
    ev_s = np.asarray(ev_s)
    ev_j = np.asarray(ev_j)
    n_ev = len(ev_s)
    f = fwd3[ev_s, ev_j]
    print(f"[events] 放量下跌 {n_ev:,} ({time.time() - t0:.0f}s)", flush=True)

    # 基础率
    nxt_rise = np.zeros(n_ev, dtype=bool)
    nxt_drop = np.zeros(n_ev, dtype=bool)
    for off in range(1, NEXT_WIN + 1):
        jj = np.clip(ev_j + off, 0, n_days - 1)
        nxt_rise |= pct[ev_s, jj] >= lu_thr[ev_s, 0]
        nxt_drop |= pct[ev_s, jj] <= FLUSH_THR
    valid = np.isfinite(f)
    base = {
        "n_events": int(n_ev),
        "fwd3_mean": round(float(np.nanmean(f)), 5),
        "fwd3_med": round(float(np.nanmedian(f)), 5),
        "cont_rate": round(float(np.mean(f[valid] <= CONT_THR)), 4),
        "bounce_rate": round(float(np.mean(f[valid] >= BOUNCE_THR)), 4),
        "next5_bigdrop_rate": round(float(np.mean(nxt_drop)), 4),
        "next5_bigrise_rate": round(float(np.mean(nxt_rise)), 4),
    }
    print(f"[base] {base}", flush=True)

    # 多指标交叉判别: AUC(CONT=1) @ lag0 (事件日) 与 lag1 (前一交易日), 半窗拆分
    half_j = n_eval // 2
    disc_rows = []
    for fam, cols in FAM.items():
        for col in cols:
            m = rank_mats[col]
            for lag in (0, 1):
                jj = np.clip(ev_j - lag, 0, n_days - 1)
                sc = m[ev_s, jj]
                ok = np.isfinite(sc) & np.isfinite(f)
                y = (f[ok] <= CONT_THR).astype(int)
                auc = _auc(sc[ok], y)
                jmask = ev_j[ok] <= half_j
                auc_h1 = _auc(sc[ok][jmask], y[jmask])
                auc_h2 = _auc(sc[ok][~jmask], y[~jmask])
                disc_rows.append(
                    {
                        "family": fam,
                        "col": col,
                        "lag": lag,
                        "n": int(ok.sum()),
                        "n_cont": int(y.sum()),
                        "auc": round(auc, 4) if np.isfinite(auc) else None,
                        "auc_h1": round(auc_h1, 4) if np.isfinite(auc_h1) else None,
                        "auc_h2": round(auc_h2, 4) if np.isfinite(auc_h2) else None,
                    }
                )

    # ---- OHLC 日频微观结构判别 (用户 09-03: 用 OHLC/量能/换手, 勿用慢变量) ----
    def _pivot_base(col: str) -> np.ndarray:
        w = (
            panel.assign(d=dt)
            .pivot_table(index="symbol", columns="d", values=col, aggfunc="last")
            .sort_index()
            .reindex(index=symbols, columns=pd.DatetimeIndex(cal))
        )
        return w.to_numpy(dtype="float64")

    O = _pivot_base("open")
    H = _pivot_base("high")
    L = _pivot_base("low")
    C = _pivot_base("close")
    V = _pivot_base("volume")
    T_ = _pivot_base("turnover_rate")
    del panel
    gc.collect()
    prevC = np.roll(C, 1, axis=1)
    prevC[:, 0] = np.nan
    rng = H - L
    with np.errstate(invalid="ignore", divide="ignore"):
        ohlc_raw = {
            "close_pos": np.where(rng > 0, (C - L) / rng, np.nan),
            "upper_wick": np.where(rng > 0, (H - np.maximum(O, C)) / rng, np.nan),
            "lower_wick": np.where(rng > 0, (np.minimum(O, C) - L) / rng, np.nan),
            "gap": O / prevC - 1.0,
            "intraday": C / O - 1.0,
            "spike_rev": C / H - 1.0,  # 收盘距最高 (冲高回落)
            "amp_pct": (H - L) / prevC,  # 日振幅占昨收
        }
    del O, H, L
    gc.collect()
    for name, base_mat, roll in (
        ("vol_x5", V, 5),
        ("turn_x5", T_, 5),
    ):
        b = pd.DataFrame(base_mat)
        m = b / b.T.rolling(roll, min_periods=3).mean().T
        ohlc_raw[name] = m.to_numpy()
        del b, m
    del V, T_, C
    gc.collect()

    OHLC_FEATS = tuple(ohlc_raw.keys())
    for col in OHLC_FEATS:
        m = pd.DataFrame(ohlc_raw[col]).rank(axis=0, pct=True).to_numpy("float32")
        rank_mats[col] = m
        for lag in (0, 1):
            jj = np.clip(ev_j - lag, 0, n_days - 1)
            sc = m[ev_s, jj]
            ok = np.isfinite(sc) & np.isfinite(f)
            y = (f[ok] <= CONT_THR).astype(int)
            auc = _auc(sc[ok], y)
            jmask = ev_j[ok] <= half_j
            disc_rows.append(
                {
                    "family": "ohlc",
                    "col": col,
                    "lag": lag,
                    "n": int(ok.sum()),
                    "n_cont": int(y.sum()),
                    "auc": round(auc, 4) if np.isfinite(auc) else None,
                    "auc_h1": round(_auc(sc[ok][jmask], y[jmask]), 4)
                    if jmask.any()
                    else None,
                    "auc_h2": round(_auc(sc[ok][~jmask], y[~jmask]), 4)
                    if (~jmask).any()
                    else None,
                }
            )
        del m

    # 事件形态分组 (原始值, 非秩): 高开杀/低开杀/长下影收回/长上影
    cp = ohlc_raw["close_pos"]
    gp = ohlc_raw["gap"]
    lw = ohlc_raw["lower_wick"]
    uw = ohlc_raw["upper_wick"]
    shape_groups = {
        "高开杀跌(gap>0,收于下半)": (gp[ev_s, ev_j] > 0) & (cp[ev_s, ev_j] < 0.5),
        "低开杀跌(gap<=0,收于下半)": (gp[ev_s, ev_j] <= 0) & (cp[ev_s, ev_j] < 0.5),
        "长下影收回(lower_wick>0.4)": lw[ev_s, ev_j] > 0.4,
        "长上影(upper_wick>0.4)": uw[ev_s, ev_j] > 0.4,
    }
    shape_rows = []
    for name, m_ in shape_groups.items():
        mm = np.nan_to_num(m_.astype(float), nan=0.0).astype(bool) & np.isfinite(f)
        if mm.sum() < 50:
            continue
        shape_rows.append(
            {
                "shape": name,
                "n": int(mm.sum()),
                "fwd3_mean": round(float(np.nanmean(f[mm])), 5),
                "bounce_rate": round(float(np.mean(f[mm] >= BOUNCE_THR)), 4),
                "next5_bigrise_rate": round(float(np.mean(nxt_rise[mm])), 4),
                "next5_bigdrop_rate": round(float(np.mean(nxt_drop[mm])), 4),
            }
        )
    shape_df = pd.DataFrame(shape_rows)
    print("=== 事件日形态分组 ===")
    print(shape_df.to_string(index=False), flush=True)
    del ohlc_raw, cp, gp, lw, uw
    gc.collect()

    disc = pd.DataFrame(disc_rows)

    # 捕获器交叉验证: 事件日 A4_warm5 top10% 与否 → 后续表现
    A1 = np.nanmean(np.stack([rank_mats[c] for c in MOM_COLS]), axis=0)
    A2 = np.nanmean(np.stack([rank_mats[c] for c in VOL_COLS]), axis=0)
    A2_5 = (
        pd.DataFrame(A2).T.rolling(NEXT_WIN, min_periods=3).mean().T.to_numpy("float32")
    )
    A4 = np.nanmean(np.stack([A1, A2_5]), axis=0)
    a4 = A4[ev_s, ev_j]
    is_top = np.zeros(n_ev, dtype=bool)
    for j in np.unique(ev_j):
        col_ok = ev_j == j
        w = a4[col_ok]
        ok = np.isfinite(w)
        if ok.sum() < 20:
            continue
        q = np.quantile(w[ok], 0.90)
        is_top[col_ok] = ok & (w >= q)
    grp = []
    for name, m_ in (("A4top10%", is_top), ("其余", ~is_top)):
        mm = m_ & np.isfinite(f)
        grp.append(
            {
                "group": name,
                "n": int(mm.sum()),
                "fwd3_mean": round(float(np.nanmean(f[mm])), 5),
                "bounce_rate": round(float(np.mean(f[mm] >= BOUNCE_THR)), 4),
                "next5_bigrise_rate": round(float(np.mean(nxt_rise[mm])), 4),
                "next5_bigdrop_rate": round(float(np.mean(nxt_drop[mm])), 4),
            }
        )
    grp_df = pd.DataFrame(grp)
    print("=== 捕获器 A4 分组 (事件内) ===")
    print(grp_df.to_string(index=False), flush=True)

    top_lag0 = (
        disc[(disc["lag"] == 0) & disc["auc"].notna()]
        .sort_values("auc", ascending=False)
        .head(12)
    )
    print("=== 判别 TOP (lag0, AUC>0.5=延续) ===")
    print(top_lag0.to_string(index=False), flush=True)
    ohlc_lag0 = disc[
        (disc["family"] == "ohlc") & (disc["lag"] == 0) & disc["auc"].notna()
    ].sort_values("auc", ascending=False)
    print("=== OHLC 判别 (lag0) ===")
    print(ohlc_lag0.to_string(index=False), flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pq_path = out_dir / f"volflush_study_{ts}.parquet"
    disc.to_parquet(pq_path, index=False)
    meta = {
        "ts": ts,
        "days": args.days,
        "eval": args.eval,
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "cost": COST,
        "flush_thr": FLUSH_THR,
        "vol_q": VOL_Q,
        "cont_bounce_thr": [CONT_THR, BOUNCE_THR],
        "base": base,
        "a4_groups": grp_df.to_dict("records"),
        "shape_groups": shape_df.to_dict("records"),
        "ohlc_disc_lag0": ohlc_lag0.to_dict("records"),
        "top_disc_lag0": top_lag0.to_dict("records"),
        "protocol_deviations": [
            "放量下跌=跌幅<=-5% 且换手秩>=0.80 (双创同阈, 未按 20cm 放宽)",
            "AUC lag0=事件日收盘可知, lag1=前一交易日可知 (可提前布防)",
            "AUC>0.5 = 指标高分→跌势延续 (出货); <0.5 = 高分→止跌反弹 (洗盘)",
        ],
    }
    (out_dir / f"volflush_study_{ts}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {pq_path} rows={len(disc)} ({time.time() - t0:.0f}s)", flush=True)
    print(json.dumps(base, ensure_ascii=False))
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raise SystemExit(main())
