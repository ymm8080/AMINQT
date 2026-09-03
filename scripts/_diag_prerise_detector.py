"""_diag_prerise_detector.py — 涨前预热捕获器 (动量漂移+量能换手预热→上涨预测, 2026-09-03).

动机: preinfo_audit_0903 定案 — 涨前月内仅有的真信号 = bias 动量漂移 + 量能/换手
预热, 且 w1 (涨前最后 5 日) 前置. 本探针把这两项做成日频可计算信号, 检验能否
提前捕获上涨 (尤其强涨日), 并产出当日 top 预测清单.

四臂 (审计直接映射, 无扫参):
  A1 mom      = bias_20/60/120/250 截面秩均值 (当日)
  A2 vol      = turnover_rate/amount/free_float_turnover_rate/volume_ratio/
                ma_vol_ratio_5_20 截面秩均值 (当日)
  A3 combined = mean(A1, A2)
  A4 warm5    = mean(A1, A2 的 5 日均值) — 审计 w1 前置的预热形态

评估 (近 250 交易日, PIT: 信号只用 t 及更早, 前向 net3 = px[t+4]/px[t+1]-1-成本):
  分位净收益曲线 (top10% vs 全池), top10 日净收益/hit3 半窗拆分, 双板拆分,
  强涨日捕获: 未来 5 日内出现 >=9.5%(主板)/>=19%(双创) 强涨日的事件中,
  此前 5 日内曾入该臂 top10% 的比例 vs 基线 10%.

WORM: DATA OTHERS/diag/prerise_detector_<ts>.parquet + .json
用法: python scripts/_diag_prerise_detector.py [--days 550] [--eval 250]
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
MOM_COLS = ("bias_20", "bias_60", "bias_120", "bias_250")
VOL_COLS = (
    "turnover_rate",
    "amount",
    "free_float_turnover_rate",
    "volume_ratio",
    "ma_vol_ratio_5_20",
)
TOP_Q = 0.10
TOP_N = 10
LU_THR_MAIN, LU_THR_DUAL = 0.095, 0.19
CAP_WINDOW = 5  # 强涨日捕获回看窗 (事件前 5 日内曾入 top10%)


def _rank_mats(panel: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, np.ndarray]:
    """每列 → (symbol × date) 截面 pct-rank 矩阵."""
    dt = pd.to_datetime(panel["date"]).dt.normalize()
    out = {}
    for col in cols:
        df = pd.DataFrame({"s": panel["symbol"], "d": dt, "v": panel[col]})
        df["rk"] = df.groupby("d")["v"].rank(pct=True)
        mat = df.pivot(index="s", columns="d", values="rk").sort_index()
        out[col] = mat.to_numpy(dtype="float32")
        del mat
    return out


def _nanmean_mats(mats: list[np.ndarray]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        stack = np.stack(mats)
        return np.nanmean(stack, axis=0).astype("float32")


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

    read_cols = ["symbol", "date", "close_hfq", *MOM_COLS, *VOL_COLS]
    panel = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=read_cols,
        filters=[("date", ">=", cutoff)],
    )
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    dt = pd.to_datetime(panel["date"]).dt.normalize()
    cal = np.sort(dt.unique())
    symbols = np.sort(panel["symbol"].unique())
    s_idx = {s: i for i, s in enumerate(symbols)}
    d_idx = {d: i for i, d in enumerate(cal)}
    print(
        f"[panel] {len(symbols)} syms x {len(cal)} days ({time.time() - t0:.0f}s)",
        flush=True,
    )

    # 价格矩阵 (ffill) + 前向净收益 + 强涨日标签
    px_wide = (
        panel.assign(d=dt)
        .pivot_table(index="symbol", columns="d", values="close_hfq", aggfunc="last")
        .sort_index()
        .reindex(index=symbols, columns=pd.DatetimeIndex(cal))
    )
    px = px_wide.ffill(axis=1).to_numpy(dtype="float64")
    del px_wide
    gc.collect()
    si = np.array([s_idx[s] for s in symbols])
    di = np.array([d_idx[d] for d in cal])
    px = px[np.ix_(si, di)]  # 行=symbols, 列=cal (对齐特征矩阵)

    pct = px / np.roll(px, 1, axis=1) - 1.0
    pct[:, 0] = np.nan
    fwd3 = (
        px[:, 4:] / px[:, 1:-3] - 1.0 - COST
    )  # 列 j → 交易日 cal[j+1]买入, cal[j+4]卖
    n_eval = fwd3.shape[1]
    dual_mask = np.array([s[:2] in ("30", "68") for s in symbols])
    lu_thr = np.where(dual_mask, LU_THR_DUAL, LU_THR_MAIN)[:, None]
    big_rise = pct >= lu_thr
    print(
        f"[fwd] eval days={n_eval} dual_syms={int(dual_mask.sum())} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    # 四臂信号矩阵
    r_mom = _rank_mats(panel, MOM_COLS)
    r_vol = _rank_mats(panel, VOL_COLS)
    del panel
    gc.collect()
    A1 = _nanmean_mats([r_mom[c] for c in MOM_COLS])
    A2 = _nanmean_mats([r_vol[c] for c in VOL_COLS])
    A2_df = pd.DataFrame(A2)
    A2_5 = A2_df.rolling(CAP_WINDOW, axis=1, min_periods=3).mean().to_numpy("float32")
    del A2_df
    arms = {
        "A1_mom": A1,
        "A2_vol": A2,
        "A3_combined": _nanmean_mats([A1, A2]),
        "A4_warm5": _nanmean_mats([A1, A2_5]),
    }
    del r_mom, r_vol, A2_5
    gc.collect()

    # 评估: 特征矩阵列对齐 cal; fwd3 列 j 对应 cal[j] 信号日 (t=j, 买 j+1 卖 j+4)
    eval_lo = n_eval - args.eval
    days = np.arange(max(0, eval_lo), n_eval)
    half = len(days) // 2

    per_arm_rows = []
    picks = {}
    for arm, W in arms.items():
        W_e = W[:, days]
        F_e = fwd3[:, days]
        top_sets = {}  # day_pos → top10% symbol set
        day_stats = []
        for k in range(len(days)):
            w, f = W_e[:, k], F_e[:, k]
            ok = np.isfinite(w) & np.isfinite(f)
            if ok.sum() < 100:
                continue
            wv, fv = w[ok], f[ok]
            q = np.quantile(wv, 1 - TOP_Q)
            topm = wv >= q
            base = float(np.median(fv))
            day_stats.append(
                {
                    "day_pos": days[k],
                    "date": str(pd.Timestamp(cal[days[k]]).date()),
                    "top_net3": float(fv[topm].mean()),
                    "all_med_net3": base,
                    "top_hit3": float((fv[topm] > 0).mean()),
                    "n": int(ok.sum()),
                }
            )
            top_sets[days[k]] = set(symbols[ok][topm])
        ds = pd.DataFrame(day_stats)
        h1, h2 = ds.iloc[:half], ds.iloc[half:]
        spread = ds["top_net3"] - ds["all_med_net3"]

        # 强涨日捕获: 事件=交易日 t+1..t+5 内出现强涨日 (t 在评估窗内)
        cap_ev = cap_hit = 0
        ev_days = set(days)
        for j in range(n_eval):
            if j not in ev_days:
                continue
            for s_i in np.nonzero(big_rise[:, j])[0]:
                cap_ev += 1
                look = top_sets.get(j - 1, set()) | top_sets.get(j - 2, set())
                look |= top_sets.get(j - 3, set()) | top_sets.get(j - 4, set())
                look |= top_sets.get(j - 5, set())
                if symbols[s_i] in look:
                    cap_hit += 1
        per_arm_rows.append(
            {
                "arm": arm,
                "days": int(len(ds)),
                "top_net3": round(float(ds["top_net3"].mean()), 5),
                "all_med_net3": round(float(ds["all_med_net3"].mean()), 5),
                "spread": round(float(spread.mean()), 5),
                "spread_h1": round(
                    float((h1["top_net3"] - h1["all_med_net3"]).mean()), 5
                ),
                "spread_h2": round(
                    float((h2["top_net3"] - h2["all_med_net3"]).mean()), 5
                ),
                "top_hit3": round(float(ds["top_hit3"].mean()), 4),
                "lu_events": cap_ev,
                "lu_capture": round(cap_hit / cap_ev, 4) if cap_ev else None,
                "lu_capture_base": TOP_Q,
            }
        )
        picks[arm] = sorted(top_sets.values(), key=len)[-1] if top_sets else set()

    res = pd.DataFrame(per_arm_rows).sort_values("spread", ascending=False)
    print(res.to_string(index=False), flush=True)

    # 当日预测清单: 面板最后交易日 (非 fwd3 末列 — 那是 4 日前), 剔当日已强涨, 各臂 top15
    last_j = len(cal) - 1
    last_date = str(pd.Timestamp(cal[last_j]).date())
    today_rise = pct[:, last_j]
    pred_rows = []
    for arm, W in arms.items():
        w = W[:, last_j]
        ok = np.isfinite(w) & ~(today_rise >= lu_thr[:, 0])
        cand = pd.DataFrame(
            {"symbol": symbols[ok], "score": w[ok], "pct_today": today_rise[ok]}
        )
        top15 = cand.nlargest(15, "score")
        for rank, (_, r) in enumerate(top15.iterrows(), 1):
            pred_rows.append(
                {"arm": arm, "rank": rank, **r.to_dict(), "asof": last_date}
            )
    pred = pd.DataFrame(pred_rows)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pq_path = out_dir / f"prerise_detector_{ts}.parquet"
    res.to_parquet(pq_path, index=False)
    pred.to_parquet(out_dir / f"prerise_detector_{ts}_picks.parquet", index=False)
    meta = {
        "ts": ts,
        "days": args.days,
        "eval": args.eval,
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "asof": last_date,
        "cost": COST,
        "arms": {a: {"mom": list(MOM_COLS), "vol": list(VOL_COLS)} for a in ()}
        | {
            "MOM_COLS": list(MOM_COLS),
            "VOL_COLS": list(VOL_COLS),
        },
        "summary": res.to_dict("records"),
        "gate": {
            "necessary": [
                "spread_h1 与 spread_h2 同向且 >0 (至少一臂)",
                "lu_capture > 0.10 基线 (同臂)",
            ],
            "note": "必要条件全过才有预测价值; 全负则与 vp 族判死同判",
        },
        "protocol_deviations": [
            "信号=当日截面秩, 前向=close_hfq t+1→t+4 净收益 (T+1 买入口径)",
            "强涨日阈值 9.5%/19% 按代码前缀 (30/68 双创) 近似涨跌停",
            "预测清单剔当日已强涨股 (非'涨前')",
        ],
    }
    (out_dir / f"prerise_detector_{ts}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {pq_path} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[asof] {last_date} 预测清单 top15/臂 已落盘", flush=True)
    print(pred[pred["arm"] == "A4_warm5"].to_string(index=False), flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
