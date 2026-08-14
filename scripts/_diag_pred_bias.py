"""_diag_pred_bias.py — 量化 PARALLEL STOCK LIST 预测值 vs 已实现 偏差 (2026-08-13 用户: 预测普遍偏高).

对比对象:
  - pred_mag_{h}  (每股 MFE 回归预期 / 10d 为 c2c)  vs  已实现 MFE_{h}
  - pred_ret_{h}  (每股 c2c 平均预期)                 vs  已实现 label_pm_{h}d_net
  - pred_prob_{h} (达到固定绝对目标概率)              vs  实际达到率 (mfe >= ABS_TARGET)

数据:
  - 预测: STOCK_LIST_DIR/parallel_shortlist_*.csv (每日交付, 用户看到的值)
  - 已实现: data/_diag_stage_{board}_3y.parquet 的 label_pm_{h}d_net (c2c 净收益, 已扣成本)
            + close_hfq/high_hfq 重算 MFE (生产 add_mfe_labels 口径)

WORM 输出: data/_diag_pred_bias_<ts>.csv + .json
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline1.label_engine import COST, slippage_tier
from config.settings import DATA_DIR, STOCK_LIST_DIR

HORIZONS = ("3d", "5d", "10d")
ABS_TARGET = {"3d": 0.03, "5d": 0.04, "10d": 0.06}
LABEL_PM = {h: f"label_pm_{h[:-1]}d_net" for h in HORIZONS}


def compute_realized(panel: pd.DataFrame) -> pd.DataFrame:
    """按生产 add_mfe_labels 口径补已实现 MFE_{h}, 复用面板已有 c2c 净收益列.

    panel 需为 (symbol,date) 排序切片 (含 date >= 预测首日-1). 返回副本加列.
    """
    df = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol")
    exec_px = g["close_hfq"].shift(-1)
    max_off = 11
    shifts = pd.concat(
        [g["high_hfq"].shift(-off) for off in range(2, max_off + 1)],
        axis=1,
        keys=range(2, max_off + 1),
    )
    slip = df["adv20"].map(slippage_tier) if "adv20" in df.columns else 0.0015
    cost_total = COST + 2 * slip
    for h in HORIZONS:
        k = int(h[:-1])
        peak = shifts.loc[:, 2 : k + 1].max(axis=1, skipna=False)
        df[f"real_mfe_{h}"] = peak / exec_px - 1 - cost_total
    df["real_c2c_3d"] = df["label_pm_3d_net"]
    df["real_c2c_5d"] = df["label_pm_5d_net"]
    df["real_c2c_10d"] = df["label_pm_10d_net"]
    return df


def main() -> int:
    pred_cols = ["date", "board", "symbol"] + [
        f"{k}_{h}" for h in HORIZONS for k in ("pred_mag", "pred_prob", "pred_ret")
    ]
    frames = []
    for fp in sorted(glob.glob(str(STOCK_LIST_DIR / "parallel_shortlist_*.csv"))):
        d = pd.read_csv(fp, dtype={"symbol": str})
        keep = [c for c in pred_cols if c in d.columns]
        d = d[keep].copy()
        d["date"] = d["date"].astype(str).str.slice(0, 10)
        frames.append(d)
    if not frames:
        print("无 parallel_shortlist_*.csv")
        return 1
    pred = pd.concat(frames, ignore_index=True)
    pred = pred.drop_duplicates(subset=["date", "board", "symbol"]).reset_index(drop=True)
    pred["date"] = pd.to_datetime(pred["date"])
    print(f"[pred] {len(pred)} 行, 日期 {pred['date'].min().date()} ~ {pred['date'].max().date()}")

    # 面板切片: 预测首日前 1 交易日起 (供 T+1 买入价) — 已实现列只取面板自身可测部分
    lo = pred["date"].min() - pd.Timedelta(days=5)
    panels = {}
    for board in ("main", "dual"):
        t = pq.read_table(
            str(DATA_DIR / f"_diag_stage_{board}_3y.parquet"),
            columns=["symbol", "date", "close_hfq", "high_hfq", "adv20"]
            + list(LABEL_PM.values()),
            filters=[("date", ">=", lo)],
        ).to_pandas()
        t["symbol"] = t["symbol"].astype(str)
        t["date"] = pd.to_datetime(t["date"])
        panels[board] = t
    # 只算预测涉及 symbol 的行, 减少 MFE shift 开销
    need_syms = set(pred["symbol"]) & set(pred["board"])
    for board, t in panels.items():
        syms = set(pred.loc[pred["board"] == board, "symbol"])
        panels[board] = t[t["symbol"].isin(syms)]
    real = pd.concat([compute_realized(t) for t in panels.values()], ignore_index=True)
    del panels
    real = real[["symbol", "date"] + [c for c in real.columns if c.startswith("real_")]]

    merged = pred.merge(real, on=["symbol", "date"], how="left")
    rows = []
    for h in HORIZONS:
        sub = merged[
            ["date", "board", f"pred_mag_{h}", f"real_mfe_{h}",
             f"pred_ret_{h}", f"real_c2c_{h}", f"pred_prob_{h}"]
        ].copy()
        sub = sub.dropna(subset=[f"real_mfe_{h}"])
        if sub.empty:
            continue
        n = len(sub)
        pm, rm = sub[f"pred_mag_{h}"], sub[f"real_mfe_{h}"]
        pr, rr = sub[f"pred_ret_{h}"], sub[f"real_c2c_{h}"]
        pp = sub[f"pred_prob_{h}"]
        hit = (rm >= ABS_TARGET[h]).mean()
        rows.append(
            {
                "horizon": h,
                "n": n,
                "pred_mag_mean": float(pm.mean()),
                "real_mfe_mean": float(rm.mean()),
                "mag_bias": float(pm.mean() - rm.mean()),
                "pred_ret_mean": float(pr.mean()),
                "real_c2c_mean": float(rr.mean()),
                "ret_bias": float(pr.mean() - rr.mean()),
                "pred_prob_mean": float(pp.mean()),
                "real_hit_rate": float(hit),
                "prob_bias": float(pp.mean() - hit),
            }
        )
        # 逐日偏差 (观察是否近期更差)
        daily = sub.groupby("date").agg(
            pred_ret=(f"pred_ret_{h}", "mean"),
            real_c2c=(f"real_c2c_{h}", "mean"),
            pred_mag=(f"pred_mag_{h}", "mean"),
            real_mfe=(f"real_mfe_{h}", "mean"),
        )
        print(f"\n=== T+{h[:-1]} (n={n}) 逐日 ===")
        print(daily.round(4).to_string())

    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_csv = DATA_DIR / f"_diag_pred_bias_{ts}.csv"
    df.to_csv(out_csv, index=False)
    (DATA_DIR / f"_diag_pred_bias_{ts}.json").write_text(
        json.dumps({"ts": ts, "rows": df.to_dict("records")}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
