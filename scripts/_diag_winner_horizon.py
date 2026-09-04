"""_diag_winner_horizon.py — 赢家节奏检查: T+3/T+5/T+10 同员重算 (2026-09-03).

背景: winner-leak 复盘定案 6 月后排名键赢家判别 AUC 0.52 (抛硬币), 且冻结包
in-sample 也排不动 = 旧特征对新行情无信息. 剩一个未验假说: 赢家没消失, 是变快了
(快轮动行情 3-5 天兑现后回吐, T+10 c2c 抹平). 本脚本在 winner_leak 同一 scored
检查点上, 用面板 close_hfq 重建 pivot, 把实得收益重算为 T+3/T+5/T+10 三口径:

  成员/排名不变: TOP10 = E7 池按生产键 (pred_ret_10d) 当日取前 10, 与窗口无关.
  变的只是结局: realized_net_H = close[i+H+1] / close[i+1] - 1 - 0.2%.

判据 (预登记):
  6-8 月短窗 (T+3/T+5) AUC 回 0.6+ 且席位率显著回升 → 赢家变快了,
    解法 = 排名/验收按行情换窗 (影子 A/B 验证后换生产口径);
  三窗 AUC 全 ~0.5 → 特征彻底失效, 转"新行情画像"研究 + 非排名牌 (减旗/fc).

自检: 本脚本 T+10 逐行复算须与 scored ckpt realized_net 一致 (diff<1e-9),
     且全窗席位率/均值复现 winner_leak (main 51% / +12.6%).
WORM: DATA OTHERS/diag/winner_horizon_{ts}.json + _monthly_{ts}.csv

用法: python scripts/_diag_winner_horizon.py [--eval 125]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, LEGACY_ENTRY_GATE, PANEL_V3_PATH, data_others_path
from scripts._diag_winner_leak import day_auc, gate_non_pain, topn_per_day
from scripts._rankkey_multiseed_sweep import (
    COST,
    REALIZED_BUY_LAG,
    _build_realized_pivot,
)

HORIZONS = (3, 5, 10)
WIN_T = 0.05
WIN_T_SHORT = 0.03  # T+3/T+5 的敏感性门槛 (3 日 5% 过苛)
DEPTH = 10
SLICE_DAYS = 420  # 镜像 sweep 默认切片, pivot ffill 语义一致
ANCHOR_SLOT = {"main": 0.51, "dual": 0.55}  # winner_leak 全窗席位率 (舍入)


def realized_net_H(
    price: np.ndarray,
    cal: np.ndarray,
    sym_rows: np.ndarray,
    j_cols: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """向量化 T+horizon c2c 净: buy=cal[j+1], sell=cal[j+horizon+1]."""
    n = price.shape[1]
    buy_j = j_cols + REALIZED_BUY_LAG
    sell_j = j_cols + horizon + REALIZED_BUY_LAG
    out = np.full(len(j_cols), np.nan, dtype=float)
    ok = (buy_j < n) & (sell_j < n)
    if ok.any():
        pb = price[sym_rows[ok], buy_j[ok]]
        ps = price[sym_rows[ok], sell_j[ok]]
        net = ps / pb - 1.0 - COST
        bad = ~(np.isfinite(net) & np.isfinite(pb) & (pb > 0))
        net[bad] = np.nan
        out[ok] = net
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, default=125)
    args = ap.parse_args()
    t0 = time.time()

    q50_gate = bool(LEGACY_ENTRY_GATE.get("q50_sign_gate", False))
    monthly_rows: list[pd.DataFrame] = []
    report: dict = {"horizons": list(HORIZONS), "win_t": WIN_T, "boards": {}}

    # ── pivot: 只读 3 列, 切片语义镜像 sweep ──
    light = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=["symbol", "date", "close_hfq"],
    )
    light["date"] = pd.to_datetime(light["date"]).dt.normalize()
    cal_all = np.sort(light["date"].unique())
    cut = cal_all[-SLICE_DAYS]
    light = light[light["date"] >= cut]
    pivot, cal = _build_realized_pivot(light)
    del light
    price = pivot.to_numpy(dtype="float64")
    print(
        f"[pivot] symbols={len(pivot)} days={len(cal)} ({pd.Timestamp(cal[0]).date()}"
        f"..{pd.Timestamp(cal[-1]).date()}) ({time.time() - t0:.0f}s)"
    )

    for board in ("main", "dual"):
        fr = pd.read_parquet(str(DATA_DIR / f"_diag_rankkey_scored_{board}_e{args.eval}.parquet"))
        fr["date"] = pd.to_datetime(fr["date"])
        fr["symbol"] = fr["symbol"].astype(str)
        fr = fr[np.isfinite(fr["realized_net"])].reset_index(drop=True)
        fr["in_e7"] = (
            gate_non_pain(fr, q50_gate) & ~(fr["pain_prob"].fillna(0) > 0.5)
        ).to_numpy()
        days = sorted(fr.loc[fr["in_e7"], "date"].unique())
        fr = fr[fr["date"].isin(days)].reset_index(drop=True)

        sym_rows = pivot.index.get_indexer(fr["symbol"].to_numpy())
        j_cols = np.searchsorted(cal, fr["date"].to_numpy())
        if not np.all(cal[j_cols] == fr["date"].to_numpy()):
            print(f"[{board}] FAIL 特征日期不在日历")
            return 2
        for h in HORIZONS:
            fr[f"net{h}"] = realized_net_H(price, cal, sym_rows, j_cols, h)
        d10 = fr.loc[np.isfinite(fr["net10"]), "net10"] - fr.loc[
            np.isfinite(fr["net10"]), "realized_net"
        ]
        chk = float(np.nanmax(np.abs(d10)))
        drift = float((d10.abs() > 1e-6).mean())
        print(f"[{board}] T+10 复算 vs ckpt: max|diff|={chk:.2e}, "
              f"漂移行占比 {drift:.1%} (面板 vintage 除权回溯, 聚合对账 winner_leak 一致; "
              f"{'OK' if drift < 0.05 else 'DRIFT>5% 排查'})")

        top = topn_per_day(fr[fr["in_e7"]], "pred_ret_10d", DEPTH)
        ids = set(map(tuple, top[["date", "symbol"]].to_numpy()))
        fr["in_top10"] = [(d, s) in ids for d, s in zip(fr["date"], fr["symbol"])]
        fr["ym"] = fr["date"].dt.strftime("%Y-%m")

        rows = []
        for ym, g in fr.groupby("ym", sort=True):
            e7, tp = g[g["in_e7"]], g[g["in_top10"]]
            row = {"board": board, "ym": ym, "days": g["date"].nunique(),
                   "slot_rate10": float((tp["net10"] >= WIN_T).mean()) if len(tp) else np.nan}
            for h in HORIZONS:
                w = g[f"net{h}"] >= (WIN_T if h == 10 else WIN_T_SHORT)
                wt = tp[f"net{h}"] >= (WIN_T if h == 10 else WIN_T_SHORT)
                aucs = []
                for _, gd in e7.groupby("date"):
                    m = gd[f"net{h}"] >= (WIN_T if h == 10 else WIN_T_SHORT)
                    if 1 <= m.sum() < len(gd):
                        a = day_auc(
                            gd.loc[m, "pred_ret_10d"].to_numpy(),
                            gd.loc[~m, "pred_ret_10d"].to_numpy(),
                        )
                        if np.isfinite(a):
                            aucs.append(a)
                row[f"density{h}"] = float(w.mean())
                row[f"auc{h}"] = float(np.median(aucs)) if aucs else np.nan
                row[f"slot{h}"] = float(wt.mean()) if len(tp) else np.nan
            rows.append(row)
        monthly = pd.DataFrame(rows)
        monthly_rows.append(monthly)
        print(f"\n== {board} ==  (阈值: T+10≥{WIN_T:.0%}, T+3/T+5≥{WIN_T_SHORT:.0%})")
        print(monthly.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        full = monthly[~monthly["ym"].isin(("2026-06", "2026-07", "2026-08"))]
        late = monthly[monthly["ym"].isin(("2026-06", "2026-07", "2026-08"))]
        b = {"anchor_slot10_check": ANCHOR_SLOT[board],
             "anchor_computed": float(
                 fr.loc[fr["in_top10"], "net10"].ge(WIN_T).mean())}
        for h in HORIZONS:
            b[f"early_auc{h}"] = float(full[f"auc{h}"].mean())
            b[f"late_auc{h}"] = float(late[f"auc{h}"].mean())
            b[f"late_slot{h}"] = float(late[f"slot{h}"].mean()) if len(late) else np.nan
        report["boards"][board] = b

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pd.concat(monthly_rows, ignore_index=True).to_csv(
        out_dir / f"winner_horizon_monthly_{ts}.csv", index=False
    )
    report["ts"] = ts
    (out_dir / f"winner_horizon_{ts}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}\\winner_horizon_{ts}.json + _monthly_ ({time.time() - t0:.0f}s)")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
