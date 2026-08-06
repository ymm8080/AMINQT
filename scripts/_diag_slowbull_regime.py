# -*- coding: utf-8 -*-
"""慢牛 regime 条件退出验证 (2026-08-06).

stability 证明 trail8 是趋势跟随放大器: 上升段 (h4/h5) +2~4pp, 下行段 (h1/h3)
-1.4~-5.2pp。用户提议: **上升阶段用 trail8, 下降阶段不用**。本脚本验证该想法:
按买入日 PIT 市场状态打标, 分段量 cur/trail8/trail15, 并测条件策略
(上升→trail8, 下降→cur) 是否稳定跑赢现行 cur。

市场代理 = 面板每日全部股票 close_hfq 中位数 (等权、PIT、自洽)。3 种 regime 定义:
  A: mkt > MA20      (经典短趋势过滤器)
  B: mkt > MA60      (中趋势)
  C: ret20 > 0       (20 日市场动量为正)
买入日 T 打标 (入场价在 T+1, 故 T 收盘信息 PIT 安全)。WORM 落盘.
"""

from __future__ import annotations

import gc
import json
import os

import numpy as np
import pandas as pd

from app.pipeline_parallel.backtest import COST, load_panel, slippage_tier
from app.pipeline_parallel.config import OOS_WINDOWS, SLOW_BULL
from app.pipeline_parallel.scoring import pool_score, select_topn
from app.pipeline_parallel.signals import trailing_stop_price

SELL_COLS = (
    "below_ma20",
    "adx_broken",
    "big_drop",
    "below_ma5_3d",
    "turnover_spike",
    "tp_80_div",
)
MODES = ("cur", "trail5", "trail8", "trail15")
REGIME_DEFS = ("A", "B", "C")
M = 40
HARD_STOP = 0.92
N_MIN = 30
HALF = 121


def gen_picks(work: pd.DataFrame, board: str, top_n: int) -> pd.DataFrame:
    wb = work[(work["board"] == board) & work["gate_slow_bull"]]
    if wb.empty:
        return pd.DataFrame()
    score = pool_score(wb, SLOW_BULL.pool, weights=SLOW_BULL.pool_weights)
    return select_topn(wb, score, top_n)


def build_sim_arrays(work: pd.DataFrame) -> dict:
    w = work.sort_values(["symbol", "date"]).reset_index(drop=True)
    uniques, codes = np.unique(w["symbol"].values, return_inverse=True)
    sizes = np.bincount(codes)
    starts = np.zeros(len(uniques), dtype=np.int64)
    starts[1:] = np.cumsum(sizes)[:-1]
    ends = np.cumsum(sizes)
    sell = {c: w[c].values.astype(bool) for c in SELL_COLS}
    any_sell = np.zeros(len(w), dtype=bool)
    for c in SELL_COLS:
        any_sell |= sell[c]
    return {
        "sym_code": {s: int(c) for c, s in enumerate(uniques)},
        "starts": starts,
        "ends": ends,
        "dates": w["date"].values.astype("datetime64[ns]"),
        "close": w["close_hfq"].values,
        "low": w["low_hfq"].values,
        "ma20": w["ma20"].values,
        "any_sell": any_sell,
        "cost": COST + 2 * np.array([slippage_tier(v) for v in w["adv20"].values]),
    }


def exit_rets(picks: pd.DataFrame, mode: str, A: dict) -> list[float]:
    """对一批 (symbol,date) 模拟指定退出, 返回逐票净收益列表."""
    sym_code, starts, ends = A["sym_code"], A["starts"], A["ends"]
    dates_dt, close, low, ma20 = A["dates"], A["close"], A["low"], A["ma20"]
    any_sell, cost_arr = A["any_sell"], A["cost"]
    rets, holds = [], []
    for sym, T in zip(picks["symbol"], picks["date"]):
        c = sym_code[str(sym)]
        lo, hi = starts[c], ends[c]
        base = lo + int(np.searchsorted(dates_dt[lo:hi], np.datetime64(T)))
        r0 = base + 1
        if r0 + M >= hi:
            continue
        entry = close[r0]
        if not np.isfinite(entry) or entry <= 0:
            continue
        cost = cost_arr[base]
        peak = 0.0
        exit_ret = None
        for k in range(1, M + 1):
            r = base + k
            if mode == "cur":
                peak = max(peak, low[r] / entry - 1.0)
                stop_hit = low[r] < trailing_stop_price(entry, peak, ma20[r])
                if any_sell[r] or stop_hit:
                    exit_ret = close[r] / entry - 1 - cost
                    holds.append(k)
                    break
            else:
                ret = close[r] / entry - 1.0
                peak = max(peak, ret)
                trail = float(mode[5:]) / 100.0
                if ret <= peak - trail or ret <= HARD_STOP - 1.0:
                    exit_ret = close[r] / entry - 1 - cost
                    holds.append(k)
                    break
        if exit_ret is None:
            exit_ret = close[r0 + M] / entry - 1 - cost
            holds.append(M)
        rets.append(exit_ret)
    return rets


def sim_agg(rets: list[float], holds: list[int] | None = None) -> dict:
    rr = np.array(rets)
    return {
        "n": int(len(rr)),
        "p_win": round(float((rr > 0).mean()), 4) if len(rr) else None,
        "avg": round(float(rr.mean()), 4) if len(rr) else None,
        "median_hold": float(np.median(holds)) if holds else None,
    }


def main() -> int:
    work = load_panel()
    dates = np.sort(work["date"].unique())
    D = len(dates)

    # 市场代理: 每日全股票 close_hfq 中位数 (PIT)
    mkt = work.groupby("date")["close_hfq"].median().sort_index()
    ma20 = mkt.rolling(20, min_periods=20).mean()
    ma60 = mkt.rolling(60, min_periods=60).mean()
    ret20 = mkt.pct_change(20)
    reg = pd.DataFrame({"mkt": mkt, "ma20": ma20, "ma60": ma60, "ret20": ret20})
    reg["A"] = mkt > ma20
    reg["B"] = mkt > ma60
    reg["C"] = ret20 > 0

    # 窗口 (复用 stability 切法)
    windows: dict[str, tuple[np.datetime64, np.datetime64]] = {
        "full": (dates[0], dates[-1])
    }
    for i in range(6):
        s, e = i * HALF, min((i + 1) * HALF, D) - 1
        windows[f"h{i + 1}"] = (dates[s], dates[e])
    windows["oos6m"] = (dates[-OOS_WINDOWS["6m"]], dates[-1])

    picks_all = {}
    for b in ("main", "dual"):
        pk = gen_picks(work, b, SLOW_BULL.top_n)
        if not pk.empty:
            pk = pk[["symbol", "date"]].copy()
            pk = pk.merge(
                reg[["A", "B", "C"]], left_on="date", right_index=True, how="left"
            )
            for d in REGIME_DEFS:
                pk[f"r_{d}"] = pk[d].fillna(False).astype(bool)
                pk = pk.drop(columns=[d])
        picks_all[b] = pk
        print(f"[{b}] 全历史池子 {len(pk):,} 条", flush=True)
        del pk
        gc.collect()

    A = build_sim_arrays(work)
    del work
    gc.collect()

    out: dict = {
        "meta": {
            "panel": f"{dates[0]} → {dates[-1]}",
            "n_days": int(D),
            "regime_defs": {"A": "mkt>MA20", "B": "mkt>MA60", "C": "ret20>0"},
            "market_proxy": "每日全股票 close_hfq 中位数 (PIT)",
            "conditional": "上升→trail8, 下降→cur",
            "note": "买入日 T 打标 (入场 T+1, PIT 安全)",
        }
    }
    for b, pk in picks_all.items():
        out.setdefault(b, {})
        for rdef in REGIME_DEFS:
            up = pk[pk[f"r_{rdef}"]]
            dn = pk[~pk[f"r_{rdef}"]]
            cell: dict = {"n_up": int(len(up)), "n_down": int(len(dn))}
            for nm, sub in (("up", up), ("down", dn)):
                if not len(sub):
                    cell[nm] = {m: {"n": 0} for m in MODES}
                    continue
                cell[nm] = {m: sim_agg(exit_rets(sub, m, A)) for m in MODES}
            # 条件策略: 上升→trail8, 下降→cur
            up_r = exit_rets(up, "trail8", A) if len(up) else []
            dn_r = exit_rets(dn, "cur", A) if len(dn) else []
            cell["conditional"] = sim_agg(up_r + dn_r)
            # 全池 cur 基准
            cell["all_cur"] = sim_agg(exit_rets(pk, "cur", A))
            out[b][rdef] = cell
        # 每定义×窗口: 条件策略 vs cur 稳定性
        for rdef in REGIME_DEFS:
            cond_rows, cur_rows = [], []
            for wname, (d0, d1) in windows.items():
                sub = (
                    pk[(pk["date"] >= d0) & (pk["date"] <= d1)]
                    if len(pk)
                    else pd.DataFrame()
                )
                if len(sub) < N_MIN:
                    continue
                up = sub[sub[f"r_{rdef}"]]
                dn = sub[~sub[f"r_{rdef}"]]
                up_r = exit_rets(up, "trail8", A) if len(up) else []
                dn_r = exit_rets(dn, "cur", A) if len(dn) else []
                c = sim_agg(up_r + dn_r)
                cc = sim_agg(exit_rets(sub, "cur", A))
                cond_rows.append((wname, c["n"], c["avg"]))
                cur_rows.append((wname, cc["n"], cc["avg"]))
            out[b].setdefault(f"windows_{rdef}", []).append(
                {
                    "def": rdef,
                    "rows": [
                        {"window": w, "n": n, "cond_avg": c, "cur_avg": cu}
                        for (w, n, c), (_, _, cu) in zip(cond_rows, cur_rows)
                    ],
                }
            )
        del pk
        gc.collect()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_regime_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    # 打印
    for b in ("main", "dual"):
        print(f"\n=== [{b}] ===")
        for rdef in REGIME_DEFS:
            c = out[b][rdef]
            print(
                f"\n定义 {rdef} (mkt{('>MA20' if rdef == 'A' else '>MA60' if rdef == 'B' else ' ret20>0')}): "
                f"up {c['n_up']:,} / down {c['n_down']:,}"
            )
            print(
                f"  up  : cur {c['up']['cur']['avg'] * 100:7.2f}%  trail8 {c['up']['trail8']['avg'] * 100:7.2f}%  "
                f"trail15 {c['up']['trail15']['avg'] * 100:7.2f}%"
            )
            print(
                f"  down: cur {c['down']['cur']['avg'] * 100:7.2f}%  trail8 {c['down']['trail8']['avg'] * 100:7.2f}%  "
                f"trail15 {c['down']['trail15']['avg'] * 100:7.2f}%"
            )
            print(
                f"  条件 (up→trail8/down→cur): {c['conditional']['avg'] * 100:7.2f}%  vs 全池cur "
                f"{c['all_cur']['avg'] * 100:7.2f}%  (delta {((c['conditional']['avg'] - c['all_cur']['avg']) * 100):+.2f}pp)"
            )
        wr = out[b].get("windows_A", [{}])[0].get("rows", [])
        if wr:
            print("\n  窗口稳定性 (定义A, 条件 vs cur):")
            for r in wr:
                d = (r["cond_avg"] - r["cur_avg"]) * 100
                print(
                    f"    {r['window']:<7} n={r['n']:>5}  cond {r['cond_avg'] * 100:7.2f}%  "
                    f"cur {r['cur_avg'] * 100:7.2f}%  delta {d:+.2f}pp"
                )
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
