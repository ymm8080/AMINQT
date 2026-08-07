"""慢牛 trail8 稳定性验证 (2026-08-06).

closetrail 在末 6 个月 OOS (2026-01-26→08-04) 发现 trail8 (收盘回落 8% 走 + 硬止损 -8%)
优于现行 cur 退出 (main +0.82% / dual +1.16% vs +0.43%/+0.47%)。本脚本把同一退出对比
拆到全 3y 面板的多个历史窗口, 验证该优势是否跨市场状态稳定, 还是单段过拟合。

窗口 (按买入日期切分, 退出用买入后行情):
  full    = 2023-08-07 → 面板末 (全窗, 参考)
  h1..h6  = 6 个连续 ~121 交易日 (半年) 段
  oos6m   = 末 126 交易日 (对标 closetrail 参照)

每窗口×板块: cur/trail5/trail8/trail15 同池同票, 仅退出规则不同 (逻辑同 closetrail).
向量化生成池子 (pool_score+select_topn 在全 gate 子集一次算, 逐日结果与
daily_slowbull_pool 逐日调用等价: cross_rank 按 date 分组, select_topn 按 date 取 Top-N).
判定 trail8 稳定: 多数有效窗口 trail8.avg > cur.avg 且 p_win 不恶化. WORM 落盘.
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
M = 40
HARD_STOP = 0.92
N_MIN = 30  # 窗口有效判定最小样本
HALF = 121  # 半年段交易日数 (726 = 6×121)


def gen_picks(work: pd.DataFrame, board: str, top_n: int) -> pd.DataFrame:
    """向量化生成该板全历史每日 Top-N 池 (与 daily_slowbull_pool 逐日调用等价)."""
    wb = work[(work["board"] == board) & work["gate_slow_bull"]]
    if wb.empty:
        return pd.DataFrame()
    score = pool_score(wb, SLOW_BULL.pool, weights=SLOW_BULL.pool_weights)
    return select_topn(wb, score, top_n)


def build_sim_arrays(work: pd.DataFrame) -> dict:
    """一次性构建向量化模拟数组 (symbol→区间, close/low/ma20/卖出信号/成本)."""
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


def sim(picks: pd.DataFrame, mode: str, A: dict) -> dict:
    """对一批 (symbol,date) 模拟指定退出, 返回 {n, p_win, avg, median_hold}."""
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
    rr = np.array(rets)
    return {
        "n": int(len(rr)),
        "p_win": round(float((rr > 0).mean()), 4),
        "avg": round(float(rr.mean()), 4),
        "median_hold": float(np.median(holds)) if holds else None,
    }


def main() -> int:
    work = load_panel()
    dates = np.sort(work["date"].unique())
    D = len(dates)

    # 窗口定义 (按买入日期)
    windows: dict[str, tuple[np.datetime64, np.datetime64]] = {
        "full": (dates[0], dates[-1])
    }
    for i in range(6):
        s, e = i * HALF, min((i + 1) * HALF, D) - 1
        windows[f"h{i + 1}"] = (dates[s], dates[e])
    windows["oos6m"] = (dates[-OOS_WINDOWS["6m"]], dates[-1])

    # 每板块全历史池子生成一次, 再按窗口过滤 (避免重叠窗口重复计算)
    picks_all = {}
    for b in ("main", "dual"):
        pk = gen_picks(work, b, SLOW_BULL.top_n)
        if not pk.empty:
            pk = pk[["symbol", "date"]].copy()
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
            "max_hold": M,
            "hard_stop": HARD_STOP,
            "n_min": N_MIN,
            "half_window_days": HALF,
            "note": "窗口按买入日期切分; 退出用买入后行情; "
            "cur=现行6信号+low触线移动止盈, trailX=收盘回落X%走+硬止损-8%",
        }
    }
    for b, pk in picks_all.items():
        for wname, (d0, d1) in windows.items():
            sub = (
                pk[(pk["date"] >= d0) & (pk["date"] <= d1)]
                if len(pk)
                else pd.DataFrame()
            )
            out.setdefault(wname, {})[b] = (
                {"n_picks": int(len(sub))}
                if sub.empty
                else {"n_picks": int(len(sub)), **{m: sim(sub, m, A) for m in MODES}}
            )
        del pk
        gc.collect()

    # 每板块稳定性摘要
    out["summary"] = {}
    for b in ("main", "dual"):
        judge = [
            w
            for w in ("h1", "h2", "h3", "h4", "h5", "h6", "oos6m")
            if out[w][b].get("cur", {}).get("n", 0) >= N_MIN
            and out[w][b].get("trail8", {}).get("n", 0) >= N_MIN
        ]
        margins = [out[w][b]["trail8"]["avg"] - out[w][b]["cur"]["avg"] for w in judge]
        pwin_delta = [
            out[w][b]["trail8"]["p_win"] - out[w][b]["cur"]["p_win"] for w in judge
        ]
        out["summary"][b] = {
            "judge_windows": judge,
            "n_windows": len(judge),
            "trail8_beats_cur": sum(1 for m in margins if m > 0),
            "pwin_not_worse": sum(1 for d in pwin_delta if d >= 0),
            "margin_mean_pp": round(float(np.mean(margins)) * 100, 2)
            if margins
            else None,
            "margin_min_pp": round(float(np.min(margins)) * 100, 2)
            if margins
            else None,
            "margin_max_pp": round(float(np.max(margins)) * 100, 2)
            if margins
            else None,
        }

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_stability_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    # 打印矩阵
    print("\n=== 稳定性矩阵 (avg 净收益%) ===")
    hdr = "窗口        " + "".join(f"{m:>10}" for m in MODES)
    for b in ("main", "dual"):
        print(f"\n[{b}]")
        print(hdr)
        for wname in ("full", "h1", "h2", "h3", "h4", "h5", "h6", "oos6m"):
            r = out[wname][b]
            if "n_picks" in r and r["n_picks"] == 0:
                print(f"{wname:<10}" + "   (空池)")
                continue
            cells = []
            for m in MODES:
                v = r.get(m, {})
                n = v.get("n", 0)
                flag = "" if n >= N_MIN else "!"
                cells.append(f"{v.get('avg', 0) * 100:>8.2f}%{flag}")
            print(f"{wname:<10}" + "".join(f"{c:>10}" for c in cells))
    print("\n=== 稳定性摘要 (判定窗口=6半年+oos6m 中 n≥30 者) ===")
    for b, s in out["summary"].items():
        print(
            f"[{b}] 判定窗口 {s['n_windows']} 个: trail8 跑赢 cur "
            f"{s['trail8_beats_cur']} 个, p_win 不恶化 {s['pwin_not_worse']} 个, "
            f"margin 均值 {s['margin_mean_pp']}pp (min {s['margin_min_pp']} / max {s['margin_max_pp']})"
        )
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
