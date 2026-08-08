"""点对点(close-to-close)复测: 3d/5d 排名 vs 10d 排名 — 用户假设 (2026-08-07).

用户质疑: 10d 排名挑的是 MFE 天花板, 实得(点对点)未必好 —
后程发力票(如 000006: 前5天跌, 第10天爆拉)会让 3-5d 持有者吃亏.
用户明确要求: 评估用 close-to-close, 不是 MFE.

本脚本复用 rank_daily.parquet 的逐日池(score/mag_*), 从并行面板收盘价算
点对点收益 (口径同生产 label_pm_kd): 买=close[T+1], 卖=close[T+1+k],
ret_k = close[T+1+k]/close[T+1]-1 (每股用自己的交易日序列, 停牌跳过).
同 250d OOS 复测四个排名:
  组A = top-10 by mag_3d
  组B = top-10 by t35 (0.5·mag_3d+0.5·mag_5d)
  组C = 池内 mag_10d>0 → top-10 by t35 (10d门)
  组D = top-10 by mag_10d
  组E = top-10 by pred_comp (含10d 加权组合, 0.25/0.40/0.25/0.10) — 头对头验证 "组合 vs 纯10d"
结局 = 点对点实得均值 (2d/3d/5d/10d) + 上涨率. WORM.

用法: python scripts/_diag_10d_point_ret.py [trailing_days=250]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline_parallel.config import PANEL
from config.settings import BACKTEST_RESULT_DIR

SRC = (
    r"D:\AMINQT\DATA OTHERS\BACKTESTING RESULT"
    r"\parallel_rank_compare_20260807_033821\rank_daily.parquet"
)
BOARDS = ("main", "dual")
POOL_N, TOP_N = 30, 10
TAIL_DAYS = 450  # 覆盖评估窗 + 前向 10 交易日
H_K = (2, 3, 5, 10)
T35 = (0.5, 0.5)
# 模块 horizon_w (T+2/3/5/10, 与 config SHORTLIST_SCORE 一致) — pred_comp 键用
W = {"2d": 0.25, "3d": 0.40, "5d": 0.25, "10d": 0.10}


def load_closes() -> dict[str, np.ndarray]:
    """按 symbol 加载 (date_int64, close_hfq) 数组 (内存安全列式)."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ckpt in (PANEL.main_checkpoint, PANEL.dual_checkpoint):
        t = pq.read_table(ckpt, columns=["symbol", "date", "close_hfq"])
        df = t.to_pandas()
        dates = sorted(df["date"].unique())
        df = df[df["date"] >= dates[-TAIL_DAYS]].reset_index(drop=True)
        d64 = df["date"].to_numpy().astype("datetime64[ns]")
        for sym, g in df.groupby("symbol", sort=False):
            idx = g.index.to_numpy()
            out[str(sym)] = (d64[idx], df["close_hfq"].to_numpy()[idx])
        del t, df
        gc.collect()
    return out


def point_rets(closes, sym: str, d: np.datetime64, ks: tuple[int, ...]) -> list[float]:
    """该股在日期 d 的前向点对点收益 (close[T+1+k]/close[T+1]-1); 数据不足→nan."""
    if sym not in closes:
        return [np.nan] * len(ks)
    dates, close = closes[sym]
    i = int(np.searchsorted(dates, d))
    if i >= len(dates) or dates[i] != d or i + 1 >= len(dates):
        return [np.nan] * len(ks)
    exec_px = close[i + 1]
    if not np.isfinite(exec_px) or exec_px <= 0:
        return [np.nan] * len(ks)
    out = []
    for k in ks:
        j = i + 1 + k
        out.append(close[j] / exec_px - 1.0 if j < len(dates) else np.nan)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trailing_days", nargs="?", type=int, default=250)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"diag_10d_point_ret_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    dates = sorted(df["date"].unique())
    keep = set(dates[-args.trailing_days :])
    df = df[df["date"].isin(keep)].reset_index(drop=True)
    df["t35"] = T35[0] * df["mag_3d"] + T35[1] * df["mag_5d"]
    # 含 10d 的模块 horizon_w 加权组合 (与 config SHORTLIST_SCORE 同权重, 0.25/0.40/0.25/0.10)
    df["pred_comp"] = (
        W["2d"] * df["mag_2d"]
        + W["3d"] * df["mag_3d"]
        + W["5d"] * df["mag_5d"]
        + W["10d"] * df["mag_10d"]
    )
    print(f"[data] {len(df):,} 行 / {df['date'].nunique()} 交易日", flush=True)

    print("[panel] 列式加载收盘价 (按 symbol)...", flush=True)
    closes = load_closes()
    print(f"[panel] {len(closes):,} 只 (含前向)", flush=True)

    d_arr = df["date"].to_numpy().astype("datetime64[ns]")
    sym_arr = df["symbol"].astype(str).to_numpy()
    pr = np.full((len(df), len(H_K)), np.nan)
    for i in range(len(df)):
        pr[i] = point_rets(closes, sym_arr[i], d_arr[i], H_K)
    for col, k in zip(H_K, range(len(H_K))):
        df[f"pr_{col}d"] = pr[:, k]
    del pr
    gc.collect()

    KEYS = (
        ("mag_3d", "纯 mag_3d 排名"),
        ("t35", "T+3/T+5 组合排名"),
        ("gate_10d_t35", "10d门+组合排名"),
        ("pred_comp", "T+2/3/5/10 加权组合排名"),
        ("mag_10d", "纯 mag_10d 排名"),
        ("mag_10d_all", "纯 mag_10d 全池排名 (无 top30 预筛)"),
    )
    groups = {
        "mag_3d": (None, "mag_3d"),
        "t35": (None, "t35"),
        "gate_10d_t35": (df["mag_10d"] > 0, "t35"),
        "pred_comp": (None, "pred_comp"),
        "mag_10d": (None, "mag_10d"),
        "mag_10d_all": (None, "mag_10d"),
    }
    UNIVERSE_KEYS = {"mag_10d_all"}  # 无 score 预筛: 全板块日截面直接按 10d 排
    rows: list[dict] = []
    for _d, g in df.groupby("date"):
        for board in BOARDS:
            bd = g[g["board"] == board]
            if len(bd) < POOL_N:
                continue
            pool = bd.nlargest(POOL_N, "score")
            for key, label in KEYS:
                base = bd if key in UNIVERSE_KEYS else pool
                mask, sort_by = groups[key]
                sub = base if mask is None else base[mask]
                if sub.empty:
                    continue
                lst = sub.sort_values(sort_by, ascending=False).head(TOP_N)
                if len(lst) < 5:
                    continue
                r = lst[["pr_2d", "pr_3d", "pr_5d", "pr_10d"]].mean(skipna=True)
                rows.append(
                    {
                        "date": _d,
                        "board": board,
                        "key": key,
                        "label": label,
                        "n": len(lst),
                        "pr2d": float(r["pr_2d"]),
                        "pr3d": float(r["pr_3d"]),
                        "pr5d": float(r["pr_5d"]),
                        "pr10d": float(r["pr_10d"]),
                        "win5d": float((lst["pr_5d"] > 0).mean()),
                    }
                )

    daily = pd.DataFrame(rows)
    daily.to_csv(out_dir / "daily.csv", index=False)

    agg_rows = []
    print(
        f"\n===== 点对点(close-to-close)实得: 3d/5d vs 10d 排名 "
        f"(最近 {args.trailing_days} 交易日) =====",
        flush=True,
    )
    print("买=close[T+1], 卖=close[T+1+k]; 结局=组合内平均实得收益", flush=True)
    for board in BOARDS:
        sub = daily[daily["board"] == board]
        if sub.empty:
            continue
        print(f"\n[{board}]", flush=True)
        print(
            f"  {'组':<20}{'日':>4}{'2d':>9}{'3d':>9}{'5d':>9}{'10d':>9}{'5d涨率':>9}",
            flush=True,
        )
        for key, label in KEYS:
            g = sub[sub["key"] == key]
            if g.empty:
                continue
            r = g.mean(numeric_only=True)
            print(
                f"  {label:<20}{int(len(g)):>4}{r['pr2d']:>+9.4f}{r['pr3d']:>+9.4f}"
                f"{r['pr5d']:>+9.4f}{r['pr10d']:>+9.4f}{r['win5d']:>9.1%}",
                flush=True,
            )
            agg_rows.append(
                {
                    "board": board,
                    "key": key,
                    "label": label,
                    "n_days": int(len(g)),
                    "pr2d": round(float(r["pr2d"]), 5),
                    "pr3d": round(float(r["pr3d"]), 5),
                    "pr5d": round(float(r["pr5d"]), 5),
                    "pr10d": round(float(r["pr10d"]), 5),
                    "win5d": round(float(r["win5d"]), 4),
                }
            )
        base = sub[sub["key"] == "mag_3d"].set_index("date")["pr5d"]
        for key, label in (
            ("t35", "组合"),
            ("gate_10d_t35", "10d门"),
            ("mag_10d", "纯10d"),
        ):
            other = sub[sub["key"] == key].set_index("date")["pr5d"]
            common = base.index.intersection(other.index)
            if len(common):
                wins = int((other[common] > base[common]).sum())
                print(
                    f"  逐日 {label} 5d实得 赢 纯3d: {wins}/{len(common)} 天",
                    flush=True,
                )

    pd.DataFrame(agg_rows).to_csv(out_dir / "agg.csv", index=False)
    summary = {
        "ts": ts,
        "trailing_days": args.trailing_days,
        "metric": "close-to-close point returns (买=close[T+1], 卖=close[T+1+k]); NOT MFE",
        "t35": T35,
        "note": "rank_daily 池/幅度复用; 点对点收益从面板 close_hfq 按每股交易日序列计算; 停牌跳过",
        "boards": agg_rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nWORM: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
