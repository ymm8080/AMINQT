"""_diag_accumulate_check.py — 蓄势/横盘突破组合最终检验 + 4 子窗口稳定性 (2026-08-19).

300911 形态: 窄幅横盘 2 个月 (6-8月 18-24 元) + 筹码集中 + 缩量回调后放量突破.
检验: 波动率压缩横盘 × 放量 × 昨跌 × 低位, 及 4 子窗口稳定性.
dual 池 420 交易日, T+1 close 买 / T+11 close 卖, 扣 0.2%.

用法: python scripts/_diag_accumulate_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from config.settings import PANEL_V3_PATH

COLS = [
    "symbol", "date", "close", "high", "low", "volume",
    "close_hfq", "chip_entropy", "chip_gini",
]


def main() -> int:
    slice_days = int(sys.argv[1]) if len(sys.argv) > 1 else 420
    p = pd.read_parquet(str(PANEL_V3_PATH), columns=COLS)
    p = p[p["symbol"].str.startswith(("30", "68"))].copy()
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["dt"] = pd.to_datetime(p["date"]).dt.normalize()
    dates = sorted(pd.unique(p["dt"]))
    p = p[p["dt"] >= dates[-slice_days if slice_days < len(dates) else 0]]

    g = p.groupby("symbol", group_keys=False)
    p["ret_1d"] = g["close_hfq"].pct_change()
    p["prev_neg"] = g["ret_1d"].shift(1) < 0
    p["vol60"] = p.groupby("symbol")["ret_1d"].transform(
        lambda v: v.rolling(60, min_periods=40).std()
    )
    p["vol60_xr"] = p.groupby("dt")["vol60"].rank(pct=True)
    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]
    traded5 = p["volume"].shift(1).rolling(5, min_periods=5).min() > 0
    shrink5 = traded5 & (p["vr"].shift(1).rolling(5, min_periods=4).max() < 1.0)
    brk = (p["volume"] > 0) & (p["vr"] > 1.5)
    acc = shrink5 & brk

    piv = p.pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
    max250 = piv.rolling(250, min_periods=60, axis=1).max()
    p = p.merge(
        (piv / max250 - 1.0).stack().rename("dd250").reset_index(),
        on=["symbol", "dt"], how="left",
    )
    cp = piv.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    t10 = (cp.shift(-11, axis=1) / cp.shift(-1, axis=1) - 1.0 - 0.002).stack().rename(
        "t10"
    ).reset_index()
    t10.columns = ["symbol", "dt", "t10"]
    p = p.merge(t10, on=["symbol", "dt"], how="left")
    ok = p["t10"].notna()

    def st(sub: pd.DataFrame) -> str:
        r = sub["t10"].dropna()
        return (
            f"n={len(r):>6} 命中={float((r > 0).mean()):>5.1%} "
            f"均值={r.mean():+6.2%} 中位={r.median():+6.2%}"
        )

    print("全池基准:      ", st(p[ok]))
    print("横盘(vol60低30%)+放量:", st(p[ok & (p["vol60_xr"] < 0.3) & brk]))
    print("横盘+放量+昨跌:    ", st(p[ok & (p["vol60_xr"] < 0.3) & brk & p["prev_neg"]]))
    print("横盘+放量+低位:    ", st(p[ok & (p["vol60_xr"] < 0.3) & brk & (p["dd250"] < -0.4)]))
    print("横盘+缩量+放量+低位:", st(p[ok & (p["vol60_xr"] < 0.3) & acc & (p["dd250"] < -0.4)]))

    seg = pd.cut(p["dt"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    print()
    for name, m in [
        ("低位+蓄势+昨跌 (全部):", ok & acc & p["prev_neg"] & (p["dd250"] < -0.4)),
        ("蓄势+放量        (全部):", ok & acc),
    ]:
        print(name, st(p[m]))
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            print(f"   {q}:", st(p[m & (seg == q)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
