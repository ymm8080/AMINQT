"""_diag_breakout_predict.py — 预测 20cm 涨停事件: 前 40-60 日演化组合 (2026-08-19).

用户问题: "如何预测 8/19 的 20% 上涨, 应该看它前一段时间段的变化"
300911: 8/19 放量涨停 +20%. 预测点 = 8/18 收盘 (T), 特征 = T 之前 40-60 日演化:
  位置回落 (pos60), 筹码持续集中 (ent_chg60/gini_chg60), 缩量 (shrink20),
  波动压缩 (vol60_xr), 低位 (dd250). 目标 = T+1..T+5 内出现 20cm 涨停 (二元事件).

此前两轮检验的是 T+10 平均收益/爆发率 (个股收益口径), 用户指明预测目标是
"涨停事件本身". 本脚本测事件概率 + 组合 + 4 子窗稳定性. 全史 879d dual 池.

用法: python scripts/_diag_breakout_predict.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

COLS = [
    "symbol",
    "date",
    "close_hfq",
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "volume",
    "chip_entropy",
    "chip_gini",
]
SLICE = 879  # dual 池全史


def main() -> int:
    p = pd.read_parquet(str(PANEL_V3_PATH), columns=COLS)
    p = p[p["symbol"].str.startswith(("30", "68"))].copy()
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["dt"] = pd.to_datetime(p["date"]).dt.normalize()
    dates = sorted(pd.unique(p["dt"]))
    p = p[p["dt"] >= dates[-SLICE if SLICE < len(dates) else 0]]

    g = p.groupby("symbol", group_keys=False)
    p["ret_1d"] = g["close_hfq"].pct_change()

    # ---- 前 40-60 日演化特征 (T 日收盘可得, PIT) ----
    p["pos60"] = p["close_hfq"] / p.groupby("symbol")["close_hfq"].shift(60) - 1.0
    p["pos40"] = p["close_hfq"] / p.groupby("symbol")["close_hfq"].shift(40) - 1.0
    p["ent_chg60"] = (
        p["chip_entropy"] / p.groupby("symbol")["chip_entropy"].shift(60) - 1.0
    )
    p["gini_chg60"] = p["chip_gini"] / p.groupby("symbol")["chip_gini"].shift(60) - 1.0
    p["ent_chg40"] = (
        p["chip_entropy"] / p.groupby("symbol")["chip_entropy"].shift(40) - 1.0
    )

    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]
    p["shrink20"] = p["vr"].shift(1).rolling(20, min_periods=10).mean() < 0.8
    p["vol60"] = (
        p.groupby("symbol")["ret_1d"]
        .transform(lambda v: v.rolling(60, min_periods=40).std())
        .shift(1)
    )
    p["vol60_xr"] = p.groupby("dt")["vol60"].rank(pct=True)
    p["low_vol"] = p["vol60_xr"] < 0.3

    piv = p.pivot_table(
        index="symbol", columns="dt", values="close_hfq", aggfunc="last"
    )
    cp = piv.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    max250 = piv.rolling(250, min_periods=60, axis=1).max()
    dd250 = (
        (cp / max250.reindex(columns=cp.columns) - 1.0)
        .stack()
        .rename("dd250")
        .reset_index()
    )
    dd250.columns = ["symbol", "dt", "dd250"]
    p = p.merge(dd250, on=["symbol", "dt"], how="left")

    # ---- 目标: T+1..T+5 内出现 20cm 涨停 ----
    ret_piv = cp.pct_change(axis=1)
    fut5 = ret_piv.iloc[:, ::-1].rolling(5, min_periods=1, axis=1).max().iloc[:, ::-1]
    fut5 = fut5.shift(-1, axis=1)  # 排除 T 日, 取 T+1..T+5
    hit5 = (fut5 >= 0.195).stack().rename("hit5").reset_index()
    hit5.columns = ["symbol", "dt", "hit5"]
    p = p.merge(hit5, on=["symbol", "dt"], how="left")

    ok = p["hit5"].notna()

    def rate(sub: pd.DataFrame) -> float:
        return float(sub["hit5"].mean())

    def report(sub: pd.DataFrame, label: str) -> None:
        r = sub[sub["hit5"].notna()]
        if len(r) < 30:
            print(f"{label:<36} n={len(r):>5} (样本过少)")
            return
        print(f"{label:<36} n={len(r):>6} 5日涨停率={rate(r):>6.2%}")

    print(
        f"== 预测 20cm 涨停事件: 前 40-60 日演化 (879d dual 池, 截至 {pd.Timestamp(dates[-1]).date()}) =="
    )
    print("   [5日涨停率] = T+1..T+5 内出现 ret>=19.5% 的概率\n")
    base = rate(p[ok])
    report(p[ok], "全池基准")
    print()
    report(p[ok & (p["pos60"] < -0.20)], "60日回落>20%")
    report(p[ok & (p["ent_chg60"] < -0.10)], "60日筹码集中(entropy降>10%)")
    report(p[ok & (p["ent_chg40"] < -0.08)], "40日筹码集中")
    report(p[ok & (p["gini_chg60"] > 0.10)], "60日gini升>10%")
    report(p[ok & (p["shrink20"])], "前20日缩量")
    report(p[ok & (p["low_vol"])], "波动压缩(前60日低30%)")
    report(p[ok & (p["dd250"] < -0.30)], "低位(dd250<-30%)")
    print()
    report(p[ok & (p["pos60"] < -0.20) & (p["ent_chg60"] < -0.10)], "回落+筹码集中")
    report(
        p[ok & (p["pos60"] < -0.20) & (p["ent_chg60"] < -0.10) & p["shrink20"]],
        "回落+筹码集中+缩量",
    )
    report(
        p[
            ok
            & (p["pos60"] < -0.20)
            & (p["ent_chg60"] < -0.10)
            & p["shrink20"]
            & p["low_vol"]
        ],
        "回落+筹码集中+缩量+波动压缩",
    )
    report(
        p[
            ok
            & (p["pos60"] < -0.20)
            & (p["ent_chg60"] < -0.10)
            & p["shrink20"]
            & p["low_vol"]
            & (p["dd250"] < -0.30)
        ],
        "全组合+低位",
    )

    print()
    seg = pd.cut(p["dt"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    combo = ok & (p["pos60"] < -0.20) & (p["ent_chg60"] < -0.10) & p["shrink20"]
    for name, m in [("基准", ok), ("回落+集中+缩量", combo)]:
        print(f"{name} 子窗口:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            report(p[m & (seg == q)], f"  {q}")

    s = p[p["symbol"] == "300911"].tail(6)[
        [
            "dt",
            "close_hfq",
            "ret_1d",
            "pos60",
            "ent_chg60",
            "shrink20",
            "low_vol",
            "dd250",
            "hit5",
        ]
    ]
    print("\n300911 最近 6 日 (8/18 为预测日):")
    print(s.round(3).to_string(index=False))

    out = {
        "as_of": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        "window": f"{SLICE}d dual pool",
        "target": "T+1..T+5 内 20cm 涨停 (ret>=19.5%)",
        "base_rate": round(base, 4),
        "combo_rate": round(rate(p[combo]), 4),
        "verdict": "",
    }
    out_path = os.path.join(data_others_path("diag"), "breakout_predict_20260819.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
