"""_diag_churn_drop.py — D24 换手稳定性罚分挤出池的股票, 250d 表现如何 (2026-08-19).

用户问 8/18 该抓哪个信号才能抓住 300911. 事实链: 8/12 在池+入选,
8/13 试盘放量日 turnover_stability_5=0.527>0.5 → D24"对倒嫌疑" → liquidity_score×0.5
→ 排名 100 名级掉到 564 → 掉出 dual serving 池 top200 → 8/13-8/18 系统看不见它 → 8/19 涨停。

本脚本量化: 罚分挤出池 (罚分前≤200 且罚分后>200) 的股票, T+10 实得 (c2c, 扣 0.2%)
vs 罚分后仍留在池内的股票. 若挤出股有超额 → churn 罚分误杀正常活跃股, 池机制是主因;
若挤出股更差 → 罚分正确, 300911 掉池是数据口径差异而非机制.

用法: python scripts/_diag_churn_drop.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

COST = 0.002
TOP_N = 200
MIN_AMOUNT = 5e7
N_DAYS = 250


def _stats(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    if len(r) < 50:
        print(f"  {label:<32} n={len(r):>5} 样本过少")
        return {"label": label, "n": int(len(r)), "insufficient": True}
    out = {
        "label": label,
        "n": int(len(r)),
        "days": int(r.index.nunique()) if r.index.nlevels > 1 else 0,
        "hit": float((r > 0).mean()),
        "mean": float(r.mean()),
        "med": float(r.median()),
        "ge5": float((r >= 0.05).mean()),
        "ge10": float((r >= 0.10).mean()),
    }
    print(
        f"  {label:<32} n={len(r):>6} 命中={out['hit']:>5.1%} 实得={out['mean']:+6.2%} "
        f"中位={out['med']:+6.2%} ≥5%={out['ge5']:>5.1%} ≥10%={out['ge10']:>5.1%}"
    )
    return out


def main() -> int:
    cols = [
        "symbol",
        "date",
        "amount",
        "board",
        "free_float_turnover_rate",
        "turnover_rate",
        "close_hfq",
    ]
    p = pd.read_parquet(str(PANEL_V3_PATH), columns=cols)
    p["symbol"] = p["symbol"].astype(str)
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    end = p["date"].max()
    start = end - pd.Timedelta(days=N_DAYS * 1.8)
    p = p[(p["date"] >= start) & (p["board"].isin(("GEM", "STAR")))].copy()

    g = p.groupby("symbol")
    p["stab"] = (
        g["turnover_rate"].rolling(5, min_periods=1).std()
        / g["turnover_rate"].rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    p["t10_ret"] = (
        p.groupby("symbol")["close_hfq"].shift(-10) / p["close_hfq"].shift(-1)
        - 1.0
        - COST
    )

    rows = []
    for _dt, d in p[p["amount"] >= MIN_AMOUNT].groupby("date"):
        if len(d) < 300:
            continue
        d = d.copy()
        rk_a = d.groupby("board")["amount"].rank(pct=True)
        rk_ff = d.groupby("board")["free_float_turnover_rate"].rank(pct=True)
        d["score_raw"] = 0.5 * rk_a + 0.5 * rk_ff
        churn = (d["stab"] > 0.5).astype(int)
        d["score_pen"] = d["score_raw"] * np.where(churn == 1, 0.5, 1.0)
        d["rk_raw"] = d.groupby("board")["score_raw"].rank(
            ascending=False, method="first"
        )
        d["rk_pen"] = d.groupby("board")["score_pen"].rank(
            ascending=False, method="first"
        )
        d["in_raw"] = d["rk_raw"] <= TOP_N
        d["in_pen"] = d["rk_pen"] <= TOP_N
        d["dropped"] = d["in_raw"] & ~d["in_pen"]  # 罚分挤出池
        d["pen_kept"] = d["in_pen"] & (d["stab"] > 0.5)  # 罚了但仍留池
        rows.append(
            d[
                [
                    "date",
                    "symbol",
                    "dropped",
                    "pen_kept",
                    "stab",
                    "rk_raw",
                    "rk_pen",
                    "t10_ret",
                    "board",
                    "in_raw",
                    "in_pen",
                ]
            ]
        )
    all_d = pd.concat(rows)
    print(f"[pool] {all_d['date'].nunique()} 日 × {all_d['symbol'].nunique()} 只")
    print(f"  罚分挤出池 (in_raw & !in_pen): {all_d['dropped'].sum()} 行")
    print(f"  罚分后留池 (churn 且 in_pen):   {all_d['pen_kept'].sum()} 行")

    print("\n== 罚分挤出池的股票, T+10 实得 (c2c 扣0.2%) ==")
    groups = []
    groups.append(_stats(all_d["t10_ret"], "全池基准 (amount≥5000万)"))
    groups.append(_stats(all_d.loc[all_d["in_raw"], "t10_ret"], "池内 (罚分前 top200)"))
    groups.append(
        _stats(all_d.loc[all_d["dropped"], "t10_ret"], "挤出池 (罚分前在/罚分后出)")
    )
    groups.append(_stats(all_d.loc[all_d["pen_kept"], "t10_ret"], "罚分后仍留池"))
    groups.append(
        _stats(
            all_d.loc[all_d["dropped"] & (all_d["stab"] > 0.8), "t10_ret"],
            "挤出池 (stab>0.8 重罚)",
        )
    )

    # 分年
    all_d["year"] = all_d["date"].dt.year
    for y in (2025, 2026):
        sub = all_d[all_d["year"] == y]
        groups.append(_stats(sub.loc[sub["dropped"], "t10_ret"], f"挤出池 {y}"))
        groups.append(_stats(sub.loc[sub["pen_kept"], "t10_ret"], f"留池 {y}"))

    # 300911 具体掉池日
    s = all_d[all_d["symbol"] == "300911"]
    print("\n== 300911 逐日: 罚分排名变化 ==")
    for _, x in s.sort_values("date").iterrows():
        tag = "挤出池!" if x["dropped"] else ("留池" if x["pen_kept"] else "本就掉出")
        print(
            f"  {x['date'].date()} rk_raw={x['rk_raw']:.0f} rk_pen={x['rk_pen']:.0f} "
            f"stab={x['stab']:.2f} t10={x['t10_ret']:+.1%} [{tag}]"
        )

    out = {
        "as_of": "2026-08-19",
        "source": "panel_full_enriched_v3.parquet 近250交易日",
        "defs": {
            "dropped": "罚分前 score 排名≤200 且罚分后 >200 (D24 对倒嫌疑 ×0.5)",
            "cost": COST,
            "t10": "close[t+10]/close[t+1]-1-cost (c2c)",
        },
        "groups": groups,
    }
    out_path = os.path.join(data_others_path("diag"), "churn_drop_20260819.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
