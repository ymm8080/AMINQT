"""_diag_ambush_breakout.py — 试盘→洗盘→突破 两阶段模式检验 (2026-08-19).

用户描述的统一做法: "大概十天内就有了上涨信号 (试盘), 但一直在横盘洗盘,
洗干净了再突破". 300911 例: 8/13 放量试盘 (vr 1.85, +3.2%) → 8/14-8/18
横盘洗盘 (不破位) → 8/19 放量涨停突破.

此前三轮检验只测"状态→未来涨", 未测"试盘信号 + 洗盘期"两阶段组合. 本脚本:

  试盘日 S: vr > 1.5 且 0 < ret < 15% (放量上涨未涨停, 埋伏阶段)
  洗盘期 S+1..S+3: 无突破级上涨 (max ret < 7%) 且 不破位 (min low >= S 日 low)
  目标: S+1..S+10 内出现突破 (max ret >= 7%) — "十天内突破"

  对比: 全池基准 (任意日未来 10 天突破率) vs 试盘 alone vs 试盘+洗盘,
        + 涨停口径 (>=19.5%), + 4 子窗稳定性. 全史 879d dual 池.

用法: python scripts/_diag_ambush_breakout.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

COLS = ["symbol", "date", "close_hfq", "low_hfq", "volume"]
SLICE = 879


def main() -> int:
    p = pd.read_parquet(str(PANEL_V3_PATH), columns=COLS)
    p = p[p["symbol"].str.startswith(("30", "68"))].copy()
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["dt"] = pd.to_datetime(p["date"]).dt.normalize()
    dates = sorted(pd.unique(p["dt"]))
    p = p[p["dt"] >= dates[-SLICE if SLICE < len(dates) else 0]]

    g = p.groupby("symbol", group_keys=False)
    p["ret_1d"] = g["close_hfq"].pct_change()
    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]

    piv_c = p.pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
    piv_l = p.pivot_table(index="symbol", columns="dt", values="low_hfq", aggfunc="last")
    cp = piv_c.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    lp = piv_l.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    ret_piv = cp.pct_change(axis=1)

    def fwd_rolling(df: pd.DataFrame, win: int, fn) -> pd.DataFrame:
        return fn(df.iloc[:, ::-1].rolling(win, min_periods=1, axis=1)).iloc[:, ::-1]

    fut10_max_ret = fwd_rolling(ret_piv, 10, lambda r: r.max())  # S+1..S+10 max ret
    fut3_max_ret = fwd_rolling(ret_piv, 3, lambda r: r.max())    # S+1..S+3 max ret
    fut3_min_low = fwd_rolling(lp, 3, lambda r: r.min())         # S+1..S+3 min low

    def join(name: str, df: pd.DataFrame) -> pd.DataFrame:
        out = df.stack().rename(name).reset_index()
        out.columns = ["symbol", "dt", name]
        return p.merge(out, on=["symbol", "dt"], how="left")

    p = join("fut10_max_ret", fut10_max_ret)
    p = join("fut3_max_ret", fut3_max_ret)
    p = join("fut3_min_low", fut3_min_low)
    p["wash"] = (p["fut3_max_ret"] < 0.07) & (p["fut3_min_low"] >= p["low_hfq"])

    ok = p["fut10_max_ret"].notna()
    probe = ok & (p["vr"] > 1.5) & (p["ret_1d"] > 0) & (p["ret_1d"] < 0.15)

    def stats(sub: pd.DataFrame, label: str) -> None:
        if len(sub) < 30:
            print(f"{label:<34} n={len(sub):>6} (样本过少)")
            return
        print(
            f"{label:<34} n={len(sub):>6} 10天突破≥7%={float((sub['fut10_max_ret'] >= .07).mean()):>6.2%} "
            f"涨停≥19.5%={float((sub['fut10_max_ret'] >= .195).mean()):>6.2%}"
        )

    print(f"== 试盘→洗盘→突破 两阶段模式 (879d dual 池, 截至 {pd.Timestamp(dates[-1]).date()}) ==")
    print("   洗盘 = S+1..S+3 无突破级上涨 (<7%) 且不破 S 日低点\n")
    stats(p[ok], "全池基准")
    stats(p[probe], "试盘日 alone")
    stats(p[probe & p["wash"]], "试盘 + 洗盘")
    stats(p[probe & ~p["wash"]], "试盘 + 未洗盘 (对照)")

    print()
    seg = pd.cut(p["dt"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    for name, m in [("基准", ok), ("试盘+洗盘", probe & p["wash"])]:
        print(f"{name} 子窗口:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            stats(p[m & (seg == q)], f"  {q}")

    s = p[p["symbol"] == "300911"].tail(8)[
        ["dt", "close_hfq", "ret_1d", "vr", "fut3_max_ret", "wash", "fut10_max_ret"]
    ]
    print("\n300911 最近 8 日 (8/13 为试盘日):")
    print(s.round(3).to_string(index=False))

    out = {
        "as_of": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        "window": f"{SLICE}d dual pool",
        "probe_def": "vr>1.5 & 0<ret<15%",
        "wash_def": "S+1..S+3 max ret<7% & min low>=S low",
        "target": "S+1..S+10 max ret>=7%",
        "base_rate": round(float((p[ok]["fut10_max_ret"] >= 0.07).mean()), 4),
        "probe_rate": round(float((p[probe]["fut10_max_ret"] >= 0.07).mean()), 4),
        "probe_wash_rate": round(float((p[probe & p["wash"]]["fut10_max_ret"] >= 0.07).mean()), 4),
    }
    out_path = os.path.join(data_others_path("diag"), "ambush_breakout_20260819.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
