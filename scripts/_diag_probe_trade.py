"""_diag_probe_trade.py — 试盘日进场规则模拟 (2026-08-19).

六轮研究结论: 试盘信号 (vr>1.5 & 0<ret<15%) 是唯一强 standalone 信号
(10 天突破率 43% vs 基准 23.7%), 但力量全在"试盘后 1-3 天直接突破"
(未洗盘 59.75%); 洗盘 3 日不突破后续突破率崩到 19.7%.

本脚本把该信号转成可执行规则测真实 P&L:
  买入: 试盘日 S 的 T+1 开盘价
  离场: 3 个交易日内 (S+1..S+3) high 首次 ≥ 买入价×1.07 → 7% 止盈卖出;
        否则 S+3 收盘卖出
  成本: 双边 0.4% (佣金+滑点)
对比: 全池基准 (任意日同规则) / 试盘日 / 试盘+筹码集中, 4 子窗.
全史 879d dual 池.

用法: python scripts/_diag_probe_trade.py
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
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "close_hfq",
    "volume",
    "chip_entropy",
]
SLICE = 879
TARGET = 0.07
COST = 0.004  # 双边 0.2%×2


def main() -> int:
    p = pd.read_parquet(str(PANEL_V3_PATH), columns=COLS)
    p = p[p["symbol"].str.startswith(("30", "68"))].copy()
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["dt"] = pd.to_datetime(p["date"]).dt.normalize()
    dates = sorted(pd.unique(p["dt"]))
    p = p[p["dt"] >= dates[-SLICE]]

    g = p.groupby("symbol", group_keys=False)
    p["ret_1d"] = g["close_hfq"].pct_change(fill_method=None)
    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]
    p["ent_chg20"] = g["chip_entropy"].pct_change(20, fill_method=None)

    # T+1 开盘买入; S+1..S+3 逐日 high 触 7% 止盈, 否则 S+3 收盘走
    p["buy"] = g["open_hfq"].shift(-1)
    p["tp"] = p["buy"] * (1 + TARGET)
    h1, h2, h3 = (
        g["high_hfq"].shift(-1),
        g["high_hfq"].shift(-2),
        g["high_hfq"].shift(-3),
    )
    c3 = g["close_hfq"].shift(-3)
    hit1 = h1 >= p["tp"]
    hit2 = (~hit1) & (h2 >= p["tp"])
    hit3 = (~hit1) & (~hit2) & (h3 >= p["tp"])
    exit_px = p["tp"].where(hit1 | hit2 | hit3, c3)
    p["trade_ret"] = exit_px / p["buy"] - 1 - COST

    ok = p["trade_ret"].notna() & p["buy"].notna()
    probe = ok & (p["vr"] > 1.5) & (p["ret_1d"] > 0) & (p["ret_1d"] < 0.15)
    collect = ok & (p["ent_chg20"] < -0.08)

    def stats(sub: pd.DataFrame, label: str) -> None:
        if len(sub) < 30:
            print(f"{label:<30} n={len(sub):>6} (样本过少)")
            return
        r = sub["trade_ret"]
        print(
            f"{label:<30} n={len(sub):>6} 均值={r.mean():+6.2%} 中位={r.median():+6.2%} "
            f"盈利占比={(r > 0).mean():5.1%} 触7%止盈={(sub['exit_win']).mean():5.1%}"
        )

    p["exit_win"] = (hit1 | hit2 | hit3).astype(float)
    print(f"== 试盘进场规则 (879d dual 池, 截至 {pd.Timestamp(dates[-1]).date()}) ==")
    print("   T+1 开盘买 → 3 日内 high 触 +7% 止盈, 否则 S+3 收盘卖, 双边成本 0.4%\n")
    stats(p[ok], "全池基准 (任意日)")
    stats(p[probe], "试盘日")
    stats(p[probe & collect], "试盘日 + 筹码集中")
    stats(p[collect], "筹码集中 alone (对照)")

    print()
    seg = pd.cut(p["dt"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    for name, m in [("基准", ok), ("试盘日", probe)]:
        print(f"{name} 子窗口:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            stats(p[m & (seg == q)], f"  {q}")

    s911 = p[p["symbol"] == "300911"]
    print("\n300911 试盘日明细 (8/13 为试盘日):")
    pd.set_option("display.width", 200)
    print(
        s911[["dt", "vr", "ret_1d", "buy", "trade_ret"]].tail(6).to_string(index=False)
    )

    out = {
        "as_of": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        "window": f"{SLICE}d dual pool",
        "rule": "T+1 open buy, 3d high>=+7% take-profit else S+3 close, cost 0.4%",
        "base_mean": round(float(p.loc[ok, "trade_ret"].mean()), 4),
        "probe_mean": round(float(p.loc[probe, "trade_ret"].mean()), 4),
        "probe_collect_mean": round(
            float(p.loc[probe & collect, "trade_ret"].mean()), 4
        ),
    }
    out_path = os.path.join(data_others_path("diag"), "probe_trade_20260819.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
