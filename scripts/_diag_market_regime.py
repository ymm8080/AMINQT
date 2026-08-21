"""_diag_market_regime.py — 市场 regime 快照诊断 (2026-08-19).

用户观察到 300911 类"低位横盘蓄势→放量爆发"行情变多, 怀疑 regime 变化.
本脚本给出市场结构当前值 vs 历史分位 (末 420 交易日):

  1. 市场宽度: 逐日上涨家数占比 / 涨停占比 (main 10% / dual 20% 口径)
  2. 高低位相对: 低位组 (dd250<-40%) vs 高位组 (dd250>-15%) 逐日等权平均收益
  3. 蓄势形态频率: 缩量5日+放量 的每日出现占比 (300911 8/13 形态)

输出: 每个指标最近 60 天 (粗线) vs 420 天历史 (中位/分位), 当前值落在历史
      分位 → 判断是否结构性变化. WORM 落盘 DATA OTHERS/diag/market_regime_<ts>.json.

用法: python scripts/_diag_market_regime.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

SLICE = 420  # 历史窗 (交易日)


def main() -> int:
    p = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=["symbol", "date", "close_hfq", "open_hfq", "high_hfq", "low_hfq", "volume"],
    )
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["dt"] = pd.to_datetime(p["date"]).dt.normalize()
    p["is_dual"] = p["symbol"].str.startswith(("30", "68"))
    dates = sorted(pd.unique(p["dt"]))
    cut = dates[-SLICE]
    p = p[p["dt"] >= cut]

    g = p.groupby("symbol", group_keys=False)
    p["ret_1d"] = g["close_hfq"].pct_change()
    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]
    prev_traded5 = p["volume"].shift(1).rolling(5, min_periods=5).min() > 0
    shrink5 = prev_traded5 & (p["vr"].shift(1).rolling(5, min_periods=4).max() < 1.0)
    brk = (p["volume"] > 0) & (p["vr"] > 1.5)
    p["acc"] = shrink5 & brk  # 蓄势突破 (300911 8/13 形态)

    # 位置 dd_250
    piv = p.pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
    max250 = piv.rolling(250, min_periods=60, axis=1).max()
    p = p.merge(
        (piv / max250 - 1.0).stack().rename("dd250").reset_index(),
        on=["symbol", "dt"], how="left",
    )

    # 剔除 8/19 不完整日 (最新日 ret_1d 用 8/18 起算仍可用, 但涨停判定含当日, 保留)
    # 逐日指标
    daily = pd.DataFrame({"dt": dates}).set_index("dt")
    daily["up_ratio"] = (
        p.groupby("dt")["ret_1d"].apply(lambda s: (s > 0).mean())
    )
    daily["limit_ratio"] = (
        p.groupby("dt")["ret_1d"].apply(
            lambda s: (s >= np.where(p.loc[s.index, "is_dual"], 0.195, 0.098)).mean()
        )
    )
    low = p["dd250"] < -0.40
    high = p["dd250"] > -0.15
    daily["low_ret"] = (
        p.loc[low].groupby("dt")["ret_1d"].mean()
    )
    daily["high_ret"] = (
        p.loc[high].groupby("dt")["ret_1d"].mean()
    )
    daily["low_high_spread"] = daily["low_ret"] - daily["high_ret"]
    daily["acc_ratio"] = (
        p.groupby("dt")["acc"].apply(lambda s: s.mean())
    )

    # 最近 60 天均线 vs 历史分位
    def _vs_hist(col: str, recent: int = 60) -> dict:
        s = daily[col].dropna()
        hist = s.iloc[:-recent] if len(s) > recent else s
        cur = s.iloc[-recent:].mean() if len(s) >= recent else s.mean()
        q = float((hist <= cur).mean())
        return {
            "current_recent60": round(float(cur), 4),
            "hist_median": round(float(hist.median()), 4),
            "hist_p25": round(float(hist.quantile(0.25)), 4),
            "hist_p75": round(float(hist.quantile(0.75)), 4),
            "hist_p90": round(float(hist.quantile(0.90)), 4),
            "current_hist_quantile": round(q, 3),
        }

    out = {
        "as_of": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        "window_days": SLICE,
        "up_ratio": _vs_hist("up_ratio"),
        "limit_ratio": _vs_hist("limit_ratio"),
        "low_ret": _vs_hist("low_ret"),
        "high_ret": _vs_hist("high_ret"),
        "low_high_spread": _vs_hist("low_high_spread"),
        "acc_ratio": _vs_hist("acc_ratio"),
    }

    ts = pd.Timestamp(dates[-1]).strftime("%Y%m%d")
    out_dir = data_others_path("diag")
    out_path = os.path.join(out_dir, f"market_regime_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    names = {
        "up_ratio": "上涨家数占比",
        "limit_ratio": "涨停占比",
        "low_ret": "低位组日均收益",
        "high_ret": "高位组日均收益",
        "low_high_spread": "低位-高位 日收益差",
        "acc_ratio": "蓄势突破形态占比",
    }
    print(f"== 市场 regime 快照 (截至 {out['as_of']}, 历史 {SLICE} 交易日) ==")
    print(f"{'指标':<14}{'近60日均值':>10}{'历史中位':>10}{'p25':>8}{'p75':>8}{'当前分位':>10} 解读")
    for k, v in out.items():
        if k in ("as_of", "window_days"):
            continue
        q = v["current_hist_quantile"]
        if q >= 0.90:
            note = ">> 历史高位区"
        elif q <= 0.10:
            note = "<< 历史低位区"
        else:
            note = "正常区间"
        print(
            f"{names[k]:<14}{v['current_recent60']:>10.4f}{v['hist_median']:>10.4f}"
            f"{v['hist_p25']:>8.4f}{v['hist_p75']:>8.4f}{q:>10.2f} {note}"
        )
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
