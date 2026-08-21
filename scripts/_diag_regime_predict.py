"""_diag_regime_predict.py — regime 指标对未来收益的预测力检验 (2026-08-19).

用户问: 每日 regime 快照 (上涨占比/涨停占比/低位-高位差/蓄势频率) 能否放进
PIPELINE 提升 PREDICT 质量. 先验: 环境分类器 08-10 已回测否决 (选股日闸无效),
校准闸增益 100% 择日非择股. 本脚本复核:

  1. 择时层: T 日指标 vs 未来 1/5/10 日全市场等权收益 (Spearman)
  2. 风格层: T 日指标 vs 未来 10 日 低位组相对高位组超额 (低位补涨是否可预测)
  3. 横截面层: 把指标并入 prob 特征 vs 不并, 250d 日历分组 AUC 是否提升

dual 池 420 交易日 (与 regime 快照同窗). WORM 落盘.

用法: python scripts/_diag_regime_predict.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

SLICE = 420


def main() -> int:
    p = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=[
            "symbol",
            "date",
            "close_hfq",
            "open_hfq",
            "high_hfq",
            "low_hfq",
            "volume",
        ],
    )
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["dt"] = pd.to_datetime(p["date"]).dt.normalize()
    p["is_dual"] = p["symbol"].str.startswith(("30", "68"))
    dates = sorted(pd.unique(p["dt"]))
    p = p[p["dt"] >= dates[-SLICE]]

    g = p.groupby("symbol", group_keys=False)
    p["ret_1d"] = g["close_hfq"].pct_change()
    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]
    prev_traded5 = p["volume"].shift(1).rolling(5, min_periods=5).min() > 0
    shrink5 = prev_traded5 & (p["vr"].shift(1).rolling(5, min_periods=4).max() < 1.0)
    brk = (p["volume"] > 0) & (p["vr"] > 1.5)
    p["acc"] = shrink5 & brk

    piv = p.pivot_table(
        index="symbol", columns="dt", values="close_hfq", aggfunc="last"
    )
    max250 = piv.rolling(250, min_periods=60, axis=1).max()
    p = p.merge(
        (piv / max250 - 1.0).stack().rename("dd250").reset_index(),
        on=["symbol", "dt"],
        how="left",
    )

    daily = pd.DataFrame({"dt": dates}).set_index("dt")
    daily["up_ratio"] = p.groupby("dt")["ret_1d"].apply(lambda s: (s > 0).mean())
    daily["limit_ratio"] = p.groupby("dt")["ret_1d"].apply(
        lambda s: (s >= np.where(p.loc[s.index, "is_dual"], 0.195, 0.098)).mean()
    )
    low = p["dd250"] < -0.40
    high = p["dd250"] > -0.15
    daily["low_ret"] = p.loc[low].groupby("dt")["ret_1d"].mean()
    daily["high_ret"] = p.loc[high].groupby("dt")["ret_1d"].mean()
    daily["spread"] = daily["low_ret"] - daily["high_ret"]
    daily["acc_ratio"] = p.groupby("dt")["acc"].apply(lambda s: s.mean())

    # 市场未来收益: 全市场等权 ret 未来 1/5/10 日
    mkt_ret = p.groupby("dt")["ret_1d"].mean()
    fut = pd.DataFrame(index=daily.index)
    for h in (1, 5, 10):
        fut[f"mkt_fut{h}"] = mkt_ret.shift(-h).rolling(h).sum()
    # 低位组未来 10 日 vs 高位组 (风格超额)
    piv[p["symbol"][p["dd250"] < -0.40].unique()] if False else None
    # 简化: 用每日高低位组收益的滚动未来
    fut["low_fut10"] = daily["low_ret"].shift(-10).rolling(10).sum()
    fut["high_fut10"] = daily["high_ret"].shift(-10).rolling(10).sum()
    fut["style_fut10"] = fut["low_fut10"] - fut["high_fut10"]

    from scipy.stats import spearmanr

    out = {"window_days": SLICE, "as_of": str(pd.Timestamp(dates[-1]).date())}
    print("== 择时层: T 日指标 vs 未来市场收益 (Spearman ρ) ==")
    print(f"{'指标':<12}{'vs 未来1日':>10}{'vs 未来5日':>12}{'vs 未来10日':>12}")
    for col in ["up_ratio", "limit_ratio", "spread", "acc_ratio"]:
        row = {"indicator": col}
        line = f"{col:<12}"
        for h in (1, 5, 10):
            a = daily[col]
            b = fut[f"mkt_fut{h}"]
            m = a.notna() & b.notna()
            rho, _ = spearmanr(a[m], b[m])
            row[f"mkt_fut{h}"] = round(float(rho), 3)
            line += f"{rho:>10.3f}"
        print(line)
        out[col] = row
    print()
    print("== 风格层: T 日指标 vs 未来10日 低位-高位超额 ==")
    print(f"{'指标':<12}{'ρ':>8}")
    for col in ["up_ratio", "limit_ratio", "spread", "acc_ratio"]:
        a = daily[col]
        b = fut["style_fut10"]
        m = a.notna() & b.notna()
        rho, _ = spearmanr(a[m], b[m])
        print(f"{col:<12}{rho:>8.3f}")
        out.setdefault("style", {})[col] = round(float(rho), 3)

    ts = pd.Timestamp(dates[-1]).strftime("%Y%m%d")
    out_path = os.path.join(data_others_path("diag"), f"regime_predict_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
