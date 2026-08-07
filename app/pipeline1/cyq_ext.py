"""CYQ 筹码分布扩展导出 (V3 候选列).

独立模块 (不修改 cyq_calculator.py 的基础导出接口): 从 150 档筹码分布
再导出 20 个统计列:

  - 成本分位线: 10%/20%/30%/40%/60%/70%/80%
  - 集中度:     60%/80% (计算口径与 70%/90% 一致)
  - 峰:         众数价位 peak_price + 峰质量 peak_mass
  - 分布形态:   chip_entropy (熵), chip_gini (基尼), chip_skew_dist (偏度)
  - 筹码位置:   mass_above_close / 1.1x / 1.2x, mass_below_0.9x
  - 支撑压力:   resistance_dist / support_dist (最近局部极值, 距离按现价归一)

算法与 cyq_calculator._compute_cyq_one_day 完全一致 (本地副本),
保证扩展列与基础列在同一 xdata 分布上自洽.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .cyq_calculator import FACTOR, RANGE_DAYS, _safe_div

BASE_OUTPUT_COLS = [
    "winner_ratio",
    "avg_cost",
    "pct_70_low",
    "pct_70_high",
    "pct_70_con",
    "pct_90_low",
    "pct_90_high",
    "pct_90_con",
    "cost_5pct",
    "cost_15pct",
    "cost_50pct",
    "cost_85pct",
    "cost_95pct",
    "weight_avg",
]

EXTENDED_OUTPUT_COLS = [
    "cost_10pct",
    "cost_20pct",
    "cost_30pct",
    "cost_40pct",
    "cost_60pct",
    "cost_70pct",
    "cost_80pct",
    "pct_60_con",
    "pct_80_con",
    "peak_price",
    "peak_mass",
    "chip_entropy",
    "chip_gini",
    "chip_skew_dist",
    "mass_above_close",
    "mass_above_1_1x",
    "mass_above_1_2x",
    "mass_below_0_9x",
    "resistance_dist",
    "support_dist",
    "peak_roc_5d",
    "peak_roc_20d",
]

# 接入 V3 面板 + 特征注册的目标列 (peak_mass 已由用户排除; 2026-08-02 加回
# chip_gini/resistance_dist/support_dist — 用户裁决全部保留)
TARGET_COLS = [
    "peak_price",
    "chip_entropy",
    "chip_skew_dist",
    "chip_gini",
    "resistance_dist",
    "support_dist",
    "peak_roc_5d",
    "peak_roc_20d",
]


def _compute_cyq_one_day(
    index: int,
    records: list[dict],
    factor: int = FACTOR,
    range_days: int = RANGE_DAYS,
) -> dict:
    """单日筹码分布 + 扩展导出 (副本, 核心算法与 cyq_calculator 一致)."""
    start = max(0, index - range_days + 1)
    kdata = records[start : max(1, index + 1)]

    maxprice = 0.0
    minprice = 0.0
    for k in kdata:
        maxprice = k["high"] if maxprice == 0 else max(maxprice, k["high"])
        minprice = k["low"] if minprice == 0 else min(minprice, k["low"])

    accuracy = max(0.01, (maxprice - minprice) / (factor - 1))
    xdata = [0.0] * factor

    for k in kdata:
        o, c, h, l = k["open"], k["close"], k["high"], k["low"]  # noqa: E741
        avg = (o + c + h + l) / 4.0
        turnover_rate = min(1.0, (k.get("hsl", k.get("turnover_rate", 0)) or 0) / 100.0)
        hsl_val = turnover_rate

        H = math.floor((h - minprice) / accuracy)
        L_idx = math.ceil((l - minprice) / accuracy)
        density = factor - 1 if h == l else _safe_div(2.0, h - l, factor - 1)
        avg_bucket = math.floor((avg - minprice) / accuracy)

        for n in range(factor):
            xdata[n] *= 1.0 - hsl_val

        if h == l:
            idx = avg_bucket
            if 0 <= idx < factor:
                xdata[idx] += density * hsl_val / 2.0
        else:
            for j in range(L_idx, H + 1):
                if j < 0 or j >= factor:
                    continue
                curprice = minprice + accuracy * j
                if curprice <= avg:
                    xdata[j] += (
                        _safe_div(curprice - l, avg - l, 1.0) * density * hsl_val
                    )
                else:
                    xdata[j] += (
                        _safe_div(h - curprice, h - avg, 1.0) * density * hsl_val
                    )

    for n in range(factor):
        xdata[n] = max(0.0, xdata[n])

    total_chips = sum(xdata)
    current_price = records[index]["close"]

    def cost_at_percentile(pct: float) -> float:
        target = total_chips * pct
        cumsum = 0.0
        for i in range(factor):
            cumsum += xdata[i]
            if cumsum >= target:
                return minprice + i * accuracy
        return minprice + (factor - 1) * accuracy

    def compute_percent_chips(pct: float) -> dict:
        if total_chips == 0:
            return {"priceRange": [0.0, 0.0], "concentration": 0.0}
        ps = [(1.0 - pct) / 2.0, (1.0 + pct) / 2.0]
        pr = [cost_at_percentile(ps[0]), cost_at_percentile(ps[1])]
        denom = pr[0] + pr[1]
        return {
            "priceRange": pr,
            "concentration": _safe_div(pr[1] - pr[0], denom, 0.0),
        }

    # ---- 基础导出 (与 cyq_calculator 一致) ----
    if total_chips > 0:
        below = 0.0
        for i in range(factor):
            if current_price >= minprice + i * accuracy:
                below += xdata[i]
        winner_ratio = below / total_chips
    else:
        winner_ratio = 0.5

    avg_cost = cost_at_percentile(0.5)
    if total_chips > 0:
        weighted_avg = (
            sum((minprice + i * accuracy) * xdata[i] for i in range(factor))
            / total_chips
        )
    else:
        weighted_avg = current_price

    pc70 = compute_percent_chips(0.7)
    pc90 = compute_percent_chips(0.9)

    # ---- 扩展导出 ----
    prices = [minprice + i * accuracy for i in range(factor)]
    total = total_chips if total_chips > 0 else 1.0
    p = [max(0.0, x / total) for x in xdata]

    peak_idx = max(range(factor), key=lambda i: xdata[i])
    peak_price = prices[peak_idx]
    peak_mass = p[peak_idx]

    chip_entropy = -sum(pi * math.log(pi) for pi in p if pi > 1e-12)

    sorted_p = sorted(p)
    n_p = factor
    chip_gini = (
        (2 * sum(i * x for i, x in enumerate(sorted_p, start=1)))
        / (n_p * sum(sorted_p))
        - (n_p + 1) / n_p
        if sum(sorted_p) > 0
        else 0.0
    )

    dist_mean = sum(pi * prices[i] for i, pi in enumerate(p))
    dist_var = sum(pi * (prices[i] - dist_mean) ** 2 for i, pi in enumerate(p))
    dist_std = math.sqrt(dist_var) if dist_var > 0 else 0.0
    chip_skew_dist = (
        sum(pi * ((prices[i] - dist_mean) / dist_std) ** 3 for i, pi in enumerate(p))
        if dist_std > 1e-12
        else 0.0
    )

    def _mass_above(price_th: float) -> float:
        if total_chips <= 0:
            return 0.0
        return (
            sum(xdata[i] for i in range(factor) if prices[i] > price_th) / total_chips
        )

    mass_above_close = _mass_above(current_price)
    mass_above_1_1x = _mass_above(current_price * 1.1)
    mass_above_1_2x = _mass_above(current_price * 1.2)
    mass_below_0_9x = 1.0 - _mass_above(current_price * 0.9) if total_chips > 0 else 0.0

    def _nearest_extremum(above: bool) -> float:
        best = None
        for i in range(1, factor - 1):
            if xdata[i] < xdata[i - 1] or xdata[i] <= xdata[i + 1]:
                continue
            is_above = prices[i] > current_price
            if is_above != above:
                continue
            dist = (
                abs(prices[i] - current_price) / current_price
                if current_price > 0
                else 0.0
            )
            if best is None or dist < best:
                best = dist
        return best if best is not None else 0.0

    resistance_dist = _nearest_extremum(above=True)
    support_dist = _nearest_extremum(above=False)

    pc60 = compute_percent_chips(0.6)
    pc80 = compute_percent_chips(0.8)

    return {
        "winner_ratio": winner_ratio,
        "avg_cost": avg_cost,
        "pct_70_low": pc70["priceRange"][0],
        "pct_70_high": pc70["priceRange"][1],
        "pct_70_con": pc70["concentration"],
        "pct_90_low": pc90["priceRange"][0],
        "pct_90_high": pc90["priceRange"][1],
        "pct_90_con": pc90["concentration"],
        "cost_5pct": cost_at_percentile(0.05),
        "cost_15pct": cost_at_percentile(0.15),
        "cost_50pct": cost_at_percentile(0.50),
        "cost_85pct": cost_at_percentile(0.85),
        "cost_95pct": cost_at_percentile(0.95),
        "weight_avg": weighted_avg,
        "cost_10pct": cost_at_percentile(0.10),
        "cost_20pct": cost_at_percentile(0.20),
        "cost_30pct": cost_at_percentile(0.30),
        "cost_40pct": cost_at_percentile(0.40),
        "cost_60pct": cost_at_percentile(0.60),
        "cost_70pct": cost_at_percentile(0.70),
        "cost_80pct": cost_at_percentile(0.80),
        "pct_60_con": pc60["concentration"],
        "pct_80_con": pc80["concentration"],
        "peak_price": peak_price,
        "peak_mass": peak_mass,
        "chip_entropy": chip_entropy,
        "chip_gini": chip_gini,
        "chip_skew_dist": chip_skew_dist,
        "mass_above_close": mass_above_close,
        "mass_above_1_1x": mass_above_1_1x,
        "mass_above_1_2x": mass_above_1_2x,
        "mass_below_0_9x": mass_below_0_9x,
        "resistance_dist": resistance_dist,
        "support_dist": support_dist,
    }


def _compute_cyq_for_stock(
    kdata: pd.DataFrame,
    factor: int = FACTOR,
    range_days: int = RANGE_DAYS,
) -> pd.DataFrame:
    """为单只股票逐日计算筹码分布 (含扩展导出). 跳过预热期 RANGE_DAYS/2."""
    records = kdata.sort_values("date").to_dict(orient="records")
    n = len(records)

    rows = []
    start_idx = int(range_days / 2)
    for idx in range(start_idx, n):
        r = _compute_cyq_one_day(idx, records, factor, range_days)
        r["date"] = records[idx]["date"]
        rows.append(r)

    out_cols = ["date"] + BASE_OUTPUT_COLS + EXTENDED_OUTPUT_COLS
    if not rows:
        return pd.DataFrame(columns=out_cols)
    out = pd.DataFrame(rows)
    # peak_roc_5d/20d: 按 symbol 序列对 peak_price 的 pct_change (与 build_variant_c 一致)
    out["peak_roc_5d"] = out["peak_price"].pct_change(5, fill_method=None)
    out["peak_roc_20d"] = out["peak_price"].pct_change(20, fill_method=None)
    return out[out_cols]


def compute_cyq_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """全市场面板 → 扩展 CYQ 面板 (基础15 + 扩展22)."""
    parts = []
    for sym, g in panel.groupby("symbol"):
        g = g.sort_values("date")
        cyq = _compute_cyq_for_stock(g)
        cyq["symbol"] = sym
        parts.append(cyq)
    result = pd.concat(parts, ignore_index=True)
    return result[
        ["symbol", "date"] + [c for c in result.columns if c not in ("symbol", "date")]
    ]


def compute_cyq_today(
    hist: pd.DataFrame,
    today: pd.DataFrame,
    factor: int = FACTOR,
    range_days: int = RANGE_DAYS,
) -> pd.DataFrame:
    """为今日每只股票计算 TARGET_COLS (8 个目标列).

    hist:  面板历史 (需含 symbol/date/open/high/low/close/turnover_rate/peak_price,
           peak_price 在回填后存在, 用于 roc 分母)
    today: 今日行 (含 symbol/open/high/low/close/turnover_rate)
    返回: symbol + TARGET_COLS, 每 symbol 一行.
    """
    hist_g = {sym: g for sym, g in hist.groupby("symbol")}
    rows = []
    for sym, g_today in today.groupby("symbol"):
        g_hist = hist_g.get(sym)
        if g_hist is not None:
            g_hist = g_hist.sort_values("date")
            kdata = pd.concat([g_hist.tail(range_days), g_today], ignore_index=True)
        else:
            kdata = g_today
        records = kdata.sort_values("date").to_dict(orient="records")
        r = _compute_cyq_one_day(len(records) - 1, records, factor, range_days)
        r["date"] = records[-1]["date"]
        new_peak = r["peak_price"]
        if g_hist is not None and "peak_price" in g_hist.columns:
            pp = g_hist["peak_price"]
            r["peak_roc_5d"] = new_peak / pp.iloc[-5] - 1 if len(pp) >= 6 else np.nan
            r["peak_roc_20d"] = new_peak / pp.iloc[-20] - 1 if len(pp) >= 21 else np.nan
        else:
            r["peak_roc_5d"] = np.nan
            r["peak_roc_20d"] = np.nan
        r["symbol"] = sym
        rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["symbol"] + TARGET_COLS)
    out = pd.DataFrame(rows)
    return out[["symbol"] + TARGET_COLS]
