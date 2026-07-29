# -*- coding: utf-8 -*-
"""CYQ 筹码分布计算器 — Python 移植自东方财富 JS 算法.

原算法: akshare stock_cyq_em (https://quote.eastmoney.com)

只依赖 OHLCV + 换手率, 计算日频筹码分布:
  - 获利盘比例 (benefit_part)
  - 平均成本 (avg_cost)
  - 70%/90% 筹码集中度 (concentration)
  - 成本分位线 (5%/15%/50%/85%/95%)

算法核心: 120日滑动窗口内, 按换手率衰减旧筹码 + 三角形分布加新筹码,
累计150档价格分桶后按分位数取成本线.
"""

from __future__ import annotations

import math

import pandas as pd

FACTOR = 150  # 价格分桶数
RANGE_DAYS = 120  # 筹码计算回溯窗口


def _compute_cyq_one_day(
    index: int,
    records: list[dict],
    factor: int = FACTOR,
    range_days: int = RANGE_DAYS,
) -> dict:
    """计算单日筹码分布 (对应 JS CYQCalculator)."""
    start = max(0, index - range_days + 1)
    kdata = records[start : max(1, index + 1)]

    # ---- 价格范围 (窗口内最高/最低) ----
    maxprice = 0.0
    minprice = 0.0
    for k in kdata:
        maxprice = k["high"] if maxprice == 0 else max(maxprice, k["high"])
        minprice = k["low"] if minprice == 0 else min(minprice, k["low"])

    accuracy = max(0.01, (maxprice - minprice) / (factor - 1))

    # ---- 筹码分布累积 ----
    xdata = [0.0] * factor

    for k in kdata:
        o, c, h, l = k["open"], k["close"], k["high"], k["low"]  # noqa: E741
        avg = (o + c + h + l) / 4.0
        turnover_rate = min(1.0, (k.get("hsl", k.get("turnover_rate", 0)) or 0) / 100.0)
        hsl_val = turnover_rate  # JS: Math.min(1, eles.hsl / 100 || 0)

        H = math.floor((h - minprice) / accuracy)
        L_idx = math.ceil((l - minprice) / accuracy)
        density = factor - 1 if h == l else 2.0 / (h - l)  # GPoint[0]
        avg_bucket = math.floor((avg - minprice) / accuracy)  # GPoint[1]

        # ---- 衰减旧筹码 ----
        for n in range(factor):
            xdata[n] *= 1.0 - hsl_val

        # ---- 加新筹码 (三角形分布) ----
        if h == l:
            # 一字板: 筹码集中在均价
            idx = avg_bucket
            if 0 <= idx < factor:
                xdata[idx] += density * hsl_val / 2.0
        else:
            for j in range(L_idx, H + 1):
                if j < 0 or j >= factor:
                    continue
                curprice = minprice + accuracy * j
                if curprice <= avg:
                    if abs(avg - l) < 1e-8:
                        xdata[j] += density * hsl_val
                    else:
                        xdata[j] += (curprice - l) / (avg - l) * density * hsl_val
                else:
                    if abs(h - avg) < 1e-8:
                        xdata[j] += density * hsl_val
                    else:
                        xdata[j] += (h - curprice) / (h - avg) * density * hsl_val

    # xdata 非负剪裁
    for n in range(factor):
        xdata[n] = max(0.0, xdata[n])

    total_chips = sum(xdata)
    current_price = records[index]["close"]

    # ---- 辅助函数 ----
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
            "concentration": (pr[1] - pr[0]) / denom if denom != 0 else 0.0,
        }

    # ---- 输出 ----
    # 获利盘比例
    if total_chips > 0:
        below = 0.0
        for i in range(factor):
            if current_price >= minprice + i * accuracy:
                below += xdata[i]
        benefit_part = below / total_chips
    else:
        benefit_part = 0.5

    avg_cost = cost_at_percentile(0.5)

    # 加权平均成本
    if total_chips > 0:
        weighted_avg = (
            sum((minprice + i * accuracy) * xdata[i] for i in range(factor))
            / total_chips
        )
    else:
        weighted_avg = current_price

    pc70 = compute_percent_chips(0.7)
    pc90 = compute_percent_chips(0.9)

    return {
        "benefit_part": benefit_part,
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
    }


def _compute_cyq_for_stock(
    kdata: pd.DataFrame,
    factor: int = FACTOR,
    range_days: int = RANGE_DAYS,
) -> pd.DataFrame:
    """为单只股票逐日计算筹码分布.

    kdata 需含列: date, open, close, high, low, turnover_rate
    按 date 升序排列.

    对于前 range_days 个交易日 (预热期), 筹码数据不足 120 天窗口,
    结果逐渐收敛; LightGBM 原生处理 NaN, IC 筛选可剔除早期不稳定的点.
    """
    records = kdata.sort_values("date").to_dict(orient="records")
    n = len(records)

    rows = []
    # 跳过前 RANGE_DAYS/2 天 (预热不足, 筹码分布不稳定)
    start_idx = int(range_days / 2)
    for idx in range(start_idx, n):
        r = _compute_cyq_one_day(idx, records, factor, range_days)
        r["date"] = records[idx]["date"]
        rows.append(r)

    if not rows:
        # 返回空结构
        cols = [
            "date",
            "benefit_part",
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
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows)[
        [
            "date",
            "benefit_part",
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
    ]


def compute_cyq_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """全市场面板 → CYQ 筹码分布面板 (逐股计算, 安全网 #5).

    panel 需含: symbol, date, open, close, high, low, turnover_rate
    """
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


# ---- 快速验证入口 ----
if __name__ == "__main__":
    import sys

    panel = pd.read_parquet("data/panel_3y.parquet")
    stock = panel[panel["symbol"] == "002881"].copy()
    if len(stock) == 0:
        print("002881 not in panel")
        sys.exit(1)

    cyq = _compute_cyq_for_stock(stock)
    recent = cyq.tail(5)
    for _, row in recent.iterrows():
        d = str(row["date"])[:10]
        print(
            f"{d} | 90%集中度={row['pct_90_con'] * 100:.2f}% | "
            f"获利盘={row['benefit_part'] * 100:.1f}% | "
            f"90%区间=[{row['pct_90_low']:.2f}, {row['pct_90_high']:.2f}] | "
            f"均价={row['weight_avg']:.2f}"
        )
