#!/usr/bin/env python3
"""验证 v3 填充后的数据覆盖率."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V3_PATH = "data/panel_full_enriched_v3.parquet"
df = pd.read_parquet(V3_PATH)

total = len(df)
dates = sorted(df["date"].unique())
last_date = dates[-1]
last_day = df[df["date"] == last_date]

print(f"v3 面板: {total} 行 {len(df.columns)} 列")
print(f"股票数: {df['symbol'].nunique()}")
print(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"最新交易日: {last_date.date()}, 当日 {len(last_day)} 只股票")
print()

# 按数据源分组检查覆盖率
groups = {
    "行情基础": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pctChg",
        "turn",
        "pre_close",
    ],
    "涨跌停": ["up_limit_raw", "down_limit_raw"],
    "融资融券": ["margin_balance", "margin_buy_amt", "short_balance", "short_sell_vol"],
    "日线指标": [
        "turnover_rate_f",
        "volume_ratio_y",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm_y",
        "total_mv",
        "circ_mv",
        "total_share",
        "float_share",
        "free_share",
    ],
    "行业": ["sw_ret_1d"],
    "股东数": ["holder_count"],
    "股东增减持": ["sh_change_vol", "sh_change_amt", "sh_net_sign"],
    "北向资金": [
        "north_net_buy_sh",
        "north_net_buy_sz",
        "north_buy_amt_sh",
        "north_sell_amt_sh",
        "north_buy_amt_sz",
        "north_sell_amt_sz",
    ],
    "龙虎榜": ["lhb_net_buy", "lhb_buy_amt", "lhb_sell_amt"],
    "筹码分布": [
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
    ],
    "财务指标": [
        "roe",
        "roe_deducted",
        "roa",
        "gross_margin",
        "rev_yoy",
        "debt_ratio",
        "current_ratio",
        "asset_turnover",
        "ar_turnover",
        "inventory_turnover",
        "ocf_to_or",
        "net_margin",
        "eps_yoy",
        "profit_yoy",
    ],
    "量价特征": [
        "ma_vol_ratio_5_20",
        "vol_surge",
        "amt_surge",
        "volume_ratio_x",
        "dv_ttm_x",
    ],
    "公告": ["announce_date"],
}

print(f"{'数据源':<12} {'列名':<22} {'非空率':>8} {'最新日非空':>10}")
print("-" * 56)
for group, cols in groups.items():
    print(f"[{group}]")
    for c in cols:
        if c not in df.columns:
            print(f"  {c:<20} {'缺失':>8}")
            continue
        nn = df[c].notna().sum()
        ratio = nn / total * 100
        latest_nn = last_day[c].notna().sum() if c in last_day.columns else 0
        flag = " OK" if ratio > 70 else (" --" if ratio > 10 else " XX")
        print(f"  {c:<20} {ratio:>7.1f}% {latest_nn:>10}{flag}")
    print()
