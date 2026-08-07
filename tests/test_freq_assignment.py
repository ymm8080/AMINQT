"""freq_of / FREQ_ASSIGNMENT 一致性单测 — 与 _classify_freq_full.py 确认判定对齐.

核心铁律: 月频特征不进日频模型. 三频模型特征路由按基列频率,
brute-force 变体继承基列频率, 事件类归事件模块, 未知列必须显式暴露.
"""

import pytest

from app.pipeline1.feature_selector import (
    FAMILY_ANALOG,
    FREQ_ASSIGNMENT,
    FREQ_ORDER,
    freq_of,
)

# ── 与 2026-08-04 全市场×3年 6格判定一致 (scripts/_classify_freq_full.py) ──
CONFIRMED = {
    # chip 筹码
    "pct_90_con": "月",
    "pct_90_high": "月",
    "weight_avg": "月",
    "conc_trend_20d": "月",
    "resistance_dist": "月",
    "chip_entropy": "日",
    "chip_gini": "日",
    "peak_roc_5d": "日",
    "chip_skew_dist": "月",
    "conc_90_industry_rank": "月",
    "peak_price": "月",
    "peak_roc_20d": "周",
    "cost_bias": "周",
    "support_dist": "周",
    # cost
    "cost_50pct": "月",
    "cost_95pct": "月",
    # price
    "close_hfq": "周",
    # vol
    "volume": "月",
    "amount": "月",
    "turnover_rate": "月",
    "free_float_turnover_rate": "月",
    "volume_ratio": "月",
    "ma_vol_ratio_5_20": "月",
    "vol_surge": "月",
    "amt_surge": "月",
    # ma
    "bias_5": "周",
    "bias_10": "周",
    "bias_20": "月",
    "bias_60": "日",
    "bias_120": "月",
    "bias_250": "月",
    "bias_5_20_cross": "日",
    "bias_20_60_cross": "日",
    # volatility
    "amplitude_5d": "月",
    "intraday_range": "月",
    "winner_ratio": "周",
    "pctChg": "周",
    # valuation
    "pe_ttm": "周",
    "pb": "周",
    "ps_ttm": "周",
    "dv_ratio": "月",
    "total_mv": "周",
    "circ_mv": "周",
    # margin
    "margin_balance": "月",
    "short_balance": "周",
    "margin_buy_amt": "月",
    "short_sell_vol": "周",
    # fundamental
    "roe": "月",
    "roe_deducted": "月",
    "roa": "月",
    "gross_margin": "月",
    "debt_ratio": "月",
    "current_ratio": "月",
    "asset_turnover": "月",
    "ar_turnover": "月",
    "inventory_turnover": "月",
    "rev_yoy": "周",
    "net_margin": "周",
    "eps_yoy": "周",
    "profit_yoy": "周",
    "ocfps": "周",
    "revenue_ps": "周",
    "bps": "周",
    "eps": "周",
    "dt_eps": "周",
    "roe_yoy": "周",
    "q_roe": "周",
    "q_ocf_to_sales": "周",
    "ocf_to_or": "日",
}


def test_freq_assignment_matches_confirmed():
    """FREQ_ASSIGNMENT 每个确认列的频率归属与 6格判定一致."""
    for col, freq in CONFIRMED.items():
        assert col in FREQ_ASSIGNMENT, f"{col} 缺失于 FREQ_ASSIGNMENT"
        assert FREQ_ASSIGNMENT[col][0] == freq, (
            f"{col}: FREQ_ASSIGNMENT={FREQ_ASSIGNMENT[col][0]} != 判定 {freq}"
        )


def test_freq_assignment_no_extra():
    """FREQ_ASSIGNMENT 不应包含未在确认表里的列 (防手工误加)."""
    extras = set(FREQ_ASSIGNMENT) - set(CONFIRMED)
    assert not extras, f"FREQ_ASSIGNMENT 多出未确认列: {extras}"


@pytest.mark.parametrize(
    "feature,expected",
    [
        # 基列直查
        ("pct_90_con", "月"),
        ("close_hfq", "周"),
        ("chip_entropy", "日"),
        # brute-force 变体继承基列频率
        ("pct_90_con_brute_pct20", "月"),
        ("pct_90_con_brute_pct1", "月"),  # 月频列日窗口变体仍归月频表
        ("close_hfq_brute_pct5", "周"),
        ("weight_avg_brute_ma60", "月"),
        # 事件类 → 事件模块
        ("sh_net_ratio", "事件"),
        ("sh_ratio_30d", "事件"),
        ("lhb_net_buy", "事件"),
        ("lhb2_inst_flow", "事件"),
        ("bt_disc_raw", "事件"),
        ("bt_count", "事件"),
        # 未知列显式暴露
        ("roe_qoq", "未分类"),
        ("lhb_inst_net_buy_5d", "事件"),
    ],
)
def test_freq_of(feature, expected):
    assert freq_of(feature) == expected


def test_no_monthly_leaks_into_daily():
    """铁律: 月频特征不进日频模型 — 无月频列会路由到日频表."""
    daily = {c for c, (f, _) in FREQ_ASSIGNMENT.items() if f == "日"}
    monthly = {c for c, (f, _) in FREQ_ASSIGNMENT.items() if f == "月"}
    assert not (daily & monthly)
    # 全部确认列均落到 {月, 周, 日} 之一
    assert set(CONFIRMED) <= set(FREQ_ASSIGNMENT)


def test_family_analog_integrity():
    """FAMILY_ANALOG 每个条目频率合法且与 FREQ_ASSIGNMENT 无键冲突."""
    for col, (f, t) in FAMILY_ANALOG.items():
        assert f in FREQ_ORDER, f"{col} 频率 {f} 非法"
        assert t in ("TS", "XS"), f"{col} 类型 {t} 非法"
        assert col not in FREQ_ASSIGNMENT, (
            f"{col} 在 FREQ_ASSIGNMENT 与 FAMILY_ANALOG 重复"
        )


# 生产精选基列 (main 1066 / dual 30 特征的去 brute 后缀基列) 全覆盖守卫:
# 任何基列不得漏成 '未分类' — 否则三频表会丢特征.
PRODUCTION_BASES = {
    # main board
    "pct_90_con",
    "weight_avg",
    "cost_50pct",
    "close_hfq",
    "bias_5",
    "bias_60",
    "bias_20_60_cross",
    "volume",
    "amount",
    "turnover_rate",
    "vol_surge",
    "amt_surge",
    "margin_balance",
    "short_balance",
    "roe",
    "roa",
    "gross_margin",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "turn",
    "dv_ttm",
    "up_limit_raw",
    "down_limit_raw",
    "total_share",
    "float_share",
    "free_share",
    "benefit_part_x",
    "benefit_part_y",
    "churn_suspect",
    "lhb_net_buy",
    "holder_count",
    "sh_change_vol",
    # dual board
    "sw_ret_1d",
    "sw_ret_5d",
    "sw_ret_20d",
    "sector_return",
    "sector_return_5d",
    "ATR_pct",
    "sw_turnover_anomaly",
    "amihud_illiq",
    "amount_xrank",
    "market_turnover_ratio_20d",
    "market_limit_up",
    "overnight_ret",
    "month",
}


def test_production_bases_fully_covered():
    """生产精选基列全部可路由, 无 '未分类' (三频表不丢特征)."""
    unknown = [b for b in PRODUCTION_BASES if freq_of(b) == "未分类"]
    assert not unknown, f"生产基列漏分类: {unknown}"
