#!/usr/bin/env python3
"""Analyze V3 columns: which are new vs covered by dim methods."""

v3 = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "close_hfq",
    "volume",
    "amount",
    "turnover_rate",
    "pre_close",
    "turn",
    "is_suspended",
    "is_st",
    "board",
    "list_days",
    "industry",
    "free_float_turnover_rate",
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
    "sh_net_change_sign",
    "sh_change_amt_total",
    "volume_ratio_x",
    "dv_ttm_x",
    "announce_date",
    "eps_yoy",
    "profit_yoy",
    "net_margin",
    "bias_5",
    "bias_10",
    "bias_20",
    "bias_60",
    "bias_120",
    "bias_250",
    "bias_5_20_cross",
    "bias_20_60_cross",
    "ma_vol_ratio_5_20",
    "amplitude_5d",
    "pctChg",
    "intraday_range",
    "vol_surge",
    "amt_surge",
    "up_limit_raw",
    "down_limit_raw",
    "margin_balance",
    "short_balance",
    "margin_buy_amt",
    "short_sell_vol",
    "turnover_rate_f",
    "volume_ratio",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_mv",
    "circ_mv",
    "total_share",
    "float_share",
    "free_share",
    "sw_ret_1d",
    "holder_count",
    "avg_shares_per_holder",
    "sh_change_vol",
    "sh_change_amt",
    "sh_net_sign",
    "north_net_buy_sh",
    "north_net_buy_sz",
    "north_buy_amt_sh",
    "north_sell_amt_sh",
    "north_buy_amt_sz",
    "north_sell_amt_sz",
    "lhb_net_buy",
    "lhb_buy_amt",
    "lhb_sell_amt",
]

v3_set = set(v3)

# Feature engine id_cols (from feature_engine_v35.py:1912-1949)
id_cols = {
    "symbol",
    "date",
    "board",
    "industry",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "turnover_rate",
    "free_float_turnover_rate",
    "is_suspended",
    "is_st",
    "list_days",
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "close_hfq",
    "limit_pct",
    "announce_date",
    "PE_TTM",
    "touched_limit_up",
    "score_rank",
    "rank_amount",
    "rank_ff_turnover",
    "liquidity_score",
    "churn_suspect",
    "is_virtual",
    "price_1455",
    "adv20",
    "time",
    "吸筹峰",
    "红在蓝上",
}

feature_cols = sorted(v3_set - id_cols)

# Which columns have dedicated dim methods computing derived features from them
dim_map = {
    "dim03_fundamentals": {
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
        "eps_yoy",
        "profit_yoy",
        "net_margin",
    },
    "dim05_turnover_liquidity": {
        "turnover_rate_f",
        "volume_ratio",
        "dv_ratio",
        "dv_ttm",
    },
    "dim06_valuation_size": {
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv",
        "circ_mv",
        "total_share",
        "float_share",
        "free_share",
    },
    "dim07_limit_gene": {"up_limit_raw", "down_limit_raw"},
    "dim10_money_flow": {"vol_surge", "amt_surge"},
    "dim12_ma_system": {
        "bias_5",
        "bias_10",
        "bias_20",
        "bias_60",
        "bias_120",
        "bias_250",
        "bias_5_20_cross",
        "bias_20_60_cross",
        "ma_vol_ratio_5_20",
    },
    "dim18/dim26_lhb": {"lhb_net_buy", "lhb_buy_amt", "lhb_sell_amt"},
    "dim21_chip": {
        "benefit_part",
        "avg_cost",
        "weight_avg",
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
    },
    "dim22_fina_pit": {"sh_net_change_sign", "sh_change_amt_total"},
    "dim23_shareholder": {"holder_count", "avg_shares_per_holder"},
    "dim24_margin": {
        "margin_balance",
        "short_balance",
        "margin_buy_amt",
        "short_sell_vol",
    },
    "dim25_northbound": {
        "north_net_buy_sh",
        "north_net_buy_sz",
        "north_buy_amt_sh",
        "north_sell_amt_sh",
        "north_buy_amt_sz",
        "north_sell_amt_sz",
    },
    "dim28_sector_index": {"sw_ret_1d"},
    "dim29_holdertrade": {"sh_change_vol", "sh_change_amt", "sh_net_sign"},
}

all_dim_cols = set()
for cols in dim_map.values():
    all_dim_cols.update(cols)

# MISSINGNESS_COLS from feature engine (cols that get is_missing_* flags)
missingness_cols = {
    "main_money_flow",
    "chip_concentration",
    "conc_90",
    "benefit_part",
    "margin_balance",
    "north_net_buy_sh",
    "holder_count",
}

# NEUTRALIZE_COLS
neutralize_cols = {"turnover_rate", "chip_concentration", "conc_90", "benefit_part"}

print("=" * 70)
print("V3 COLUMNS — FEATURE ENGINEERING COVERAGE ANALYSIS")
print("=" * 70)

print("\n--- COLUMNS WITH DEDICATED dim METHOD ---")
for c in feature_cols:
    if c in all_dim_cols:
        for d, cols in dim_map.items():
            if c in cols:
                flags = []
                if c in missingness_cols:
                    flags.append("has_missingness_flag")
                if c in neutralize_cols:
                    flags.append("neutralized")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                print(f"  {c:30s} <- {d}{flag_str}")
                break

print("\n--- COLUMNS WITHOUT DEDICATED dim (raw pass-through only) ---")
no_dim = sorted(set(feature_cols) - all_dim_cols)
for c in no_dim:
    flags = []
    if c in missingness_cols:
        flags.append("has_missingness_flag")
    if c in neutralize_cols:
        flags.append("neutralized")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    print(f"  {c:30s}{flag_str}")

print("\n--- EXCLUDED (in id_cols) ---")
for c in sorted(v3_set & id_cols):
    print(f"  {c}")

print("\n--- SUMMARY ---")
print(f"Total V3 columns: {len(v3_set)}")
print(f"Excluded (id_cols): {len(v3_set & id_cols)}")
print(f"Auto-discovered features: {len(feature_cols)}")
print(f"  With dim method: {len(set(feature_cols) & all_dim_cols)}")
print(f"  Raw pass-through: {len(no_dim)}")
