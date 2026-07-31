# Auto-Adoption Delta Report

**Generated**: 2026-07-29T11:52:12.305301
**Panel**: 2.7M rows × 102 cols, 3,244 symbols

## 1. Per-Dim IC BEFORE (current state)

| Dim Group | Candidates | Strong | Weak | Dead | Top IC | Top3 Mean IC |
|-----------|-----------|--------|------|------|--------|-------------|
| price_volume |  92 |   0 |  82 |  10 | 0.2838 | 0.2256 |
| volatility |  92 |   0 |  82 |  10 | 0.2838 | 0.2256 |
| fundamentals |  92 |   0 |  82 |  10 | 0.2838 | 0.2256 |
| limit_gene | 105 |   0 |  90 |  15 | 0.1950 | 0.1941 |
| sector_effect |  60 |   0 |  56 |   4 | 0.1950 | 0.1950 |
| turnover_liquidity |  60 |   0 |  56 |   4 | 0.1950 | 0.1950 |
| valuation_size |  60 |   0 |  56 |   4 | 0.1950 | 0.1950 |
| active_pit |   ? |   ? |   ? |   ? | 0.0000 | 0.0000 |
| calendar |  60 |   0 |  56 |   4 | 0.1950 | 0.1950 |
| custom_formulas | 105 |   0 |  90 |  15 | 0.1950 | 0.1941 |
| money_flow | 105 |   0 |  90 |  15 | 0.1950 | 0.1941 |
| float_limits |  84 |   0 |  76 |   8 | 0.2838 | 0.2246 |
| ma_system |  84 |   0 |  76 |   8 | 0.2838 | 0.2246 |
| holiday |  84 |   0 |  76 |   8 | 0.2838 | 0.2246 |
| market_sentiment | 105 |   0 |  90 |  15 | 0.1950 | 0.1941 |
| alpha_factors |  84 |   0 |  76 |   8 | 0.2838 | 0.2246 |
| candlestick |  84 |   0 |  76 |   8 | 0.2838 | 0.2246 |
| extended_factors |  84 |   0 |  76 |   8 | 0.2838 | 0.2246 |
| short_horizon |   ? |   ? |   ? |   ? | 0.0000 | 0.0000 |
| lhb |  62 |   0 |  59 |   3 | 0.1950 | 0.1941 |
| amihud |  62 |   0 |  59 |   3 | 0.1950 | 0.1941 |
| chip |  70 |   1 |  64 |   5 | 0.1950 | 0.1941 |
| fina_pit |  70 |   1 |  64 |   5 | 0.1950 | 0.1941 |
| shareholder |  70 |   1 |  64 |   5 | 0.1950 | 0.1941 |
| margin |  57 |   0 |  49 |   8 | 0.1950 | 0.1941 |
| lhb_enhanced |  57 |   0 |  49 |   8 | 0.1950 | 0.1941 |
| industry_flow |  94 |   0 |  80 |  14 | 0.1950 | 0.1941 |
| sector_index |  94 |   0 |  80 |  14 | 0.1950 | 0.1941 |
| holdertrade |  94 |   0 |  80 |  14 | 0.1950 | 0.1941 |
| kline_geometry |  94 |   0 |  80 |  14 | 0.1950 | 0.1941 |
| announcement |   ? |   ? |   ? |   ? | 0.0000 | 0.0000 |

**Total features evaluated**: 2321
**Total weak (usable)**: 2063

## 2. Unused Panel Columns → Auto-Adopt Candidates

| Column | NaN% | Dtype | Would Generate | Dim Group |
|--------|------|-------|---------------|-----------|
| amt_surge | 2.8% | float64 | 6 trial features | _auto_adopted |
| ar_turnover | 51.8% | float64 | 6 trial features | _auto_adopted |
| avg_cost | 0.5% | float64 | 6 trial features | _auto_adopted |
| bias_10 | 0.1% | float64 | 6 trial features | _auto_adopted |
| bias_120 | 0.1% | float64 | 6 trial features | _auto_adopted |
| bias_250 | 0.1% | float64 | 6 trial features | _auto_adopted |
| bps | 0.1% | float64 | 6 trial features | _auto_adopted |
| circ_mv | 76.1% | float64 | 0 trial features | _auto_adopted |
| dt_eps | 1.3% | float64 | 6 trial features | _auto_adopted |
| dv_ttm | 83.1% | float64 | 0 trial features | _auto_adopted |
| eps | 0.1% | float64 | 6 trial features | _auto_adopted |
| intraday_range | 0.1% | float64 | 6 trial features | _auto_adopted |
| lhb_sell_amt | 99.0% | float64 | 0 trial features | _auto_adopted |
| ocf_to_or | 49.8% | float64 | 6 trial features | _auto_adopted |
| ocfps | 0.1% | float64 | 6 trial features | _auto_adopted |
| pctChg | 0.3% | float64 | 6 trial features | _auto_adopted |
| pct_70_high | 0.5% | float64 | 6 trial features | _auto_adopted |
| pct_70_low | 0.5% | float64 | 6 trial features | _auto_adopted |
| pct_90_high | 0.5% | float64 | 6 trial features | _auto_adopted |
| pct_90_low | 0.5% | float64 | 6 trial features | _auto_adopted |
| ps_ttm | 76.1% | float64 | 0 trial features | _auto_adopted |
| q_ocf_to_sales | 0.2% | float64 | 6 trial features | _auto_adopted |
| q_roe | 0.4% | float64 | 6 trial features | _auto_adopted |
| revenue_ps | 0.2% | float64 | 6 trial features | _auto_adopted |
| roe_deducted | 49.9% | float64 | 6 trial features | _auto_adopted |
| roe_yoy | 1.7% | float64 | 6 trial features | _auto_adopted |
| sh_change_amt | 49.0% | float64 | 6 trial features | _auto_adopted |
| sh_change_vol | 49.0% | float64 | 6 trial features | _auto_adopted |
| sh_net_sign | 49.0% | float64 | 6 trial features | _auto_adopted |
| short_sell_vol | 25.8% | float64 | 6 trial features | _auto_adopted |
| vol_surge | 2.8% | float64 | 6 trial features | _auto_adopted |
| weight_avg | 0.5% | float64 | 6 trial features | _auto_adopted |

**Adoptable columns**: 28/32
**New trial features after adoption**: 168

## 3. AFTER — Projected Registry State

(_auto_adopted dim adds {total_new} trial features; IC scores TBD by next screening)

| | Features | Active | Strong | Weak | Dead |
|---|---|---|---|---|---|
| **BEFORE** | 2321 | 2066 | 3 | 2063 | 255 |
| **AFTER** | 2489 | 2234 (trial) | same | same | same |
| **DELTA** | +168 | +168 trial | 0 | 0 | 0 |

> Trial features start as `grade=trial, active=True` — they will be IC-screened in the next training window. Features that pass (>0.02 |IC|) promote to strong/weak. Features that fail 3 consecutive windows are deactivated.

## 4. Auto-Adopted Feature Manifest

| Source Column | NaN% | Trial Features |
|--------------|------|---------------|
| amt_surge | 2.8% | amt_surge_zscore_20d, amt_surge_chg5d, amt_surge_chg20d, amt_surge_sector_rank... |
| ar_turnover | 51.8% | ar_turnover_zscore_20d, ar_turnover_chg5d, ar_turnover_chg20d, ar_turnover_sector_rank... |
| avg_cost | 0.5% | avg_cost_zscore_20d, avg_cost_chg5d, avg_cost_chg20d, avg_cost_sector_rank... |
| bias_10 | 0.1% | bias_10_zscore_20d, bias_10_chg5d, bias_10_chg20d, bias_10_sector_rank... |
| bias_120 | 0.1% | bias_120_zscore_20d, bias_120_chg5d, bias_120_chg20d, bias_120_sector_rank... |
| bias_250 | 0.1% | bias_250_zscore_20d, bias_250_chg5d, bias_250_chg20d, bias_250_sector_rank... |
| bps | 0.1% | bps_zscore_20d, bps_chg5d, bps_chg20d, bps_sector_rank... |
| dt_eps | 1.3% | dt_eps_zscore_20d, dt_eps_chg5d, dt_eps_chg20d, dt_eps_sector_rank... |
| eps | 0.1% | eps_zscore_20d, eps_chg5d, eps_chg20d, eps_sector_rank... |
| intraday_range | 0.1% | intraday_range_zscore_20d, intraday_range_chg5d, intraday_range_chg20d, intraday_range_sector_rank... |
| ocf_to_or | 49.8% | ocf_to_or_zscore_20d, ocf_to_or_chg5d, ocf_to_or_chg20d, ocf_to_or_sector_rank... |
| ocfps | 0.1% | ocfps_zscore_20d, ocfps_chg5d, ocfps_chg20d, ocfps_sector_rank... |
| pctChg | 0.3% | pctChg_zscore_20d, pctChg_chg5d, pctChg_chg20d, pctChg_sector_rank... |
| pct_70_high | 0.5% | pct_70_high_zscore_20d, pct_70_high_chg5d, pct_70_high_chg20d, pct_70_high_sector_rank... |
| pct_70_low | 0.5% | pct_70_low_zscore_20d, pct_70_low_chg5d, pct_70_low_chg20d, pct_70_low_sector_rank... |
| pct_90_high | 0.5% | pct_90_high_zscore_20d, pct_90_high_chg5d, pct_90_high_chg20d, pct_90_high_sector_rank... |
| pct_90_low | 0.5% | pct_90_low_zscore_20d, pct_90_low_chg5d, pct_90_low_chg20d, pct_90_low_sector_rank... |
| q_ocf_to_sales | 0.2% | q_ocf_to_sales_zscore_20d, q_ocf_to_sales_chg5d, q_ocf_to_sales_chg20d, q_ocf_to_sales_sector_rank... |
| q_roe | 0.4% | q_roe_zscore_20d, q_roe_chg5d, q_roe_chg20d, q_roe_sector_rank... |
| revenue_ps | 0.2% | revenue_ps_zscore_20d, revenue_ps_chg5d, revenue_ps_chg20d, revenue_ps_sector_rank... |
| roe_deducted | 49.9% | roe_deducted_zscore_20d, roe_deducted_chg5d, roe_deducted_chg20d, roe_deducted_sector_rank... |
| roe_yoy | 1.7% | roe_yoy_zscore_20d, roe_yoy_chg5d, roe_yoy_chg20d, roe_yoy_sector_rank... |
| sh_change_amt | 49.0% | sh_change_amt_zscore_20d, sh_change_amt_chg5d, sh_change_amt_chg20d, sh_change_amt_sector_rank... |
| sh_change_vol | 49.0% | sh_change_vol_zscore_20d, sh_change_vol_chg5d, sh_change_vol_chg20d, sh_change_vol_sector_rank... |
| sh_net_sign | 49.0% | sh_net_sign_zscore_20d, sh_net_sign_chg5d, sh_net_sign_chg20d, sh_net_sign_sector_rank... |
| short_sell_vol | 25.8% | short_sell_vol_zscore_20d, short_sell_vol_chg5d, short_sell_vol_chg20d, short_sell_vol_sector_rank... |
| vol_surge | 2.8% | vol_surge_zscore_20d, vol_surge_chg5d, vol_surge_chg20d, vol_surge_sector_rank... |
| weight_avg | 0.5% | weight_avg_zscore_20d, weight_avg_chg5d, weight_avg_chg20d, weight_avg_sector_rank... |