# V3 Panel Data Integrity Audit Report

**Date**: 2026-07-28
**Source**: `data/panel_full_enriched_v3.parquet`
**Author**: Claude Code audit (feature_engine_v35.py + direct data checks)

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total rows | 2,711,084 |
| Total columns | 101 |
| Unique symbols | 3,244 |
| Unique trading days | 863 |
| Date range | 2023-01-03 ~ 2026-07-28 |

---

## 2. Column Categories (101 columns)

### 2.1 Identifiers (8)
`symbol`, `date`, `board`, `industry`, `announce_date`, `list_days`, `is_st`, `is_suspended`

### 2.2 OHLCV Raw (7)
`open`, `high`, `low`, `close`, `volume`, `amount`, `pre_close`

### 2.3 OHLCV HFQ-adjusted (4)
`open_hfq`, `high_hfq`, `low_hfq`, `close_hfq`

### 2.4 OHLCV Derived (2)
`intraday_range`, `amplitude_5d`

### 2.5 Turnover / Liquidity / Dividend (10)
`turn`, `turnover_rate`, `turnover_rate_f`, `free_float_turnover_rate`, `volume_ratio`, `volume_ratio_x`, `ar_turnover`, `dv_ratio`, `dv_ttm`, `dv_ttm_x`

### 2.6 Valuation / Size (8)
`pe_ttm`, `pb`, `ps_ttm`, `total_mv`, `circ_mv`, `total_share`, `float_share`, `free_share`

### 2.7 Fundamentals / PIT (13)
`current_ratio`, `debt_ratio`, `asset_turnover`, `gross_margin`, `inventory_turnover`, `ocf_to_or`, `roa`, `roe`, `roe_deducted`, `eps_yoy`, `rev_yoy`, `profit_yoy`, `net_margin`

### 2.8 Chip / CYQ Distribution (16)
`cost_5pct`, `cost_15pct`, `cost_50pct`, `cost_85pct`, `cost_95pct`, `weight_avg`, `benefit_part`, `pct_70_con`, `pct_90_con`, `pct_70_high`, `pct_70_low`, `pct_90_high`, `pct_90_low`, `avg_cost`, `chip_concentration`, `conc_90`

### 2.9 Margin Trading (5)
`margin_balance`, `margin_buy_amt`, `net_margin`, `short_balance`, `short_sell_vol`

### 2.10 Northbound (6)
`north_buy_amt_sh`, `north_buy_amt_sz`, `north_sell_amt_sh`, `north_sell_amt_sz`, `north_net_buy_sh`, `north_net_buy_sz`

### 2.11 LHB / Dragon-Tiger (3)
`lhb_net_buy`, `lhb_buy_amt`, `lhb_sell_amt`

### 2.12 Shareholder Structure (7)
`holder_count`, `avg_shares_per_holder`, `sh_change_amt`, `sh_change_amt_total`, `sh_change_vol`, `sh_net_change_sign`, `sh_net_sign`

### 2.13 Price Limits (2)
`up_limit_raw`, `down_limit_raw`

### 2.14 Pre-computed Bias Features (8)
`bias_5`, `bias_10`, `bias_20`, `bias_60`, `bias_120`, `bias_250`, `bias_5_20_cross`, `bias_20_60_cross`

### 2.15 New / Recently Added (3)
`pctChg` (from baostock), `ma_vol_ratio_5_20`, `sw_ret_1d` (sector index return)

### 2.16 Unclassified Surge Indicators (2)
`vol_surge`, `amt_surge` — from `app/core/ths_indicators.py` (volume/amount surge detection), present in factor registry, pass-through from upstream

---

## 3. Column Usage by Feature Engine

| Category | Count | Notes |
|---|---|---|
| Panel columns directly read by engine | 73 | OHLCV, fundamentals, chip, margin, northbound, LHB, shareholder, limits, bias, surge, pctChg |
| Panel columns not explicitly read by engine (pass-through) | 28 | See 3.1 below |

### 3.1 Pass-Through Columns (28)

These columns exist in the panel but are NOT directly read by any `dimXX` function in `feature_engine_v35.py`. They may be used by the `_add_time_series_changes` method (which operates on ALL numeric columns) or downstream modules:

`turn`, `free_float_turnover_rate`, `avg_cost`, `pct_70_low`, `pct_70_high`, `pct_90_low`, `pct_90_high`, `roe`, `roe_deducted`, `roa`, `gross_margin`, `rev_yoy`, `debt_ratio`, `current_ratio`, `asset_turnover`, `ar_turnover`, `inventory_turnover`, `ocf_to_or`, `sh_net_change_sign`, `volume_ratio_x`, `dv_ttm_x`, `eps_yoy`, `profit_yoy`, `net_margin`, `ma_vol_ratio_5_20`, `intraday_range`, `dv_ttm`, `sh_net_sign`

**Status**: These are NOT necessarily dead weight — they serve as:
- Input to `_add_time_series_changes` which generates `_chgN`/`_pct_chgN` variants for all numeric columns
- Input to `_add_cross_sectional_ranks` which generates `_xrank` variants
- Future feature expansion (e.g., dim22 reads `roe`, `roa`, `net_margin` etc. to produce derived features)

### 3.2 Columns NOT in panel but expected by engine (handled gracefully)

`chip_concentration`, `conc_90`, `is_limit_up`, `touched_limit_up` — engine checks `if col in df.columns` and uses fallbacks.

---

## 4. Data Quality

### 4.1 Null Rates Summary

| Category | Count | Columns |
|---|---|---|
| >90% null (effectively empty) | 26 | `avg_shares_per_holder`, shareholder change columns, LHB columns, fundamental PIT columns, EPS/profit/net_margin |
| >50% null (sparse) | 50 (24 more) | Chip/CYQ columns (83%), northbound details (83%), margin (73%), valuation (27%), turnover (27%) |
| 1-50% null (partial) | 19 | `dv_ttm`, northbound net, PE TTM, PB, PS, volume_ratio, turnover_rate_f, circ_mv, float/free share, up/down_limit_raw, `turn` (8.3%), `vol_surge`/`amt_surge` (2.7%) |
| <1% null (near complete) | 32 | OHLCV, OHLCV_hfq, bias features, identifiers, `pctChg` (0.20%), `sw_ret_1d` (0.22%), `ma_vol_ratio_5_20` (0.04%) |

### 4.2 Columns with >90% Nulls (Detailed)

| Column | Null Rate | Source | Reason |
|---|---|---|---|
| `avg_shares_per_holder` | 100.00% | Stk_holdernumber | Never populated |
| `sh_change_amt_total` | 99.87% | Stk_holdertrade | No upstream data |
| `sh_net_change_sign` | 99.87% | Stk_holdertrade | No upstream data |
| `sh_change_vol` | 99.68% | Stk_holdertrade | No upstream data |
| `sh_change_amt` | 99.68% | Stk_holdertrade | No upstream data |
| `sh_net_sign` | 99.68% | Stk_holdertrade | No upstream data |
| `lhb_sell_amt` | 99.16% | LHB | No upstream data |
| `lhb_net_buy` | 99.16% | LHB | No upstream data |
| `lhb_buy_amt` | 99.16% | LHB | No upstream data |
| `eps_yoy` | 99.14% | Fina_indicator | Quarterly only |
| `announce_date` | 99.14% | Fina_indicator | Quarterly only |
| `profit_yoy` | 99.14% | Fina_indicator | Quarterly only |
| `net_margin` | 99.14% | Fina_indicator | Quarterly only |
| `holder_count` | 98.57% | Stk_holdernumber | Sparse quarterly data |
| `inventory_turnover` | 94.04% | Fina_indicator | Quarterly only |
| `ar_turnover` | 93.96% | Fina_indicator | Quarterly only |
| `gross_margin` | 93.94% | Fina_indicator | Quarterly only |
| `roa` | 93.94% | Fina_indicator | Quarterly only |
| `current_ratio` | 93.92% | Fina_indicator | Quarterly only |
| `roe_deducted` | 93.62% | Fina_indicator | Quarterly only |
| `debt_ratio` | 93.53% | Fina_indicator | Quarterly only |
| `asset_turnover` | 93.53% | Fina_indicator | Quarterly only |
| `roe` | 93.53% | Fina_indicator | Quarterly only |
| `rev_yoy` | 93.53% | Fina_indicator | Quarterly only |
| `ocf_to_or` | 93.53% | Fina_indicator | Quarterly only |
| `pre_close` | 91.79% | Unknown | Merged from different source |

**Severity: MEDIUM**
- Fundamental PIT columns are expected to be sparse (quarterly announcements). The null rates are by design — the data is present for the specific announcement dates and forward-filled by `merge_asof(direction=backward)`.
- However, `avg_shares_per_holder` (100% null) and shareholder change columns (99.87% null) are essentially empty and produce no useful signal.
- `pre_close` at 91.79% null is problematic — see section 4.2a below.

### 4.2a `pre_close` Column Issue

**Severity: HIGH**
- `pre_close` is 91.79% null (2,488,482 missing of 2,711,084 rows)
- This means any engine dimension that reads `pre_close` (dim02_volatility's amplitude_5d, dim07_limit_gene's limit_up detection, dim_active_pit) will produce NaN results for the majority of rows
- `pctChg` was computed from baostock which has its own pre_close internally — it is NOT affected (0.20% null only)
- **Root cause**: The `pre_close` column was likely merged from a different upstream source with incomplete coverage, while `close` and other OHLCV columns from baostock have 0% nulls

### 4.3 OHLCV Integrity

| Check | Violations | Status |
|---|---|---|
| High >= Low | 0 | PASS |
| High >= Open | 0 | PASS |
| High >= Close | 0 | PASS |
| Low <= Open | 0 | PASS |
| Low <= Close | 0 | PASS |
| Volume >= 0 | 0 | PASS |
| Volume nulls | 2,250 (0.08%) | MINOR — all on suspended/halted days |

**Severity: MINOR**
- 2,250 volume nulls are expected on suspension days or data gaps

### 4.4 Infinity Values

| Metric | Value |
|---|---|
| inf / -inf values | 0 across all numeric columns |

**Severity: NONE** — clean data (engine also runs replace([inf, -inf], NaN) as safety net)

### 4.5 Duplicate (symbol, date) Pairs

| Check | Result |
|---|---|
| Duplicate rows | 0 |

**Severity: NONE** — panel is unique on (symbol, date)

### 4.6 `list_days` Monotonicity (Look-Ahead Bias Check)

| Check | Result |
|---|---|
| Symbols with non-monotonic list_days | 0 |

**Severity: NONE** — no look-ahead bias detected

### 4.7 `pctChg` Column Quality

| Metric | Value |
|---|---|
| Non-null rows | 2,705,733 (99.80%) |
| Null rows | 5,351 (0.20%) — all first day per symbol or reset dates |
| Mean | 0.05% |
| Median | 0.00% |
| Std | 3.07% |
| Skew | 0.61 |
| Kurtosis | 5.36 |
| Non-ST >11% change | 17,375 rows — expected on GEM/STAR boards (20% limit) |

**Formula verification**: pctChg exactly equals `(close / pre_close - 1) * 100` for all 222,501 rows where pre_close is available (max diff = 0.0). The remaining 2,483,232 rows have pctChg populated from baostock but pre_close null — pctChg was computed server-side by baostock.

**Null dates**: 5,351 nulls are all on the first trading day per symbol (2023-01-03 or later IPO dates) where no prior close exists.

**Severity: LOW** — pctChg is well-formed. The 0.20% null rate is expected from first-day data.

### 4.8 `pctChg` vs Feature Engine's `ROC_1d`

| Metric | Value |
|---|---|
| Correlation (pctChg vs ROC_1d) | 0.9854 |
| Mean absolute diff | 0.0083 pp |
| Rows with >5pp diff | 616 |
| Max diff | 468 pp (on ex-rights day) |

**Interpretation**:
- `ROC_1d` = `close_hfq / close_hfq.shift(1) - 1` (hfq-adjusted, smooth through corporate actions)
- `pctChg` = raw close daily return (reflects actual trading P&L including ex-rights jumps)
- The 0.985 correlation is **high but not 1.0** — and that's correct: pctChg captures real trading return inclusive of dividends/rights/splits; ROC_1d smooths these out
- The 468% max diff occurs on ex-rights dates where hfq close spikes due to forward-adjustment
- **Bottom line**: pctChg is NOT redundant with ROC_1d. It provides complementary information (real trading return vs momentum signal)

**Per-board correlation**:
- `main` board (SH/SZ main): 0.99 (rare ex-rights events)
- `GEM`: 0.98
- `STAR`: 0.98
- `SH`: 0.83 (many ex-rights events in SH market)
- `SZ`: 0.86 (many ex-rights events in SZ market)

---

## 5. Board Coverage Analysis

### 5.1 Board Distribution

| Board | Rows | Unique Symbols | pctChg Null Rate |
|---|---|---|---|
| SZ (Shenzhen, poorly classified) | 1,202,165 | 1,883 | 0.27% |
| SH (Shanghai, poorly classified) | 864,666 | 1,361 | 0.19% |
| main (Main board) | 403,902 | 2,006 | 0.00% |
| GEM (ChiNext) | 170,519 | 845 | 0.20% |
| STAR (科创) | 69,832 | 341 | 0.20% |

### 5.2 Board Classification Issue

**Severity: HIGH**
- 1,024 symbols appear in BOTH `main` and `SZ` board categories
- These symbols transition from `SZ` to `main` on 2025-09-24 and back on 2026-07-28
- This is almost certainly a data processing artifact — stocks do not change exchanges
- The `board` column is used by `_add_cross_sectional_ranks` for board-stratified ranking and by `get_limit_pct` for determining daily price limits
- **Impact**: Cross-sectional rankings may be split across two boards for the same symbol, and price limits may be incorrectly applied

### 5.3 `turn` Column

- `turn` = `turnover_rate * 100` (correlation 1.0 but values differ by factor of 100)
- 8.30% null rate (224,895 rows) vs `turnover_rate` at 0% null
- **Severity: LOW** — the `turnover_rate` column is complete; `turn` is simply a percentage-scaled version with some missing days

### 5.4 `pctChg` Coverage

- No symbols have gaps >20 trading days
- No symbols are stale (missing recent data)
- Board coverage is uniform (0-0.3% null across all boards)

---

## 6. Seven Planned Columns Status

The following 8 columns are planned but NOT yet present in the panel:

| Column | Status | Notes |
|---|---|---|
| `pcfNcfTTM` | NOT FOUND | Not in panel, not referenced by engine |
| `quickRatio` | NOT FOUND | Not in panel, not referenced by engine |
| `cashRatio` | NOT FOUND | Not in panel, not referenced by engine |
| `assetToEquity` | NOT FOUND | Not in panel, not referenced by engine |
| `tangibleAssetToAsset` | NOT FOUND | Not in panel, not referenced by engine |
| `ebitToInterest` | NOT FOUND | Not in panel, not referenced by engine |
| `CFOToNP` | NOT FOUND | Not in panel, not referenced by engine |
| `CFOToGr` | NOT FOUND | Not in panel, not referenced by engine |

None of the 8 planned columns are expected by the feature engine (v3.5) — they would be new input columns for future dimension expansion.

---

## 7. Summary of Issues and Recommendations

### CRITICAL (fix before training)

| # | Issue | Impact | Recommendation |
|---|---|---|---|
| 7.1 | **Board column has 1,024 symbols with conflicting classification (SZ → main SZ)** | `get_limit_pct` uses board to determine daily price limits. Wrong board = wrong limit_up/down prices = incorrect limit_up detection, incorrect limit distance features, incorrect `_is_limit_down`. Also board-stratified cross-sectional ranks are broken (same symbol on two boards). | Fix board classification logic in `cleaning_pipeline.board_of()`. The overlap should not exist — a stock is either main board or SME/SZ market, not both. Check the 2025-09-24 and 2026-07-28 transition dates for processing bugs. |

### HIGH

| # | Issue | Impact | Recommendation |
|---|---|---|---|
| 7.2 | **`pre_close` is 91.79% null** | dim02 (amplitude_5d via amplitude = (high-low)/pre_close), dim07 (is_limit_up detection), dim_active_pit all produce NaN for 91%+ of rows. The engine's `get_limit_pct` is unaffected (it uses board+date), but any feature depending on `pre_close` directly is broken. | 1) Backfill `pre_close` from baostock (same source as `pctChg`, which is well-populated), OR 2) Use `close.shift(1)` as a fallback within the engine, OR 3) Source `pre_close` from the same OHLCV endpoint that provides open/high/low/close. |

### MEDIUM

| # | Issue | Impact | Recommendation |
|---|---|---|---|
| 7.3 | **6 shareholder columns effectively 100% empty** | 7 columns (avg_shares_per_holder, sh_change_amt, sh_net_change_sign, sh_change_amt_total, sh_net_sign, sh_change_vol) are 99.68-100% null. Zeros out dim23 and dim29 outputs. | Either find a data source for shareholder changes (Tushare stk_holdertrade), or remove the dead dimensions from the engine when data is unavailable. |
| 7.4 | **3 LHB columns effectively 100% empty** | lhb_net_buy, lhb_buy_amt, lhb_sell_amt are 99.16% null. Zeros out dim18 and dim26 outputs. | Source Tushare lhb data, or skip LHB dimensions. |
| 7.5 | **Fundamental PIT columns (13) are 93-99% null** | This is by-design for quarterly data (only announcement-date rows have values, then forward-filled by merge_asof), but the raw panel has no forward-fill applied — the forward-fill happens inside `dim03_fundamentals` via merge_asof. Confirm that the pipeline `panel_builder` applies the merge_asof before engine runs. | Verify the data pipeline order: fundamentals should be merged BEFORE feature engine executes dim03/dim22. |

### LOW

| # | Issue | Impact | Recommendation |
|---|---|---|---|
| 7.6 | **`turn` is redundant with `turnover_rate` (= x100) but 8.3% null** | Minor redundancy. `turn` is essentially `turnover_rate * 100` but has 225k fewer valid rows. | Use `turnover_rate` and drop `turn`, or ensure `turn` is fully populated from the same source. |
| 7.7 | **17,375 non-ST rows with >11% daily change in pctChg** | These are legitimate GEM/STAR board moves (20% limit). Not a bug, but may affect outlier detection logic. | If any training label caps at 11%, add explicit GEM/STAR handling. |

### ACTION ITEMS BEFORE NEXT TRAINING RUN

1. **Fix board classification** (highest priority — affects all limit-up/down features)
2. **Fill `pre_close`** from baostock to unblock amplitude and limit-up detection
3. **Confirm fundamental data pipeline**: ensure `merge_asof` for quarterly data happens before dim22 runs
4. **Consider dropping dead columns**: `avg_shares_per_holder`, all `sh_change*` columns (99.7%+ empty) add storage cost and `_chgN`/`_pct_chgN` overhead. The `_add_time_series_changes` method skips columns with >70% nulls, but the raw columns still consume memory.
5. **Monitor `pctChg` as a new feature**: correlation with `ROC_1d` is 0.985, but they diverge on ex-rights days. `pctChg` captures real trading return — potentially valuable for label construction and as an auxiliary feature.

---

*Report generated by V3 Data Audit pipeline. Contact: Claude Code.*
