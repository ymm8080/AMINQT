# Pipeline 1 Feature Selection — Final Solution

**Date**: 2026-07-29
**Test**: 12-agent A/B/C matrix → 5000 brute-force → 3-year validation
**Data**: v3 panel (2023-01-03 ~ 2026-07-29, 3 years for MAIN; 1 year for DUAL)

---

## Final Pipeline

| | **MAIN Board** (CSI 300) | **DUAL Board** (GEM/STAR) |
|---|---|---|
| **Data window** | **3 years** | **1 year** |
| **Pre-filter** | NaN only (>95% drop) | NaN only |
| **Feature pool** | 5000 brute-force | 420 curated (FeatureEngineV35) |
| **Module** | Dedup L2 (\|r\|>0.7 per base group) | Gate D (forward ablation, 95% sat, min=30) |
| **IC screen?** | No | No |
| **Var filter?** | No | No |

## Performance (3Y MAIN / 1Y DUAL)

| | **MAIN (3Y)** | **DUAL (1Y)** |
|---|---|---|
| **Pool → Selected** | 3,232 → 1,069 | 429 → 30 |
| **OOS Rank IC** | **+0.0035** | +0.0072 |
| **ICIR** | 0.017 | 0.045 |
| **Top-10 Sharpe** | **+0.32** (★ first positive!) | -0.41 |
| **Top-10 Win Rate** | **49.8%** | 48.1% |
| **Composite** | **+0.243** (★ first positive!) | -0.005 |

> Composite = ICIR × 0.40 + Sharpe × 0.35 + WinRate × 0.25

## 3Y vs 1Y Comparison

| | MAIN 1Y | MAIN 3Y | DUAL 1Y | DUAL 3Y |
|---|---|---|---|---|
| IC | -0.005 | **+0.004** | +0.007 | -0.002 |
| Sharpe | -0.60 | **+0.32** | -0.41 | -1.86 |
| Composite | -0.089 | **+0.243** | -0.005 | -0.535 |

**MAIN 3Y beats 1Y across all metrics.** DUAL 3Y degrades → keep 1Y.

## Key Findings

1. **5000 brute > 420 curated on Main**: More features → more signal after dedup
2. **Gate D only on Dual, not Main**: Main board Gate D always underperforms Dedup L2
3. **IC screening always hurts**: Module's own selection (Dedup/Gate D) is sufficient
4. **Var filter always hurts**: NaN-only is the best pre-filter
5. **3 years data critical for Main**: Cross-regime training produces first positive Sharpe
6. **1 year better for Dual**: Gate D on longer data overfits

## Discarded

- IC screen (pre/post): degrades all pipelines
- Gate D on Main: all configs < Dedup L2
- Gate D + Dedup combo: worse than solo
- 5000 pool on Dual: no improvement
- Var filter: removes useful signals
