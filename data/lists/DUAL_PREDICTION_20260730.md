# DUAL Board Prediction List — 2026-07-30

**Pipeline**: 3-Layer (Gate D) | **Model**: `dual_2026W31.pkl` | **Features**: 30 selected from 294 | **Window**: 1Y rolling

---

## Summary

| Metric | Value |
|--------|-------|
| Candidates | 15 |
| Board Mix | GEM: 10, STAR: 5 |
| Score Range | -0.858 ~ +1.355 |
| Mean pred_ret_1d | +5.04% |
| Mean prob_up | 69.1% |
| Mean uncertainty_width | 0.101 |
| Mean pain_prob | 0.410 |

---

## Column Reference

| Column | Description |
|--------|-------------|
| `symbol` | Stock ticker |
| `board` | GEM (创业板) / STAR (科创板) |
| `pred_ret_1d` | Predicted 1-day return (LightGBM reg) |
| `pred_ret_3d` | Predicted 3-day return |
| `pred_ret_5d` | Predicted 5-day return |
| `prob_up` | Platt-calibrated probability of positive return (1d_cls) |
| `score` | Composite selection score (1d:0.25 + 3d:0.45 + 5d:0.35) |
| `pred_q10` | **E1 Quantile** — 10th percentile predicted return (downside) |
| `pred_q50` | **E1 Quantile** — median predicted return |
| `pred_q90` | **E1 Quantile** — 90th percentile predicted return (upside) |
| `uncertainty_width` | **E1** — q90 − q10 spread (wider = more uncertain) |
| `pain_prob` | **E2 Pain Alert** — probability of >3% drawdown within 5 days |
| `momentum` | Momentum classification (low/neutral/high) |
| `weight` | Position weight (inverse volatility scaling) |
| `market_state` | Market regime (bull/bear/range) |
| `is_limit_up_close` | Stock hit daily limit-up |
| `is_one_word_limit` | One-word limit (一字板) |
| `consensus_score` | Model consensus across horizons |
| `signal_conflict` | Conflict flag between reg and cls signals |
| `announce_score` | Announcement event score |
| `schema_version` | List schema version (1.2) |

---

## Full List (Ranked by Score)

### #1 — 300903 (GEM)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **+1.355** | pred_ret_1d | **+5.14%** |
| pred_ret_3d | +6.31% | pred_ret_5d | +9.32% |
| prob_up | 61.4% | momentum | low |
| pred_q10 | -5.98% | pred_q50 | +2.92% |
| pred_q90 | +6.10% | **uncertainty** | **0.121** |
| **pain_prob** | **40.4%** | weight | 0.054 |

### #2 — 300655 (GEM)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **+0.954** | pred_ret_1d | **+8.20%** |
| pred_ret_3d | +6.04% | pred_ret_5d | +5.48% |
| prob_up | 73.2% | momentum | low |
| pred_q10 | -4.17% | pred_q50 | +3.87% |
| pred_q90 | +6.04% | **uncertainty** | **0.102** |
| **pain_prob** | **43.9%** | weight | 0.064 |

### #3 — 688313 (STAR)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **+0.479** | pred_ret_1d | **+5.37%** |
| pred_ret_3d | +1.86% | pred_ret_5d | +7.74% |
| prob_up | 61.1% | momentum | low |
| pred_q10 | -5.30% | pred_q50 | +2.51% |
| pred_q90 | +8.29% | **uncertainty** | **0.136** |
| **pain_prob** | **49.7%** | weight | 0.048 |

### #4 — 688122 (STAR)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **+0.414** | pred_ret_1d | **+3.59%** |
| pred_ret_3d | +0.85% | pred_ret_5d | +6.03% |
| prob_up | 73.8% | momentum | low |
| pred_q10 | -1.36% | pred_q50 | +3.02% |
| pred_q90 | +5.40% | **uncertainty** | **0.068** |
| **pain_prob** | **48.3%** | weight | 0.097 |

### #5 — 301392 (GEM)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **+0.349** | pred_ret_1d | **+6.24%** |
| pred_ret_3d | +3.12% | pred_ret_5d | +3.77% |
| prob_up | 69.3% | momentum | low |
| pred_q10 | -3.32% | pred_q50 | +3.08% |
| pred_q90 | +5.92% | **uncertainty** | **0.092** |
| **pain_prob** | **40.0%** | weight | 0.071 |

### #6 — 300153 (GEM)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **+0.328** | pred_ret_1d | **+4.97%** |
| pred_ret_3d | -0.72% | pred_ret_5d | +4.93% |
| prob_up | 60.2% | momentum | low |
| pred_q10 | -4.22% | pred_q50 | +2.73% |
| pred_q90 | +6.20% | **uncertainty** | **0.104** |
| **pain_prob** | **41.0%** | weight | 0.063 |

### #7 — 688325 (STAR)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **+0.205** | pred_ret_1d | **+7.69%** |
| pred_ret_3d | +8.92% | pred_ret_5d | +6.09% |
| prob_up | 86.2% | momentum | low |
| pred_q10 | -6.03% | pred_q50 | +3.44% |
| pred_q90 | +6.56% | **uncertainty** | **0.126** |
| **pain_prob** | **49.7%** | weight | 0.052 |

### #8 — 301021 (GEM)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **+0.157** | pred_ret_1d | **+7.82%** |
| pred_ret_3d | +8.21% | pred_ret_5d | +6.48% |
| prob_up | 70.4% | momentum | low |
| pred_q10 | -3.16% | pred_q50 | +3.95% |
| pred_q90 | +6.79% | **uncertainty** | **0.100** |
| **pain_prob** | **31.0%** | weight | 0.066 |

### #9 — 301489 (GEM)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **+0.047** | pred_ret_1d | **+6.37%** |
| pred_ret_3d | +8.01% | pred_ret_5d | +4.62% |
| prob_up | 73.7% | momentum | low |
| pred_q10 | -4.41% | pred_q50 | +3.34% |
| pred_q90 | +4.96% | **uncertainty** | **0.094** |
| **pain_prob** | **45.0%** | weight | 0.070 |

### #10 — 300382 (GEM)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **-0.151** | pred_ret_1d | **+5.48%** |
| pred_ret_3d | +2.22% | pred_ret_5d | +2.16% |
| prob_up | 72.5% | momentum | low |
| pred_q10 | -3.20% | pred_q50 | +2.48% |
| pred_q90 | +5.65% | **uncertainty** | **0.088** |
| **pain_prob** | **34.7%** | weight | 0.074 |

### #11 — 300726 (GEM)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **-0.466** | pred_ret_1d | **+5.22%** |
| pred_ret_3d | +5.08% | pred_ret_5d | -0.68% |
| prob_up | 64.8% | momentum | low |
| pred_q10 | -3.40% | pred_q50 | +3.40% |
| pred_q90 | +6.67% | **uncertainty** | **0.101** |
| **pain_prob** | **33.4%** | weight | 0.065 |

### #12 — 688578 (STAR)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **-0.626** | pred_ret_1d | **+1.44%** |
| pred_ret_3d | -1.39% | pred_ret_5d | -2.91% |
| prob_up | 64.3% | momentum | low |
| pred_q10 | -4.65% | pred_q50 | +1.67% |
| pred_q90 | +4.58% | **uncertainty** | **0.092** |
| **pain_prob** | **39.8%** | weight | 0.071 |

### #13 — 688180 (STAR)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **-0.793** | pred_ret_1d | **+2.95%** |
| pred_ret_3d | +0.94% | pred_ret_5d | -0.09% |
| prob_up | 77.4% | momentum | low |
| pred_q10 | -2.73% | pred_q50 | +2.07% |
| pred_q90 | +5.89% | **uncertainty** | **0.086** |
| **pain_prob** | **32.2%** | weight | 0.076 |

### #14 — 300835 (GEM)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **-0.844** | pred_ret_1d | **+5.13%** |
| pred_ret_3d | +2.96% | pred_ret_5d | +4.29% |
| prob_up | 66.6% | momentum | low |
| pred_q10 | -3.99% | pred_q50 | +3.97% |
| pred_q90 | +8.36% | **uncertainty** | **0.124** |
| **pain_prob** | **45.4%** | weight | 0.053 |

### #15 — 300171 (GEM)
| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Score | **-0.858** | pred_ret_1d | **+0.72%** |
| pred_ret_3d | -0.15% | pred_ret_5d | +5.43% |
| prob_up | 62.0% | momentum | low |
| pred_q10 | -4.70% | pred_q50 | +1.63% |
| pred_q90 | +3.60% | **uncertainty** | **0.083** |
| **pain_prob** | **41.5%** | weight | 0.079 |

---

## Risk Dashboard

| Symbol | Board | Score | pred_1d | prob_up | pred_q10 | pred_q90 | uncertainty | pain_prob |
|--------|-------|-------|---------|---------|----------|----------|-------------|-----------|
| 300903 | GEM | +1.355 | +5.14% | 61.4% | -5.98% | +6.10% | 0.121 | 40.4% |
| 300655 | GEM | +0.954 | +8.20% | 73.2% | -4.17% | +6.04% | 0.102 | 43.9% |
| 688313 | STAR | +0.479 | +5.37% | 61.1% | -5.30% | +8.29% | 0.136 | **49.7%** |
| 688122 | STAR | +0.414 | +3.59% | 73.8% | -1.36% | +5.40% | 0.068 | 48.3% |
| 301392 | GEM | +0.349 | +6.24% | 69.3% | -3.32% | +5.92% | 0.092 | 40.0% |
| 300153 | GEM | +0.328 | +4.97% | 60.2% | -4.22% | +6.20% | 0.104 | 41.0% |
| 688325 | STAR | +0.205 | +7.69% | 86.2% | -6.03% | +6.56% | 0.126 | **49.7%** |
| 301021 | GEM | +0.157 | +7.82% | 70.4% | -3.16% | +6.79% | 0.100 | **31.0%** |
| 301489 | GEM | +0.047 | +6.37% | 73.7% | -4.41% | +4.96% | 0.094 | 45.0% |
| 300382 | GEM | -0.151 | +5.48% | 72.5% | -3.20% | +5.65% | 0.088 | 34.7% |
| 300726 | GEM | -0.466 | +5.22% | 64.8% | -3.40% | +6.67% | 0.101 | 33.4% |
| 688578 | STAR | -0.626 | +1.44% | 64.3% | -4.65% | +4.58% | 0.092 | 39.8% |
| 688180 | STAR | -0.793 | +2.95% | 77.4% | -2.73% | +5.89% | 0.086 | 32.2% |
| 300835 | GEM | -0.844 | +5.13% | 66.6% | -3.99% | +8.36% | 0.124 | 45.4% |
| 300171 | GEM | -0.858 | +0.72% | 62.0% | -4.70% | +3.60% | 0.083 | 41.5% |

---

## Distribution

| Stat | pred_ret_1d | prob_up | uncertainty | pain_prob | score |
|------|-------------|---------|-------------|-----------|-------|
| **Mean** | +5.04% | 69.1% | 0.101 | 41.0% | +0.024 |
| **Std** | 2.44% | 7.3% | 0.019 | 6.1% | 0.666 |
| **Min** | +0.72% | 60.2% | 0.068 | 31.0% | -0.858 |
| **25%** | +3.59% | 61.4% | 0.090 | 34.7% | -0.626 |
| **50%** | +5.22% | 69.3% | 0.101 | 41.5% | +0.157 |
| **75%** | +6.37% | 73.7% | 0.120 | 45.4% | +0.349 |
| **Max** | +8.20% | 86.2% | 0.136 | 49.7% | +1.355 |

---

## Pipeline Trace

```
Layer 1: data/factor_registry/features_dual_20260730T115732.parquet
  → 300 DUAL stocks (GEM+STAR), 1Y rolling window (2025-07-30 ~ 2026-07-27)
  → FeatureEngineV35: 294 curated features (32 dim groups, cross_sectional_rank=True)

Layer 2: data/factor_registry/selected_dual_20260730T115834.json
  → NaN filter (>95%): 294 → 265
  → Gate D (forward ablation, min=30, sat=95%): 265 → 30 features
  → Best ICIR @ n=10: 0.3781

Layer 3: models/pipeline1/dual_2026W31.pkl
  → LightGBM: 4 base models (1d_reg, 3d_reg, 5d_reg, 1d_cls)
  → E1: 5 quantile models (q10/q25/q50/q75/q90)
  → E2: Pain prediction model (≥3% drawdown, next 5d)
  → LambdaRank: Learning-to-rank ensemble
  → OOS IC(1d) = 0.7893 | switched = True | 94s training

Prediction: 2026-07-30 (15 candidates)
  → data/lists/list_20260730.parquet
```

---

*Generated 2026-07-30 12:10 UTC+8 | Pipeline: 3-Layer Gate D | Model: dual_2026W31*
*⚠ NOT investment advice. All predictions are model outputs for research purposes only.*
