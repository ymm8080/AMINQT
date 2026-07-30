# -*- coding: utf-8 -*-
"""Auto-Adoption Delta Report: per-dim IC before vs after.
只算 delta impact — 不跑完整 seed.
"""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

REPORT_PATH = "data/factor_registry/auto_adoption_delta_report.md"

# ── 1. Load existing per-dim IC data ──
registry_dir = "data/factor_registry"
dim_files = sorted(
    f for f in os.listdir(registry_dir)
    if f.startswith("ic_dim_main_") and f.endswith(".json")
)

per_dim = {}  # dim_name → summary
for f in dim_files:
    with open(os.path.join(registry_dir, f), encoding="utf-8") as fh:
        data = json.load(fh)
    for dim in data.get("dims", []):
        per_dim[dim] = {
            "group": data["dim_group"],
            "candidates": data.get("n_candidates", 0),
            "strong": data.get("n_strong", 0),
            "weak": data.get("n_weak", 0),
            "dead": data.get("n_dead", 0),
            "selected": data.get("n_selected", 0),
            "top_ic_max": max(
                (abs(t.get("ic_1d", 0)) for t in data.get("top_20", [])),
                default=0.0,
            ),
            "top_3_mean_ic": np.mean(sorted(
                [abs(t.get("ic_1d", 0)) for t in data.get("top_20", [])],
                reverse=True,
            )[:3]) if data.get("top_20") else 0.0,
        }

# ── 2. Identify unused panel columns ──
# From earlier audit: 33 panel cols not referenced by feature engine
UNUSED_COLS = [
    "eps", "ocfps", "bps", "revenue_ps",        # new fina_indicator per-share
    "dt_eps", "roe_deducted", "roe_yoy",         # extra fina
    "q_roe", "q_ocf_to_sales",
    "ocf_to_or", "ar_turnover", "circ_mv",
    "ps_ttm", "dv_ttm",
    "bias_10", "bias_120", "bias_250",           # extra bias periods
    "pct_70_low", "pct_70_high",                  # chip detail
    "pct_90_low", "pct_90_high",
    "avg_cost", "weight_avg",
    "intraday_range", "pctChg", "vol_surge", "amt_surge",  # intraday
    "lhb_sell_amt", "short_sell_vol",             # partial alt
    "sh_change_vol", "sh_change_amt", "sh_net_sign",
]

# ── 3. Load actual panel to check column quality ──
print("Loading panel metadata...")
panel = pd.read_parquet("data/panel_full_enriched_v3.parquet", engine="pyarrow",
                        columns=["symbol", "date"] + [c for c in UNUSED_COLS
                                                       if c not in ("eps", "ocfps", "bps", "revenue_ps")])
# Add the new fina cols if they exist
for c in ["eps", "ocfps", "bps", "revenue_ps"]:
    try:
        ext = pd.read_parquet("data/panel_full_enriched_v3.parquet", engine="pyarrow",
                              columns=[c])
        panel[c] = ext[c]
    except Exception:
        panel[c] = np.nan

# ── 4. Build BEFORE vs AFTER per dim ──
lines = []
lines.append("# Auto-Adoption Delta Report")
lines.append(f"\n**Generated**: {pd.Timestamp.now().isoformat()}")
lines.append(f"**Panel**: 2.7M rows × 102 cols, 3,244 symbols\n")

lines.append("## 1. Per-Dim IC BEFORE (current state)\n")
lines.append("| Dim Group | Candidates | Strong | Weak | Dead | Top IC | Top3 Mean IC |")
lines.append("|-----------|-----------|--------|------|------|--------|-------------|")

dim_order = [
    ("dim01", "price_volume"), ("dim02", "volatility"), ("dim03", "fundamentals"),
    ("dim07", "limit_gene"), ("dim04", "sector_effect"), ("dim05", "turnover_liquidity"),
    ("dim06", "valuation_size"), ("dim_active_pit", "active_pit"),
    ("dim08", "calendar"), ("dim09", "custom_formulas"), ("dim10", "money_flow"),
    ("dim11", "float_limits"), ("dim12", "ma_system"), ("dim13", "holiday"),
    ("dim14", "market_sentiment"), ("dim15", "alpha_factors"),
    ("dim16", "candlestick"), ("dim17", "extended_factors"),
    ("dim20", "short_horizon"), ("dim18", "lhb"), ("dim19", "amihud"),
    ("dim21", "chip"), ("dim22", "fina_pit"), ("dim23", "shareholder"),
    ("dim24", "margin"), ("dim26", "lhb_enhanced"), ("dim27", "industry_flow"),
    ("dim28", "sector_index"), ("dim29", "holdertrade"), ("dim30", "kline_geometry"),
    ("dim31", "announcement"),
]

for dim_name, dim_label in dim_order:
    info = per_dim.get(dim_name, {})
    top_ic = info.get("top_ic_max", 0)
    top3 = info.get("top_3_mean_ic", 0)
    lines.append(
        f"| {dim_label} | {info.get('candidates','?'):>3} | "
        f"{info.get('strong','?'):>3} | {info.get('weak','?'):>3} | "
        f"{info.get('dead','?'):>3} | {top_ic:.4f} | {top3:.4f} |"
    )

lines.append(f"\n**Total features evaluated**: {sum(v['candidates'] for v in per_dim.values())}")
lines.append(f"**Total weak (usable)**: {sum(v['weak'] for v in per_dim.values())}")

# ── 5. New columns → trial features ──
lines.append("\n## 2. Unused Panel Columns → Auto-Adopt Candidates\n")
lines.append("| Column | NaN% | Dtype | Would Generate | Dim Group |")
lines.append("|--------|------|-------|---------------|-----------|")

ADOPTION_TEMPLATES = ["zscore_20d", "chg5d", "chg20d", "sector_rank", "ma5_cross", "vol_adj"]
col_quality = []

for col in sorted(UNUSED_COLS):
    if col not in panel.columns:
        continue
    nan_rate = panel[col].isna().mean()
    dtype = str(panel[col].dtype)
    is_numeric = dtype in ("float64", "float32", "int64", "int32")
    is_usable = is_numeric and nan_rate < 0.7
    n_features = len(ADOPTION_TEMPLATES) if is_usable else 0
    quality = "✅ adopt" if is_usable else ("❌ sparse" if nan_rate > 0.7 else "❌ non-numeric")
    col_quality.append((col, nan_rate, dtype, quality, n_features))
    lines.append(
        f"| {col} | {nan_rate:.1%} | {dtype} | "
        f"{n_features} trial features | _auto_adopted |"
    )

total_new = sum(c[-1] for c in col_quality)
adoptable = sum(1 for c in col_quality if c[3].startswith("✅"))

lines.append(f"\n**Adoptable columns**: {adoptable}/{len(col_quality)}")
lines.append(f"**New trial features after adoption**: {total_new}")

# ── 6. AFTER: projected per-dim IC impact ──
lines.append("\n## 3. AFTER — Projected Registry State\n")
lines.append("(_auto_adopted dim adds {total_new} trial features; IC scores TBD by next screening)\n")

total_before = sum(v["candidates"] for v in per_dim.values())
total_active = sum(v["selected"] for v in per_dim.values())
lines.append(f"| | Features | Active | Strong | Weak | Dead |")
lines.append(f"|---|---|---|---|---|---|")
lines.append(f"| **BEFORE** | {total_before} | {total_active} | {sum(v['strong'] for v in per_dim.values())} | {sum(v['weak'] for v in per_dim.values())} | {sum(v['dead'] for v in per_dim.values())} |")
lines.append(f"| **AFTER** | {total_before + total_new} | {total_active + total_new} (trial) | same | same | same |")
lines.append(f"| **DELTA** | +{total_new} | +{total_new} trial | 0 | 0 | 0 |")

lines.append(f"\n> Trial features start as `grade=trial, active=True` — they will be IC-screened in the next training window. Features that pass (>0.02 |IC|) promote to strong/weak. Features that fail 3 consecutive windows are deactivated.\n")

# ── 7. Per-column detail ──
lines.append("## 4. Auto-Adopted Feature Manifest\n")
lines.append("| Source Column | NaN% | Trial Features |")
lines.append("|--------------|------|---------------|")
for col, nan_rate, dtype, quality, nf in sorted(col_quality, key=lambda x: -x[-1]):
    if nf == 0:
        continue
    feat_names = [f"{col}_{t}" for t in ADOPTION_TEMPLATES]
    lines.append(f"| {col} | {nan_rate:.1%} | {', '.join(feat_names[:4])}... |")

# ── Write report ──
report = "\n".join(lines)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)
print(report)
print(f"\nReport saved to: {REPORT_PATH}")
