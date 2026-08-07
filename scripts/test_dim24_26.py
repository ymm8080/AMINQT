"""
Test DIM24 (margin), DIM25 (northbound), DIM26 (LHB) end-to-end.
Simplified: use existing cache, sample to 1500 stocks.
"""

import warnings

warnings.filterwarnings("ignore")
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE = os.path.join(ROOT, "data", "supply_cache", "alt_data")

# ── 1. Load panel, sample 1500 stocks ──
panel = pd.read_parquet(os.path.join(ROOT, "data", "panel_full_enriched.parquet"))
np.random.seed(42)
stocks = list(panel["symbol"].unique())
sampled = set(np.random.choice(stocks, min(1500, len(stocks)), replace=False))
panel = panel[panel["symbol"].isin(sampled)].copy()
print(f"Panel after sampling: {panel['symbol'].nunique()} stocks, {len(panel)} rows")

# Basic columns
from app.pipeline1.cleaning_pipeline import board_of, get_limit_pct  # noqa: E402

panel["board"] = panel["symbol"].map(board_of)
panel["limit_pct"] = [
    get_limit_pct(b, d) for b, d in zip(panel["board"], panel["date"], strict=False)
]
panel["is_suspended"] = False
panel["is_st"] = False
panel["industry"] = panel.get("industry", "综合")

# ── Merge margin ──
mg_path = os.path.join(CACHE, "margin", "20240102_20260727.parquet")
if os.path.exists(mg_path):
    mg = pd.read_parquet(mg_path)
    mg_cols = [
        c for c in mg.columns if c not in ["symbol", "date"] and not c.startswith("_")
    ]
    panel = panel.merge(
        mg[["symbol", "date"] + mg_cols], on=["symbol", "date"], how="left"
    )
    n_dates = panel.dropna(subset=["margin_balance"])["date"].nunique()
    print(
        f"Margin merged: non-NaN={panel['margin_balance'].notna().sum()}/{len(panel)}, dates={n_dates}"
    )

# ── Merge northbound ──
nb_path = os.path.join(CACHE, "northbound", "20240102_20260727.parquet")
if os.path.exists(nb_path):
    nb = pd.read_parquet(nb_path)
    nb_cols = [
        c for c in nb.columns if c not in ["symbol", "date"] and not c.startswith("_")
    ]
    nb_daily = nb[["date"] + nb_cols].drop_duplicates(subset=["date"])
    panel = panel.merge(nb_daily, on="date", how="left")
    nnb = panel["north_net_buy_sh"].notna().sum()
    nd = panel.loc[panel["north_net_buy_sh"].notna(), "date"].nunique()
    print(f"Northbound merged: non-NaN={nnb}/{len(panel)}, dates={nd}")

# ── Merge LHB ──
lhb_path = os.path.join(CACHE, "lhb", "all_20240102_20260727.parquet")
if os.path.exists(lhb_path):
    lhb = pd.read_parquet(lhb_path)
    lhb_cols = [
        c for c in lhb.columns if c not in ["symbol", "date"] and not c.startswith("_")
    ]
    panel = panel.merge(
        lhb[["symbol", "date"] + lhb_cols], on=["symbol", "date"], how="left"
    )
    print(f"LHB merged: non-NaN={panel['lhb_net_buy'].notna().sum()}/{len(panel)}")

# ── DATA QUALITY SUMMARY ──
print(f"\n{'=' * 70}")
print("DATA QUALITY SUMMARY (pre-feature)")
print(f"{'=' * 70}")
for col in [
    "margin_balance",
    "short_balance",
    "margin_buy_amt",
    "north_net_buy_sh",
    "north_net_buy_sz",
    "lhb_net_buy",
    "lhb_institutional_net_buy",
    "lhb_institutional_count",
]:
    if col in panel.columns:
        n_na = panel[col].isna().sum()
        n_tot = len(panel)
        if n_na < n_tot:
            vmin, vmax = panel[col].min(), panel[col].max()
        else:
            vmin = vmax = 0
        print(
            f"  {col:35s}: NaN {n_na:>8d}/{n_tot} ({n_na / n_tot * 100:5.1f}%),  range=[{vmin:>15.4f}, {vmax:>15.4f}]"
        )

# Date range coverage
for label, col in [
    ("Margin", "margin_balance"),
    ("Northbound", "north_net_buy_sh"),
    ("LHB", "lhb_net_buy"),
]:
    if col in panel.columns:
        dts = panel.dropna(subset=[col])["date"].unique()
        if len(dts):
            print(
                f"  {label} coverage: {len(dts)} dates, {pd.Timestamp(min(dts)).strftime('%Y-%m-%d')} ~ {pd.Timestamp(max(dts)).strftime('%Y-%m-%d')}"
            )

# ── 6. BUILD ALL FEATURES ──
panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
panel["list_days"] = panel.groupby("symbol").cumcount() + 1

from app.pipeline1.feature_engine_v35 import FeatureEngineV35  # noqa: E402
from app.pipeline1.label_engine import LabelEngine  # noqa: E402

t0 = time.time()
fe = FeatureEngineV35()
panel = fe.build(panel)
panel = LabelEngine.build_path_labels(panel)
panel = LabelEngine.build_labels(panel)
panel = LabelEngine.mask_suspension(panel)
panel = LabelEngine.mask_recent_days(panel, days=6)
print(f"Features + labels built: {len(panel.columns)} cols in {time.time() - t0:.1f}s")


# ── 7. IC EVALUATION ──
def rank_ic(df, factor, label, min_stocks=30):
    """Daily rank IC, return mean abs IC and hit rate."""
    sub = df[[factor, "date", label]].dropna()
    if len(sub) < 500:
        return 0.0, 0.0, 0
    ics = []
    for _, g in sub.groupby("date"):
        g2 = g.dropna()
        if len(g2) >= min_stocks:
            ic = g2[[factor, label]].corr(method="spearman").iloc[0, 1]
            if not np.isnan(ic):
                ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        return 0.0, 0.0, 0
    return float(np.mean(ics)), float(np.mean(ics > 0)), len(ics)


dims = {
    "DIM24_MARGIN": [
        "margin_balance_chg_1d",
        "margin_balance_chg_5d",
        "short_balance_ratio",
        "margin_buy_ratio",
        "margin_balance_ma20_dev",
        "margin_balance_yoy",
        "margin_pressure_score",
    ],
    "DIM25_NORTHBOUND": [
        "north_net_buy_5d",
        "north_net_buy_20d",
        "north_net_buy_streak",
        "north_buy_ratio",
        "north_sh_sz_divergence",
        "north_momentum_5d",
        "north_flow_zscore",
    ],
    "DIM26_LHB": [
        "lhb_inst_net_buy_5d",
        "lhb_inst_net_buy_20d",
        "lhb_inst_count_5d",
        "lhb_inst_buy_ratio",
        "lhb_abnormal_score",
    ],
}

# Find available labels
all_labels = [c for c in panel.columns if c.startswith("label_")]
print(f"Available labels: {all_labels}")
label_1d = [c for c in all_labels if c.startswith("label_1d")]
label_1d = label_1d[0] if label_1d else "label_1d"
label_5d = [c for c in all_labels if c.startswith("label_5d")]
label_5d = label_5d[0] if label_5d else "label_5d"
label_20d = [c for c in all_labels if c.startswith("label_20d")]
label_20d = label_20d[0] if label_20d else None

print("\n" + "=" * 110)
print(f"IC EVALUATION  |  Label: {label_1d}")
print("=" * 110)
for dim, feats in dims.items():
    best_ic, best_f, best_hit = 0, "", 0
    has_any = False
    print(f"\n── {dim} ──")
    for f in feats:
        if f not in panel.columns:
            print(f"  {f:30s}: ✗ MISSING")
            continue
        has_any = True
        nan_r = panel[f].isna().mean()
        if nan_r > 0.98:
            print(f"  {f:30s}: NaN={nan_r:.1%} (skip)")
            continue
        ic_1d, hit_1d, n = rank_ic(panel, f, label_1d)
        ic_5d, hit_5d, _ = rank_ic(panel, f, label_5d)
        ic_20d, hit_20d, _ = (
            rank_ic(panel, f, label_20d) if label_20d else (0.0, 0.0, 0)
        )

        sig = (
            "***STRONG"
            if ic_1d >= 0.03
            else ("**OK" if ic_1d >= 0.02 else ("*weak" if ic_1d >= 0.01 else " -"))
        )
        print(
            f"  {f:30s}: |IC|_1d={ic_1d:.5f} hit={hit_1d:.1%}  "
            f"|IC|_5d={ic_5d:.5f}  |IC|_20d={ic_20d:.5f}  "
            f"NaN={nan_r:.1%}  [{sig}]  (n_dates={n})"
        )
        if ic_1d > best_ic:
            best_ic, best_f, best_hit = ic_1d, f, hit_1d
    if has_any:
        print(f"  → BEST in {dim}: {best_f}  |IC|_1d={best_ic:.5f}  hit={best_hit:.1%}")

# ── 8. Correlation summary ──
print("\n" + "=" * 110)
print("FEATURE CORRELATION MATRIX (Spearman, among DIM24/25/26)")
print("=" * 110)
all_feats = []
for feats in dims.values():
    all_feats.extend(feats)
existing = [f for f in all_feats if f in panel.columns]
if len(existing) >= 2:
    corr = panel[existing].dropna().corr(method="spearman")
    # Print compact heatmap-style
    for i, f1 in enumerate(existing):
        row = []
        for j, f2 in enumerate(existing):
            if j <= i:
                row.append("      ")
            else:
                v = corr.loc[f1, f2]
                if abs(v) >= 0.5:
                    row.append(f"{v:+.3f}H")
                elif abs(v) >= 0.3:
                    row.append(f"{v:+.3f}M")
                else:
                    row.append(f"{v:+.3f}L")
        print(f"  {f1:30s}: {' '.join(row)}")

print("\nDone.")
