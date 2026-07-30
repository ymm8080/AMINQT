# -*- coding: utf-8 -*-
"""Auto-Adoption ICIR Impact — slim: 100 stocks, 200 sampled features, delta only."""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYARROW_LEGACY_MEMORY_POOL"] = "1"

import numpy as np
import pandas as pd

from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.cleaning_pipeline import CleaningPipeline, board_of
from app.utils.daily_rank_ic import icir as calc_icir

# ── 1. Load: July 2026, 100 stocks ──
print("Loading...")
meta = pd.read_parquet("data/panel_full_enriched_v3.parquet",
                       columns=["symbol","date"], engine="pyarrow")
mask = (meta["date"]>="2026-07-01") & (meta["date"]<="2026-07-28")
np.random.seed(42)
syms = np.random.choice(meta.loc[mask,"symbol"].unique(), size=min(100,len(meta.loc[mask,"symbol"].unique())), replace=False)
idx = meta[meta["symbol"].isin(syms) & mask].index; del meta
panel = pd.read_parquet("data/panel_full_enriched_v3.parquet", engine="pyarrow")
df = panel.iloc[idx].copy(); del panel
print(f"Loaded: {len(df)} rows, {df['symbol'].nunique()} symbols")

# ── 2. Clean + labels ──
df["board"] = df["symbol"].map(lambda s: board_of(s))
cleaner = CleaningPipeline()
m, d = cleaner.run_train(df)
df = pd.concat([m,d]).sort_values(["symbol","date"]).reset_index(drop=True)
# Fix: run step4 manually to get is_limit_up
df, _ = cleaner.step4_tradability(df)
if "is_limit_up_close" in df.columns and "is_limit_up" not in df.columns:
    df["is_limit_up"] = df["is_limit_up_close"]
df = LabelEngine.build_path_labels(df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=6)
label_1d = "label_1d_net" if "label_1d_net" in df.columns else "label_1d"
print(f"Cleaned: {len(df)} rows, label={label_1d}")

# ── 3. BEFORE ──
print("\n=== BEFORE ===")
fe = FeatureEngineV35()
t0 = time.time()
df_before = fe.build(df.copy())
bt = time.time()-t0
before_cols = FeatureEngineV35.feature_columns(df_before)

# Per-dim quota sampling: seed registry to get dim groups, then pick top-N per dim
reg = FeatureRegistry(path="data/factor_registry/_ic_test_registry.json")
reg._seed(df_before)
print(f"Seeded: {len(reg.features)} features in {len(reg.get_dim_groups())} dims")

sample_feats = []
QUOTA_PER_DIM = 6
for dim_name in sorted(reg.get_dim_groups()):
    feats = [f for f in reg.get_active(dim_name) if f in df_before.columns]
    if not feats:
        continue
    # Pick top by std within this dim
    stds = df_before[feats].std()
    valid = stds[stds > 0].dropna().sort_values(ascending=False)
    sample_feats.extend(list(valid.index[:QUOTA_PER_DIM]))
sample_feats = sample_feats[:200]  # cap
print(f"Time: {bt:.1f}s, Features: {len(before_cols)}, Sampled: {len(sample_feats)} per-dim")

b_icirs = []
for f in sample_feats:
    v = calc_icir(df_before, f, label_1d)
    if not np.isnan(v): b_icirs.append(abs(v))

# ── 5. AFTER (reuse seeded registry, enable adoption) ──
print("\n=== AFTER ===")
reg.enable_adoption()
reg._data["adoption"]["registered_source_cols"] = []
reg.save()
print("\n=== AFTER ===")
t0 = time.time()
df_after = fe.build(df.copy(), registry=reg)
at = time.time()-t0
after_cols = FeatureEngineV35.feature_columns(df_after)
new_feats = sorted(set(after_cols)-set(before_cols))
print(f"Time: {at:.1f}s, Features: {len(after_cols)} (+{len(new_feats)} trial)")

# Same sample + all new
a_icirs = []
for f in sample_feats:
    if f in df_after.columns:
        v = calc_icir(df_after, f, label_1d)
        if not np.isnan(v): a_icirs.append(abs(v))

trial_icirs = []
for f in new_feats:
    if f in df_after.columns:
        v = calc_icir(df_after, f, label_1d)
        if not np.isnan(v): trial_icirs.append((f, v))

# ── 6. Report ──
mb, ma = np.mean(b_icirs), np.mean(a_icirs)
trial_vals = [abs(v) for _,v in trial_icirs]

print("\n" + "="*60)
print("DELTA REPORT")
print("="*60)
print(f"Stocks: {df['symbol'].nunique()}, Days: {df['date'].nunique()}")
print(f"Features: {len(before_cols)} → {len(after_cols)} (+{len(new_feats)})")
print(f"Build time: {bt:.1f}s → {at:.1f}s")
print(f"\nSampled ICIR (n={len(b_icirs)}):")
print(f"  BEFORE: mean={mb:.4f} median={np.median(b_icirs):.4f}")
print(f"  AFTER:  mean={ma:.4f} median={np.median(a_icirs):.4f}")
print(f"  DELTA:  {ma-mb:+.4f}")

if trial_vals:
    print(f"\nTrial features (n={len(trial_vals)}):")
    print(f"  mean={np.mean(trial_vals):.4f} median={np.median(trial_vals):.4f} max={np.max(trial_vals):.4f}")
    s = sum(1 for v in trial_vals if v>=0.05)
    w = sum(1 for v in trial_vals if 0.02<=v<0.05)
    n = sum(1 for v in trial_vals if v<0.02)
    print(f"  strong(>=0.05):{s}  weak(0.02-0.05):{w}  noise(<0.02):{n}")

    best = sorted(trial_icirs, key=lambda x:-abs(x[1]))[:10]
    print(f"\nTop 10 by |ICIR|:")
    for f,v in best: print(f"  {f:<50s} ICIR={v:+.4f}")

print(f"\nVerdict: {'ADOPT' if ma>mb else 'REJECT'} (ICIR delta={ma-mb:+.4f})")
