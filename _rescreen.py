# -*- coding: utf-8 -*-
"""Quick re-screen: feature engine + ICScreener on panel_18m, with current thresholds.
   Compares new grades vs stored factors_dual_2026W31.json."""
import json, sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rescreen")

import pandas as pd

PANEL = "data/panel_18m.parquet"
REF_JSON = "data/factor_registry/factors_dual_2026W31.json"

t0 = time.time()
panel = pd.read_parquet(PANEL)
logger.info("Panel: %d rows, %d cols, %d stocks, %s ~ %s",
            len(panel), len(panel.columns), panel["symbol"].nunique(),
            str(panel["date"].min())[:10], str(panel["date"].max())[:10])

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.ic_screener import ICScreener

cleaner = CleaningPipeline()
main_df, dual_df = cleaner.run_train(panel)
logger.info("After clean: main=%d rows, dual=%d rows", len(main_df), len(dual_df))

features = FeatureEngineV35()
df = features.build(dual_df, cross_sectional_rank=True)
df = LabelEngine.build_path_labels(df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=6)
logger.info("Features: %d rows, %d cols", len(df), len(df.columns))

candidates = FeatureEngineV35.feature_columns(df)
candidates = [c for c in candidates if c in df.columns and df[c].dtype != object and df[c].isna().mean() < 0.95]
logger.info("Candidates: %d", len(candidates))

screener = ICScreener(registry_path="data/factor_registry")
result = screener.screen(df, candidates, window_id="dual_rescreen")

strong = [k for k, v in result["detail"].items() if v["grade"] == "strong"]
weak   = [k for k, v in result["detail"].items() if v["grade"] == "weak"]
dead   = [k for k, v in result["detail"].items() if v["grade"] == "dead"]
logger.info("New grades: %d strong, %d weak, %d dead, %d total", len(strong), len(weak), len(dead), len(result["detail"]))

# Compare with stored
if os.path.exists(REF_JSON):
    ref = json.load(open(REF_JSON, encoding="utf-8"))
    ref_detail = ref.get("detail", {})
    changed = {}  # factor -> (old_grade, new_grade)
    for f, v in result["detail"].items():
        old_g = ref_detail.get(f, {}).get("grade", "N/A")
        new_g = v["grade"]
        if old_g != new_g and old_g != "N/A":
            changed[f] = (old_g, new_g, v.get("ic_1d", 0), v.get("icir", 0))

    print()
    print("=" * 100)
    print(f"GRADE CHANGES under current thresholds (IC_STRONG=0.02, ICIR_MIN=0.05)")
    print(f"Reference: {REF_JSON}")
    print("=" * 100)
    if changed:
        upgraded   = [(f, o, n, ic, ir) for f, (o, n, ic, ir) in changed.items() if o in ("dead", "N/A") and n in ("strong", "weak")]
        downgraded = [(f, o, n, ic, ir) for f, (o, n, ic, ir) in changed.items() if o in ("strong", "weak") and n in ("dead", "weak")]
        same = [(f, o, n, ic, ir) for f, (o, n, ic, ir) in changed.items() if o == "weak" and n == "weak"]

        print(f"\n>>> UPGRADED (dead→alive): {len(upgraded)}")
        for f, o, n, ic, icir in sorted(upgraded, key=lambda x: -abs(x[3])):
            print(f"  {f:<40s} {o:>6s} → {n:<6s}  |IC|={abs(ic):.4f}  ICIR={icir:.4f}")

        print(f"\n>>> DOWNGRADED: {len(downgraded)}")
        for f, o, n, ic, icir in sorted(downgraded, key=lambda x: abs(x[3])):
            print(f"  {f:<40s} {o:>6s} → {n:<6s}  |IC|={abs(ic):.4f}  ICIR={icir:.4f}")

        # CHG features specifically
        chg_features = [f for f in result["detail"] if "_chg" in f.lower()]
        chg_strong = [f for f in chg_features if result["detail"][f]["grade"] == "strong"]
        chg_weak   = [f for f in chg_features if result["detail"][f]["grade"] == "weak"]
        chg_dead   = [f for f in chg_features if result["detail"][f]["grade"] == "dead"]
        print(f"\n>>> CHG FEATURES: {len(chg_features)} total, {len(chg_strong)} strong, {len(chg_weak)} weak, {len(chg_dead)} dead")
        if chg_dead:
            print("  DEAD CHG features:")
            for f in sorted(chg_dead):
                v = result["detail"][f]
                print(f"    {f:<45s} |IC|={abs(v['ic_1d']):.4f}  ICIR={v['icir']:.4f}  roll_mean={v['rolling_mean']:.4f}")
    else:
        print("No grade changes found — thresholds don't shift any factor's grade.")

print()
print(f"Elapsed: {time.time() - t0:.1f}s")
