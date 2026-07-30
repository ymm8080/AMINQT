# -*- coding: utf-8 -*-
"""Aggregate all per-dimension IC evaluations into a unified decision report.
Reads existing ic_dim_main_g*.json files and produces:
1. Consolidated ranking (top 100 by |IC_5d|)
2. Dimension-level summary (strongest feature per dimension)
3. Decision table: which dimensions should be INCLUDE vs SKIP vs DELETE
"""

import json
import glob
import os
from collections import defaultdict
from datetime import datetime

REGISTRY_DIR = "data/factor_registry"

def load_all_groups():
    """Load all ic_dim_main_g*.json group evaluations."""
    files = sorted(glob.glob(os.path.join(REGISTRY_DIR, "ic_dim_main_g*.json")))
    all_factors = {}  # factor_name -> best metrics
    dim_results = []  # raw group data
    for f in files:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        dim_results.append(d)
        for t in d.get("top_20", []):
            fac = t["factor"]
            # Keep the entry with highest |IC_5d|
            if fac not in all_factors or abs(t["ic_5d"]) > abs(all_factors[fac]["ic_5d"]):
                all_factors[fac] = {**t, "source_dims": d["dims"]}
        # Also add non-top-20 if available in the full results
    return dim_results, all_factors

def main():
    dim_results, all_factors = load_all_groups()

    if not all_factors:
        print("No factor data found. Run ic_eval_dim.py first.")
        return

    # Sort by |IC_5d| descending
    ranked = sorted(all_factors.values(), key=lambda x: abs(x["ic_5d"]), reverse=True)

    # ── 1. TOP FEATURES ──
    print("=" * 100)
    print("  CONSOLIDATED FEATURE EVALUATION — Main Board, 2024-now (Full 2003 stocks)")
    print("=" * 100)
    print(f"  Total unique factors: {len(all_factors)}")
    print()

    print(f"{'Rank':<5s} {'Factor':<42s} {'IC_1d':>8s} {'IC_3d':>8s} {'IC_5d':>8s} {'Grade':>8s} {'Source Dims'}")
    print("-" * 100)
    for i, t in enumerate(ranked[:60], 1):
        dim_str = ",".join(t.get("source_dims", [])[:3])
        print(f"{i:<5d} {t['factor']:<42s} {t['ic_1d']:>+8.4f} {t['ic_3d']:>+8.4f} {t['ic_5d']:>+8.4f} {t['grade']:>8s} {dim_str}")

    # ── 2. GRADE DISTRIBUTION ──
    grades = defaultdict(int)
    for t in all_factors.values():
        grades[t["grade"]] += 1
    print(f"\n  Grade distribution: strong={grades.get('strong', 0)}, weak={grades.get('weak', 0)}, dead={grades.get('dead', 0)}")

    # ── 3. DIMENSION SUMMARY ──
    print(f"\n{'=' * 100}")
    print("  DIMENSION-LEVEL DECISION TABLE")
    print(f"{'=' * 100}")
    print(f"{'Dim':<8s} {'Status':>8s} {'Best Feature':<38s} {'IC_5d':>8s} {'Grade':>8s} {'#Factors'}")
    print("-" * 100)

    for d in dim_results:
        dims_str = ",".join(d["dims"])
        top = d.get("top_20", [{}])[0] if d.get("top_20") else {}
        best_fac = top.get("factor", "N/A")
        best_ic = top.get("ic_5d", 0)
        best_grade = top.get("grade", "dead")
        status = "INCLUDE" if d["n_strong"] + d["n_weak"] > 0 and abs(best_ic) >= 0.15 else "WEAK" if d["n_weak"] > 0 else "DEAD"
        print(f"{dims_str:<8s} {status:>8s} {best_fac:<38s} {best_ic:>+8.4f} {best_grade:>8s} {d['n_candidates']}")

    # ── 4. DECISION MATRIX ──
    print(f"\n{'=' * 100}")
    print("  FEATURE INCLUSION DECISION MATRIX")
    print(f"{'=' * 100}")
    print()
    print("  THRESHOLDS:")
    print("    |IC_5d| >= 0.15  → INCLUDE (strong predictive signal)")
    print("    0.10 < |IC_5d| < 0.15 → WEAK (marginal, keep if complementary)")
    print("    |IC_5d| < 0.10  → DEAD (no predictive value, remove)")
    print()

    strong = [t for t in ranked if abs(t["ic_5d"]) >= 0.15]
    weak = [t for t in ranked if 0.10 <= abs(t["ic_5d"]) < 0.15]
    dead = [t for t in ranked if abs(t["ic_5d"]) < 0.10]

    print(f"  STRONG (|IC|>=0.15): {len(strong)} features")
    for t in strong:
        print(f"    [+] {t['factor']:<40s} IC_5d={t['ic_5d']:+.4f}")

    print(f"\n  WEAK (0.10<=|IC|<0.15): {len(weak)} features")
    for t in weak[:20]:
        print(f"    [~] {t['factor']:<40s} IC_5d={t['ic_5d']:+.4f}")
    if len(weak) > 20:
        print(f"    ... and {len(weak)-20} more")

    print(f"\n  DEAD (|IC|<0.10): {len(dead)} features -- REMOVE from pipeline")

    # ── 5. SAVE ──
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REGISTRY_DIR, f"feature_decision_{tag}.json")
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_factors": len(all_factors),
        "strong_count": len(strong),
        "weak_count": len(weak),
        "dead_count": len(dead),
        "strong": [{"factor": t["factor"], "ic_5d": t["ic_5d"], "grade": t["grade"]} for t in strong],
        "weak": [{"factor": t["factor"], "ic_5d": t["ic_5d"], "grade": t["grade"]} for t in weak],
        "dead": [{"factor": t["factor"], "ic_5d": t["ic_5d"]} for t in dead],
        "dimension_summary": [
            {
                "dims": d["dims"],
                "best_feature": (d.get("top_20", [{}])[0] or {}).get("factor"),
                "best_ic_5d": (d.get("top_20", [{}])[0] or {}).get("ic_5d", 0),
                "n_strong": d["n_strong"],
                "n_weak": d["n_weak"],
                "n_dead": d["n_dead"],
            }
            for d in dim_results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Decision report saved: {out_path}")

if __name__ == "__main__":
    main()
