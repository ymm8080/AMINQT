import pickle
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for p in [
    r"D:\AMINQT\AMINQT CODES\models\pipeline1\main_v20260730_2235_635feats_7mod.pkl",
    r"D:\AMINQT\AMINQT CODES\models\pipeline1\main_2026W31_3y.pkl",
    r"D:\AMINQT\AMINQT CODES\models\pipeline1\dual_2026W31_3y.pkl",
]:
    print("=" * 70)
    print("BUNDLE:", p.split("\\")[-1])
    try:
        d = pickle.load(open(p, "rb"))
    except Exception as e:
        print("  LOAD ERR:", e)
        continue
    print("  keys:", list(d.keys()))
    m = d.get("models", {})
    for k, v in m.items():
        try:
            md = v[0]
            print(
                f"  {k}: n_feat_in={getattr(md, 'n_features_in_', None)} "
                f"best_iter={getattr(md, 'best_iteration_', None)} "
                f"n_estimators={getattr(md, 'n_estimators', None)}"
            )
        except Exception as e:
            print("  ", k, "ERR", e)
    ics = d.get("oos", {}).get("ics", {})
    print(
        "  OOS ics:", {k: round(float(v), 4) for k, v in ics.items()} if ics else None
    )
    fcols = d.get("feature_cols") or d.get("features") or []
    print("  feature_cols count:", len(fcols) if fcols else None)
