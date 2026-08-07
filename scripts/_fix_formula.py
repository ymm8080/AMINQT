"""One-time fix: pct_90_con fallback formula in feature_engine_v35.py."""

f = "app/pipeline1/feature_engine_v35.py"
s = open(f, encoding="utf-8").read()

# Fix 1: docstring
s = s.replace(
    "pct_90_con   = (cost_95pct - cost_5pct) / cost_50pct",
    "pct_90_con   = (cost_95pct - cost_5pct) / (cost_95pct + cost_5pct)",
)

# Fix 2: fallback formula + add pct_70_con
old = '            if "pct_90_con" not in df.columns:\n                df["pct_90_con"] = (df["cost_95pct"] - df["cost_5pct"]) / df[\n                    "cost_50pct"\n                ].replace(0, np.nan)'
new = '            if "pct_90_con" not in df.columns:\n                df["pct_90_con"] = (df["cost_95pct"] - df["cost_5pct"]) / (\n                    df["cost_95pct"] + df["cost_5pct"]\n                ).replace(0, np.nan)\n            if "pct_70_con" not in df.columns:\n                df["pct_70_con"] = (df["cost_85pct"] - df["cost_15pct"]) / (\n                    df["cost_85pct"] + df["cost_15pct"]\n                ).replace(0, np.nan)'
s = s.replace(old, new)

open(f, "w", encoding="utf-8").write(s)
print("Done: formula fixed")
