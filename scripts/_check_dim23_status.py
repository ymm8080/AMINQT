# -*- coding: utf-8 -*-
"""Count active registry features per dim_group, to reconcile the '17 columns' claim."""

import json
from collections import Counter

REG = r"D:\AMINQT\DATA OTHERS\factor_registry\feature_registry.json"
feats = json.load(open(REG, encoding="utf-8"))["features"]

by_dim = Counter()
for name, v in feats.items():
    if v.get("active", True):
        by_dim[v.get("dim_group") or "UNGROUPED"] += 1

print("=== active features per dim_group ===")
for g, c in by_dim.most_common():
    print(f"  {c:4d}  {g}")

total = sum(by_dim.values())
print(f"\nTOTAL active registry features: {total} / {len(feats)} registered")
