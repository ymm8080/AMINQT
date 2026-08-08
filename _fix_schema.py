# -*- coding: utf-8 -*-
"""Fix schema consistency: add day_change + pred_ret_2d, update schema_version to 1.3."""
import os

# Fix 1: tests/test_pipeline1_list_to_panel.py
path = os.path.join("tests", "test_pipeline1_list_to_panel.py")
with open(path, "r", encoding="utf-8-sig") as f:
    content = f.read()

# Add day_change column before pred_ret_1d in _make_schema_list
old = '"board": ["main"] * n,\n            "pred_ret_1d": rng.uniform(-0.02, 0.05, n),\n            "pred_ret_2d"'
new = '"board": ["main"] * n,\n            "day_change": rng.uniform(-0.03, 0.06, n),\n            "pred_ret_1d": rng.uniform(-0.02, 0.05, n),\n            "pred_ret_2d"'
if old in content:
    content = content.replace(old, new)
    print("[OK] Added day_change to _make_schema_list")
else:
    print("[SKIP] day_change already present or pattern not found")

# Update schema_version from 1.2 to 1.3
content = content.replace('"schema_version": "1.2"', '"schema_version": "1.3"')
# Update V1.2 references to V1.3
content = content.replace("V1.2", "V1.3")
# Update '1.2' assertions to '1.3'
content = content.replace("'1.2'", "'1.3'")
content = content.replace('== "1.2"', '== "1.3"')

with open(path, "w", encoding="utf-8-sig") as f:
    f.write(content)
print("[OK] Test file fixed")

# Fix 2: app/streamlit/data_service.py
path2 = os.path.join("app", "streamlit", "data_service.py")
with open(path2, "r", encoding="utf-8") as f:
    content2 = f.read()

# Add pred_ret_2d after pred_ret_1d in demo_list
old2 = '"pred_ret_1d": rng.uniform(-0.02, 0.05, n),\n            "pred_ret_3d"'
new2 = '"pred_ret_1d": rng.uniform(-0.02, 0.05, n),\n            "pred_ret_2d": rng.uniform(-0.03, 0.08, n),\n            "pred_ret_3d"'
if old2 in content2:
    content2 = content2.replace(old2, new2)
    print("[OK] Added pred_ret_2d to demo_list")
else:
    print("[SKIP] pred_ret_2d already present or pattern not found")

# Update schema_version from 1.2 to 1.3
content2 = content2.replace('"schema_version": "1.2"', '"schema_version": "1.3"')

with open(path2, "w", encoding="utf-8") as f:
    f.write(content2)
print("[OK] data_service.py fixed")
