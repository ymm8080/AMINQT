# -*- coding: utf-8 -*-
"""Fix winner_ratio = winner_rate / 100 (restore [0,1] range)."""

import pathlib

# 1. _daily_fetch.py
p1 = pathlib.Path("_daily_fetch.py")
t1 = p1.read_text(encoding="utf-8")
t1 = t1.replace(
    '# --- winner_ratio = winner_rate (0-100, percentage) ---\nif "winner_rate" in df.columns and "winner_ratio" in panel_cols:\n    df["winner_ratio"] = df["winner_rate"]',
    '# --- winner_ratio = winner_rate / 100 ([0,1] ratio) ---\nif "winner_rate" in df.columns and "winner_ratio" in panel_cols:\n    df["winner_ratio"] = df["winner_rate"] / 100.0',
)
p1.write_text(t1, encoding="utf-8")
print("_daily_fetch.py: done")

# 2. feature_engine_v35.py
p2 = pathlib.Path("app/pipeline1/feature_engine_v35.py")
t2 = p2.read_text(encoding="utf-8")
# Fix the computation line
t2 = t2.replace(
    'df["winner_ratio"] = df["winner_rate"]',
    'df["winner_ratio"] = df["winner_rate"] / 100.0',
)
# Fix docstring
t2 = t2.replace(
    "winner_ratio = winner_rate (0-100, 百分比)",
    "winner_ratio = winner_rate / 100  # [0,1]",
)
t2 = t2.replace(
    "winner_rate (获利盘比例 %, 即 winner_ratio)",
    "winner_rate (获利盘比例 %, winner_ratio = winner_rate/100)",
)
p2.write_text(t2, encoding="utf-8")
print("feature_engine_v35.py: done")

# Verify
import py_compile

py_compile.compile("_daily_fetch.py", doraise=True)
py_compile.compile("app/pipeline1/feature_engine_v35.py", doraise=True)
print("Syntax OK")
