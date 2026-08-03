# -*- coding: utf-8 -*-
"""Resolve merge conflict in data_service.py."""
import pathlib

p = pathlib.Path("app/streamlit/data_service.py")
t = p.read_text(encoding="utf-8")

old = """<<<<<<< HEAD
            "pred_ret_2d": rng.uniform(-0.03, 0.08, n),
=======
            "pred_ret_2d": rng.uniform(-0.02, 0.07, n),
>>>>>>> origin/main
"""
new = '            "pred_ret_2d": rng.uniform(-0.03, 0.08, n),\n'

if old in t:
    t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")
    print("Conflict resolved.")
else:
    print("Conflict marker not found!")
