# -*- coding: utf-8 -*-
"""ovd_wdist / ovd_15p 登记 feature_registry (443 -> 445), WORM 追加."""
import json
from pathlib import Path

REG = Path(r"D:\AMINQT\DATA OTHERS\factor_registry\feature_registry.json")

NOTE = (
    "09-09 筹码缺口实验 (tmp_chip/_full_true150.py 真150档分布, 500stk x 500d seed42): "
    "IC +0.0845 t=23.2 残差(winner_ratio+cost_bias+mom5) +0.0074 t=19.2 双半窗同号; "
    "proxy单桶对照方向守住 (IC缩32%系采样+单桶近似). 注入 cyq_ext._compute_cyq_one_day"
)

entries = {
    "ovd_wdist": {
        "dim_group": "chip_morphology",
        "active": True,
        "grade": "trial",
        "transform": "raw",
        "created": "2026-09-09",
        "last_eval": "",
        "source_cols": ["ovd_wdist"],
        "note": "上方套牢筹码距离加权均值 " + NOTE,
    },
    "ovd_15p": {
        "dim_group": "chip_morphology",
        "active": True,
        "grade": "trial",
        "transform": "raw",
        "created": "2026-09-09",
        "last_eval": "",
        "source_cols": ["ovd_15p"],
        "note": "距现价>=15%以远筹码占比, IC +0.0818 t=22.4 残差 +0.0070 t=18.9; " + NOTE,
    },
}

reg = json.loads(REG.read_text(encoding="utf-8"))
feats = reg["features"]
for name, meta in entries.items():
    if name in feats:
        print(f"SKIP {name}: already registered")
        continue
    feats[name] = meta
backup = REG.with_suffix(".json.bak_0909_ovd")
if not backup.exists():
    backup.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
REG.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"OK total={len(feats)} (backup: {backup.name})")
