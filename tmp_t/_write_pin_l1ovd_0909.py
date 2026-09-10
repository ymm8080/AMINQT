# -*- coding: utf-8 -*-
"""WRITE-PIN: 09-09 三臂 A/B 终判落盘 (tmp_t/_pin_ab_l1ovd_0909_result.json).

主判据 B2 vs A 3/4 PASS; 按板读组 (pins 本为分板文件):
  main 2/2 B2 PASS (ovd 全正) → main pin += dim20三 + ovd两 (5)
  dual 2/2 B1 PASS (ovd 双负) → dual pin += dim20三 (3)
WORM: .bak_0909_l1ovd 备份后写; selected_count 同步.
"""
import json
import shutil
from pathlib import Path

REG = Path(r"D:\AMINQT\DATA OTHERS\factor_registry")
DIM20 = ["days_since_board_break", "vol_decay_ratio", "quiet_drift"]
OVD = ["ovd_wdist", "ovd_15p"]

for fname, add in [("selected_main_pinned.json", DIM20 + OVD),
                   ("selected_dual_pinned.json", DIM20)]:
    p = REG / fname
    pin = json.loads(p.read_text(encoding="utf-8"))
    feats = pin["features"]
    new = [c for c in add if c not in feats]
    if not new:
        print(f"{fname}: nothing to add")
        continue
    bak = REG / (fname + ".bak_0909_l1ovd")
    if not bak.exists():
        shutil.copy2(p, bak)
    pin["features"] = feats + new
    pin["selected_count"] = len(pin["features"])
    pin.setdefault("pin_updates", []).append({
        "date": "2026-09-09",
        "added": new,
        "evidence": "tmp_t/_pin_ab_l1ovd_0909_result.json 三臂A/B "
                    "OOS250d LGBM 预注册判据: B2-A 3/4; main B2 2/2, dual B1 2/2",
    })
    p.write_text(json.dumps(pin, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{fname}: +{new} -> {len(pin['features'])} (backup {bak.name})")
