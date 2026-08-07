"""一次性注册: 把 5 个筹码形态目标列注册进特征注册中心.

用法:
  python scripts/_register_cyq_features.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv()

from app.pipeline1 import cyq_ext
from app.pipeline1.feature_registry import FeatureRegistry

META = {
    "dim_group": "chip_morphology",
    "active": True,
    "grade": "trial",
    "transform": "raw",
    "created": "2026-08-02",
    "last_eval": "",
}


def main() -> None:
    reg = FeatureRegistry()
    for name in cyq_ext.TARGET_COLS:
        meta = dict(META)
        # source_cols 指向自身 → 阻止 _auto_adopt_new_columns 再生成模板特征
        meta["source_cols"] = [name]
        reg.register_new(name, meta)
    reg.save()

    feats = reg.features
    missing = [n for n in cyq_ext.TARGET_COLS if n not in feats]
    if missing:
        raise SystemExit(f"注册缺失: {missing}")
    inactive = [n for n in cyq_ext.TARGET_COLS if not feats[n].get("active", True)]
    if inactive:
        raise SystemExit(f"未激活: {inactive}")
    if "peak_mass" in feats:
        raise SystemExit("peak_mass 不应被注册")

    print(f"[ok] 已注册 {len(cyq_ext.TARGET_COLS)} 个筹码形态特征:")
    for name in cyq_ext.TARGET_COLS:
        m = feats[name]
        print(
            f"    {name}: dim={m['dim_group']} active={m['active']} "
            f"grade={m['grade']} source_cols={m['source_cols']}"
        )
    print(f"    注册中心: {reg.path} (共 {len(feats)} 特征)")


if __name__ == "__main__":
    main()
