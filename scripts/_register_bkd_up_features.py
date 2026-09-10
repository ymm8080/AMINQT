"""一次性注册: 把 dim36 双向短期族 14 特征注册进特征注册中心.

背景: 训练侧 build(registry=...) 按 has_dim_group 门控 dim — 未注册时
dim36_bkd_up 整族不物化, force_include 名全部落空. 本脚本即"先有鸡"的一次性动作
(同 scripts/_register_cyq_features.py 先例).

用法:
  python scripts/_register_bkd_up_features.py
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

from app.pipeline1.feature_registry import FeatureRegistry

# 与 app/pipeline1/feature_engine_v35.py dim36_bkd_up / A/B v6 FAMILY 逐字一致
TARGET_COLS = [
    "bkd_dn_streak", "bkd_dn_days5", "bkd_dd5_high20", "bkd_dd_high60",
    "bkd_min10_dist", "bkd_below_ma_cnt", "bkd_ma_bear_align", "bkd_ma5_slope5",
    "up_vol_confirm5", "up_body5", "up_break20_vol", "up_followthrough",
    "up_pullback_depth", "up_gap_hold",
]

META = {
    "dim_group": "dim36_bkd_up",
    "active": True,
    "grade": "trial",
    "transform": "raw",
    "created": "2026-09-10",
    "last_eval": "",
}


def main() -> None:
    reg = FeatureRegistry()
    for name in TARGET_COLS:
        meta = dict(META)
        # source_cols 指向自身 → 阻止 _auto_adopt_new_columns 再生成模板特征
        meta["source_cols"] = [name]
        reg.register_new(name, meta)
    reg.save()

    feats = reg.features
    missing = [n for n in TARGET_COLS if n not in feats]
    if missing:
        raise SystemExit(f"注册缺失: {missing}")
    inactive = [n for n in TARGET_COLS if not feats[n].get("active", True)]
    if inactive:
        raise SystemExit(f"未激活: {inactive}")
    if not reg.has_dim_group("dim36_bkd_up"):
        raise SystemExit("has_dim_group('dim36_bkd_up') 仍为 False, 门控不会放行")

    print(f"[ok] 已注册 {len(TARGET_COLS)} 个 dim36 双向短期特征:")
    for name in TARGET_COLS:
        m = feats[name]
        print(
            f"    {name}: dim={m['dim_group']} active={m['active']} "
            f"grade={m['grade']} source_cols={m['source_cols']}"
        )
    print(f"    注册中心: {reg.path} (共 {len(feats)} 特征)")


if __name__ == "__main__":
    main()
