# -*- coding: utf-8 -*-
"""V3 CYQ 特征注册中心迁移 (2026-08-02 A/B/C 删列决策).

- DEACTIVATE: 所有不在 KEEP-15 的 CYQ 条目 (过时 _x/_y 孪生、benefit_*、
  chip_skew/conc_90/cost50_rank/cost_spread、未落盘扩展 chip_gini/resistance_dist/support_dist)
- REGISTER active: 7 裸名基础列 + conc_90_industry_rank

迁移后 active CYQ 特征 = 恰好 KEEP 15 列 (7 基础 + 5 扩展 + 3 派生; peak_mass 等已否决列不注册).
2026-08-02 用户裁决: 面板 CYQ = 15, chip_gini/resistance_dist/support_dist 不回填 → 去激活.
原子写回 (tmp + os.replace).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REG = Path(
    os.getenv(
        "FACTOR_REGISTRY",
        r"D:\AMINQT\DATA OTHERS\factor_registry\feature_registry.json",
    )
)

# KEEP 15: 7 基础裸名 + 5 扩展 + 3 派生 (chip_gini/resistance_dist/support_dist 未落盘, 不注册)
KEEP_BASE = [
    "winner_ratio",
    "avg_cost",
    "pct_90_high",
    "pct_90_con",
    "cost_50pct",
    "cost_95pct",
    "weight_avg",
]
KEEP_EXT = [
    "chip_entropy",
    "chip_skew_dist",
    "peak_price",
    "peak_roc_5d",
    "peak_roc_20d",
]
KEEP_DERIVED = ["cost_bias", "conc_trend_20d", "conc_90_industry_rank"]
KEEP = set(KEEP_BASE + KEEP_EXT + KEEP_DERIVED)

# 派生列删列清单中 5 个名字不命中 is_cyq 标记 (无 cost_/chip/conc_90 等前缀),
# 显式列出以便同样去激活 (feature_engine 已不再产出, 注册中心需同步).
EXTRA_DEACTIVATE = {
    "conc_streak",
    "conc_streak_3d",
    "conc70_streak",
    "conc70_streak_3d",
    "conc_reversal",
}

# 命中 CYQ 命名空间但属于其他维度、不在本次删列范围 (K线几何)
_NON_CYQ = {"body_pct_ma20", "body_pct_ma5"}
_CYQ_MARKERS = (
    "chip",
    "peak",
    "entropy",
    "skew",
    "support",
    "resist",
    "winner",
    "benefit",
    "conc_90",
    "conc_trend",
    "cost50",
    "avg_cost",
    "cost_",
    "weight_avg",
    "pct_70",
    "pct_90",
)


def is_cyq(name: str) -> bool:
    return name not in _NON_CYQ and any(m in name for m in _CYQ_MARKERS)


def main() -> None:
    if not REG.exists():
        raise SystemExit(f"注册中心不存在: {REG}")
    d = json.load(open(REG, encoding="utf-8"))
    feats = d["features"]

    deactivated = []
    for name in list(feats):
        if not feats[name].get("active", False):
            continue
        if is_cyq(name) and name not in KEEP:
            feats[name]["active"] = False
            deactivated.append(name)
        elif name in EXTRA_DEACTIVATE:
            feats[name]["active"] = False
            deactivated.append(name)

    registered = []
    today = str(date.today())
    for name in KEEP_BASE:
        if name not in feats:
            feats[name] = {
                "dim_group": "dim21_chip_tushare",
                "active": True,
                "grade": "trial",
                "transform": "raw",
                "source_cols": [name],
                "created": today,
                "last_eval": "",
            }
            registered.append(name)
    for name in KEEP_EXT:
        if name not in feats:
            feats[name] = {
                "dim_group": "chip_morphology",
                "active": True,
                "grade": "trial",
                "transform": "raw",
                "source_cols": [name],
                "created": today,
                "last_eval": "",
            }
            registered.append(name)
    if "conc_90_industry_rank" not in feats:
        feats["conc_90_industry_rank"] = {
            "dim_group": "dim21_chip_tushare",
            "active": True,
            "grade": "unknown",
            "transform": "dim21_chip_tushare",
            "source_cols": [],
            "created": today,
            "last_eval": "",
        }
        registered.append("conc_90_industry_rank")

    active_cyq = sorted(
        n for n, f in feats.items() if is_cyq(n) and f.get("active", False)
    )
    if set(active_cyq) != KEEP:
        missing = sorted(KEEP - set(active_cyq))
        extra = sorted(set(active_cyq) - KEEP)
        raise SystemExit(
            f"断言失败: active CYQ != KEEP {len(KEEP)}\n  缺: {missing}\n  多: {extra}"
        )

    d["last_update"] = today
    tmp = REG.with_suffix(".json.tmp")
    json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, REG)

    print(f"deactivated ({len(deactivated)}):")
    for n in sorted(deactivated):
        print("  -", n)
    print(f"registered ({len(registered)}):")
    for n in sorted(registered):
        print("  +", n)
    print(f"\nactive CYQ = {len(active_cyq)} == KEEP 15 [ok]")


if __name__ == "__main__":
    main()
