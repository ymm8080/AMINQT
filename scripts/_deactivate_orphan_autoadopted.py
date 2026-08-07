"""去激活 5 个失源列的 _auto_adopted 注册项 (2026-08-02 审计).

面板已删除 source_cols: holder_count / sw_ret_1d / sw_index_close / sw_index_vol / turn.
注册中心仍 active → 与 V3 面板不一致, 用户裁决不需要 → 去激活.
WORM 备份 + 原子写回 (tmp + os.replace), 沿用 _migrate_cyq_registry.py 模式.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path

REG = Path(
    os.getenv(
        "FACTOR_REGISTRY",
        r"D:\AMINQT\DATA OTHERS\factor_registry\feature_registry.json",
    )
)
ORPHANS = ["holder_count", "sw_ret_1d", "sw_index_close", "sw_index_vol", "turn"]


def main() -> None:
    if not REG.exists():
        raise SystemExit(f"注册中心不存在: {REG}")
    d = json.load(open(REG, encoding="utf-8"))
    feats = d["features"]

    deactivated = []
    for name in ORPHANS:
        f = feats.get(name)
        if f is None:
            print(f"  [skip] 不存在: {name}")
            continue
        if not f.get("active", False):
            print(f"  [skip] 已非 active: {name}")
            continue
        f["active"] = False
        deactivated.append(name)

    if not deactivated:
        print("无变化, 退出.")
        return

    backup = REG.with_name(
        f"feature_registry_pre_deact5_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    shutil.copy2(REG, backup)
    print(f"备份: {backup}")

    d["last_update"] = str(date.today())
    tmp = REG.with_suffix(".json.tmp")
    json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, REG)

    print(f"deactivated ({len(deactivated)}): {sorted(deactivated)}")
    print("[done]")


if __name__ == "__main__":
    main()
