# -*- coding: utf-8 -*-
"""V3 CYQ KEEP-15 落地验证 (2026-08-02): 面板列 + 注册中心 + 切片 build.

面板 CYQ = 15 (7 基础 + 5 扩展 + 3 派生). chip_gini/resistance_dist/support_dist
用户 2026-08-02 裁决不回填 → 不在 KEEP.

用法:
  python scripts/_verify_v3_cyq_keep15.py            # 面板/注册断言 (快)
  python scripts/_verify_v3_cyq_keep15.py --build    # 追加切片 build 断言 (慢)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from config.settings import CYQ_BASE_DELETE, CYQ_BASE_KEEP, PANEL_V3_PATH
from app.pipeline1 import cyq_ext

REG = Path(
    os.getenv(
        "FACTOR_REGISTRY",
        r"D:\AMINQT\DATA OTHERS\factor_registry\feature_registry.json",
    )
)

KEEP_BASE = set(CYQ_BASE_KEEP)
DROPPED_EXT = {"chip_gini", "resistance_dist", "support_dist"}  # 裁决不回填
KEEP_EXT = set(cyq_ext.TARGET_COLS) - DROPPED_EXT
KEEP_DERIVED = {"cost_bias", "conc_trend_20d", "conc_90_industry_rank"}
KEEP = KEEP_BASE | KEEP_EXT | KEEP_DERIVED
DELETE_BASE = set(CYQ_BASE_DELETE)

CYQ_MARKERS = ("chip", "peak", "entropy", "skew", "support", "resist", "winner",
               "benefit", "conc_90", "conc_trend", "cost50", "avg_cost", "cost_",
               "weight_avg", "pct_70", "pct_90")
_NON_CYQ = {"body_pct_ma20", "body_pct_ma5"}


def is_cyq(name: str) -> bool:
    return name not in _NON_CYQ and any(m in name for m in CYQ_MARKERS)


def main() -> None:
    fail = 0

    # ── 1. 面板断言 ──
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    cols = set(panel.columns)
    print(f"[panel] {PANEL_V3_PATH}  {len(panel):,} 行, {len(panel.columns)} 列")

    missing_keep_base = sorted(KEEP_BASE - cols)
    missing_keep_ext = sorted(KEEP_EXT - cols)
    missing_keep_derived = sorted(KEEP_DERIVED - cols)
    leftover_del = sorted(DELETE_BASE & cols)
    print(f"  KEEP 基础列 ({len(KEEP_BASE)}): {sorted(KEEP_BASE)}")
    print(f"  KEEP 扩展列 ({len(KEEP_EXT)}): {sorted(KEEP_EXT)}")
    print(f"  KEEP 派生列 ({len(KEEP_DERIVED)}): {sorted(KEEP_DERIVED)}")
    print(f"  缺 KEEP 基础: {missing_keep_base}")
    print(f"  缺 KEEP 扩展: {missing_keep_ext}")
    print(f"  缺 KEEP 派生: {missing_keep_derived}")
    print(f"  残留 DELETE 基础列: {leftover_del}")

    panel_cyq = sorted(c for c in cols if is_cyq(c))
    print(f"  面板 CYQ 相关列 ({len(panel_cyq)}): {panel_cyq}")

    if missing_keep_base or missing_keep_ext or missing_keep_derived or leftover_del:
        print("[FAIL] 面板 CYQ 列面 != 7 基础 + 5 扩展 + 3 派生")
        fail += 1
    else:
        print("[ok] 面板 = 15 CYQ 列 (7 基础 + 5 扩展 + 3 派生), 无 DELETE 残留")

    # ── 2. 注册中心断言 ──
    d = json.load(open(REG, encoding="utf-8"))
    feats = d["features"]
    active_cyq = sorted(
        n for n, f in feats.items() if is_cyq(n) and f.get("active", False)
    )
    print(f"\n[registry] {REG}  active CYQ = {len(active_cyq)}")
    print(f"  {active_cyq}")
    if set(active_cyq) != KEEP:
        print(f"[FAIL] active CYQ != KEEP 15\n  缺: {sorted(KEEP - set(active_cyq))}\n  多: {sorted(set(active_cyq) - KEEP)}")
        fail += 1
    else:
        print("[ok] active CYQ == KEEP 15")

    # ── 3. 切片 build 断言 (可选) ──
    if "--build" in sys.argv:
        from app.pipeline1.feature_registry import FeatureRegistry
        from app.pipeline1.feature_engine_v35 import FeatureEngineV35

        main = panel[panel["board"] == "main"]
        syms = sorted(main["symbol"].unique())[:3]
        sl = main[main["symbol"].isin(syms)]
        print(f"\n[build] 切片 {len(syms)} 只主板股票: {len(sl):,} 行")
        out = FeatureEngineV35().build(sl, registry=FeatureRegistry())
        out_cyq = sorted(c for c in out.columns if is_cyq(c))
        print(f"  build 输出 CYQ 列 ({len(out_cyq)}): {out_cyq}")
        if set(out_cyq) != KEEP:
            print(f"[FAIL] build 输出 CYQ != KEEP 15\n  缺: {sorted(KEEP - set(out_cyq))}\n  多: {sorted(set(out_cyq) - KEEP)}")
            fail += 1
        else:
            print("[ok] build 输出 CYQ == KEEP 15")

    print(f"\n==> {'[FAIL] ' + str(fail) + ' 项失败' if fail else 'ALL PASS [ok]'}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
