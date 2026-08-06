# -*- coding: utf-8 -*-
"""_diag_fresh_selection.py — 按"目标分层设计"在当前面板上重选 MAIN 特征, 输出 B/C 归属 (只读).

目标设计 (用户裁决 + 提案):
  A = 日频变化列 → brute 展开 (32 transforms)
  B = 事件列(lhb_*/bt_*/sh_*) + float_share + chg∈[0.01,0.1) 慢变列 → 仅 level
  C = chg<0.01 常量列 → 仅 level
  注意: 季度财务列在 feature_selector.EXCLUDE_COLS 里已排除 brute (代码早已如此).
  全部 level (A+B+C) 照常进 MAIN dedup_l2.

brute 展开用 generate_family(float32, 逐家族) 内存安全, 规避全量 float64 OOM.

用法: python scripts/_diag_fresh_selection.py
"""
import gc
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_selector import (
    BruteForceGenerator,
    FeatureSelector,
    apply_event_scope_screens,
)
from app.pipeline1.train_runner import prepare_board_frame
from _diag_column_feed import (
    temporal_variation,
    tier_of,
)

logging.disable(logging.CRITICAL)


class TieredGen(BruteForceGenerator):
    """只对 A 列 brute 展开; generate_family float32 逐家族, 内存安全."""

    def __init__(self, a_cols, **kw):
        super().__init__(**kw)
        self._a = list(a_cols)

    def _eligible(self, df):
        return [c for c in self._a if c in df.columns]

    def generate(self, df, raw_cols=None):
        raw = raw_cols or self._eligible(df)
        new = None
        # generate() 只用 rolling_max 产出 max+min; 跳过 rolling_min 避免重复列
        for fam in self.BASE_TRANSFORM_DEFS:
            if fam == "rolling_min":
                continue
            part = self.generate_family(df, fam, raw_cols=raw, dtype="float32")
            new = part if new is None else new.join(part)
        return new


def base_of(c):
    return c.split("_brute_")[0] if "_brute_" in c else c


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()

    panel = pd.read_parquet(PANEL_V3_PATH)
    print(f"[1/5] 面板 {len(panel):,} 行 x {panel.shape[1]} 列 "
          f"({(time.time()-t0):.0f}s)", flush=True)

    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    main_df, dual_df = cleaner.run_train(panel)
    del panel, dual_df
    gc.collect()
    print(f"[2/5] 清洗后 main {len(main_df):,} 行, {main_df['symbol'].nunique()} 股 "
          f"({(time.time()-t0):.0f}s)", flush=True)

    df = prepare_board_frame(main_df, features, float_shares_map=None,
                             cross_sectional_rank=False)
    del main_df
    gc.collect()
    print(f"[3/5] 特征+标签面板 {len(df):,} 行 x {df.shape[1]} 列 "
          f"({(time.time()-t0):.0f}s)", flush=True)

    gen0 = BruteForceGenerator()
    elig = gen0._eligible(df)
    tv = temporal_variation(df, elig)
    tiers = {c: tier_of(tv[c], c) for c in elig}
    nA = sum(1 for t in tiers.values() if t == "A")
    nB = sum(1 for t in tiers.values() if t == "B")
    nC = sum(1 for t in tiers.values() if t == "C")
    a_cols = [c for c in elig if tiers[c] == "A"]
    print(f"[4/5] eligible={len(elig)} | A={nA} (brute->{nA*32:,}) "
          f"| B={nB} (仅level) | C={nC} (仅level)  ({time.time()-t0:.0f}s)", flush=True)

    sel = FeatureSelector()
    selected = sel.select(df, "main", generator=TieredGen(a_cols))
    selected = apply_event_scope_screens(selected, df)
    selected = [c for c in selected if c in df.columns]
    print(f"[5/5] 选中 {len(selected)} 个 ({(time.time()-t0):.0f}s)", flush=True)

    # ── 归属分类 ──
    comp = {"A_brute": [], "A_level": [], "B_level": [], "C_level": [], "other": []}
    for c in selected:
        b = base_of(c)
        is_brute = "_brute_" in c
        t = tiers.get(b, "?")
        if is_brute:
            comp["A_brute"].append(c)
        elif t == "A":
            comp["A_level"].append(c)
        elif t == "B":
            comp["B_level"].append(c)
        elif t == "C":
            comp["C_level"].append(c)
        else:
            comp["other"].append(c)

    print("\n===== 新选中特征 B/C 归属 =====")
    for k, lst in comp.items():
        print(f"  {k:<10} {len(lst):>4}   e.g. {lst[:4]}")

    print(f"\n-- 从 B 列选中的 level 特征 ({len(comp['B_level'])}) --")
    for c in comp["B_level"]:
        print(f"    {c}")
    print(f"\n-- 从 C 列选中的 level 特征 ({len(comp['C_level'])}) --")
    for c in comp["C_level"]:
        print(f"    {c}")
    if comp["other"]:
        print(f"\n-- 未在 eligible 中的选中特征 other ({len(comp['other'])}) --")
        for c in comp["other"]:
            print(f"    {c}")

    # 全展开会额外出现的 B/C brute 变体数 (目标设计不生成)
    excluded_brute = sum(32 for c in elig if tiers[c] in ("B", "C"))
    print(f"\n[对比] 全展开会比目标设计多 {excluded_brute:,} 个 B/C brute 变体进池 "
          f"(目标设计全部不生成)")

    out = {
        "board": "main",
        "design": "tiered A-brute + B/C-level",
        "created": time.strftime("%Y%m%dT%H%M%S"),
        "pool": {"eligible": len(elig), "A": nA, "B": nB, "C": nC,
                 "A_brute_feats": nA * 32},
        "selected_count": len(selected),
        "features": selected,
    }
    path = os.path.join("data", f"fresh_selected_main_{out['created']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已存 {path}")


if __name__ == "__main__":
    main()
