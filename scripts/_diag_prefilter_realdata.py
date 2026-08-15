# -*- coding: utf-8 -*-
"""族级预过滤真实数据验证 (2026-08-15, 一次性诊断).

用最新生产选择集 selected_main_20260814T115937.json 的 86 个 brute 列作 need,
对真实 V3 面板末段 60 天切片逐族对比:
  new = generate_columns (预过滤开启, 生产路径)
  old = 同函数短路禁用 (逐字节等价旧路径)
断言: 空族两边都 None; 有交集族两边列名+数值逐字节一致 (含 NaN 位置).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from config import settings
from app.pipeline1.feature_selector import BRUTE_FAMILIES, BruteForceGenerator

SEL = r"D:\AMINQT\DATA OTHERS\factor_registry\selected_main_20260814T115937.json"


def load_slice():
    dset = ds.dataset(settings.PANEL_V3_PATH)
    max_date = dset.scanner(columns=["date"]).to_table().column("date").to_pandas().max()
    cut = max_date - pd.Timedelta(days=60)
    print(f"面板 max_date={max_date}, 切片 date>={cut}")
    tbl = dset.to_table(filter=ds.field("date") >= cut)
    df = tbl.to_pandas()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def main():
    d = json.load(open(SEL, encoding="utf-8"))
    feats = d["features"] if isinstance(d["features"], list) else list(d["features"])
    need = {f for f in feats if "_brute_" in str(f)}
    print(f"need={len(need)} brute 列 (来自 {SEL.split(chr(92))[-1]})")

    df = load_slice()
    gen = BruteForceGenerator()
    raw_cols = gen._eligible(df)
    print(f"切片 rows={len(df)} syms={df['symbol'].nunique()} raw_cols={len(raw_cols)}")

    def run(fam, disable_shortcircuit):
        orig = gen._family_candidate_names
        if disable_shortcircuit:
            # 返回与 need 必有交集的候选集 → isdisjoint 恒 False → 短路禁用 = 旧路径
            gen._family_candidate_names = lambda f, r: set(need)
        try:
            return gen.generate_columns(df, fam, need, raw_cols=raw_cols, dtype="float32")
        finally:
            gen._family_candidate_names = orig

    n_skipped = 0
    for fam in BRUTE_FAMILIES:
        t0 = time.time(); new = run(fam, False); t_new = time.time() - t0
        t0 = time.time(); old = run(fam, True); t_old = time.time() - t0
        if old is None:
            assert new is None, f"{fam}: old=None 而 new={type(new)} (误短路漏列!)"
            n_skipped += 1
            print(f"{fam}: 空族一致 None  [new {t_new:.1f}s / old {t_old:.1f}s]")
            continue
        assert new is not None, f"{fam}: old 有产出而 new 被误短路"
        assert set(new.columns) == set(old.columns), (
            f"{fam}: 列漂移 {set(old.columns) ^ set(new.columns)}"
        )
        for c in old.columns:
            np.testing.assert_array_equal(
                new[c].to_numpy(), old[c].to_numpy(), err_msg=f"{fam}/{c}"
            )
        print(f"{fam}: 有交集 {len(new.columns)} 列逐字节一致 [new {t_new:.1f}s / old {t_old:.1f}s]")

    print(f"PASS: {len(BRUTE_FAMILIES)} 族全等, 其中 {n_skipped} 族空族短路 (生产省掉该族全 symbol 白算)")


if __name__ == "__main__":
    main()
