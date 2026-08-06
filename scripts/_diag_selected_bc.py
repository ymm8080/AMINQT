# -*- coding: utf-8 -*-
"""_diag_selected_bc.py — 统计当前选中特征里, 有多少(哪些)从 B/C 原始列衍生 (只读).

"衍生" = brute 变体 (base_brute_*) 或该 B/C 列本身的 level.
B/C 分层与 _diag_column_feed 完全一致 (A=brute展开候选 / B=事件+float_share+慢变 / C=常量).
被 B/C 基列衍生的选中特征, 在"B/C 不展开"的分层下会从候选池消失 → 选中结果会变.

用法: python scripts/_diag_selected_bc.py
"""

import gc
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)  # import _diag_column_feed

import pandas as pd

from config.settings import PANEL_V3_PATH
from _diag_column_feed import (
    B_EVENT_PREFIX,
    B_EXTRA,
    temporal_variation,
    tier_of,
)

B_EVENT_OR_EXTRA = {c for c in B_EXTRA} | set(B_EVENT_PREFIX)

logging.disable(logging.CRITICAL)

REGISTRY = r"D:\AMINQT\DATA OTHERS\factor_registry"
YEARS = 3
MASK_RECENT_DAYS = 6

BASE_KEYS = ("date", "symbol", "close_hfq", "amount", "volume", "is_suspended")


def base_of(c):
    return c.split("_brute_")[0] if "_brute_" in c else c


def load_current(board):
    path = os.path.join(REGISTRY, f"selected_{board}_current.json")
    with open(path, encoding="utf-8") as f:
        cur = json.load(f)
    if "features" in cur:
        return cur
    vp = os.path.join(REGISTRY, cur["active_version"])
    with open(vp, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    sel = {b: load_current(b) for b in ("main", "dual")}
    bases = {}
    for b, doc in sel.items():
        for c in doc["features"]:
            bases.setdefault(base_of(c), set()).add(b)
    all_bases = list(bases)

    # 当前面板 schema — 选中清单里基列已不在面板的 (旧列/改名) 标记 stale
    import pyarrow.parquet as pq

    schema_names = {f.name for f in pq.read_schema(PANEL_V3_PATH)}
    present = [b for b in all_bases if b in schema_names]
    sorted(set(all_bases) - set(present))

    read_cols = list(dict.fromkeys(list(BASE_KEYS) + present))
    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)
    from app.pipeline1.label_engine import LabelEngine

    df = LabelEngine.build_labels(df, session="PM")
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    latest = df["date"].max()
    cutoff = latest - pd.DateOffset(years=YEARS)
    tr = df[df["date"] >= cutoff].reset_index(drop=True)
    del df
    gc.collect()
    tv = temporal_variation(tr, present)
    del tr
    gc.collect()

    tier = {b: tier_of(tv.get(b, float("nan")), b) for b in present}

    for board in ("main", "dual"):
        doc = sel[board]
        feats = doc["features"]
        print(f"\n===== {board} | {doc.get('pipeline', '?')} | 选中 {len(feats)} =====")
        nB = nC = nA = nS = 0
        b_list, c_list, s_list = [], [], []
        for c in feats:
            b = base_of(c)
            if b in tier:
                t = tier[b]
            elif b in B_EVENT_OR_EXTRA:
                t = "B"
            else:
                t = "stale"
            if t == "B":
                nB += 1
                b_list.append(c)
            elif t == "C":
                nC += 1
                c_list.append(c)
            elif t == "A":
                nA += 1
            else:
                nS += 1
                s_list.append(c)
        print(f"  [基列分层] A={nA} | B={nB} | C={nC} | 基列不在当前面板(stale)={nS}")
        if b_list:
            print(f"  -- 从 B 列衍生的选中特征 ({nB}) --")
            for c in b_list:
                print(f"    {c}")
        if c_list:
            print(f"  -- 从 C 列衍生的选中特征 ({nC}) --")
            for c in c_list:
                print(f"    {c}")
        if s_list:
            print(f"  -- 基列不在当前面板的选中特征 ({nS}) --")
            for c in s_list[:60]:
                print(f"    {c}")
            if len(s_list) > 60:
                print(f"    ... 其余 {len(s_list) - 60} 个省略")

    print("\n[基列分层明细] base -> tier (chg); 不在当前面板=stale")
    for b in sorted(all_bases, key=lambda x: -abs(tv.get(x, 0) or 0)):
        if b in tier:
            t = tier[b]
        elif b in B_EVENT_OR_EXTRA:
            t = "B"
        else:
            t = "stale"
        v = tv.get(b, float("nan"))
        v_s = f"{v:.3f}" if v == v else "  nan"
        print(f"  {b:<26} {t:<6} {v_s}")


if __name__ == "__main__":
    main()
