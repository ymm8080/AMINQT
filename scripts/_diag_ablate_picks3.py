# -*- coding: utf-8 -*-
"""归因: 消融 picks 差异是否全部来自停牌缺口符号 (rolling 行计数跨缺口污染)."""
import glob
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

JSON = glob.glob("data/_ablate_train_window_quality_*.json")[-1]
print(f"分析 JSON: {JSON}")
out = json.load(open(JSON, encoding="utf-8"))
print(f"verdict: faithful={out['verdict']['faithful_3y_reduced_vs_json']} "
      f"picks_identical={out['verdict']['picks_identical']} "
      f"n_picks_diffs={out['verdict']['n_picks_diffs']}")

SYS_KEYS = ["sniper", "fusion", "slow_bull"]


def picks_of(board, wn, sname, k):
    return set(tuple(x) for x in out["boards"][board][wn]["picks"][sname][k])


# 汇总每窗总选股对数 (仅 3y/2y/1y, 不含 ref3y_json 无 picks)
total_pairs = 0
diff_syms: set[str] = set()
for b in out["boards"]:
    for lab in out["windows_days"]:
        for sname in SYS_KEYS:
            for k, pk in out["boards"][b][lab]["picks"][sname].items():
                p3 = set(tuple(x) for x in pk)
                p1 = picks_of(b, "1y", sname, k)
                p2 = picks_of(b, "2y", sname, k)
                total_pairs += len(p3 | p1 | p2)
                for wn_picks in (p2, p1):
                    diff_syms |= (p3 ^ wn_picks)

# 收集所有 diff 的 (symbol) —— 需要日期来查缺口, 故重建 (date, symbol) 差异集
diff_cells = []
for b in out["boards"]:
    for lab in out["windows_days"]:
        for sname in SYS_KEYS:
            for k, pk in out["boards"][b][lab]["picks"][sname].items():
                p3 = set(tuple(x) for x in pk)
                for wn in ("2y", "1y"):
                    pwn = picks_of(b, wn, sname, k)
                    if p3 != pwn:
                        diff_cells.append((b, sname, k, wn, p3 ^ pwn))

# 查 3y 面板: 每个 diff (symbol,date) 前 60 交易日是否有 >10 自然日的缺口 (停牌)
from scripts._ablate_train_window_quality import load_window

print("加载 3y 面板判定缺口...", flush=True)
work = load_window(726)
sub = work[work["board"].isin(out["boards"])][["symbol", "date", "is_suspended"]]
sub = sub.sort_values(["symbol", "date"])
g = sub.groupby("symbol")["date"]
prev = g.shift(1)
gap_days = (pd.to_datetime(sub["date"]) - pd.to_datetime(prev)).dt.days
sub = sub.assign(gap_days=gap_days)
gc.collect()

n_gap_pick, n_total_diff_pick = 0, 0
all_gap = True
for b, sname, k, wn, diffset in diff_cells:
    syms = {x[0] for x in diffset}
    for sym in syms:
        n_total_diff_pick += 1
        row = sub[sub["symbol"] == sym]
        # 符号在窗口内任意 60 交易日窗口里有 >10 自然日缺口 → 判定停牌缺口符号
        has_gap = bool((row["gap_days"] > 10).any())
        if has_gap:
            n_gap_pick += 1
        else:
            all_gap = False
            print(f"  非缺口差异: {b}/{sname}/{k}/{wn} sym={sym}")

print(f"\n=== 归因汇总 ===")
print(f"总对比 pick 对数: {total_pairs:,}")
print(f"差异 pick 数: {n_total_diff_pick} ({n_total_diff_pick/total_pairs*100:.3f}%)")
print(f"其中停牌缺口符号差异: {n_gap_pick} ({n_gap_pick/max(n_total_diff_pick,1)*100:.1f}%)")
print(f"全部差异均为停牌缺口符号: {all_gap}")
print(f"差异去重符号数: {len(diff_syms)}")
