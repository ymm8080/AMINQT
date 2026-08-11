"""探针4: 5 个"非缺口差异"符号 — 验证是否为 1y 截断冷启动 (窗口起点滚动特征 NaN) 而非真实质量差异."""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

JSON = glob.glob("data/_ablate_train_window_quality_*.json")[-1]
out = json.load(open(JSON, encoding="utf-8"))
SYMS = ["605162", "603387", "300717", "688530", "688551"]

# 1y 截断日期 (从 load_window 日志: 242d cutoff=2025-08-11)
CUTOFF = pd.Timestamp("2025-08-11")

# 从 JSON 拿 3y 面板的行日期范围
from scripts._ablate_train_window_quality import load_window

print("加载 3y 面板...", flush=True)
work = load_window(726)
work["date"] = pd.to_datetime(work["date"])
for s in SYMS:
    sub = work[work["symbol"] == s].sort_values("date")
    if sub.empty:
        print(f"{s}: 3y 帧无数据")
        continue
    first, last = sub["date"].iloc[0], sub["date"].iloc[-1]
    n = len(sub)
    max_gap = sub["date"].diff().dt.days.max()
    # 1y 帧起点后的首行 = 该符号在 1y 帧的首行
    after_cut = sub[sub["date"] >= CUTOFF]
    n_before_cut = (sub["date"] < CUTOFF).sum()
    print(f"\n{s}: rows={n}  span={first.date()}→{last.date()}  max_gap={max_gap:.0f}d")
    print(
        f"    截止日前行数={n_before_cut}  截止日后首行={after_cut['date'].iloc[0].date() if not after_cut.empty else 'N/A'}"
    )
    # 若 max_gap 大 → 停牌; 若 n_before_cut 多且连续 → 截断冷启动候选
    if after_cut.empty:
        print("    => 全部在截止日后 (次新/数据晚起)")

# 差异 pick 的日期分布: 是否集中在 1y 帧起点附近 (非 OOS)
print("\n=== 差异 pick 日期分布 (相对各窗起点) ===")
for b in out["boards"]:
    for sname in ("sniper", "fusion", "slow_bull"):
        for k in ("primary", "alt"):
            try:
                p3 = set(tuple(x) for x in out["boards"][b]["3y"]["picks"][sname][k])
                p1 = set(tuple(x) for x in out["boards"][b]["1y"]["picks"][sname][k])
            except KeyError:
                continue
            d = p3 ^ p1
            if not d:
                continue
            dates = pd.to_datetime([x[1] for x in d])
            early1y = (dates <= CUTOFF + pd.Timedelta("21d")).sum()
            print(
                f"{b}/{sname}/{k}: diff={len(d)}  其中1y起点后≤21交易日={early1y} ({early1y / len(d) * 100:.0f}%)"
            )
