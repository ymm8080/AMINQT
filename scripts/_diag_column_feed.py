"""_diag_column_feed.py — 诊断列喂入质量 (只读, 不改任何训练逻辑).

指标 (训练窗口最近 3 年, 无未来函数):
1. 每列日频 rank IC vs 实际目标 label_pm_{2,3,5}d_net
   (B9 执行口径: close[T+1+k]/close[T+1]-1, 与训练同口径 + 停牌遮蔽 + 近端 6 天遮蔽).
   加权 IC = 0.45×IC_2d + 0.35×IC_3d + 0.2×IC_5d (LABEL_WEIGHTS).
   用户裁决 (2026-08-03): 若只选一个视界, 取 t+3.
2. 每列时序变化率: 300 只抽样股票内, 相邻日 diff≠0 的占比 (忽略 NaN 过渡).
   ~0 → 常量; ~0.02 → 季度阶跃; ~0.5-1.0 → 日频变化.
3. 行卫生: 零成交量/零成交额行数.

分层 (只决定是否 brute 展开, 不决定是否入选择):
  A = 日频变化 → brute 展开候选 (32 transforms)
  B = 事件列 + float_share (用户裁决: 仅 level, 不展开) + 季度/慢变列
  C = 常量/阶跃静态列 → 仅 level
  全部列 (A raw + A brute + B + C) 仍流经 MAIN dedup_l2 / DUAL gate_d.

用法: python scripts/_diag_column_feed.py
"""

import gc
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from app.pipeline1.feature_selector import (
    TIER_EVENT_EXTRA,
    TIER_EVENT_PREFIX,
    BruteForceGenerator,
    temporal_variation,
    tier_of,
)
from app.pipeline1.label_engine import LabelEngine
from config.settings import PANEL_V3_PATH

# 用户裁决: 事件列 + float_share → B (仅 level, 不 brute 展开)
# 别名保留给 _diag_selected_bc 等下游脚本 (feature_selector 为唯一实现源).
B_EVENT_PREFIX = TIER_EVENT_PREFIX
B_EXTRA = TIER_EVENT_EXTRA

logging.disable(logging.CRITICAL)

YEARS = 3
MASK_RECENT_DAYS = 6  # 与 train_runner.prepare_board_frame 一致
LABELS = (f"label_pm_{k}d_net" for k in (2, 3, 5))
LABELS = tuple(LABELS)
WEIGHTS = {2: 0.45, 3: 0.35, 5: 0.2}  # LABEL_WEIGHTS 去 1d(权重0)

def _schema_cols(path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    sch = pq.read_schema(path)
    return [f.name for f in sch if f.type in (pa.float64(), pa.int64())]


def daily_rank_ic_multi(df, feats, labels):
    """按日截面 Spearman rank IC, 逐列×逐视界两两剔 NaN, mean over dates.

    Spearman = Pearson on within-date ranks. 居中必须用 rank 的日内均值
    (不能是原始列均值 — 否则伪 IC 退化为 -√3/2).
    Returns {label: {feat: ic}}.
    """
    df = df.dropna(subset=labels).copy()
    allc = list(feats) + list(labels)
    rank = df.groupby("date")[allc].rank(method="average")
    c = rank - rank.groupby(df["date"]).transform("mean")
    d = df["date"]
    out = {lab: {} for lab in labels}
    for f in feats:
        cf = c[f]
        for lab in labels:
            mask = cf.notna() & c[lab].notna()
            cfl = cf.where(mask)
            cl = c[lab].where(mask)
            num = (cfl * cl).groupby(d).sum()
            s2f = (cfl * cfl).groupby(d).sum()
            s2l = (cl * cl).groupby(d).sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                ic = num / np.sqrt(s2f * s2l)
            out[lab][f] = float(ic.mean(skipna=True))
    return out


def weighted_ic(ics):
    """按 LABEL_WEIGHTS (2/3/5d) 加权 IC; 仅用非 NaN 视界, 权重按占比归一."""
    acc = 0.0
    wsum = 0.0
    for k, w in WEIGHTS.items():
        v = ics[f"label_pm_{k}d_net"]
        if v == v:
            acc += w * v
            wsum += w
    return acc / wsum if wsum > 0 else float("nan")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    schema_names = _schema_cols(PANEL_V3_PATH)
    gen = BruteForceGenerator(eligible_cols=schema_names)
    elig = gen._eligible(pd.DataFrame(columns=schema_names))
    read_cols = [
        "date",
        "symbol",
        "close_hfq",
        "amount",
        "volume",
        "is_suspended",
    ] + list(elig)
    read_cols = list(dict.fromkeys(read_cols))

    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)
    df = LabelEngine.build_labels(df, session="PM")  # 同训练口径
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    latest = df["date"].max()
    cutoff = latest - pd.DateOffset(years=YEARS)
    tr = df[df["date"] >= cutoff].reset_index(drop=True)
    del df
    gc.collect()
    print(
        f"训练窗口 {cutoff.date()} .. {latest.date()} | rows={len(tr):,} "
        f"stocks={tr['symbol'].nunique()} | "
        f"目标 label_pm_2d/3d/5d_net 非空率: "
        f"{tr['label_pm_2d_net'].notna().mean():.1%}/{tr['label_pm_3d_net'].notna().mean():.1%}"
        f"/{tr['label_pm_5d_net'].notna().mean():.1%}"
    )

    vol0 = int((tr["volume"] == 0).sum())
    amt0 = int((tr["amount"] == 0).sum())
    print(
        f"[行卫生] volume==0: {vol0:,} ({vol0 / len(tr):.2%}) | "
        f"amount==0: {amt0:,} ({amt0 / len(tr):.2%})"
    )

    ics = daily_rank_ic_multi(tr, elig, LABELS)
    tv = temporal_variation(tr, elig)
    rows = []
    for c in elig:
        per_horizon = {lab: ics[lab][c] for lab in LABELS}
        wic = weighted_ic(per_horizon)
        rows.append(
            (
                c,
                wic,
                per_horizon["label_pm_2d_net"],
                per_horizon["label_pm_3d_net"],
                per_horizon["label_pm_5d_net"],
                tv[c],
            )
        )
    rows.sort(key=lambda t: -(abs(t[1]) if t[1] == t[1] else float("inf")))

    print(f"\n{'col':<26}{'wIC':>8}{'IC2d':>8}{'IC3d':>8}{'IC5d':>8}{'chg':>7}  tier")
    print("-" * 74)
    print("wIC = 0.45*IC2d + 0.35*IC3d + 0.2*IC5d | tier: A=brute展开 B/C=仅level")
    for c, wic, i2, i3, i5, chg in rows:
        cat = tier_of(chg, c)

        def fmt(v):
            return f"{v:+.4f}" if v == v else "   nan"

        chg_s = f"{chg:.3f}" if chg == chg else "  nan"
        print(
            f"{c:<26}{fmt(wic):>8}{fmt(i2):>8}{fmt(i3):>8}{fmt(i5):>8}{chg_s:>7}  {cat}"
        )

    counts = {"A": 0, "B": 0, "C": 0}
    for c, _, _, _, _, chg in rows:
        counts[tier_of(chg, c)] += 1
    nA = counts["A"]
    print(
        f"\n汇总: A={nA} (brute展开 -> {nA * 32:,} 特征) | B={counts['B']} (仅level) | "
        f"C={counts['C']} (静态仅level)"
    )


if __name__ == "__main__":
    main()
