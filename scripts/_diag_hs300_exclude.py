# -*- coding: utf-8 -*-
"""_diag_hs300_exclude.py — HS300 成分股 × 最近1年 列诊断 → EXCLUDE_COLS 候选.

复用 _diag_column_feed.py 的 IC/chg/tier 函数, 只把窗口从全市场3年换成 HS300×1年:
1. HS300 成分股: ak.index_stock_cons("000300") (当前时点快照, 作为1年窗口近似)
2. 面板最近 1 年, 同口径 label_pm_{2,3,5}d_net + 停牌遮蔽 + 近端6天遮蔽
3. 每列 wIC (=0.45*IC2d+0.35*IC3d+0.2*IC5d), chg (时序变化率), tier
4. 输出落盘 data/_diag_hs300_exclude_<ts>.log + .json (WORM, 只报告不 implement)

用法: python scripts/_diag_hs300_exclude.py
"""
import gc
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.label_engine import LabelEngine
from scripts._diag_column_feed import (
    B_EVENT_PREFIX,
    B_EXTRA,
    LABELS,
    MASK_RECENT_DAYS,
    WEIGHTS,
    daily_rank_ic_multi,
    temporal_variation,
    tier_of,
    weighted_ic,
)

logging.disable(logging.CRITICAL)

YEARS = 1


def _fetch_hs300() -> list:
    """HS300 当前成分股, 归一化为面板 6 位 symbol."""
    import akshare as ak

    df = ak.index_stock_cons(symbol="000300")
    col = "品种代码" if "品种代码" in df.columns else "成分券代码"
    syms = []
    for s in df[col].astype(str):
        s = s.strip()
        if "." in s:  # e.g. 600519.SH
            s = s.split(".")[0]
        syms.append(s)
    return list(dict.fromkeys(syms))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from scripts._diag_column_feed import _schema_cols
    from app.pipeline1.feature_selector import BruteForceGenerator

    hs300 = _fetch_hs300()
    print(f"HS300 成分股 {len(hs300)} 只")

    schema_names = _schema_cols(PANEL_V3_PATH)
    gen = BruteForceGenerator(eligible_cols=schema_names)
    elig = gen._eligible(pd.DataFrame(columns=schema_names))
    read_cols = list(
        dict.fromkeys(
            ["date", "symbol", "close_hfq", "amount", "volume", "is_suspended"]
            + list(elig)
        )
    )

    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)
    df = df[df["symbol"].isin(hs300)].reset_index(drop=True)
    print(f"HS300 面板行 {len(df):,} / 股 {df['symbol'].nunique()}")

    df = LabelEngine.build_labels(df, session="PM")
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    latest = df["date"].max()
    cutoff = latest - pd.DateOffset(years=YEARS)
    tr = df[df["date"] >= cutoff].reset_index(drop=True)
    del df
    gc.collect()
    print(
        f"训练窗口 {cutoff.date()} .. {latest.date()} | rows={len(tr):,} "
        f"stocks={tr['symbol'].nunique()} | 目标非空率 "
        f"{tr['label_pm_2d_net'].notna().mean():.1%}/"
        f"{tr['label_pm_3d_net'].notna().mean():.1%}/"
        f"{tr['label_pm_5d_net'].notna().mean():.1%}"
    )

    ics = daily_rank_ic_multi(tr, elig, LABELS)
    n_samp = min(300, tr["symbol"].nunique())
    tv = temporal_variation(tr, elig, n_stocks=n_samp)
    rows = []
    for c in elig:
        per_horizon = {lab: ics[lab][c] for lab in LABELS}
        wic = weighted_ic(per_horizon)
        rows.append(
            {
                "col": c,
                "wIC": wic,
                "IC2d": per_horizon["label_pm_2d_net"],
                "IC3d": per_horizon["label_pm_3d_net"],
                "IC5d": per_horizon["label_pm_5d_net"],
                "chg": tv[c],
            }
        )
    rows.sort(key=lambda r: -(abs(r["wIC"]) if r["wIC"] == r["wIC"] else float("inf")))

    lines = []
    lines.append(f"训练窗口 {cutoff.date()} .. {latest.date()} | rows={len(tr):,} "
                 f"stocks={tr['symbol'].nunique()}")
    lines.append("")
    lines.append(f"{'col':<26}{'wIC':>8}{'IC2d':>8}{'IC3d':>8}{'IC5d':>8}{'chg':>7}  tier")
    lines.append("-" * 74)
    lines.append("wIC = 0.45*IC2d + 0.35*IC3d + 0.2*IC5d | tier: A=brute展开 B/C=仅level")
    for r in rows:
        cat = tier_of(r["chg"], r["col"])
        fmt = lambda v: f"{v:+.4f}" if v == v else "   nan"
        chg_s = f"{r['chg']:.3f}" if r["chg"] == r["chg"] else "  nan"
        lines.append(
            f"{r['col']:<26}{fmt(r['wIC']):>8}{fmt(r['IC2d']):>8}"
            f"{fmt(r['IC3d']):>8}{fmt(r['IC5d']):>8}{chg_s:>7}  {cat}"
        )

    counts = {"A": 0, "B": 0, "C": 0}
    for r in rows:
        counts[tier_of(r["chg"], r["col"])] += 1
    lines.append("")
    lines.append(
        f"汇总: A={counts['A']} (brute展开 -> {counts['A']*32:,} 特征) | "
        f"B={counts['B']} (仅level) | C={counts['C']} (静态仅level)"
    )

    text = "\n".join(lines)
    print(text)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("data", f"_diag_hs300_exclude_{ts}.log")
    json_path = os.path.join("data", f"_diag_hs300_exclude_{ts}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n已落盘: {log_path}\n        {json_path}")


if __name__ == "__main__":
    main()
