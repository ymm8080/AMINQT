# -*- coding: utf-8 -*-
"""_diag_analog_pipeline.py — 定位类比列跑的价格判定污染源 (2026-08-04).

现象: 面板直读口径 (LabelEngine) 下 close = TS·周 负 -0.042; 类比列跑 (run_train +
FeatureEngineV35.build) 下 close = TS·月 正 +0.069. 本脚本分解:
  1) run_train 是否改写 close/close_hfq (同 (symbol,date) 对比面板);
  2) 虚拟退市行数量 + delisted_virtual_ret 值;
  3) 只用 run_train 输出 (不跑特征引擎) 重算 close 的 6格 → 看清洗是否单独翻转判定.
输出: data/_diag_analog_pipeline_<ts>.log (WORM).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.label_engine import LabelEngine
from scripts._diag_column_feed import LABELS, MASK_RECENT_DAYS
from scripts._classify_freq_full import (
    MIN_CROSS,
    MIN_OBS,
    WINDOWS,
    _f,
    _wtsic,
    group_spearman,
)


def classify(work, col):
    g_grp = work.groupby("symbol")
    wins = {}
    for w in WINDOWS.values():
        wins[f"{col}_p{w}"] = (work[col] / g_grp[col].shift(w) - 1.0).astype("float64")
    g_sym, g_date = work["symbol"], work["date"]
    lab_sym = {label: work.groupby("symbol")[label].rank() for label in LABELS}
    lab_date = {label: work.groupby("date")[label].rank() for label in LABELS}
    tsic, xic = {}, {}
    for f, wc in wins.items():
        wr_sym = wc.groupby(g_sym.values).rank()
        wr_date = wc.groupby(g_date.values).rank()
        tsic[f] = {
            label: group_spearman(wr_sym, lab_sym[label], g_sym, MIN_OBS)
            for label in LABELS
        }
        xic[f] = {
            label: group_spearman(wr_date, lab_date[label], g_date, MIN_CROSS)
            for label in LABELS
        }
    ts = {w: _wtsic(tsic[f"{col}_p{w}"]) for w in (1, 5, 20)}
    xs = {w: _wtsic(xic[f"{col}_p{w}"]) for w in (1, 5, 20)}
    cells = {
        "TS日": ts[1],
        "TS周": ts[5],
        "TS月": ts[20],
        "XS日": xs[1],
        "XS周": xs[5],
        "XS月": xs[20],
    }
    return cells


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    lines = []
    lines.append("=" * 78)
    lines.append("  类比列跑污染源定位 — run_train 分解测试")
    lines.append("=" * 78)

    panel = pd.read_parquet(PANEL_V3_PATH)
    lines.append(f"面板 rows={len(panel):,} stocks={panel['symbol'].nunique():,}")

    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)

    # 1) run_train 是否改写 close/close_hfq
    for b, bdf in (("main", main_df), ("dual", dual_df)):
        if bdf is None or len(bdf) == 0:
            continue
        bdf = bdf.sort_values(["symbol", "date"]).reset_index(drop=True)
        sub = panel[["symbol", "date", "close", "close_hfq"]].merge(
            bdf[["symbol", "date", "close", "close_hfq"]],
            on=["symbol", "date"],
            suffixes=("_panel", "_clean"),
            how="inner",
        )
        n = len(sub)
        d_close = (sub["close_panel"] - sub["close_clean"]).abs().max()
        d_hfq = (sub["close_hfq_panel"] - sub["close_hfq_clean"]).abs().max()
        n_virt = int(bdf["is_virtual"].fillna(0).sum()) if "is_virtual" in bdf else 0
        lines.append(
            f"[{b}] 清洗后 rows={len(bdf):,} 虚拟行={n_virt:,} | "
            f"重叠(symbol,date)={n:,} | close 最大差={d_close:.4f} | "
            f"close_hfq 最大差={d_hfq:.4f}"
        )

    # delisted_virtual_ret 值
    dvr = getattr(cleaner.cfg, "delisted_virtual_ret", None)
    lines.append(f"delisted_virtual_ret = {dvr}")

    # 2) 只用 run_train 输出重算 close 6格 (不跑特征引擎)
    both = pd.concat([main_df, dual_df], ignore_index=True)
    both = both.sort_values(["symbol", "date"]).reset_index(drop=True)
    both = LabelEngine.build_labels(both, session="PM")
    both = LabelEngine.mask_suspension(both)
    both = LabelEngine.mask_recent_days(both, days=MASK_RECENT_DAYS)
    cutoff = both["date"].max() - pd.DateOffset(years=3)
    work = both[both["date"] >= cutoff].reset_index(drop=True)
    lines.append(
        f"run_train 口径 3y: rows={len(work):,} stocks={work['symbol'].nunique():,}"
    )

    for c in ["close", "close_hfq"]:
        if c not in work.columns:
            continue
        cells = classify(work, c)
        best = max(cells, key=lambda k: abs(cells[k]))
        lines.append(
            f"  {c:<12}{_f(cells['TS日']):>8}{_f(cells['TS周']):>8}{_f(cells['TS月']):>8}"
            f"{_f(cells['XS日']):>8}{_f(cells['XS周']):>8}{_f(cells['XS月']):>8}"
            f"  ← {best} ({abs(cells[best]):.4f})  [run_train 口径]"
        )

    text = "\n".join(lines)
    print(text, flush=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_diag_analog_pipeline_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}", flush=True)


if __name__ == "__main__":
    main()
