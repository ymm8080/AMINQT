# -*- coding: utf-8 -*-
"""_classify_freq_analog.py — FAMILY_ANALOG 类比列 6格实测 (替换同族猜测, 2026-08-04).

对 feature_selector.FAMILY_ANALOG 的全部类比列 (84) 跑真实 6格判定:
  TS/XR × 日/周/月 (1/5/20 变化窗口), 取 |IC| 最强格为判定,
与 FAMILY_ANALOG 同族猜测 (freq, type) 对比 → 一致 / 不一致清单.
不一致列是后续把 FAMILY_ANALOG → FREQ_ASSIGNMENT 升级的依据.

数据: 多数类比列不在持久化面板 (换手/筹码_x_y/行业/市场/快信号为训练时生成),
故走真实训练序列 cleaner.run_train → FeatureEngineV35.build → prepare_board_frame
(含标签 + 停牌/近端掩码). 逐板块构建 → 只留 [date,symbol,is_suspended,close_hfq]
+ 类比列 + labels, 立即 drop build 输出并 gc → 无 brute-force generate, 无 OOM.

输出: data/_classify_freq_analog_<ts>.log (WORM) + data/_classify_freq_analog_summary_<ts>.json
用法: python scripts/_classify_freq_analog.py
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
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_selector import FAMILY_ANALOG
from app.pipeline1.train_runner import prepare_board_frame
from scripts._diag_column_feed import LABELS
from scripts._classify_freq_full import (
    MIN_CROSS,
    MIN_OBS,
    WINDOWS,
    _wtsic,
    group_spearman,
)

logging.disable(logging.CRITICAL)
np.seterr(all="ignore")


def family_of(col: str) -> str:
    """按列名前缀推断族 (仅供报告可读性)."""
    if col.startswith(("open", "high", "low", "close", "pre_close")):
        return "价格"
    if col.startswith(("pct_", "cost_", "avg_cost", "weight_avg")):
        return "筹码/成本变体"
    if col.startswith(("sw_", "sector_return", "ind_", "market_")):
        return "行业/市场"
    if col in ("month", "list_days"):
        return "日历"
    if col.startswith(("benefit_part", "churn_suspect")):
        return "赢家占比/洗盘"
    if col in (
        "dv_ttm",
        "up_limit_raw",
        "down_limit_raw",
        "total_share",
        "float_share",
        "free_share",
    ):
        return "股息/股本/涨跌停"
    if col in (
        "turn",
        "turnover_rate_f",
        "rank_ff_turnover",
        "rank_amount",
        "liquidity_score",
        "adv20",
        "turnover_stability_5",
        "amihud_illiq",
        "amihud_illiquidity",
        "turnover_f_chg_5d",
        "free_float_turnover_rate_xrank",
        "amount_xrank",
        "ATR_pct",
    ):
        return "换手/流动性/波动"
    if col in (
        "close_vs_low",
        "overnight_ret",
        "ROC_3d",
        "gap_strength_5d",
        "gap_strength_20d",
        "gap_vs_ma5",
    ):
        return "快价格信号"
    return "其他"


def _load_train() -> pd.DataFrame:
    """真实训练序列 → 只保留类比列+labels+id 的合并 df (main+dual)."""
    panel = pd.read_parquet(PANEL_V3_PATH)
    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)
    del panel
    gc.collect()

    fe = FeatureEngineV35()
    base = ["date", "symbol", "is_suspended", "close_hfq"]
    need = set(base) | set(FAMILY_ANALOG.keys()) | set(LABELS)
    parts = []
    for board, bdf in (("main", main_df), ("dual", dual_df)):
        if bdf is None or len(bdf) == 0:
            continue
        print(f"[{board}] build ... (rows={len(bdf):,})", flush=True)
        d = prepare_board_frame(bdf, fe, None, cross_sectional_rank=(board != "main"))
        keep = [c for c in need if c in d.columns]
        parts.append(d[keep].assign(board=board))
        ncols = d.shape[1]
        del d
        gc.collect()
        print(f"[{board}] 训练面板 cols={ncols} → 裁剪保留 {len(keep)} 列", flush=True)
    del main_df, dual_df, fe
    gc.collect()
    merged = pd.concat(parts, ignore_index=True)
    return merged.sort_values(["symbol", "date"]).reset_index(drop=True)


def classify_col(work, g_sym, g_date, lab_sym, lab_date, col):
    """单列 6格判定 → (cells, best_cell, best_ic)."""
    g_grp = work.groupby("symbol")
    wins = {}
    for w in WINDOWS.values():
        wins[f"{col}_p{w}"] = (work[col] / g_grp[col].shift(w) - 1.0).astype("float64")
    tsic = {}
    xic = {}
    for f, wc in wins.items():
        wr_sym = wc.groupby(g_sym.values).rank()
        wr_date = wc.groupby(g_date.values).rank()
        tsic[f] = {
            lab: group_spearman(wr_sym, lab_sym[lab], g_sym, MIN_OBS) for lab in LABELS
        }
        xic[f] = {
            lab: group_spearman(wr_date, lab_date[lab], g_date, MIN_CROSS)
            for lab in LABELS
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
    best = max(cells, key=lambda k: abs(cells[k]))
    return cells, best, abs(cells[best])


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    def prog(msg):
        print(msg, flush=True)

    df = _load_train()
    prog(
        f"合并训练 df rows={len(df):,} stocks={df['symbol'].nunique()} "
        f"cols={df.shape[1]}"
    )
    latest = df["date"].max()
    cutoff = latest - pd.DateOffset(years=3)
    work = df[df["date"] >= cutoff].reset_index(drop=True)
    prog(
        f"全市场×3年 rows={len(work):,} stocks={work['symbol'].nunique()} "
        f"| {latest:%Y-%m-%d} 往前3年"
    )
    del df
    gc.collect()

    g_sym = work["symbol"]
    g_date = work["date"]
    lab_sym = {lab: work.groupby("symbol")[lab].rank() for lab in LABELS}
    lab_date = {lab: work.groupby("date")[lab].rank() for lab in LABELS}

    out = []
    out.append("=" * 86)
    out.append("  FAMILY_ANALOG 类比列 6格实测 — 与同族猜测对比")
    out.append("  实测: TS/XS × 日/周/月 (1/5/20窗口), 取 |IC| 最强格")
    out.append("=" * 86)
    out.append(f"{'feature':<24}{'族':<10}{'实测':>8}{'|IC|':>8}{'猜测':>8}{'结果':>4}")
    out.append("-" * 86)

    summary = []
    n_match = n_mismatch = n_missing = 0
    for col, (gfreq, gtype) in FAMILY_ANALOG.items():
        fam = family_of(col)
        if col not in work.columns:
            out.append(f"{col:<24}{fam:<10}{'缺失':>8}{'':>8}{gfreq:>8}{'??':>4}")
            summary.append(
                {
                    "feat": col,
                    "family": fam,
                    "status": "missing",
                    "guess_freq": gfreq,
                    "guess_type": gtype,
                }
            )
            n_missing += 1
            continue
        cells, best, ic = classify_col(work, g_sym, g_date, lab_sym, lab_date, col)
        real_freq, real_type = best[2:], best[:2]
        match = (real_freq == gfreq) and (real_type == gtype)
        tag = "✅" if match else "❌"
        if match:
            n_match += 1
        else:
            n_mismatch += 1
        out.append(
            f"{col:<24}{fam:<10}{best:>8}{ic:>8.4f}{gfreq + ':' + gtype:>8}{tag:>4}"
        )
        summary.append(
            {
                "feat": col,
                "family": fam,
                "status": "ok",
                "cells": {k: round(v, 4) if v == v else None for k, v in cells.items()},
                "verdict": best,
                "ic": round(ic, 4),
                "guess_freq": gfreq,
                "guess_type": gtype,
                "match": match,
            }
        )
        del cells
        gc.collect()
        prog(f"    done {col} ({fam})")

    out.append("-" * 86)
    out.append(
        f"一致 {n_match} / 不一致 {n_mismatch} / 缺失 {n_missing} / "
        f"合计 {len(FAMILY_ANALOG)}"
    )

    if n_mismatch:
        out.append("\n── 不一致明细 (升级 FAMILY_ANALOG 依据) ──")
        for s in summary:
            if s["status"] == "ok" and not s["match"]:
                out.append(
                    f"  {s['feat']:<24} 实测 {s['verdict']} (|IC| {s['ic']:.4f})"
                    f"  ≠ 猜测 {s['guess_freq']}:{s['guess_type']}"
                )

    if n_missing:
        out.append("\n── 缺失列 (训练 df 中不存在, 需查生成源) ──")
        for s in summary:
            if s["status"] == "missing":
                out.append(f"  ? {s['feat']}")

    text = "\n".join(out)
    print(text, flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    with open(
        os.path.join("data", f"_classify_freq_analog_summary_{ts}.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            {
                "ts": ts,
                "rows": len(work),
                "n_match": n_match,
                "n_mismatch": n_mismatch,
                "n_missing": n_missing,
                "features": summary,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    p = os.path.join("data", f"_classify_freq_analog_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}", flush=True)
    print(f"摘要: data/_classify_freq_analog_summary_{ts}.json", flush=True)


if __name__ == "__main__":
    main()
