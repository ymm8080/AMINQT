# -*- coding: utf-8 -*-
"""_reclassify_all_features.py — 统一口径重审全部特征 (2026-08-04).

用户裁决口径 (2026-08-04 定稿): 只关心 TOP-10 — 每日期截面按特征排名取 TOP-10,
量这 10 只的「绝对上涨幅度(平均净收益) + 上涨概率(胜率)」; 后面的都不关心.
验收规则: 任一视界 胜率>=55% 且 平均净收益>0 → 保留.
**单端 (2026-08-04 用户修正): 只测高值端 (特征值降序取 TOP-10), 不做双向.**
路由 (--route, 默认关): 仅对验收通过的特征做 6格 rankIC (TS/XS × 日/周/月)
  → 归 月/周/日/事件 模型 (rankIC 只管路由, 不参与验收 — LHB 教训).

行集 = 生产行集 (与 prepare_board_frame 完全一致):
  run_train(每板块) → features.build → build_path_labels → build_labels(B9晚盘净标签)
  → mask_suspension → mask_recent_days → 3y 窗口.
主板切片复用 _diag_analog_stage 检查点 data/_diag_stage_main_3y.parquet (若存在, 省重建).

输出 (WORM): data/_reclassify_all_<ts>.json + <ts>.log
"""

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import COST, LabelEngine, slippage_tier
from scripts._diag_column_feed import MASK_RECENT_DAYS
from scripts._measure_topn import HORIZONS, measure_topn
from scripts._classify_freq_analog import family_of
from scripts._classify_freq_full import (
    MIN_CROSS,
    MIN_OBS,
    WINDOWS,
    _wtsic,
    group_spearman,
)

MAIN_CHECKPOINT = os.path.join("data", "_diag_stage_main_3y.parquet")
DUAL_CHECKPOINT = os.path.join("data", "_diag_stage_dual_3y.parquet")

# 非特征列排除
META_EXCLUDE = {"symbol", "date", "is_suspended", "board", "name", "code", "exec_px"}


def add_label_pm_10d_net(df: pd.DataFrame) -> pd.DataFrame:
    """生产口径补算 label_pm_10d(_net) (2026-08-04 验收视界扩至 5d/10d).

    与 label_engine.build_labels B9 + add_net_labels 完全一致 (检查点无 price_1455
    → 日K近似执行价 exec=close_hfq[T+1]):
      label_pm_10d     = close_hfq[T+11]/close_hfq[T+1] - 1
      label_pm_10d_net = label_pm_10d - (COST + 2×分层滑点)
    再按生产 mask_suspension 逻辑遮蔽 [T,T+10] 含停牌的行.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol")["close_hfq"]
    exec_px = g.shift(-1)
    fut = g.shift(-11)
    df["label_pm_10d"] = fut / exec_px - 1
    slip = df["adv20"].map(slippage_tier) if "adv20" in df.columns else 0.0015
    df["label_pm_10d_net"] = df["label_pm_10d"] - (COST + 2 * slip)
    n = 10
    rolling_sum = (
        df.groupby("symbol")["is_suspended"]
        .rolling(n + 1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    vals = rolling_sum.values
    suspended_vals = np.zeros(len(vals), dtype=bool)
    if len(vals) > n:
        suspended_vals[: len(vals) - n] = vals[n:] > 0
    suspended = pd.Series(suspended_vals, index=rolling_sum.index)
    df["label_pm_10d"] = df["label_pm_10d"].where(~suspended, np.nan)
    df["label_pm_10d_net"] = df["label_pm_10d_net"].where(~suspended, np.nan)
    return df


def _finalize_slice(d3: pd.DataFrame) -> pd.DataFrame:
    """补 10d 净标签 + 重跑 mask_recent_days (对既有 2/3/5 幂等)."""
    d3 = add_label_pm_10d_net(d3)
    d3 = LabelEngine.mask_recent_days(d3, days=MASK_RECENT_DAYS)
    return d3


def build_board_slice(cleaner, fe, board_df, board, checkpoint) -> pd.DataFrame:
    """单板块生产切片 (features.build + labels + 掩码 + 3y), 复用检查点省重建."""
    if checkpoint and os.path.exists(checkpoint):
        df = pd.read_parquet(checkpoint)
        print(f"[{board}] 复用检查点 {checkpoint} rows={len(df):,}", flush=True)
        return _finalize_slice(df)
    d = fe.build(board_df, None, cross_sectional_rank=(board != "main"), registry=None)
    del board_df
    gc.collect()
    d = LabelEngine.build_path_labels(d)
    d = LabelEngine.build_labels(d, session="PM")
    d = LabelEngine.mask_suspension(d)
    d = LabelEngine.mask_recent_days(d, days=MASK_RECENT_DAYS)
    latest = d["date"].max()
    cutoff = latest - pd.DateOffset(years=3)
    d3 = d[d["date"] >= cutoff].reset_index(drop=True)
    del d
    gc.collect()
    print(
        f"[{board}] 构建 rows={len(d3):,} stocks={d3['symbol'].nunique():,} "
        f"cols={d3.shape[1]:,} | latest={latest:%Y-%m-%d}",
        flush=True,
    )
    d3 = _finalize_slice(d3)
    if checkpoint:
        d3.to_parquet(checkpoint, index=False)
        print(
            f"[{board}] 检查点已落盘 {checkpoint} ({os.path.getsize(checkpoint) / 1e9:.2f} GB)",
            flush=True,
        )
    return d3


def feature_cols(work: pd.DataFrame) -> list:
    """全部可验收特征列: 数值、非元数据、非标签、有变异性."""
    cols = []
    for c in work.columns:
        if c in META_EXCLUDE or c.startswith("label_"):
            continue
        if not pd.api.types.is_numeric_dtype(work[c]):
            continue
        s = work[c].dropna()
        if s.nunique() <= 2:
            continue
        if len(s) < 1000:
            continue
        cols.append(c)
    return sorted(cols)


def accept_one(work: pd.DataFrame, col: str) -> dict:
    """单端 TOP-10 绝对验收 (2026-08-04 用户口径: 只测高值端, 不做双向).
    特征值降序 → 每日期截面取 TOP-10 → 测幅度(平均净收益)+胜率.
    任一视界 胜率>=55% 且 平均>0 即通过, 取综合分最高的视界为裁决视界."""
    r = measure_topn(work, col, top_n=10, per="date", ascending=False)
    best_h, best_score = None, -1.0
    if not (r.get("missing") or r.get("insufficient")):
        for k in HORIZONS:
            d_ = r.get(k, {})
            if d_.get("ok"):
                score = d_.get("mag", 0.0) + d_.get("winrate", 0.0)  # 幅度+概率综合
                if score > best_score:
                    best_h, best_score = k, score
    return {"res": r, "horizon": best_h}


def route_one(work, g_sym, g_date, lab_sym, lab_date, col):
    """6格 rankIC (路由用)."""
    g_grp = work.groupby("symbol")
    wins = {}
    for w in WINDOWS.values():
        wins[f"{col}_p{w}"] = (work[col] / g_grp[col].shift(w) - 1.0).astype("float64")
    tsic, xic = {}, {}
    for f, wc in wins.items():
        wr_sym = wc.groupby(g_sym.values).rank()
        wr_date = wc.groupby(g_date.values).rank()
        tsic[f] = {
            lab: group_spearman(wr_sym, lab_sym[lab], g_sym, MIN_OBS) for lab in lab_sym
        }
        xic[f] = {
            lab: group_spearman(wr_date, lab_date[lab], g_date, MIN_CROSS) for lab in lab_date
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", action="store_true", help="验收通过后做 6格 rankIC 路由")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    def prog(msg):
        print(msg, flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("data", f"_reclassify_all_{ts}.log")

    # ── 1. 生产行集 (两检查点齐全 → 快速路径, 跳过 panel/run_train/features.build) ──
    both_ckpt = all(os.path.exists(p) for p in (MAIN_CHECKPOINT, DUAL_CHECKPOINT))
    fe = FeatureEngineV35()
    cleaner = CleaningPipeline()
    if both_ckpt:
        prog("快速路径: 两检查点齐全, 跳过 panel/run_train/features.build")
        board_dfs = {}
    else:
        panel = pd.read_parquet(PANEL_V3_PATH)
        main_df, dual_df = cleaner.run_train(panel)
        del panel
        gc.collect()
        board_dfs = {"main": main_df, "dual": dual_df}

    slices = []
    for board, ckpt in (("main", MAIN_CHECKPOINT), ("dual", DUAL_CHECKPOINT)):
        bdf = board_dfs.get(board)
        if not both_ckpt and (bdf is None or len(bdf) == 0):
            print(f"[{board}] 空, 跳过", flush=True)
            continue
        slices.append(build_board_slice(cleaner, fe, bdf, board, ckpt))
        if bdf is not None:
            del bdf
            gc.collect()
    del board_dfs, fe, cleaner
    gc.collect()

    # ignore_index 排序 → 不额外深拷贝 (降峰值内存, 上次 OOM 在 reset_index 5.92GiB)
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    del slices
    gc.collect()
    cols = feature_cols(work)
    latest = work["date"].max()
    prog(
        f"生产行集 rows={len(work):,} stocks={work['symbol'].nunique():,} "
        f"cols={len(cols):,} 特征 | latest={latest:%Y-%m-%d}"
    )

    # ── 2. 验收 (单端: 高值端 TOP-10 绝对幅度+胜率) ──
    header = (
        "="
        * 100
        + "\n  TOP-10 绝对验收 (高值端 | 每日期截面 | 净收益标签 label_pm_*d_net | "
        f"胜率>=55% 且 平均>0 通过 | 视界 {list(HORIZONS)})" + "\n=" * 100
    )
    prog(header)
    summary = []
    n_pass = n_skip = 0
    for i, col in enumerate(cols):
        a = accept_one(work, col)
        res, best_h = a["res"], a["horizon"]
        if best_h is None:
            n_skip += 1
            prog(
                f"[{i + 1}/{len(cols)}] ✗ {col:<28} 不达标(高值端TOP10无胜率>=55%且幅度>0) | {family_of(col)}"
            )
            summary.append(
                {
                    "col": col,
                    "family": family_of(col),
                    "accepted": False,
                    "horizon": None,
                }
            )
            continue
        n_pass += 1
        d_ = res.get(best_h, {})
        prog(
            f"[{i + 1}/{len(cols)}] ✓ {col:<28} T+{best_h} "
            f"幅度={d_.get('mag', float('nan')):+.2%} 胜率={d_.get('winrate', float('nan')):>6.1%} "
            f"n={d_.get('n', 0):,} | {family_of(col)}"
        )
        summary.append(
            {
                "col": col,
                "family": family_of(col),
                "accepted": True,
                "horizon": best_h,
                "top10_high": {str(k): res.get(k) for k in HORIZONS},
            }
        )

    prog("-" * 100)
    prog(f"通过 {n_pass} / 未达标 {n_skip} / 合计 {len(cols)}")

    # ── 3. 路由 (可选): 仅验收通过特征做 6格 rankIC ──
    if args.route:
        passed = [s for s in summary if s["accepted"]]
        prog(f"\n路由: 对 {len(passed)} 个通过特征做 6格 rankIC ...")
        g_sym = work["symbol"]
        g_date = work["date"]
        lab_sym = {
            lab: work.groupby("symbol")[lab].rank()
            for lab in (f"label_pm_{k}d_net" for k in (2, 3, 5))
        }
        lab_date = {
            lab: work.groupby("date")[lab].rank()
            for lab in (f"label_pm_{k}d_net" for k in (2, 3, 5))
        }
        for i, s in enumerate(passed):
            col = s["col"]
            cells, best, ic = route_one(work, g_sym, g_date, lab_sym, lab_date, col)
            s["routing"] = {
                "cells": {k: round(v, 4) for k, v in cells.items()},
                "verdict": best,
                "ic": round(ic, 4),
            }
            prog(f"  [{i + 1}/{len(passed)}] {col:<28} {best} (|IC| {ic:.4f})")
        del g_sym, g_date, lab_sym, lab_date
        gc.collect()

    # ── 4. WORM 落盘 ──
    out = {
        "ts": ts,
        "window": {"end": str(latest), "years": 3},
        "rows": len(work),
        "stocks": int(work["symbol"].nunique()),
        "n_features": len(cols),
        "meta": {
            "top_n": 10,
            "per": "date",
            "end": "high_only",  # 单端: 只测高值端
            "min_winrate": 0.55,
            "min_mag": 0.0,
            "horizons": list(HORIZONS),
            "labels": "label_pm_*d_net",
        },
        "n_pass": n_pass,
        "n_skip": n_skip,
        "features": summary,
    }
    p = os.path.join("data", f"_reclassify_all_{ts}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    text = "\n".join(
        [header]
        + [
            f"[{s.get('accepted') and 'PASS' or '---'}] "
            f"{s['col']:<28} "
            f"T+{s.get('horizon') or '-'} {s.get('family')}"
            for s in summary
        ]
    )
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    prog(f"\n落盘: {p}\n落盘: {log_path}")


if __name__ == "__main__":
    main()
