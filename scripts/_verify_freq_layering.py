# -*- coding: utf-8 -*-
"""_verify_freq_layering.py — 三频模型: 特征频率归属判定 (全市场×3年).

用户 2026-08-04 方法论升级:
① 按功能分组特征, 判定每组特征在 日/周/月 哪个频率上对预测有意义 → 定义频率;
② 同频率特征做交互组合实验;
最终 = 月频模型 + 周频模型 + 日频模型融合, 月度特征不进日频模型.

本脚本验证第①步: 对每个功能组, 用 1/5/20 日变化窗口构造特征(日/周/月代理),
per-stock TSIC 看信号在哪一档最强 → 输出频率归属判定表。
另加同频 vs 跨频组合对照 (月度筹码 × 月度MA状态 vs × 日频MA状态)。

用法: python scripts/_verify_freq_layering.py
"""
import gc
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.label_engine import LabelEngine
from scripts._diag_column_feed import (
    LABELS,
    MASK_RECENT_DAYS,
    daily_rank_ic_multi,
    weighted_ic,
)
from scripts._diag_chip_weekly import _apply, per_stock_ts_ic

logging.disable(logging.CRITICAL)
np.seterr(all="ignore")

WINDOWS = {"日(1)": 1, "周(5)": 5, "月(20)": 20}
MIN_OBS = 60

CHIP_COLS = ["pct_90_con", "chip_entropy", "chip_gini"]
COST_COLS = ["cost_95pct", "cost_50pct"]


def _load() -> pd.DataFrame:
    read_cols = list(
        dict.fromkeys(
            ["date", "symbol", "is_suspended", "close_hfq", "volume", "amount"]
            + CHIP_COLS + COST_COLS
        )
    )
    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)
    df = LabelEngine.build_labels(df, session="PM")
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    return df


def _f(v):
    return f"{v:+.4f}" if v == v else "   nan"


def _build(work):
    g = work.groupby("symbol")
    for c in CHIP_COLS + COST_COLS:
        for w in WINDOWS.values():
            work[f"{c}_p{w}"] = g[c].transform(lambda x, ww=w: _apply("pct_change", ww, x))
    for w in WINDOWS.values():
        work[f"close_hfq_p{w}"] = g["close_hfq"].transform(
            lambda x, ww=w: _apply("pct_change", ww, x)
        )
        work[f"vol_p{w}"] = g["volume"].transform(
            lambda x, ww=w: _apply("pct_change", ww, x)
        )
        work[f"pv{w}"] = work[f"close_hfq_p{w}"] * np.sign(work[f"vol_p{w}"])
    work["bias20"] = work["close_hfq"] / g["close_hfq"].transform(
        lambda x: x.rolling(20).mean()
    ) - 1.0
    work["bias5"] = work["close_hfq"] / g["close_hfq"].transform(
        lambda x: x.rolling(5).mean()
    ) - 1.0


def _wtsic(tsic, k):
    per = {lab: tsic[lab][k] for lab in LABELS}
    return weighted_ic(per)


def _report(tr, out):
    out.append(f"--- 全市场×3年 | rows={len(tr):,} stocks={tr['symbol'].nunique()} ---")
    work = tr.dropna(subset=LABELS).copy()
    _build(work)

    # --- 6格判定: 口径(TS时序/XS截面) × 频率(日/周/月) ---
    def freq6(title, base_cols, col_fn):
        out.append(f"  [{title}] 6格判定 (TS=个股时序 IC / XS=日截面 rank IC):")
        out.append(
            f"{'feature':<18}{'TS日':>8}{'TS周':>8}{'TS月':>8}"
            f"{'XS日':>8}{'XS周':>8}{'XS月':>8}  ← 最强格"
        )
        out.append("-" * 82)
        for c in base_cols:
            ks = {w: col_fn(c, w) for w in (1, 5, 20)}
            feats = {ks[w]: work[ks[w]] for w in (1, 5, 20)}
            tsic = per_stock_ts_ic(work, feats, LABELS, min_obs=MIN_OBS)
            xic = daily_rank_ic_multi(work, list(feats), LABELS)
            ts = {w: _wtsic(tsic, ks[w]) for w in (1, 5, 20)}
            xs = {
                w: weighted_ic({lab: xic[lab][ks[w]] for lab in LABELS})
                for w in (1, 5, 20)
            }
            cells = {
                "TS日": ts[1], "TS周": ts[5], "TS月": ts[20],
                "XS日": xs[1], "XS周": xs[5], "XS月": xs[20],
            }
            best = max(cells, key=lambda k: abs(cells[k]))
            out.append(
                f"{c:<18}{_f(cells['TS日']):>8}{_f(cells['TS周']):>8}{_f(cells['TS月']):>8}"
                f"{_f(cells['XS日']):>8}{_f(cells['XS周']):>8}{_f(cells['XS月']):>8}"
                f"  ← {best} ({abs(cells[best]):.4f})"
            )
        out.append("")

    def win_fn(c, w):
        return f"{c}_p{w}"

    freq6("筹码组", CHIP_COLS, win_fn)
    freq6("COST成本线组", COST_COLS, win_fn)
    freq6("价格组", ["close_hfq"], win_fn)

    def pv_fn(c, w):
        return f"pv{w}"

    freq6("价量组(price×vol)", ["pv"], pv_fn)

    # 同频 vs 跨频组合对照 (月度筹码 × MA状态)
    out.append("  [同频/跨频组合对照] 月度筹码 × MA 状态:")
    out.append(f"{'feature':<30}{'wTSIC':>9}")
    work["p20_x_sgnb20"] = work["pct_90_con_p20"] * np.sign(work["bias20"])
    work["p20_x_sgnb5"] = work["pct_90_con_p20"] * np.sign(work["bias5"])
    combos = [
        ("p20 单独(月)", "pct_90_con_p20"),
        ("p20×sgn(bias20) 月×月", "p20_x_sgnb20"),
        ("p20×sgn(bias5) 月×日", "p20_x_sgnb5"),
    ]
    feats = {col: work[col] for _, col in combos}
    tsic = per_stock_ts_ic(work, feats, LABELS, min_obs=MIN_OBS)
    for label, col in combos:
        out.append(f"{label:<30}{_f(_wtsic(tsic, col)):>9}")
    out.append("")

    del work
    gc.collect()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    out = []
    df = _load()
    latest = df["date"].max()
    cutoff = latest - pd.DateOffset(years=3)
    _report(df[df["date"] >= cutoff].reset_index(drop=True), out)
    del df
    gc.collect()

    text = "\n".join(out)
    print(text)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_verify_freq_layering_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}")


if __name__ == "__main__":
    main()
