# -*- coding: utf-8 -*-
"""_verify_chip_ma_interaction.py — 筹码结构变化 × 价格均线 组合信号验证.

用户纠正 (2026-08-04): 「筹码结构变化是要与价格均线一起做组合来预测的，
不是单独预测」。本脚本用 per-stock TSIC 验证组合假设:
1. 单独: chip_pct20 (参考 — 已知单独≈0)
2. 交互: chip_pct20 × sign(bias_20), chip_pct20 × bias_20
3. 边际: chip_pct20 对 bias_20 残差 (控制价格均线后筹码是否还有增量)
4. 条件: bias_20>0 (站上均线) vs bias_20<0 两个子窗口内 chip_pct20 的 TSIC

HS300×1年 + 全市场×3年. 输出落盘 data/_verify_chip_ma_interaction_<ts>.log (WORM).

用法: python scripts/_verify_chip_ma_interaction.py
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
from scripts._diag_column_feed import LABELS, MASK_RECENT_DAYS, weighted_ic
from scripts._diag_chip_weekly import _apply, per_stock_ts_ic
from scripts._verify_chip_tsic import _residualize

logging.disable(logging.CRITICAL)

CHIP_COLS = ["pct_90_con", "chip_entropy", "chip_gini"]
MA_COLS = ["bias_20", "bias_60"]
MIN_OBS = 20


def _load(hs300_only: bool) -> pd.DataFrame:
    read_cols = list(
        dict.fromkeys(
            ["date", "symbol", "is_suspended", "close_hfq"] + CHIP_COLS + MA_COLS
        )
    )
    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)
    if hs300_only:
        import akshare as ak

        cons = ak.index_stock_cons(symbol="000300")
        col = "品种代码" if "品种代码" in cons.columns else "成分券代码"
        syms = [str(s).strip().split(".")[0] for s in cons[col]]
        df = df[df["symbol"].isin(syms)].reset_index(drop=True)
    df = LabelEngine.build_labels(df, session="PM")
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    return df


def _f(v):
    return f"{v:+.4f}" if v == v else "   nan"


def _wtsic(tsic, name):
    per = {lab: tsic[lab][name] for lab in LABELS}
    return weighted_ic(per)


def _feat_table(work, feats, out):
    tsic = per_stock_ts_ic(work, feats, LABELS, min_obs=MIN_OBS)
    out.append(
        f"{'feature':<26}{'wTSIC':>9}{'TS2d':>9}{'TS3d':>9}{'TS5d':>9}"
    )
    out.append("-" * 62)
    for name in feats:
        per = {lab: tsic[lab][name] for lab in LABELS}
        out.append(
            f"{name:<26}{_f(weighted_ic(per)):>9}"
            f"{_f(per['label_pm_2d_net']):>9}{_f(per['label_pm_3d_net']):>9}"
            f"{_f(per['label_pm_5d_net']):>9}"
        )
    out.append("")


def _report(tr, label, out):
    out.append(f"--- {label} | rows={len(tr):,} stocks={tr['symbol'].nunique()} ---")
    work = tr.dropna(subset=LABELS).copy()
    for c in CHIP_COLS:
        work[f"{c}_p20"] = work.groupby("symbol")[c].transform(
            lambda x: _apply("pct_change", 20, x)
        )
        work[f"{c}_p20_x_sgn"] = work[f"{c}_p20"] * np.sign(work["bias_20"])
        work[f"{c}_p20_x_bias"] = work[f"{c}_p20"] * work["bias_20"]
    for c in CHIP_COLS:
        _residualize(work, f"{c}_p20", "bias_20", f"{c}_p20_r20", min_obs=MIN_OBS)

    feats = {}
    for m in MA_COLS:
        feats[m] = work[m]
    for c in CHIP_COLS:
        for k in (f"{c}_p20", f"{c}_p20_x_sgn", f"{c}_p20_x_bias", f"{c}_p20_r20"):
            feats[k] = work[k]
    _feat_table(work, feats, out)

    # 条件: 站上/跌破 bias_20 子窗口内 chip_p20 的 TSIC
    out.append("  条件子窗口 (bias_20 站上 vs 跌破) 内 chip_p20 TSIC:")
    out.append(
        f"{'feature':<20}{'bias20>0 wTSIC':>15}{'bias20<0 wTSIC':>15}"
    )
    for c in CHIP_COLS:
        up = work[work["bias_20"] > 0]
        dn = work[work["bias_20"] <= 0]
        fu = {f"{c}_p20": up[f"{c}_p20"]}
        fd = {f"{c}_p20": dn[f"{c}_p20"]}
        tu = per_stock_ts_ic(up, fu, LABELS, min_obs=MIN_OBS)
        td = per_stock_ts_ic(dn, fd, LABELS, min_obs=MIN_OBS)
        out.append(
            f"{c:<20}{_f(_wtsic(tu, f'{c}_p20')):>15}{_f(_wtsic(td, f'{c}_p20')):>15}"
        )
    out.append("")
    del work, feats
    gc.collect()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    out = []
    hs = _load(hs300_only=True)
    latest = hs["date"].max()
    cutoff = latest - pd.DateOffset(years=1)
    _report(hs[hs["date"] >= cutoff].reset_index(drop=True), "HS300×1年", out)
    del hs
    gc.collect()

    full = _load(hs300_only=False)
    cutoff3 = latest - pd.DateOffset(years=3)
    _report(full[full["date"] >= cutoff3].reset_index(drop=True), "全市场×3年", out)
    del full
    gc.collect()

    text = "\n".join(out)
    print(text)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_verify_chip_ma_interaction_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}")


if __name__ == "__main__":
    main()
