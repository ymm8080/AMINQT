# -*- coding: utf-8 -*-
"""_verify_chip_tsic.py — 筹码 per-stock TSIC 结论稳健性验证.

验证「筹码=个股 time-series IC, 月频最强」是否成立:
1. 跨窗口一致性: HS300×1年 全窗 / 上半年 / 下半年 / 全市场×3年 —
   同一筹码列 pct20 的 wTSIC 符号与量级应一致.
2. 与价格正交性: 个股内把 chip_pct20 对 close_hfq_pct20 回归取残差,
   残差 wTSIC 若仍在 → 筹码信号独立于价格动量(真实筹码信息);
   若消失 → 只是价格动量代理.
3. 对照: close_hfq_pct20 自身的 wTSIC(价格动量/反转参考系).

输出落盘 data/_verify_chip_tsic_<ts>.log (WORM).

用法: python scripts/_verify_chip_tsic.py
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

logging.disable(logging.CRITICAL)

CHIP_COLS = ["chip_entropy", "chip_gini", "chip_skew_dist", "pct_90_con"]
PRICE = "close_hfq"
MIN_OBS = 20


def _load(hs300_only: bool) -> pd.DataFrame:
    read_cols = list(
        dict.fromkeys(["date", "symbol", "is_suspended", "close_hfq"] + CHIP_COLS)
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


def _build_feats(df, cols):
    feats = {}
    for c in cols:
        feats[f"{c}_pct20"] = df.groupby("symbol")[c].transform(
            lambda x: _apply("pct_change", 20, x)
        )
    return feats


def _residualize(work, target, on, out_name, min_obs=MIN_OBS):
    """个股内 OLS: target ~ on + const, 残差为 target 正交于 on 的部分."""
    work[out_name] = np.nan
    for sym, g in work.groupby("symbol"):
        x = g[on].astype(float)
        y = g[target].astype(float)
        m = x.notna() & y.notna()
        if m.sum() < min_obs:
            continue
        X = np.column_stack([x[m].values, np.ones(int(m.sum()))])
        beta, *_ = np.linalg.lstsq(X, y[m].values, rcond=None)
        work.loc[g.index[m], out_name] = y[m].values - X @ beta


def _wtsic(tsic, name):
    per = {lab: tsic[lab][name] for lab in LABELS}
    return weighted_ic(per)


def _report(df, label, out):
    out.append(f"--- {label} | rows={len(df):,} stocks={df['symbol'].nunique()} ---")
    feats = _build_feats(df, [PRICE] + CHIP_COLS)
    work = df.dropna(subset=LABELS).copy()
    work = pd.concat([work, pd.DataFrame(feats, index=df.index)], axis=1)
    # 正交性: chip_pct20 残差于 close_hfq_pct20
    for c in CHIP_COLS:
        _residualize(work, f"{c}_pct20", f"{PRICE}_pct20", f"{c}_resid")
    feat_dict = dict(feats)
    for c in CHIP_COLS:
        feat_dict[f"{c}_resid"] = work[f"{c}_resid"]
    tsic = per_stock_ts_ic(work, feat_dict, LABELS)
    out.append(
        f"{'col':<22}{'wTSIC(pct20)':>14}{'wTSIC(正交价格)':>18}"
        f"{'TS2d(resid)':>13}{'TS5d(resid)':>13}"
    )
    out.append("-" * 84)
    for name in feats:
        w = _wtsic(tsic, name)
        f = lambda v: f"{v:+.4f}" if v == v else "   nan"
        out.append(f"{name:<22}{f(w):>14}{'':>18}{'':>13}{'':>13}")
    for c in CHIP_COLS:
        w = _wtsic(tsic, f"{c}_resid")
        per = {lab: tsic[lab][f"{c}_resid"] for lab in LABELS}
        f = lambda v: f"{v:+.4f}" if v == v else "   nan"
        out.append(
            f"{c+'_resid':<22}{'':>14}{f(w):>18}"
            f"{f(per['label_pm_2d_net']):>13}{f(per['label_pm_5d_net']):>13}"
        )
    out.append("")
    del work, tsic
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
    tr = hs[hs["date"] >= cutoff].reset_index(drop=True)
    mid = tr["date"].median()
    _report(tr, "HS300×1年 全窗", out)
    _report(
        tr[tr["date"] <= mid].reset_index(drop=True), "HS300×1年 上半年", out
    )
    _report(
        tr[tr["date"] > mid].reset_index(drop=True), "HS300×1年 下半年", out
    )
    del hs, tr
    gc.collect()

    full = _load(hs300_only=False)
    cutoff3 = latest - pd.DateOffset(years=3)
    f3 = full[full["date"] >= cutoff3].reset_index(drop=True)
    _report(f3, "全市场×3年", out)
    del full, f3
    gc.collect()

    text = "\n".join(out)
    print(text)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_verify_chip_tsic_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}")


if __name__ == "__main__":
    main()
