# -*- coding: utf-8 -*-
"""_diag_chip_weekly.py ?筹码形态列?双周/月频变换 **个股 time-series IC** 测试.

用户修正: 筹码形态是**个股自身的时间序列状态变?*, 跨股票横截面比较混入
个股结构性差? 应采?**per-stock time-series IC**: 对每只股? 将该股自?
历史上筹码特征序列与未来收益序列?rank 相关, 再对全样本股票取平均.

对每?chip ?× 各变?windows 5/10/20)计算个股时序 IC, ?HS300×1年窗?
输出落盘 data/_diag_chip_weekly_<ts>.log (WORM).

用法: python scripts/_diag_chip_weekly.py [--hs300|--full]
"""

import argparse
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

logging.disable(logging.CRITICAL)

CHIP_COLS = [
    "chip_entropy",
    "chip_gini",
    "chip_skew_dist",
    "pct_90_con",
    "pct_90_high",
    "cost_95pct",
    "peak_price",
    "peak_roc_5d",
    "peak_roc_20d",
    "winner_ratio",
]

# 对照: 纯价格水平列 ?检验成本线?成本线列本身是价??level TSIC
# 是否只是个股价格均值回? 而非筹码结构信息.
CONTROL_COLS = ["close_hfq", "low_hfq"]

# (name, transform spec) ?周频=5, 双周=10, 月频=20
TRANSFORMS = [
    ("level", None),
    ("ma5", ("rolling_mean", 5)),
    ("ma10", ("rolling_mean", 10)),
    ("ma20", ("rolling_mean", 20)),
    ("pct5", ("pct_change", 5)),
    ("pct10", ("pct_change", 10)),
    ("pct20", ("pct_change", 20)),
    ("mom5", ("momentum", 5)),
    ("mom10", ("momentum", 10)),
    ("mom20", ("momentum", 20)),
    ("diff5", ("diff", 5)),
    ("diff20", ("diff", 20)),
    ("std5", ("rolling_std", 5)),
    ("max5", ("rolling_max", 5)),
]


def _apply(family, win, s):
    if family == "rolling_mean":
        return s.rolling(win).mean()
    if family == "pct_change":
        return s.pct_change(win)
    if family == "momentum":
        return s / s.shift(win) - 1.0
    if family == "diff":
        return s.diff(win)
    if family == "rolling_std":
        return s.rolling(win).std()
    if family == "rolling_max":
        return s.rolling(win).max()
    raise ValueError(family)


def per_stock_ts_ic(work, feats, labels, min_obs=20):
    """个股时序 IC: 对每只股? 用该?*自身**历史序列, 计算特征?T+2/T+3/T+5
    涨幅(label_pm_{2,3,5}d_net)?rank 相关(Spearman), 再对全样本股票取平均.
    feats: {name: pd.Series index 对齐 work}. 返回 {label: {name: mean_tsic}}."""
    res = {lab: {t: [] for t in feats} for lab in labels}
    for sym, g in work.groupby("symbol"):
        for tname, fvals in feats.items():
            f = g[tname]
            if f.notna().sum() < min_obs:
                continue
            for lab in labels:
                label = g[lab]
                m = f.notna() & label.notna()
                if m.sum() < min_obs:
                    continue
                fr = f[m].astype(float).rank()
                lr = label[m].astype(float).rank()
                c = fr.corr(lr)
                if c == c:
                    res[lab][tname].append(c)
    return {
        lab: {t: (float(np.mean(v)) if v else np.nan) for t, v in res[lab].items()}
        for lab in labels
    }


def _emit(df, col, out):
    feats = {}
    for tname, spec in TRANSFORMS:
        if spec is None:
            feats[tname] = df[col].astype(float)
        else:
            feats[tname] = df.groupby("symbol")[col].transform(
                lambda x, fam=spec[0], w=spec[1]: _apply(fam, w, x)
            )
    fdf = pd.DataFrame(feats, index=df.index)
    work = df.dropna(subset=LABELS).copy()
    work = pd.concat([work, fdf], axis=1)
    tsic = per_stock_ts_ic(work, feats, LABELS)
    for tname, _ in TRANSFORMS:
        per = {lab: tsic[lab][tname] for lab in LABELS}
        wic = weighted_ic(per)
        def f(v):
            return f"{v:+.4f}" if v == v else "   nan"
        out.append(
            f"{col:<20}{tname:<9}{f(wic):>9}"
            f"{f(per['label_pm_2d_net']):>9}{f(per['label_pm_3d_net']):>9}"
            f"{f(per['label_pm_5d_net']):>9}"
        )


def run(df, label, out):
    out.append(f"=== {label} | rows={len(df):,} stocks={df['symbol'].nunique()} ===")
    out.append(
        f"{'col':<20}{'transform':<9}{'wTSIC':>9}{'TS2d':>9}{'TS3d':>9}{'TS5d':>9}"
    )
    out.append("-" * 66)
    for col in CHIP_COLS:
        if col not in df.columns:
            continue
        _emit(df, col, out)
    out.append("--- 对照: 纯价格水平列 (level TSIC 是否=价格均值回? ---")
    for col in CONTROL_COLS:
        if col not in df.columns:
            continue
        _emit(df, col, out)
    out.append("")
    del df
    gc.collect()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["hs300", "full"], default="hs300")
    args = ap.parse_args()

    from scripts._diag_column_feed import _schema_cols

    schema = _schema_cols(PANEL_V3_PATH)
    read_cols = list(
        dict.fromkeys(
            ["date", "symbol", "is_suspended", "close_hfq"]
            + [c for c in CHIP_COLS if c in schema]
            + [c for c in CONTROL_COLS if c in schema]
        )
    )
    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)

    if args.universe == "hs300":
        import akshare as ak

        cons = ak.index_stock_cons(symbol="000300")
        col = "品种代码" if "品种代码" in cons.columns else "成分券代?"
        syms = [str(s).strip().split(".")[0] for s in cons[col]]
        df = df[df["symbol"].isin(syms)].reset_index(drop=True)

    df = LabelEngine.build_labels(df, session="PM")
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    latest = df["date"].max()
    cutoff = latest - pd.DateOffset(years=1 if args.universe == "hs300" else 3)
    tr = df[df["date"] >= cutoff].reset_index(drop=True)
    del df
    gc.collect()

    out = []
    run(tr, f"{args.universe} {cutoff.date()}..{latest.date()}", out)
    text = "\n".join(out)
    print(text)

    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_diag_chip_weekly_{ts}.log")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n落盘: {p}")


if __name__ == "__main__":
    main()
