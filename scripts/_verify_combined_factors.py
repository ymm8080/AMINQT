# -*- coding: utf-8 -*-
"""_verify_combined_factors.py — 用户方法论全量实测 (per-stock TSIC).

用户 2026-08-04 方法论 (全部实验验证):
A. 筹码结构变化 × 价格均线 (月频筹码配月均线) 组合预测, 不是单独预测
   - chip_p20 单独 (参考) / ×sign(bias20) / ×bias20 / 对 bias20 残差 / 站上·跌破均线子窗口
B. 价格: 5/10/20 日均线**变化**的组合作为预测因子
   - bias5/bias10/bias20 偏离, MA 间 gap (g5_10/g10_20), MA 自身动量 (chg5/10/20),
     MA 组合动量 (chg5+chg10+chg20), 多空排列一致 (align)
C. 价格 × 量: 一阶/二阶/三阶组合作为训练预测因子
   - 1阶: pct1(收益), vol1(量变), amt1(额变)
   - 2阶: pct2=d(pct1,5), vol2=d(vol1,5)
   - 3阶: pct3=d(pct1,10), vol3=d(vol1,10)
   - 组合: pv1=pct1×sign(vol1), pv2=pct2×sign(vol2), pv3=pct3×sign(vol3)
D. 筹码月度 × COST月度成本线 组合 (用户 2026-08-04 追加)
   - cost95_p20 / cost50_p20 单独 (月频成本线变化)
   - chip_p20 × cost95_p20, chip_p20 × (close/cost95-1), chip_p20 × sign(cost95_p20)

HS300×1年 + 全市场×3年. 输出落盘 data/_verify_combined_factors_<ts>.log (WORM).

用法: python scripts/_verify_combined_factors.py
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
COST_COLS = ["cost_95pct", "cost_50pct"]
MIN_OBS = 20


def _load(hs300_only: bool) -> pd.DataFrame:
    read_cols = list(
        dict.fromkeys(
            ["date", "symbol", "is_suspended", "close_hfq", "volume", "amount"]
            + CHIP_COLS
            + COST_COLS
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


def _feat_table(work, cols, title, out):
    out.append(f"  [{title}]")
    feats = {c: work[c] for c in cols}
    tsic = per_stock_ts_ic(work, feats, LABELS, min_obs=MIN_OBS)
    out.append(f"{'feature':<30}{'wTSIC':>9}{'TS2d':>9}{'TS3d':>9}{'TS5d':>9}")
    out.append("-" * 66)
    for name in cols:
        per = {lab: tsic[lab][name] for lab in LABELS}
        out.append(
            f"{name:<30}{_f(weighted_ic(per)):>9}"
            f"{_f(per['label_pm_2d_net']):>9}{_f(per['label_pm_3d_net']):>9}"
            f"{_f(per['label_pm_5d_net']):>9}"
        )
    out.append("")
    return tsic


def _build(work):
    g = work.groupby("symbol")
    # --- A. 筹码月频变化 + 月均线组合 ---
    for c in CHIP_COLS:
        work[f"{c}_p20"] = g[c].transform(lambda x: _apply("pct_change", 20, x))
    work["bias20"] = (
        work["close_hfq"] / g["close_hfq"].transform(lambda x: x.rolling(20).mean())
        - 1.0
    )
    for c in CHIP_COLS:
        work[f"{c}_p20_x_sgn"] = work[f"{c}_p20"] * np.sign(work["bias20"])
        work[f"{c}_p20_x_bias"] = work[f"{c}_p20"] * work["bias20"]
    for c in CHIP_COLS:
        _residualize(work, f"{c}_p20", "bias20", f"{c}_r20", min_obs=MIN_OBS)

    # --- B. 5/10/20 日均线变化的组合 ---
    for w in (5, 10, 20):
        work[f"ma{w}"] = g["close_hfq"].transform(lambda x: x.rolling(w).mean())
        work[f"bias{w}"] = work["close_hfq"] / work[f"ma{w}"] - 1.0
    work["g5_10"] = work["ma5"] / work["ma10"] - 1.0
    work["g10_20"] = work["ma10"] / work["ma20"] - 1.0
    for w, n in ((5, 5), (10, 10), (20, 20)):
        work[f"ma_chg{w}"] = g[f"ma{w}"].transform(lambda x: _apply("pct_change", n, x))
    work["ma_combo"] = work["ma_chg5"] + work["ma_chg10"] + work["ma_chg20"]
    work["align"] = np.sign(work["g5_10"]) * np.sign(work["g10_20"])

    # --- D. 筹码月度 × COST月度成本线 组合 ---
    for cc in COST_COLS:
        work[f"{cc}_p20"] = g[cc].transform(lambda x: _apply("pct_change", 20, x))
        work[f"close_div_{cc}"] = work["close_hfq"] / work[cc] - 1.0
    for c in CHIP_COLS:
        for cc in COST_COLS:
            work[f"{c}__x__{cc}_p20"] = work[f"{c}_p20"] * work[f"{cc}_p20"]
            work[f"{c}__x__sgn{cc}"] = work[f"{c}_p20"] * np.sign(work[f"{cc}_p20"])
            work[f"{c}__x__div{cc}"] = work[f"{c}_p20"] * work[f"close_div_{cc}"]

    # --- C. 价格 × 量 一/二/三阶组合 ---
    work["pct1"] = g["close_hfq"].transform(lambda x: _apply("pct_change", 1, x))
    work["vol1"] = g["volume"].transform(lambda x: _apply("pct_change", 1, x))
    work["amt1"] = g["amount"].transform(lambda x: _apply("pct_change", 1, x))
    work["pct2"] = work["pct1"] - work["pct1"].shift(5)
    work["vol2"] = work["vol1"] - work["vol1"].shift(5)
    work["pct3"] = work["pct1"] - work["pct1"].shift(10)
    work["vol3"] = work["vol1"] - work["vol1"].shift(10)
    work["pv1"] = work["pct1"] * np.sign(work["vol1"])
    work["pv2"] = work["pct2"] * np.sign(work["vol2"])
    work["pv3"] = work["pct3"] * np.sign(work["vol3"])


def _feat_sets(work):
    a = ["bias20"]
    for c in CHIP_COLS:
        a += [f"{c}_p20", f"{c}_p20_x_sgn", f"{c}_p20_x_bias", f"{c}_r20"]

    b = []
    for w in (5, 10, 20):
        b += [f"bias{w}", f"ma_chg{w}"]
    b += ["g5_10", "g10_20", "ma_combo", "align"]

    c_ = ["pct1", "vol1", "amt1", "pct2", "vol2", "pct3", "vol3", "pv1", "pv2", "pv3"]

    d_ = []
    for cc in COST_COLS:
        d_ += [f"{cc}_p20", f"close_div_{cc}"]
    for c in CHIP_COLS:
        for cc in COST_COLS:
            d_ += [f"{c}__x__{cc}_p20", f"{c}__x__sgn{cc}", f"{c}__x__div{cc}"]

    return a, b, c_, d_


def _report(tr, label, out):
    out.append(f"--- {label} | rows={len(tr):,} stocks={tr['symbol'].nunique()} ---")
    work = tr.dropna(subset=LABELS).copy()
    _build(work)
    a, b, c_, d_ = _feat_sets(work)
    _feat_table(work, a, "A. 筹码×月均线组合", out)
    _feat_table(work, b, "B. 5/10/20日MA变化组合", out)
    _feat_table(work, c_, "C. 价格×量 1/2/3阶组合", out)
    _feat_table(work, d_, "D. 筹码×COST月度成本线组合", out)
    # A 的条件子窗口
    out.append(
        "  [A 条件子窗口] chip_p20 在 站上(bias20>0) vs 跌破(bias20<=0) 的 TSIC:"
    )
    out.append(f"{'feature':<20}{'bias20>0':>10}{'bias20<=0':>10}")
    for c in CHIP_COLS:
        for tag, sub in (
            ("up", work[work["bias20"] > 0]),
            ("dn", work[work["bias20"] <= 0]),
        ):
            fe = {f"{c}_p20": sub[f"{c}_p20"]}
            t = per_stock_ts_ic(sub, fe, LABELS, min_obs=MIN_OBS)
            w = weighted_ic({lab: t[lab][f"{c}_p20"] for lab in LABELS})
            if tag == "up":
                row = [f"{c}_p20", _f(w)]
            else:
                row.append(_f(w))
                out.append(f"{row[0]:<20}{row[1]:>10}{row[2]:>10}")
    out.append("")
    del work
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

    text = "\n".join(out)
    print(text)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_verify_combined_factors_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}")


if __name__ == "__main__":
    main()
