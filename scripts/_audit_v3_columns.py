# -*- coding: utf-8 -*-
"""V3 面板逐列覆盖率 + 数据质量审计 (2023-01-01 ~ 今日).

对 panel_full_enriched_v3.parquet (data/ 下的已恢复版本) 每个列:
  1. 非空行数 / 覆盖率 / 覆盖 symbol 数 / 覆盖日期数 / 首末非空日期
  2. 分年覆盖率 (2023 / 2024 / 2025 / 2026YTD) → 识别新增或断更列
数据质量:
  - OHLCV 铁律校验 (high>=low 等), volume>=0, 负价格
  - 重复 (symbol,date), inf, 常数列, 全空列
  - board 分布 / is_st / is_suspended / pre_close 覆盖率
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audit_v3_columns")

PANEL_PATH = os.getenv(
    "PANEL_PATH", "data/panel_full_enriched_v3.parquet"
)
OUT_REPORT = os.getenv(
    "AUDIT_OUT", "data/v3_column_audit_{}.md".format(date.today().strftime("%Y%m%d"))
)
YEAR_START = 2023
TODAY = pd.Timestamp(date.today())

OHLCV = ["open", "high", "low", "close", "volume", "amount"]


def _cov_stats(df: pd.DataFrame, col: str) -> dict:
    nn = df[col].notna()
    n = len(df)
    if not nn.any():
        return {
            "n_nn": 0, "cov": 0.0, "n_sym": 0, "n_date": 0,
            "first": None, "last": None,
        }
    d = df.loc[nn, "date"]
    return {
        "n_nn": int(nn.sum()),
        "cov": float(nn.sum() / n * 100),
        "n_sym": int(df.loc[nn, "symbol"].nunique()),
        "n_date": int(d.nunique()),
        "first": d.min(),
        "last": d.max(),
    }


def _year_cov(df: pd.DataFrame) -> pd.DataFrame:
    """年份 x 列 的非空覆盖率矩阵 (%)."""
    d = df.assign(_year=df["date"].dt.year)
    g = d.groupby("_year")
    return g.apply(lambda s: s.notna().mean().mul(100).round(2))


def main() -> None:
    logger.info("载入面板: %s", PANEL_PATH)
    df = pd.read_parquet(PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(YEAR_START, 1, 1)].copy()
    df["symbol"] = df["symbol"].astype(str)

    rows = len(df)
    n_sym = df["symbol"].nunique()
    n_day = df["date"].nunique()
    dmin, dmax = df["date"].min(), df["date"].max()
    cols = list(df.columns)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    lines: list[str] = []
    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    # ── 0. 概览 ──
    out("=" * 100)
    out("V3 PANEL COLUMN COVERAGE & DATA QUALITY AUDIT")
    out(f"  面板: {PANEL_PATH}")
    out(f"  窗口: {YEAR_START}-01-01 ~ {TODAY.date()} (数据 {dmin.date()} ~ {dmax.date()})")
    out(f"  行数: {rows:,} | symbol: {n_sym:,} | 交易日: {n_day:,}")
    out(f"  列数: {len(cols)} (数值 {len(num_cols)})")
    out("=" * 100)

    # ── 1. 逐列覆盖率 ──
    out(f"\n## 1. 逐列覆盖率 ({len(cols)} 列, 按覆盖率升序)")
    hdr = (f"{'col':<30}{'cov%':>7}{'n_nn':>10}{'n_sym':>7}{'n_date':>7}"
           f"{'first':>11}{'last':>11}")
    out(hdr)
    out("-" * len(hdr))
    year_cov = _year_cov(df)
    years_present = [y for y in (2023, 2024, 2025, 2026) if y in year_cov.index]
    year_str = "  ".join(str(y) for y in years_present)
    out(f"{'':<30}{'':>7}{'':>10}{'':>7}{'':>7}  per-year cov%: {year_str}")

    # 先按覆盖率排序做主体表
    stats = {c: _cov_stats(df, c) for c in cols}
    for c in sorted(cols, key=lambda c: stats[c]["cov"]):
        s = stats[c]
        cov = f"{s['cov']:6.2f}"
        first = str(s["first"].date()) if s["first"] is not None else "-"
        last = str(s["last"].date()) if s["last"] is not None else "-"
        yc = "  ".join(
            f"{year_cov.loc[y, c]:6.1f}" if c in year_cov.columns else "     -"
            for y in years_present
        )
        out(f"{c:<30}{cov:>7}{s['n_nn']:>10}{s['n_sym']:>7}{s['n_date']:>7}"
            f"{first:>11}{last:>11}  {yc}")

    # ── 2. 时间断更检测: 首年无数据或末年无数据 ──
    out("\n## 2. 时间覆盖异常列")
    flag = False
    for c in cols:
        yc = year_cov[c] if c in year_cov.columns else pd.Series(dtype=float)
        # 首年/末年缺失但中间存在 → 断更或新列
        first_y, last_y = years_present[0], years_present[-1]
        has_first = c not in year_cov.columns or yc.get(first_y, 0) > 0.5
        has_last = c not in year_cov.columns or yc.get(last_y, 0) > 0.5
        mid = [y for y in years_present if yc.get(y, 0) > 50]
        if (not has_first or not has_last) and mid:
            out(f"  {c:<30} 首年cov={yc.get(first_y, 0):.1f}% "
                f"末年cov={yc.get(last_y, 0):.1f}% 但中间覆盖良好 → 断更/新列")
            flag = True
    if not flag:
        out("  (无)")

    # ── 3. 数据质量 ──
    out("\n## 3. 数据质量")
    # 3.1 OHLCV 铁律
    out("\n### 3.1 OHLCV 完整性")
    if all(c in df.columns for c in ["high", "low"]):
        v = df["high"] < df["low"]
        out(f"  high < low:            {int(v.sum()):>8,}")
    if all(c in df.columns for c in ["high", "open"]):
        out(f"  high < open:            {int((df['high'] < df['open']).sum()):>8,}")
    if all(c in df.columns for c in ["high", "close"]):
        out(f"  high < close:           {int((df['high'] < df['close']).sum()):>8,}")
    if all(c in df.columns for c in ["low", "open"]):
        out(f"  low > open:             {int((df['low'] > df['open']).sum()):>8,}")
    if all(c in df.columns for c in ["low", "close"]):
        out(f"  low > close:            {int((df['low'] > df['close']).sum()):>8,}")
    if "volume" in df.columns:
        out(f"  volume < 0:             {int((df['volume'] < 0).sum()):>8,}")
        out(f"  volume null:            {int(df['volume'].isna().sum()):>8,}")
    # 负价格
    neg = [c for c in ["open", "high", "low", "close", "pre_close"] if c in df.columns]
    for c in neg:
        n = int((df[c] < 0).sum())
        if n:
            out(f"  负值 {c}: {n:,}")

    # 3.2 重复键
    out("\n### 3.2 重复 (symbol,date)")
    dup = int(df.duplicated(subset=["symbol", "date"]).sum())
    out(f"  重复行: {dup:,}")

    # 3.3 inf / 常数 / 全空
    out("\n### 3.3 inf / 常数列 / 全空列")
    inf_cols, const_cols, empty_cols = [], [], []
    for c in num_cols:
        s = df[c]
        n_inf = int(np.isinf(s).sum())
        if n_inf:
            inf_cols.append(f"{c}({n_inf})")
        if s.dropna().nunique() == 1:
            const_cols.append(c)
        if s.notna().sum() == 0:
            empty_cols.append(c)
    out(f"  inf 列: {', '.join(inf_cols) if inf_cols else '(无)'}")
    out(f"  常数列 (单值): {', '.join(const_cols) if const_cols else '(无)'}")
    out(f"  全空列: {', '.join(empty_cols) if empty_cols else '(无)'}")

    # 3.4 标识列
    out("\n### 3.4 标识 / 状态列")
    for c in ["board", "industry", "is_st", "is_suspended", "pre_close"]:
        if c not in df.columns:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            out(f"  {c:<14} True={int(df[c].sum()):>8,} False={int((~df[c]).sum()):>8,}")
        else:
            n_nn = int(df[c].notna().sum())
            out(f"  {c:<14} 非空={n_nn:>9,} ({n_nn/rows*100:6.2f}%)")
    if "board" in df.columns:
        out("\n  board 分布:")
        for b, n in df["board"].value_counts().items():
            out(f"    {str(b):<10} {n:>9,} ({n/rows*100:5.2f}%)")

    # ── 4. 关键缺口汇总 (覆盖率 < 50% 且非设计性稀疏) ──
    out("\n## 4. 覆盖率 < 50% 的列 (前 25)")
    low = [c for c in cols if stats[c]["cov"] < 50]
    low.sort(key=lambda c: stats[c]["cov"])
    for c in low[:25]:
        yc = " ".join(f"{year_cov.loc[y, c]:.0f}%" if c in year_cov.columns else "  -" for y in years_present)
        out(f"  {c:<30} cov={stats[c]['cov']:6.2f}%  per-year: {yc}")

    # ── 写报告 ──
    report = "\n".join(lines)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    logger.info("报告已写入: %s", OUT_REPORT)


if __name__ == "__main__":
    main()
