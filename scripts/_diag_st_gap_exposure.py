"""_diag_st_gap_exposure.py — ST 期间 + 停牌缺口暴露诊断 (2026-08-14).

背景 (数据质量两大污染源, 见 memory):
1. ST: 面板无 is_st 列 (ingest 入口按当日名称剔 ST, 中途戴帽的股留在池里当正常股),
   训练含 ST 期间行 (主板 5% 涨跌停 + 退市风险, 标签/特征失真).
2. 停牌缺口: 停牌行在面板构建时被删 (is_suspended 全 False), 滚动特征
   (ret60/rps_60/sharpe_20/ADX 等) 跨缺口桥接 — dual 53.6% 行位于缺口后 60 行内.

本脚本只做暴露量化 (轻量, 无校准重算):
- ST 历史: Tushare namechange 缓存 data/supply_cache/namechange/namechange_full_*.parquet
  (ST/*ST 期间 + 退市整理期), 命中面板行 / 交付清单行.
- 停牌缺口: 面板 symbol×date 序列跳日 (>1 交易日) → 缺口后 60 行 = 60d 类特征污染带,
  统计面板行占比 + 交付清单命中.
WORM 输出 data/_diag_st_gap_exposure_<ts>.json.

用法: python scripts/_diag_st_gap_exposure.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.settings import DATA_DIR

STOCK_LIST_DIR = os.path.join(
    os.path.dirname(DATA_DIR), "DAILY OPERATION", "STOCK LIST"
)
NAME_CACHE = DATA_DIR / "supply_cache" / "namechange"
GAP_WINDOW = 60  # ret60/rps_60 等 60d 类特征的污染带 (5/20d 类为其子集)


def _st_periods() -> pd.DataFrame:
    """namechange 缓存 → ST/*ST/退 期间表 [symbol, start, end, name]."""
    files = sorted(glob.glob(str(NAME_CACHE / "namechange_full_*.parquet")))
    if not files:
        return pd.DataFrame(columns=["symbol", "start", "end", "name"])
    nc = pd.read_parquet(files[-1])
    nc["symbol"] = nc["ts_code"].str.split(".").str[0]
    bad = nc[nc["name"].str.contains("ST|退", na=False)].copy()
    bad["start"] = pd.to_datetime(bad["start_date"])
    bad["end"] = pd.to_datetime(bad["end_date"]).fillna(pd.Timestamp("2027-01-01"))
    return bad[["symbol", "start", "end", "name"]].reset_index(drop=True)


def _hit_mask(symbols: pd.Series, dates: pd.Series, periods: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=symbols.index)
    for _, r in periods.iterrows():
        m |= (symbols == r["symbol"]) & (dates >= r["start"]) & (dates <= r["end"])
    return m


def _gap_mask(t: pd.DataFrame) -> pd.Series:
    """面板行级: 该行是否位于停牌缺口后 GAP_WINDOW 行内 (60d 类特征污染)."""
    t = t.copy()
    t["date"] = pd.to_datetime(t["date"])
    cal = np.unique(t["date"].values)
    idx = np.searchsorted(cal, t["date"].values)
    skip = pd.Series(idx, index=t.index).groupby(t["symbol"]).diff().fillna(1.0)
    roll = skip.groupby(t["symbol"]).transform(
        lambda s: s.rolling(GAP_WINDOW, min_periods=1).max()
    )
    return roll > 1


def main() -> int:
    periods = _st_periods()
    out: dict = {
        "ts": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "st_periods": int(len(periods)),
        "st_symbols": int(periods["symbol"].nunique()),
        "panels": {},
        "delivered": {},
    }
    for board in ("main", "dual"):
        fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
        t = pq.read_table(str(fp), columns=["symbol", "date"]).to_pandas()
        t["symbol"] = t["symbol"].astype(str)
        t["date"] = pd.to_datetime(t["date"])
        st_m = _hit_mask(t["symbol"], t["date"], periods)
        gap_m = _gap_mask(t)
        rec = t["date"] >= (t["date"].max() - pd.Timedelta(days=400))
        out["panels"][board] = {
            "rows": int(len(t)),
            "symbols": int(t["symbol"].nunique()),
            "st_rows": int(st_m.sum()),
            "st_rows_pct": float(st_m.mean()),
            "st_symbols": int(t.loc[st_m, "symbol"].nunique()),
            "gap_contaminated_rows": int(gap_m.sum()),
            "gap_contaminated_pct": float(gap_m.mean()),
            "recent400d_gap_pct": float(gap_m[rec.values].mean()),
            "recent400d_st_pct": float(st_m[rec.values].mean()),
        }
    for f in sorted(glob.glob(os.path.join(STOCK_LIST_DIR, "*.csv"))):
        name = os.path.basename(f)
        if not any(
            k in name
            for k in (
                "parallel_shortlist",
                "overall_shortlist",
                "slowbull_pool",
                "legacy_stocklist",
            )
        ):
            continue
        try:
            d = pd.read_csv(f, dtype={"symbol": str})
        except Exception:
            continue
        if "symbol" not in d.columns or "date" not in d.columns:
            continue
        dd = pd.to_datetime(d["date"], errors="coerce")
        st_hits = [
            str(sy).zfill(6)
            for sy, dt_ in zip(d["symbol"], dd)
            if not pd.isna(dt_)
            and _hit_mask(
                pd.Series([str(sy).zfill(6)]),
                pd.Series([dt_]),
                periods,
            ).iloc[0]
        ]
        if st_hits:
            out["delivered"][name] = {"st_hits": st_hits}
    ts = out["ts"]
    fp_out = DATA_DIR / f"_diag_st_gap_exposure_{ts}.json"
    fp_out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print(f"\n[saved] {fp_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
