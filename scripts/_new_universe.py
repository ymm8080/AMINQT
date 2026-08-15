"""全市场宇宙修复 Step 1: 生成待补股票清单 (2026-08-15 用户拍板).

目标口径 = "全市场剔 ST/次新" (与 ingest gate 一致):
  - stock_basic list_status=L, 沪深 A 股 (60/68/00/30 开头, 剔北交所/B股)
  - 名称非 ST/*ST/退 (name_is_st)
  - 减去现有 V3 面板 3225 只 → 待补清单

历史 ST 时段行保留 (已实测: 面板内 000007 等 *ST 期间行全在, gate 只按当日名剔).
次新 gate (上市 <150 交易日剔行) 在 build 阶段用 apply_ingest_scan 同口径处理.

WORM: data/new_universe/new_symbols_<ts>.parquet (+ should_be_universe_<ts>.parquet)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tushare as ts  # noqa: E402

from app.core.universe_manager import name_is_st  # noqa: E402
from config import settings  # noqa: E402

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
OUT_DIR = "data/new_universe"
MIN_LIST_DAYS = settings.INGEST_MIN_LIST_DAYS


def main() -> None:
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    basic = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,list_date,market,industry",
    )
    basic = basic.drop_duplicates(subset="symbol").copy()
    basic["symbol"] = basic["symbol"].astype(str).str.strip()

    # 沪深 A 股: 60 沪主板 / 68 科创板 / 00 深主板 / 30 创业板
    hs = basic[basic["symbol"].str.match(r"^(60|68|00|30)\d{4}$", na=False)].copy()
    hs = hs[~hs["name"].map(name_is_st)]
    print(f"[stock_basic] 沪深A 剔ST: {len(hs)} 只")

    panel_syms = set(
        pd.read_parquet(PANEL, columns=["symbol"])["symbol"].astype(str).str.strip()
    )
    new = hs[~hs["symbol"].isin(panel_syms)].copy()
    print(f"[new] 待补: {len(new)} 只 (面板现有 {len(panel_syms)} 只)")

    # 次新 gate: 上市 <150 交易日 → 现在还没到入库时点, 但列出来 (unfreeze 后 daily fetch 会自动收)
    # 这里只标注, 不剔清单 — build 阶段用 apply_ingest_scan 精确按面板日历剔行.
    new = new.sort_values("symbol").reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"new_symbols_{ts_}.parquet")
    new.to_parquet(out, index=False)
    print(f"[save] {out}")

    should = pd.concat(
        [hs, hs[hs["symbol"].isin(panel_syms)].assign(in_panel=True)],
        ignore_index=True,
    ).drop_duplicates(subset="symbol")
    out2 = os.path.join(OUT_DIR, f"should_be_universe_{ts_}.parquet")
    should[
        ["symbol", "name", "list_date", "market", "industry", "in_panel"]
    ].to_parquet(out2, index=False)
    print(f"[save] {out2}")

    print("\n== 待补清单按上市年份 ==")
    ld = pd.to_datetime(new["list_date"], format="%Y%m%d", errors="coerce")
    print(ld.dt.year.value_counts().sort_index().to_string())
    print("\n== 按板块 ==")
    board = (
        new["symbol"]
        .str[:3]
        .map(
            lambda s: (
                "沪主板"
                if s.startswith("60")
                else "科创板"
                if s.startswith("68")
                else "深主板"
                if s.startswith("00")
                else "创业板"
            )
        )
    )
    print(board.value_counts().to_string())
    # 上市 <150 交易日暂不入库的数量
    print(
        f"\n上市未满 {MIN_LIST_DAYS} 交易日 (unfreeze 后自动进): "
        f"{(ld >= '2025-11-20').sum()} 只 (按日历粗估)"
    )


if __name__ == "__main__":
    main()
