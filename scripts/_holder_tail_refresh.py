# -*- coding: utf-8 -*-
"""一次性拉取 holdertrade 尾部缺口 (20260801~20260803) 追加进 data/_holder_cmp_raw.parquet.

保持与 _holder_cmp_fetch.py 完全相同的 schema 与字段语义:
  symbol, date, change_vol, change_ratio, holder_type, in_de,
  evt_start_date, evt_end_date, sign, signed_ratio

Tushare 不可用/限流时优雅跳过 (打印提示, 退出码 0), 不阻塞后续回填.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline1.data_supply import DataSupplyChain, _with_timeout
from config.settings import TUSHARE_TOKEN  # noqa: F401  (触发配置校验)

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "_holder_cmp_raw.parquet",
)
START, END = "20260801", "20260803"
PAGE = 3000
SLEEP = 0.3

_FIELDS = (
    "ts_code,ann_date,holder_name,holder_type,in_de,change_vol,change_ratio,"
    "after_share,after_ratio,avg_price,total_share,begin_date,close_date"
)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def main() -> None:
    supply = DataSupplyChain()
    pro = supply._tushare_pro()
    if pro is None:
        print("Tushare pro 不可用, 跳过尾部刷新 (08-03 列保持 NaN)")
        return

    def _fetch_page(offset: int) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return _with_timeout(
                    lambda: pro.stk_holdertrade(
                        start_date=START,
                        end_date=END,
                        limit=PAGE,
                        offset=offset,
                        fields=_FIELDS,
                    ),
                    timeout=120,
                )
            except Exception as exc:  # noqa: BLE001 — 网络抖动重试
                last_exc = exc
                print(f"  page offset={offset} attempt={attempt + 1} failed: {exc!r}")
                time.sleep(2.0 * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    all_pages: list[pd.DataFrame] = []
    offset = 0
    try:
        while True:
            raw = _fetch_page(offset)
            if raw is None or len(raw) == 0:
                break
            all_pages.append(raw)
            if len(raw) < PAGE:
                break
            offset += PAGE
            time.sleep(SLEEP)
    except Exception as exc:  # noqa: BLE001 — 限流/超时 → 优雅跳过
        print(f"Tushare stk_holdertrade 尾部拉取失败, 跳过: {exc!r}")
        return

    if not all_pages:
        print(f"{START}~{END} 无新增事件, 不改写文件")
        return

    raw = pd.concat(all_pages, ignore_index=True)
    print(f"tail raw rows: {len(raw)}")

    change_vol = pd.to_numeric(raw.get("change_vol", 0), errors="coerce").fillna(0)
    change_ratio = pd.to_numeric(raw.get("change_ratio", np.nan), errors="coerce")
    holder_type = raw.get("holder_type", "").fillna("").astype(str).str.upper()
    in_de = raw.get("in_de", "").fillna("").astype(str).str.upper()
    out = pd.DataFrame(
        {
            "symbol": raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", ""),
            "date": _to_dt(raw.get("ann_date", "")),
            "change_vol": change_vol,
            "change_ratio": change_ratio,
            "holder_type": holder_type,
            "in_de": in_de,
            "evt_start_date": _to_dt(raw.get("begin_date", "")),
            "evt_end_date": _to_dt(raw.get("close_date", "")),
        }
    )
    out["sign"] = out["in_de"].map({"IN": 1.0, "DE": -1.0}).fillna(0.0)
    out["signed_ratio"] = out["sign"] * out["change_ratio"]

    # 去重 (分页可能返回重复行; raw 无 holder_name, 用全字段去重)
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) != before:
        print(f"tail dedup: {before} -> {len(out)}")

    old = pd.read_parquet(OUT)
    # 幂等: 删除已存在的 (symbol,date) 尾部行再追加
    tail_dates = out["date"]
    old = old[~old["date"].isin(tail_dates)]
    new_rows = len(out)
    combined = pd.concat([old, out], ignore_index=True)
    combined.to_parquet(OUT, index=False)
    print(f"saved: {OUT} ({combined.shape})")
    print(
        "date range now: %s ~ %s, new rows added: %d"
        % (
            combined["date"].min().strftime("%Y%m%d"),
            combined["date"].max().strftime("%Y%m%d"),
            new_rows,
        )
    )
    print("08-03 rows:", len(combined[combined["date"] == "2026-08-03"]))


if __name__ == "__main__":
    main()
