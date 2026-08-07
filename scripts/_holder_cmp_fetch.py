"""一次性拉取 holdertrade 全字段 (含 change_ratio/holder_type) 用于 GLM vs Kimi 方案 IC 对比.

不修改生产 fetch_holdertrade (其输出裁剪了 change_ratio/holder_type);
本脚本直接调用 Tushare pro API 全字段分页拉取, 缓存到 data/_holder_cmp_raw.parquet.
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
START, END = "20230103", "20260731"
PAGE = 3000
SLEEP = 0.3

_FIELDS = (
    "ts_code,ann_date,holder_name,holder_type,in_de,change_vol,change_ratio,"
    "after_share,after_ratio,avg_price,total_share,begin_date,close_date"
)


def main() -> None:
    supply = DataSupplyChain()
    pro = supply._tushare_pro()
    if pro is None:
        raise SystemExit("Tushare pro 不可用")

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
    t0 = time.time()
    while True:
        raw = _fetch_page(offset)
        if raw is None or len(raw) == 0:
            break
        all_pages.append(raw)
        if len(raw) < PAGE:
            break
        offset += PAGE
        time.sleep(SLEEP)
    raw = pd.concat(all_pages, ignore_index=True)
    print(f"raw rows: {len(raw)}, pages fetched, {time.time() - t0:.1f}s")

    def _to_dt(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    change_vol = pd.to_numeric(raw.get("change_vol", 0), errors="coerce").fillna(0)
    change_ratio = pd.to_numeric(raw.get("change_ratio", np.nan), errors="coerce")
    holder_type = raw.get("holder_type", "").astype(str).str.upper()
    in_de = raw.get("in_de", "").astype(str).str.upper()
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
    out.to_parquet(OUT, index=False)
    print(f"saved: {OUT} ({out.shape})")
    print("holder_type dist:", out["holder_type"].value_counts().to_dict())
    print("in_de dist:", out["in_de"].value_counts().to_dict())
    print("ratio missing: %.2f%%" % (out["change_ratio"].isna().mean() * 100))
    print(
        "evt_start missing: {:.2f}%, evt_end missing: {:.2f}%".format(
            out["evt_start_date"].isna().mean() * 100,
            out["evt_end_date"].isna().mean() * 100,
        )
    )
    print(
        "sample:",
        out[
            [
                "symbol",
                "date",
                "holder_type",
                "in_de",
                "change_ratio",
                "evt_start_date",
                "evt_end_date",
            ]
        ]
        .head(8)
        .to_string(),
    )


if __name__ == "__main__":
    main()
